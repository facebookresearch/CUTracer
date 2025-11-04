/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed
 * under MIT, based on original NVIDIA NVBit sample code licensed under
 * BSD-3-Clause. See LICENSE file in the root directory for Meta's license
 * terms. See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* contains definition of the reg_info_t and mem_access_t structure */
#include "common.h"

/* analysis functionality */
#include "analysis.h"

/* instrumentation functionality */
#include "instrument.h"

/* env config */
#include "env_config.h"
#include "trace_writer.h"

/* logging functionality */
#include "log.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)

#define CUDA_CHECK_LAST_ERROR()                                                                       \
  do {                                                                                                \
    cudaError_t err = cudaGetLastError();                                                             \
    if (err != cudaSuccess) {                                                                         \
      loprintf("FATAL: CUDA Last Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      fflush(stderr);                                                                                 \
      assert(err == cudaSuccess);                                                                     \
    }                                                                                                 \
  } while (0)

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* The kernel launch is identified by a global id */
uint64_t global_kernel_launch_id = 0;

// Global mapping tables for kernel launch tracking
std::map<uint64_t, std::pair<CUcontext, CUfunction>> kernel_launch_to_func_map;
std::map<uint64_t, uint32_t> kernel_launch_to_iter_map;
std::map<uint64_t, KernelDimensions> kernel_launch_to_dimensions_map;

// map to store the iteration count for each kernel
static std::map<CUfunction, uint32_t> kernel_iter_map;

/* ===== Main Functionality ===== */
/**
 * @brief Conditionally instruments a CUDA function by delegating to specialized
 * instrumentation functions.
 *
 * This function is based on the `instrument_kernel` function from NVIDIA's
 * `mem_trace` example. It inspects a function and, if it matches the filters,
 * iterates through its SASS instructions, delegating the actual instrumentation
 * to functions in `instrument.cu`.
 *
 * Meta's enhancements include:
 *  - Combining memory (MREF) and register (REG) tracing.
 *  - Adding support for Unified Registers (UREG).
 *  - Kernel Filtering: A system (`kernel_filters`) to selectively
 *    instrument kernels based on environment variables, avoiding unnecessary
 *    overhead.
 *  - Modular Instrumentation API: The core instrumentation logic is refactored
 *    into `instrument.cu`, providing a clear API for different tracing types.
 *
 * @param ctx The CUDA context of the function.
 * @param func The `CUfunction` to inspect and potentially instrument.
 * @return `true` if any related function was matched by the filters, `false` otherwise.
 */
bool instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  bool any_related_function_matched = kernel_filters.empty();
  /* iterate on function */
  for (auto f : related_functions) {
    // Get function name (both mangled and unmangled versions)
    const char* unmangled_name = nvbit_get_func_name(ctx, f, false);
    const char* mangled_name = nvbit_get_func_name(ctx, f, true);

    // Check if function name contains any of the patterns
    bool should_instrument = true;  // Default to true if no filters specified

    if (!kernel_filters.empty()) {
      should_instrument = false;  // Start with false when we have filters
      for (const auto& filter : kernel_filters) {
        if ((unmangled_name && strstr(unmangled_name, filter.c_str()) != NULL) ||
            (mangled_name && strstr(mangled_name, filter.c_str()) != NULL)) {
          should_instrument = true;
          any_related_function_matched = true;  // Mark that at least one kernel matched
          if (verbose) {
            loprintf("Found matching kernel for filter '%s': %s (mangled: %s)\n", filter.c_str(),
                     unmangled_name ? unmangled_name : "unknown", mangled_name ? mangled_name : "unknown");
          }
          break;
        }
      }
    } else if (verbose) {
      loprintf("Instrumenting kernel: %s (mangled: %s)\n", unmangled_name ? unmangled_name : "unknown",
               mangled_name ? mangled_name : "unknown");
    }

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
      loprintf("Inspecting kernel %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
    }

    if (!should_instrument) {
      continue;
    }
    if (dump_cubin) {
      std::string kernel_hash_hex = compute_kernel_name_hash_hex(ctx, f);
      std::string cubin_filename = std::string(nvbit_get_func_name(ctx, f)) + "_0x" + kernel_hash_hex + ".cubin";
      nvbit_dump_cubin(ctx, f, cubin_filename.c_str());
    }
    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }
      if (verbose) {
        instr->printDecoded();
      }

      std::vector<int> reg_num_list;
      std::vector<int> ureg_num_list;
      int mref_idx = 0;
      int opcode_id = instr->getIdx();
      ctx_state->id_to_sass_map[f][opcode_id] = std::string(instr->getSass());
      /* iterate on the operands */
      for (int i = 0; i < instr->getNumOperands(); i++) {
        /* get the operand "i" */
        const InstrType::operand_t* op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::REG) {
          for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
            reg_num_list.push_back(op->u.reg.num + reg_idx);
          }
        } else if (op->type == InstrType::OperandType::UREG) {
          for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
            ureg_num_list.push_back(op->u.reg.num + reg_idx);
          }
        } else if (op->type == InstrType::OperandType::MREF) {
          // TODO: double check this with NVIDIA people
          if (op->u.mref.has_desc) {
            ureg_num_list.push_back(op->u.mref.desc_ureg_num);
            ureg_num_list.push_back(op->u.mref.desc_ureg_num + 1);
          }

          // Use new instrumentation interface for memory tracing
          if (is_instrument_type_enabled(InstrumentType::MEM_TRACE)) {
            instrument_memory_trace(instr, opcode_id, ctx_state, mref_idx);
          }
          mref_idx++;
        }
      }

      // Choose instrumentation based on enabled types.
      if (is_instrument_type_enabled(InstrumentType::OPCODE_ONLY)) {
        // Lightweight instrumentation for instruction histogram analysis.
        // This sends minimal data (opcode_id, warp_id) to the CPU, reducing
        // overhead.
        instrument_opcode_only(instr, opcode_id, ctx_state);
      } else if (is_instrument_type_enabled(InstrumentType::REG_TRACE)) {
        // Full register tracing.
        instrument_register_trace(instr, opcode_id, ctx_state, reg_num_list, ureg_num_list);
      }
    }

    // Statically identify special instructions for this function:
    // - "clock" (CS2R SR_CLOCKLO) used for histogram region boundaries
    // - "EXIT" used as a candidate signal for warp completion on host side
    if (should_instrument) {
      for (std::map<int, std::string>::const_iterator it_sass = ctx_state->id_to_sass_map[f].begin();
           it_sass != ctx_state->id_to_sass_map[f].end(); ++it_sass) {
        const char* sass_cstr = it_sass->second.c_str();
        if (strstr(sass_cstr, "CS2R") && strstr(sass_cstr, "SR_CLOCKLO")) {
          ctx_state->clock_opcode_ids[f].insert(it_sass->first);
        }
        // Simple substring detection for EXIT mnemonic (predication handled by extract on host side)
        if (strstr(sass_cstr, "EXIT")) {
          ctx_state->exit_opcode_ids[f].insert(it_sass->first);
        }
      }
    }
    /* ============================================ */
  }
  return any_related_function_matched;
}

// Reference code from NVIDIA nvbit mem_trace tool
/* flush channel */
__global__ void flush_channel(ChannelDev* ch_dev) {
  ch_dev->flush();
}

// Reference code from NVIDIA nvbit mem_trace tool
void init_context_state(CUcontext ctx) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];
  ctx_state->recv_thread_done = RecvThreadState::WORKING;
  cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE, ctx_state->channel_dev, recv_thread_fun,
                               ctx);
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

// Reference code from NVIDIA nvbit mem_trace tool
/**
 * @brief Prepares for a kernel launch, conditionally instrumenting and logging.
 *
 * Based on `enter_kernel_launch` from NVIDIA's `mem_trace` example, this
 * function is called just before a kernel launch. It determines if a kernel
 * should be instrumented based on active filters.
 *
 * Key modifications by Meta include making instrumentation and log file creation
 * conditional on the filtering result. This prevents empty log files for
 * skipped kernels.
 *
 * @param ctx The current CUDA context.
 * @param func The kernel function being launched.
 * @param kernel_launch_id A reference to the global kernel launch counter.
 * @param cbid The NVBit callback ID for the CUDA API call.
 * @param params A pointer to the parameters of the CUDA launch call.
 * @param stream_capture True if the launch is part of a CUDA stream capture.
 * @param build_graph True if the launch is part of a manual graph build.
 * @return `true` if the kernel was instrumented, `false` otherwise.
 */
static bool enter_kernel_launch(CUcontext ctx, CUfunction func, uint64_t& kernel_launch_id, nvbit_api_cuda_t cbid,
                                void* params, bool stream_capture = false, bool build_graph = false) {
  CTXstate* ctx_state = ctx_state_map[ctx];
  // no need to sync during stream capture or manual graph build, since no
  // kernel is actually launched.
  if (!stream_capture && !build_graph) {
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
  }

  bool should_instrument = instrument_function_if_needed(ctx, func);

  int nregs = 0;
  CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

  int shmem_static_nbytes = 0;
  CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

  /* get function name and pc */
  const char* func_name = nvbit_get_func_name(ctx, func);
  uint64_t pc = nvbit_get_func_addr(ctx, func);
  std::string kernel_hash_hex = compute_kernel_name_hash_hex(ctx, func);

  // during stream capture or manual graph build, no kernel is launched, so
  // do not set launch argument, do not print kernel info, do not increase
  // grid_launch_id. All these should be done at graph node launch time.
  if (!stream_capture && !build_graph) {
    /* set kernel launch id at launch time */
    nvbit_set_at_launch(ctx, func, (uint64_t)kernel_launch_id);

    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
      loprintf(
          "CUTracer: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
          "Kernel name %s - kernel hash 0x%s - kernel launch id %ld - grid size %d,%d,%d "
          "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
          "id %ld\n",
          (uint64_t)ctx, pc, func_name, kernel_hash_hex.c_str(), kernel_launch_id, p->config->gridDimX,
          p->config->gridDimY, p->config->gridDimZ, p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ,
          nregs, shmem_static_nbytes + p->config->sharedMemBytes, (uint64_t)p->config->hStream);
    } else {
      cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
      loprintf(
          "CUTracer: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
          "Kernel name %s - kernel hash 0x%s - kernel launch id %ld - grid size %d,%d,%d "
          "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
          "id %ld\n",
          (uint64_t)ctx, pc, func_name, kernel_hash_hex.c_str(), kernel_launch_id, p->gridDimX, p->gridDimY,
          p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes,
          (uint64_t)p->hStream);
    }

    // For histogram analysis, we need to map the kernel launch ID back to its
    // function and context to access the correct SASS and clock opcode maps.
    // We also track the iteration count for unique file naming.
    uint32_t current_iter = kernel_iter_map[func];
    kernel_launch_to_func_map[kernel_launch_id] = {ctx, func};
    kernel_launch_to_iter_map[kernel_launch_id] = current_iter;

    // Store kernel dimensions for warp statistics tracking
    KernelDimensions dims;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
      dims.gridDimX = p->config->gridDimX;
      dims.gridDimY = p->config->gridDimY;
      dims.gridDimZ = p->config->gridDimZ;
      dims.blockDimX = p->config->blockDimX;
      dims.blockDimY = p->config->blockDimY;
      dims.blockDimZ = p->config->blockDimZ;
    } else {
      cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
      dims.gridDimX = p->gridDimX;
      dims.gridDimY = p->gridDimY;
      dims.gridDimZ = p->gridDimZ;
      dims.blockDimX = p->blockDimX;
      dims.blockDimY = p->blockDimY;
      dims.blockDimZ = p->blockDimZ;
    }
    kernel_launch_to_dimensions_map[kernel_launch_id] = dims;

    // increment kernel launch id for next launch
    // kernel id can be changed here, since nvbit_set_at_launch() has copied
    // its value above.
    kernel_launch_id++;
  }

  // Create TraceWriter for all trace modes (0/1/2)
  // Note: log_open_kernel_file() has been removed - TraceWriter now handles all trace output
  if (should_instrument) {
    if (!ctx_state->trace_writer) {
      std::string base_filename = generate_kernel_log_basename(ctx, func, kernel_iter_map[func]++);
      ctx_state->trace_writer = new TraceWriter(base_filename, trace_format_ndjson);

      // Initialize trace_index for this kernel
      if (!stream_capture && !build_graph) {
        ctx_state->trace_index_by_kernel[kernel_launch_id - 1] = 0;
      }

      if (verbose) {
        loprintf("Created TraceWriter for mode %d, file: %s\n", trace_format_ndjson, base_filename.c_str());
      }
    }
  }
  /* enable instrumented code to run */
  nvbit_enable_instrumented(ctx, func, should_instrument);
  return should_instrument;
}

// the function is only called for non cuda graph launch cases.
static void leave_kernel_launch(CTXstate* ctx_state, uint64_t& grid_launch_id) {
  // make sure user kernel finishes to avoid deadlock
  cudaDeviceSynchronize();
  /* push a flush channel kernel */
  flush_channel<<<1, 1>>>(ctx_state->channel_dev);

  /* Make sure GPU is idle */
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST_ERROR();

  // Flush TraceWriter buffer
  if (ctx_state->trace_writer) {
    ctx_state->trace_writer->flush();
  }
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params,
                         CUresult* pStatus) {
  pthread_mutex_lock(&cuda_event_mutex);

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag) {
    pthread_mutex_unlock(&cuda_event_mutex);
    return;
  }
  skip_callback_flag = true;

  CTXstate* ctx_state = ctx_state_map[ctx];

  switch (cbid) {
    // Identify all the possible CUDA launch events without stream
    // parameters, they will not get involved with cuda graph
    case API_CUDA_cuLaunch:
    case API_CUDA_cuLaunchGrid: {
      cuLaunch_params* p = (cuLaunch_params*)params;
      CUfunction func = p->f;
      if (!is_exit) {
        enter_kernel_launch(ctx, func, global_kernel_launch_id, cbid, params);
      } else {
        leave_kernel_launch(ctx_state, global_kernel_launch_id);
      }
    } break;
    // To support kernel launched by cuda graph (in addition to existing kernel
    // launche method), we need to do:
    //
    // 1. instrument kernels at cudaGraphAddKernelNode event. This is for cases
    // that kernels are manually added to a cuda graph.
    // 2. distinguish captured kernels when kernels are recorded to a graph
    // using stream capture. cudaStreamIsCapturing() tells us whether a stream
    // is capturiong.
    // 3. per-kernel instruction counters, since cuda graph can launch multiple
    // kernels at the same time.
    //
    // Three cases:
    //
    // 1. original kernel launch:
    //     1a. for any kernel launch without using a stream, we instrument it
    //     before it is launched, call cudaDeviceSynchronize after it is
    //     launched and read the instruction counter of the kernel.
    //     1b. for any kernel launch using a stream, but the stream is not
    //     capturing, we do the same thing as 1a.
    //
    //  2. cuda graph using stream capturing: if a kernel is launched in a
    //  stream and the stream is capturing. We instrument the kernel before it
    //  is launched and do nothing after it is launched, because the kernel is
    //  not running until cudaGraphLaunch. Instead, we issue a
    //  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
    //  instruction counters, since a cloned graph might be launched afterwards.
    //
    //  3. cuda graph manual: we instrument the kernel added by
    //  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch as 2.
    //
    // The above method should handle most of cuda graph launch cases.
    // kernel launches with stream parameter, they can be used for cuda graph
    case API_CUDA_cuLaunchKernel_ptsz:
    case API_CUDA_cuLaunchKernel:
    case API_CUDA_cuLaunchCooperativeKernel:
    case API_CUDA_cuLaunchCooperativeKernel_ptsz:
    case API_CUDA_cuLaunchKernelEx:
    case API_CUDA_cuLaunchKernelEx_ptsz:
    case API_CUDA_cuLaunchGridAsync: {
      CUfunction func;
      CUstream hStream;

      if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        func = p->f;
        hStream = p->config->hStream;
      } else if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
                 cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz || cbid == API_CUDA_cuLaunchCooperativeKernel) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        func = p->f;
        hStream = p->hStream;
      } else {
        cuLaunchGridAsync_params* p = (cuLaunchGridAsync_params*)params;
        func = p->f;
        hStream = p->hStream;
      }

      cudaStreamCaptureStatus streamStatus;
      /* check if the stream is capturing, if yes, do not sync */
      CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
      if (!is_exit) {
        bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
        enter_kernel_launch(ctx, func, global_kernel_launch_id, cbid, params, stream_capture);
      } else {
        if (streamStatus != cudaStreamCaptureStatusActive) {
          if (verbose >= 1) {
            loprintf("kernel %s not captured by cuda graph\n", nvbit_get_func_name(ctx, func));
          }
          leave_kernel_launch(ctx_state, global_kernel_launch_id);
        } else {
          if (verbose >= 1) {
            loprintf("kernel %s captured by cuda graph\n", nvbit_get_func_name(ctx, func));
          }
        }
      }
    } break;
    case API_CUDA_cuGraphAddKernelNode: {
      cuGraphAddKernelNode_params* p = (cuGraphAddKernelNode_params*)params;
      CUfunction func = p->nodeParams->func;

      if (!is_exit) {
        // cuGraphAddKernelNode_params->nodeParams is the same as
        // cuLaunchKernel_params up to sharedMemBytes
        enter_kernel_launch(ctx, func, global_kernel_launch_id, cbid, (void*)p->nodeParams, false, true);
      }
    } break;
    case API_CUDA_cuGraphLaunch: {
      // if we are exiting a cuda graph launch:
      // Wait until the graph is completed using
      // cudaStreamSynchronize()
      if (is_exit) {
        cuGraphLaunch_params* p = (cuGraphLaunch_params*)params;

        CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
        CUDA_CHECK_LAST_ERROR();
        /* push a flush channel kernel */
        flush_channel<<<1, 1, 0, p->hStream>>>(ctx_state->channel_dev);
        CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
        CUDA_CHECK_LAST_ERROR();
      }

    } break;
    default:
      break;
  };

  skip_callback_flag = false;
  pthread_mutex_unlock(&cuda_event_mutex);
}

// Reference NVIDIA record_reg_vals example
void nvbit_tool_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  init_context_state(ctx);
  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_ctx_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  if (verbose) {
    printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
  }
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  CTXstate* ctx_state = new CTXstate;
  ctx_state_map[ctx] = ctx_state;
  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_ctx_term(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  skip_callback_flag = true;
  if (verbose) {
    loprintf("CUTracer: TERMINATING CONTEXT %p\n", ctx);
  }
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  /* Notify receiver thread and wait for receiver thread to
   * notify back */
  ctx_state->recv_thread_done = RecvThreadState::STOP;
  while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);

  // Cleanup TraceWriter
  if (ctx_state->trace_writer) {
    ctx_state->trace_writer->flush();
    delete ctx_state->trace_writer;
    ctx_state->trace_writer = nullptr;
  }

  // Clean up any remaining kernel mapping entries
  // (in case there were kernels launched but no data received)
  kernel_launch_to_func_map.clear();
  kernel_launch_to_iter_map.clear();

  ctx_state->channel_host.destroy(false);
  cudaFree(ctx_state->channel_dev);
  skip_callback_flag = false;
  delete ctx_state;
  pthread_mutex_unlock(&mutex);
  // Cleanup log handle system

  cleanup_log_handle();
}

// Reference code from NVIDIA nvbit mem_trace tool
void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func, CUstream stream, uint64_t launch_handle) {
  func_config_t config = {0};
  const char* func_name = nvbit_get_func_name(ctx, func);
  uint64_t pc = nvbit_get_func_addr(ctx, func);

  pthread_mutex_lock(&mutex);
  nvbit_set_at_launch(ctx, func, (uint64_t)global_kernel_launch_id, stream, launch_handle);
  nvbit_get_func_config(ctx, func, &config);

  loprintf(
      "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
      "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
      "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
      "id %ld\n",
      (uint64_t)ctx, pc, func_name, global_kernel_launch_id, config.gridDimX, config.gridDimY, config.gridDimZ,
      config.blockDimX, config.blockDimY, config.blockDimZ, config.num_registers,
      config.shmem_static_nbytes + config.shmem_dynamic_nbytes, (uint64_t)stream);
  // grid id can be changed here, since nvbit_set_at_launch() has copied its
  // value above.
  global_kernel_launch_id++;
  pthread_mutex_unlock(&mutex);
}

// Reference code from NVIDIA nvbit mem_trace tool with Meta modifications for
// env config
void nvbit_at_init() {
  // Initialize configuration from environment variables
  init_config_from_env();
  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex, &attr);

  pthread_mutex_init(&cuda_event_mutex, &attr);
}
