/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed under MIT,
 * based on original NVIDIA nvbit sample code licensed under BSD-3-Clause.
 * See LICENSE file in the root directory for Meta's license terms.
 * See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <string>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* analysis functionality */
#include "analysis.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

/* lock */
pthread_mutex_t cuda_event_mutex;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

std::map<int, std::string> id_to_sass_map;

// Based on NVIDIA code with Meta modifications for unified register support
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
      printf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
          instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
          instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
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
      id_to_sass_map[opcode_id] = std::string(instr->getSass());
      /* iterate on the operands */
      for (int i = 0; i < instr->getNumOperands(); i++) {
        /* get the operand "i" */
        const InstrType::operand_t *op = instr->getOperand(i);
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
          /* insert call to the instrumentation function with its
           * arguments */
          nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
          /* predicate value */
          nvbit_add_call_arg_guard_pred_val(instr);
          /* opcode id */
          nvbit_add_call_arg_const_val32(instr, opcode_id);
          /* memory reference 64 bit address */
          nvbit_add_call_arg_mref_addr64(instr, mref_idx);
          /* add instruction PC */
          nvbit_add_call_arg_const_val64(instr, instr->getOffset());
          /* add pointer to channel_dev*/
          nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
          mref_idx++;
        }
      }

      /* insert call to the instrumentation function with its arguments */
      nvbit_insert_call(instr, "record_reg_val", IPOINT_BEFORE);
      /* guard predicate value */
      nvbit_add_call_arg_guard_pred_val(instr);
      /* opcode id */
      nvbit_add_call_arg_const_val32(instr, opcode_id);
      /* add pointer to channel_dev*/
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
      /* add instruction PC */
      nvbit_add_call_arg_const_val64(instr, instr->getOffset());
      /* how many register values are passed next */
      nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
      nvbit_add_call_arg_const_val32(instr, ureg_num_list.size());
      for (int num : reg_num_list) {
        /* last parameter tells it is a variadic parameter passed to
         * the instrument function record_reg_val() */
        nvbit_add_call_arg_reg_val(instr, num, true);
      }
      for (int num : ureg_num_list) {
        nvbit_add_call_arg_ureg_val(instr, num, true);
      }
    }
  }
}

// Based on NVIDIA code with Meta modifications for message type support
__global__ void flush_channel(ChannelDev *ch_dev = NULL) {
  // Get the channel to use
  ChannelDev *channel = (ch_dev == NULL) ? &channel_dev : ch_dev;

  /* push completion marker with negative cta id */
  reg_info_t ri;
  ri.header.type = MSG_TYPE_REG_INFO;
  ri.cta_id_x = -1;  // Completion marker
  ri.pc = 0;
  ri.num_uregs = 0;

  channel->push(&ri, sizeof(reg_info_t));
  channel->flush();
}

// Original NVIDIA implementation
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params,
                         CUresult *pStatus) {
  pthread_mutex_lock(&cuda_event_mutex);

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag) {
    pthread_mutex_unlock(&cuda_event_mutex);
    return;
  }
  skip_callback_flag = true;

  /* Identify all the possible CUDA launch events */
  if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchGrid ||
      cbid == API_CUDA_cuLaunchGridAsync || cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernelEx ||
      cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
    /* cast params to launch parameter based on cbid since if we are here
     * we know these are the right parameters types */
    CUfunction func;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
      func = p->f;
    } else {
      cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
      func = p->f;
    }

    if (!is_exit) {
      /* Make sure GPU is idle */
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      int nregs = 0;
      CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

      int shmem_static_nbytes = 0;
      CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

      instrument_function_if_needed(ctx, func);

      nvbit_enable_instrumented(ctx, func, true);

      if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
        printf(
            "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
            "%d - shmem %d - cuda stream id %ld\n",
            nvbit_get_func_name(ctx, func), p->config->gridDimX, p->config->gridDimY, p->config->gridDimZ,
            p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ, nregs,
            shmem_static_nbytes + p->config->sharedMemBytes, (uint64_t)p->config->hStream);
      } else {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        printf(
            "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
            "%d - shmem %d - cuda stream id %ld\n",
            nvbit_get_func_name(ctx, func), p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
            p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
      }
    } else {
      /* make sure current kernel is completed */
      cudaDeviceSynchronize();
      cudaError_t kernelError = cudaGetLastError();
      if (kernelError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
        assert(0);
      }

      /* issue flush of channel so we are sure all the memory accesses
       * have been pushed */
      flush_channel<<<1, 1>>>();
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);
    }
  }
  skip_callback_flag = false;
  pthread_mutex_unlock(&cuda_event_mutex);
}

// Original NVIDIA implementation
void nvbit_tool_init(CUcontext ctx) {
  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&cuda_event_mutex, &attr);

  /* Initialize analysis dependencies */
  init_recv_thread_deps(&channel_host, &recv_thread_done, &id_to_sass_map);

  recv_thread_done = RecvThreadState::WORKING;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, recv_thread_fun, NULL);
  nvbit_set_tool_pthread(channel_host.get_thread());
}

// Original NVIDIA implementation
void nvbit_at_ctx_term(CUcontext ctx) {
  skip_callback_flag = true;
  /* Notify receiver thread and wait for receiver thread to
   * notify back */
  recv_thread_done = RecvThreadState::STOP;
  while (recv_thread_done != RecvThreadState::FINISHED);
  channel_host.destroy(false);
  skip_callback_flag = false;
}
