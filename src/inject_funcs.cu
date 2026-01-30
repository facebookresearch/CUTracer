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

#include <cuda_runtime.h>
#include <stdarg.h>
#include <stdint.h>

#include "utils/utils.h"

/* for channel */
#include "common.h"
#include "utils/channel.hpp"

/* Based on NVIDIA NVBit reg_trace example with Meta modifications for
message type, unified register, and kernel launch id support*/
extern "C" __device__ __noinline__ void instrument_reg_val(int pred, int opcode_id, uint64_t pchannel_dev,
                                                           uint64_t kernel_launch_id, uint64_t pc, int32_t num_regs,
                                                           int32_t num_uregs, ...) {
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  ri.header.type = MSG_TYPE_REG_INFO;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_global_warp_id();
  ri.opcode_id = opcode_id;
  ri.num_regs = num_regs;
  ri.num_uregs = num_uregs;
  ri.kernel_launch_id = kernel_launch_id;
  ri.pc = pc;

  if (num_regs || num_uregs) {
    // Initialize variable argument list
    va_list vl;
    va_start(vl, num_uregs);
    for (int i = 0; i < num_regs; i++) {
      uint32_t val = va_arg(vl, uint32_t);

      /* collect register values from other threads */
      for (int tid = 0; tid < 32; tid++) {
        ri.reg_vals[tid][i] = __shfl_sync(active_mask, val, tid);
      }
    }
    // Only the first thread in the warp needs to process unified registers
    if (first_laneid == laneid) {
      for (int i = 0; i < num_uregs; i++) {
        ri.ureg_vals[i] = va_arg(vl, uint32_t);
      }
    }

    va_end(vl);
  }

  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&ri, sizeof(reg_info_t));
  }
}

/* Based on NVIDIA NVBit mem_trace example with Meta modifications for message type */
extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id, uint64_t addr,
                                                       uint64_t kernel_launch_id, uint64_t pc, uint64_t pchannel_dev) {
  /* if thread is predicated off, return */
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  mem_addr_access_t ma;

  ma.header.type = MSG_TYPE_MEM_ADDR_ACCESS;

  /* collect memory address information from other threads */
  for (int i = 0; i < 32; i++) {
    ma.addrs[i] = __shfl_sync(active_mask, addr, i);
  }
  ma.kernel_launch_id = kernel_launch_id;
  int4 cta = get_ctaid();
  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.pc = pc;
  ma.warp_id = get_global_warp_id();
  ma.opcode_id = opcode_id;

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&ma, sizeof(mem_addr_access_t));
  }
}

extern "C" __device__ __noinline__ void instrument_opcode(int pred, int opcode_id, uint64_t pchannel_dev,
                                                          uint64_t kernel_launch_id, uint64_t pc) {
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  opcode_only_t oi;
  oi.header.type = MSG_TYPE_OPCODE_ONLY;

  int4 cta = get_ctaid();
  oi.cta_id_x = cta.x;
  oi.cta_id_y = cta.y;
  oi.cta_id_z = cta.z;
  oi.warp_id = get_global_warp_id();
  oi.opcode_id = opcode_id;
  oi.kernel_launch_id = kernel_launch_id;
  oi.pc = pc;

  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&oi, sizeof(opcode_only_t));
  }
}

/**
 * @brief Helper function to read a 32-bit value from shared memory using byte-level access.
 *
 * Uses __cvta_shared_to_generic to convert shared memory segment address
 * to a generic pointer, then reads bytes individually to handle arbitrary alignment.
 * This avoids "misaligned address" errors that occur when reading uint32_t from
 * non-4-byte-aligned addresses.
 *
 * @param smemAddr The shared memory segment address (not a generic pointer)
 * @return The 32-bit value at the given shared memory address (little-endian)
 */
__device__ __forceinline__ uint32_t loadSmemValue32(uint64_t smemAddr) {
  const auto ptr = static_cast<unsigned char*>(__cvta_shared_to_generic(static_cast<unsigned>(smemAddr)));
  // Read bytes individually and assemble into uint32_t (little-endian)
  uint32_t value = 0;
  value |= static_cast<uint32_t>(ptr[0]);
  value |= static_cast<uint32_t>(ptr[1]) << 8;
  value |= static_cast<uint32_t>(ptr[2]) << 16;
  value |= static_cast<uint32_t>(ptr[3]) << 24;
  return value;
}

/* Memory space constants matching InstrType::MemorySpace */
#define MEM_SPACE_NONE 0
#define MEM_SPACE_GLOBAL 1
#define MEM_SPACE_SHARED 4
#define MEM_SPACE_LOCAL 5

/**
 * @brief Device function to trace memory access with values.
 *
 * This function collects both memory addresses AND values for detailed data flow analysis.
 * It uses IPOINT_AFTER timing to capture values after the memory operation completes.
 *
 * For Global/Local memory: values are read from registers (passed as variadic args)
 * For Shared memory: values are read directly from memory using address space conversion
 *
 * @param pred Guard predicate value
 * @param opcode_id The opcode identifier for this instruction
 * @param addr Memory address accessed by this thread
 * @param access_size Access size in bytes (1, 2, 4, 8, 16)
 * @param mem_space Memory space type (GLOBAL=1, SHARED=4, LOCAL=5)
 * @param is_load 1 for load operations, 0 for store operations
 * @param num_regs Number of register values following (variadic)
 * @param pchannel_dev Pointer to the channel device
 * @param kernel_launch_id Global kernel launch identifier
 * @param pc Program counter (instruction offset)
 * @param ... Variadic register values (num_regs uint32_t values)
 */
extern "C" __device__ __noinline__ void instrument_mem_value(int pred, int opcode_id, uint64_t addr, int access_size,
                                                             int mem_space, int is_load, int num_regs,
                                                             uint64_t pchannel_dev, uint64_t kernel_launch_id,
                                                             uint64_t pc, ...) {
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  mem_value_access_t mv;
  mv.header.type = MSG_TYPE_MEM_VALUE_ACCESS;

  // Fill basic fields
  mv.kernel_launch_id = kernel_launch_id;
  int4 cta = get_ctaid();
  mv.cta_id_x = cta.x;
  mv.cta_id_y = cta.y;
  mv.cta_id_z = cta.z;
  mv.pc = pc;
  mv.warp_id = get_global_warp_id();
  mv.opcode_id = opcode_id;
  mv.mem_space = mem_space;
  mv.is_load = is_load;
  mv.access_size = access_size;

  // Collect addresses from all lanes
  for (int i = 0; i < 32; i++) {
    mv.addrs[i] = __shfl_sync(active_mask, addr, i);
  }

  // Initialize values array to zero
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 4; j++) {
      mv.values[i][j] = 0;
    }
  }

  // Collect values based on memory space type
  if (mem_space == MEM_SPACE_SHARED) {
    // Shared memory: use address space conversion to read values directly
    // Read up to 4 32-bit words based on access_size
    int words_to_read = (access_size + 3) / 4;
    if (words_to_read > 4) words_to_read = 4;

    uint32_t my_values[4] = {0, 0, 0, 0};
    if (addr != 0) {
      for (int w = 0; w < words_to_read; w++) {
        my_values[w] = loadSmemValue32(addr + w * 4);
      }
    }

    // Broadcast values from each lane
    for (int i = 0; i < 32; i++) {
      for (int w = 0; w < words_to_read; w++) {
        mv.values[i][w] = __shfl_sync(active_mask, my_values[w], i);
      }
    }
  } else {
    // Global/Local memory: read values from variadic register arguments
    if (num_regs > 0) {
      va_list vl;
      va_start(vl, pc);

      int regs_to_process = num_regs;
      if (regs_to_process > 4) regs_to_process = 4;

      for (int r = 0; r < regs_to_process; r++) {
        uint32_t val = va_arg(vl, uint32_t);
        // Broadcast this register value from all lanes
        for (int tid = 0; tid < 32; tid++) {
          mv.values[tid][r] = __shfl_sync(active_mask, val, tid);
        }
      }

      va_end(vl);
    }
  }

  // First active lane pushes the data to the channel
  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&mv, sizeof(mem_value_access_t));
  }
}

/**
 * @brief Device function to inject delay before any instrumented instruction.
 *
 * Injects a nanosleep delay before an instruction to expose potential race
 * conditions. The delay value is computed on the host during instrumentation,
 * so each instruction receives a unique, fixed delay value.
 *
 * Uses __nanosleep() intrinsic on SM 7.0+ (Volta and later), with a fallback
 * to busy-wait using clock64() on older architectures.
 *
 * @param pred Guard predicate value (from nvbit_add_call_arg_guard_pred_val)
 * @param delay_ns Delay in nanoseconds (passed from host, 0 = no delay)
 */
extern "C" __device__ __noinline__ void instrument_delay(int pred, uint32_t delay_ns) {
  if (!pred) {
    return;
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  __nanosleep(delay_ns);
#else
  // Fallback: busy-wait using clock64 (approximate 2 cycles per ns)
  uint64_t delay_cycles = delay_ns * 2;
  uint64_t start = clock64();
  while ((clock64() - start) < delay_cycles) {
  }
#endif
}
