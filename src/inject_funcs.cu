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

#include <stdarg.h>
#include <stdint.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* Based on NVIDIA code with Meta modifications for message type and unified register support */
#include "common.h"
extern "C" __device__ __noinline__ void record_reg_val(int pred, int opcode_id, uint64_t pchannel_dev, uint64_t pc,
                                                       int32_t num_regs, int32_t num_uregs, ...) {
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
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ri, sizeof(reg_info_t));
  }
}