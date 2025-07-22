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

#include <stdio.h>
#include <stdlib.h>

#include <unordered_map>

#include "analysis.h"
#include "common.h"
#include "cuda.h"
#include "log.h"
#include "utils/channel.hpp"
extern std::map<int, std::string> id_to_sass_map;
extern pthread_mutex_t mutex;
extern std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

// Based on NVIDIA NVBit record_reg_vals example.
void* recv_thread_fun(void* args) {
  CUcontext ctx = (CUcontext)args;

  pthread_mutex_lock(&mutex);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  ChannelHost* ch_host = &ctx_state->channel_host;
  pthread_mutex_unlock(&mutex);
  char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

  while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);

    if (num_recv_bytes > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        // First read the message header to determine the message type
        message_header_t* header = (message_header_t*)&recv_buffer[num_processed_bytes];
        if (header->type == MSG_TYPE_REG_INFO) {
          reg_info_t* ri = (reg_info_t*)&recv_buffer[num_processed_bytes];

          /* when we get this cta_id_x it means the kernel has completed */
          if (ri->cta_id_x == -1) {
            break;
          }
          trace_lprintf("CTX %p - CTA %d,%d,%d - warp %d - %s:\n", ctx, ri->cta_id_x, ri->cta_id_y, ri->cta_id_z,
                        ri->warp_id, (id_to_sass_map)[ri->opcode_id].c_str());

          // Print register values
          for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
            trace_lprintf("  * ");
            for (int i = 0; i < 32; i++) {
              trace_lprintf("Reg%d_T%d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
            }
            trace_lprintf("\n");
          }
          trace_lprintf("\n");
          num_processed_bytes += sizeof(reg_info_t);
        } else if (header->type == MSG_TYPE_MEM_ACCESS) {
          // Process memory access message
          mem_access_t* mem = (mem_access_t*)&recv_buffer[num_processed_bytes];
          trace_lprintf("CTX %p - grid_launch_id %ld - CTA %d,%d,%d - warp %d - PC %ld - %s:\n", ctx,
                        mem->grid_launch_id, mem->cta_id_x, mem->cta_id_y, mem->cta_id_z, mem->warp_id, mem->pc,
                        id_to_sass_map[mem->opcode_id].c_str());
          trace_lprintf("  Memory Addresses:\n  * ");
          int printed = 0;
          for (int i = 0; i < 32; i++) {
            if (mem->addrs[i] != 0) {  // Only print non-zero addresses
              trace_lprintf("T%d: 0x%016lx ", i, mem->addrs[i]);
              printed++;
              if (printed % 4 == 0 && i < 31) {
                trace_lprintf("\n    ");
              }
            }
          }
          trace_lprintf("\n\n");
          num_processed_bytes += sizeof(mem_access_t);
        }
      }
    }
  }
  free(recv_buffer);
  ctx_state->recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}
