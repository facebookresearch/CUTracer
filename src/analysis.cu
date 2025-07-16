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

#include "analysis.h"
#include "common.h"
#include "utils/channel.hpp"

/* Global pointers to dependencies - initialized by init_recv_thread_deps */
static ChannelHost* g_channel_host = nullptr;
static volatile RecvThreadState* g_recv_thread_done = nullptr;
static std::map<int, std::string>* g_id_to_sass_map = nullptr;

/* Initialize receiver thread dependencies */
void init_recv_thread_deps(ChannelHost* host, volatile RecvThreadState* thread_state,
                           std::map<int, std::string>* sass_map) {
  g_channel_host = host;
  g_recv_thread_done = thread_state;
  g_id_to_sass_map = sass_map;
}

// Based on NVIDIA code
void* recv_thread_fun(void*) {
  char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

  while (*g_recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = g_channel_host->recv(recv_buffer, CHANNEL_SIZE);

    if (num_recv_bytes > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        reg_info_t* ri = (reg_info_t*)&recv_buffer[num_processed_bytes];

        /* when we get this cta_id_x it means the kernel has completed */
        if (ri->cta_id_x == -1) {
          break;
        }

        // Simple instruction trace output
        printf("CTA %d,%d,%d - warp %d - PC %ld - %s:\n", ri->cta_id_x, ri->cta_id_y, ri->cta_id_z, ri->warp_id, ri->pc,
               (*g_id_to_sass_map)[ri->opcode_id].c_str());

        // Print register values
        for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
          printf("  * ");
          for (int i = 0; i < 32; i++) {
            printf("Reg%d_T%d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
          }
          printf("\n");
        }
        printf("\n");
        num_processed_bytes += sizeof(reg_info_t);
      }
    }
  }
  free(recv_buffer);
  *g_recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}
