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

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "analysis.h"
#include "common.h"
#include "cuda.h"
#include "env_config.h"
#include "log.h"
#include "utils/channel.hpp"
extern std::map<int, std::string> id_to_sass_map;
extern pthread_mutex_t mutex;
extern std::unordered_map<CUcontext, CTXstate *> ctx_state_map;

void dump_region_histograms_to_csv(CUcontext ctx, CUfunction func, uint32_t iteration) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  pthread_mutex_lock(&ctx_state->histogram_mutex);
  if (ctx_state->last_kernel_histograms.empty()) {
    pthread_mutex_unlock(&ctx_state->histogram_mutex);
    return;  // Nothing to dump
  }

  std::string basename = generate_kernel_log_basename(ctx, func, iteration);
  std::string csv_filename = basename + "_hist.csv";

  auto histograms = std::move(ctx_state->last_kernel_histograms);
  ctx_state->last_kernel_histograms.clear();
  pthread_mutex_unlock(&ctx_state->histogram_mutex);

  FILE *fp = fopen(csv_filename.c_str(), "w");
  if (!fp) {
    oprintf("ERROR: Could not open histogram file %s\n", csv_filename.c_str());
    return;
  }

  fprintf(fp, "cta_x,cta_y,cta_z,warp_id,region_id,opcode_id,instruction,count\n");
  for (const auto &region_result : histograms) {
    for (const auto &pair : region_result.histogram) {
      int opcode_id = pair.first;
      int count = pair.second;
      fprintf(fp, "%d,%d,%d,%d,%d,%d,\"%s\",%d\n", region_result.warp_key.cta_id_x, region_result.warp_key.cta_id_y,
              region_result.warp_key.cta_id_z, region_result.warp_key.warp_id, region_result.region_id, opcode_id,
              id_to_sass_map.count(opcode_id) ? id_to_sass_map[opcode_id].c_str() : "N/A", count);
    }
  }
  fclose(fp);
  loprintf("Histogram data dumped to %s\n", csv_filename.c_str());
}

// Based on NVIDIA NVBit record_reg_vals example.
void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;

  pthread_mutex_lock(&mutex);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  ChannelHost *ch_host = &ctx_state->channel_host;
  pthread_mutex_unlock(&mutex);
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  /* ===== New feature: Instruction Histogram ===== */
  std::unordered_set<int> clock_opcode_ids;
  for (const auto &pair : id_to_sass_map) {
    if (strstr(pair.second.c_str(), "CS2R") && strstr(pair.second.c_str(), "SR_CLOCKLO")) {
      clock_opcode_ids.insert(pair.first);
    }
  }

  struct WarpState {
    bool is_collecting = false;
    std::map<int, int> histogram;
    int region_counter = 0;
  };

  std::unordered_map<WarpKey, WarpState, WarpKey::Hash> warp_states;
  std::vector<RegionHistogram> local_completed_histograms;
  /* ============================================ */

  while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);

    if (num_recv_bytes > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        message_header_t *header = (message_header_t *)&recv_buffer[num_processed_bytes];

        int cta_id_x = -1, cta_id_y = -1, cta_id_z = -1, warp_id = -1, opcode_id = -1;

        if (header->type == MSG_TYPE_REG_INFO) {
          reg_info_t *ri = (reg_info_t *)&recv_buffer[num_processed_bytes];
          trace_lprintf("CTX %p - CTA %d,%d,%d - warp %d - %s:\n", ctx, ri->cta_id_x, ri->cta_id_y, ri->cta_id_z,
                        ri->warp_id, (id_to_sass_map)[ri->opcode_id].c_str());

          for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
            trace_lprintf("  * ");
            for (int i = 0; i < 32; i++) {
              trace_lprintf("Reg%d_T%02d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
            }
            trace_lprintf("\n");
          }
          trace_lprintf("\n");
          num_processed_bytes += sizeof(reg_info_t);

          cta_id_x = ri->cta_id_x;
          cta_id_y = ri->cta_id_y;
          cta_id_z = ri->cta_id_z;
          warp_id = ri->warp_id;
          opcode_id = ri->opcode_id;

        } else if (header->type == MSG_TYPE_MEM_ACCESS) {
          mem_access_t *mem = (mem_access_t *)&recv_buffer[num_processed_bytes];
          trace_lprintf(
              "CTX %p - grid_launch_id %ld - CTA %d,%d,%d - warp %d - PC %ld - "
              "%s:\n",
              ctx, mem->kernel_launch_id, mem->cta_id_x, mem->cta_id_y, mem->cta_id_z, mem->warp_id, mem->pc,
              id_to_sass_map[mem->opcode_id].c_str());
          trace_lprintf("  Memory Addresses:\n  * ");
          int printed = 0;
          for (int i = 0; i < 32; i++) {
            if (mem->addrs[i] != 0) {  // Only print non-zero addresses
              trace_lprintf("T%02d: 0x%016lx ", i, mem->addrs[i]);
              printed++;
              if (printed % 4 == 0 && i < 31) {
                trace_lprintf("\n    ");
              }
            }
          }
          trace_lprintf("\n\n");
          num_processed_bytes += sizeof(mem_access_t);

          cta_id_x = mem->cta_id_x;
          cta_id_y = mem->cta_id_y;
          cta_id_z = mem->cta_id_z;
          warp_id = mem->warp_id;
          opcode_id = mem->opcode_id;
        } else {
          // Unknown message type, print error and break loop
          fprintf(stderr,
                  "ERROR: Unknown message type %d received in recv_thread_fun. "
                  "Stopping processing of this chunk.\n",
                  header->type);
          continue;
        }

        /* ===== New feature: Instruction Histogram ===== */
        WarpKey warp_key = {cta_id_x, cta_id_y, cta_id_z, warp_id};
        WarpState &current_state = warp_states[warp_key];
        bool is_clock_instruction = clock_opcode_ids.count(opcode_id) > 0;

        if (is_clock_instruction) {
          if (current_state.is_collecting && !current_state.histogram.empty()) {
            local_completed_histograms.push_back({warp_key, current_state.region_counter, current_state.histogram});
            current_state.histogram.clear();
            current_state.region_counter++;
          }
          current_state.is_collecting = true;
        }

        if (current_state.is_collecting) {
          current_state.histogram[opcode_id]++;
        }
        /* ============================================ */
      }
    }
  }

  /* ===== New feature: Instruction Histogram ===== */
  // Dump any remaining histograms at the end of the kernel
  for (auto &pair : warp_states) {
    if (pair.second.is_collecting && !pair.second.histogram.empty()) {
      local_completed_histograms.push_back({pair.first, pair.second.region_counter, pair.second.histogram});
    }
  }

  pthread_mutex_lock(&ctx_state->histogram_mutex);
  ctx_state->last_kernel_histograms = std::move(local_completed_histograms);
  pthread_mutex_unlock(&ctx_state->histogram_mutex);
  /* ============================================ */

  free(recv_buffer);
  ctx_state->recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}
