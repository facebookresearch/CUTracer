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

extern pthread_mutex_t mutex;
extern std::unordered_map<CUcontext, CTXstate *> ctx_state_map;
extern std::map<uint64_t, std::pair<CUcontext, CUfunction>> kernel_launch_to_func_map;
extern std::map<uint64_t, uint32_t> kernel_launch_to_iter_map;

std::string extract_instruction_name(const std::string &sass_line) {
  // SASS format examples:
  // /*0240*/                   CS2R.32 R7, SR_CLOCKLO ;
  // /*0250*/              @!P0 IMAD.MOV.U32 R6, RZ, RZ, 0x800000 ;

  // Find the start of the instruction part (after the address)
  size_t start_pos = sass_line.find("*/");
  if (start_pos == std::string::npos) {
    return "UNKNOWN";
  }
  start_pos += 2;  // Skip "*/"

  // Skip whitespace
  while (start_pos < sass_line.length() && isspace(sass_line[start_pos])) {
    start_pos++;
  }

  // Skip predicate if present (starts with @)
  if (start_pos < sass_line.length() && sass_line[start_pos] == '@') {
    // Find the end of predicate part (next space)
    while (start_pos < sass_line.length() && !isspace(sass_line[start_pos])) {
      start_pos++;
    }
    // Skip whitespace after predicate
    while (start_pos < sass_line.length() && isspace(sass_line[start_pos])) {
      start_pos++;
    }
  }

  // Extract instruction name (until first space)
  size_t end_pos = start_pos;
  while (end_pos < sass_line.length() && !isspace(sass_line[end_pos])) {
    end_pos++;
  }

  if (start_pos >= sass_line.length() || end_pos <= start_pos) {
    return "UNKNOWN";
  }

  return sass_line.substr(start_pos, end_pos - start_pos);
}

void process_instruction_histogram(const reg_info_t *ri, CTXstate *ctx_state,
                                   std::unordered_map<WarpKey, WarpState, WarpKey::Hash> &warp_states,
                                   std::vector<RegionHistogram> &completed_histograms) {
  // Get current function from kernel launch ID
  auto func_iter = kernel_launch_to_func_map.find(ri->kernel_launch_id);
  if (func_iter == kernel_launch_to_func_map.end()) {
    return;  // Unknown kernel, skip histogram processing
  }

  CUfunction current_func = func_iter->second.second;

  // Get clock opcode IDs for this function
  const std::unordered_set<int> *clock_opcode_ids = nullptr;
  if (ctx_state->clock_opcode_ids.count(current_func)) {
    clock_opcode_ids = &ctx_state->clock_opcode_ids.at(current_func);
  }

  // Get SASS mapping for this function
  const std::map<int, std::string> *sass_map_for_func = nullptr;
  if (ctx_state->id_to_sass_map.count(current_func)) {
    sass_map_for_func = &ctx_state->id_to_sass_map.at(current_func);
  }

  if (!clock_opcode_ids || !sass_map_for_func) {
    return;  // No mapping available for this function
  }

  WarpKey warp_key = {ri->cta_id_x, ri->cta_id_y, ri->cta_id_z, ri->warp_id};
  WarpState &current_state = warp_states[warp_key];
  bool is_clock_instruction = clock_opcode_ids->count(ri->opcode_id) > 0;

  if (is_clock_instruction) {
    if (current_state.is_collecting && !current_state.histogram.empty()) {
      completed_histograms.push_back({warp_key, current_state.region_counter, current_state.histogram});
      current_state.histogram.clear();
      current_state.region_counter++;
    }
    current_state.is_collecting = true;
  }

  if (current_state.is_collecting && sass_map_for_func->count(ri->opcode_id)) {
    // Extract instruction name from SASS string
    const std::string &sass_line = sass_map_for_func->at(ri->opcode_id);
    std::string instruction_name = extract_instruction_name(sass_line);
    current_state.histogram[instruction_name]++;
  }
}

void dump_previous_kernel_data(uint64_t kernel_launch_id, const std::vector<RegionHistogram> &histograms) {
  if (histograms.empty()) {
    return;  // Nothing to dump
  }

  // Find kernel info from global mapping
  if (kernel_launch_to_func_map.find(kernel_launch_id) != kernel_launch_to_func_map.end()) {
    auto [ctx, func] = kernel_launch_to_func_map[kernel_launch_id];
    uint32_t iteration = kernel_launch_to_iter_map[kernel_launch_id];

    // Use existing CSV generation logic
    dump_histograms_to_csv(ctx, func, iteration, histograms);

    // Clean up mapping tables to free memory
    kernel_launch_to_func_map.erase(kernel_launch_id);
    kernel_launch_to_iter_map.erase(kernel_launch_id);
  }
}

void dump_histograms_to_csv(CUcontext ctx, CUfunction func, uint32_t iteration,
                            const std::vector<RegionHistogram> &histograms) {
  if (histograms.empty()) {
    return;  // Nothing to dump
  }

  std::string basename = generate_kernel_log_basename(ctx, func, iteration);
  std::string csv_filename = basename + "_hist.csv";

  FILE *fp = fopen(csv_filename.c_str(), "w");
  if (!fp) {
    oprintf("ERROR: Could not open histogram file %s\n", csv_filename.c_str());
    return;
  }

  fprintf(fp, "cta_x,cta_y,cta_z,warp_id,region_id,instruction,count\n");
  for (const auto &region_result : histograms) {
    for (const auto &pair : region_result.histogram) {
      const std::string &instruction_name = pair.first;
      int count = pair.second;
      fprintf(fp, "%d,%d,%d,%d,%d,\"%s\",%d\n", region_result.warp_key.cta_id_x, region_result.warp_key.cta_id_y,
              region_result.warp_key.cta_id_z, region_result.warp_key.warp_id, region_result.region_id,
              instruction_name.c_str(), count);
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
  std::unordered_map<WarpKey, WarpState, WarpKey::Hash> warp_states;
  std::vector<RegionHistogram> local_completed_histograms;

  // Kernel boundary detection
  uint64_t last_seen_kernel_launch_id = UINT64_MAX;  // Initial invalid value
  /* ============================================ */

  while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);

    if (num_recv_bytes > 0) {
      // Process data packets in this chunk

      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        message_header_t *header = (message_header_t *)&recv_buffer[num_processed_bytes];

        const char *sass_str = "N/A";

        if (header->type == MSG_TYPE_REG_INFO) {
          reg_info_t *ri = (reg_info_t *)&recv_buffer[num_processed_bytes];

          // Kernel boundary detection - check for launch ID change
          if (ri->kernel_launch_id != last_seen_kernel_launch_id) {
            if (last_seen_kernel_launch_id != UINT64_MAX) {
              // Dump any remaining histograms for warps that were collecting
              for (auto &pair : warp_states) {
                if (pair.second.is_collecting && !pair.second.histogram.empty()) {
                  local_completed_histograms.push_back({pair.first, pair.second.region_counter, pair.second.histogram});
                }
              }

              // Dump the previous kernel's data
              dump_previous_kernel_data(last_seen_kernel_launch_id, local_completed_histograms);

              // Clear state for new kernel
              local_completed_histograms.clear();
              warp_states.clear();
            }
            last_seen_kernel_launch_id = ri->kernel_launch_id;
          }

          // Get SASS string for trace output
          auto func_iter = kernel_launch_to_func_map.find(ri->kernel_launch_id);
          if (func_iter != kernel_launch_to_func_map.end()) {
            CUfunction current_func = func_iter->second.second;
            if (ctx_state->id_to_sass_map.count(current_func) &&
                ctx_state->id_to_sass_map[current_func].count(ri->opcode_id)) {
              sass_str = ctx_state->id_to_sass_map[current_func][ri->opcode_id].c_str();
            }
          }
          trace_lprintf("CTX %p - CTA %d,%d,%d - warp %d - %s:\n", ctx, ri->cta_id_x, ri->cta_id_y, ri->cta_id_z,
                        ri->warp_id, sass_str);

          for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
            trace_lprintf("  * ");
            for (int i = 0; i < 32; i++) {
              trace_lprintf("Reg%d_T%02d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
            }
            trace_lprintf("\n");
          }
          trace_lprintf("\n");
          num_processed_bytes += sizeof(reg_info_t);

          /* ===== New feature: Instruction Histogram ===== */
          process_instruction_histogram(ri, ctx_state, warp_states, local_completed_histograms);
          /* ============================================ */

        } else if (header->type == MSG_TYPE_MEM_ACCESS) {
          mem_access_t *mem = (mem_access_t *)&recv_buffer[num_processed_bytes];

          // Get SASS string for trace output
          auto func_iter = kernel_launch_to_func_map.find(mem->kernel_launch_id);
          if (func_iter != kernel_launch_to_func_map.end()) {
            CUfunction current_func = func_iter->second.second;
            if (ctx_state->id_to_sass_map.count(current_func) &&
                ctx_state->id_to_sass_map[current_func].count(mem->opcode_id)) {
              sass_str = ctx_state->id_to_sass_map[current_func][mem->opcode_id].c_str();
            }
          }
          trace_lprintf(
              "CTX %p - grid_launch_id %ld - CTA %d,%d,%d - warp %d - PC %ld - "
              "%s:\n",
              ctx, mem->kernel_launch_id, mem->cta_id_x, mem->cta_id_y, mem->cta_id_z, mem->warp_id, mem->pc, sass_str);
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
        } else {
          // Unknown message type, print error and break loop
          fprintf(stderr,
                  "ERROR: Unknown message type %d received in recv_thread_fun. "
                  "Stopping processing of this chunk.\n",
                  header->type);
          continue;
        }
      }
    }
  }

  /* ===== Handle the last kernel's data ===== */
  // Dump any remaining histograms for warps that were collecting
  for (auto &pair : warp_states) {
    if (pair.second.is_collecting && !pair.second.histogram.empty()) {
      local_completed_histograms.push_back({pair.first, pair.second.region_counter, pair.second.histogram});
    }
  }

  // Dump the last kernel's data if any
  if (last_seen_kernel_launch_id != UINT64_MAX && !local_completed_histograms.empty()) {
    dump_previous_kernel_data(last_seen_kernel_launch_id, local_completed_histograms);
  }
  /* ========================================== */

  free(recv_buffer);
  ctx_state->recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}
