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
/* Warp Tracking */
// <global kernel launch id, <WarpKey, TraceRecord>>
std::map<uint64_t, std::map<WarpKey, std::vector<TraceRecord>>> warp_traces;
extern uint64_t global_kernel_launch_id;
std::map<WarpKey, WarpLoopState> loop_states;

extern int verbose;
// Loop detection configuration variables
extern int loop_win_size;
extern uint32_t loop_repeat_thresh;

// Updates the loop state for a warp based on its current PC.
//
// Args:
// - key: The warp key
// - pc: The program counter
// - current_trace: The current trace record
void update_loop_state(const WarpKey& key, uint64_t pc, const TraceRecord& current_trace) {

  if (loop_states.find(key) == loop_states.end()) loop_states.emplace(key, WarpLoopState(
    
  ));
  auto& state = loop_states[key];

  state.pcs[state.head] = pc;
  state.head = (state.head + 1) % loop_win_size;
  active_warps.insert(key);
  if (state.head != 0) return;

  uint8_t period = loop_win_size;
  for (uint8_t p = 1; p < loop_win_size; ++p) {
    bool ok = true;
    for (uint8_t i = p; i < loop_win_size && ok; ++i) ok &= (state.pcs[i] == state.pcs[i - p]);
    if (ok) {
      period = p;
      break;
    }
  }

  if (period == loop_win_size) {
    state.repeat_cnt = 0;
    state.loop_flag = false;
    state.last_sig = 0;
    return;
  }

  // Build rotation-invariant signature
  auto canonical_hash = [&](uint8_t P) -> uint64_t {
    // Find min_rot to make the first P pcs sorted
    uint8_t min_rot = 0;
    for (uint8_t r = 1; r < P; ++r) {
      for (uint8_t i = 0; i < P; ++i) {
        uint64_t a = state.pcs[(i + r) % P];
        uint64_t b = state.pcs[(i + min_rot) % P];
        if (a == b) continue;
        if (a < b) min_rot = r;
        break;
      }
    }
    // Build signature
    uint64_t h = 14695981039346656037ULL ^ P;
    for (uint8_t i = 0; i < P; ++i) {
      uint64_t pc_rot = state.pcs[(i + min_rot) % P];
      h = (h ^ pc_rot) * 1099511628211ULL;
    }
    return h;
  };

  uint64_t sig = canonical_hash(period);

  if (sig == state.last_sig) {
    if (++state.repeat_cnt >= loop_repeat_thresh && !state.loop_flag) {
      state.loop_flag = true;
      state.first_loop_time = time(nullptr);

      // Store detailed loop information
      state.current_loop.period = period;
      state.current_loop.instructions.clear();

      // Save complete instruction information for each instruction in the loop
      for (uint8_t i = 0; i < period; ++i) {
        WarpLoopState::LoopInstruction instr;
        instr.pc = state.pcs[i];
        instr.opcode_id = current_trace.opcode_id;
        instr.reg_values = current_trace.reg_values;
        instr.ureg_values = current_trace.ureg_values;
        state.current_loop.instructions.push_back(instr);
      }

      if (verbose >= 2) {
        trace_lprintf("Warp loop detected (P=%u): CTA %d,%d,%d warp %d\n", period, key.cta_id_x, key.cta_id_y, key.cta_id_z,
                 key.warp_id);
      }
    }
  } else {
    state.repeat_cnt = 1;
    state.loop_flag = false;
    state.last_sig = sig;
  }
}

// Based on NVIDIA NVBit record_reg_vals example with Meta modifications for message type and analysis support.
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

          trace_lprintf("CTX %p - grid_launch_id %ld - CTA %d,%d,%d - warp %d - %s:\n", ctx, global_kernel_launch_id, ri->cta_id_x,
                 ri->cta_id_y, ri->cta_id_z, ri->warp_id, (id_to_sass_map)[ri->opcode_id].c_str());

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
