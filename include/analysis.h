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

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <stdint.h>

#include <map>
#include <string>
#include <vector>

/* for channel */
#include "utils/channel.hpp"

/* Channel buffer size */
#define CHANNEL_SIZE (1l << 20)

/* Thread state enum */
enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};

struct CTXstate {
  /* context id */
  int id;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev *channel_dev;
  ChannelHost channel_host;

  // After initialization, set it to WORKING to make recv thread get data,
  // parent thread sets it to STOP to make recv thread stop working.
  // recv thread sets it to FINISHED when it cleans up.
  // parent thread should wait until the state becomes FINISHED to clean up.
  volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;
};

/* ===== Data Structures ===== */
struct TraceRecord {
  int opcode_id;
  uint64_t pc;
  std::vector<std::vector<uint32_t>> reg_values;  // [reg_idx][thread_idx]
  std::vector<uint32_t> ureg_values;              // [ureg_idx]
  std::vector<std::vector<uint64_t>> addrs;       // [thread_idx][addr_idx]
};

/* Structure to uniquely identify a warp */
struct WarpKey {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  // global warp id
  int warp_id;

  // Operator for map comparison
  bool operator<(const WarpKey &other) const {
    if (cta_id_x != other.cta_id_x) return cta_id_x < other.cta_id_x;
    if (cta_id_y != other.cta_id_y) return cta_id_y < other.cta_id_y;
    if (cta_id_z != other.cta_id_z) return cta_id_z < other.cta_id_z;
    return warp_id < other.warp_id;
  }

  // Hash function for unordered_map
  struct Hash {
    size_t operator()(const WarpKey &k) const {
      return (size_t)k.cta_id_x ^ ((size_t)k.cta_id_y << 10) ^ ((size_t)k.cta_id_z << 20) ^ ((size_t)k.warp_id << 30);
    }
  };

  // Equality operator for unordered_map
  bool operator==(const WarpKey &other) const {
    return cta_id_x == other.cta_id_x && cta_id_y == other.cta_id_y && cta_id_z == other.cta_id_z &&
           warp_id == other.warp_id;
  }
};

/* Structure to track the loop state of a warp */
struct WarpLoopState {
  std::vector<uint64_t> pcs;  // Circular buffer of PCs
  uint8_t head;               // Current position in circular buffer
  uint64_t last_sig;          // Last computed signature
  uint8_t last_period;        // Last loop period
  uint32_t repeat_cnt;        // Number of consecutive pattern repetitions
  bool loop_flag;             // Flag indicating loop detection
  time_t first_loop_time;     // Time when loop was first detected

  // Structure to store detailed information for each instruction in the loop
  struct LoopInstruction {
    uint64_t pc;                                    // Program counter
    int opcode_id;                                  // Instruction opcode ID
    std::vector<std::vector<uint32_t>> reg_values;  // Register values [reg_idx][thread_idx]
    std::vector<uint32_t> ureg_values;              // Unified register values
    std::vector<uint64_t> addrs;                    // Memory addresses (if any)
  };

  // Structure to hold complete loop information
  struct LoopInfo {
    std::vector<LoopInstruction> instructions;  // Complete instruction trace for the loop
    uint8_t period;                             // Loop period
  };
  LoopInfo current_loop;

  // Default constructor (using a default window size of 32)
  WarpLoopState() : pcs(32, 0), head(0), last_sig(0), repeat_cnt(0), loop_flag(false), first_loop_time(0) {}

  WarpLoopState(int window_size)
      : pcs(window_size, 0), head(0), last_sig(0), repeat_cnt(0), loop_flag(false), first_loop_time(0) {}
};

/* Receiver thread function */
void *recv_thread_fun(void *);

// Clear loop detection state when a kernel completes
void clear_loop_state();
#endif /* ANALYSIS_H */
