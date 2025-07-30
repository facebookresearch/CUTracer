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
  std::vector<struct RegionHistogram> last_kernel_histograms;
  pthread_mutex_t histogram_mutex;
};

/* Receiver thread function */
void *recv_thread_fun(void *);
void dump_region_histograms_to_csv(CUcontext ctx, CUfunction func, uint32_t iteration);

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

struct RegionHistogram {
  struct WarpKey warp_key;
  int region_id;
  std::map<int, int> histogram;
};

#endif /* ANALYSIS_H */
