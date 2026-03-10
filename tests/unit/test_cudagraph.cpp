/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

/**
 * @file test_cudagraph.cpp
 * @brief Unit tests for CUDA graph support logic.
 *
 * Tests the CUDA graph-related data structures and decision logic used by
 * CUTracer, including KernelDimensions, KernelWarpStats, and the
 * stream-capture / build-graph flag handling that determines whether
 * synchronization and metadata writes should be deferred.
 *
 * Compile and run:
 *   g++ -std=c++17 -I../../include -o test_cudagraph test_cudagraph.cpp
 *   ./test_cudagraph
 */

#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>

// Include the header that defines KernelDimensions, KernelWarpStats, WarpKey,
// KernelFuncMetadata, and CTXstate.
// We forward-declare / stub types it depends on so we can compile without CUDA.

// Stubs for CUDA / NVBit types that analysis.h references
typedef void* CUcontext;
typedef void* CUfunction;
typedef void* CUstream;
struct ChannelDev {
  void flush() {}
};
struct ChannelHost {
  void init(int, int, ChannelDev*, void* (*)(void*), CUcontext) {}
  void destroy(bool) {}
  pthread_t get_thread() { return {}; }
};
// Minimal Instr stub
struct Instr {};
// nvbit stubs
inline const char* nvbit_get_func_name(CUcontext, CUfunction, bool = false) { return "stub"; }
inline uint64_t nvbit_get_func_addr(CUcontext, CUfunction) { return 0; }
inline void nvbit_set_tool_pthread(pthread_t) {}
// common.h types
#include <cstdint>
struct reg_info_t {
  uint64_t kernel_launch_id;
  int32_t cta_id_x;
  int32_t cta_id_y;
  int32_t cta_id_z;
  int32_t warp_id;
  int32_t opcode_id;
  uint64_t pcAddr;
  uint32_t active_mask;
  uint64_t reg_vals[32];
  uint64_t ureg_vals[8];
  uint64_t pred_mask;
  uint64_t upred_mask;
};
struct mem_addr_access_t {
  uint64_t kernel_launch_id;
  int32_t cta_id_x;
  int32_t cta_id_y;
  int32_t cta_id_z;
  int32_t warp_id;
  int32_t opcode_id;
  uint64_t addrs[32];
};
struct opcode_only_t {};
struct mem_value_access_t {};
struct RegIndices {};

// Provide stub nlohmann::json so analysis.h compiles
namespace nlohmann {
class json {
 public:
  json() = default;
  json& operator[](const char*) { return *this; }
  json& operator[](const std::string&) { return *this; }
  json& operator=(const char*) { return *this; }
  json& operator=(const std::string&) { return *this; }
  json& operator=(int) { return *this; }
  json& operator=(uint64_t) { return *this; }
  json& operator=(std::initializer_list<unsigned int>) { return *this; }
  bool empty() const { return true; }
  std::string dump(int = -1) const { return "{}"; }
};
}  // namespace nlohmann

// Stub TraceWriter
class TraceWriter {};

// Now include the real header that defines the data structures we want to test
#include "instr_category.h"

// ---- Manually include the data structures from analysis.h ----
// (We cannot include analysis.h directly because it pulls in channel.hpp and
// nvbit.h which require CUDA headers. Instead we reproduce the exact structs.)

struct WarpKey {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warp_id;

  bool operator<(const WarpKey& other) const {
    if (cta_id_x != other.cta_id_x) return cta_id_x < other.cta_id_x;
    if (cta_id_y != other.cta_id_y) return cta_id_y < other.cta_id_y;
    if (cta_id_z != other.cta_id_z) return cta_id_z < other.cta_id_z;
    return warp_id < other.warp_id;
  }

  struct Hash {
    size_t operator()(const WarpKey& k) const {
      return (size_t)k.cta_id_x ^ ((size_t)k.cta_id_y << 10) ^
             ((size_t)k.cta_id_z << 20) ^ ((size_t)k.warp_id << 30);
    }
  };

  bool operator==(const WarpKey& other) const {
    return cta_id_x == other.cta_id_x && cta_id_y == other.cta_id_y &&
           cta_id_z == other.cta_id_z && warp_id == other.warp_id;
  }
};

struct KernelDimensions {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
};

struct KernelWarpStats {
  uint32_t total_warps;
  KernelDimensions dimensions;
  std::unordered_set<WarpKey, WarpKey::Hash> all_seen_warps;
  std::unordered_set<WarpKey, WarpKey::Hash> finished_warps;

  KernelWarpStats() : total_warps(0) {}
};

// ============================================================================
// Test helpers
// ============================================================================

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                             \
  std::cout << "Testing: " << #name << "... "; \
  if (test_##name()) {                         \
    std::cout << "PASSED" << std::endl;        \
    tests_passed++;                            \
  } else {                                     \
    std::cout << "FAILED" << std::endl;        \
    tests_failed++;                            \
  }

// ============================================================================
// Replicate the stream_capture / build_graph decision logic from cutracer.cu
// so we can unit-test the branching without a real GPU.
// ============================================================================

/**
 * Determines whether to synchronize the device before/after the kernel launch.
 * In CUTracer, synchronization is skipped when stream_capture or build_graph
 * is true because no kernel actually runs at that point.
 */
static bool should_sync_device(bool stream_capture, bool build_graph) {
  return !stream_capture && !build_graph;
}

/**
 * Determines whether to set launch arguments and write kernel metadata.
 * During stream capture or manual graph build the launch arguments are
 * deferred to the actual graph-node launch time.
 */
static bool should_set_launch_args(bool stream_capture, bool build_graph) {
  return !stream_capture && !build_graph;
}

/**
 * Determines whether kernel metadata should be written to the trace file
 * at enter_kernel_launch time. Metadata is only written for normal (non-graph)
 * launches; graph launches get metadata at nvbit_at_graph_node_launch time.
 */
static bool should_write_metadata_at_launch(bool stream_capture,
                                            bool build_graph) {
  return !stream_capture && !build_graph;
}

/**
 * Calculate total warp count from grid/block dimensions.
 * This matches the formula used throughout CUTracer for warp tracking.
 */
static uint32_t calculate_total_warps(const KernelDimensions& dims) {
  uint32_t total_threads_per_block = dims.blockDimX * dims.blockDimY * dims.blockDimZ;
  uint32_t warps_per_block = (total_threads_per_block + 31) / 32;
  uint32_t total_blocks = dims.gridDimX * dims.gridDimY * dims.gridDimZ;
  return total_blocks * warps_per_block;
}

// ============================================================================
// Test Cases: Stream capture / build graph decision logic
// ============================================================================

bool test_normal_launch_should_sync() {
  // Normal launch: both flags false -> should sync
  return should_sync_device(false, false) == true;
}

bool test_stream_capture_should_not_sync() {
  // Stream capture: no actual kernel launch, skip sync
  return should_sync_device(true, false) == false;
}

bool test_build_graph_should_not_sync() {
  // Manual graph build (cuGraphAddKernelNode): skip sync
  return should_sync_device(false, true) == false;
}

bool test_both_flags_should_not_sync() {
  // Both flags true: definitely skip sync
  return should_sync_device(true, true) == false;
}

bool test_normal_launch_writes_metadata() {
  return should_write_metadata_at_launch(false, false) == true;
}

bool test_stream_capture_defers_metadata() {
  return should_write_metadata_at_launch(true, false) == false;
}

bool test_build_graph_defers_metadata() {
  return should_write_metadata_at_launch(false, true) == false;
}

bool test_normal_launch_sets_args() {
  return should_set_launch_args(false, false) == true;
}

bool test_stream_capture_defers_args() {
  return should_set_launch_args(true, false) == false;
}

bool test_build_graph_defers_args() {
  return should_set_launch_args(false, true) == false;
}

// ============================================================================
// Test Cases: KernelDimensions
// ============================================================================

bool test_kernel_dimensions_1d_grid() {
  KernelDimensions dims = {128, 1, 1, 256, 1, 1};
  // 256 threads per block = 8 warps, 128 blocks = 1024 warps
  return calculate_total_warps(dims) == 128 * 8;
}

bool test_kernel_dimensions_3d_grid() {
  KernelDimensions dims = {4, 4, 4, 32, 1, 1};
  // 32 threads = 1 warp per block, 4*4*4 = 64 blocks
  return calculate_total_warps(dims) == 64;
}

bool test_kernel_dimensions_non_warp_aligned() {
  KernelDimensions dims = {1, 1, 1, 33, 1, 1};
  // 33 threads -> ceil(33/32) = 2 warps, 1 block
  return calculate_total_warps(dims) == 2;
}

bool test_kernel_dimensions_single_thread() {
  KernelDimensions dims = {1, 1, 1, 1, 1, 1};
  // 1 thread -> 1 warp, 1 block
  return calculate_total_warps(dims) == 1;
}

bool test_kernel_dimensions_full_block() {
  KernelDimensions dims = {1, 1, 1, 1024, 1, 1};
  // 1024 threads = 32 warps
  return calculate_total_warps(dims) == 32;
}

bool test_kernel_dimensions_3d_block() {
  KernelDimensions dims = {2, 1, 1, 8, 8, 4};
  // 8*8*4 = 256 threads = 8 warps per block, 2 blocks = 16 warps
  return calculate_total_warps(dims) == 16;
}

// ============================================================================
// Test Cases: WarpKey
// ============================================================================

bool test_warp_key_equality() {
  WarpKey a = {0, 0, 0, 0};
  WarpKey b = {0, 0, 0, 0};
  return a == b;
}

bool test_warp_key_inequality() {
  WarpKey a = {0, 0, 0, 0};
  WarpKey b = {0, 0, 0, 1};
  return !(a == b);
}

bool test_warp_key_ordering() {
  WarpKey a = {0, 0, 0, 0};
  WarpKey b = {1, 0, 0, 0};
  return a < b && !(b < a);
}

bool test_warp_key_hash_distinct() {
  WarpKey::Hash hasher;
  WarpKey a = {0, 0, 0, 0};
  WarpKey b = {1, 0, 0, 0};
  // Different keys should produce different hashes (not guaranteed in
  // general, but for these simple values they should differ).
  return hasher(a) != hasher(b);
}

bool test_warp_key_in_unordered_set() {
  std::unordered_set<WarpKey, WarpKey::Hash> warps;
  warps.insert({0, 0, 0, 0});
  warps.insert({0, 0, 0, 1});
  warps.insert({0, 0, 0, 0});  // duplicate
  return warps.size() == 2;
}

// ============================================================================
// Test Cases: KernelWarpStats
// ============================================================================

bool test_warp_stats_default_init() {
  KernelWarpStats stats;
  return stats.total_warps == 0 && stats.all_seen_warps.empty() &&
         stats.finished_warps.empty();
}

bool test_warp_stats_tracking() {
  KernelWarpStats stats;
  stats.total_warps = 4;
  stats.dimensions = {2, 1, 1, 64, 1, 1};

  WarpKey w0 = {0, 0, 0, 0};
  WarpKey w1 = {0, 0, 0, 1};
  WarpKey w2 = {1, 0, 0, 0};

  stats.all_seen_warps.insert(w0);
  stats.all_seen_warps.insert(w1);
  stats.all_seen_warps.insert(w2);
  stats.finished_warps.insert(w0);

  if (stats.all_seen_warps.size() != 3) return false;
  if (stats.finished_warps.size() != 1) return false;
  if (stats.finished_warps.count(w0) != 1) return false;
  if (stats.finished_warps.count(w1) != 0) return false;
  return true;
}

bool test_warp_stats_all_finished() {
  KernelWarpStats stats;
  stats.total_warps = 2;

  WarpKey w0 = {0, 0, 0, 0};
  WarpKey w1 = {0, 0, 0, 1};

  stats.all_seen_warps.insert(w0);
  stats.all_seen_warps.insert(w1);
  stats.finished_warps.insert(w0);
  stats.finished_warps.insert(w1);

  return stats.all_seen_warps.size() == stats.finished_warps.size();
}

// ============================================================================
// Test Cases: Kernel launch ID mapping (map-based tracking)
// ============================================================================

bool test_kernel_launch_to_func_map() {
  // Simulate the map<uint64_t, pair<CUcontext, CUfunction>> structure
  std::map<uint64_t, std::pair<CUcontext, CUfunction>> launch_map;

  CUcontext ctx1 = (CUcontext)0x1;
  CUfunction func1 = (CUfunction)0x100;

  launch_map[0] = {ctx1, func1};
  launch_map[1] = {ctx1, func1};
  launch_map[2] = {ctx1, func1};

  if (launch_map.size() != 3) return false;
  if (launch_map[0].first != ctx1) return false;
  if (launch_map[0].second != func1) return false;
  return true;
}

bool test_kernel_launch_to_iter_map() {
  // Simulate the iteration-count tracking used for graph launches
  std::map<uint64_t, uint32_t> iter_map;
  std::map<CUfunction, uint32_t> kernel_iter;

  CUfunction func1 = (CUfunction)0x100;
  CUfunction func2 = (CUfunction)0x200;

  // First launch of func1
  iter_map[0] = kernel_iter[func1]++;
  // Second launch of func1
  iter_map[1] = kernel_iter[func1]++;
  // First launch of func2
  iter_map[2] = kernel_iter[func2]++;

  if (iter_map[0] != 0) return false;  // func1 iter 0
  if (iter_map[1] != 1) return false;  // func1 iter 1
  if (iter_map[2] != 0) return false;  // func2 iter 0
  return true;
}

bool test_kernel_dimensions_map() {
  // Simulate graph-node dimension storage
  std::map<uint64_t, KernelDimensions> dims_map;

  dims_map[0] = {4, 1, 1, 128, 1, 1};
  dims_map[1] = {8, 2, 1, 256, 1, 1};

  if (dims_map[0].gridDimX != 4) return false;
  if (dims_map[1].blockDimX != 256) return false;
  return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "CUDA Graph Support Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  // Stream capture / build graph decision logic
  TEST(normal_launch_should_sync);
  TEST(stream_capture_should_not_sync);
  TEST(build_graph_should_not_sync);
  TEST(both_flags_should_not_sync);
  TEST(normal_launch_writes_metadata);
  TEST(stream_capture_defers_metadata);
  TEST(build_graph_defers_metadata);
  TEST(normal_launch_sets_args);
  TEST(stream_capture_defers_args);
  TEST(build_graph_defers_args);

  // KernelDimensions / warp calculation
  TEST(kernel_dimensions_1d_grid);
  TEST(kernel_dimensions_3d_grid);
  TEST(kernel_dimensions_non_warp_aligned);
  TEST(kernel_dimensions_single_thread);
  TEST(kernel_dimensions_full_block);
  TEST(kernel_dimensions_3d_block);

  // WarpKey
  TEST(warp_key_equality);
  TEST(warp_key_inequality);
  TEST(warp_key_ordering);
  TEST(warp_key_hash_distinct);
  TEST(warp_key_in_unordered_set);

  // KernelWarpStats
  TEST(warp_stats_default_init);
  TEST(warp_stats_tracking);
  TEST(warp_stats_all_finished);

  // Kernel launch mapping
  TEST(kernel_launch_to_func_map);
  TEST(kernel_launch_to_iter_map);
  TEST(kernel_dimensions_map);

  std::cout << "========================================" << std::endl;
  std::cout << "Results: " << tests_passed << " passed, " << tests_failed
            << " failed" << std::endl;
  std::cout << "========================================" << std::endl;

  return tests_failed > 0 ? 1 : 0;
}
