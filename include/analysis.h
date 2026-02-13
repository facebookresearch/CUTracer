/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef ANALYSIS_H
#define ANALYSIS_H
#include <cstdint>
#include <ctime>
#include <deque>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "instr_category.h"
#include "nvbit.h"
/* for channel */
#include "utils/channel.hpp"

// Forward declaration for TraceWriter (defined in trace_writer.h)
class TraceWriter;

/* Channel buffer size - increased for mem_value_trace support */
#define CHANNEL_SIZE (1l << 22)  // 4MB

/* Thread state enum */
enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};

/* ===== Data Structures ===== */

/* Structure to uniquely identify a warp */
struct WarpKey {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  // global warp id
  int warp_id;

  // Operator for map comparison
  bool operator<(const WarpKey& other) const {
    if (cta_id_x != other.cta_id_x) return cta_id_x < other.cta_id_x;
    if (cta_id_y != other.cta_id_y) return cta_id_y < other.cta_id_y;
    if (cta_id_z != other.cta_id_z) return cta_id_z < other.cta_id_z;
    return warp_id < other.warp_id;
  }

  // Hash function for unordered_map
  struct Hash {
    size_t operator()(const WarpKey& k) const {
      return (size_t)k.cta_id_x ^ ((size_t)k.cta_id_y << 10) ^ ((size_t)k.cta_id_z << 20) ^ ((size_t)k.warp_id << 30);
    }
  };

  // Equality operator for unordered_map
  bool operator==(const WarpKey& other) const {
    return cta_id_x == other.cta_id_x && cta_id_y == other.cta_id_y && cta_id_z == other.cta_id_z &&
           warp_id == other.warp_id;
  }
};

// Merged trace record containing mandatory reg trace and optional mem trace
struct TraceRecordMerged {
  reg_info_t reg;
  bool has_mem = false;
  uint64_t mem_addrs[32] = {0};
};

// Structure to track the loop state of a warp
struct WarpLoopState {
  // Circular buffer of recent merged trace records
  std::vector<TraceRecordMerged> history;
  uint8_t head;    // Next write position in circular buffer
  uint8_t filled;  // Number of valid entries written, capped at buffer size
  uint64_t last_sig;
  uint8_t last_period;
  uint32_t repeat_cnt;
  bool loop_flag;
  time_t first_loop_time;

  // Structure to hold complete loop information
  struct LoopInfo {
    std::vector<TraceRecordMerged> instructions;  // Copy of one canonical period
    uint8_t period;
  };

  LoopInfo current_loop;

  WarpLoopState()
      : head(0), filled(0), last_sig(0), last_period(0), repeat_cnt(0), loop_flag(false), first_loop_time(0) {
  }
};

/**
 * @brief Represents the state of a single warp during instruction histogram
 * analysis.
 *
 * This structure tracks whether a warp is currently in a region of interest
 * for collection and stores the histogram data for that region.
 */
struct WarpState {
  /**
   * @brief A flag indicating whether instruction collection is active for this
   * warp.
   *
   * This acts as a switch, turned on by a "start" clock instruction and off
   * by an "end" clock instruction.
   */
  bool is_collecting = false;
  /**
   * @brief A counter for the number of regions analyzed for this warp.
   *
   * This helps in uniquely identifying each region within a warp's execution.
   */
  int region_counter = 0;
  /**
   * @brief The histogram of instructions collected for the current region.
   *
   * Maps an instruction name (string) to its execution count (int).
   */
  std::map<std::string, int> histogram;
};

/**
 * @brief Stores the completed instruction histogram for a specific region of a
 * warp.
 */
struct RegionHistogram {
  /**
   * @brief The ID of the warp.
   */
  int warp_id;
  /**
   * @brief The ID of the region within the warp.
   */
  int region_id;
  /**
   * @brief The completed histogram for this region.
   */
  std::map<std::string, int> histogram;
};

/**
 * @brief Grid and block dimensions for a kernel launch.
 */
struct KernelDimensions {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
};

/**
 * @brief Per-function static metadata collected once during instrumentation.
 *
 * Aggregates all per-function attributes that do not change across launches,
 * eliminating the need to re-query CUDA driver APIs on every launch.
 */
struct KernelFuncMetadata {
  std::string mangled_name;
  std::string unmangled_name;
  std::string kernel_checksum;  // FNV-1a hash hex string
  std::string cubin_path;       // Only set when dump_cubin is enabled
  uint64_t func_addr = 0;       // nvbit_get_func_addr()
  int nregs = 0;                // CU_FUNC_ATTRIBUTE_NUM_REGS
  int shmem_static_nbytes = 0;  // CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES

  /// Serialize per-function static attributes to JSON.
  nlohmann::json to_json() const {
    nlohmann::json j;
    j["mangled_name"] = mangled_name;
    j["unmangled_name"] = unmangled_name;
    j["kernel_checksum"] = kernel_checksum;
    std::ostringstream oss;
    oss << "0x" << std::hex << func_addr;
    j["func_addr"] = oss.str();
    j["nregs"] = nregs;
    j["shmem_static"] = shmem_static_nbytes;
    if (!cubin_path.empty()) {
      j["cubin_path"] = cubin_path;
    }
    return j;
  }
};

/**
 * @brief Tracks warp statistics for a single kernel launch.
 *
 * This structure maintains complete information about all warps in a kernel:
 * - Total number of warps (calculated from grid/block dimensions)
 * - All warps ever seen executing
 * - Warps that have finished execution
 * - Currently active warps (maintained elsewhere in CTXstate)
 */
struct KernelWarpStats {
  // Total number of warps in this kernel launch
  uint32_t total_warps;

  // Grid and block dimensions
  KernelDimensions dimensions;

  // All warps that have ever been observed executing
  std::unordered_set<WarpKey, WarpKey::Hash> all_seen_warps;

  // Warps that were once active but have finished
  std::unordered_set<WarpKey, WarpKey::Hash> finished_warps;

  KernelWarpStats() : total_warps(0) {
  }
};

/**
 * @brief Stores the completed instruction histogram for a specific region of a
 * warp.
 */
struct CTXstate {
  /* context id */
  int id;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev* channel_dev;
  ChannelHost channel_host;

  // After initialization, set it to WORKING to make recv thread get data,
  // parent thread sets it to STOP to make recv thread stop working.
  // recv thread sets it to FINISHED when it cleans up.
  // parent thread should wait until the state becomes FINISHED to clean up.
  volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

  // Per-function SASS mappings for instruction histogram feature
  std::unordered_map<CUfunction, std::map<int, std::string>> id_to_sass_map;
  // Per-function register indices mapping (static data, collected at instrumentation time)
  std::unordered_map<CUfunction, std::map<int, RegIndices>> id_to_reg_indices_map;
  std::unordered_map<CUfunction, std::unordered_set<int>> clock_opcode_ids;
  // Per-function EXIT opcode ids (statically identified at instrumentation time)
  std::unordered_map<CUfunction, std::unordered_set<int>> exit_opcode_ids;

  /* State for Deadlock/Hang Detection */
  std::map<WarpKey, WarpLoopState> loop_states;
  std::set<WarpKey> active_warps;
  time_t last_hang_check_time;

  // Pending mem traces per warp for out-of-order arrival (mem before reg)
  std::unordered_map<WarpKey, std::deque<mem_addr_access_t>, WarpKey::Hash> pending_mem_by_warp;

  // Per-warp activity timestamps for inactive cleanup
  std::unordered_map<WarpKey, time_t, WarpKey::Hash> last_seen_time_by_warp;
  std::unordered_map<WarpKey, time_t, WarpKey::Hash> exit_candidate_since_by_warp;

  // Per-warp last observed state: whether last instruction was BAR.SYNC.DEFER_BLOCKING
  std::unordered_map<WarpKey, bool, WarpKey::Hash> last_is_defer_blocking_by_warp;

  // Deadlock handling
  int deadlock_consecutive_hits = 0;
  bool deadlock_termination_initiated = false;

  // Warp statistics tracking per kernel launch
  std::unordered_map<uint64_t, KernelWarpStats> kernel_warp_tracking;

  // Per-launch TraceWriter map for JSON output (mode 1/2)
  // Maps kernel_launch_id -> TraceWriter*
  // Each kernel launch gets its own trace file
  std::map<uint64_t, TraceWriter*> trace_writers;
  mutable std::shared_mutex writers_mutex;

  // Per-kernel trace index counter (monotonically increasing)
  // Maps kernel_launch_id -> current trace_index
  std::unordered_map<uint64_t, uint64_t> trace_index_by_kernel;
};

// =================================================================================
// Function Declarations
// =================================================================================

/**
 * @brief The main thread function for receiving and processing data from the
 * GPU.
 *
 * This function runs in a separate CPU thread, continuously receiving data
 * packets (like `reg_info_t`, `mem_access_t`, `opcode_only_t`) from the GPU
 * channel and dispatching them for analysis.
 *
 * @param args A pointer to the `CUcontext` for which this thread is launched.
 * @return void*
 */
void* recv_thread_fun(void* args);

/**
 * @brief Writes a set of histograms to a formatted CSV file.
 *
 * @param ctx The CUDA context.
 * @param func The kernel function.
 * @param iteration The iteration number of the kernel launch.
 * @param histograms The histogram data to be written.
 */
void dump_histograms_to_csv(CUcontext ctx, CUfunction func, uint32_t iteration,
                            const std::vector<RegionHistogram>& histograms);

#endif /* ANALYSIS_H */
