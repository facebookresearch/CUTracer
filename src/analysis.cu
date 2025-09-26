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

#include <assert.h>
#include <pthread.h>
#include <signal.h>
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

#define PC_HISTORY_LEN 32
#define LOOP_REPEAT_THRESH 3
// Throttle interval for hang checks (seconds)
#define HANG_CHECK_THROTTLE_SECS 1

extern pthread_mutex_t mutex;
extern std::unordered_map<CUcontext, CTXstate *> ctx_state_map;
extern std::map<uint64_t, std::pair<CUcontext, CUfunction>> kernel_launch_to_func_map;
extern std::map<uint64_t, uint32_t> kernel_launch_to_iter_map;

// Forward declaration for helper defined below in this file
std::string extract_instruction_name(const std::string &sass_line);

/**
 * Print one instruction line in the exact same format as loop body entries.
 *
 * Purpose:
 * - Provide a single-line renderer that matches the loop body formatting so
 *   different states (looping/barrier/progressing) can share the same
 *   presentation logic for an instruction row.
 *
 * Output format (aligned columns):
 *   Example: [idx] PC <dec>; Offset <dec> [0x<hex>];  <SASS> (has_mem)\n
 * - The trailing "(has_mem)" is printed only when the merged record has
 *   associated memory info.
 * - The index field width is currently fixed to 1 for consistency with our
 *   existing loop body output. Adjust if you later support wider indices.
 *
 * Parameters:
 * - index: The row index to print inside square brackets. For barrier/
 *   progressing single-line outputs, callers typically pass 0.
 * - instr: The merged trace record containing the reg info (and optional mem).
 * - sass_map_for_func: Per-function opcode->SASS map to resolve the mnemonic.
 *   If null or missing entry, prints "UNKNOWN" as fallback.
 * - pc_dec_width: Precomputed decimal width for PC/Offset alignment.
 * - hex_nibbles_max: Precomputed hex width (in nibbles), rounded to multiples
 *   of 4 and clamped to [4,16], used for the 0x... field.
 *
 * Notes:
 * - This function does not perform width computation. Callers should compute
 *   widths across the intended set of rows (entire loop body, or just the last
 *   single record) and pass them here to ensure consistent alignment.
 */
static inline void print_instruction_line(size_t index, const TraceRecordMerged &instr,
                                          const std::map<int, std::string> *sass_map_for_func, int pc_dec_width,
                                          int hex_nibbles_max) {
  uint64_t pc_val = instr.reg.pc;
  const char *sass_cstr = "UNKNOWN";
  if (sass_map_for_func && sass_map_for_func->count(instr.reg.opcode_id)) {
    sass_cstr = sass_map_for_func->at(instr.reg.opcode_id).c_str();
  }
  loprintf("        [%*zu] PC %*lu; Offset %*lu /*0x%0*lx*/;  %s", 1, index, pc_dec_width, (unsigned long)pc_val,
           pc_dec_width, (unsigned long)pc_val, hex_nibbles_max, (unsigned long)pc_val, sass_cstr);
  if (instr.has_mem) {
    loprintf(" (has_mem)");
  }
  loprintf("\n");
}

/**
 * Print the most recent instruction for a warp as a single line (loop-body style).
 *
 * Purpose:
 * - Provide a one-liner for BARRIER/PROGRESSING states that mirrors the loop
 *   body row format for visual consistency with LOOPING output.
 *
 * Behavior:
 * - Fetches the latest entry from the per-warp ring buffer
 *   ((head + filled - 1) % PC_HISTORY_LEN).
 * - Computes widths (decimal/hex) based solely on this last record and prints
 *   a single formatted line prefixed by "last: ".
 * - If there is no available entry (null state or empty), prints a line with
 *   UNKNOWN mnemonic and zero PC/Offset.
 *
 * Parameters:
 * - loop_state: Pointer to the warp's loop state holding the ring buffer.
 * - sass_map_for_func: Per-function opcode->SASS map to resolve the mnemonic.
 *
 * Notes:
 * - This function intentionally recomputes widths for the single last record.
 *   For multi-row loop bodies, prefer computing widths once across all rows and
 *   calling print_instruction_line_like_loop per row.
 */
static inline void print_last_instruction_line(const WarpLoopState *loop_state,
                                               const std::map<int, std::string> *sass_map_for_func) {
  uint64_t last_pc_val = 0;
  bool last_has_mem = false;
  const char *last_sass_cstr = "UNKNOWN";

  if (loop_state && loop_state->filled > 0 && sass_map_for_func) {
    int idx_last = (loop_state->head + (int)loop_state->filled + PC_HISTORY_LEN - 1) % PC_HISTORY_LEN;
    int opcode_last = loop_state->history[idx_last].reg.opcode_id;
    last_pc_val = loop_state->history[idx_last].reg.pc;
    last_has_mem = loop_state->history[idx_last].has_mem;
    if (sass_map_for_func->count(opcode_last)) {
      last_sass_cstr = sass_map_for_func->at(opcode_last).c_str();
    }
  }

  int pc_dec_width = 1;
  int hex_nibbles_max = 4;
  {
    uint64_t td = last_pc_val;
    int dec_w = 1;
    while (td >= 10) {
      td /= 10;
      dec_w++;
    }
    if (dec_w > pc_dec_width) pc_dec_width = dec_w;
    int nibbles = 1;
    if (last_pc_val != 0) {
      nibbles = 0;
      uint64_t th = last_pc_val;
      while (th) {
        th >>= 4;
        nibbles++;
      }
    }
    if (nibbles > hex_nibbles_max) hex_nibbles_max = nibbles;
    hex_nibbles_max = ((hex_nibbles_max + 3) / 4) * 4;
    if (hex_nibbles_max < 4) hex_nibbles_max = 4;
    if (hex_nibbles_max > 16) hex_nibbles_max = 16;
  }
  loprintf("      Last: [%*d] PC %*lu; Offset %*lu /*0x%0*lx*/;  %s", 1, 0, pc_dec_width, (unsigned long)last_pc_val,
           pc_dec_width, (unsigned long)last_pc_val, hex_nibbles_max, (unsigned long)last_pc_val, last_sass_cstr);
  if (last_has_mem) {
    loprintf(" (has_mem)");
  }
}

static inline bool matches_barrier_defer_blocking(const std::string &mnemonic) {
  if (mnemonic == "BAR.SYNC.DEFER_BLOCKING") return true;
  // Conservative fallback: prefix BAR.SYNC and contains .DEFER_BLOCKING
  if (mnemonic.rfind("BAR.SYNC", 0) == 0 && mnemonic.find(".DEFER_BLOCKING") != std::string::npos) return true;
  return false;
}

static bool is_barrier_defer_blocking_for_opcode(CTXstate *ctx_state, CUfunction func, int opcode_id) {
  if (!ctx_state) return false;
  if (!ctx_state->id_to_sass_map.count(func)) return false;
  const std::map<int, std::string> &sass_map = ctx_state->id_to_sass_map[func];
  std::map<int, std::string>::const_iterator it = sass_map.find(opcode_id);
  if (it == sass_map.end()) return false;
  const std::string &sass_line = it->second;
  std::string mnemonic = extract_instruction_name(sass_line);
  return matches_barrier_defer_blocking(mnemonic);
}

/**
 * @brief Extracts the full instruction mnemonic from a SASS line.
 *
 * This function parses a SASS instruction string to extract the mnemonic,
 * which includes the base instruction and any dot-separated modifiers
 * (e.g., "IMAD.MOV.U32"). It correctly handles and skips optional
 * predicates (e.g., "@!P0").
 *
 * @param sass_line The full SASS instruction line.
 * @return The extracted instruction mnemonic as a string.
 */
std::string extract_instruction_name(const std::string &sass_line) {
  // SASS format examples:
  // CS2R.32 R7, SR_CLOCKLO ;
  // @!P0 IMAD.MOV.U32 R6, RZ, RZ, 0x800000 ;

  size_t start_pos = 0;

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

/**
 * @brief Processes a single instruction packet for histogram analysis.
 *
 * This function is the core of the instruction histogram feature. It uses
 * special "clock" instructions (generated by `pl.scope`) as markers to define
 * regions of interest.
 *
 * The logic operates in a start/stop fashion:
 * - The first clock instruction encountered by a warp starts the collection.
 * - The second clock instruction stops the collection and saves the histogram for
 *   the completed region.
 * - The third starts a new region, the fourth stops it, and so on.
 *
 * @warning This start/stop model does not support nested `pl.scope` blocks.
 * A nested scope will be flattened into a single sequence of start/stop
 * markers, which may lead to unintended region definitions.
 *
 * @param ri Pointer to the received opcode data packet (`opcode_only_t`).
 * @param ctx_state Pointer to the state for the current CUDA context.
 * @param warp_states A map tracking the collection state of each warp.
 * @param completed_histograms A vector where histograms of completed regions are
 * stored.
 */
void process_instruction_histogram(const opcode_only_t *ri, CTXstate *ctx_state,
                                   std::unordered_map<int, WarpState> &warp_states,
                                   std::vector<RegionHistogram> &completed_histograms) {
  // Get current function from kernel launch ID to find the correct SASS maps.
  std::map<uint64_t, std::pair<CUcontext, CUfunction>>::iterator func_iter =
      kernel_launch_to_func_map.find(ri->kernel_launch_id);
  if (func_iter == kernel_launch_to_func_map.end()) {
    return;  // Unknown kernel, skip histogram processing
  }

  CUfunction current_func = func_iter->second.second;

  // Get clock opcode IDs for this function, which mark region boundaries.
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
    return;  // No SASS or clock instruction mapping available for this function.
  }

  int warp_id = ri->warp_id;
  WarpState &current_state = warp_states[warp_id];
  bool is_clock_instruction = clock_opcode_ids->count(ri->opcode_id) > 0;

  // This block implements the start/stop logic for regions.
  if (is_clock_instruction) {
    if (current_state.is_collecting) {
      // This is an "end" clock: the region is complete.
      if (!current_state.histogram.empty()) {
        // Save the completed histogram.
        completed_histograms.push_back({warp_id, current_state.region_counter, current_state.histogram});
        current_state.histogram.clear();
        current_state.region_counter++;
      }
      // Stop collecting until the next "start" clock is found.
      current_state.is_collecting = false;
    } else {
      // This is a "start" clock: begin collecting instructions.
      current_state.is_collecting = true;
    }
  }

  // If collection is active, record the current instruction.
  if (current_state.is_collecting && sass_map_for_func->count(ri->opcode_id)) {
    // Extract the base instruction name from the full SASS string.
    const std::string &sass_line = sass_map_for_func->at(ri->opcode_id);
    std::string instruction_name = extract_instruction_name(sass_line);
    current_state.histogram[instruction_name]++;
  }
}

/**
 * @brief Dumps the collected histograms for a completed kernel launch to a file.
 *
 * This function is called when a kernel boundary is detected (i.e., when a new
 * kernel_launch_id is seen). It collates all histograms from the *previous*
 * kernel launch and triggers the process to write them to a CSV file.
 *
 * @param kernel_launch_id The ID of the kernel launch that has just finished.
 * @param histograms A vector containing all the completed region histograms for
 * that kernel.
 */
void dump_previous_kernel_data(uint64_t kernel_launch_id, const std::vector<RegionHistogram> &histograms) {
  if (histograms.empty()) {
    return;  // Nothing to dump.
  }

  // Find kernel info from global mapping
  if (kernel_launch_to_func_map.find(kernel_launch_id) != kernel_launch_to_func_map.end()) {
    auto [ctx, func] = kernel_launch_to_func_map[kernel_launch_id];
    uint32_t iteration = kernel_launch_to_iter_map[kernel_launch_id];

    // Use existing CSV generation logic.
    dump_histograms_to_csv(ctx, func, iteration, histograms);

    // Clean up mapping tables to free memory for subsequent kernels.
    kernel_launch_to_func_map.erase(kernel_launch_id);
    kernel_launch_to_iter_map.erase(kernel_launch_id);
  }
}

/**
 * @brief Writes a set of histograms to a formatted CSV file.
 *
 * This function handles the file I/O for persisting the analysis results. It
 * creates a uniquely named CSV file for a given kernel launch and writes the
 * histogram data in a structured format.
 *
 * @param ctx The CUDA context of the kernel.
 * @param func The kernel function.
 * @param iteration The iteration number of the kernel launch.
 * @param histograms The histogram data to be written to the file.
 */
void dump_histograms_to_csv(CUcontext ctx, CUfunction func, uint32_t iteration,
                            const std::vector<RegionHistogram> &histograms) {
  if (histograms.empty()) {
    return;  // Nothing to dump.
  }

  std::string basename = generate_kernel_log_basename(ctx, func, iteration);
  std::string csv_filename = basename + "_hist.csv";

  FILE *fp = fopen(csv_filename.c_str(), "w");
  if (!fp) {
    oprintf("ERROR: Could not open histogram file %s\n", csv_filename.c_str());
    return;
  }

  // Header for the CSV file.
  fprintf(fp, "warp_id,region_id,instruction,count\n");
  // Iterate through each completed region and write its histogram data.
  for (const RegionHistogram &region_result : histograms) {
    for (const std::pair<const std::string, int> &pair : region_result.histogram) {
      const std::string &instruction_name = pair.first;
      int count = pair.second;
      fprintf(fp, "%d,%d,\"%s\",%d\n", region_result.warp_id, region_result.region_id, instruction_name.c_str(), count);
    }
  }
  fclose(fp);
  loprintf("Histogram data dumped to %s\n", csv_filename.c_str());
}

/**
 * @brief Extract kernel launch ID from different message types
 *
 * This helper function provides a unified interface to retrieve the kernel_launch_id
 * field from various message structures. It's used for kernel boundary detection
 * to determine when processing transitions from one CUDA kernel to another.
 *
 * @param header Pointer to the message header containing the message type
 * @return The kernel launch ID for the message, or 0 if the message type is unknown
 */
static uint64_t get_kernel_launch_id(const message_header_t *header) {
  switch (header->type) {
    case MSG_TYPE_REG_INFO:
      return ((const reg_info_t *)header)->kernel_launch_id;
    case MSG_TYPE_OPCODE_ONLY:
      return ((const opcode_only_t *)header)->kernel_launch_id;
    case MSG_TYPE_MEM_ACCESS:
      return ((const mem_access_t *)header)->kernel_launch_id;
    default:
      return 0;  // Invalid/unknown message type - no kernel ID available
  }
}

/**
 * @brief Computes a canonical signature for a sequence of PCs to detect loops.
 *
 * This function analyzes the recent history of Program Counters (PCs) for a warp
 * to identify repeating patterns, which indicate a loop. The process involves:
 * 1.  **Period Detection**: It finds the shortest repeating sequence of PCs in
 *     the history buffer. If no repeating pattern is found, it returns 0.
 * 2.  **Canonicalization**: To ensure that the same loop produces the same
 *     signature regardless of the entry point, it finds the lexicographically
 *     smallest rotation of the detected period. For example, `[3,1,2]` becomes
 *     `[1,2,3]`.
 * 3.  **Hashing**: It computes an FNV-1a hash of the canonical sequence, seeded
 *     with the period length, to produce the final signature.
 *
 * @param history A constant reference to the ring buffer of merged trace records.
 * @param ring_size The total size of the ring buffer (must be `PC_HISTORY_LEN`).
 * @param head The index of the oldest element in the ring buffer.
 * @param out_period A reference to a `uint8_t` that will be set to the detected
 *                   period length. If no period is found, it's set to 0.
 * @return A 64-bit canonical signature of the loop, or 0 if no loop is detected.
 */
static uint64_t compute_canonical_signature(const std::vector<TraceRecordMerged> &history, int ring_size, uint8_t head,
                                            uint8_t &out_period) {
  // Reconstruct linear PC sequence in chronological order (oldest -> newest).
  uint64_t pcs[PC_HISTORY_LEN];
  for (int i = 0; i < ring_size; ++i) {
    int idx = (head + i) % ring_size;  // head points to oldest element position
    pcs[i] = history[idx].reg.pc;
  }

  // Detect the shortest repeating period p (1..N-1) such that pcs[i] == pcs[i-p]
  uint8_t period = 0;
  for (uint8_t p = 1; p < ring_size; ++p) {
    bool match = true;
    for (uint8_t i = p; i < ring_size; ++i) {
      if (pcs[i] != pcs[i - p]) {
        match = false;
        break;
      }
    }
    if (match) {
      period = p;
      break;
    }
  }
  if (period == 0) {
    out_period = 0;
    return 0;
  }

  // Find minimal rotation of the period segment to get canonical representation
  // Build candidate of length period from the first period entries
  uint64_t seg[PC_HISTORY_LEN];
  for (uint8_t i = 0; i < period; ++i) seg[i] = pcs[i];
  uint8_t min_rot = 0;
  for (uint8_t r = 1; r < period; ++r) {
    // Compare rotation r with current min_rot lexicographically
    bool smaller = false;
    for (uint8_t i = 0; i < period; ++i) {
      uint64_t a = seg[(i + r) % period];
      uint64_t b = seg[(i + min_rot) % period];
      if (a == b) continue;
      if (a < b) smaller = true;
      break;
    }
    if (smaller) min_rot = r;
  }

  // FNV-1a style hash seeded by period for better distribution
  const uint64_t FNV_OFFSET = 14695981039346656037ULL;
  const uint64_t FNV_PRIME = 1099511628211ULL;
  uint64_t h = FNV_OFFSET ^ period;
  for (uint8_t i = 0; i < period; ++i) {
    uint64_t pcv = seg[(i + min_rot) % period];
    h = (h ^ pcv) * FNV_PRIME;
  }
  out_period = period;
  return h;
}

/**
 * @brief Updates the loop detection state for a given warp.
 *
 * This function is the core of the host-side loop detection logic. It maintains
 * a history of instructions for each warp and uses it to detect when a warp
 * enters a stable loop.
 *
 * The process for each new instruction is as follows:
 * 1.  **History Update**: The new instruction record (`reg_info_t`) is added to
 *     the warp's ring buffer. The function also tries to match it with any
 *     pending memory access records for the same instruction.
 * 2.  **Signature Calculation**: Once the history buffer is full, it calls
 *     `compute_canonical_signature` to get a signature of the current PC sequence.
 * 3.  **Loop State Tracking**:
 *     - If the new signature and period match the previous one, a `repeat_cnt`
 *       is incremented.
 *     - If they don't match, the counter is reset.
 *     - When `repeat_cnt` exceeds `LOOP_REPEAT_THRESH`, the warp is officially
 *       considered to be in a loop (`loop_flag` is set to true), and the loop
 *       body (one full period) is captured and stored.
 *
 * @param ctx_state Pointer to the state for the current CUDA context.
 * @param key The `WarpKey` identifying the warp to be updated.
 * @param ri Pointer to the `reg_info_t` packet for the current instruction.
 */
static void update_loop_state(CTXstate *ctx_state, const WarpKey &key, const reg_info_t *ri) {
  WarpLoopState &state = ctx_state->loop_states[key];
  // One-time buffer allocation per warp
  if (state.history.size() != PC_HISTORY_LEN) {
    state.history.assign(PC_HISTORY_LEN, TraceRecordMerged());
    state.head = 0;
    state.filled = 0;
  }

  // Write the incoming reg record into ring buffer
  TraceRecordMerged &slot = state.history[state.head];
  slot.reg = *ri;
  slot.has_mem = false;  // will be flipped if we find a matching pending mem
  memset(slot.mem_addrs, 0, sizeof(slot.mem_addrs));

  // Try to match any pending mem for this warp (mem may arrive before reg)
  auto &pending = ctx_state->pending_mem_by_warp[key];
  if (!pending.empty()) {
    // Find first matching mem by pc/opcode
    for (auto it = pending.begin(); it != pending.end(); ++it) {
      if (it->pc == ri->pc && it->opcode_id == ri->opcode_id) {
        slot.has_mem = true;
        memcpy(slot.mem_addrs, it->addrs, sizeof(slot.mem_addrs));
        pending.erase(it);
        break;
      }
    }
  }

  // Advance ring pointers
  state.head = (uint8_t)((state.head + 1) % PC_HISTORY_LEN);
  if (state.filled < PC_HISTORY_LEN) state.filled++;

  // Only check for loops once the history buffer is full
  if (state.filled < PC_HISTORY_LEN) {
    return;
  }

  // Compute canonical signature and period from the ring buffer
  uint8_t period = 0;
  uint64_t current_sig = compute_canonical_signature(state.history, PC_HISTORY_LEN, state.head, period);

  if (current_sig != 0 && current_sig == state.last_sig && period == state.last_period) {
    state.repeat_cnt++;
  } else {
    state.repeat_cnt = 1;  // current observed once
    state.loop_flag = false;
  }

  if (state.repeat_cnt > LOOP_REPEAT_THRESH) {
    if (!state.loop_flag) {
      state.loop_flag = true;
      state.first_loop_time = time(nullptr);
      // Capture the loop body records (one period) from the ring buffer in chronological order
      state.current_loop.period = period;
      state.current_loop.instructions.clear();
      state.current_loop.instructions.reserve(period);
      // head points to oldest, so sequence starts from head index
      for (uint8_t i = 0; i < period; ++i) {
        int idx = (state.head + i) % PC_HISTORY_LEN;
        state.current_loop.instructions.push_back(state.history[idx]);
      }
    }
  }
  state.last_sig = current_sig;
  state.last_period = period;
}

/**
 * @brief Clears all state related to deadlock and hang detection.
 *
 * This function is called at the boundary of a new kernel launch to ensure that
 * the state from the previous kernel does not interfere with the analysis of the
 * new one. It clears all maps and sets that track warp activity, loop states,
 * and pending memory operations.
 *
 * @param ctx_state Pointer to the state for the current CUDA context.
 */
static void clear_deadlock_state(CTXstate *ctx_state) {
  ctx_state->loop_states.clear();
  ctx_state->active_warps.clear();
  ctx_state->pending_mem_by_warp.clear();
  ctx_state->last_seen_time_by_warp.clear();
  ctx_state->exit_candidate_since_by_warp.clear();
  ctx_state->last_is_defer_blocking_by_warp.clear();
}

/**
 * @brief Prints detailed status information for all active warps including loop states.
 *
 * This function provides a comprehensive view of each warp's current state including:
 * - Basic warp identification (CTA coordinates, warp ID)
 * - Loop detection status (whether in loop, loop period, repeat count)
 * - Activity timestamps and exit candidate status
 * - Loop body instruction details if available
 *
 * @param ctx_state Pointer to the state for the current CUDA context
 * @param current_kernel_launch_id The current kernel launch ID for context
 */
static void print_warp_status_summary(CTXstate *ctx_state, uint64_t current_kernel_launch_id) {
  if (ctx_state->active_warps.empty()) {
    loprintf("==> WARP STATUS: No active warps for kernel_launch_id=%lu\n", current_kernel_launch_id);
    return;
  }

  time_t now = time(nullptr);
  loprintf("==> WARP STATUS SUMMARY for kernel_launch_id=%lu (%zu active warps):\n", current_kernel_launch_id,
           ctx_state->active_warps.size());
  loprintf("    Format: WarpID[CTA_x,y,z] - LoopStatus - Activity\n");
  loprintf("    -----------------------------------------------------------------------\n");

  // Resolve SASS map for the current function, if available
  const std::map<int, std::string> *sass_map_for_func = nullptr;
  {
    std::map<uint64_t, std::pair<CUcontext, CUfunction>>::iterator func_iter =
        kernel_launch_to_func_map.find(current_kernel_launch_id);
    if (func_iter != kernel_launch_to_func_map.end()) {
      CUfunction f_func = func_iter->second.second;
      if (ctx_state->id_to_sass_map.count(f_func)) {
        sass_map_for_func = &ctx_state->id_to_sass_map[f_func];
      }
    }
  }

  for (const auto &warp_key : ctx_state->active_warps) {
    // Basic warp info
    loprintf("    Warp%d[%d,%d,%d]: ", warp_key.warp_id, warp_key.cta_id_x, warp_key.cta_id_y, warp_key.cta_id_z);

    // Loop state info
    auto loop_iter = ctx_state->loop_states.find(warp_key);
    bool is_looping = false;
    if (loop_iter != ctx_state->loop_states.end()) {
      const WarpLoopState &loop_state = loop_iter->second;
      if (loop_state.loop_flag) {
        is_looping = true;
        time_t last_seen_secs = 0;
        auto seen_it2 = ctx_state->last_seen_time_by_warp.find(warp_key);
        if (seen_it2 != ctx_state->last_seen_time_by_warp.end()) {
          last_seen_secs = now - seen_it2->second;
        }
        loprintf("LOOPING(period=%d, repeat=%d) last_seen=%lds ", loop_state.last_period, loop_state.repeat_cnt,
                 last_seen_secs);
      }
    }

    // Activity info
    time_t inactive_duration = 0;
    auto seen_iter = ctx_state->last_seen_time_by_warp.find(warp_key);
    if (seen_iter != ctx_state->last_seen_time_by_warp.end()) {
      inactive_duration = now - seen_iter->second;
    }

    if (!is_looping) {
      // If not looping, distinguish barrier vs progressing by last_is_defer_blocking_by_warp
      bool is_barrier = false;
      auto itBar = ctx_state->last_is_defer_blocking_by_warp.find(warp_key);
      if (itBar != ctx_state->last_is_defer_blocking_by_warp.end()) {
        is_barrier = itBar->second;
      }

      // Use unified single-line printer for last instruction; no temporary variables needed here

      if (is_barrier) {
        // Barrier category (last observed instruction is BAR.SYNC.DEFER_BLOCKING)
        // Include inactivity seconds for quick assessment
        int period_val = 0;
        int repeat_val = 0;
        if (loop_iter != ctx_state->loop_states.end()) {
          period_val = loop_iter->second.last_period;
          repeat_val = loop_iter->second.repeat_cnt;
        }
        loprintf("BARRIER(inactive=%lds) no_loop(period=%d, repeat=%d)\n", inactive_duration, period_val, repeat_val);
        print_last_instruction_line(loop_iter != ctx_state->loop_states.end() ? &loop_iter->second : nullptr,
                                    sass_map_for_func);
      } else {
        // Progressing category
        int period_val = 0;
        int repeat_val = 0;
        if (loop_iter != ctx_state->loop_states.end()) {
          period_val = loop_iter->second.last_period;
          repeat_val = loop_iter->second.repeat_cnt;
        }
        loprintf("PROGRESSING no_loop(period=%d, repeat=%d)\n", period_val, repeat_val);
        print_last_instruction_line(loop_iter != ctx_state->loop_states.end() ? &loop_iter->second : nullptr,
                                    sass_map_for_func);
      }
    }

    // Exit candidate status
    auto exit_iter = ctx_state->exit_candidate_since_by_warp.find(warp_key);
    if (exit_iter != ctx_state->exit_candidate_since_by_warp.end()) {
      time_t exit_duration = now - exit_iter->second;
      loprintf("- EXIT_CANDIDATE(%lds)", exit_duration);
    }

    loprintf("\n");

    // Print loop body details if warp is in a confirmed loop
    if (loop_iter != ctx_state->loop_states.end() && loop_iter->second.loop_flag) {
      const WarpLoopState &loop_state = loop_iter->second;
      if (!loop_state.current_loop.instructions.empty()) {
        loprintf("      Loop Body (%d instructions):\n", loop_state.current_loop.period);
        // Pre-compute alignment for PC/Offset columns across the loop body
        int pc_dec_width = 1;
        int hex_nibbles_max = 4;
        {
          size_t upper = std::min(static_cast<size_t>(loop_state.current_loop.period),
                                  loop_state.current_loop.instructions.size());
          for (size_t j = 0; j < upper; ++j) {
            uint64_t pc = loop_state.current_loop.instructions[j].reg.pc;
            int dec_w = 1;
            uint64_t td = pc;
            while (td >= 10) {
              td /= 10;
              dec_w++;
            }
            if (dec_w > pc_dec_width) pc_dec_width = dec_w;
            int nibbles = 1;
            if (pc != 0) {
              nibbles = 0;
              uint64_t th = pc;
              while (th) {
                th >>= 4;
                nibbles++;
              }
            }
            if (nibbles > hex_nibbles_max) hex_nibbles_max = nibbles;
          }
          hex_nibbles_max = ((hex_nibbles_max + 3) / 4) * 4;
          if (hex_nibbles_max < 4) hex_nibbles_max = 4;
          if (hex_nibbles_max > 16) hex_nibbles_max = 16;
        }
        for (size_t i = 0;
             i < static_cast<size_t>(loop_state.current_loop.period) && i < loop_state.current_loop.instructions.size();
             ++i) {
          const auto &instr = loop_state.current_loop.instructions[i];
          print_instruction_line(i, instr, sass_map_for_func, pc_dec_width, hex_nibbles_max);
        }
      }
    }
  }
  loprintf("    -----------------------------------------------------------------------\n");
}

// Checks for potential kernel hangs by determining if all active warps are stuck in loops.
// The check is throttled to run at most once every HANG_CHECK_THROTTLE_SECS.
// If all warps are looping and the condition persists for several checks, it terminates the process.
// Before checking, it prunes warps that are candidates for exiting and have been inactive.
static void check_kernel_hang(CTXstate *ctx_state, uint64_t current_kernel_launch_id) {
  time_t now = time(nullptr);
  if (now - ctx_state->last_hang_check_time < HANG_CHECK_THROTTLE_SECS) {  // Throttle to configured interval
    return;
  }
  ctx_state->last_hang_check_time = now;

  if (ctx_state->active_warps.empty()) {
    return;
  }

  // Cleanup exit-candidate warps before evaluating loop state
  std::vector<WarpKey> to_remove;
  to_remove.reserve(ctx_state->active_warps.size());
  for (const WarpKey &key : ctx_state->active_warps) {
    auto itExit = ctx_state->exit_candidate_since_by_warp.find(key);
    if (itExit == ctx_state->exit_candidate_since_by_warp.end()) continue;
    time_t exit_since = itExit->second;
    time_t last_seen = 0;
    auto itSeen = ctx_state->last_seen_time_by_warp.find(key);
    if (itSeen != ctx_state->last_seen_time_by_warp.end()) last_seen = itSeen->second;
    // Remove only if no activity since EXIT and at least one throttle interval has passed
    if (last_seen <= exit_since && (now - exit_since) >= HANG_CHECK_THROTTLE_SECS) {
      to_remove.push_back(key);
    }
  }
  if (!to_remove.empty()) {
    for (const WarpKey &key : to_remove) {
      ctx_state->active_warps.erase(key);
      ctx_state->loop_states.erase(key);
      ctx_state->pending_mem_by_warp.erase(key);
      ctx_state->last_seen_time_by_warp.erase(key);
      ctx_state->exit_candidate_since_by_warp.erase(key);
    }
  }

  size_t looping_cnt = 0;
  size_t barrier_cnt = 0;
  size_t progressing_cnt = 0;
  bool candidate_hang = false;
  for (const WarpKey &warp_key : ctx_state->active_warps) {
    bool is_looping = false;
    {
      std::map<WarpKey, WarpLoopState>::const_iterator it = ctx_state->loop_states.find(warp_key);
      if (it != ctx_state->loop_states.end() && it->second.loop_flag) is_looping = true;
    }

    bool is_barrier = false;
    {
      std::unordered_map<WarpKey, bool, WarpKey::Hash>::const_iterator itBar =
          ctx_state->last_is_defer_blocking_by_warp.find(warp_key);
      if (itBar != ctx_state->last_is_defer_blocking_by_warp.end()) is_barrier = itBar->second;
    }

    if (is_barrier)
      barrier_cnt++;
    else if (is_looping)
      looping_cnt++;
    else
      progressing_cnt++;
  }

  // Hang trigger: no warp is progressing → kernel is effectively stalled.
  // We partition active warps into three mutually exclusive categories:
  //  - barrier: the last observed instruction is BAR.SYNC.DEFER_BLOCKING
  //  - looping: not barrier and loop_flag == true (stable PC cycle detected)
  //  - progressing: not barrier and loop_flag == false (still making forward progress)
  // If progressing_cnt == 0 (and there are active warps), the system is stalled.
  // This single condition covers the three intended hang scenarios:
  //  (1) All barrier: barrier_cnt == active_warps.size(), looping_cnt == 0, progressing_cnt == 0
  //  (2) Barrier + the rest looping: barrier_cnt > 0, looping_cnt > 0, progressing_cnt == 0
  //  (3) All looping: looping_cnt == active_warps.size(), barrier_cnt == 0, progressing_cnt == 0
  if (!ctx_state->active_warps.empty() && progressing_cnt == 0) {
    candidate_hang = true;
  }

  if (candidate_hang) {
    time_t hang_time = 0;
    if (!ctx_state->loop_states.empty()) {
      hang_time = now - ctx_state->loop_states.begin()->second.first_loop_time;
    }
    loprintf(
        "Possible kernel hang: launch_id=%lu — state(looping=%zu, barrier=%zu, progressing=%zu) for %ld seconds.\n",
        current_kernel_launch_id, looping_cnt, barrier_cnt, progressing_cnt, hang_time);
    print_warp_status_summary(ctx_state, current_kernel_launch_id);
    if (!ctx_state->deadlock_termination_initiated) {
      ctx_state->deadlock_consecutive_hits++;
      if (ctx_state->deadlock_consecutive_hits >= 3) {
        ctx_state->deadlock_termination_initiated = true;
        loprintf("Deadlock sustained for %d checks; sending SIGTERM.\n", ctx_state->deadlock_consecutive_hits);
        fflush(stdout);
        fflush(stderr);
        raise(SIGTERM);
        sleep(2);
        loprintf("Process still alive after SIGTERM; sending SIGKILL.\n");
        raise(SIGKILL);
      }
    }
  } else {
    ctx_state->deadlock_consecutive_hits = 0;
  }
}

/**
 * @brief The main thread function for receiving and processing data from the
 * GPU.
 *
 * This function is based on the `recv_thread_fun` from NVIDIA's `mem_trace`
 * example. It runs in a separate CPU thread for each CUDA context, continuously
 * receiving data packets from the GPU channel and processing them.
 *
 * Meta's enhancements transform this from a simple single-purpose function to a
 * versatile multi-analysis pipeline:
 *  - **Generic Message-Passing System**: The original function only handled one
 *    data type (`mem_access_t`). This version uses a `message_header_t` to
 *    identify different packet types (`reg_info_t`, `opcode_only_t`, etc.) and
 *    dispatch them to the appropriate analysis logic.
 *  - **Instruction Histogram Analysis**: It contains the complete host-side logic
 *    for the `PROTON_INSTR_HISTOGRAM` feature, including state management for
 *    each warp (`warp_states`) and tracking completed regions.
 *  - **Kernel Boundary Detection**: It introduces robust state management across
 *    kernel launches by tracking `kernel_launch_id`. This allows it to detect
 *    when a kernel has finished, ensuring that all pending data for that kernel
 *    is finalized and dumped before processing the next one.
 *  - **SASS String Enrichment**: For richer logging, it looks up the SASS string
 *    for a given `opcode_id` to provide more context in the trace output.
 *
 * @param args A void pointer to the `CUcontext` for which this thread is
 * launched.
 * @return void* Always returns NULL.
 */
void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;

  pthread_mutex_lock(&mutex);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  ChannelHost *ch_host = &ctx_state->channel_host;
  pthread_mutex_unlock(&mutex);
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  // Per-thread, per-context state for histogram analysis.
  std::unordered_map<int, WarpState> warp_states;
  std::vector<RegionHistogram> local_completed_histograms;

  // Used to detect when a new kernel begins.
  uint64_t last_seen_kernel_launch_id = UINT64_MAX;  // Initial invalid value

  while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);

    if (num_recv_bytes > 0) {
      // Process data packets in this chunk

      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        // First read the message header to determine the message type
        message_header_t *header = (message_header_t *)&recv_buffer[num_processed_bytes];
        const char *sass_str = "N/A";

        uint64_t current_launch_id = get_kernel_launch_id(header);
        bool is_new_kernel = false;
        if (current_launch_id != 0 && current_launch_id != last_seen_kernel_launch_id) {
          is_new_kernel = true;
          if (last_seen_kernel_launch_id != UINT64_MAX) {
            // Cleanup for the previous kernel
            if (is_analysis_type_enabled(AnalysisType::PROTON_INSTR_HISTOGRAM)) {
              // Dump any remaining histograms for warps that were collecting
              for (auto &pair : warp_states) {
                if (pair.second.is_collecting && !pair.second.histogram.empty()) {
                  local_completed_histograms.push_back({pair.first, pair.second.region_counter, pair.second.histogram});
                }
              }
              dump_previous_kernel_data(last_seen_kernel_launch_id, local_completed_histograms);
              local_completed_histograms.clear();
              warp_states.clear();
            }
            if (is_analysis_type_enabled(AnalysisType::DEADLOCK_DETECTION)) {
              clear_deadlock_state(ctx_state);
            }
          }
          last_seen_kernel_launch_id = current_launch_id;
        }

        if (header->type == MSG_TYPE_REG_INFO) {
          reg_info_t *ri = (reg_info_t *)&recv_buffer[num_processed_bytes];

          if (is_analysis_type_enabled(AnalysisType::DEADLOCK_DETECTION)) {
            WarpKey key = {ri->cta_id_x, ri->cta_id_y, ri->cta_id_z, ri->warp_id};
            if (is_new_kernel) {
              ctx_state->last_hang_check_time = time(nullptr);
            }
            ctx_state->active_warps.insert(key);
            // Update last seen time for this warp
            ctx_state->last_seen_time_by_warp[key] = time(nullptr);
            update_loop_state(ctx_state, key, ri);

            // Determine if current instruction is BAR.SYNC.DEFER_BLOCKING
            bool is_barrier_defer = false;
            CUfunction f_func2 = nullptr;
            auto func_iter2 = kernel_launch_to_func_map.find(ri->kernel_launch_id);
            if (func_iter2 != kernel_launch_to_func_map.end()) {
              f_func2 = func_iter2->second.second;
              is_barrier_defer = is_barrier_defer_blocking_for_opcode(ctx_state, f_func2, ri->opcode_id);
            }
            ctx_state->last_is_defer_blocking_by_warp[key] = is_barrier_defer;

            // Mark EXIT candidate if this opcode_id is an EXIT for the current function
            if (func_iter2 != kernel_launch_to_func_map.end()) {
              if (ctx_state->exit_opcode_ids.count(f_func2) &&
                  ctx_state->exit_opcode_ids[f_func2].count(ri->opcode_id)) {
                if (!ctx_state->exit_candidate_since_by_warp.count(key)) {
                  ctx_state->exit_candidate_since_by_warp[key] = time(nullptr);
                }
              }
            }
          }
          // Get SASS string for trace output
          auto func_iter = kernel_launch_to_func_map.find(ri->kernel_launch_id);
          if (func_iter != kernel_launch_to_func_map.end()) {
            auto [f_ctx, f_func] = func_iter->second;
            if (ctx_state->id_to_sass_map.count(f_func) && ctx_state->id_to_sass_map[f_func].count(ri->opcode_id)) {
              sass_str = ctx_state->id_to_sass_map[f_func][ri->opcode_id].c_str();
            }
          }

          trace_lprintf("CTX %p - CTA %d,%d,%d - warp %d - %s:\n", ctx, ri->cta_id_x, ri->cta_id_y, ri->cta_id_z,
                        ri->warp_id, sass_str);

          // Print register values
          for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
            trace_lprintf("  * ");
            for (int i = 0; i < 32; i++) {
              trace_lprintf("Reg%d_T%02d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
            }
            trace_lprintf("\n");
          }
          trace_lprintf("\n");
          num_processed_bytes += sizeof(reg_info_t);

        } else if (header->type == MSG_TYPE_OPCODE_ONLY) {
          if (is_analysis_type_enabled(AnalysisType::PROTON_INSTR_HISTOGRAM)) {
            opcode_only_t *oi = (opcode_only_t *)&recv_buffer[num_processed_bytes];

            process_instruction_histogram(oi, ctx_state, warp_states, local_completed_histograms);
          }
          num_processed_bytes += sizeof(opcode_only_t);

        } else if (header->type == MSG_TYPE_MEM_ACCESS) {
          mem_access_t *mem = (mem_access_t *)&recv_buffer[num_processed_bytes];

          // Get SASS string for trace output.
          std::map<uint64_t, std::pair<CUcontext, CUfunction>>::iterator func_iter =
              kernel_launch_to_func_map.find(mem->kernel_launch_id);
          if (func_iter != kernel_launch_to_func_map.end()) {
            std::pair<CUcontext, CUfunction> kernel_info = func_iter->second;
            CUfunction f_func = kernel_info.second;
            if (ctx_state->id_to_sass_map.count(f_func) && ctx_state->id_to_sass_map[f_func].count(mem->opcode_id)) {
              sass_str = ctx_state->id_to_sass_map[f_func][mem->opcode_id].c_str();
            }
          }

          trace_lprintf(
              "CTX %p - kernel_launch_id %ld - CTA %d,%d,%d - warp %d - PC %ld - "
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
          // TODO: handle error message in our current log mechanism
          fprintf(stderr,
                  "ERROR: Unknown message type %d received in recv_thread_fun. "
                  "Stopping processing of this chunk.\n",
                  header->type);
          continue;
        }
      }
    }

    if (is_analysis_type_enabled(AnalysisType::DEADLOCK_DETECTION) && last_seen_kernel_launch_id != UINT64_MAX) {
      check_kernel_hang(ctx_state, last_seen_kernel_launch_id);
    }
  }

  // Dump data for the very last kernel if it exists.
  if (last_seen_kernel_launch_id != UINT64_MAX) {
    // Dump any remaining histograms for warps that were still collecting.
    for (const std::pair<const int, WarpState> &pair : warp_states) {
      if (pair.second.is_collecting && !pair.second.histogram.empty()) {
        local_completed_histograms.push_back({pair.first, pair.second.region_counter, pair.second.histogram});
      }
    }
    if (!local_completed_histograms.empty()) {
      dump_previous_kernel_data(last_seen_kernel_launch_id, local_completed_histograms);
    }
  }

  free(recv_buffer);
  ctx_state->recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}
