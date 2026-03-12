/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#pragma once

#include <stdint.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration to avoid circular dependency
enum class InstrumentType;
enum class InstrCategory;

/**
 * @brief Instrumentation point type for NVBit
 *
 * Controls whether instrumentation is inserted before or after the instruction.
 */
enum class IPointType {
  DEFAULT = -1,  // Not set - use the default_ipoint passed to get_ipoint_from_config
  BEFORE = 0,    // IPOINT_BEFORE - instrument before instruction execution
  AFTER = 1,     // IPOINT_AFTER - instrument after instruction execution
};

/**
 * @brief Defines the type of analysis to be performed on the collected trace
 * data.
 */
enum class AnalysisType {
  /**
   * @brief No analysis is performed.
   */
  ANALYSIS_NONE = 0,
  /**
   * @brief Enables instruction histogram analysis.
   *
   * This corresponds to the `proton_instr_histogram` setting and requires
   * `OPCODE_ONLY` instrumentation.
   */
  PROTON_INSTR_HISTOGRAM = 1,

  /**
   * @brief Enables deadlock detection analysis.
   */
  DEADLOCK_DETECTION = 2,

  /**
   * @brief Enables random delay injection for data race detection.
   */
  RANDOM_DELAY = 3,
};

// Configuration variables
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;
extern int verbose;
extern bool dump_cubin;

// Kernel name filters
extern std::vector<std::string> kernel_filters;

// Instrumentation configuration
// Uses instrument_type_to_index map for O(1) lookup (replaces the old unordered_set)
// Ordered list of enabled instrument types (preserves insertion order for IPOINT mapping)
extern std::vector<InstrumentType> enabled_instrument_types_ordered;
// Map from InstrumentType to its index in enabled_instrument_types_ordered (O(1) lookup)
extern std::unordered_map<InstrumentType, int> instrument_type_to_index;

// Analysis configuration
extern std::unordered_set<AnalysisType> enabled_analysis_types;

// Initialize configuration from environment variables
void init_config_from_env();

// Check if a specific instrumentation type is enabled
bool is_instrument_type_enabled(InstrumentType type);

// Check if any instrumentation type is enabled
bool has_any_instrumentation_enabled();

// Check if a specific analysis type is enabled
bool is_analysis_type_enabled(AnalysisType type);

// Initialize instrumentation configuration
void init_instrumentation(const std::string& instrument_str);

// Initialize analysis configuration
void init_analysis(const std::string& analysis_str);

// Trace format configuration
// 0 = text format (default)
// 1 = NDJSON+Zstd (compressed JSON)
// 2 = NDJSON only (uncompressed JSON, good for debugging)
extern int trace_format_ndjson;

// Zstd compression level (1-22, higher = better compression but slower)
// Default: 9 (good compression with reasonable speed)
extern int zstd_compression_level;

// Delay value in nanoseconds for random delay instrumentation
extern uint32_t g_delay_ns;

// Delay dump output path (optional)
// If set, instrumentation points will be written to this JSON file for later replay
extern std::string delay_dump_path;

// Delay load path (optional)
// If set, instrumentation points will be read from this JSON file for replay mode
extern std::string delay_load_path;

// Output directory for all CUTracer files (trace files and log files)
// When set, all output is written to this directory instead of the current directory
// Set via CUTRACER_OUTPUT_DIR environment variable
extern std::string output_dir;

// CPU call stack capture at kernel launch (default: enabled)
// Set CUTRACER_CPU_CALLSTACK=0 to disable
extern bool cpu_callstack_enabled;

// GPU channel buffer size for GPU→CPU communication (in bytes)
// Computed from CUTRACER_CHANNEL_RECORDS (number of records the buffer can hold)
// or defaults to 4MB if not set. Smaller values force more frequent flushes,
// which is useful for hang debugging (ensures trace data reaches CPU promptly).
// Set via CUTRACER_CHANNEL_RECORDS environment variable.
extern int channel_buffer_size;

// Instruction category filtering for conditional instrumentation
// If empty, all instructions are instrumented
// If set, only instructions in the specified categories are instrumented
extern std::unordered_set<InstrCategory> enabled_instr_categories;

// Kernel execution time limit in seconds (0 = disabled)
// When set, any kernel running longer than this value is automatically
// terminated with SIGTERM. Acts as a general safety valve independent of
// deadlock detection (does not require -a deadlock_detection).
// Set via CUTRACER_KERNEL_TIMEOUT_S environment variable
extern uint32_t kernel_timeout_s;

// Trace file size limit in MB (0 = disabled)
// When the trace file exceeds this limit, the process is automatically terminated.
// Set via CUTRACER_TRACE_SIZE_LIMIT_MB environment variable
extern uint32_t trace_size_limit_mb;

// Initialize instruction category filtering from environment variable
void init_instr_categories(const std::string& categories_str);

// Check if a specific instruction category should be instrumented
bool should_instrument_category(InstrCategory category);

// Check if category-based filtering is enabled
bool has_category_filter_enabled();

// IPOINT configuration for instrumentation
//
// Two environment variables control IPOINT behavior:
//
// 1. CUTRACER_INSTRUMENT_IPOINT_UNIFORM=a|b
//    Applies the same IPOINT to ALL enabled instruments.
//    Example: CUTRACER_INSTRUMENT_IPOINT_UNIFORM=b  (all use IPOINT_BEFORE)
//
// 2. CUTRACER_INSTRUMENT_IPOINT=a,b,a,...
//    Per-instrument IPOINT list. Count MUST match the final enabled instruments
//    (including analysis-added ones). Order follows enabled_instrument_types_ordered.
//    Example: CUTRACER_INSTRUMENT=reg_trace,mem_value_trace
//             CUTRACER_INSTRUMENT_IPOINT=b,a  (reg_trace=BEFORE, mem_value_trace=AFTER)
//
// Error if both are set simultaneously.
// Warning if mem_value_trace uses IPOINT_BEFORE (may not capture loaded values).
//
// Values: 'a'/'after' = IPOINT_AFTER, 'b'/'before' = IPOINT_BEFORE

// Uniform IPOINT for all instrumentation (set via CUTRACER_INSTRUMENT_IPOINT_UNIFORM)
extern IPointType uniform_ipoint;

// Per-instrument IPOINT overrides (set via CUTRACER_INSTRUMENT_IPOINT)
// Indexed by position in enabled_instrument_types_ordered
extern std::vector<IPointType> ipoint_overrides;

// Initialize IPOINT configuration from environment variables
void init_instrument_ipoint();
