/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#pragma once

#include <stdint.h>

#include <string>
#include <unordered_set>
#include <vector>
// Forward declaration to avoid circular dependency
enum class InstrumentType;

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
};

// Configuration variables
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;
extern int verbose;
extern bool dump_cubin;

// Kernel name filters
extern std::vector<std::string> kernel_filters;
// Instrumentation configuration
extern std::unordered_set<InstrumentType> enabled_instrument_types;

// Analysis configuration
extern std::unordered_set<AnalysisType> enabled_analysis_types;

// Initialize configuration from environment variables
void init_config_from_env();

// Check if a specific instrumentation type is enabled
bool is_instrument_type_enabled(InstrumentType type);

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
