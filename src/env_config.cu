/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#include <stdio.h>
#include <stdlib.h>

#include "env_config.h"
#include "instrument.h"
#include "log.h"

// Define configuration variables
// EVERY VARIABLE MUST BE INITIALIZED IN init_config_from_env()
uint32_t instr_begin_interval;
uint32_t instr_end_interval;
int verbose;
// kernel name filters
std::vector<std::string> kernel_filters;
// enabled instrumentation types
std::unordered_set<InstrumentType> enabled_instrument_types;
// enabled analysis types
std::unordered_set<AnalysisType> enabled_analysis_types;

/**
 * @brief Parses a comma-separated string of kernel name filters for substring matching.
 *
 * This function takes a string from an environment variable, splits it by commas,
 * and populates the global `kernel_filters` vector with the individual filters.
 * These filters are then used to determine which CUDA kernels to instrument by
 * checking if a filter string appears as a **substring** of the kernel's name
 * (either mangled or unmangled).
 * Empty tokens resulting from ",," or trailing/leading commas are ignored.
 *
 * @param filters_env A C-style string containing comma-separated kernel name filters.
 *                     If NULL, the function does nothing.
 *
 * @example
 * If the environment variable (e.g., `KERNEL_FILTERS`) is set as:
 * `export KERNEL_FILTERS="add,_Z2_gemm,reduce"`
 *
 * A kernel named "add_kernel" would be matched by the "add" filter. A kernel
 * named "my_reduce_kernel" would be matched by "reduce".
 *
 * After calling this function with the example string, the `kernel_filters`
 * vector will contain: `{"add", "_Z2_gemm", "reduce"}`
 */
static void parse_kernel_filters(const std::string &filters_env) {
  if (filters_env.empty()) return;

  std::string filters_str = filters_env;
  size_t pos = 0;
  std::string token;

  // Split by commas
  while ((pos = filters_str.find(',')) != std::string::npos) {
    token = filters_str.substr(0, pos);
    if (!token.empty()) {
      kernel_filters.push_back(token);
    }
    filters_str.erase(0, pos + 1);
  }

  // Add the last token (if it exists)
  if (!filters_str.empty()) {
    kernel_filters.push_back(filters_str);
  }

  printf("Kernel name filters to instrument:\n");
  for (const auto &filter : kernel_filters) {
    printf("  - %s\n", filter.c_str());
  }
}

// Helper function for reading environment variables
static void get_var_int(int &var, const char *env_name, int default_val, const char *description) {
  const char *env_val = getenv(env_name);
  if (env_val) {
    var = atoi(env_val);
  } else {
    var = default_val;
  }
  loprintf("%s = %d (%s)\n", env_name, var, description);
}

static void get_var_uint32(uint32_t &var, const char *env_name, uint32_t default_val, const char *description) {
  const char *env_val = getenv(env_name);
  if (env_val) {
    var = (uint32_t)atoll(env_val);
  } else {
    var = default_val;
  }
  loprintf("%s = %u (%s)\n", env_name, var, description);
}

static void get_var_str(std::string &var, const char *env_name, const std::string &default_val,
                        const char *description) {
  const char *env_val = getenv(env_name);
  if (env_val) {
    var = std::string(env_val);
  } else {
    var = default_val;
  }
  loprintf("%s = %s (%s)\n", env_name, var.c_str(), description);
}

/**
 * @brief Initialize instrumentation system based on environment variables
 *
 * Parses CUTRACER_INSTRUMENT environment variable and sets up enabled types.
 * This function is called within init_config_from_env().
 */
void init_instrumentation(const std::string &instrument_str) {
  if (instrument_str.empty()) {
    return;
  }
  loprintf("Using instrumentation types: %s\n", instrument_str.c_str());

  if (instrument_str.find("reg_trace") != std::string::npos) {
    enabled_instrument_types.insert(InstrumentType::REG_TRACE);
    loprintf("  - Enabled: reg_trace (register value tracing)\n");
  }
  if (instrument_str.find("mem_trace") != std::string::npos) {
    enabled_instrument_types.insert(InstrumentType::MEM_TRACE);
    loprintf("  - Enabled: mem_trace (memory access tracing)\n");
  }
}

void init_analysis(const std::string &analysis_str) {
  enabled_analysis_types.clear();

  if (analysis_str.empty()) {
    loprintf("No analysis types specified.\n");
    return;
  }
  loprintf("Using analysis types: %s\n", analysis_str.c_str());

  // Parse comma-separated values
  if (analysis_str.find("proton_instr_histogram") != std::string::npos) {
    enabled_analysis_types.insert(AnalysisType::PROTON_INSTR_HISTOGRAM);
    loprintf("  - Enabled: proton_instr_histogram\n");

    // If proton_instr_histogram is enabled, force opcode_only instrumentation
    if (!is_instrument_type_enabled(InstrumentType::OPCODE_ONLY)) {
      enabled_instrument_types.insert(InstrumentType::OPCODE_ONLY);
      loprintf(
          "`proton_instr_histogram` analysis is enabled, forcing `opcode_only` "
          "instrumentation.\n");
    }
  }

  if (analysis_str.find("deadlock_detection") != std::string::npos) {
    enabled_analysis_types.insert(AnalysisType::DEADLOCK_DETECTION);
    loprintf("  - Enabled: deadlock_detection\n");

    // If deadlock_detection is enabled, force reg_trace instrumentation
    if (!is_instrument_type_enabled(InstrumentType::REG_TRACE)) {
      enabled_instrument_types.insert(InstrumentType::REG_TRACE);
      loprintf(
          "`deadlock_detection` analysis is enabled, forcing `reg_trace` "
          "instrumentation.\n");
    }
  }
}

/**
 * @brief Check if a specific instrumentation type is enabled
 *
 * @param type The instrumentation type to check
 * @return true if the instrumentation type is enabled
 */
bool is_instrument_type_enabled(InstrumentType type) {
  return enabled_instrument_types.count(type);
}

bool is_analysis_type_enabled(AnalysisType type) {
  return enabled_analysis_types.count(type);
}

// Initialize all configuration variables
void init_config_from_env() {
  // Enable device memory allocation
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  // Initialize log handle
  init_log_handle();
  // Get other configuration variables
  get_var_int(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  // If INSTRS is not set, fall back to the old INSTR_BEGIN/INSTR_END behavior
  get_var_uint32(instr_begin_interval, "INSTR_BEGIN", 0,
                 "Beginning of the instruction interval where to apply instrumentation");
  get_var_uint32(instr_end_interval, "INSTR_END", UINT32_MAX,
                 "End of the instruction interval where to apply instrumentation");
  std::string instrument_str;
  get_var_str(instrument_str, "CUTRACER_INSTRUMENT", "",
              "Instrumentation types to enable (opcode_only,reg_trace,mem_trace)");
  std::string kernel_filters_env;
  get_var_str(kernel_filters_env, "KERNEL_FILTERS", "", "Kernel name filters");
  std::string analysis_str;
  get_var_str(analysis_str, "CUTRACER_ANALYSIS", "", "Analysis types to enable (proton_instr_histogram,deadlock_detection)");

  //===== Initializations ==========
  // Get kernel name filters
  parse_kernel_filters(kernel_filters_env);

  // Clear enabled types at the beginning
  enabled_instrument_types.clear();

  // Initialize analysis first, as it may enable instrumentation types
  init_analysis(analysis_str);
  // Initialize instrumentation from user settings
  init_instrumentation(instrument_str);

  std::string pad(100, '-');
  loprintf("%s\n", pad.c_str());
}
