/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#include <stdio.h>
#include <stdlib.h>


#include "env_config.h"
#include "log.h"

// Define configuration variables
// EVERY VARIABLE MUST BE INITIALIZED IN init_config_from_env()
uint32_t instr_begin_interval;
uint32_t instr_end_interval;
int verbose;
// kernel name filters
std::vector<std::string> kernel_patterns;
bool any_kernel_matched = false;

/**
 * @brief Parses a comma-separated string of kernel name patterns.
 *
 * This function takes a string from an environment variable, splits it by commas,
 * and populates the global `kernel_patterns` vector with the individual patterns.
 * These patterns are later used to filter which CUDA kernels should be instrumented.
 * Empty tokens resulting from ",," or trailing/leading commas are ignored.
 *
 * @param patterns_env A C-style string containing comma-separated kernel name patterns.
 *                     If NULL, the function does nothing.
 *
 * @example
 * If the environment variable (e.g., `CUTOMP_KERNEL_PATTERNS`) is set as:
 * `export CUTOMP_KERNEL_PATTERNS="add_kernel,_Z2_gemm,reduce"`
 *
 * After calling this function, the `kernel_patterns` vector will contain:
 * `{"add_kernel", "_Z2_gemm", "reduce"}`
 */
static void parse_kernel_patterns(const char *patterns_env) {
  if (!patterns_env) return;

  std::string patterns_str(patterns_env);
  size_t pos = 0;
  std::string token;

  // Split by commas
  while ((pos = patterns_str.find(',')) != std::string::npos) {
    token = patterns_str.substr(0, pos);
    if (!token.empty()) {
      kernel_patterns.push_back(token);
    }
    patterns_str.erase(0, pos + 1);
  }

  // Add the last token (if it exists)
  if (!patterns_str.empty()) {
    kernel_patterns.push_back(patterns_str);
  }

  if (verbose) {
    printf("Kernel name filters to instrument:\n");
    for (const auto &pattern : kernel_patterns) {
      printf("  - %s\n", pattern.c_str());
    }
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
  // Get kernel name filters
  const char *kernel_patterns_env = getenv("KERNEL_PATTERNS");
  if (kernel_patterns_env) {
    parse_kernel_patterns(kernel_patterns_env);
  }
  std::string pad(100, '-');
  loprintf("%s\n", pad.c_str());
}
