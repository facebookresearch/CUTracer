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
bool dump_cubin;
// kernel name filters
std::vector<std::string> kernel_filters;
// enabled instrumentation types
std::unordered_set<InstrumentType> enabled_instrument_types;
// enabled analysis types
std::unordered_set<AnalysisType> enabled_analysis_types;

// Trace format configuration variable
int trace_format_ndjson;

// Zstd compression level
int zstd_compression_level;

// Delay value in nanoseconds for synchronization instrumentation
uint32_t g_delay_ns;

// Delay config dump output path (optional)
std::string delay_dump_path;

// Delay config load path (optional)
std::string delay_load_path;

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
static void parse_kernel_filters(const std::string& filters_env) {
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
  for (const auto& filter : kernel_filters) {
    printf("  - %s\n", filter.c_str());
  }
}

// Helper function for reading environment variables
static void get_var_int(int& var, const char* env_name, int default_val, const char* description) {
  const char* env_val = getenv(env_name);
  if (env_val) {
    var = atoi(env_val);
  } else {
    var = default_val;
  }
  loprintf("%s = %d (%s)\n", env_name, var, description);
}

static void get_var_uint32(uint32_t& var, const char* env_name, uint32_t default_val, const char* description) {
  const char* env_val = getenv(env_name);
  if (env_val) {
    var = (uint32_t)atoll(env_val);
  } else {
    var = default_val;
  }
  loprintf("%s = %u (%s)\n", env_name, var, description);
}

static void get_var_uint64(uint64_t& var, const char* env_name, uint64_t default_val, const char* description) {
  const char* env_val = getenv(env_name);
  if (env_val) {
    var = (uint64_t)strtoull(env_val, nullptr, 10);
  } else {
    var = default_val;
  }
  loprintf("%s = %lu (%s)\n", env_name, var, description);
}

static void get_var_str(std::string& var, const char* env_name, const std::string& default_val,
                        const char* description) {
  const char* env_val = getenv(env_name);
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
void init_instrumentation(const std::string& instrument_str) {
  if (instrument_str.empty()) {
    return;
  }
  loprintf("Using instrumentation types: %s\n", instrument_str.c_str());

  if (instrument_str.find("reg_trace") != std::string::npos) {
    enabled_instrument_types.insert(InstrumentType::REG_TRACE);
    loprintf("  - Enabled: reg_trace (register value tracing)\n");
  }
  if (instrument_str.find("mem_addr_trace") != std::string::npos) {
    enabled_instrument_types.insert(InstrumentType::MEM_ADDR_TRACE);
    loprintf("  - Enabled: mem_addr_trace (memory access address tracing)\n");
  }
  if (instrument_str.find("mem_value_trace") != std::string::npos) {
    enabled_instrument_types.insert(InstrumentType::MEM_VALUE_TRACE);
    loprintf("  - Enabled: mem_value_trace (memory access with value tracing)\n");
  }
  if (instrument_str.find("random_delay") != std::string::npos) {
    enabled_instrument_types.insert(InstrumentType::RANDOM_DELAY);
    loprintf("  - Enabled: random_delay (random delay injection)\n");
  }

  // Warn if both mem_addr_trace and mem_value_trace are enabled
  if (enabled_instrument_types.count(InstrumentType::MEM_ADDR_TRACE) &&
      enabled_instrument_types.count(InstrumentType::MEM_VALUE_TRACE)) {
    loprintf("WARNING: Both 'mem_addr_trace' and 'mem_value_trace' are enabled.\n");
    loprintf("- mem_addr_trace: records addresses at IPOINT_BEFORE\n");
    loprintf("- mem_value_trace: records addresses+values at IPOINT_AFTER\n");
    loprintf("Note: mem_value_trace already includes address information.\n");
    loprintf("If you only need value tracing, consider using mem_value_trace alone.\n");
  }
}

void parse_delay_config() {
  uint64_t delay_val = 0;
  get_var_uint64(delay_val, "CUTRACER_DELAY_NS", 0, "Delay in nanoseconds for synchronization instructions");

  // If random_delay analysis is enabled but no valid delay value, error out.
  if (delay_val == 0 && is_analysis_type_enabled(AnalysisType::RANDOM_DELAY)) {
    fprintf(stderr,
            "FATAL: CUTRACER_ANALYSIS includes 'random_delay' but no delay value is set.\n"
            "Please set CUTRACER_DELAY_NS to a positive value (in nanoseconds).\n"
            "Example: export CUTRACER_DELAY_NS=1000000  (1ms delay)\n");
    exit(1);
  }

  // Validate range: nanosleep uses uint32_t, so delay must fit in 32 bits.
  if (delay_val > UINT32_MAX) {
    fprintf(stderr, "FATAL: Delay value %lu exceeds maximum value of %u.\n", delay_val, UINT32_MAX);
    exit(1);
  }

  g_delay_ns = (uint32_t)delay_val;

  // Get delay config dump output path
  get_var_str(delay_dump_path, "CUTRACER_DELAY_DUMP_PATH", "", "Output path to dump delay config JSON for replay");

  // Get delay load path (for replay mode)
  get_var_str(delay_load_path, "CUTRACER_DELAY_LOAD_PATH", "",
              "Load delay config JSON for replay mode (uses saved delay values instead of random)");

  // Validate that load and dump paths are not both set
  if (!delay_dump_path.empty() && !delay_load_path.empty()) {
    fprintf(stderr,
            "FATAL: Both CUTRACER_DELAY_DUMP_PATH and CUTRACER_DELAY_LOAD_PATH are set.\n"
            "Please use only one: DUMP for recording, LOAD for replay.\n");
    exit(1);
  }
}

void init_analysis(const std::string& analysis_str) {
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

  // deadlock_detection: enable analysis type and ensure REG_TRACE is on
  if (analysis_str.find("deadlock_detection") != std::string::npos) {
    enabled_analysis_types.insert(AnalysisType::DEADLOCK_DETECTION);
    loprintf("  - Enabled: deadlock_detection\n");
    if (!is_instrument_type_enabled(InstrumentType::REG_TRACE)) {
      enabled_instrument_types.insert(InstrumentType::REG_TRACE);
      loprintf("  - deadlock_detection: forcing reg_trace instrumentation\n");
    }
  }

  // random_delay: enable analysis type and ensure RANDOM_DELAY instrumentation is on
  // Note: CUTRACER_DELAY_NS is validated later in init_config_from_env()
  if (analysis_str.find("random_delay") != std::string::npos) {
    enabled_analysis_types.insert(AnalysisType::RANDOM_DELAY);
    loprintf("  - Enabled: random_delay\n");
    if (!is_instrument_type_enabled(InstrumentType::RANDOM_DELAY)) {
      enabled_instrument_types.insert(InstrumentType::RANDOM_DELAY);
      loprintf("  - random_delay: forcing random_delay instrumentation\n");
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

/**
 * @brief Check if any instrumentation type is enabled
 *
 * This function checks if CUTRACER_INSTRUMENT was set to a non-empty value,
 * meaning at least one instrumentation type (reg_trace, mem_addr_trace, etc.)
 * is enabled. This is used to decide whether to create TraceWriter instances -
 * if no instrumentation is enabled, there's no point creating trace files
 * since they will be empty.
 *
 * @return true if at least one instrumentation type is enabled
 */
bool has_any_instrumentation_enabled() {
  return !enabled_instrument_types.empty();
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
  int dump_cubin_int = 0;
  get_var_int(dump_cubin_int, "CUTRACER_DUMP_CUBIN", 0, "Dump cubin files for instrumented kernels");
  dump_cubin = (dump_cubin_int != 0);
  // If INSTRS is not set, fall back to the old INSTR_BEGIN/INSTR_END behavior
  get_var_uint32(instr_begin_interval, "INSTR_BEGIN", 0,
                 "Beginning of the instruction interval where to apply instrumentation");
  get_var_uint32(instr_end_interval, "INSTR_END", UINT32_MAX,
                 "End of the instruction interval where to apply instrumentation");
  std::string instrument_str;
  get_var_str(instrument_str, "CUTRACER_INSTRUMENT", "",
              "Instrumentation types to enable (opcode_only,reg_trace,mem_addr_trace)");
  std::string kernel_filters_env;
  get_var_str(kernel_filters_env, "KERNEL_FILTERS", "", "Kernel name filters");
  std::string analysis_str;
  get_var_str(analysis_str, "CUTRACER_ANALYSIS", "",
              "Analysis types to enable (proton_instr_histogram, deadlock_detection, random_delay)");

  //===== Initializations ==========
  // Get kernel name filters
  parse_kernel_filters(kernel_filters_env);

  // Clear enabled types at the beginning
  enabled_instrument_types.clear();

  // Initialize analysis first, as it may enable instrumentation types
  init_analysis(analysis_str);
  // Initialize instrumentation from user settings
  init_instrumentation(instrument_str);

  // Trace format configuration
  get_var_int(trace_format_ndjson, "TRACE_FORMAT_NDJSON", 1,
              "Trace format: 0=text, 1=NDJSON+Zstd, 2=NDJSON only, 3=CLP Archive");

  // Validate trace format range
  if (trace_format_ndjson < 0 || trace_format_ndjson > 3) {
    printf("WARNING: Invalid TRACE_FORMAT_NDJSON=%d. Using default=0 (text).\n", trace_format_ndjson);
    trace_format_ndjson = 0;
  }

  // Zstd compression level (only used when trace_format_ndjson == 1)
  get_var_int(zstd_compression_level, "CUTRACER_ZSTD_LEVEL", 22, "Zstd compression level (1-22, default 22)");

  // Validate compression level range
  if (zstd_compression_level < 1 || zstd_compression_level > 22) {
    printf("WARNING: Invalid CUTRACER_ZSTD_LEVEL=%d. Using default=22.\n", zstd_compression_level);
    zstd_compression_level = 22;
  }

  // Parse and validate delay configuration (includes config paths)
  parse_delay_config();

  std::string pad(100, '-');
  loprintf("%s\n", pad.c_str());
}
