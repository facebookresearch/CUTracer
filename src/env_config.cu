/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#include <stdio.h>
#include <stdlib.h>

#include <filesystem>

#include "env_config.h"
#include "instr_category.h"
#include "instrument.h"
#include "log.h"

namespace fs = std::filesystem;

// Define configuration variables
// EVERY VARIABLE MUST BE INITIALIZED IN init_config_from_env()
uint32_t instr_begin_interval;
uint32_t instr_end_interval;
int verbose;
bool dump_cubin;
// kernel name filters
std::vector<std::string> kernel_filters;

// Instrumentation configuration - single source of truth
// enabled_instrument_types_ordered: preserves insertion order for IPOINT mapping
// instrument_type_to_index: O(1) lookup for "is enabled" check (replaces the old set)
std::vector<InstrumentType> enabled_instrument_types_ordered;
std::unordered_map<InstrumentType, int> instrument_type_to_index;

// enabled analysis types
std::unordered_set<AnalysisType> enabled_analysis_types;
// enabled instruction categories for conditional instrumentation
std::unordered_set<InstrCategory> enabled_instr_categories;

// Uniform IPOINT for all instrumentation (CUTRACER_INSTRUMENT_IPOINT_UNIFORM)
IPointType uniform_ipoint = IPointType::DEFAULT;

// Per-instrument IPOINT overrides (CUTRACER_INSTRUMENT_IPOINT)
std::vector<IPointType> ipoint_overrides;

// Trace format configuration variable (CUTRACER_TRACE_FORMAT / legacy TRACE_FORMAT_NDJSON)
int trace_format;

// Zstd compression level
int zstd_compression_level;

// Delay value in nanoseconds for synchronization instrumentation (max)
uint32_t g_delay_ns;

// Minimum delay value in nanoseconds (floor for random mode)
uint32_t g_delay_min_ns;

// Delay mode: 0 = fixed (same delay for all threads), 1 = random (per-thread random)
int g_delay_mode;

// Delay config dump output path (optional)
std::string delay_dump_path;

// Delay config load path (optional)
std::string delay_load_path;

// Output directory for all CUTracer files (traces and logs)
std::string output_dir;

// CPU call stack capture at kernel launch (default: enabled)
bool cpu_callstack_enabled;

// GPU channel buffer size (in bytes), computed from CUTRACER_CHANNEL_RECORDS
// Default: 4MB (1 << 22)
int channel_buffer_size;

// Kernel execution time limit in seconds (0 = disabled)
uint32_t kernel_timeout_s;

// No-data hang detection timeout in seconds (default: 15)
uint32_t no_data_timeout_s;

// Trace file size limit in MB (0 = disabled)
uint32_t trace_size_limit_mb;

/**
 * @brief Compute the largest record size among all currently enabled instrument types.
 *
 * Must be called after init_analysis() and init_instrumentation(), since
 * analysis types can implicitly enable instrument types (e.g., deadlock_detection
 * forces REG_TRACE).
 *
 * @return The sizeof() of the largest enabled record type, or sizeof(reg_info_t) as fallback.
 */
static size_t compute_max_record_size() {
  size_t max_size = 0;

  if (is_instrument_type_enabled(InstrumentType::REG_TRACE)) {
    max_size = std::max(max_size, sizeof(reg_info_t));
  }
  if (is_instrument_type_enabled(InstrumentType::MEM_ADDR_TRACE)) {
    max_size = std::max(max_size, sizeof(mem_addr_access_t));
  }
  if (is_instrument_type_enabled(InstrumentType::MEM_VALUE_TRACE)) {
    max_size = std::max(max_size, sizeof(mem_value_access_t));
  }
  if (is_instrument_type_enabled(InstrumentType::OPCODE_ONLY)) {
    max_size = std::max(max_size, sizeof(opcode_only_t));
  }
  if (is_instrument_type_enabled(InstrumentType::TMA_TRACE)) {
    max_size = std::max(max_size, sizeof(tma_access_t));
  }

  if (max_size == 0) {
    max_size = sizeof(reg_info_t);
  }

  return max_size;
}

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
 * @brief Helper function to convert InstrumentType to its string name (forward declaration)
 */
static const char* instrument_type_to_name(InstrumentType type);

/**
 * @brief Helper to add an instrument type (avoids duplicates via map check)
 *
 * Only adds if not already present. Updates both ordered list and index map.
 * Returns true if added, false if already present (duplicate).
 */
static bool add_enabled_instrument_type(InstrumentType type) {
  // Check if already added (O(1) lookup)
  if (instrument_type_to_index.find(type) != instrument_type_to_index.end()) {
    return false;  // Already enabled, skip duplicate
  }
  int index = static_cast<int>(enabled_instrument_types_ordered.size());
  enabled_instrument_types_ordered.push_back(type);
  instrument_type_to_index[type] = index;
  return true;
}

/**
 * @brief Initialize instrumentation system based on environment variables
 *
 * Parses CUTRACER_INSTRUMENT environment variable and sets up enabled types.
 */
void init_instrumentation(const std::string& instrument_str) {
  if (instrument_str.empty()) {
    return;
  }
  loprintf("Using instrumentation types: %s\n", instrument_str.c_str());

  if (instrument_str.find("opcode_only") != std::string::npos) {
    add_enabled_instrument_type(InstrumentType::OPCODE_ONLY);
    loprintf("  - Enabled: opcode_only\n");
  }
  if (instrument_str.find("reg_trace") != std::string::npos) {
    add_enabled_instrument_type(InstrumentType::REG_TRACE);
    loprintf("  - Enabled: reg_trace (register value tracing)\n");
  }
  if (instrument_str.find("mem_addr_trace") != std::string::npos) {
    add_enabled_instrument_type(InstrumentType::MEM_ADDR_TRACE);
    loprintf("  - Enabled: mem_addr_trace (memory access address tracing)\n");
  }
  if (instrument_str.find("mem_value_trace") != std::string::npos) {
    add_enabled_instrument_type(InstrumentType::MEM_VALUE_TRACE);
    loprintf("  - Enabled: mem_value_trace (memory access with value tracing)\n");
  }
  if (instrument_str.find("random_delay") != std::string::npos) {
    add_enabled_instrument_type(InstrumentType::RANDOM_DELAY);
    loprintf("  - Enabled: random_delay (random delay injection)\n");
  }
  if (instrument_str.find("tma_trace") != std::string::npos) {
    add_enabled_instrument_type(InstrumentType::TMA_TRACE);
    loprintf("  - Enabled: tma_trace (TMA descriptor tracing)\n");
  }

  // Warn if both mem_addr_trace and mem_value_trace are enabled
  if (is_instrument_type_enabled(InstrumentType::MEM_ADDR_TRACE) &&
      is_instrument_type_enabled(InstrumentType::MEM_VALUE_TRACE)) {
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

  // Parse minimum delay (optional, default 0)
  uint64_t delay_min_val = 0;
  get_var_uint64(delay_min_val, "CUTRACER_DELAY_MIN_NS", 0,
                 "Minimum delay in nanoseconds (floor for random mode, default: 0)");
  if (delay_min_val > UINT32_MAX) {
    fprintf(stderr, "FATAL: Min delay value %lu exceeds maximum value of %u.\n", delay_min_val, UINT32_MAX);
    exit(1);
  }
  if (delay_min_val > delay_val) {
    fprintf(stderr,
            "FATAL: CUTRACER_DELAY_MIN_NS (%lu) > CUTRACER_DELAY_NS (%lu).\n"
            "Min delay must be <= max delay.\n",
            delay_min_val, delay_val);
    exit(1);
  }
  g_delay_min_ns = (uint32_t)delay_min_val;

  // Parse delay mode: "random" (1, default) or "fixed" (0)
  // random: each thread gets a random delay in [min_delay_ns, delay_ns] for asymmetric timing (recommended)
  // fixed: all threads get the same delay (preserves relative timing, often masks races)
  std::string delay_mode_str;
  get_var_str(delay_mode_str, "CUTRACER_DELAY_MODE", "random",
              "Delay mode: 'random' (per-thread random delay, default) or 'fixed' (same delay for all threads)");
  if (delay_mode_str == "random") {
    g_delay_mode = 1;
  } else if (delay_mode_str == "fixed") {
    g_delay_mode = 0;
  } else {
    fprintf(stderr,
            "FATAL: Invalid CUTRACER_DELAY_MODE '%s'.\n"
            "Valid values: 'random' (default) or 'fixed'.\n",
            delay_mode_str.c_str());
    exit(1);
  }

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
    if (add_enabled_instrument_type(InstrumentType::OPCODE_ONLY)) {
      loprintf("  NOTE: proton_instr_histogram requires opcode_only instrumentation (auto-added).\n");
    }
  }

  // deadlock_detection: enable analysis type and ensure REG_TRACE is on
  if (analysis_str.find("deadlock_detection") != std::string::npos) {
    enabled_analysis_types.insert(AnalysisType::DEADLOCK_DETECTION);
    loprintf("  - Enabled: deadlock_detection\n");
    if (add_enabled_instrument_type(InstrumentType::REG_TRACE)) {
      loprintf("  NOTE: deadlock_detection requires reg_trace instrumentation (auto-added).\n");
    }
  }

  // random_delay: enable analysis type and ensure RANDOM_DELAY instrumentation is on
  // Note: CUTRACER_DELAY_NS is validated later in init_config_from_env()
  if (analysis_str.find("random_delay") != std::string::npos) {
    enabled_analysis_types.insert(AnalysisType::RANDOM_DELAY);
    loprintf("  - Enabled: random_delay\n");
    if (add_enabled_instrument_type(InstrumentType::RANDOM_DELAY)) {
      loprintf("  NOTE: random_delay analysis requires random_delay instrumentation (auto-added).\n");
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
  return instrument_type_to_index.find(type) != instrument_type_to_index.end();
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
  return !enabled_instrument_types_ordered.empty();
}

bool is_analysis_type_enabled(AnalysisType type) {
  return enabled_analysis_types.count(type);
}

/**
 * @brief Initialize instruction category filtering from environment variable
 *
 * Parses CUTRACER_INSTR_CATEGORIES environment variable and sets up enabled categories.
 * If empty, no category filtering is applied (all instructions are instrumented).
 * If set, only instructions matching the specified categories are instrumented.
 *
 * @param categories_str Comma-separated category names (e.g., "mma,tma,sync")
 */
void init_instr_categories(const std::string& categories_str) {
  enabled_instr_categories.clear();

  if (categories_str.empty()) {
    // No category filtering - all instructions will be instrumented
    return;
  }

  loprintf("Using instruction category filters: %s\n", categories_str.c_str());

  // Parse comma-separated values (case-insensitive)
  std::string str = categories_str;

  // Convert to lowercase for case-insensitive matching
  for (char& c : str) {
    c = std::tolower(c);
  }

  if (str.find("mma") != std::string::npos) {
    enabled_instr_categories.insert(InstrCategory::MMA);
    loprintf("  - Enabled category: MMA (UTCMMA, HMMA, etc.)\n");
  }
  if (str.find("tma") != std::string::npos) {
    enabled_instr_categories.insert(InstrCategory::TMA);
    loprintf("  - Enabled category: TMA (UTMALDG, UTMASTG, etc.)\n");
  }
  if (str.find("sync") != std::string::npos) {
    enabled_instr_categories.insert(InstrCategory::SYNC);
    loprintf("  - Enabled category: SYNC (WARPGROUP.DEPBAR, etc.)\n");
  }

  if (enabled_instr_categories.empty()) {
    loprintf("WARNING: CUTRACER_INSTR_CATEGORIES set but no valid categories found.\n");
    loprintf("         Valid categories: mma, tma, sync\n");
  }
}

/**
 * @brief Check if a specific instruction category should be instrumented
 *
 * @param category The instruction category to check
 * @return true if the category should be instrumented (either no filter or category is enabled)
 */
bool should_instrument_category(InstrCategory category) {
  // If no category filter is set, instrument all categories
  if (enabled_instr_categories.empty()) {
    return true;
  }
  // Otherwise, only instrument if the category is explicitly enabled
  return enabled_instr_categories.count(category) > 0;
}

/**
 * @brief Check if category-based filtering is enabled
 *
 * @return true if CUTRACER_INSTR_CATEGORIES was set to a non-empty value
 */
bool has_category_filter_enabled() {
  return !enabled_instr_categories.empty();
}

/**
 * @brief Helper function to convert InstrumentType to its string name
 */
static const char* instrument_type_to_name(InstrumentType type) {
  switch (type) {
    case InstrumentType::OPCODE_ONLY:
      return "opcode_only";
    case InstrumentType::REG_TRACE:
      return "reg_trace";
    case InstrumentType::MEM_ADDR_TRACE:
      return "mem_addr_trace";
    case InstrumentType::MEM_VALUE_TRACE:
      return "mem_value_trace";
    case InstrumentType::RANDOM_DELAY:
      return "random_delay";
    case InstrumentType::TMA_TRACE:
      return "tma_trace";
    default:
      return "unknown";
  }
}

/**
 * @brief Parse IPOINT value from string
 *
 * @param str Input string (should be lowercase and trimmed)
 * @return IPointType parsed value, or DEFAULT if invalid
 */
static IPointType parse_ipoint_value(const std::string& str) {
  if (str == "b" || str == "before") {
    return IPointType::BEFORE;
  } else if (str == "a" || str == "after") {
    return IPointType::AFTER;
  }
  return IPointType::DEFAULT;
}

/**
 * @brief Trim whitespace and convert string to lowercase in-place
 */
static void trim_and_lowercase(std::string& s) {
  while (!s.empty() && std::isspace(s.front())) s.erase(0, 1);
  while (!s.empty() && std::isspace(s.back())) s.pop_back();
  for (char& c : s) c = std::tolower(c);
}

/**
 * @brief Parse comma-separated IPOINT list
 *
 * @param ipoint_str The IPOINT configuration string
 * @return Vector of IPointType values
 */
static std::vector<IPointType> parse_ipoint_list(const std::string& ipoint_str) {
  std::vector<IPointType> result;
  std::string str = ipoint_str;
  size_t pos = 0;

  while ((pos = str.find(',')) != std::string::npos) {
    std::string token = str.substr(0, pos);
    trim_and_lowercase(token);

    IPointType ipoint = parse_ipoint_value(token);
    if (ipoint == IPointType::DEFAULT) {
      loprintf("WARNING: Invalid IPOINT value '%s' at index %zu\n", token.c_str(), result.size());
    }
    result.push_back(ipoint);
    str.erase(0, pos + 1);
  }

  // Handle last token
  trim_and_lowercase(str);
  if (!str.empty()) {
    IPointType ipoint = parse_ipoint_value(str);
    if (ipoint == IPointType::DEFAULT) {
      loprintf("WARNING: Invalid IPOINT value '%s' at index %zu\n", str.c_str(), result.size());
    }
    result.push_back(ipoint);
  }

  return result;
}

/**
 * @brief Warn if mem_value_trace uses IPOINT_BEFORE
 */
static void warn_mem_value_trace_before(IPointType ipoint) {
  if (ipoint == IPointType::BEFORE && is_instrument_type_enabled(InstrumentType::MEM_VALUE_TRACE)) {
    loprintf(
        "WARNING: mem_value_trace with IPOINT_BEFORE may not capture loaded values correctly.\n"
        "         Load instructions need IPOINT_AFTER to capture values after memory read.\n");
  }
}

/**
 * @brief Handle CUTRACER_INSTRUMENT_IPOINT_UNIFORM env var
 *
 * Applies the same IPOINT to all enabled instruments.
 */
static void init_uniform_ipoint(const char* uniform_env) {
  std::string str = uniform_env;
  trim_and_lowercase(str);

  uniform_ipoint = parse_ipoint_value(str);
  if (uniform_ipoint == IPointType::DEFAULT) {
    fprintf(stderr,
            "FATAL: Invalid CUTRACER_INSTRUMENT_IPOINT_UNIFORM value '%s'\n"
            "Valid values: 'a'/'after' or 'b'/'before'\n",
            uniform_env);
    exit(1);
  }

  const char* name = (uniform_ipoint == IPointType::BEFORE) ? "BEFORE" : "AFTER";
  loprintf("CUTRACER_INSTRUMENT_IPOINT_UNIFORM: IPOINT_%s for all instruments\n", name);
  warn_mem_value_trace_before(uniform_ipoint);
}

/**
 * @brief Handle CUTRACER_INSTRUMENT_IPOINT env var (per-instrument list)
 *
 * Count must match the final enabled_instrument_types_ordered (including
 * both user-specified and analysis-added instruments).
 */
static void init_per_instrument_ipoint(const char* list_env) {
  ipoint_overrides = parse_ipoint_list(list_env);

  size_t enabled_count = enabled_instrument_types_ordered.size();
  size_t ipoint_count = ipoint_overrides.size();

  if (ipoint_count != enabled_count) {
    fprintf(stderr,
            "FATAL: CUTRACER_INSTRUMENT_IPOINT has %zu values, but %zu instruments are enabled.\n"
            "The IPOINT list count must match the number of enabled instruments.\n"
            "\nEnabled instruments (in order):\n",
            ipoint_count, enabled_count);
    for (size_t i = 0; i < enabled_count; ++i) {
      fprintf(stderr, "  %zu: %s\n", i + 1, instrument_type_to_name(enabled_instrument_types_ordered[i]));
    }
    fprintf(stderr,
            "\nExample: CUTRACER_INSTRUMENT=reg_trace,mem_value_trace\n"
            "         CUTRACER_INSTRUMENT_IPOINT=b,a\n");
    exit(1);
  }

  loprintf("CUTRACER_INSTRUMENT_IPOINT per-instrument configuration:\n");
  for (size_t i = 0; i < ipoint_count; ++i) {
    const char* ipoint_name = (ipoint_overrides[i] == IPointType::BEFORE)  ? "BEFORE"
                              : (ipoint_overrides[i] == IPointType::AFTER) ? "AFTER"
                                                                           : "DEFAULT";
    const char* instr_name = instrument_type_to_name(enabled_instrument_types_ordered[i]);
    loprintf("  - %s: IPOINT_%s\n", instr_name, ipoint_name);

    if (enabled_instrument_types_ordered[i] == InstrumentType::MEM_VALUE_TRACE) {
      warn_mem_value_trace_before(ipoint_overrides[i]);
    }
  }
}

/**
 * @brief Initialize IPOINT configuration from environment variables
 *
 * Reads CUTRACER_INSTRUMENT_IPOINT_UNIFORM and CUTRACER_INSTRUMENT_IPOINT,
 * validates mutual exclusivity, and delegates to the appropriate handler.
 */
void init_instrument_ipoint() {
  uniform_ipoint = IPointType::DEFAULT;
  ipoint_overrides.clear();

  const char* uniform_env = getenv("CUTRACER_INSTRUMENT_IPOINT_UNIFORM");
  const char* list_env = getenv("CUTRACER_INSTRUMENT_IPOINT");

  bool has_uniform = (uniform_env != nullptr && uniform_env[0] != '\0');
  bool has_list = (list_env != nullptr && list_env[0] != '\0');

  if (has_uniform && has_list) {
    fprintf(stderr,
            "FATAL: Both CUTRACER_INSTRUMENT_IPOINT_UNIFORM and CUTRACER_INSTRUMENT_IPOINT are set.\n"
            "Please use only one:\n"
            "  - CUTRACER_INSTRUMENT_IPOINT_UNIFORM=a|b for uniform IPOINT for all instruments\n"
            "  - CUTRACER_INSTRUMENT_IPOINT=a,b,... for per-instrument IPOINT list\n");
    exit(1);
  }

  if (has_uniform) {
    init_uniform_ipoint(uniform_env);
  } else if (has_list) {
    init_per_instrument_ipoint(list_env);
  }
}

/**
 * @brief Initialize output directory from environment variable
 *
 * Reads CUTRACER_OUTPUT_DIR and validates:
 * 1. The path exists
 * 2. It is a directory
 * 3. The directory has write permission
 * If not set, all output files are written to the current directory.
 *
 * This must run before init_log_handle() so the main log file is placed
 * in the configured directory. Fatal errors use fprintf(stderr) directly
 * since the logger is not yet available.
 */
void init_output_dir() {
  // Read directly with getenv() because this runs before init_log_handle(),
  // so the logging system is not yet available.
  const char* env_val = getenv("CUTRACER_OUTPUT_DIR");
  if (env_val) {
    output_dir = std::string(env_val);
  }

  if (!output_dir.empty()) {
    fs::path dir_path(output_dir);

    if (!fs::exists(dir_path)) {
      fprintf(stderr,
              "FATAL: CUTRACER_OUTPUT_DIR '%s' does not exist.\n"
              "Please create the directory first or specify a valid directory.\n",
              output_dir.c_str());
      exit(1);
    }
    if (!fs::is_directory(dir_path)) {
      fprintf(stderr,
              "FATAL: CUTRACER_OUTPUT_DIR '%s' is not a directory.\n"
              "Please specify a valid directory.\n",
              output_dir.c_str());
      exit(1);
    }
    auto perms = fs::status(dir_path).permissions();
    if ((perms & fs::perms::owner_write) == fs::perms::none) {
      fprintf(stderr,
              "FATAL: CUTRACER_OUTPUT_DIR '%s' is not writable.\n"
              "Please check directory permissions (chmod) or choose a different directory.\n",
              output_dir.c_str());
      exit(1);
    }
  }
}

// Initialize all configuration variables
void init_config_from_env() {
  // Enable device memory allocation
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  // Initialize output directory first (log file is placed there)
  init_output_dir();
  // Initialize log handle
  init_log_handle();
  // Log after init so it appears in both stdout and the log file
  loprintf("CUTRACER_OUTPUT_DIR = %s\n", output_dir.empty() ? "(not set)" : output_dir.c_str());
  // Get other configuration variables
  get_var_int(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
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

  // Clear all instrumentation data structures at the beginning
  enabled_instrument_types_ordered.clear();
  instrument_type_to_index.clear();

  // Initialize instrumentation from CUTRACER_INSTRUMENT
  init_instrumentation(instrument_str);

  // Initialize analysis - may add additional instruments (always use defaults for IPOINT)
  init_analysis(analysis_str);

  // Cubin dump configuration (after instrumentation/analysis init so we can auto-detect)
  // Auto-enable when instrumentation is active; user can override with CUTRACER_DUMP_CUBIN=0/1
  const char* dump_cubin_env = getenv("CUTRACER_DUMP_CUBIN");
  if (dump_cubin_env) {
    dump_cubin = (atoi(dump_cubin_env) != 0);
  } else {
    dump_cubin = has_any_instrumentation_enabled();
  }
  loprintf("CUTRACER_DUMP_CUBIN = %s%s\n", dump_cubin ? "enabled" : "disabled", dump_cubin_env ? "" : " (auto)");

  // Trace format configuration (CUTRACER_TRACE_FORMAT with TRACE_FORMAT_NDJSON as legacy fallback)
  // Try new name first, fall back to legacy name for backward compatibility
  const char* tf_env = getenv("CUTRACER_TRACE_FORMAT");
  if (!tf_env) {
    tf_env = getenv("TRACE_FORMAT_NDJSON");
    if (tf_env) {
      loprintf("NOTE: TRACE_FORMAT_NDJSON is deprecated. Please use CUTRACER_TRACE_FORMAT instead.\n");
    }
  }
  if (tf_env) {
    trace_format = atoi(tf_env);
  } else {
    trace_format = 2;
  }
  loprintf("CUTRACER_TRACE_FORMAT = %d (Trace format: 0=text, 1=NDJSON+Zstd, 2=NDJSON only, 3=CLP Archive)\n",
           trace_format);

  // Validate trace format range
  if (trace_format < 0 || trace_format > 3) {
    printf("WARNING: Invalid CUTRACER_TRACE_FORMAT=%d. Using default=2 (NDJSON only).\n", trace_format);
    trace_format = 2;
  }
  // Zstd compression level (only used when trace_format == 1)
  get_var_int(zstd_compression_level, "CUTRACER_ZSTD_LEVEL", 9, "Zstd compression level (1-22, default 9)");

  // Validate compression level range
  if (zstd_compression_level < 1 || zstd_compression_level > 22) {
    printf("WARNING: Invalid CUTRACER_ZSTD_LEVEL=%d. Using default=9.\n", zstd_compression_level);
    zstd_compression_level = 9;
  }

  // Parse and validate delay configuration (includes config paths)
  parse_delay_config();

  // Parse instruction category filters (optional)
  std::string instr_categories_str;
  get_var_str(instr_categories_str, "CUTRACER_INSTR_CATEGORIES", "",
              "Instruction categories to instrument (mma,tma,sync). Empty = all instructions");
  init_instr_categories(instr_categories_str);

  // CPU call stack capture (default: enabled)
  int cpu_callstack_int;
  get_var_int(cpu_callstack_int, "CUTRACER_CPU_CALLSTACK", 1, "CPU call stack capture (1=enabled, 0=disabled)");
  cpu_callstack_enabled = (cpu_callstack_int != 0);

  // Channel buffer size configuration
  // Users specify the number of records the buffer can hold.
  // The actual byte size is computed as: max_record_size * channel_records.
  // This must be after init_analysis() + init_instrumentation() since analysis
  // types can implicitly enable instrument types (e.g., deadlock_detection → REG_TRACE).
  int channel_records = 0;
  get_var_int(channel_records, "CUTRACER_CHANNEL_RECORDS", 0,
              "Channel buffer capacity in records (0=default 4MB). "
              "Set to 1 for per-record flush (useful for hang debugging)");

  if (channel_records > 0) {
    int max_record = (int)compute_max_record_size();
    channel_buffer_size = max_record * channel_records;
    loprintf("Channel buffer: %d records x %d bytes/record = %d bytes\n", channel_records, max_record,
             channel_buffer_size);
  } else {
    channel_buffer_size = (1 << 22);  // Default 4MB
    loprintf("Channel buffer: default %d bytes (4MB)\n", channel_buffer_size);
  }

  // Parse IPOINT configuration (optional) - after instrumentation is initialized
  init_instrument_ipoint();

  // Kernel execution time limit (seconds, 0 = disabled)
  get_var_uint32(kernel_timeout_s, "CUTRACER_KERNEL_TIMEOUT_S", 0,
                 "Kernel execution time limit in seconds (0 = disabled). Auto-terminate when exceeded");

  // No-data hang detection timeout (seconds, default: 15)
  // Independent of deadlock detection. Detects silent kernel hangs
  get_var_uint32(no_data_timeout_s, "CUTRACER_NO_DATA_TIMEOUT_S", 15,
                 "No-data hang timeout in seconds (default: 15). Detect silent kernel hangs. "
                 "Independent of deadlock detection (0 = disabled)");

  // Trace file size limit (MB, 0 = disabled)
  get_var_uint32(trace_size_limit_mb, "CUTRACER_TRACE_SIZE_LIMIT_MB", 0,
                 "Max trace file size in MB (0 = disabled). Auto-terminate when exceeded");
  std::string pad(100, '-');
  loprintf("%s\n", pad.c_str());
}
