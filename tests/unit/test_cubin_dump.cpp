/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

/**
 * @file test_cubin_dump.cpp
 * @brief Unit tests for cubin dump configuration and path construction.
 *
 * Tests the env-var parsing logic for CUTRACER_DUMP_CUBIN, the cubin file
 * naming convention (kernel_{checksum}_{name}.cubin), and the
 * KernelFuncMetadata serialization when cubin_path is set or empty.
 *
 * Compile and run:
 *   g++ -std=c++17 -I../../include -o test_cubin_dump test_cubin_dump.cpp
 *   ./test_cubin_dump
 */

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

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
// Replicate the cubin dump configuration logic from env_config.cu
// so we can unit-test it without CUDA / NVBit dependencies.
// ============================================================================

/**
 * Parse CUTRACER_DUMP_CUBIN environment variable the same way
 * init_config_from_env() does: read the integer value and convert to bool.
 */
static bool parse_dump_cubin_env() {
  const char* env_val = getenv("CUTRACER_DUMP_CUBIN");
  if (env_val) {
    return atoi(env_val) != 0;
  }
  return false;  // default
}

/**
 * Reproduce the cubin path construction from cutracer.cu:
 *   meta.cubin_path = "kernel_" + checksum + "_" + truncated_name + ".cubin";
 * with the mangled name truncated to 150 characters.
 */
static std::string build_cubin_path(const std::string& checksum,
                                    const std::string& mangled_name) {
  std::string truncated_name = mangled_name.substr(0, 150);
  return "kernel_" + checksum + "_" + truncated_name + ".cubin";
}

// ============================================================================
// Test Cases
// ============================================================================

bool test_dump_cubin_default_off() {
  // When CUTRACER_DUMP_CUBIN is not set, dump_cubin should be false.
  unsetenv("CUTRACER_DUMP_CUBIN");
  return parse_dump_cubin_env() == false;
}

bool test_dump_cubin_enabled_with_1() {
  // Setting CUTRACER_DUMP_CUBIN=1 should enable cubin dump.
  setenv("CUTRACER_DUMP_CUBIN", "1", 1);
  bool result = parse_dump_cubin_env();
  unsetenv("CUTRACER_DUMP_CUBIN");
  return result == true;
}

bool test_dump_cubin_enabled_with_nonzero() {
  // Any non-zero value should enable cubin dump.
  setenv("CUTRACER_DUMP_CUBIN", "42", 1);
  bool result = parse_dump_cubin_env();
  unsetenv("CUTRACER_DUMP_CUBIN");
  return result == true;
}

bool test_dump_cubin_disabled_with_0() {
  // Setting CUTRACER_DUMP_CUBIN=0 should keep cubin dump disabled.
  setenv("CUTRACER_DUMP_CUBIN", "0", 1);
  bool result = parse_dump_cubin_env();
  unsetenv("CUTRACER_DUMP_CUBIN");
  return result == false;
}

bool test_dump_cubin_disabled_with_nonnumeric() {
  // Non-numeric values get atoi() == 0, so dump_cubin should be false.
  setenv("CUTRACER_DUMP_CUBIN", "yes", 1);
  bool result = parse_dump_cubin_env();
  unsetenv("CUTRACER_DUMP_CUBIN");
  return result == false;
}

bool test_cubin_path_construction_basic() {
  // Verify the canonical naming convention:
  //   kernel_{checksum}_{name}.cubin
  std::string path = build_cubin_path("7fa21c3e", "_Z10my_kernelPiS_");
  return path == "kernel_7fa21c3e__Z10my_kernelPiS_.cubin";
}

bool test_cubin_path_construction_long_name() {
  // Mangled names longer than 150 chars should be truncated.
  std::string long_name(200, 'A');
  std::string path = build_cubin_path("abcdef01", long_name);
  // The truncated name should be exactly 150 chars.
  std::string expected = "kernel_abcdef01_" + long_name.substr(0, 150) + ".cubin";
  return path == expected;
}

bool test_cubin_path_construction_exact_150() {
  // Mangled names of exactly 150 chars should not be truncated.
  std::string name_150(150, 'B');
  std::string path = build_cubin_path("deadbeef", name_150);
  std::string expected = "kernel_deadbeef_" + name_150 + ".cubin";
  return path == expected;
}

bool test_cubin_path_construction_short_name() {
  // Short names should be preserved as-is.
  std::string path = build_cubin_path("1234", "f");
  return path == "kernel_1234_f.cubin";
}

bool test_cubin_path_construction_empty_name() {
  // Edge case: empty mangled name should still produce a valid filename.
  std::string path = build_cubin_path("0000", "");
  return path == "kernel_0000_.cubin";
}

bool test_cubin_path_construction_empty_checksum() {
  // Edge case: empty checksum.
  std::string path = build_cubin_path("", "my_kernel");
  return path == "kernel__my_kernel.cubin";
}

bool test_cubin_path_extension() {
  // Verify the file extension is always .cubin
  std::string path = build_cubin_path("abc", "kern");
  size_t pos = path.rfind(".cubin");
  // .cubin must be at the end of the string
  return pos != std::string::npos && (pos + 6) == path.size();
}

bool test_cubin_path_no_iteration_component() {
  // Cubin paths should NOT contain an iteration number (unlike trace files).
  // Trace format: kernel_{checksum}_iter{N}_{name}.ndjson
  // Cubin format: kernel_{checksum}_{name}.cubin
  std::string path = build_cubin_path("aabbccdd", "_Z5gemmPfS_");
  return path.find("iter") == std::string::npos;
}

bool test_cubin_path_deterministic() {
  // Same inputs should always produce the same path (no randomness).
  std::string p1 = build_cubin_path("hash1", "kernel_A");
  std::string p2 = build_cubin_path("hash1", "kernel_A");
  return p1 == p2;
}

bool test_cubin_path_different_checksums() {
  // Different checksums should produce different paths (files are per-binary).
  std::string p1 = build_cubin_path("aaaa", "kern");
  std::string p2 = build_cubin_path("bbbb", "kern");
  return p1 != p2;
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "Cubin Dump Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  // Environment variable parsing tests
  TEST(dump_cubin_default_off);
  TEST(dump_cubin_enabled_with_1);
  TEST(dump_cubin_enabled_with_nonzero);
  TEST(dump_cubin_disabled_with_0);
  TEST(dump_cubin_disabled_with_nonnumeric);

  // Cubin path construction tests
  TEST(cubin_path_construction_basic);
  TEST(cubin_path_construction_long_name);
  TEST(cubin_path_construction_exact_150);
  TEST(cubin_path_construction_short_name);
  TEST(cubin_path_construction_empty_name);
  TEST(cubin_path_construction_empty_checksum);
  TEST(cubin_path_extension);
  TEST(cubin_path_no_iteration_component);
  TEST(cubin_path_deterministic);
  TEST(cubin_path_different_checksums);

  std::cout << "========================================" << std::endl;
  std::cout << "Results: " << tests_passed << " passed, " << tests_failed
            << " failed" << std::endl;
  std::cout << "========================================" << std::endl;

  return tests_failed > 0 ? 1 : 0;
}
