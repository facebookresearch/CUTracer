/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef DELAY_INJECT_CONFIG_H
#define DELAY_INJECT_CONFIG_H

#include <stdint.h>

#include <map>
#include <string>

#include "nvbit.h"

/**
 * @brief Represents a single instrumentation point for delay injection.
 *
 * Each instrumentation point captures:
 * - pc_offset: Program counter offset for the instruction (used as key)
 * - sass: The SASS assembly instruction string
 * - delay_ns: The delay value in nanoseconds
 * - enabled: Whether this point is enabled (randomly set, 50% probability)
 */
struct DelayInstrumentationPoint {
  uint64_t pc_offset;
  std::string sass;
  uint32_t delay_ns;
  bool enabled;
};

/**
 * @brief Configuration for delay instrumentation points in a single kernel.
 */
struct KernelDelayInjectConfig {
  std::string kernel_name;
  std::string timestamp;  // ISO 8601 timestamp when kernel was instrumented
  std::map<uint64_t, DelayInstrumentationPoint> instrumentation_points;  // Keyed by pc_offset
};

/**
 * @brief Master configuration containing all kernel delay configs.
 */
struct DelayInjectConfig {
  std::string version = "1.0";
  uint32_t delay_ns;
  std::map<std::string, KernelDelayInjectConfig> kernels;  // Indexed by kernel name

  bool save_to_file(const std::string& filepath) const;
};

// Global delay injection configuration
extern DelayInjectConfig g_delay_inject_config;

/**
 * @brief Initialize delay JSON configuration for export.
 *
 * Sets up the global delay config with the delay value from env.
 * This is used for exporting instrumentation points to JSON for replay.
 */
void init_delay_json_config();

/**
 * @brief Create a new kernel delay config for a given kernel name.
 *
 * Should be called once per kernel before the instruction iteration loop.
 *
 * @param kernel_name The kernel name
 * @return Pointer to the newly created kernel config
 */
KernelDelayInjectConfig* create_kernel_delay_config(const std::string& kernel_name);

/**
 * @brief Register an instrumentation point for a kernel.
 *
 * Creates and registers a DelayInstrumentationPoint from the instruction.
 *
 * @param kdc Pointer to the kernel delay config (from create_kernel_delay_config)
 * @param instr The NVBit instruction to register
 * @param delay_ns The delay value in nanoseconds
 * @param enabled Whether this point is enabled
 */
void register_delay_instrumentation_point(KernelDelayInjectConfig* kdc, Instr* instr, uint32_t delay_ns, bool enabled);

/**
 * @brief Finalize and save delay config at context termination.
 */
void finalize_delay_config();

#endif /* DELAY_INJECT_CONFIG_H */
