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

// Configuration variables
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;
extern int verbose;

// Kernel name filters
extern std::vector<std::string> kernel_filters;
// Instrumentation configuration
extern std::unordered_set<InstrumentType> enabled_instrument_types;

// Initialize configuration from environment variables
void init_config_from_env();

// Check if a specific instrumentation type is enabled
bool is_instrument_type_enabled(InstrumentType type);

// Initialize instrumentation configuration
void init_instrumentation(const std::string &instrument_str);
