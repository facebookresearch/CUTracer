/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#pragma once

#include <stdint.h>
#include <vector>
#include <string>

// Configuration variables
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;
extern int verbose;

// Kernel name filters
extern std::vector<std::string> kernel_filters;
extern bool any_kernel_matched;

// Initialize configuration from environment variables
void init_config_from_env();
