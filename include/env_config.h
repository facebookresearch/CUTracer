/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#pragma once

#include <stdint.h>

// Configuration variables
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;
extern int verbose;

// Initialize configuration from environment variables
void init_config_from_env();
