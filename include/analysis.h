/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed under MIT,
 * based on original NVIDIA nvbit sample code licensed under BSD-3-Clause.
 * See LICENSE file in the root directory for Meta's license terms.
 * See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <stdint.h>

#include <map>
#include <string>

// Forward declarations
class ChannelHost;

/* Channel buffer size */
#define CHANNEL_SIZE (1l << 20)

/* Thread state enum */
enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};

/* Receiver thread function */
void* recv_thread_fun(void*);

/* Initialize receiver thread dependencies */
void init_recv_thread_deps(ChannelHost* host, volatile RecvThreadState* thread_state,
                           std::map<int, std::string>* sass_map);

#endif /* ANALYSIS_H */
