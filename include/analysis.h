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

#include "nvbit.h"
/* for channel */
#include "utils/channel.hpp"

/* Channel buffer size */
#define CHANNEL_SIZE (1l << 20)

/* Thread state enum */
enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};

struct CTXstate {
  /* context id */
  int id;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev* channel_dev;
  ChannelHost channel_host;

  // After initialization, set it to WORKING to make recv thread get data,
  // parent thread sets it to STOP to make recv thread stop working.
  // recv thread sets it to FINISHED when it cleans up.
  // parent thread should wait until the state becomes FINISHED to clean up.
  volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

  // map from function to map from offset to SASS string
  std::map<CUfunction, std::map<int, std::string>> id_to_sass_map;
};

/* Receiver thread function */
void* recv_thread_fun(void*);

#endif /* ANALYSIS_H */
