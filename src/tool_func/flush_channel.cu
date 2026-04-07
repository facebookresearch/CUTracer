/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * See LICENSE file in the root directory for license terms.
 */

#include "utils/channel.hpp"

extern "C" __global__ void flush_channel(ChannelDev* ch_dev) {
  ch_dev->flush();
}
