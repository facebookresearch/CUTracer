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

#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

/* Message type enum to identify different message types */
typedef enum { MSG_TYPE_REG_INFO = 0, MSG_TYPE_MEM_ACCESS = 1, MSG_TYPE_OPCODE_ONLY = 2 } message_type_t;

/* Common header for all message types */
typedef struct {
  message_type_t type;  // Type of the message
} message_header_t;

/* Based on NVIDIA record_reg_vals example with Meta modifications for message type support and adds CUTracer specific
 * extensions */
typedef struct {
  message_header_t header;  // Common header with type=MSG_TYPE_REG_INFO
  int32_t cta_id_x;
  int32_t cta_id_y;
  int32_t cta_id_z;
  int32_t warp_id;
  int32_t opcode_id;
  int32_t num_regs;
  /* 32 lanes, each thread can store up to 8 register values */
  uint32_t reg_vals[32][8];

  // CUTracer extensions
  uint64_t kernel_launch_id;  // Global kernel launch id
  uint64_t pc;                // Program counter for the instruction
  int32_t num_uregs;          // Number of unified registers
  uint32_t ureg_vals[8];      // Unified registers shared by all threads in the same warp
} reg_info_t;

/* Based on NVIDIA mem_trace example with Meta modifications for message type support */
typedef struct {
  message_header_t header;  // Common header with type=MSG_TYPE_MEM_ACCESS
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;
  int warp_id;
  int opcode_id;
  uint64_t addrs[32];
} mem_access_t;

typedef struct {
  message_header_t header;  // Common header with type=MSG_TYPE_OPCODE_ONLY
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;
  int warp_id;
  int opcode_id;
} opcode_only_t;

#endif /* COMMON_H */
