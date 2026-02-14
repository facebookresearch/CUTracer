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

/* Maximum number of register operands tracked per instruction.
 * This is a configurable buffer limit, NOT a hardware constraint.
 * Increase if instructions with more operands need to be traced. */
#define MAX_REG_OPERANDS 16
#define MAX_UREG_OPERANDS 16

/* Message type enum to identify different message types */
typedef enum {
  MSG_TYPE_REG_INFO = 0,
  MSG_TYPE_MEM_ADDR_ACCESS = 1,
  MSG_TYPE_OPCODE_ONLY = 2,
  MSG_TYPE_MEM_VALUE_ACCESS = 3,  // Memory access with value tracing
  MSG_TYPE_TMA_ACCESS = 4         // TMA (Tensor Memory Accelerator) access
} message_type_t;

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
  /* 32 lanes, each thread can store up to MAX_REG_OPERANDS register values */
  uint32_t reg_vals[32][MAX_REG_OPERANDS];

  // CUTracer extensions
  uint64_t kernel_launch_id;              // Global kernel launch id
  uint64_t pc;                            // Program counter for the instruction
  int32_t num_uregs;                      // Number of unified registers
  uint32_t ureg_vals[MAX_UREG_OPERANDS];  // Unified registers shared by all threads in the same warp
} reg_info_t;

/* Based on NVIDIA mem_trace example with Meta modifications for message type support */
typedef struct {
  message_header_t header;  // Common header with type=MSG_TYPE_MEM_ADDR_ACCESS
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;
  int warp_id;
  int opcode_id;
  uint64_t addrs[32];
} mem_addr_access_t;

/**
 * @brief Memory access with value tracing structure.
 *
 * This structure captures both memory addresses AND values for detailed
 * data flow analysis. It is larger than mem_addr_access_t (~820 bytes vs ~300 bytes)
 * due to the values array.
 *
 * Used when CUTRACER_INSTRUMENT=mem_value_trace is enabled.
 * Always captured at IPOINT_AFTER for consistent timing semantics.
 */
typedef struct {
  message_header_t header;  // type=MSG_TYPE_MEM_VALUE_ACCESS
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;
  int warp_id;
  int opcode_id;
  int mem_space;           // Memory space: GLOBAL=1, SHARED=4, LOCAL=5 (matches InstrType::MemorySpace)
  int is_load;             // 1=load, 0=store
  int access_size;         // Access size in bytes (1, 2, 4, 8, 16)
  uint64_t addrs[32];      // Memory addresses for each lane
  uint32_t values[32][4];  // Values: [lane][reg_idx], max 128-bit (4x32-bit) per lane
} mem_value_access_t;

/**
 * @brief A lightweight data packet for instruction histogram analysis.
 *
 * This structure is sent from the GPU to the CPU when `OPCODE_ONLY`
 * instrumentation is enabled. It contains the minimal information required
 * to identify an instruction and its execution context without the overhead
 * of register or memory data.
 */
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

/* Host-only C++ structures */
#ifdef __cplusplus
#include <cstdint>
#include <vector>

/**
 * @brief Register indices for CPU-side static mapping.
 *
 * Maps reg_vals/ureg_vals array positions to actual register numbers.
 * Collected at instrumentation time, not transmitted over GPU channel.
 * This avoids runtime overhead and buffer size increase for static data.
 */
struct RegIndices {
  std::vector<uint8_t> reg_indices;   // R register numbers: 0-254 (R0-R254)
  std::vector<uint8_t> ureg_indices;  // UR register numbers: 0-62 (UR0-UR62)
};

#endif /* __cplusplus */

/**
 * @brief TMA (Tensor Memory Accelerator) access tracing structure.
 *
 * This structure captures TMA descriptor information for UTMALDG.2D (load)
 * and UTMASTG.2D (store) instructions. It reads the 128-byte TMA descriptor
 * from the address pointed to by the uniform register operand.
 *
 * The TMA descriptor contains tensor metadata for bulk async memory transfers
 * between global and shared memory on Hopper/Blackwell GPUs.
 *
 * Enabled via CUTRACER_INSTR_CATEGORIES=TMA.
 *
 * NOTE: This structure only contains the raw descriptor data. Parsing of
 * the descriptor fields is done on the host side (in analysis.cu or trace_writer)
 * to reduce GPU overhead and allow easier modification of decoding logic.
 */
typedef struct {
  message_header_t header;  // type=MSG_TYPE_TMA_ACCESS
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;
  int warp_id;
  int opcode_id;

  // TMA descriptor address (from uniform register, e.g., UR28)
  uint64_t desc_addr;

  // Raw TMA descriptor (128 bytes = 16 x 64-bit words)
  // The descriptor is opaque but contains:
  //   - Base address in global memory
  //   - Tensor dimensions and strides
  //   - Box (tile) dimensions
  //   - Data type, swizzle mode, etc.
  // Decoding is done on the host side for flexibility.
  uint64_t desc_raw[16];
} tma_access_t;

/**
 * @brief Decoded TMA descriptor fields (for host-side use only).
 *
 * This structure contains the decoded fields from the TMA descriptor.
 * It is populated by decode_tma_descriptor() on the host side.
 *
 * The decoding is based on NVIDIA documentation and reverse engineering.
 * The exact layout may vary by CUDA version, but the following is typical for Hopper:
 *
 * Qword 0: Global address (64 bits)
 * Qword 1-2: Global dimensions packed (up to 5 dimensions, each 32 bits)
 * Qword 3-5: Global strides packed (up to 4 strides, each 64 bits for byte offsets)
 * Qword 6-7: Box dimensions packed (up to 5 dimensions, each 32 bits)
 * Qword 8: Element strides, data type, swizzle, interleave, etc.
 */
typedef struct {
  uint64_t global_addr;       // Base address in global memory (extracted from desc)
  uint32_t global_dim[5];     // Global tensor dimensions (up to 5D)
  uint32_t global_stride[4];  // Global tensor strides (in bytes, up to 4)
  uint32_t box_dim[5];        // Box (tile) dimensions to transfer
  uint32_t element_size;      // Element size in bytes (derived from data type)
  uint32_t data_type;         // TMA data type enum value
  uint32_t swizzle_mode;      // Swizzle mode for shared memory bank conflict avoidance
  uint32_t interleave;        // Interleave mode
  uint32_t tensor_rank;       // Number of dimensions (1-5)
} tma_decoded_desc_t;

#endif /* COMMON_H */
