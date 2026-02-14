/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Instruction Category Tracing - Public Interface
 *
 * This header provides the public API for instruction-specific tracing
 * (TMA, MMA, etc.). The actual implementation is in fb/instrument_fb.cu
 * for internal builds; OSS builds get no-op stubs.
 */

#pragma once

#include <vector>

#include "common.h"

// Forward declarations
class Instr;
struct CTXstate;

// Include internal implementation if available, otherwise provide stubs
#if __has_include("fb/instrument_fb.h")
#include "fb/instrument_fb.h"
#else

/**
 * @brief Opaque operand structure for instruction categories (OSS stub).
 *
 * Internal builds use a richer structure defined in fb/instrument_fb.h.
 */
struct CategoryOperands {};

/**
 * @brief Extract category-specific operands from a TMA instruction.
 *
 * OSS stub - returns false (no extraction).
 *
 * @param instr The instruction to extract operands from
 * @param operands Output structure for extracted operands
 * @return true if extraction successful, false otherwise
 */
inline bool extract_tma_operands(Instr* instr, CategoryOperands& operands) {
  (void)instr;
  (void)operands;
  return false;
}

/**
 * @brief Instrument a TMA instruction for tracing.
 *
 * OSS stub - does nothing.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier
 * @param ctx_state The context state
 * @param operands The extracted operands
 */
inline void instrument_tma_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, const CategoryOperands& operands) {
  (void)instr;
  (void)opcode_id;
  (void)ctx_state;
  (void)operands;
}

/**
 * @brief Parse TMA descriptor raw data into structured fields.
 *
 * OSS stub - does nothing.
 *
 * @param desc_raw The raw 128-byte descriptor (16 x 64-bit words)
 * @param decoded Output structure for decoded fields
 */
inline void decode_tma_descriptor(const uint64_t* desc_raw, tma_decoded_desc_t& decoded) {
  (void)desc_raw;
  memset(&decoded, 0, sizeof(decoded));
}

#endif  // __has_include
