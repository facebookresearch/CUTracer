/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef INSTRUMENT_H
#define INSTRUMENT_H

#include <vector>

#include "analysis.h"

/**
 * @brief Instrumentation types for different data collection modes
 */
enum class InstrumentType {
  OPCODE_ONLY,  // Lightweight: only collect opcode information
  REG_TRACE,    // Medium: collect register values
  MEM_TRACE,    // Heavy: collect memory access information
  RANDOM_DELAY  // Inject random delays on synchronization instructions
};

/**
 * @brief Structure to hold operand information for instrumentation
 *
 * This structure groups operand data needed for instrumentation,
 * making it easy to extend support for additional operand types without
 * changing function signatures.
 */
struct OperandLists {
  std::vector<int> reg_nums;   // Regular register numbers
  std::vector<int> ureg_nums;  // Uniform register numbers
  // Future: add support for other types like pred_nums, generic_vals, etc.
};

/**
 * @brief Insert lightweight opcode-only instrumentation for instruction histogram analysis
 *
 * This is optimized for Proton instruction statistic analysis where only opcode
 * information is needed for histogram generation.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 */
void instrument_opcode_only(Instr* instr, int opcode_id, CTXstate* ctx_state);

/**
 * @brief Insert register tracing instrumentation
 *
 * Collects register values for detailed register flow analysis.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 * @param operands Structure containing all operand information (reg, ureg, etc.)
 */
void instrument_register_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, const OperandLists& operands);

/**
 * @brief Insert memory access tracing instrumentation
 *
 * Collects memory access information for memory pattern analysis.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 * @param mref_idx Memory reference index
 */
void instrument_memory_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx);

/**
 * @brief Insert random delay instrumentation for synchronization instructions
 *
 * Injects a random delay before eligible synchronization instructions.
 * The delay is computed on the host side, so each instruction gets a unique
 * random value. This is useful for exposing potential race conditions.
 *
 * @param instr The instruction to instrument
 * @param max_delay_ns Maximum random delay in nanoseconds
 */
void instrument_random_delay(Instr* instr, uint32_t max_delay_ns);

/**
 * @brief SASS instruction patterns for delay injection.
 */
static const std::vector<const char*> DELAY_INJECTION_PATTERNS = {
    "SYNCS.PHASECHK.TRANS64.TRYWAIT",  // mbarrier try_wait
    "SYNCS.ARRIVE.TRANS64.RED.A1T0",   // mbarrier arrive
    "UTMALDG.2D",                      // TMA load
    "WARPGROUP.DEPBAR.LE",             // MMA wait
};

/**
 * @brief Check if an instruction should have delay injected
 *
 * @param instr The instruction to check
 * @param patterns Vector of SASS substrings to match against
 * @return true if the instruction matches any delay injection pattern
 */
bool isInstrForDelayInjection(Instr* instr, const std::vector<const char*>& patterns);

#endif /* INSTRUMENT_H */
