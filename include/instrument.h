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
  OPCODE_ONLY,      // Lightweight: only collect opcode information
  REG_TRACE,        // Medium: collect register values
  MEM_ADDR_TRACE,   // Heavy: collect memory access information (address only)
  MEM_VALUE_TRACE,  // Heavy: collect memory access with values
  RANDOM_DELAY      // Inject random delays on synchronization instructions
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
void instrument_memory_addr_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx);

/**
 * @brief Insert memory access tracing with value capture instrumentation
 *
 * Collects memory addresses AND values for data flow analysis.
 * Always uses IPOINT_AFTER for consistent timing semantics.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 * @param mref_idx Memory reference index
 * @param mem_space Memory space type (obtained via instr->getMemorySpace() in cutracer.cu)
 */
void instrument_memory_value_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx, int mem_space);

/**
 * @brief Instruments an instruction to inject a fixed delay.
 *
 * Inserts a call to the `instrument_delay` device function before the
 * instruction. The delay value is a fixed value determined by CUTRACER_DELAY_NS.
 *
 * @param instr The instruction to instrument
 * @param delay_ns Fixed delay in nanoseconds
 */
void instrument_delay_injection(Instr* instr, uint32_t delay_ns);

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
bool shouldInjectDelay(Instr* instr, const std::vector<const char*>& patterns);

#endif /* INSTRUMENT_H */
