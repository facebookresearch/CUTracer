/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed
 * under MIT, based on original NVIDIA NVBit sample code licensed under
 * BSD-3-Clause. See LICENSE file in the root directory for Meta's license
 * terms. See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <cstdlib>

#include "analysis.h"
#include "instrument.h"
#include "nvbit.h"

/**
 * @brief Instruments an instruction to record its opcode for lightweight analysis.
 *
 * This function was developed by Meta to support analyses like instruction
 * histograms, where only the instruction's identity is needed, not its operand
 * values.
 *
 * It injects a call to the `instrument_opcode` device function, passing the
 * opcode ID, program counter (PC), and global kernel launch ID.
 */
void instrument_opcode_only(Instr* instr, int opcode_id, CTXstate* ctx_state) {
  /* insert call to the instrumentation function with its arguments */
  nvbit_insert_call(instr, "instrument_opcode", IPOINT_BEFORE);
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* opcode id */
  nvbit_add_call_arg_const_val32(instr, opcode_id);
  /* pass the pointer to the channel on the device */
  nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
  /* add "space" for kernel function pointer that will be set
   * at launch time (64 bit value at offset 0 of the dynamic
   * arguments). it is used to pass global kernel launch id*/
  nvbit_add_call_arg_launch_val64(instr, 0);
  /* add instruction offset */
  nvbit_add_call_arg_const_val64(instr, instr->getOffset());
}

/**
 * @brief Instruments an instruction to trace the values of its register operands.
 *
 * This function is based on the instrumentation logic from NVIDIA's
 * `record_reg_vals` example (`third_party/nvbit/tools/record_reg_vals/record_reg_vals.cu`).
 * It injects a call to the `instrument_reg_val` device function to capture operand values.
 *
 * Key enhancements by Meta include:
 *  - **Unified Register (UREG) Support**: Added tracing for UREGs, which was
 *    not present in the original example.
 *  - **Enhanced Context**: Passes the global kernel launch ID and the
 *    instruction's program counter (PC) to correlate data more effectively
 *    during analysis.
 *  - **Refactoring**: Encapsulated the logic into this dedicated function,
 *    separating it from the main instruction iteration loop in `cutracer.cu`.
 */
void instrument_register_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, const OperandLists& operands) {
  /* insert call to the instrumentation function with its arguments */
  nvbit_insert_call(instr, "instrument_reg_val", IPOINT_BEFORE);
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* opcode id */
  nvbit_add_call_arg_const_val32(instr, opcode_id);
  /* add pointer to channel_dev*/
  nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
  /* add "space" for kernel function pointer that will be set
   * at launch time (64 bit value at offset 0 of the dynamic
   * arguments). it is used to pass global kernel launch id*/
  nvbit_add_call_arg_launch_val64(instr, 0);
  /* add instruction offset */
  nvbit_add_call_arg_const_val64(instr, instr->getOffset());
  /* how many register values are passed next */
  nvbit_add_call_arg_const_val32(instr, operands.reg_nums.size());
  nvbit_add_call_arg_const_val32(instr, operands.ureg_nums.size());

  for (int num : operands.reg_nums) {
    /* last parameter tells it is a variadic parameter passed to
     * the instrument function record_reg_val() */
    nvbit_add_call_arg_reg_val(instr, num, true);
  }
  for (int num : operands.ureg_nums) {
    nvbit_add_call_arg_ureg_val(instr, num, true);
  }
}

/**
 * @brief Instruments a memory instruction to trace memory access details.
 *
 * This function is based on the instrumentation logic from NVIDIA's `mem_trace`
 * example. It injects a call to the `instrument_mem` device function.
 *
 * Meta's enhancements include:
 *  - **Refactoring**: Moving the instrumentation logic from the main loop in
 *    `cutracer.cu` into this modular function.
 *  - **Contextual Information**: Passing the global kernel launch ID and the
 *    instruction's program counter (PC) alongside the memory address. This
 *    allows for more detailed analysis by linking each memory access to a
 *    specific kernel launch and instruction.
 */
void instrument_memory_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx) {
  /* insert call to the instrumentation function with its
   * arguments */
  nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
  /* predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* opcode id */
  nvbit_add_call_arg_const_val32(instr, opcode_id);
  /* memory reference 64 bit address */
  nvbit_add_call_arg_mref_addr64(instr, mref_idx);
  /* add "space" for kernel function pointer that will be set
   * at launch time (64 bit value at offset 0 of the dynamic
   * arguments). it is used to pass global kernel launch id*/
  nvbit_add_call_arg_launch_val64(instr, 0);
  /* add instruction offset */
  nvbit_add_call_arg_const_val64(instr, instr->getOffset());
  /* add pointer to channel_dev*/
  nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
}
