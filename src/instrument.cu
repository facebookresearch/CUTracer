/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 */

#include <cstdlib>
#include <cstring>

#include "env_config.h"
#include "instrument.h"
#include "nvbit.h"

void instrument_opcode_only(Instr* instr, int opcode_id, CTXstate* ctx_state) {
  /* insert call to the instrumentation function with its arguments */
  nvbit_insert_call(instr, "instrument_opcode", IPOINT_BEFORE);
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
}

void instrument_register_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, const std::vector<int>& reg_num_list,
                               const std::vector<int>& ureg_num_list) {
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
  nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
  nvbit_add_call_arg_const_val32(instr, ureg_num_list.size());

  for (int num : reg_num_list) {
    /* last parameter tells it is a variadic parameter passed to
     * the instrument function record_reg_val() */
    nvbit_add_call_arg_reg_val(instr, num, true);
  }
  for (int num : ureg_num_list) {
    nvbit_add_call_arg_ureg_val(instr, num, true);
  }
}

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
