/*
 * MIT License
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Implicit Registers Collection for SASS Instructions
 *
 * NVBit's operand API only exposes registers explicitly written in assembly.
 * Some instructions implicitly use additional sequential registers based on
 * their semantics (e.g., UTMALDG uses URb+1 for barrier address).
 *
 * This module augments the operand list with these implicit registers
 * for more accurate tracing.
 */

#pragma once

#include <vector>

#include "common.h"

/**
 * @brief Context capturing operand information for implicit register collection.
 *
 * Stores operand data by type, preserving order of appearance.
 * Each vector index corresponds to the nth operand of that type.
 */
struct OperandContext {
  std::vector<int> mref_urs;              // UR numbers from MREF operands (in order)
  std::vector<int> mref_ras;              // RA numbers from MREF operands (in order)
  std::vector<int> desc_urs;              // UR numbers from MEM_DESC operands (in order)
  std::vector<int> generic_urs;           // UR numbers from GENERIC operands (in order)
  std::vector<std::string> generic_strs;  // Raw GENERIC operand strings (e.g., "gdesc[UR44]", "tmem[UR53]")
};

// Try to load internal implementation first
#if __has_include("fb/implicit_regs_fb.h")
#include "fb/implicit_regs_fb.h"
#define IMPLICIT_REGS_IMPL_DEFINED
#endif

// Fallback: no-op for OSS builds
#ifndef IMPLICIT_REGS_IMPL_DEFINED
/**
 * @brief Collect implicit registers for SASS instructions (OSS stub).
 *
 * @param sass      SASS instruction string (used to identify instruction type)
 * @param ctx       Operand context with all explicit operand info
 * @param operands  OperandLists to augment (modified in-place)
 *
 * In OSS builds, this is a no-op. Internal builds provide instruction-specific
 * logic.
 */
inline void collect_implicit_regs(const char* /*sass*/, const OperandContext& /*ctx*/, OperandLists& /*operands*/) {
  // No-op in OSS builds
}
#endif
