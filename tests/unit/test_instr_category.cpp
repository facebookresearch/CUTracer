/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

/**
 * @file test_instr_category.cpp
 * @brief Unit tests for instruction category detection.
 *
 * Compile and run:
 *   g++ -std=c++17 -I../../include -o test_instr_category test_instr_category.cpp
 *   ./test_instr_category
 */

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

#include "instr_category.h"

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                             \
  std::cout << "Testing: " << #name << "... "; \
  if (test_##name()) {                         \
    std::cout << "✅ PASSED" << std::endl;     \
    tests_passed++;                            \
  } else {                                     \
    std::cout << "❌ FAILED" << std::endl;     \
    tests_failed++;                            \
  }

// ============================================================================
// Test Cases
// ============================================================================

bool test_detect_utcmma() {
  // Test UTCHMMA detection (Blackwell tensor core)
  const char* sass = "UTCHMMA gdesc[UR44], gdesc[UR46], tmem[UR53], tmem[UR60], idesc[UR61], UP1 ;";
  InstrCategory cat = detect_instr_category(sass);
  return cat == InstrCategory::MMA;
}

bool test_detect_utcmma_variants() {
  // Test different UTC*MMA variants (Blackwell tensor core instructions)
  const char* variants[] = {
      // Blackwell specific patterns
      "UTCHMMA gdesc[UR44], gdesc[UR46], tmem[UR53], tmem[UR60], idesc[UR61], UP1 ;",
      "UTCIMMA gdesc[UR44], gdesc[UR46], tmem[UR53], tmem[UR60], idesc[UR61], UP1 ;",
      "UTCQMMA gdesc[UR44], gdesc[UR46], tmem[UR53], tmem[UR60], idesc[UR61], UP1 ;",
      "UTCOMMA gdesc[UR44], gdesc[UR46], tmem[UR53], tmem[UR60], idesc[UR61], UP1 ;",
      "/*0a00*/   UTCHMMA gdesc[UR44], gdesc[UR46], tmem[UR53], tmem[UR60], idesc[UR61], UP1 ;",  // With PC prefix
      // Hopper GMMA
      "HGMMA.16816.F16 desc[UR8], desc[UR12], desc[UR0], R24 ;",
  };

  for (const char* sass : variants) {
    if (detect_instr_category(sass) != InstrCategory::MMA) {
      std::cerr << "Failed for: " << sass << std::endl;
      return false;
    }
  }
  return true;
}

bool test_detect_tma_load() {
  // Test TMA load detection
  const char* sass = "UTMALDG.2D.CTA_GROUP::1 [UR8], [R16], P0";
  InstrCategory cat = detect_instr_category(sass);
  return cat == InstrCategory::TMA;
}

bool test_detect_tma_store() {
  // Test TMA store detection
  const char* sass = "UTMASTG.2D [UR8], [R16], R32";
  InstrCategory cat = detect_instr_category(sass);
  return cat == InstrCategory::TMA;
}

bool test_detect_sync() {
  // Test sync instruction detection
  const char* sass = "WARPGROUP.DEPBAR.LE 0x2";
  InstrCategory cat = detect_instr_category(sass);
  return cat == InstrCategory::SYNC;
}

bool test_detect_none() {
  // Test that non-matching instructions return NONE
  const char* sass_list[] = {
      "FADD R0, R1, R2", "LDG.E.64 R8, [R16]", "STG.E.64 [R16], R8", "MOV R0, R1", "EXIT", nullptr,
  };

  for (int i = 0; sass_list[i] != nullptr; i++) {
    if (detect_instr_category(sass_list[i]) != InstrCategory::NONE) {
      std::cerr << "Should be NONE: " << sass_list[i] << std::endl;
      return false;
    }
  }

  // Test nullptr
  if (detect_instr_category(nullptr) != InstrCategory::NONE) {
    std::cerr << "nullptr should return NONE" << std::endl;
    return false;
  }

  return true;
}

bool test_category_name() {
  // Test category name lookup
  if (strcmp(get_instr_category_name(InstrCategory::MMA), "MMA") != 0) return false;
  if (strcmp(get_instr_category_name(InstrCategory::TMA), "TMA") != 0) return false;
  if (strcmp(get_instr_category_name(InstrCategory::SYNC), "SYNC") != 0) return false;
  if (strcmp(get_instr_category_name(InstrCategory::NONE), "NONE") != 0) return false;
  return true;
}

bool test_pattern_description() {
  // Test pattern description lookup - Blackwell pattern
  const char* desc = get_instr_pattern_description("UTCHMMA gdesc[UR44], gdesc[UR46], tmem[UR53] ;");
  if (desc == nullptr) return false;
  if (strstr(desc, "Blackwell UTCHMMA") == nullptr) return false;

  // Non-matching should return nullptr
  desc = get_instr_pattern_description("FADD R0, R1, R2");
  if (desc != nullptr) return false;

  return true;
}

bool test_is_instr_category() {
  // Test isInstrCategory helper - Blackwell pattern
  const char* mma_sass = "UTCHMMA gdesc[UR44], gdesc[UR46], tmem[UR53], tmem[UR60], idesc[UR61], UP1 ;";
  if (!is_instr_category(mma_sass, InstrCategory::MMA)) return false;
  if (is_instr_category(mma_sass, InstrCategory::TMA)) return false;
  if (is_instr_category(mma_sass, InstrCategory::SYNC)) return false;

  return true;
}

bool test_get_patterns_for_category() {
  // Test getting patterns for a category
  auto mma_patterns = get_patterns_for_category(InstrCategory::MMA);
  if (mma_patterns.empty()) return false;

  // Should contain UTCHMMA (Blackwell)
  bool found_utchmma = false;
  for (const char* p : mma_patterns) {
    if (strcmp(p, "UTCHMMA") == 0) {
      found_utchmma = true;
      break;
    }
  }
  if (!found_utchmma) return false;

  // NONE category should have no patterns
  auto none_patterns = get_patterns_for_category(InstrCategory::NONE);
  if (!none_patterns.empty()) return false;

  return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "Instruction Category Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  TEST(detect_utcmma);
  TEST(detect_utcmma_variants);
  TEST(detect_tma_load);
  TEST(detect_tma_store);
  TEST(detect_sync);
  TEST(detect_none);
  TEST(category_name);
  TEST(pattern_description);
  TEST(is_instr_category);
  TEST(get_patterns_for_category);

  std::cout << "========================================" << std::endl;
  std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
  std::cout << "========================================" << std::endl;

  return tests_failed > 0 ? 1 : 0;
}
