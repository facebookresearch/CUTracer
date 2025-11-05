// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include "trace_writer.h"

#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

// ============================================================================
// Constructor & Destructor
// ============================================================================

TraceWriter::TraceWriter(const std::string& filename, int trace_mode, size_t buffer_threshold)
    : filename_(filename),
      file_handle_(nullptr),
      buffer_threshold_(buffer_threshold),
      trace_mode_(trace_mode),
      enabled_(true),
      zstd_ctx_(nullptr),
      compression_level_(9) {  // Default compression level 9 (balanced)

  // Validate trace mode
  if (trace_mode < 0 || trace_mode > 2) {
    fprintf(stderr, "TraceWriter: Invalid trace_mode %d (must be 0, 1, or 2)\n", trace_mode);
    enabled_ = false;
    return;
  }

  // Determine filename and open mode based on trace mode
  std::string actual_filename;
  const char* open_mode;

  if (trace_mode == 0) {
    // Mode 0: Text format
    actual_filename = filename + ".log";
    open_mode = "a";  // Append text mode

  } else if (trace_mode == 1) {
    // Mode 1: NDJSON + Zstd compression
    actual_filename = filename + ".ndjson.zstd";
    open_mode = "ab";  // Append binary mode (required for compressed data)

    // Initialize Zstd compression context
    zstd_ctx_ = ZSTD_createCCtx();
    if (!zstd_ctx_) {
      fprintf(stderr, "TraceWriter: Failed to initialize Zstd compression context\n");
      enabled_ = false;
      return;
    }

    // Pre-allocate compression buffer to avoid runtime allocation
    // ZSTD_compressBound() returns worst-case compressed size
    size_t max_compressed_size = ZSTD_compressBound(buffer_threshold);
    compressed_buffer_.resize(max_compressed_size);

  } else {  // trace_mode == 2
    // Mode 2: NDJSON uncompressed
    actual_filename = filename + ".ndjson";
    open_mode = "a";  // Append text mode
  }

  // Open output file
  file_handle_ = fopen(actual_filename.c_str(), open_mode);

  if (!file_handle_) {
    fprintf(stderr, "TraceWriter: Failed to open %s\n", actual_filename.c_str());
    enabled_ = false;
    return;
  }
}

TraceWriter::~TraceWriter() {
  // Flush any remaining data
  flush();

  // Close file
  if (file_handle_) {
    fclose(file_handle_);
  }

  // Release Zstd compression context
  if (zstd_ctx_) {
    ZSTD_freeCCtx(zstd_ctx_);
    zstd_ctx_ = nullptr;
  }
}

// ============================================================================
// Public API
// ============================================================================

bool TraceWriter::write_trace(const TraceRecord& record) {
  if (!enabled_) return false;

  // Dispatch based on trace mode
  if (trace_mode_ == 0) {
    write_text_format(record);
  } else {
    write_json_format(record);
  }

  return true;
}

void TraceWriter::flush() {
  // Dispatch based on trace mode
  if (trace_mode_ == 1) {
    write_compressed();
  } else if (trace_mode_ == 2) {
    write_uncompressed();
  }
  // Mode 0 (text) doesn't buffer, so no flush needed
}

// ============================================================================
// Private Helpers
// ============================================================================

void TraceWriter::write_uncompressed() {
  if (json_buffer_.empty() || !enabled_) return;

  // Write JSON buffer directly to file
  if (file_handle_) {
    size_t written = fwrite(json_buffer_.data(), 1, json_buffer_.size(), file_handle_);
    if (written != json_buffer_.size()) {
      fprintf(stderr, "TraceWriter: Write error (wrote %zu of %zu bytes)\n", written, json_buffer_.size());
    }

    fflush(file_handle_);
  }

  // Clear buffer
  json_buffer_.clear();
}

void TraceWriter::write_compressed() {
  if (json_buffer_.empty() || !enabled_ || !zstd_ctx_) return;

  // Compress JSON buffer using Zstd
  size_t compressed_size = ZSTD_compressCCtx(zstd_ctx_, compressed_buffer_.data(), compressed_buffer_.size(),
                                             json_buffer_.data(), json_buffer_.size(), compression_level_);

  // Check for compression errors
  if (ZSTD_isError(compressed_size)) {
    fprintf(stderr, "TraceWriter: Zstd compression error: %s\n", ZSTD_getErrorName(compressed_size));
    return;
  }

  // Write compressed data to file
  if (file_handle_) {
    size_t written = fwrite(compressed_buffer_.data(), 1, compressed_size, file_handle_);
    if (written != compressed_size) {
      fprintf(stderr, "TraceWriter: Write error (wrote %zu of %zu compressed bytes)\n", written, compressed_size);
    }

    fflush(file_handle_);
  }

  // Clear buffer
  json_buffer_.clear();
}

void TraceWriter::serialize_reg_info(nlohmann::json& j, const reg_info_t* reg) {
  if (!reg) return;

  using json = nlohmann::json;

  // Basic fields
  j["grid_launch_id"] = reg->kernel_launch_id;
  j["cta"] = {reg->cta_id_x, reg->cta_id_y, reg->cta_id_z};
  j["warp"] = reg->warp_id;
  j["opcode_id"] = reg->opcode_id;
  j["pc"] = reg->pc;

  // CRITICAL: Transpose register array
  // C layout: reg_vals[thread][reg] → JSON: regs[reg][thread]
  // This ensures all values for the same register across all threads
  // are grouped together in the JSON output.
  json::array_t regs_array;
  for (int reg_idx = 0; reg_idx < reg->num_regs; reg_idx++) {
    json::array_t thread_vals;
    for (int thread = 0; thread < 32; thread++) {
      thread_vals.push_back(reg->reg_vals[thread][reg_idx]);
    }
    regs_array.push_back(thread_vals);
  }
  j["regs"] = regs_array;

  // Add unified registers if present
  if (reg->num_uregs > 0) {
    json::array_t uregs_array;
    for (int i = 0; i < reg->num_uregs; i++) {
      uregs_array.push_back(reg->ureg_vals[i]);
    }
    j["uregs"] = uregs_array;
  }
}

void TraceWriter::serialize_mem_access(nlohmann::json& j, const mem_access_t* mem) {
  if (!mem) return;

  // Basic fields
  j["grid_launch_id"] = mem->kernel_launch_id;
  j["cta"] = {mem->cta_id_x, mem->cta_id_y, mem->cta_id_z};
  j["warp"] = mem->warp_id;
  j["opcode_id"] = mem->opcode_id;
  j["pc"] = mem->pc;

  // Convert address array (32 addresses)
  std::vector<uint64_t> addrs(mem->addrs, mem->addrs + 32);
  j["addrs"] = addrs;
}

void TraceWriter::serialize_opcode_only(nlohmann::json& j, const opcode_only_t* opcode) {
  if (!opcode) return;

  // Basic fields (minimal - opcode_only is lightweight)
  j["grid_launch_id"] = opcode->kernel_launch_id;
  j["cta"] = {opcode->cta_id_x, opcode->cta_id_y, opcode->cta_id_z};
  j["warp"] = opcode->warp_id;
  j["opcode_id"] = opcode->opcode_id;
  j["pc"] = opcode->pc;
}

// ============================================================================
// Format-specific output methods
// ============================================================================

void TraceWriter::write_text_format(const TraceRecord& record) {
  if (!file_handle_) return;

  // Dispatch by trace type
  switch (record.type) {
    case MSG_TYPE_REG_INFO: {
      const reg_info_t* ri = record.data.reg_info;

      // Print header line
      fprintf(file_handle_, "CTX %p - CTA %d,%d,%d - warp %d - %s:\n", record.context, ri->cta_id_x, ri->cta_id_y,
              ri->cta_id_z, ri->warp_id, record.sass_instruction.c_str());

      // Print register values
      for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
        fprintf(file_handle_, "  * ");
        for (int i = 0; i < 32; i++) {
          fprintf(file_handle_, "Reg%d_T%02d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
        }
        fprintf(file_handle_, "\n");
      }

      // Print uniform register values (if present)
      if (ri->num_uregs > 0) {
        fprintf(file_handle_, "  * UR: ");
        for (int i = 0; i < ri->num_uregs; i++) {
          fprintf(file_handle_, "UR%d: 0x%08x ", i, ri->ureg_vals[i]);
        }
        fprintf(file_handle_, "\n");
      }

      fprintf(file_handle_, "\n");
      break;
    }

    case MSG_TYPE_MEM_ACCESS: {
      const mem_access_t* mem = record.data.mem_access;

      // Print header
      fprintf(file_handle_, "CTX %p - kernel_launch_id %ld - CTA %d,%d,%d - warp %d - PC %ld - %s:\n", record.context,
              mem->kernel_launch_id, mem->cta_id_x, mem->cta_id_y, mem->cta_id_z, mem->warp_id, mem->pc,
              record.sass_instruction.c_str());

      // Print memory addresses
      fprintf(file_handle_, "  Memory Addresses:\n  * ");
      int printed = 0;
      for (int i = 0; i < 32; i++) {
        if (mem->addrs[i] != 0) {
          fprintf(file_handle_, "T%02d: 0x%016lx ", i, mem->addrs[i]);
          printed++;
          if (printed % 4 == 0 && i < 31) {
            fprintf(file_handle_, "\n    ");
          }
        }
      }
      fprintf(file_handle_, "\n\n");
      break;
    }

    case MSG_TYPE_OPCODE_ONLY: {
      // Opcode_only typically doesn't output to trace files (only for histogram)
      // But we can add support if needed
      break;
    }

    default:
      fprintf(stderr, "TraceWriter: Unknown message type %d in text mode\n", record.type);
      break;
  }

  fflush(file_handle_);
}

void TraceWriter::write_json_format(const TraceRecord& record) {
  try {
    using json = nlohmann::json;
    json j;

    // ========== Serialize metadata ==========

    // Type string
    switch (record.type) {
      case MSG_TYPE_REG_INFO:
        j["type"] = "reg_trace";
        break;
      case MSG_TYPE_MEM_ACCESS:
        j["type"] = "mem_trace";
        break;
      case MSG_TYPE_OPCODE_ONLY:
        j["type"] = "opcode_only";
        break;
      default:
        fprintf(stderr, "TraceWriter: Unknown message type %d\n", record.type);
        return;
    }

    // Context pointer (as hex string)
    std::stringstream ss;
    ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(record.context);
    j["ctx"] = ss.str();

    // SASS instruction (if available)
    if (!record.sass_instruction.empty()) {
      j["sass"] = record.sass_instruction;
    }

    // Trace index
    j["trace_index"] = record.trace_index;

    // Timestamp
    j["timestamp"] = record.timestamp;

    // ========== Serialize data (dispatch by type) ==========

    switch (record.type) {
      case MSG_TYPE_REG_INFO:
        serialize_reg_info(j, record.data.reg_info);
        break;
      case MSG_TYPE_MEM_ACCESS:
        serialize_mem_access(j, record.data.mem_access);
        break;
      case MSG_TYPE_OPCODE_ONLY:
        serialize_opcode_only(j, record.data.opcode_only);
        break;
    }

    // ========== Buffer and flush ==========

    // Append to JSON buffer (NDJSON format - newline is critical!)
    json_buffer_ += j.dump() + "\n";

    // Check if buffer threshold reached
    if (json_buffer_.size() >= buffer_threshold_) {
      // Dispatch based on trace mode
      if (trace_mode_ == 1) {
        write_compressed();
      } else {
        write_uncompressed();
      }
    }

  } catch (const std::exception& e) {
    fprintf(stderr, "TraceWriter: JSON error in write_json_format: %s\n", e.what());
  }
}
