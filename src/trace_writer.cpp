// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include "trace_writer.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

#include "env_config.h"

// ============================================================================
// Constructor & Destructor
// ============================================================================

TraceWriter::TraceWriter(const std::string& filename, int trace_mode, size_t buffer_threshold)
    : filename_(filename),
      file_handle_(nullptr),
      fd_(-1),
      buffer_threshold_(buffer_threshold),
      trace_mode_(static_cast<TraceMode>(trace_mode)),
      enabled_(true),
      zstd_ctx_(nullptr),
      compression_level_(zstd_compression_level) {  // Use configurable compression level from env_config

  // Validate trace mode
  if (trace_mode < 0 || trace_mode > 3) {
    fprintf(stderr, "TraceWriter: Invalid trace_mode %d (must be 0, 1, 2, or 3)\n", trace_mode);
    enabled_ = false;
    return;
  }

  // Determine filename based on trace mode
  std::string actual_filename;

  if (trace_mode_ == TraceMode::TEXT) {
    // Mode 0: Text format - use FILE* for fprintf compatibility
    actual_filename = filename + ".log";
    file_handle_ = fopen(actual_filename.c_str(), "a");

    if (!file_handle_) {
      fprintf(stderr, "TraceWriter: Failed to open %s\n", actual_filename.c_str());
      enabled_ = false;
      return;
    }

  } else if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
    // Mode 1: NDJSON + Zstd compression - use POSIX write() for reliability
    actual_filename = filename + ".ndjson.zst";

    // Open with O_CREAT | O_WRONLY | O_APPEND
    fd_ = open(actual_filename.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
    if (fd_ < 0) {
      fprintf(stderr, "TraceWriter: Failed to open %s (errno=%d)\n", actual_filename.c_str(), errno);
      enabled_ = false;
      return;
    }

    // Initialize Zstd compression context
    zstd_ctx_ = ZSTD_createCCtx();
    if (!zstd_ctx_) {
      fprintf(stderr, "TraceWriter: Failed to initialize Zstd compression context\n");
      close(fd_);
      fd_ = -1;
      enabled_ = false;
      return;
    }

    // Pre-allocate compression buffer to avoid runtime allocation
    size_t max_compressed_size = ZSTD_compressBound(buffer_threshold);
    compressed_buffer_.resize(max_compressed_size);

  } else {  // trace_mode == UNCOMPRESSED_NDJSON || trace_mode == CLP
    // Mode 2/3: NDJSON uncompressed - use POSIX write() for reliability
    actual_filename = filename + ".ndjson";

    // Open with O_CREAT | O_WRONLY | O_APPEND
    fd_ = open(actual_filename.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
    if (fd_ < 0) {
      fprintf(stderr, "TraceWriter: Failed to open %s (errno=%d)\n", actual_filename.c_str(), errno);
      enabled_ = false;
      return;
    }
  }
}

TraceWriter::~TraceWriter() {
  // Flush any remaining data
  flush();

  // Close file handle (Mode 0)
  if (file_handle_) {
    fclose(file_handle_);
    file_handle_ = nullptr;
  }

  // Close file descriptor (Mode 1/2/3)
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }

  // If in CLP mode, write to CLP archive file after fd is closed
  if (trace_mode_ == TraceMode::CLP) {
    write_clp_archive();
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
  if (trace_mode_ == TraceMode::TEXT) {
    write_text_format(record);
  } else {
    write_json_format(record);
  }

  return true;
}

void TraceWriter::flush() {
  // Dispatch based on trace mode
  if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
    write_compressed();
  } else if (trace_mode_ == TraceMode::UNCOMPRESSED_NDJSON || trace_mode_ == TraceMode::CLP) {
    write_uncompressed();
  }
  // Mode 0 (text) doesn't buffer, so no flush needed
}

// ============================================================================
// Private Helpers
// ============================================================================

template <typename T>
void serialize_common_fields(nlohmann::json& j, const T* data) {
  j["grid_launch_id"] = data->kernel_launch_id;
  j["cta"] = {data->cta_id_x, data->cta_id_y, data->cta_id_z};
  j["warp"] = data->warp_id;
  j["opcode_id"] = data->opcode_id;

  std::stringstream pc_ss;
  pc_ss << "0x" << std::hex << data->pc;
  j["pc"] = pc_ss.str();
}

bool TraceWriter::write_data(const char* data, size_t size, const char* data_type) {
  if (fd_ < 0) return false;

  size_t total_written = 0;

  // Retry until all data is written or a fatal error occurs
  while (total_written < size) {
    ssize_t written = write(fd_, data + total_written, size - total_written);

    if (written < 0) {
      // Error occurred
      if (errno == EINTR) {
        // Interrupted by signal, retry
        continue;
      }
      // Fatal error
      fprintf(stderr, "TraceWriter: Fatal write error after %zu of %zu %s (errno=%d: %s)\n", total_written, size,
              data_type, errno, strerror(errno));
      enabled_ = false;
      return false;
    }

    // Check for write() returning 0 (no progress)
    if (written == 0) {
      fprintf(stderr, "TraceWriter: write() returned 0 after %zu of %zu %s (disk full or quota exceeded?)\n",
              total_written, size, data_type);
      enabled_ = false;
      return false;
    }

    total_written += written;
  }

  // Force data to disk (optional but recommended for reliability)
  fsync(fd_);
  return true;
}

void TraceWriter::write_uncompressed() {
  if (json_buffer_.empty() || !enabled_) return;

  // CRITICAL FIX: Move json_buffer_ to temp to prevent data corruption
  //
  // Problem: Previously used json_buffer_.data() directly during write_data(),
  // which caused random NULL bytes to appear at line starts in output files.
  //
  // Root cause: If json_buffer_ internal pointer becomes invalid during write
  // (e.g., memory reallocation, or buffer state inconsistency), we'd be
  // writing from a stale pointer. This manifested as:
  //   - Random single NULL bytes replacing '{' at JSON line starts
  //   - Different error lines on each run (non-deterministic)
  //   - Mode 1 unaffected (uses separate compressed_buffer_)
  //
  // Solution: std::move() transfers ownership to temp_buffer BEFORE write,
  // ensuring json_buffer_ is immediately emptied and safe for new data,
  // regardless of write_data() success/failure.
  std::string temp_buffer = std::move(json_buffer_);

  // json_buffer_ is now empty (moved-from state)
  // Write from the temporary buffer
  write_data(temp_buffer.data(), temp_buffer.size(), "bytes");
}

void TraceWriter::write_clp_archive() {
  // Write to CLP archive file from uncompressed ndjson file
  std::string uncompressed_ndjson_file = filename_ + ".ndjson";
  std::string clp_archive_file = filename_ + ".clp";
  std::string clp_run_cmd = "clp-s c --single-file-archive " + clp_archive_file + " " + uncompressed_ndjson_file;
  // run the clp command line to compress and remove the uncompressed ndjson file
  int rc = std::system(clp_run_cmd.c_str());
  if (rc != 0) {
    fprintf(stderr, "TraceWriter: clp-s command line failed with error code %d\n", rc);
    return;
  }
  // remove the uncompressed ndjson file
  int ec = std::remove(uncompressed_ndjson_file.c_str());
  if (ec != 0) {
    fprintf(stderr, "TraceWriter: Failed to remove uncompressed ndjson file with error code %d\n", ec);
    return;
  }
}

void TraceWriter::write_compressed() {
  if (json_buffer_.empty() || !enabled_ || !zstd_ctx_) return;

  // CRITICAL FIX: Move json_buffer_ to temp to prevent data loss
  //
  // Problem: Previously compressed json_buffer_ directly, then cleared only on
  // write success. This caused Mode 1 to lose ~50% of records (e.g., 6,835 of
  // 13,008 records written, 6,173 lost).
  //
  // Root cause: If compression or write failed, json_buffer_ wasn't cleared,
  // causing data to accumulate beyond buffer_threshold_ (1MB). When buffer
  // exceeded the pre-allocated compressed_buffer_ capacity, subsequent
  // compressions failed silently, and all remaining data was lost.
  //
  // Solution: std::move() transfers ownership to temp_buffer BEFORE compression,
  // ensuring json_buffer_ is immediately emptied. This prevents buffer overflow
  // and ensures consistent behavior whether compression/write succeeds or fails.
  std::string temp_buffer = std::move(json_buffer_);

  // json_buffer_ is now empty (moved-from state)
  // Compress from the temporary buffer
  size_t compressed_size = ZSTD_compressCCtx(zstd_ctx_, compressed_buffer_.data(), compressed_buffer_.size(),
                                             temp_buffer.data(), temp_buffer.size(), compression_level_);

  // Check for compression errors
  if (ZSTD_isError(compressed_size)) {
    fprintf(stderr, "TraceWriter: Zstd compression error: %s\n", ZSTD_getErrorName(compressed_size));
    return;  // temp_buffer is automatically destroyed, json_buffer_ remains empty
  }

  // Write the compressed data
  write_data(compressed_buffer_.data(), compressed_size, "compressed bytes");
}

void TraceWriter::serialize_reg_info(nlohmann::json& j, const reg_info_t* reg, const RegIndices* indices) {
  if (!reg) return;

  using json = nlohmann::json;

  serialize_common_fields(j, reg);

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

  // Add register indices from CPU-side static mapping
  if (indices && !indices->reg_indices.empty()) {
    json::array_t regs_indices_array;
    for (auto idx : indices->reg_indices) {
      regs_indices_array.push_back(idx);
    }
    j["regs_indices"] = regs_indices_array;
  }

  // Add unified registers if present
  if (reg->num_uregs > 0) {
    json::array_t uregs_array;
    for (int i = 0; i < reg->num_uregs; i++) {
      uregs_array.push_back(reg->ureg_vals[i]);
    }
    j["uregs"] = uregs_array;

    // Add unified register indices from CPU-side static mapping
    if (indices && !indices->ureg_indices.empty()) {
      json::array_t uregs_indices_array;
      for (auto idx : indices->ureg_indices) {
        uregs_indices_array.push_back(idx);
      }
      j["uregs_indices"] = uregs_indices_array;
    }
  }
}

void TraceWriter::serialize_mem_access(nlohmann::json& j, const mem_addr_access_t* mem) {
  if (!mem) return;

  serialize_common_fields(j, mem);

  // Convert address array (32 addresses)
  std::vector<uint64_t> addrs(mem->addrs, mem->addrs + 32);
  j["addrs"] = addrs;
}

void TraceWriter::serialize_opcode_only(nlohmann::json& j, const opcode_only_t* opcode) {
  if (!opcode) return;

  serialize_common_fields(j, opcode);
}

void TraceWriter::serialize_mem_value_access(nlohmann::json& j, const mem_value_access_t* mem) {
  if (!mem) return;

  using json = nlohmann::json;

  serialize_common_fields(j, mem);

  // Memory access metadata
  j["mem_space"] = mem->mem_space;
  j["is_load"] = (mem->is_load == 1);
  j["access_size"] = mem->access_size;

  // Convert address array (32 addresses)
  std::vector<uint64_t> addrs(mem->addrs, mem->addrs + 32);
  j["addrs"] = addrs;

  // Convert values array (32 lanes x up to 4 registers based on access_size)
  // Only include registers needed for the access size
  int regs_needed = (mem->access_size + 3) / 4;
  if (regs_needed > 4) regs_needed = 4;

  json::array_t values_array;
  for (int lane = 0; lane < 32; lane++) {
    json::array_t lane_vals;
    for (int r = 0; r < regs_needed; r++) {
      lane_vals.push_back(mem->values[lane][r]);
    }
    values_array.push_back(lane_vals);
  }
  j["values"] = values_array;
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

    case MSG_TYPE_MEM_ADDR_ACCESS: {
      const mem_addr_access_t* mem = record.data.mem_access;

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
      case MSG_TYPE_MEM_ADDR_ACCESS:
        j["type"] = "mem_addr_trace";
        j["ipoint"] = "B";  // IPOINT_BEFORE
        break;
      case MSG_TYPE_MEM_VALUE_ACCESS:
        j["type"] = "mem_value_trace";
        j["ipoint"] = "A";  // IPOINT_AFTER
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
        serialize_reg_info(j, record.data.reg_info, record.reg_indices);
        break;
      case MSG_TYPE_MEM_ADDR_ACCESS:
        serialize_mem_access(j, record.data.mem_access);
        break;
      case MSG_TYPE_MEM_VALUE_ACCESS:
        serialize_mem_value_access(j, record.data.mem_value_access);
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
      if (trace_mode_ == TraceMode::COMPRESSED_NDJSON) {
        write_compressed();
      } else {
        write_uncompressed();
      }
    }

  } catch (const std::exception& e) {
    fprintf(stderr, "TraceWriter: JSON error in write_json_format: %s\n", e.what());
  }
}
