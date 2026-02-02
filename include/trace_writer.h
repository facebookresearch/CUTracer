// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <zstd.h>

#include <cstdint>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "common.h"
#include "nvbit.h"

/**
 * @brief Rich trace record combining GPU trace data with metadata.
 *
 * This structure acts as an intermediate layer between raw GPU communication
 * structs (reg_info_t, mem_addr_access_t, opcode_only_t) and the JSON output format.
 *
 * Benefits:
 * - GPU communication protocol remains unchanged
 * - Easy to add metadata fields without modifying GPU structs
 * - One record = one JSON line (atomic write operation)
 */
struct TraceRecord {
  // ========== Metadata (not in original GPU structs) ==========

  /**
   * @brief CUDA context pointer (for debugging/correlation).
   *
   * Source: recv_thread_fun() args
   * JSON field: "ctx" (hex string)
   */
  CUcontext context;

  /**
   * @brief SASS instruction string for human readability.
   *
   * Source: ctx_state->id_to_sass_map[func][opcode_id]
   * JSON field: "sass"
   */
  std::string sass_instruction;

  /**
   * @brief Per-kernel trace sequence number (monotonically increasing).
   *
   * Used to track trace ordering within a kernel for analysis.
   * Source: Maintained by analysis.cu per kernel_launch_id
   * JSON field: "trace_index"
   */
  uint64_t trace_index;

  /**
   * @brief Host-side timestamp when trace was received (nanoseconds).
   *
   * Source: Current time when trace is processed
   * JSON field: "timestamp"
   */
  uint64_t timestamp;

  // ========== Original trace data ==========

  /**
   * @brief Type of the trace record.
   */
  message_type_t type;

  /**
   * @brief Union containing pointer to original GPU data.
   *
   * Lifetime: Caller ensures the pointed-to struct remains valid
   * until write_trace() completes (typically local variable in analysis.cu).
   */
  union {
    const reg_info_t* reg_info;
    const mem_addr_access_t* mem_access;
    const mem_value_access_t* mem_value_access;
    const opcode_only_t* opcode_only;
  } data;

  // ========== Constructors for convenience ==========

  /**
   * @brief Create a TraceRecord for reg_info_t.
   */
  static TraceRecord create_reg_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                      const reg_info_t* reg) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_REG_INFO;
    record.data.reg_info = reg;
    return record;
  }

  /**
   * @brief Create a TraceRecord for mem_addr_access_t.
   */
  static TraceRecord create_mem_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                      const mem_addr_access_t* mem) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_MEM_ADDR_ACCESS;
    record.data.mem_access = mem;
    return record;
  }

  /**
   * @brief Create a TraceRecord for mem_value_access_t.
   */
  static TraceRecord create_mem_value_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                            const mem_value_access_t* mem_value) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_MEM_VALUE_ACCESS;
    record.data.mem_value_access = mem_value;
    return record;
  }

  /**
   * @brief Create a TraceRecord for opcode_only_t.
   */
  static TraceRecord create_opcode_trace(CUcontext ctx, const std::string& sass, uint64_t trace_idx, uint64_t ts,
                                         const opcode_only_t* opcode) {
    TraceRecord record;
    record.context = ctx;
    record.sass_instruction = sass;
    record.trace_index = trace_idx;
    record.timestamp = ts;
    record.type = MSG_TYPE_OPCODE_ONLY;
    record.data.opcode_only = opcode;
    return record;
  }
};

/**
 * @brief TraceWriter for trace output in multiple formats.
 *
 * Unified writer supporting three output modes:
 * - Mode 0: Text format (legacy .log files)
 * - Mode 1: NDJSON + Zstd compression (.ndjson.zst)
 * - Mode 2: NDJSON uncompressed (.ndjson)
 *
 * Key features:
 * - Single unified API: write_trace(TraceRecord)
 * - Automatic format dispatch based on trace_mode
 * - Buffering for I/O efficiency
 * - Metadata (ctx, sass, trace_index, timestamp) automatically included
 */
class TraceWriter {
 private:
  std::string filename_;
  FILE* file_handle_;
  int fd_;  // File descriptor for POSIX write() (Mode 1/2 only)
  std::string json_buffer_;
  size_t buffer_threshold_;
  int trace_mode_;  // 0, 1, or 2
  bool enabled_;

  // ========== Mode 1 (Zstd compression) support ==========
  ZSTD_CCtx* zstd_ctx_;                  // Zstd compression context
  std::vector<char> compressed_buffer_;  // Pre-allocated compression output buffer
  int compression_level_;                // Zstd compression level (1-22, default 22)

 public:
  /**
   * @brief Construct TraceWriter.
   *
   * @param filename Base filename (extension added automatically based on mode)
   * @param trace_mode 0 for text, 1 for compressed JSON, 2 for uncompressed JSON
   * @param buffer_threshold Buffer flush threshold (default 1MB)
   */
  TraceWriter(const std::string& filename, int trace_mode, size_t buffer_threshold = 1024 * 1024);

  /**
   * @brief Destructor - flushes remaining data and closes file.
   */
  ~TraceWriter();

  // Disable copy
  TraceWriter(const TraceWriter&) = delete;
  TraceWriter& operator=(const TraceWriter&) = delete;

  /**
   * @brief Write a trace record to output.
   *
   * Mode 0: Formats as text and writes to .log file
   * Mode 1/2: Serializes to JSON and writes to .ndjson[.zst] file
   *
   * @param record Complete trace record with all information
   * @return true if successful, false on error
   */
  bool write_trace(const TraceRecord& record);

  /**
   * @brief Flush buffered data to disk.
   */
  void flush();

  /**
   * @brief Check if writer is functional.
   */
  bool is_enabled() const {
    return enabled_;
  }

 private:
  // ========== Output methods ==========

  /**
   * @brief Write record in text format (mode 0).
   */
  void write_text_format(const TraceRecord& record);

  /**
   * @brief Write record in JSON format (mode 1/2).
   */
  void write_json_format(const TraceRecord& record);

  /**
   * @brief Write uncompressed buffer to file (mode 2).
   */
  void write_uncompressed();

  /**
   * @brief Compress and write buffer to file (mode 1).
   *
   * Compresses json_buffer_ using Zstd and writes to .ndjson.zst file.
   * Each flush creates an independent Zstd frame for incremental writing.
   */
  void write_compressed();

  /**
   * @brief Reliably write data to file with retry logic.
   *
   * Handles partial writes, EINTR interrupts, and detects zero-write deadlock.
   * On fatal error, sets enabled_ = false.
   *
   * @param data Pointer to data buffer
   * @param size Number of bytes to write
   * @param data_type Description for error messages (e.g., "bytes", "compressed bytes")
   * @return true if all data written successfully, false on error
   */
  bool write_data(const char* data, size_t size, const char* data_type);

  // ========== JSON serialization ==========

  /**
   * @brief Serialize reg_info_t fields to JSON object.
   */
  void serialize_reg_info(nlohmann::json& j, const reg_info_t* reg);

  /**
   * @brief Serialize mem_addr_access_t fields to JSON object.
   */
  void serialize_mem_access(nlohmann::json& j, const mem_addr_access_t* mem);

  /**
   * @brief Serialize opcode_only_t fields to JSON object.
   */
  void serialize_opcode_only(nlohmann::json& j, const opcode_only_t* opcode);

  /**
   * @brief Serialize mem_value_access_t fields to JSON object.
   */
  void serialize_mem_value_access(nlohmann::json& j, const mem_value_access_t* mem);
};
