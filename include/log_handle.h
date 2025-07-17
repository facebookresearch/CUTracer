/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed under MIT,
 * based on original NVIDIA nvbit sample code licensed under BSD-3-Clause.
 * See LICENSE file in the root directory for Metas license terms.
 * See LICENSE-BSD file in the root directory for NVIDIAs license terms.
 */

#ifndef LOG_HANDLE_H
#define LOG_HANDLE_H

#include <stdio.h>
#include <stdint.h>
#include <time.h>

// Forward declarations - use void* to avoid conflicts with CUDA types
typedef void* CUcontext_ptr;
typedef void* CUfunction_ptr;

/* ===== Log Handle Management ===== */

// Global log file handles
extern FILE *log_handle;
extern FILE *log_handle_main_trace;

/* ===== Logging Functions ===== */

/**
 * Base template function for formatted output to different destinations
 * @param file_output if true, output to log file
 * @param stdout_output if true, output to stdout
 * @param format format string
 * @param args variable argument list
 */
template <typename... Args>
void base_fprintf(bool file_output, bool stdout_output, const char *format, Args... args);

/**
 * lprintf - print to log file only (log print)
 */
template <typename... Args>
void lprintf(const char *format, Args... args);

/**
 * oprintf - print to stdout only (output print)
 */
template <typename... Args>
void oprintf(const char *format, Args... args);

/**
 * loprintf - print to log file and stdout (log and output print)
 */
template <typename... Args>
void loprintf(const char *format, Args... args);

/* ===== File Management Functions ===== */

/**
 * Creates the intermediate trace file if needed
 * @param custom_filename Optional custom filename for the log file
 * @param create_new_file Whether to create a new file or reuse existing
 */
void create_trace_file(const char *custom_filename = nullptr, bool create_new_file = false);

/**
 * Truncates a mangled function name to make it suitable for use as a filename
 * @param mangled_name The original mangled name
 * @param truncated_buffer The buffer to store the truncated name in
 * @param buffer_size Size of the provided buffer
 */
void truncate_mangled_name(const char *mangled_name, char *truncated_buffer, size_t buffer_size);

/**
 * Creates a log file specifically for a kernel based on its mangled name and iteration count
 * @param ctx CUDA context
 * @param func CUfunction representing the kernel
 * @param iteration Current iteration of the kernel execution
 */
void create_kernel_log_file(CUcontext_ptr ctx, CUfunction_ptr func, uint32_t iteration);

/**
 * Initializes the log handle system
 */
void init_log_handle();

/**
 * Cleans up the log handle system
 */
void cleanup_log_handle();

#endif /* LOG_HANDLE_H */ 