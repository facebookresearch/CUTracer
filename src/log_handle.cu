/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed under MIT,
 * based on original NVIDIA nvbit sample code licensed under BSD-3-Clause.
 * See LICENSE file in the root directory for Meta's license terms.
 * See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <string>

/* nvbit interface file */
#include "nvbit.h"

/* include environment configuration */
#include "env_config.h"

/* include log handle header */
#include "log_handle.h"

/* ===== Global Variables ===== */

/* File handling */
FILE *log_handle = NULL;
FILE *log_handle_main_trace = NULL;

/* ===== Utility Functions for Logging ===== */

/**
 * Base function for formatted output to different destinations
 */
void base_fprintf(bool file_output, bool stdout_output, const char *format, ...) {
  // if no output, return
  if (!file_output && !stdout_output) return;

  char output_buffer[2048];  // use a large enough buffer
  
  va_list args;
  va_start(args, format);
  vsnprintf(output_buffer, sizeof(output_buffer), format, args);
  va_end(args);

  // output to stdout
  if (stdout_output) {
    fprintf(stdout, "%s", output_buffer);
  }

  // output to log file (if not stdout)
  if (file_output && log_handle != NULL && log_handle != stdout) {
    fprintf(log_handle, "%s", output_buffer);
  }
}

/**
 * lprintf - print to log file only (log print)
 */
void lprintf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  
  char output_buffer[2048];
  vsnprintf(output_buffer, sizeof(output_buffer), format, args);
  va_end(args);
  
  base_fprintf(true, false, "%s", output_buffer);
}

/**
 * oprintf - print to stdout only (output print)
 */
void oprintf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  
  char output_buffer[2048];
  vsnprintf(output_buffer, sizeof(output_buffer), format, args);
  va_end(args);
  
  base_fprintf(false, true, "%s", output_buffer);
}

/**
 * loprintf - print to log file and stdout (log and output print)
 */
void loprintf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  
  char output_buffer[2048];
  vsnprintf(output_buffer, sizeof(output_buffer), format, args);
  va_end(args);
  
  base_fprintf(true, true, "%s", output_buffer);
}

/* ===== File Management Functions ===== */

/**
 * Creates the intermediate trace file if needed
 */
void create_trace_file(const char *custom_filename, bool create_new_file) {
  // For cutracer.cu, we'll use a simplified approach
  // Always log to stdout for now, can be enhanced later
  if (log_handle != NULL && log_handle != stdout) {
    fclose(log_handle);
  }
  log_handle = stdout;
  
  if (custom_filename != nullptr) {
    oprintf("Writing traces to %s\n", custom_filename);
  } else {
    oprintf("Writing traces to stdout\n");
  }
}

/**
 * Truncates a mangled function name to make it suitable for use as a filename
 */
void truncate_mangled_name(const char *mangled_name, char *truncated_buffer, size_t buffer_size) {
  if (!truncated_buffer || buffer_size == 0) {
    return;
  }

  // Default to unknown if no name provided
  if (!mangled_name) {
    snprintf(truncated_buffer, buffer_size, "unknown_kernel");
    return;
  }

  // Truncate the name if it's longer than buffer_size - 1 (leave room for null terminator)
  size_t max_length = buffer_size - 1;
  size_t name_len = strlen(mangled_name);

  if (name_len > max_length) {
    strncpy(truncated_buffer, mangled_name, max_length);
    truncated_buffer[max_length] = '\0';  // Ensure null termination
  } else {
    strcpy(truncated_buffer, mangled_name);
  }
}

/**
 * Creates a log file specifically for a kernel based on its mangled name and iteration count
 */
void create_kernel_log_file(CUcontext_ptr ctx, CUfunction_ptr func, uint32_t iteration) {
  // For cutracer.cu, we'll use a simplified approach
  // Get mangled function name for file naming
  const char *mangled_name = nvbit_get_func_name((CUcontext)ctx, (CUfunction)func, true);

  // Create a buffer for the truncated name
  char truncated_name[201];  // 200 chars + null terminator

  // Truncate the name
  truncate_mangled_name(mangled_name, truncated_name, sizeof(truncated_name));

  // Create a filename with the truncated name
  char filename[256];
  snprintf(filename, sizeof(filename), "%s_iter%u.log", truncated_name, iteration);

  // Create trace file with the custom filename
  create_trace_file(filename, true);
}

/**
 * Initializes the log handle system
 */
void init_log_handle() {
  // Initialize log handle to stdout by default
  log_handle = stdout;
  log_handle_main_trace = stdout;
  
  if (verbose) {
    oprintf("Log handle system initialized\n");
  }
}

/**
 * Cleans up the log handle system
 */
void cleanup_log_handle() {
  if (log_handle != NULL && log_handle != stdout) {
    fclose(log_handle);
    log_handle = NULL;
  }
  
  if (log_handle_main_trace != NULL && log_handle_main_trace != stdout) {
    fclose(log_handle_main_trace);
    log_handle_main_trace = NULL;
  }
  
  if (verbose) {
    oprintf("Log handle system cleaned up\n");
  }
} 