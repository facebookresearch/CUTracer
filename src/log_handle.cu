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
#include <functional>

/* nvbit interface file */
#include "nvbit.h"

/* include environment configuration */
#include "env_config.h"

/* include log handle header */
#include "log_handle.h"

/* ===== Global Variables ===== */

// The main log file for the entire process run
static FILE *g_main_log_file = NULL;
// The currently active log file for kernel traces
static FILE *g_kernel_log_file = NULL;


/* ===== Utility Functions for Logging ===== */

/**
 * @brief Base function for formatted output. Uses va_list to avoid re-formatting.
 * 
 * @param file_output if true, output to the active log file
 * @param stdout_output if true, output to stdout
 * @param format format string
 * @param args variable argument list
 */
static void vfprintf_base(bool file_output, bool stdout_output, const char *format, va_list args) {
  if (!file_output && !stdout_output) {
    return;
  }

  char output_buffer[2048];
  vsnprintf(output_buffer, sizeof(output_buffer), format, args);

  if (stdout_output) {
    fprintf(stdout, "%s", output_buffer);
  }
  
  if (file_output && g_main_log_file) {
    fprintf(g_main_log_file, "%s", output_buffer);
  }
}

void lprintf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf_base(true, false, format, args);
  va_end(args);
}

void oprintf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf_base(false, true, format, args);
  va_end(args);
}

void loprintf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf_base(true, true, format, args);
  va_end(args);
}

void trace_lprintf(const char *format, ...) {
  if (!g_kernel_log_file) {
    return;
  }

  va_list args;
  va_start(args, format);
  vfprintf(g_kernel_log_file, format, args);
  va_end(args);
}

/* ===== File Management Functions ===== */

void log_open_kernel_file(CUcontext_ptr ctx, CUfunction_ptr func, uint32_t iteration) {
  // First, ensure any previous kernel log is closed.
  log_close_kernel_file();

  const char *mangled_name = nvbit_get_func_name((CUcontext)ctx, (CUfunction)func, true);
  if (!mangled_name) {
    mangled_name = "unknown_kernel";
  }

  std::hash<std::string> hasher;
  size_t name_hash = hasher(mangled_name);

  char filename[256];
  snprintf(filename, sizeof(filename), "kernel_%.150s_%zx_iter%u.log", mangled_name, name_hash, iteration);

  g_kernel_log_file = fopen(filename, "w");
  if (g_kernel_log_file) {
    loprintf("Opened kernel trace log: %s\n", filename);
  } else {
    oprintf("ERROR: Failed to open kernel trace log file: %s\n", filename);
  }
}

void log_close_kernel_file() {
  if (g_kernel_log_file) {
    fclose(g_kernel_log_file);
    g_kernel_log_file = NULL;
  }
}


void init_log_handle() {
  // Get current timestamp for filename
  time_t now = time(0);
  struct tm* timeinfo = localtime(&now);
  char timestamp[32];
  strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);
  
  char main_log_filename[256];
  snprintf(main_log_filename, sizeof(main_log_filename), "cutracer_main_%s.log", timestamp);
  
  g_main_log_file = fopen(main_log_filename, "w");
  if (!g_main_log_file) {
    // Fallback to stdout if file creation fails
    g_main_log_file = stdout;
    oprintf("WARNING: Failed to create main log file. Falling back to stdout.\n");
  }
  
  if (verbose) {
    loprintf("Log handle system initialized. Main log is %s.\n", (g_main_log_file == stdout) ? "stdout" : main_log_filename);
  }
}

void cleanup_log_handle() {
  log_close_kernel_file();
  
  if (g_main_log_file && g_main_log_file != stdout) {
    fclose(g_main_log_file);
  }

  g_main_log_file = NULL;
  
  if (verbose) {
    oprintf("Log handle system cleaned up.\n");
  }
} 