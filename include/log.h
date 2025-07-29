/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef LOG_HANDLE_H
#define LOG_HANDLE_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>

// Forward declarations - use void* to avoid conflicts with CUDA types
typedef void *CUcontext_ptr;
typedef void *CUfunction_ptr;

/* ===== Logging Functions ===== */

/**
 * lprintf - print to the currently active log file only (log print)
 */
void lprintf(const char *format, ...);

/**
 * oprintf - print to stdout only (output print)
 */
void oprintf(const char *format, ...);

/**
 * loprintf - print to the currently active log file and stdout (log and output print)
 */
void loprintf(const char *format, ...);

/**
 * trace_lprintf - print to the kernel trace log file only
 */
void trace_lprintf(const char *format, ...);

/* ===== File Management Functions ===== */

/**
 * Opens a new log file for a specific kernel invocation.
 * This should be called on kernel entry.
 * @param ctx CUDA context
 * @param func CUfunction representing the kernel
 * @param iteration Current iteration of the kernel execution
 */
void log_open_kernel_file(CUcontext_ptr ctx, CUfunction_ptr func, uint32_t iteration);

/**
 * Closes the kernel-specific log file.
 * This should be called on kernel exit.
 */
void log_close_kernel_file();

/**
 * Initializes the log handle system. Creates the main process log file.
 */
void init_log_handle();

/**
 * Cleans up the log handle system. Closes the main process log file.
 */
void cleanup_log_handle();

#endif /* LOG_HANDLE_H */
