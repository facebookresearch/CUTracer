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

/* ===== File Management Functions ===== */

/**
 * Switches the active log to a new file for a specific kernel invocation.
 * This should be called on kernel entry.
 * @param ctx CUDA context
 * @param func CUfunction representing the kernel
 * @param iteration Current iteration of the kernel execution
 */
void log_switch_to_kernel(CUcontext_ptr ctx, CUfunction_ptr func, uint32_t iteration);

/**
 * Closes the kernel-specific log and reverts to the main process log file.
 * This should be called on kernel exit.
 */
void log_revert_to_main();


/**
 * Initializes the log handle system. Creates the main process log file.
 */
void init_log_handle();

/**
 * Cleans up the log handle system. Closes the main process log file.
 */
void cleanup_log_handle();

#endif /* LOG_HANDLE_H */ 