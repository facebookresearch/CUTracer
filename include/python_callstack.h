/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <string>
#include <vector>

/**
 * @brief Capture the Python call stack via PyTorch's CapturedTraceback API.
 *
 * Uses dlsym to dynamically resolve Python C API functions at runtime,
 * then calls torch.utils._traceback.CapturedTraceback.extract().summary()
 * to obtain symbolized Python stack frames.
 *
 * Returns an empty vector if:
 * - Python interpreter is not loaded in the process
 * - The current thread does not hold the GIL
 * - PyTorch (torch.utils._traceback) is not available
 * - Any error occurs during the Python API calls
 *
 * This function has zero compile-time dependencies on Python or PyTorch.
 */
std::vector<std::string> capture_cpu_callstack_pytorch();

/**
 * @brief Capture the Python call stack, acquiring the GIL if necessary.
 *
 * Like capture_cpu_callstack_pytorch(), but when the current thread does
 * not hold the GIL (e.g., Triton kernel launchers release it via
 * Py_BEGIN_ALLOW_THREADS before cuLaunchKernelEx), this function
 * temporarily acquires the GIL via PyGILState_Ensure() and inspects the
 * current thread's CPython frame chain directly.
 *
 * This is safe when:
 * - The current thread previously held the GIL and released it
 *   (PyThreadState and frame chain are preserved)
 * - No other thread holds the GIL while waiting for this thread
 *
 * Returns an empty vector if:
 * - Python interpreter is not loaded in the process
 * - PyGILState_Ensure/Release symbols are not available
 * - Required CPython frame-inspection symbols are not available
 * - Any error occurs during the Python API calls
 */
std::vector<std::string> capture_cpu_callstack_pytorch_acquire_gil();
