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
