/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Dynamically calls PyTorch's CapturedTraceback Python API to capture
 * the Python call stack at CUDA kernel launch time. Uses dlsym to resolve
 * Python C API functions at runtime — no compile-time dependency on
 * Python.h or libpython.
 *
 * This is the same approach used by tritonparse (structured_logging.py),
 * adapted for C++ via the Python C API.
 */

#include "python_callstack.h"

#include <dlfcn.h>
#include <sys/types.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Python C API types — defined locally to avoid requiring Python.h
using PyObject = void;
using Py_ssize_t = ssize_t;

// ---------------------------------------------------------------------------
// Python C API function pointers (resolved lazily via dlsym)
// ---------------------------------------------------------------------------

static int (*p_Py_IsInitialized)() = nullptr;
static int (*p_PyGILState_Check)() = nullptr;
static PyObject* (*p_PyImport_ImportModule)(const char*) = nullptr;
static PyObject* (*p_PyObject_GetAttrString)(PyObject*, const char*) = nullptr;
static PyObject* (*p_PyObject_CallMethod)(PyObject*, const char*, const char*, ...) = nullptr;
static Py_ssize_t (*p_PyList_Size)(PyObject*) = nullptr;
static PyObject* (*p_PyList_GetItem)(PyObject*, Py_ssize_t) = nullptr;
static PyObject* (*p_PyObject_GetAttr)(PyObject*, PyObject*) = nullptr;
static PyObject* (*p_PyUnicode_FromString)(const char*) = nullptr;
static const char* (*p_PyUnicode_AsUTF8)(PyObject*) = nullptr;
static long (*p_PyLong_AsLong)(PyObject*) = nullptr;
static void (*p_Py_DecRef)(PyObject*) = nullptr;
static void (*p_Py_IncRef)(PyObject*) = nullptr;
static PyObject* (*p_PyErr_Occurred)() = nullptr;
static void (*p_PyErr_Clear)() = nullptr;

// Tri-state resolution: allows retry if Python is not yet loaded
enum class PythonApiState {
  UNRESOLVED,
  UNAVAILABLE_RETRYABLE,
  RESOLVED,
};

static PythonApiState python_api_state = PythonApiState::UNRESOLVED;
static std::mutex python_api_mutex;

// Cached CapturedTraceback class (set after first successful import)
static PyObject* cached_captured_traceback_cls = nullptr;

/**
 * @brief Resolves Python C API symbols from the current process, retrying
 *        later calls until resolution succeeds.
 *
 * Uses dlopen(NULL) to search all symbols already loaded into the process.
 * If libpython is not yet loaded or its symbols are not globally visible,
 * this returns false for now but does not permanently cache that failure.
 * Once all required symbols resolve, the result is cached for the remainder
 * of the process.
 *
 * Thread-safe: serialized with a mutex so concurrent NVBit callback threads
 * do not race while publishing the resolved function pointers.
 */
static bool resolve_python_api() {
  std::lock_guard<std::mutex> lock(python_api_mutex);
  if (python_api_state == PythonApiState::RESOLVED) {
    return true;
  }

  void* handle = dlopen(nullptr, RTLD_NOW);
  if (!handle) {
    python_api_state = PythonApiState::UNAVAILABLE_RETRYABLE;
    return false;
  }

  // Resolve into local temporaries first, then publish atomically
  // to avoid leaving globals in a half-resolved state on failure.
  decltype(p_Py_IsInitialized) r_Py_IsInitialized = nullptr;
  decltype(p_PyGILState_Check) r_PyGILState_Check = nullptr;
  decltype(p_PyImport_ImportModule) r_PyImport_ImportModule = nullptr;
  decltype(p_PyObject_GetAttrString) r_PyObject_GetAttrString = nullptr;
  decltype(p_PyObject_CallMethod) r_PyObject_CallMethod = nullptr;
  decltype(p_PyList_Size) r_PyList_Size = nullptr;
  decltype(p_PyList_GetItem) r_PyList_GetItem = nullptr;
  decltype(p_PyObject_GetAttr) r_PyObject_GetAttr = nullptr;
  decltype(p_PyUnicode_FromString) r_PyUnicode_FromString = nullptr;
  decltype(p_PyUnicode_AsUTF8) r_PyUnicode_AsUTF8 = nullptr;
  decltype(p_PyLong_AsLong) r_PyLong_AsLong = nullptr;
  decltype(p_Py_DecRef) r_Py_DecRef = nullptr;
  decltype(p_Py_IncRef) r_Py_IncRef = nullptr;
  decltype(p_PyErr_Occurred) r_PyErr_Occurred = nullptr;
  decltype(p_PyErr_Clear) r_PyErr_Clear = nullptr;

#define RESOLVE(name)                                         \
  r_##name = (decltype(r_##name))dlsym(handle, #name);        \
  if (!r_##name) {                                            \
    dlclose(handle);                                          \
    python_api_state = PythonApiState::UNAVAILABLE_RETRYABLE; \
    return false;                                             \
  }

  RESOLVE(Py_IsInitialized);
  RESOLVE(PyGILState_Check);
  RESOLVE(PyImport_ImportModule);
  RESOLVE(PyObject_GetAttrString);
  RESOLVE(PyObject_CallMethod);
  RESOLVE(PyList_Size);
  RESOLVE(PyList_GetItem);
  RESOLVE(PyObject_GetAttr);
  RESOLVE(PyUnicode_FromString);
  RESOLVE(PyUnicode_AsUTF8);
  RESOLVE(PyLong_AsLong);
  RESOLVE(Py_DecRef);
  RESOLVE(Py_IncRef);
  RESOLVE(PyErr_Occurred);
  RESOLVE(PyErr_Clear);

#undef RESOLVE

  dlclose(handle);

  // Publish all resolved pointers atomically
  p_Py_IsInitialized = r_Py_IsInitialized;
  p_PyGILState_Check = r_PyGILState_Check;
  p_PyImport_ImportModule = r_PyImport_ImportModule;
  p_PyObject_GetAttrString = r_PyObject_GetAttrString;
  p_PyObject_CallMethod = r_PyObject_CallMethod;
  p_PyList_Size = r_PyList_Size;
  p_PyList_GetItem = r_PyList_GetItem;
  p_PyObject_GetAttr = r_PyObject_GetAttr;
  p_PyUnicode_FromString = r_PyUnicode_FromString;
  p_PyUnicode_AsUTF8 = r_PyUnicode_AsUTF8;
  p_PyLong_AsLong = r_PyLong_AsLong;
  p_Py_DecRef = r_Py_DecRef;
  p_Py_IncRef = r_Py_IncRef;
  p_PyErr_Occurred = r_PyErr_Occurred;
  p_PyErr_Clear = r_PyErr_Clear;

  python_api_state = PythonApiState::RESOLVED;
  return true;
}

// ---------------------------------------------------------------------------
// RAII helper for PyObject* reference counting
// ---------------------------------------------------------------------------

namespace {
struct PyRef {
  PyObject* obj;

  explicit PyRef(PyObject* o) : obj(o) {
  }

  ~PyRef() {
    if (obj) {
      p_Py_DecRef(obj);
    }
  }

  // Non-copyable, movable
  PyRef(const PyRef&) = delete;
  PyRef& operator=(const PyRef&) = delete;

  PyRef(PyRef&& other) noexcept : obj(other.obj) {
    other.obj = nullptr;
  }

  PyRef& operator=(PyRef&& other) noexcept {
    if (this != &other) {
      if (obj) {
        p_Py_DecRef(obj);
      }
      obj = other.obj;
      other.obj = nullptr;
    }
    return *this;
  }

  // Implicit conversion to PyObject* for passing to API calls
  operator PyObject*() const {
    return obj;
  }  // NOLINT

  explicit operator bool() const {
    return obj != nullptr;
  }

  // Release ownership without decrementing the reference count
  PyObject* release() {
    PyObject* tmp = obj;
    obj = nullptr;
    return tmp;
  }
};
}  // namespace

// ---------------------------------------------------------------------------
// CapturedTraceback class caching
// ---------------------------------------------------------------------------

/**
 * @brief Resolve and cache the CapturedTraceback class.
 *
 * Imports torch.utils._traceback and caches the CapturedTraceback class
 * after first successful resolution. On failure, does not cache so the
 * next call can retry (e.g., if PyTorch was not yet imported).
 *
 * The caller must hold the GIL.
 *
 * @return Borrowed reference to CapturedTraceback class, or nullptr.
 */
static PyObject* get_captured_traceback_cls() {
  if (cached_captured_traceback_cls) {
    return cached_captured_traceback_cls;
  }

  PyRef mod(p_PyImport_ImportModule("torch.utils._traceback"));
  if (!mod) {
    p_PyErr_Clear();
    return nullptr;
  }

  PyRef cls(p_PyObject_GetAttrString(mod, "CapturedTraceback"));
  if (!cls) {
    p_PyErr_Clear();
    return nullptr;
  }

  // Cache with an owned reference (prevent GC from collecting it)
  cached_captured_traceback_cls = cls.release();
  return cached_captured_traceback_cls;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<std::string> capture_cpu_callstack_pytorch() {
  // Reentrancy guard: if Python code executed during capture (e.g., module
  // import side effects) triggers a CUDA operation, the NVBit callback could
  // re-enter this function. Detect and bail out to prevent infinite recursion.
  static thread_local bool capturing = false;
  if (capturing) {
    return {};
  }
  capturing = true;

  std::vector<std::string> result;

  // Ensure capturing is reset on all exit paths
  auto reset_guard = [](bool* flag) { *flag = false; };
  std::unique_ptr<bool, decltype(reset_guard)> guard(&capturing, reset_guard);

  // Step 1: Check if Python C API symbols are available in this process
  if (!resolve_python_api()) {
    return result;
  }
  if (!p_Py_IsInitialized()) {
    return result;
  }

  // Step 2: Only proceed if the current thread already holds the GIL.
  // In the typical PyTorch kernel launch path:
  //   Python user code (holds GIL) → PyTorch Python → ATen C++ → CUDA runtime
  //     → NVBit callback → here
  // The GIL is still held. For C++ background threads (e.g., CUDA Graph replay,
  // DataLoader workers), this check returns false and we fall back to backtrace.
  if (!p_PyGILState_Check()) {
    return result;
  }

  // Current thread already holds the GIL — safe to call Python APIs directly.

  // Step 3/4: Resolve CapturedTraceback class (cached after first success)
  PyObject* cls = get_captured_traceback_cls();
  if (!cls) {
    return result;
  }

  // Step 5: tb = CapturedTraceback.extract()
  // The C++ NVBit frames are not visible in the Python stack, so skip=0.
  // CapturedTraceback.extract() internally adds skip=1 to elide itself.
  PyRef tb(p_PyObject_CallMethod(cls, "extract", nullptr));
  if (!tb) {
    p_PyErr_Clear();
    return result;
  }

  // Step 6: frames = tb.summary()
  // Returns a traceback.StackSummary (list of traceback.FrameSummary)
  PyRef summary(p_PyObject_CallMethod(tb, "summary", nullptr));
  if (!summary) {
    p_PyErr_Clear();
    return result;
  }

  // Step 7: Iterate frames and extract filename, name, lineno
  Py_ssize_t n = p_PyList_Size(summary);
  if (n < 0) {
    p_PyErr_Clear();
    return result;
  }

  // Pre-create attribute name strings (reused across all frames)
  PyRef attr_filename(p_PyUnicode_FromString("filename"));
  PyRef attr_name(p_PyUnicode_FromString("name"));
  PyRef attr_lineno(p_PyUnicode_FromString("lineno"));
  if (!attr_filename || !attr_name || !attr_lineno) {
    p_PyErr_Clear();
    return result;
  }

  result.reserve(n);
  for (Py_ssize_t i = 0; i < n; i++) {
    // PyList_GetItem returns a borrowed reference — do NOT Py_DECREF
    PyObject* frame = p_PyList_GetItem(summary, i);
    if (!frame) {
      continue;
    }

    PyRef py_filename(p_PyObject_GetAttr(frame, attr_filename));
    PyRef py_name(p_PyObject_GetAttr(frame, attr_name));
    PyRef py_lineno(p_PyObject_GetAttr(frame, attr_lineno));

    const char* filename = py_filename ? p_PyUnicode_AsUTF8(py_filename) : nullptr;
    const char* funcname = py_name ? p_PyUnicode_AsUTF8(py_name) : nullptr;
    long lineno = py_lineno ? p_PyLong_AsLong(py_lineno) : 0;

    // Clear any per-frame errors from PyUnicode_AsUTF8 / PyLong_AsLong
    // to avoid leaving a stale exception that could affect subsequent
    // Python API calls in the loop.
    if (p_PyErr_Occurred()) {
      p_PyErr_Clear();
      if (lineno == -1) {
        lineno = 0;
      }
    }

    // Format: "filename:lineno in funcname"
    // Matches the style used by traceback.format_list()
    std::string frame_str = std::string(filename ? filename : "??") + ":" + std::to_string(lineno) + " in " +
                            std::string(funcname ? funcname : "??");
    result.push_back(std::move(frame_str));
  }

  // Final safety clear for any stale errors
  if (p_PyErr_Occurred()) {
    p_PyErr_Clear();
  }

  return result;
}
