# Use static libzstd for self-contained builds

## Summary

This PR modifies the build system to statically link libzstd, making cutracer.so self-contained and portable across different environments where the dynamic libzstd may not be available.

## Changes

- **Makefile**: Dynamically search for `libzstd.a` in common system paths (`/usr/lib64`, `/usr/lib`, `/usr/local/lib64`, `/usr/local/lib`) with pkg-config fallback
- **Makefile**: Improved warning messages when static library is not found - now explicitly states that the build will fall back to dynamic linking and the resulting binary will NOT be self-contained/portable
- **Makefile**: Added `-lpthread` to link flags (required because zstd uses POSIX threads internally for multi-threaded compression)
- **README**: Updated installation instructions with verification steps for Ubuntu/Debian users to confirm `libzstd.a` is available

## Why

When using CUTracer via `CUDA_INJECTION64_PATH` in certain environments, the dynamic `libzstd.so.1` may not be available in the library search path, causing library load failures. Static linking resolves this dependency issue by embedding zstd into cutracer.so.

## Test Plan

1. Install static zstd library:
   - RHEL/Fedora: `sudo dnf install libzstd-static`
   - Ubuntu/Debian: `sudo apt install libzstd-dev` (verify with `dpkg -L libzstd-dev | grep libzstd.a`)
2. Rebuild CUTracer: `make clean && make`
3. Verify no dynamic zstd dependency: `ldd lib/cutracer.so | grep zstd` (should be empty)
4. Test with `CUDA_INJECTION64_PATH` in target environment
