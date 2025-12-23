# PR: Make zstd compression level configurable via environment variable

## Summary

Add `CUTRACER_ZSTD_LEVEL` environment variable to configure zstd compression level (1-22). This allows users to trade off compression speed vs compression ratio based on their use case.

## Changes

- `include/env_config.h`: Add extern declaration for `zstd_compression_level`
- `src/env_config.cu`: Add environment variable reading logic with validation (range 1-22)
- `src/trace_writer.cpp`: Use configurable compression level instead of hardcoded value 22
- `readme.md`: Document new `CUTRACER_ZSTD_LEVEL` environment variable

## Configuration

`CUTRACER_ZSTD_LEVEL`: Zstd compression level (1-22, default 3)
- Lower values (1-3): Faster compression, slightly larger output
- Higher values (19-22): Maximum compression, slower but smallest output
- Default of 3 provides balanced speed and compression ratio

## Motivation

The previous hardcoded compression level of 22 (maximum) prioritized compression ratio over speed. For most use cases, a lower compression level (e.g., 3) provides nearly the same compression ratio with significantly faster compression. This change allows users to choose the trade-off that best suits their workflow.

## Example Usage

```bash
# Fast compression (level 1)
CUTRACER_ZSTD_LEVEL=1 CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so ./app

# Maximum compression (level 22)
CUTRACER_ZSTD_LEVEL=22 CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so ./app

# Default (level 3 - balanced)
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so ./app
```

## Test Plan

1. Build CUTracer: `make -j$(nproc)`
2. Verify default level works: Run with `TRACE_FORMAT_NDJSON=1` and check output
3. Verify custom level: Set `CUTRACER_ZSTD_LEVEL=1` and verify faster compression
4. Verify validation: Set invalid value (e.g., 25) and verify warning + fallback to default
