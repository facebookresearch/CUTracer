# CUTracer

CUTracer is an NVBit-based CUDA binary instrumentation tool. It cleanly separates lightweight data collection (instrumentation) from host-side processing (analysis). Typical workflows include per-warp instruction histograms (delimited by GPU clock reads) and kernel hang detection.

## Features

-   NVBit-powered, runtime attach via `CUDA_INJECTION64_PATH` (no app rebuild needed)
-   Multiple instrumentation modes: opcode-only, register trace, memory trace
-   Built-in analyses:
    -   Instruction Histogram (for Proton/Triton workflows)
    -   Deadlock/Hang Detection
-   CUDA Graph and stream-capture aware flows
-   Deterministic kernel log file naming and CSV outputs

## Requirements

All requirements are aligned with NVBit.

Unique requirements:
- **libzstd**: Required for trace compression

## Installation

1. Clone the repository:

```bash
git clone git@github.com:facebookresearch/CUTracer.git
cd CUTracer
```

2. Install system dependencies (libzstd static library for self-contained builds):

```bash
# Ubuntu/Debian
# On most Ubuntu/Debian systems, libzstd-dev provides both shared and static libs (libzstd.a).
# You can verify this with: dpkg -L libzstd-dev | grep 'libzstd.a'
# If your distribution does not ship the static library in libzstd-dev, you may need to
# build zstd from source or install a distro-specific static libzstd package.
sudo apt-get install libzstd-dev

# CentOS/RHEL/Fedora (static library for portable builds)
sudo dnf install libzstd-static

# If static library is not available, the build will fall back to dynamic linking
# and display a warning. The resulting binary will not be self-contained.
```

3. Download third-party dependencies:

```bash
./install_third_party.sh
```

This will download:
- NVBit (NVIDIA Binary Instrumentation Tool)
- nlohmann/json (JSON library for C++)

4. Build the tool:

```bash
make -j$(nproc)
```

## Quickstart

Run your CUDA app with CUTracer (example: No instrumentation):

```bash
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
./your_app
```

## Configuration (env vars)

-   `CUTRACER_INSTRUMENT`: comma-separated modes: `opcode_only`, `reg_trace`, `mem_trace`
-   `CUTRACER_ANALYSIS`: comma-separated analyses: `proton_instr_histogram`, `deadlock_detection`
    -   Enabling `proton_instr_histogram` auto-enables `opcode_only`
    -   Enabling `deadlock_detection` auto-enables `reg_trace`
-   `KERNEL_FILTERS`: comma-separated substrings matching unmangled or mangled kernel names
-   `INSTR_BEGIN`, `INSTR_END`: static instruction index gate during instrumentation
-   `TOOL_VERBOSE`: 0/1/2
-   `TRACE_FORMAT_NDJSON`: trace output format
    -   **1** (default): NDJSON+Zstd compressed (`.ndjson.zst`, ~12x compression, 92% space savings)
    -   0: Plain text (`.log`, legacy format, verbose)
    -   2: NDJSON uncompressed (`.ndjson`, for debugging)
-   `CUTRACER_ZSTD_LEVEL`: Zstd compression level (1-22, default 9)
    -   Lower values (1-3): Faster compression, slightly larger output
    -   Higher values (19-22): Maximum compression, slower but smallest output
    -   Default of 9 provides good compression with reasonable speed

Note: The tool sets `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` to simplify channel memory handling.

## Analyses

### Instruction Histogram (proton_instr_histogram)

-   Counts SASS instruction mnemonics per warp within regions delimited by clock reads (start/stop model; nested regions not supported)
-   Output: one CSV per kernel launch with columns `warp_id,region_id,instruction,count`

### Deadlock / Hang Detection (deadlock_detection)

-   Detects sustained hangs by identifying warps stuck in stable PC loops; logs and issues SIGTERM→SIGKILL if sustained
-   Requires `reg_trace` (auto-enabled)

## Examples

### Triton/Proton (histogram + IPC):

```bash
cd ~/CUTracer/tests/proton_tests

# 1) Collect histogram with CUTracer
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=proton_instr_histogram \
KERNEL_FILTERS=add_kernel \
python ./vector-add-instrumented.py

# 2) Run without CUTracer to generate a clean Chrome trace
python ./vector-add-instrumented.py

# 3) Merge and compute IPC
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

### Deadlock/Hang detection (intentional loop example):

```bash
cd ~/CUTracer/tests/hang_test
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=deadlock_detection \
python ./test_hang.py
```

## Troubleshooting

-   No CSV/log: check `CUDA_INJECTION64_PATH`, `KERNEL_FILTERS`, and write permissions
-   Empty histogram: ensure kernels emit clock instructions (e.g., Triton `pl.scope`)
-   High overhead: prefer opcode-only; narrow filters; use `INSTR_BEGIN/INSTR_END`
-   CUDA Graph/stream capture: data is flushed at `cuGraphLaunch` exit; ensure stream sync
-   IPC merge issues: resolve warp mismatches and kernel hash ambiguity with parser flags

## License

This repository contains code under the MIT license (Meta) and the BSD-3-Clause license (NVIDIA). See [LICENSE](LICENSE) and [LICENSE-BSD](LICENSE-BSD) for details.

## More Documentation

The full documentation lives in the [Wiki](https://github.com/facebookresearch/CUTracer/wiki). Key topics include Quickstart, Analyses, Post-processing, Configuration, Outputs, API & Data Structures, Developer Guide, and Troubleshooting.
