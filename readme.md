# CUTracer

CUTracer is an NVBit-based CUDA binary instrumentation tool. It cleanly separates lightweight data collection (instrumentation) from host-side processing (analysis). Typical workflows include per-warp instruction histograms (delimited by GPU clock reads) and kernel hang detection.

## Features

- NVBit-powered, runtime attach via `CUDA_INJECTION64_PATH` (no app rebuild needed)
- Multiple instrumentation modes: opcode-only, register trace, memory trace
- Built-in analyses:
  - Instruction Histogram (for Proton/Triton workflows)
  - Deadlock/Hang Detection
- CUDA Graph and stream-capture aware flows
- Deterministic kernel log file naming and CSV outputs

## Quickstart

1) Install third-party dependency (NVBit):

```bash
git clone <repo_url>
cd CUTracer
./install_third_party.sh
```

2) Build the tool:

```bash
make -j$(nproc)
```

3) Run your CUDA app with CUTracer (example: instruction histogram):

```bash
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=proton_instr_histogram \
KERNEL_FILTERS=add_kernel \
./your_app
```

Outputs in the working directory:
- `cutracer_main_YYYYMMDD_HHMMSS.log` (main log)
- `kernel_<hash>_iter<idx>_<mangled>_hist.csv` (per-kernel instruction histogram)

For a complete Triton/Proton IPC example, see Examples below and the “Post-processing: IPC Merge” section.

## Configuration (env vars)

- `CUTRACER_INSTRUMENT`: comma-separated modes: `opcode_only`, `reg_trace`, `mem_trace`
- `CUTRACER_ANALYSIS`: comma-separated analyses: `proton_instr_histogram`, `deadlock_detection`
  - Enabling `proton_instr_histogram` auto-enables `opcode_only`
  - Enabling `deadlock_detection` auto-enables `reg_trace`
- `KERNEL_FILTERS`: comma-separated substrings matching unmangled or mangled kernel names
- `INSTR_BEGIN`, `INSTR_END`: static instruction index gate during instrumentation
- `TOOL_VERBOSE`: 0/1/2

Note: The tool sets `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` to simplify channel memory handling.

## Analyses

### Instruction Histogram (proton_instr_histogram)

- Counts SASS instruction mnemonics per warp within regions delimited by clock reads (start/stop model; nested regions not supported)
- Output: one CSV per kernel launch with columns `warp_id,region_id,instruction,count`

### Deadlock / Hang Detection (deadlock_detection)

- Detects sustained hangs by identifying warps stuck in stable PC loops; logs and issues SIGTERM→SIGKILL if sustained
- Requires `reg_trace` (auto-enabled)

## Post-processing: IPC Merge

IPC is computed offline by merging a clean Chrome trace with the CUTracer histogram and log:

```bash
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

Tips:
- Provide `--kernel-hash` if multiple launches share the same kernel hash
- The script validates warp coverage; mismatches usually mean partial/filtered inputs

## Examples

VectorAdd (CUDA C++):

```bash
cd ~/CUTracer/tests/vectoradd
make && ./vectoradd
```

Triton/Proton (histogram + IPC):

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

Deadlock/Hang detection (intentional loop example):

```bash
cd ~/CUTracer/tests/hang_test
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=deadlock_detection \
python ./test_hang.py
```

## Build & Test

```bash
# Build tool
cd ~/CUTracer && make -j

# VectorAdd sample
cd ~/CUTracer/tests/vectoradd && make && ./vectoradd
```

See the tests under `tests/` for Triton/Proton and hang detection flows.

## Troubleshooting

- No CSV/log: check `CUDA_INJECTION64_PATH`, `KERNEL_FILTERS`, and write permissions
- Empty histogram: ensure kernels emit clock instructions (e.g., Triton `pl.scope`)
- High overhead: prefer opcode-only; narrow filters; use `INSTR_BEGIN/INSTR_END`
- CUDA Graph/stream capture: data is flushed at `cuGraphLaunch` exit; ensure stream sync
- IPC merge issues: resolve warp mismatches and kernel hash ambiguity with parser flags

## Code Formatting

Use the helper script:

```bash
./format.sh check   # verify formatting (no changes)
./format.sh format  # apply formatting and list changed files
```

- C/C++/CUDA: `clang-format`
- Python: `usort` → `ruff --fix` → `black`

Install the tools via your package manager (clang-format) and `pip install usort ruff black`.

## License

This repository contains code under the MIT license (Meta) and the BSD-3-Clause license (NVIDIA). See [LICENSE](LICENSE) and [LICENSE-BSD](LICENSE-BSD) for details.

## More Documentation

The full documentation lives in the [project Wiki](https://github.com/findhao/CUTracer/wiki). Key topics include Quickstart, Analyses, Post-processing, Configuration, Outputs, API & Data Structures, Developer Guide, and Troubleshooting.
