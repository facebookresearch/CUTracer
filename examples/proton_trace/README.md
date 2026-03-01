# Example Proton Trace

This directory contains example output files from a CUTracer proton instruction
histogram run. These files illustrate the data CUTracer produces when tracing a
Triton kernel compiled via Proton instrumentation.

## Background

[Triton](https://github.com/triton-lang/triton) is a language and compiler for
writing GPU kernels.  [Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton)
is Triton's built-in profiler.  When Proton is configured with the
`instrumentation` backend, it inserts clock-read instructions into the generated
SASS that delimit user-defined regions (via `pl.scope`).

CUTracer hooks into this flow through its **proton_instr_histogram** analysis.
It intercepts every executed SASS instruction at runtime and buckets them by
`(warp_id, region_id, mnemonic)`.  The result is a per-launch CSV histogram that
tells you exactly which instructions each warp executed in each region and how
many times.

## Source Kernel

The trace was generated from the vector-add kernel in
`tests/proton_tests/vector-add-instrumented.py`.  Key parameters:

| Parameter     | Value   |
|---------------|---------|
| `n_elements`  | 98 432  |
| `BLOCK_SIZE`  | 1 024   |
| `num_warps`   | 1       |
| Grid size     | 97,1,1  |
| Block size    | 32,1,1  |

Because `num_warps=1`, each CTA uses a single warp.  The grid launches 97 CTAs,
giving 97 global warps (warp IDs 0-96).

## Files

### `example_hist.csv`

Instruction histogram CSV produced by CUTracer with
`CUTRACER_ANALYSIS=proton_instr_histogram`.

**Format:** `warp_id,region_id,instruction,count`

- `warp_id` -- global warp ID across all CTAs
- `region_id` -- region index (delimited by Proton clock reads inside `pl.scope`)
- `instruction` -- SASS mnemonic (e.g., `LDG.E.64`, `FADD`, `STG.E.64`)
- `count` -- number of times the instruction was executed by this warp in this region

Region 0 covers the body of the `load_ops` scope (loads, add, and supporting
instructions).  Region 1 covers the final store outside the scope.

### `example_cutracer.log`

CUTracer main log (`cutracer_main_*.log`) emitted during the same run.  It
records kernel instrumentation events such as:

- Kernel launch metadata (hash, grid/block dimensions)
- Instrumentation enable/disable decisions based on `KERNEL_FILTERS`
- Histogram flush events

## How to Reproduce

```bash
cd tests/proton_tests

# 1) Collect instruction histogram with CUTracer
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=proton_instr_histogram \
KERNEL_FILTERS=add_kernel \
python ./vector-add-instrumented.py

# Generated files:
#   kernel_<hash>_add_kernel_hist.csv   (instruction histogram)
#   cutracer_main_<pid>.log             (CUTracer log)

# 2) Generate a clean Chrome trace (no CUTracer overhead)
python ./vector-add-instrumented.py

# 3) Merge histogram + Chrome trace to compute per-warp IPC
python scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

## Parsing the Histogram

The histogram CSV can be loaded directly with pandas:

```python
import pandas as pd

df = pd.read_csv("example_hist.csv")

# Total instructions per warp
per_warp = df.groupby("warp_id")["count"].sum()
print(per_warp.describe())

# Instruction mix across all warps in region 0
region0 = df[df["region_id"] == 0]
mix = region0.groupby("instruction")["count"].sum().sort_values(ascending=False)
print(mix)
```
