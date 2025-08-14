## Feature Overview: Kernel Hash + IPC Parsing + Example Workflows

> NOTE: This feature set depends on the Triton proton profiler available on the latest Triton `main` branch. If your local Triton install is outdated you may be missing `triton.profiler` / `proton` APIs and the examples will fail.
> Quick checklist:
> 1. Clone or update: `git clone git@github.com:triton-lang/triton.git` (or `git pull origin main` in an existing clone).
> 2. Install editable: `pip install -e .`
> 3. Sanity check: `python -c "import triton.profiler as proton; print('proton OK')"`
>
> If step 3 fails, upgrade your Triton environment before using these examples.

This directory now contains example scripts demonstrating the end-to-end workflow for:

1. Adding stable kernel hash identifiers to CUTracer logs (branch: `findhao/add_hash_to_log`).
2. Parsing and merging trace timing + instruction histogram data to compute per-warp IPC (branch: `findhao/add_ipc_parser`).
3. A richer Triton fused attention example instrumented for trace collection (branch: `findhao/add_ipc_example`).

These three parts together enable correlating Triton kernel launches to per-warp instruction counts and timing for performance analysis.

### 1. Kernel Hash in Logs (`findhao/add_hash_to_log`)
Changes (C++/CUDA):
- Added `compute_kernel_name_hash_hex` and enhanced `generate_kernel_log_basename` in `include/log.h` / `src/log.cu`.
- Injected `kernel hash 0x<hex>` into each LAUNCH line in `src/cutracer.cu`.

Why:
- Provides a deterministic, short identifier to match runtime logs with downstream analysis outputs (e.g., CSV/Chrome trace) independent of long mangled names.
- Facilitates filtering/selecting a specific kernel instance when multiple launches exist.

Log line excerpt after change:
```
CUTracer: ... LAUNCH - Kernel name <mangled> - kernel hash 0x7fa21c3 - kernel launch id 42 - grid size ...
```

### 2. IPC Parser Script (`findhao/add_ipc_parser`)
Added file: `scripts/parse_instr_hist_trace.py` (Python).

Capabilities:
- Parses Chrome trace JSON (warp events) and CUTRacer histogram CSV (instruction counts per warp/region).
- Extracts grid/block dimensions (and optional `--kernel-hash` filter) from CUTracer log.
- Derives `global_warp_id = cta * warps_per_block + local_warp_id`.
- Validates warp coverage and merges timing + counts; computes IPC = instructions / cycles.
- Standalone modes: Chrome-only or histogram-only parsing.

Usage (merge mode):
```bash
python scripts/parse_instr_hist_trace.py \
  --chrome-trace chrome_trace.json \
  --cutracer-trace instruction_hist.csv \
  --cutracer-log cutracer_run.log \
  --output merged_ipc.csv
```

Outputs CSV columns (subset):
```
core,cta,local_warp_id,global_warp_id,region_id,name,category,cycles,timestamp_ns,total_instruction_count,ipc
```

Notes:
- Requires `pandas`. Install if needed: `pip install pandas`.
- Continues even if warp ID mismatches are detected (warns explicitly).

### 3. Fused Attention IPC Example (`findhao/add_ipc_example`)
Moved example to: `tutorial/ipc_parser_example.py`.

Highlights:
- Triton FlashAttention v2 style forward/backward reference & Triton kernels.
- Integrates Triton profiler (`proton`) with instrumentation backend collecting cycle-level trace.
- Demonstrates a more complex multi-kernel workload than the simple vector add, suitable for IPC + histogram correlation.

Minimal run (forward path example):
```bash
python tutorial/ipc_parser_example.py
```
This will produce trace/log artifacts that can be fed into the parser script to compute IPC by warp/region.

### Suggested End-to-End Workflow
1. Run CUTracer-instrumented workload (e.g., `python tutorial/ipc_parser_example.py`) to generate:
   - CUTracer log with kernel hash lines.
   - Chrome trace JSON (if enabled by your workflow).
   - Instruction histogram CSV (from histogram instrumentation feature).
2. Identify the target kernel hash from the log.
3. Run the parser script with `--kernel-hash` to produce merged IPC CSV.
   ```bash
   python scripts/parse_instr_hist_trace.py --chrome-trace ...
   ```
4. Load the CSV into Python / notebook for deeper analysis (group by region, aggregate IPC, etc.).

### Detailed Reproduction Steps (Fused Attention + IPC + Histogram)
These steps show how to override a Triton kernel, inject proton region markers, collect histogram + trace, and compute per-warp IPC.

Prereqs:
- Triton main installed editable.
- CUTracer built, with `cutracer.so` path correctly configured.

**Note on paths**: The `ttgir_dump` are part of the Triton repository. The following steps may require path adjustments depending on your directory structure.

Steps:

Step 1: Dump TTGIR for the original fused attention kernel
```bash
PYTEST_VERSION=1 \
  <path_to_triton>/third_party/proton/scripts/dump_ttgir.sh \
  python3 tutorial/ipc_parser_example.py
```
This produces a `ttgir_dump/` directory containing kernel IR and metadata.

Step 2: Create an override working copy
```bash
cp -r ttgir_dump override
```

Step 3: Edit kernel TTGIR / generated code
Insert non-nested proton region markers (cannot be nested):
```
proton.record start "cutracer"
  ... code region of interest ...
proton.record end "cutracer"
```
Make sure you do not place a `start` inside another active `start` before its matching `end`.

Step 4: Run with kernel override to validate correctness
```bash
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=override \
PYTEST_VERSION=1 \
python3 tutorial/ipc_parser_example.py
```
Confirm the application still runs and the region markers appear in the trace (Chrome trace / proton output).

Step 5: Collect histogram + trace with CUTracer + analysis mode
```bash
CUDA_INJECTION64_PATH=<path_to>/lib/cutracer.so \
KERNEL_FILTERS=attn_fwd \
TRITON_ALWAYS_COMPILE=1 \
CUTRACER_ANALYSIS=proton_instr_histogram \
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=override \
python3 tutorial/ipc_parser_example.py
```

Outputs to look for:
- CUTracer log: `cutracer_main_<timestamp>.log` (contains LAUNCH lines with kernel hash)
- Histogram CSV: `kernel_<hash>_iter0__attn_fwd_hist.csv`
- Chrome trace: e.g. `fa.chrome_trace`

Step 6: Merge and compute per-warp IPC
```bash
python ../scripts/parse_instr_hist_trace.py \
  --chrome-trace ./fa.chrome_trace \
  --cutracer-trace ./kernel_..._hist.csv \
  --cutracer-log ./cutracer_main_...log \
  --output ipc_merged.csv
```
Open `ipc_merged.csv` to analyze IPC by warp/region.

Optional sanity check: run without `--kernel-hash` first (if multiple kernels) to ensure the parser sees the correct warp span.

### Future Enhancements
- Add automated pytest + golden output for parser and hash extraction.
- Provide a small sample dataset in `tests/data/` for offline parsing demonstration.
- Extend parser to output summary stats (mean/median IPC per region) and JSON.
- Document environment variables controlling trace/histogram generation.

### Troubleshooting
| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Missing `ipc` column values | Zero or missing cycles/instruction counts | Check trace generation settings; ensure histogram collection is enabled |
| No rows after merge | Kernel hash mismatch | Re-run without `--kernel-hash` to list available events; verify hash in log |
| Pandas import error | `pandas` not installed | `pip install pandas` |
| Warp mismatch warnings | Partial / filtered trace or histogram | Verify both sources cover identical launches |

---
Existing simple example (vector add) remains available for quick sanity checks.
