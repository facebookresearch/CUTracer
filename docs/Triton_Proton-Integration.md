## Example kernel 🧪
`tests/proton_tests/vector-add-instrumented.py` uses `pl.scope("load_ops")` to inject implicit clock reads delimiting regions.

## Two-pass Workflow for IPC Calculation 🔁

IPC (Instructions Per Cycle) is not calculated directly by CUTracer. It is a **post-processing step** performed by the `scripts/parse_instr_hist_trace.py` Python script, which merges outputs from two separate application runs. See also: [Post-processing: IPC Merge](Post-processing-IPC-Merge.md).

## Prerequisites 📦
- `pandas` library must be installed: `pip install pandas`

## Step 1: Run with CUTracer to Generate Histogram and Log ▶️

First, run your application with CUTracer enabled to collect per-warp instruction counts.

**Recommended (wrapper):**
```bash
cutracer trace --analysis=proton_instr_histogram --kernel-filters=add_kernel \
  -- python ./vector-add-instrumented.py
```

**Equivalent (raw env vars):**
```bash
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=proton_instr_histogram \
KERNEL_FILTERS=add_kernel \
python ./vector-add-instrumented.py
```
*   **Outputs:**
    1.  **Instruction Histogram CSV** (`kernel_*_hist.csv`): Contains `warp_id,region_id,instruction,count`.
    2.  **CUTracer Log** (`cutracer_main_*.log`): Contains metadata like kernel hashes and grid dimensions.

## Step 2: Run without CUTracer to Generate a Clean Performance Trace 🧼

Next, run the same application *without* CUTracer to get accurate, low-overhead performance data.
```bash
python ./vector-add-instrumented.py
```
*   **Output:**
    3.  **Chrome Trace JSON** (`*.chrome_trace`): Contains warp execution events and cycle counts from Triton's profiler.

## Step 3: Merge Traces and Calculate IPC 🧮 (see the dedicated page for more)
```bash
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

*   **Output** 📄: The script produces a CSV with key columns: `core,cta,local_warp_id,global_warp_id,region_id,cycles,total_instruction_count,ipc`.

**Troubleshooting the Parser**

| Symptom | Likely Cause | Action |
|---|---|---|
| Missing `ipc` column values | Zero or missing cycles/instruction counts | Check trace generation; ensure histogram collection is enabled |
| No rows after merge | Kernel hash mismatch | Re-run parser without `--kernel-hash`; verify hash in log |
| `ImportError: No module named pandas` | `pandas` not installed | `pip install pandas` |
| Warp mismatch warnings | Partial / filtered trace or histogram | Verify both sources cover identical kernel launches |


## Advanced: Manually Instrumenting Kernels with Triton Override 🧩

For kernels where you cannot easily modify the source code to add `pl.scope`, you can use Triton's kernel override mechanism to manually inject region markers (`proton.record`) into the kernel's intermediate representation (TTGIR).

This allows you to define custom regions for analysis in complex kernels like those found in libraries (e.g., FlashAttention).

**Workflow:**

**Step 1: Dump TTGIR for the Target Kernel**

First, you need to get the intermediate representation of the kernel you want to analyze.

```bash
# This example uses a script from the Triton repo and a hypothetical example script.
# Adjust paths as needed.
PYTEST_VERSION=1 \
  <path_to_triton_repo>/third_party/proton/scripts/dump_ttgir.sh \
  python3 your_fused_attention_example.py
```
This command runs your script and tells Triton to dump the TTGIR for the kernels it compiles into a `ttgir_dump/` directory.

**Step 2: Create an Override Working Copy**

Copy the dumped files to a new directory. This will be your workspace for editing.

```bash
cp -r ttgir_dump override
```

**Step 3: Edit the Kernel's TTGIR to Insert Region Markers**

Navigate into the `override/` directory and find the `.ttgir` file corresponding to your kernel. Edit the file to insert `proton.record` calls to define your analysis region.

```
...
proton.record start "my_custom_region"
  ... code region of interest ...
proton.record end "my_custom_region"
...
```
**Important:** CUTracer's analysis logic does not support nested regions. It processes clock instructions as a simple start/stop sequence, so ensure every `start` has a matching `end` before another `start` is encountered to avoid misinterpreting regions.

**Step 4: Run the Application with the Override Enabled**

Tell Triton to use your modified kernel instead of recompiling it. This step is for validating that your changes are correct and don't break the kernel.

```bash
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=override \
PYTEST_VERSION=1 \
python3 your_fused_attention_example.py
```
Check that the application still runs correctly and produces the expected outputs.

**Step 5: Collect Histogram and Trace Data with CUTracer**

Now, run the application with both the Triton override and CUTracer enabled to collect the instruction histogram for your custom regions.

**Recommended (wrapper):**
```bash
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=override \
cutracer trace --analysis=proton_instr_histogram \
  --kernel-filters=<your_kernel_substring> \
  -- python3 your_fused_attention_example.py
```

**Equivalent (raw env vars):**
```bash
CUDA_INJECTION64_PATH=<path_to_cutracer>/lib/cutracer.so \
KERNEL_FILTERS=<your_kernel_substring> \
CUTRACER_ANALYSIS=proton_instr_histogram \
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=override \
python3 your_fused_attention_example.py
```

This will produce:
- A CUTracer log: `cutracer_main_*.log`
- An instruction histogram: `kernel_*_hist.csv`
- A Chrome trace from Triton's profiler.

**Step 6: Merge Traces and Calculate IPC**

Finally, use the parser script to merge the generated files and calculate IPC for your custom-defined regions.

```bash
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./your_trace.chrome_trace \
  --cutracer-trace ./kernel_..._hist.csv \
  --cutracer-log ./cutracer_main_...log \
  --output ipc_merged.csv
```

You can now analyze the `ipc_merged.csv` file to inspect performance within the regions you defined.
