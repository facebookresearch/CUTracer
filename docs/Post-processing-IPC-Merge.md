IPC is computed offline by merging a clean Chrome trace with CUTracer's instruction histogram and log.

## Inputs 📥
- Chrome Trace JSON (from Triton profiler)
- CUTracer Histogram CSV (`kernel_*_hist.csv`)
- CUTracer Main Log (`cutracer_main_*.log`)

## Script 🧮
Use `scripts/parse_instr_hist_trace.py`:
```bash
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

The script:
- Extracts grid/block info and kernel hash from the main log
- Aggregates histogram counts per `global_warp_id,region_id`
- Joins with Chrome trace per warp and computes `ipc = total_instruction_count / cycles`

## Notes 📝
- If multiple launches share the same kernel hash, provide `--kernel-hash` to disambiguate.
- The script validates warp coverage; mismatches usually indicate filtered or partial traces.

