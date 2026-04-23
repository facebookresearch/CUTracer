## No CSV/log produced 🚫
- Ensure `CUDA_INJECTION64_PATH` points to `lib/cutracer.so`
- Set `KERNEL_FILTERS` to match the actual kernel name (mangled or unmangled)
- Verify working directory has write permission

## Histogram empty 📉
- Ensure the kernel contains clock reads (e.g., Triton `pl.scope` adds them)
- Check that `CUTRACER_ANALYSIS=proton_instr_histogram` is set

## High overhead 🐢
- Use `opcode_only` only; avoid `reg_trace`/`mem_addr_trace`/`mem_value_trace` unless required
- Narrow `KERNEL_FILTERS` and use `INSTR_BEGIN/INSTR_END`
- Use `CUTRACER_INSTR_CATEGORIES` to limit instrumentation to specific instruction categories (e.g., `mma,tma`)

## CUDA graph / stream capture behavior 🕸️
- Tool handles these paths; if missing outputs, validate stream capture status and synchronization
 - For captured graphs, data is flushed at `cuGraphLaunch` exit; ensure proper stream sync.

## Version issues 🧩
- `nvcc`/`ptxas` minimums are enforced by `Makefile`; check errors and adjust `ARCH`

## IPC merge issues 🔗
- Warp ID mismatch: ensure both runs target the same kernel launch; avoid filtering one side only.
- Missing `ipc` values: cycles or instruction counts missing/zero; re-check both inputs.
- Kernel hash ambiguity: pass `--kernel-hash` explicitly to the parser.

## Output directory issues 📁
- `FATAL: CUTRACER_OUTPUT_DIR '...' does not exist` → Create the directory before running
- `FATAL: ... is not a directory` → The path points to a file, not a directory
- `FATAL: ... is not writable` → Check directory permissions (`chmod`)

## Instruction category filter not working 🏷️
- `WARNING: CUTRACER_INSTR_CATEGORIES set but no valid categories found` → Valid values: `mma`, `tma`, `sync` (case-insensitive)
- Ensure the target kernel actually contains instructions in the expected categories

## Trace format issues 📄
- `FATAL: Invalid CUTRACER_TRACE_FORMAT=X` → Valid string values: `text`, `zstd`, `ndjson`, `clp`. Valid numeric values: 0, 1, 2, 3
- `WARNING: Invalid CUTRACER_ZSTD_LEVEL=X. Using default=9` → Valid range: 1–22
- To debug NDJSON content, use `CUTRACER_TRACE_FORMAT=ndjson` (or numeric `2`) for uncompressed output
- **Deprecated:** `TRACE_FORMAT_NDJSON` is still accepted but prints a deprecation notice. Migrate to `CUTRACER_TRACE_FORMAT`
