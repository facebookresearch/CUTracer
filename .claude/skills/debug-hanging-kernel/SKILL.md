---
description: >-
  End-to-end deadlock diagnosis for hanging GPU kernels. Captures instruction-level
  trace, runs deterministic detection, invokes AI root cause analysis, and suggests
  fixes. Use when a kernel hangs, never returns, or has a suspected deadlock.
---

# Debug Hanging GPU Kernel

Automated deadlock diagnosis for GPU kernels that hang or never return.

## Step 1: Gather Information

Collect from the user if not provided:

1. **Test command** — the command that reproduces the hang (e.g., `python my_test.py`)
2. **Kernel name** — for `--kernel-filters`. If unknown, omit the flag and
   identify the kernel from trace output later
3. **GPU architecture** — auto-detect:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

Map: H100 → `h100`, B200 → `b200a`, B300 → `sm_110a`.
If detection fails, ask the user.

## Step 2: Capture Instruction-Level Trace

```bash
mkdir -p /tmp/cutracer_debug/deadlock

buck2 run //triton/tools/CUTracer:cutracer -c fbcode.nvcc_arch=<arch> -- trace \
    --instrument reg_trace \
    --trace-format ndjson \
    --channel-records 1 \
    --kernel-filters <kernel_name> \
    --output-dir /tmp/cutracer_debug/deadlock \
    -- <test_command>
```

### Decision point after trace completes

- **Process killed by timeout** (15s default `--no-data-timeout-s`) →
  Kernel is hanging. Proceed to Step 3.
- **Process exited normally** →
  Not a hang. Inform user and suggest `/debug-data-race` instead.
- **No .ndjson files in output directory** →
  Kernel never launched. Check `--kernel-filters` spelling or remove the flag.
- **Buck build failure** →
  Verify `nvcc_arch` matches the actual GPU.

## Step 3: Deadlock Detection + AI Analysis

This step can take up to 30 minutes for complex kernels. Run in background:

```bash
buck2 run //triton/tools/CUTracer:cutracer -- analyze deadlock \
    /tmp/cutracer_debug/deadlock/*.ndjson \
    --ai -o /tmp/cutracer_debug/deadlock/report.md
```

If `claude` CLI is not available, fall back to Phase 1 only (deterministic
detection without AI root cause analysis):

```bash
buck2 run //triton/tools/CUTracer:cutracer -- analyze deadlock \
    /tmp/cutracer_debug/deadlock/*.ndjson
```

## Step 4: Summarize Report

Read the generated report and present to the user:

1. **Hang confirmation** — how many warps are BARRIER / LOOPING / UNKNOWN
2. **Root cause** — which deadlock pattern was identified:
   - Barrier pairing mismatch
   - Circular wait
   - Divergent execution + barrier
   - Missing barrier wait
   - Scheduler-induced deadlock
   - Warp routing asymmetry
3. **Key evidence** — SASS instruction addresses and source code locations
4. **Determinism** — is this a deterministic or probabilistic deadlock

## Step 5: Suggest Fix

Based on the identified root cause pattern:

| Root Cause | Fix Strategy |
|-----------|-------------|
| Divergent execution + barrier | Add `convergent` attribute to barrier inline asm, or replace with NVVM intrinsics (ref: triton#1040) |
| Barrier pairing mismatch | Verify barrier ID and thread count match across all warp groups |
| Circular wait | Reorder barrier operations to break the cycle |
| Missing barrier wait | Add the missing barrier on the code path that skips it |
| Scheduler-induced | Likely a compiler bug — file issue against Triton compiler |
| Warp routing asymmetry | Check `async_tasks` warp group configuration |

If the fix involves a Triton compiler change, help the user locate the
relevant compiler pass and suggest a patch.

## Troubleshooting

| Error | Solution |
|-------|---------|
| `CUDA_INJECTION64_PATH is set` | Run `unset CUDA_INJECTION64_PATH` before retrying |
| Build fails with arch error | Re-detect GPU via `nvidia-smi` and correct the flag |
| No .ndjson files produced | Remove `--kernel-filters` to trace all kernels, then identify the target |
| Phase 2 timeout after 30min | Use `--no-ai` for Phase 1 report only, then inspect manually |
| `claude: command not found` | Install claude CLI, or use `--no-ai` for Phase 1 only |
