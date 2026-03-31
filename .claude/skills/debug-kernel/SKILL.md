---
description: >-
  Unified GPU kernel debugging entry point. Classifies the problem (hang vs
  data race) and routes to the appropriate specialized skill. Use when a user
  says "debug this kernel" without specifying the exact issue type.
---

# Debug GPU Kernel

Unified entry point that classifies the kernel problem and routes to the
appropriate debugging workflow.

## Step 1: Classify the Problem

Ask the user to describe the symptom:

| Symptom | Action |
|---------|--------|
| Kernel never returns / program hangs / stuck | Use `/debug-hanging-kernel` |
| Non-deterministic results / sometimes wrong / numeric mismatch | Use `/debug-data-race` |
| Not sure | Proceed to Step 2 for diagnostic |

If the user clearly describes one of the first two, skip Step 2 and route
directly to the appropriate skill.

## Step 2: Diagnostic Trace (only if symptom is unclear)

Collect the test command and kernel name from the user, auto-detect GPU
architecture via `nvidia-smi`, then run a lightweight trace without
instrumentation to observe kernel behavior:

```bash
buck2 run //triton/tools/CUTracer:cutracer -c fbcode.nvcc_arch=<arch> -- trace \
    --kernel-filters <kernel_name> \
    -- <test_command>
```

This adds zero instrumentation overhead — CUTracer only logs kernel launches
(names, grid/block dims, shared memory) without producing trace files.

### Decision logic

- **Process killed by no-data timeout (15s)** →
  Kernel is hanging. Route to `/debug-hanging-kernel` with the same
  test command and kernel name.
- **Process exits normally** → ask the user if the output is correct:
  - **Wrong results** → route to `/debug-data-race`
  - **Correct results** → no bug detected on this run. Suggest running
    multiple times to check for intermittent issues.
