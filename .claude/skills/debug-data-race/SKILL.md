---
description: >-
  End-to-end data race detection for GPU kernels. Uses random delay injection
  to discover races, deterministic replay to confirm, and delta debugging to
  minimize the trigger. Use when a kernel produces non-deterministic results,
  numeric errors, or suspected data races.
---

# Debug GPU Kernel Data Race

Automated data race detection using random delay injection, deterministic
replay, and delta debugging minimization.

## Step 1: Gather Information

Collect from the user if not provided:

1. **Test command** — must exit 0 when the race manifests, non-zero otherwise.
   If the user's test doesn't follow this convention, help write a wrapper:
   ```bash
   #!/bin/bash
   output=$(<test_command> 2>&1)
   echo "$output" | grep -q "<error_pattern>" && exit 0 || exit 1
   ```
2. **Kernel name** — for `--kernel-filters`
3. **GPU architecture** — auto-detect via:
   ```bash
   nvidia-smi --query-gpu=name --format=csv,noheader | head -1
   ```
   Map: H100 → `h100`, B200 → `b200a`, B300 → `sm_110a`
4. **Error pattern** — how to tell if the race manifested (wrong output,
   assertion failure, specific error message, etc.)

## Step 2: Discover — Random Delay Injection

Run the test with delay injection. Try up to 5 attempts per delay level:

```bash
mkdir -p /tmp/cutracer_debug/race

buck2 run //triton/tools/CUTracer:cutracer -c fbcode.nvcc_arch=<arch> -- trace \
    --instrument random_delay \
    --analysis random_delay \
    --delay-ns 1000 \
    --delay-mode random \
    --delay-dump-path /tmp/cutracer_debug/race/config_attempt_<N>.json \
    --kernel-filters <kernel_name> \
    --output-dir /tmp/cutracer_debug/race \
    -- <test_command>
```

### Decision logic

Check the test command exit code after each run:

- **Exit 0** (race triggered) → save the working config path, proceed to Step 3
- **Exit non-zero after 5 attempts** → escalate `--delay-ns`:
  1000 → 5000 → 10000 → 50000 → 100000.
  Try up to 5 runs at each level.
- **All levels exhausted, no race** → inform the user:
  - The race may not be timing-sensitive
  - Try adding `--delay-min-ns 100` to ensure a minimum delay floor
  - Verify the test script exit code convention is correct
- **Process killed by timeout** → kernel is hanging, not a race.
  Suggest `/debug-hanging-kernel` instead.

## Step 3: Replay — Confirm Reproducibility

Replay the triggering config 3 times to assess determinism:

```bash
buck2 run //triton/tools/CUTracer:cutracer -c fbcode.nvcc_arch=<arch> -- trace \
    --instrument random_delay \
    --analysis random_delay \
    --delay-load-path /tmp/cutracer_debug/race/<working_config>.json \
    --kernel-filters <kernel_name> \
    --output-dir /tmp/cutracer_debug/race \
    -- <test_command>
```

### Decision logic

- **3/3 exit 0** → deterministic reproduction. Use `--confidence-runs 1`
  in Step 4.
- **2/3 or 1/3 exit 0** → probabilistic race. Use `--confidence-runs 3`
  in Step 4 for majority voting.
- **0/3 exit 0** → reproduction failed. Go back to Step 2 and try a
  different config, or re-run discovery.

## Step 4: Reduce — Delta Debugging

First, generate a test script for the reduce command. The reduce command
sets `CUTRACER_DELAY_LOAD_PATH` automatically — do NOT hardcode it:

```bash
cat > /tmp/cutracer_debug/race/test_race.sh << 'SCRIPT'
#!/bin/bash
buck2 run //triton/tools/CUTracer:cutracer -c fbcode.nvcc_arch=<arch> -- trace \
    --instrument random_delay \
    --analysis random_delay \
    --kernel-filters <kernel_name> \
    -- <test_command>
SCRIPT
chmod +x /tmp/cutracer_debug/race/test_race.sh
```

Then run delta debugging (this can take a long time — run in background):

```bash
buck2 run //triton/tools/CUTracer:cutracer -- reduce \
    --config /tmp/cutracer_debug/race/<working_config>.json \
    --test /tmp/cutracer_debug/race/test_race.sh \
    --strategy bisect \
    --confidence-runs <1_or_3> \
    --output /tmp/cutracer_debug/race/reduce_report.json \
    --minimal-config /tmp/cutracer_debug/race/minimal_config.json
```

## Step 5: Analyze Results

Read `reduce_report.json` and `minimal_config.json`. Present to the user:

1. **Reduction summary** — N total delay points reduced to M essential points
2. **Essential delay points** — for each one:
   - PC offset (SASS instruction address)
   - SASS instruction mnemonic
   - Delay amount in nanoseconds
3. **Source mapping** — if cubin files were dumped, map PC offsets to
   Triton source locations using nvdisasm
4. **Race type** — infer from SASS instructions at essential delay points:
   - Barrier instructions (BAR.SYNC, WARPSYNC) → synchronization race
   - Shared memory (LDS, STS) → shared memory race
   - Global memory (LDG, STG) → global memory race
   - TMA instructions → TMA descriptor race

## Step 6: Suggest Fix

| Race Type | Fix Strategy |
|----------|-------------|
| Shared memory RAW | Add missing barrier between the write and read |
| TMA descriptor race | Ensure TMA fence before descriptor consumption |
| Named barrier count mismatch | Verify thread count in `named_barrier_wait` |
| Missing warpgroup sync | Add `warpgroup_arrive` + `warpgroup_wait` pair |

## Troubleshooting

| Error | Solution |
|-------|---------|
| Race never triggers | Increase `--delay-ns`; try `--delay-min-ns 100` for a delay floor |
| Reduce takes too long | Use `--strategy linear` for simpler cases |
| Test script always exits 0 | The error pattern matches normal output — fix the grep |
| Test script always exits 1 | Verify the race exists without CUTracer first |
| Replay not deterministic | Use higher `--confidence-runs` (3 or 5) in reduce |
| Process hangs during discover | Kernel deadlock, not a race — use `/debug-hanging-kernel` |
