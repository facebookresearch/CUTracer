#!/bin/bash
# =============================================================================
# cutracer_reduce_test.sh - Demo test script for cutracer reduce
# =============================================================================
# Exit 0 = Race detected (interesting)
# Exit 1 = No race (not interesting)
#
# IMPORTANT: The python reducer sets CUTRACER_DELAY_LOAD_PATH env var to pass
# the modified config path. Do NOT set CUTRACER_DELAY_LOAD_PATH in this script
# as it will be overwritten by the reducer.
#
# Example:
#   cutracer reduce --config cutracer_config.json \
#     --test ./tests/datarace_test/cutracer_reduce_test.sh -v
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTRACER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG="/tmp/cutracer_reduce_test.log"

# Run test with delay config replay mode
# Note: CUTRACER_DELAY_LOAD_PATH is set by the python reducer
CUTRACER_ANALYSIS=random_delay \
CUTRACER_DELAY_NS=10000 \
CUTRACER_KERNEL_FILTERS=matmul_kernel_bug1_late_barrier_a \
CUDA_INJECTION64_PATH="$CUTRACER_ROOT/lib/cutracer.so" \
python3 "$SCRIPT_DIR/hopper-gemm-ws_data_race_test.py" --bug 1 --iters 10 > "$LOG" 2>&1

# Parse failures from output (matches "Bug 1:", "Bug 2:", etc.)
FAILURES=$(grep -A1 "Bug.*:" "$LOG" | grep -oP 'Result: \K\d+(?=/)' || echo 0)
echo "Failures: $FAILURES/10"

# Race detected if >= 30% failure rate (3 out of 10 iterations)
if [[ "$FAILURES" -ge 3 ]]; then
    echo "✓ Race detected"
    exit 0
else
    echo "✗ No race"
    exit 1
fi
