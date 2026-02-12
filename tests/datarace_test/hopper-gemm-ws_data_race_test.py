"""
Data Race Test for Hopper GEMM Warp-Specialized Kernel

This file contains intentionally buggy versions of the warp-specialized GEMM kernel
to test data race detectors. Each bug is designed to:
1. Have high probability of passing in normal execution
2. Be triggered by random delay injection

The correct pattern is:
    barrier_wait(full_a)       # Wait for TMA to complete
    barrier_wait(full_b)       # Wait for TMA to complete
    async_dot(data_a, data_b)  # MMA uses ready data

Bug Pattern 1: Late Barrier Wait (Moved After MMA Load)
=======================================================
The barrier_wait for A is MOVED to AFTER async_dot:
    spin_loop()                # Spin to give TMA time (usually enough)
    barrier_wait(full_b)       # Only wait for B
    async_dot(data_a, data_b)  # DATA RACE: A may not be ready yet!
    barrier_wait(full_a)       # TOO LATE: MMA already used the data!

Bug Pattern 2: Missing Barrier Wait (Spin-Wait Only)
====================================================
The barrier_wait for B is REMOVED entirely:
    barrier_wait(full_a)       # Only wait for A
    spin_loop()                # Spin to give TMA time for B (usually enough)
    async_dot(data_a, data_b)  # DATA RACE: B may not be ready yet!
                               # No barrier_wait for B at all!

Why they usually pass:
- TMA loads are very fast (~100-200 cycles for 64KB)
- The spin loop provides actual delay cycles (clock64 cannot be optimized away)
- This is usually MORE than enough time for TMA to complete

Why random delay injection triggers the bugs:
- Delays slow down TMA initiation or data arrival
- The spin loop becomes insufficient
- async_dot reads partially loaded or stale data
"""

from typing import Optional

import click
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_cuda
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Use smaller matrices for faster testing
# CRITICAL: Smaller matrices = fewer loop iterations = less chance for race to manifest
# This makes the race condition timing-dependent rather than guaranteed
M, N, K = (512, 512, 512)  # Small size: only 8 K-loop iterations with BK=64


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]


# ==============================================================================
# CORRECT REFERENCE KERNEL (for comparison)
# ==============================================================================
@triton.jit
def matmul_kernel_correct(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    """Correct kernel - should always pass."""
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc(
        (BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS
    )
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    bars_full_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    bars_empty_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS
    )
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 1
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES
                offset_k = k * BK

                empty_a_1st = tlx.local_view(bars_empty_a, buf)
                full_a_1st = tlx.local_view(bars_full_a, buf)
                tlx.barrier_wait(bar=empty_a_1st, phase=p)
                tlx.barrier_expect_bytes(
                    full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc))
                )
                data_a_1st = tlx.local_view(a, buf)
                tlx.async_descriptor_load(
                    a_desc, data_a_1st, [offset_am, offset_k], full_a_1st
                )

                empty_b = tlx.local_view(bars_empty_b, buf)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=empty_b, phase=p)
                tlx.barrier_expect_bytes(
                    full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc))
                )
                data_b = tlx.local_view(b, buf)
                tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                tlx.barrier_expect_bytes(
                    bar=full_a_2nd,
                    size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)),
                )
                data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_2nd,
                    [offset_am + BLOCK_M_SPLIT, offset_k],
                    full_a_2nd,
                )

                p = p ^ (buf == (NUM_STAGES - 1))

        with tlx.async_task(num_warps=4, replicate=2):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 0
            acc = tl.zeros([BM // 2, BN], dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES

                full_a = tlx.local_view(
                    bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=full_a, phase=p)
                tlx.barrier_wait(bar=full_b, phase=p)  # CORRECT: wait for B

                data_a = tlx.local_view(
                    a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                data_b = tlx.local_view(b, buf)

                acc = tlx.async_dot(data_a, data_b, acc)
                acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                empty_a = tlx.local_view(
                    bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                empty_b = tlx.local_view(bars_empty_b, buf)
                tlx.barrier_arrive(empty_a)
                tlx.barrier_arrive(empty_b)

                p = p ^ (buf == (NUM_STAGES - 1))

            offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()

            if EPILOGUE_SUBTILE:
                acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn], c0)
                c1 = acc1.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn + BN // 2], c1)
            else:
                c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))


# ==============================================================================
# BUG 1: Late barrier_wait for A (moved after MMA load)
# ==============================================================================
# The barrier_wait for matrix A is MOVED to AFTER the async_dot call.
# This creates a data race where MMA loads potentially unready data.
#
# Correct pattern:
#     barrier_wait(full_a)       # Wait for TMA to complete
#     barrier_wait(full_b)
#     async_dot(data_a, data_b)  # MMA uses ready data
#
# Buggy pattern (barrier_wait moved after async_dot):
#     spin_loop()                # Spin to give TMA time (usually enough)
#     barrier_wait(full_b)       # Only wait for B
#     async_dot(data_a, data_b)  # DATA RACE: A may not be ready yet!
#     barrier_wait(full_a)       # TOO LATE: MMA already used the data!
#
# The spin loop uses tlx.clock64() which:
#   1. Cannot be optimized away by the compiler (hardware register read)
#   2. Actually burns GPU cycles
#   3. Provides "usually enough" delay for TMA to complete
#
# Why it usually passes without delays:
#   - TMA loads are extremely fast (~100-200 cycles for 64KB)
#   - The spin loop provides actual delay cycles
#   - This is usually MORE than enough time
#
# Why random delay injection triggers the bug:
#   - Delays slow down TMA initiation or data arrival
#   - The spin loop becomes insufficient
#   - async_dot reads partially loaded or stale data
#   - barrier_wait after async_dot doesn't help - damage is already done
# ==============================================================================
@triton.jit
def matmul_kernel_bug1_late_barrier_a(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    SPIN_COUNT: tl.constexpr,  # Number of clock cycles to spin
):
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc(
        (BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS
    )
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    bars_full_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    bars_empty_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS
    )
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 1
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES
                offset_k = k * BK

                # Load A for consumer 0
                empty_a_1st = tlx.local_view(bars_empty_a, buf)
                full_a_1st = tlx.local_view(bars_full_a, buf)
                tlx.barrier_wait(bar=empty_a_1st, phase=p)
                tlx.barrier_expect_bytes(
                    full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc))
                )
                data_a_1st = tlx.local_view(a, buf)
                tlx.async_descriptor_load(
                    a_desc, data_a_1st, [offset_am, offset_k], full_a_1st
                )

                # Load B
                empty_b = tlx.local_view(bars_empty_b, buf)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=empty_b, phase=p)
                tlx.barrier_expect_bytes(
                    full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc))
                )
                data_b = tlx.local_view(b, buf)
                tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                # Load A for consumer 1
                empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                tlx.barrier_expect_bytes(
                    bar=full_a_2nd,
                    size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)),
                )
                data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_2nd,
                    [offset_am + BLOCK_M_SPLIT, offset_k],
                    full_a_2nd,
                )

                p = p ^ (buf == (NUM_STAGES - 1))

        with tlx.async_task(num_warps=4, replicate=2):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 0
            acc = tl.zeros([BM // 2, BN], dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES

                full_a = tlx.local_view(
                    bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                full_b = tlx.local_view(bars_full_b, buf)

                # BUG: barrier_wait for A is MOVED to AFTER async_dot!
                # We use spin-wait to give TMA time, but MMA reads A before
                # we actually confirm it's ready.
                #
                # CORRECT pattern:
                #   tlx.barrier_wait(bar=full_a, phase=p)  # Wait for A
                #   tlx.barrier_wait(bar=full_b, phase=p)  # Wait for B
                #   async_dot(data_a, data_b, acc)         # MMA uses ready data
                #
                # BUG pattern (barrier moved after async_dot):
                #   spin_loop()                            # Give TMA time (usually enough)
                #   barrier_wait(bar=full_b, phase=p)      # Only wait for B
                #   async_dot(data_a, data_b, acc)         # DATA RACE: A may not be ready!
                #   barrier_wait(bar=full_a, phase=p)      # TOO LATE: damage done
                #
                # Spin for a fixed number of cycles using clock64():
                start_clock = tlx.clock64()
                while tlx.clock64() - start_clock < SPIN_COUNT:
                    pass  # Spin until enough cycles have passed
                # The spin is usually enough, but with delays it won't be!

                tlx.barrier_wait(
                    bar=full_b, phase=p
                )  # B barrier is correct (before MMA)

                data_a = tlx.local_view(
                    a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                data_b = tlx.local_view(b, buf)

                acc = tlx.async_dot(
                    data_a, data_b, acc
                )  # DATA RACE: A may not be ready!
                tlx.barrier_wait(
                    bar=full_a, phase=p
                )  # TOO LATE: MMA already used the data!

                acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                empty_a = tlx.local_view(
                    bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                empty_b = tlx.local_view(bars_empty_b, buf)
                tlx.barrier_arrive(empty_a)
                tlx.barrier_arrive(empty_b)

                p = p ^ (buf == (NUM_STAGES - 1))

            offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()

            if EPILOGUE_SUBTILE:
                acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn], c0)
                c1 = acc1.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn + BN // 2], c1)
            else:
                c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))


# ==============================================================================
# BUG 2: Missing barrier_wait for B (spin-wait only, no barrier)
# ==============================================================================
# The barrier_wait for matrix B is REMOVED entirely.
# We use only a spin loop instead of proper synchronization.
#
# Correct pattern:
#     barrier_wait(full_a)       # Wait for TMA to complete
#     barrier_wait(full_b)       # Wait for TMA to complete
#     async_dot(data_a, data_b)  # MMA uses ready data
#
# Buggy pattern (barrier_wait removed, spin only):
#     barrier_wait(full_a)       # Only wait for A
#     spin_loop()                # Spin to give TMA time for B (usually enough)
#     async_dot(data_a, data_b)  # DATA RACE: B may not be ready yet!
#                                # No barrier_wait for B at all!
#
# The spin loop uses tlx.clock64() which:
#   1. Cannot be optimized away by the compiler (hardware register read)
#   2. Actually burns GPU cycles
#   3. Provides "usually enough" delay for TMA to complete
#
# Why it usually passes without delays:
#   - TMA loads are extremely fast (~100-200 cycles for 64KB)
#   - The spin loop provides actual delay cycles
#   - This is usually MORE than enough time
#
# Why random delay injection triggers the bug:
#   - Delays slow down TMA initiation or data arrival
#   - The spin loop becomes insufficient
#   - async_dot reads partially loaded or stale data
# ==============================================================================
@triton.jit
def matmul_kernel_bug2_missing_barrier_b(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    SPIN_COUNT: tl.constexpr,  # Number of clock cycles to spin
):
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc(
        (BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS
    )
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    bars_full_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    bars_empty_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS
    )
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 1
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES
                offset_k = k * BK

                empty_a_1st = tlx.local_view(bars_empty_a, buf)
                full_a_1st = tlx.local_view(bars_full_a, buf)
                tlx.barrier_wait(bar=empty_a_1st, phase=p)
                tlx.barrier_expect_bytes(
                    full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc))
                )
                data_a_1st = tlx.local_view(a, buf)
                tlx.async_descriptor_load(
                    a_desc, data_a_1st, [offset_am, offset_k], full_a_1st
                )

                empty_b = tlx.local_view(bars_empty_b, buf)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=empty_b, phase=p)
                tlx.barrier_expect_bytes(
                    full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc))
                )
                data_b = tlx.local_view(b, buf)
                tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                tlx.barrier_expect_bytes(
                    bar=full_a_2nd,
                    size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)),
                )
                data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_2nd,
                    [offset_am + BLOCK_M_SPLIT, offset_k],
                    full_a_2nd,
                )

                p = p ^ (buf == (NUM_STAGES - 1))

        with tlx.async_task(num_warps=4, replicate=2):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 0
            acc = tl.zeros([BM // 2, BN], dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES

                full_a = tlx.local_view(
                    bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(
                    bar=full_a, phase=p
                )  # A barrier is correct (before MMA)

                # BUG: barrier_wait for B is REMOVED entirely!
                # We use only a spin loop instead of proper synchronization.
                #
                # CORRECT pattern:
                #   tlx.barrier_wait(bar=full_b, phase=p)  # Wait for B
                #   async_dot(data_a, data_b, acc)         # MMA uses ready data
                #
                # BUG pattern (barrier_wait removed, spin only):
                #   spin_loop()                            # Spin to give TMA time (usually enough)
                #   async_dot(data_a, data_b, acc)         # DATA RACE: B may not be ready!
                #                                          # No barrier_wait for B at all!
                #
                # Spin for a fixed number of cycles using clock64():
                start_clock = tlx.clock64()
                while tlx.clock64() - start_clock < SPIN_COUNT:
                    pass  # Spin until enough cycles have passed
                # The spin is usually enough, but with delays it won't be!

                data_a = tlx.local_view(
                    a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                data_b = tlx.local_view(b, buf)

                acc = tlx.async_dot(
                    data_a, data_b, acc
                )  # DATA RACE: B may not be ready!
                # NO barrier_wait for B here - completely removed!
                acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                empty_a = tlx.local_view(
                    bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()
                )
                empty_b = tlx.local_view(bars_empty_b, buf)
                tlx.barrier_arrive(empty_a)
                tlx.barrier_arrive(empty_b)

                p = p ^ (buf == (NUM_STAGES - 1))

            offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()

            if EPILOGUE_SUBTILE:
                acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn], c0)
                c1 = acc1.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn + BN // 2], c1)
            else:
                c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))


# ==============================================================================
# Test Harness
# ==============================================================================


def run_matmul_with_kernel(
    kernel, a, b, check_correctness=True, spin_count_a=1200, spin_count_b=200
):
    """Run matrix multiplication with a specific kernel."""
    M_dim, K_dim = a.shape
    K_dim2, N_dim = b.shape
    assert K_dim == K_dim2, "Dimension mismatch"

    c = torch.zeros((M_dim, N_dim), dtype=torch.float16, device=DEVICE)

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(
        a,
        shape=[M_dim, K_dim],
        strides=[K_dim, 1],
        block_shape=dummy_block,
    )
    desc_in_2 = TensorDescriptor(
        b,
        shape=[K_dim, N_dim],
        strides=[N_dim, 1],
        block_shape=dummy_block,
    )
    desc_out = TensorDescriptor(
        c,
        shape=[M_dim, N_dim],
        strides=[N_dim, 1],
        block_shape=dummy_block,
    )

    # Fixed config for testing
    BM, BN, BK = 128, 256, 64
    GROUP_SIZE_M = 8
    NUM_STAGES = 4
    NUM_MMA_WARPS = 8
    NUM_MMA_GROUPS = 2
    EPILOGUE_SUBTILE = True

    # Set block shapes
    BLOCK_M_SPLIT = BM // NUM_MMA_GROUPS
    desc_in_1.block_shape = [BLOCK_M_SPLIT, BK]
    desc_in_2.block_shape = [BK, BN]
    desc_out.block_shape = (
        [BLOCK_M_SPLIT, BN // 2] if EPILOGUE_SUBTILE else [BLOCK_M_SPLIT, BN]
    )

    grid = (triton.cdiv(M_dim, BM) * triton.cdiv(N_dim, BN),)

    # Check if this kernel needs SPIN_COUNT parameter
    if kernel == matmul_kernel_bug1_late_barrier_a:
        kernel[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M_dim,
            N_dim,
            K_dim,
            BM=BM,
            BN=BN,
            BK=BK,
            GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_STAGES=NUM_STAGES,
            NUM_MMA_WARPS=NUM_MMA_WARPS,
            NUM_MMA_GROUPS=NUM_MMA_GROUPS,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            SPIN_COUNT=spin_count_a,  # Spin cycles before MMA (for A timing)
            num_stages=1,
            num_warps=4,
        )
    elif kernel == matmul_kernel_bug2_missing_barrier_b:
        kernel[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M_dim,
            N_dim,
            K_dim,
            BM=BM,
            BN=BN,
            BK=BK,
            GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_STAGES=NUM_STAGES,
            NUM_MMA_WARPS=NUM_MMA_WARPS,
            NUM_MMA_GROUPS=NUM_MMA_GROUPS,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            SPIN_COUNT=spin_count_b,  # Spin cycles before MMA (for B timing)
            num_stages=1,
            num_warps=4,
        )
    else:
        kernel[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M_dim,
            N_dim,
            K_dim,
            BM=BM,
            BN=BN,
            BK=BK,
            GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_STAGES=NUM_STAGES,
            NUM_MMA_WARPS=NUM_MMA_WARPS,
            NUM_MMA_GROUPS=NUM_MMA_GROUPS,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            num_stages=1,
            num_warps=4,
        )

    if check_correctness:
        output_ref = torch.matmul(a, b)
        if not torch.allclose(c, output_ref, atol=1e-2, rtol=1e-2):
            max_diff = (c - output_ref).abs().max().item()
            mean_diff = (c - output_ref).abs().mean().item()
            print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
            return False, c, output_ref
        return True, c, output_ref

    return True, c, None


def run_multiple_iterations(kernel, num_iterations=10, matrix_size=None):
    """Run kernel multiple times to increase chance of triggering race.

    Returns a dict with test results for summary report.
    """
    if matrix_size is None:
        matrix_size = (M, N, K)
    m, n, k = matrix_size

    triton.set_allocator(alloc_fn)

    results = {
        "failures": 0,
        "total": num_iterations,
        "failed_iterations": [],
        "exception_iterations": [],
    }

    for i in range(num_iterations):
        # Use different random seed each iteration for variety
        torch.manual_seed(i * 42)
        a = torch.randn((m, k), dtype=torch.float16, device=DEVICE)
        b = torch.randn((k, n), dtype=torch.float16, device=DEVICE)

        try:
            correct, output, ref = run_matmul_with_kernel(kernel, a, b)
            if not correct:
                results["failures"] += 1
                max_diff = (output - ref).abs().max().item()
                results["failed_iterations"].append((i, max_diff))
        except Exception as e:
            results["failures"] += 1
            results["exception_iterations"].append((i, str(e)))

    return results


# ==============================================================================
# Pytest Tests
# ==============================================================================


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires Hopper GPU",
)
def test_bug1_late_barrier_a():
    """Test Bug 1: Late barrier_wait for A (moved after MMA load).

    The barrier_wait for matrix A is MOVED to AFTER the async_dot call.
    A spin loop provides "usually enough" delay for TMA to complete.
    async_dot reads potentially unready data - DATA RACE!
    barrier_wait after async_dot is TOO LATE - damage is already done.

    With random delays, this should fail more consistently.
    """
    results = run_multiple_iterations(
        matmul_kernel_bug1_late_barrier_a,
        num_iterations=100,
    )
    return results


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires Hopper GPU",
)
def test_bug2_missing_barrier_b():
    """Test Bug 2: Missing barrier_wait for B (spin-wait only, no barrier).

    The barrier_wait for matrix B is REMOVED entirely.
    A spin loop provides "usually enough" delay for TMA to complete.
    async_dot reads potentially unready data - DATA RACE!
    No barrier_wait for B at all - completely removed!

    With random delays, this should fail more consistently.
    """
    results = run_multiple_iterations(
        matmul_kernel_bug2_missing_barrier_b,
        num_iterations=100,
    )
    return results


def print_summary_report(all_results):
    """Print a summary report of all test results."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY REPORT")
    print("=" * 70)

    for test_name, results in all_results.items():
        failures = results["failures"]
        total = results["total"]
        status = "PASS" if failures == 0 else "FAIL"

        print(f"\n{test_name}:")
        print(f"  Result: {failures}/{total} failures [{status}]")

        if results["failed_iterations"]:
            print("  Failed iterations (with max diff):")
            for iter_num, max_diff in results["failed_iterations"]:
                print(f"    - Iteration {iter_num}: max_diff = {max_diff:.6f}")

        if results["exception_iterations"]:
            print("  Exception iterations:")
            for iter_num, exc in results["exception_iterations"]:
                print(f"    - Iteration {iter_num}: {exc}")

    print("\n" + "=" * 70)


# ==============================================================================
# Main: Run all tests with verbose output
# ==============================================================================


@click.command()
@click.option(
    "--iters",
    "-i",
    default=100,
    type=int,
    help="Number of iterations for all tests (default: 100)",
)
@click.option(
    "--bug",
    "-b",
    default="all",
    type=click.Choice(["all", "1", "2"]),
    help="Which bug test to run: 'all' (default), '1' (late barrier), or '2' (missing barrier)",
)
@click.option(
    "--tritonparse",
    "tritonparse_path",
    default=None,
    is_flag=False,
    flag_value="/tmp/tritonparse_logs/",
    help="Enable tritonparse structured logging. Optionally specify output path (default: /tmp/tritonparse_logs/)",
)
def main(iters, bug, tritonparse_path):
    if not is_cuda() or torch.cuda.get_device_capability()[0] != 9:
        print("Skipping: No Hopper GPU found")
        return

    if tritonparse_path:
        import tritonparse.structured_logging

        tritonparse.structured_logging.init(
            tritonparse_path,
            enable_trace_launch=True,
            enable_sass_dump=True,
        )

    print("=" * 70)
    print("Data Race Detection Test Suite for Hopper GEMM")
    print("=" * 70)
    print(f"Matrix size: {M} x {N} x {K}")
    print(f"Device: {DEVICE}")
    print(f"Bug filter: {bug}")
    print()
    print("These tests contain intentional bugs that cause data races.")
    print("Without random delay injection, they may pass due to timing luck.")
    print("With your random delay detector, they should fail consistently.")
    print()
    print("Running tests... (results will be shown at the end)")
    print("=" * 70)

    # Run tests and collect results
    all_results = {}
    if bug in ("all", "1"):
        all_results["Bug 1: Late barrier_wait for A"] = run_multiple_iterations(
            matmul_kernel_bug1_late_barrier_a,
            num_iterations=iters,
        )
    if bug in ("all", "2"):
        all_results["Bug 2: Missing barrier_wait for B"] = run_multiple_iterations(
            matmul_kernel_bug2_missing_barrier_b,
            num_iterations=iters,
        )

    # Print summary report at the end
    print_summary_report(all_results)

    print("Test suite complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
