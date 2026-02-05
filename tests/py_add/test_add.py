# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Test script for CUTracer trace format validation.

Uses a PT2 compiled kernel to ensure deterministic kernel generation.
This guarantees the same Triton kernel is traced across different runs,
enabling reliable cross-format validation (Mode 0/1/2 comparison).
"""

import torch


@torch.compile
def simple_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simple addition compiled to a single Triton kernel."""
    return a + b - 1.0


def test_compiled_add():
    """Test with PT2 compiled kernel for deterministic tracing."""
    device = torch.device("cuda")

    # Use fixed size for deterministic kernel generation
    # 1024 elements is small enough to be fast, large enough for meaningful traces
    a = torch.randn(1024, dtype=torch.float32, device=device)
    b = torch.randn(1024, dtype=torch.float32, device=device)

    print("Testing PT2 compiled kernel (simple_add)")
    print(f"  Input shape: {a.shape}")
    print(f"  Input dtype: {a.dtype}")
    compiled_function = torch.compile(simple_add)
    result = compiled_function(a, b)
    torch.cuda.synchronize()

    print(f"  Result shape: {result.shape}")
    print(f"  Result sum: {result.sum().item():.4f}")
    print("  ✅ PT2 compiled kernel executed successfully")

    return result


# Keep the old eager mode test for backward compatibility
def test_tensor_addition_on_gpu():
    """Legacy eager mode test (deprecated, use test_compiled_add instead)."""
    device = torch.device("cuda")

    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = torch.tensor([2, 3, 4], dtype=torch.float32, device=device)

    print("Tensor A:", a)
    print("Tensor B:", b)
    for _ in range(3):
        a = a + b

    print("Result (A + B):", a)

    return a


if __name__ == "__main__":
    # Use PT2 compiled kernel by default for deterministic tracing
    test_compiled_add()
