# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid == 0:
        while pid == pid:
            tl.atomic_add(a_ptr, 1)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


def tensor_add(a, b):
    n_elements = a.numel()
    c = torch.empty_like(a)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
    return c


def test_tensor_add():
    torch.manual_seed(0)
    size = (1024, 1024)
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)

    # Test Triton kernel
    c_triton = tensor_add(a, b)

    # Verify the result by comparing with PyTorch's native addition
    c_expected = a + b

    # Assert that the results are close (allowing for floating point precision)
    assert torch.allclose(
        c_triton, c_expected, rtol=1e-5, atol=1e-5
    ), "Triton kernel result doesn't match expected result"

    # Test multiple calls to ensure consistency
    c_triton2 = tensor_add(a, b)
    assert torch.allclose(
        c_triton, c_triton2, rtol=1e-5, atol=1e-5
    ), "Multiple calls to tensor_add should produce consistent results"
