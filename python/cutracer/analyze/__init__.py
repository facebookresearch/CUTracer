# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer analyze module.

Provides analysis algorithms that produce derived insights:
- warp-summary: Warp execution status analysis (completed, in-progress, missing)
- tma-decoder: TMA descriptor parsing and decoding
- (future: hot-instructions, data-race, memory-pattern, divergence, ...)
"""

from cutracer.analyze.fb.sass_decoder import (
    DATA_TYPE_TO_BYTES,
    DATA_TYPE_TO_STR,
    decode_tma_descriptor,
    INTERLEAVE_TO_STR,
    SWIZZLE_MODE_TO_STR,
    TMADescriptor,
)
from cutracer.query.warp_summary import (
    compute_warp_summary,
    format_ranges,
    format_warp_summary_text,
    is_exit_instruction,
    merge_to_ranges,
    warp_summary_to_dict,
    WarpSummary,
)

__all__ = [
    # Warp summary
    "WarpSummary",
    "is_exit_instruction",
    "merge_to_ranges",
    "format_ranges",
    "compute_warp_summary",
    "format_warp_summary_text",
    "warp_summary_to_dict",
    # TMA decoder
    "DATA_TYPE_TO_STR",
    "DATA_TYPE_TO_BYTES",
    "SWIZZLE_MODE_TO_STR",
    "INTERLEAVE_TO_STR",
    "TMADescriptor",
    "decode_tma_descriptor",
]
