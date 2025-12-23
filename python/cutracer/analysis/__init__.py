# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer analysis module.

This module provides analysis functions for trace files:
- TraceReader: Read and iterate over trace records
- parse_filter_expr: Parse filter expressions for record filtering
- StreamingGrouper: Memory-efficient grouping of trace records
"""

from .grouper import StreamingGrouper
from .reader import parse_filter_expr, TraceReader

__all__ = [
    "parse_filter_expr",
    "StreamingGrouper",
    "TraceReader",
]
