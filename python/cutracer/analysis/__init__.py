# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer analysis module.

Provides analysis utilities for trace files:
- TraceReader: Read and iterate over trace records
- parse_filter_expr: Parse filter expressions for record filtering
- select_records: Memory-efficient record selection
"""

from .reader import parse_filter_expr, select_records, TraceReader

__all__ = ["TraceReader", "parse_filter_expr", "select_records"]
