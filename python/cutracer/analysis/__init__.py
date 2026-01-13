# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer analysis module.

Provides analysis utilities for trace files:
- TraceReader: Read and iterate over trace records
"""

from .reader import TraceReader

__all__ = ["TraceReader"]
