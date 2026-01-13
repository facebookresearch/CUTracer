# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer analysis module.

Provides analysis utilities for trace files:
- TraceReader: Read and iterate over trace records
- parse_filter_expr: Parse filter expressions for record filtering
- select_records: Memory-efficient record selection
- StreamingGrouper: Memory-efficient grouped analysis
- Formatters: Output formatting for table/json/csv
- WarpSummary: Warp execution status summary for GPU hang analysis
"""

from .formatters import (
    DEFAULT_FIELDS,
    format_records_csv,
    format_records_json,
    format_records_table,
    format_value,
    get_display_fields,
)
from .grouper import StreamingGrouper
from .reader import parse_filter_expr, select_records, TraceReader
from .warp_summary import (
    compute_warp_summary,
    format_ranges,
    format_warp_summary_text,
    is_exit_instruction,
    merge_to_ranges,
    warp_summary_to_dict,
    WarpSummary,
)

__all__ = [
    "TraceReader",
    "parse_filter_expr",
    "select_records",
    # Grouper
    "StreamingGrouper",
    # Formatters
    "DEFAULT_FIELDS",
    "format_value",
    "get_display_fields",
    "format_records_table",
    "format_records_json",
    "format_records_csv",
    # Warp Summary
    "WarpSummary",
    "is_exit_instruction",
    "merge_to_ranges",
    "format_ranges",
    "compute_warp_summary",
    "format_warp_summary_text",
    "warp_summary_to_dict",
]
