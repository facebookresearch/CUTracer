# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Common Analysis Context

"""
Common analysis context shared across analyze submodules.

Provides AnalysisContext and build_analysis_context() to unify trace reading,
kernel metadata parsing, and source mapping across data_race, dataflow,
and deadlock analysis modules.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from cutracer.kernel_config import KernelConfig, parse_kernel_metadata
from cutracer.query.reader import TraceReader
from cutracer.types import TraceRecord


@dataclass
class AnalysisContext:
    """Common context for trace analysis.

    Encapsulates data that multiple analysis modules need:
    kernel metadata, trace records, PC-to-source mapping, and SASS text.
    """

    kernel_config: Optional[KernelConfig]
    trace_records: list[TraceRecord]
    pc_mappings: dict[int, dict[str, Any]]
    sass_text: Optional[str]
    trace_path: Path


def build_analysis_context(
    trace_path: Path,
    include_sass: bool = False,
) -> AnalysisContext:
    """Build common analysis context from a trace file.

    Steps:
    1. Read all trace records
    2. Extract kernel_metadata (if present as first record)
    3. Build PC -> source mapping (if cubin available)
    4. Optionally extract full SASS text via nvdisasm

    Args:
        trace_path: Path to .ndjson or .ndjson.zst trace file.
        include_sass: If True and cubin is available, extract full SASS text.

    Returns:
        AnalysisContext with all common analysis data.
    """
    reader = TraceReader(trace_path)
    all_records = list(reader.iter_records())

    kernel_config = None
    trace_records = all_records
    if all_records and all_records[0].get("type") == "kernel_metadata":
        kernel_config = parse_kernel_metadata(all_records[0])
        trace_records = all_records[1:]

    pc_mappings: dict[int, dict[str, Any]] = {}
    sass_text: Optional[str] = None

    if kernel_config and kernel_config.cubin_path:
        from cutracer.analyze.fb.source_mapping import (
            build_pc_source_mapping,
            get_sass_with_source_info,
        )

        pc_mappings = build_pc_source_mapping(
            kernel_config.cubin_path, trace_path.parent
        )
        if include_sass:
            cubin_full_path = trace_path.parent / kernel_config.cubin_path
            sass_text = get_sass_with_source_info(cubin_full_path)

    return AnalysisContext(
        kernel_config=kernel_config,
        trace_records=trace_records,
        pc_mappings=pc_mappings,
        sass_text=sass_text,
        trace_path=trace_path,
    )
