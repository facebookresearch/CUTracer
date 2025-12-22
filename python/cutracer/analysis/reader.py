# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Trace reader for CUTracer analysis.

This module provides the TraceReader class for reading and iterating
over trace records from NDJSON files (plain or Zstd-compressed).
"""

import json
from pathlib import Path
from typing import Iterator, Union

from cutracer.validation.compression import detect_compression, open_trace_file


class TraceReader:
    """
    Reader for CUTracer trace files.

    Supports NDJSON format with optional Zstd compression.
    Provides efficient iteration over trace records.

    Example:
        >>> reader = TraceReader("trace.ndjson.zst")
        >>> for record in reader.iter_records():
        ...     print(record["sass"])
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize the TraceReader.

        Args:
            file_path: Path to the trace file (.ndjson or .ndjson.zst)

        Raises:
            FileNotFoundError: If the file does not exist
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.compression = detect_compression(self.file_path)

    def iter_records(self) -> Iterator[dict]:
        """
        Iterate over all trace records in the file.

        Yields:
            dict: Each trace record as a dictionary

        Raises:
            json.JSONDecodeError: If a line contains invalid JSON

        Example:
            >>> reader = TraceReader("trace.ndjson")
            >>> for record in reader.iter_records():
            ...     process(record)
        """
        with open_trace_file(self.file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
