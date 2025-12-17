# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Compression utilities for CUTracer trace files.

This module provides transparent handling of compressed trace files,
supporting Zstd compression (Mode 1) used by CUTracer.

CUTracer's Zstd format:
- Multiple independent Zstd frames appended together
- Each frame contains ~1MB of uncompressed NDJSON
- Compression level: 22 (maximum)
"""

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO, Union

import zstandard as zstd

# Zstd magic number (little-endian): 0xFD2FB528
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def detect_compression(filepath: Union[str, Path]) -> str:
    """
    Detect compression format of a file.

    Uses magic number detection for reliability (works regardless of extension).

    Args:
        filepath: Path to the file to check

    Returns:
        Compression type: "zstd" or "none"

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check magic number (works for all files regardless of extension)
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic == ZSTD_MAGIC:
            return "zstd"

    return "none"


def get_trace_format(filepath: Union[str, Path]) -> tuple[str, str]:
    """
    Determine the base format and compression of a trace file.

    Args:
        filepath: Path to the trace file

    Returns:
        Tuple of (base_format, compression) where:
            - base_format: "ndjson", "text", or "unknown"
            - compression: "zstd" or "none"

    Examples:
        >>> get_trace_format("trace.ndjson")
        ('ndjson', 'none')
        >>> get_trace_format("trace.ndjson.zst")
        ('ndjson', 'zstd')
        >>> get_trace_format("trace.log")
        ('text', 'none')
    """
    filepath = Path(filepath)
    compression = detect_compression(filepath)

    # Determine base format from extension
    suffixes = "".join(filepath.suffixes).lower()

    if ".ndjson" in suffixes:
        return ("ndjson", compression)
    elif suffixes.endswith(".log"):
        return ("text", compression)
    else:
        return ("unknown", compression)


@contextmanager
def open_trace_file(filepath: Union[str, Path]) -> Iterator[TextIO]:
    """
    Open a trace file, automatically handling compression.

    This is a context manager that returns a text stream for reading.
    Compression is detected automatically and handled transparently.

    Args:
        filepath: Path to the trace file

    Yields:
        Text stream for reading the file contents

    Raises:
        FileNotFoundError: If file does not exist

    Example:
        >>> with open_trace_file("trace.ndjson.zst") as f:
        ...     for line in f:
        ...         process(line)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    compression = detect_compression(filepath)

    if compression == "zstd":
        # Use zstandard's stream_reader which handles multiple frames
        dctx = zstd.ZstdDecompressor()
        with open(filepath, "rb") as binary_file:
            with dctx.stream_reader(binary_file) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                    yield text_stream
    else:
        # Plain text file
        with open(filepath, "r", encoding="utf-8") as f:
            yield f


def iter_lines(filepath: Union[str, Path]) -> Iterator[str]:
    """
    Iterate over lines in a trace file, handling compression transparently.

    This is a memory-efficient way to process large trace files line by line.

    Args:
        filepath: Path to the trace file

    Yields:
        Lines from the file (stripped of trailing newlines)

    Raises:
        FileNotFoundError: If file does not exist

    Example:
        >>> for line in iter_lines("trace.ndjson.zst"):
        ...     record = json.loads(line)
    """
    with open_trace_file(filepath) as f:
        for line in f:
            yield line.rstrip("\n\r")


def get_file_size(filepath: Union[str, Path], compressed: bool = True) -> int:
    """
    Get the size of a trace file.

    Args:
        filepath: Path to the trace file
        compressed: If True, return compressed size; if False, return
                   uncompressed size (requires reading the entire file
                   for compressed files)

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if compressed:
        return filepath.stat().st_size

    # For uncompressed size of compressed files, we need to read through
    compression = detect_compression(filepath)
    if compression == "none":
        return filepath.stat().st_size

    # Read raw bytes directly for accurate size calculation
    total_size = 0
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as f:
        with dctx.stream_reader(f) as reader:
            while True:
                chunk = reader.read(65536)  # 64KB chunks
                if not chunk:
                    break
                total_size += len(chunk)

    return total_size
