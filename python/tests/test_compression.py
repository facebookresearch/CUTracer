# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for compression module."""

import unittest

from cutracer.validation.compression import (
    detect_compression,
    get_trace_format,
    iter_lines,
    open_trace_file,
)
from tests.test_base import (
    EXAMPLE_INPUTS_DIR,
    REG_TRACE_LOG,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_ZST,
)


class CompressionDetectionTest(unittest.TestCase):
    """Tests for compression detection functions."""

    def test_detect_compression_zstd(self) -> None:
        """Test detection of Zstd compressed file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = detect_compression(REG_TRACE_NDJSON_ZST)
        self.assertEqual(result, "zstd")

    def test_detect_compression_none(self) -> None:
        """Test detection of uncompressed file."""
        result = detect_compression(REG_TRACE_NDJSON)
        self.assertEqual(result, "none")

    def test_detect_compression_nonexistent(self) -> None:
        """Test detection raises error for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            detect_compression(EXAMPLE_INPUTS_DIR / "nonexistent.ndjson")


class TraceFormatDetectionTest(unittest.TestCase):
    """Tests for trace format detection."""

    def test_get_trace_format_ndjson(self) -> None:
        """Test format detection for uncompressed NDJSON."""
        base_format, compression = get_trace_format(REG_TRACE_NDJSON)
        self.assertEqual(base_format, "ndjson")
        self.assertEqual(compression, "none")

    def test_get_trace_format_ndjson_zst(self) -> None:
        """Test format detection for compressed NDJSON."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        base_format, compression = get_trace_format(REG_TRACE_NDJSON_ZST)
        self.assertEqual(base_format, "ndjson")
        self.assertEqual(compression, "zstd")

    def test_get_trace_format_text(self) -> None:
        """Test format detection for text log file."""
        base_format, compression = get_trace_format(REG_TRACE_LOG)
        self.assertEqual(base_format, "text")
        self.assertEqual(compression, "none")


class OpenTraceFileTest(unittest.TestCase):
    """Tests for open_trace_file function."""

    def test_open_uncompressed(self) -> None:
        """Test opening uncompressed file."""
        with open_trace_file(REG_TRACE_NDJSON) as f:
            first_line = next(iter(f))
            self.assertTrue(first_line.strip().startswith("{"))

    def test_open_compressed(self) -> None:
        """Test opening Zstd compressed file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        with open_trace_file(REG_TRACE_NDJSON_ZST) as f:
            first_line = next(iter(f))
            self.assertTrue(first_line.strip().startswith("{"))

    def test_open_nonexistent(self) -> None:
        """Test opening non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            with open_trace_file(EXAMPLE_INPUTS_DIR / "nonexistent.ndjson"):
                pass


class IterLinesTest(unittest.TestCase):
    """Tests for iter_lines function."""

    def test_iter_lines_uncompressed(self) -> None:
        """Test iterating lines from uncompressed file."""
        lines = list(iter_lines(REG_TRACE_NDJSON))
        self.assertGreater(len(lines), 0)
        # Each line should be valid JSON (starts with '{')
        for line in lines[:10]:
            self.assertTrue(line.startswith("{"), f"Invalid line: {line[:50]}")

    def test_iter_lines_compressed(self) -> None:
        """Test iterating lines from compressed file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        lines = list(iter_lines(REG_TRACE_NDJSON_ZST))
        self.assertGreater(len(lines), 0)
        # Each line should be valid JSON (starts with '{')
        for line in lines[:10]:
            self.assertTrue(line.startswith("{"), f"Invalid line: {line[:50]}")

    def test_iter_lines_compressed_has_content(self) -> None:
        """Test that compressed file can be iterated and has content."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        compressed_lines = list(iter_lines(REG_TRACE_NDJSON_ZST))
        self.assertGreater(len(compressed_lines), 0)


if __name__ == "__main__":
    unittest.main()
