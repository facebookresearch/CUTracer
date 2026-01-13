# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for TraceReader class.
"""

import json
import types
import unittest

from cutracer.analysis import TraceReader
from tests.test_base import (
    BaseValidationTest,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_RECORD_COUNT,
    REG_TRACE_NDJSON_ZST,
    REG_TRACE_NDJSON_ZST_RECORD_COUNT,
)


class TestTraceReaderInit(BaseValidationTest):
    """Tests for TraceReader initialization."""

    def test_init_with_ndjson_file(self):
        """Test initialization with a plain NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON)
        self.assertEqual(reader.file_path, REG_TRACE_NDJSON)
        self.assertEqual(reader.compression, "none")

    def test_init_with_zst_file(self):
        """Test initialization with a Zstd-compressed NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON_ZST)
        self.assertEqual(reader.file_path, REG_TRACE_NDJSON_ZST)
        self.assertEqual(reader.compression, "zstd")

    def test_init_with_string_path(self):
        """Test initialization with a string path."""
        reader = TraceReader(str(REG_TRACE_NDJSON))
        self.assertEqual(reader.file_path, REG_TRACE_NDJSON)

    def test_init_with_nonexistent_file(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            TraceReader("/nonexistent/path/file.ndjson")


class TestTraceReaderIterRecords(BaseValidationTest):
    """Tests for TraceReader.iter_records() method."""

    def test_iter_records_ndjson(self):
        """Test iterating over records in a plain NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON)
        records = list(reader.iter_records())

        self.assertEqual(len(records), REG_TRACE_NDJSON_RECORD_COUNT)

        # Verify first record structure
        first_record = records[0]
        self.assertIn("type", first_record)
        self.assertIn("warp", first_record)
        self.assertIn("sass", first_record)
        self.assertEqual(first_record["type"], "reg_trace")

    def test_iter_records_zst(self):
        """Test iterating over records in a Zstd-compressed NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON_ZST)
        records = list(reader.iter_records())

        self.assertEqual(len(records), REG_TRACE_NDJSON_ZST_RECORD_COUNT)

    def test_iter_records_zst_has_valid_structure(self):
        """Test that Zstd compressed file records have valid structure."""
        reader = TraceReader(REG_TRACE_NDJSON_ZST)
        records = list(reader.iter_records())

        self.assertGreater(len(records), 0)

        # Verify record structure
        first_record = records[0]
        self.assertIn("type", first_record)
        self.assertIn("warp", first_record)
        self.assertIn("sass", first_record)

    def test_iter_records_is_generator(self):
        """Test that iter_records returns a generator (lazy evaluation)."""
        reader = TraceReader(REG_TRACE_NDJSON)
        result = reader.iter_records()

        # Should be a generator, not a list
        self.assertIsInstance(result, types.GeneratorType)

    def test_iter_records_can_iterate_multiple_times(self):
        """Test that iter_records can be called multiple times."""
        reader = TraceReader(REG_TRACE_NDJSON)

        # First iteration
        records1 = list(reader.iter_records())
        # Second iteration
        records2 = list(reader.iter_records())

        self.assertEqual(len(records1), len(records2))
        self.assertEqual(records1[0], records2[0])


class TestTraceReaderEdgeCases(BaseValidationTest):
    """Edge case tests for TraceReader."""

    def test_empty_file(self):
        """Test reading an empty file."""
        filepath = self.create_temp_file("empty.ndjson", "")

        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 0)

    def test_single_record_file(self):
        """Test reading a file with a single record."""
        content = '{"type": "reg_trace", "warp": 0, "sass": "NOP ;"}\n'
        filepath = self.create_temp_file("single.ndjson", content)

        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "reg_trace")

    def test_empty_lines_skipped(self):
        """Test that empty lines in the file are skipped."""
        content = (
            '{"type": "reg_trace", "warp": 0}\n\n{"type": "reg_trace", "warp": 1}\n'
        )
        filepath = self.create_temp_file("test.ndjson", content)

        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["warp"], 0)
        self.assertEqual(records[1]["warp"], 1)

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        content = '{"type": "reg_trace"}\n{invalid json}\n'
        filepath = self.create_temp_file("invalid.ndjson", content)

        reader = TraceReader(filepath)
        iterator = reader.iter_records()

        # First record should work
        first = next(iterator)
        self.assertEqual(first["type"], "reg_trace")

        # Second record should raise
        with self.assertRaises(json.JSONDecodeError):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
