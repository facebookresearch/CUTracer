# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for consistency checker module.
"""

import unittest

from cutracer.validation.consistency import (
    compare_record_counts,
    compare_trace_formats,
    get_trace_statistics,
)
from tests.test_base import (
    BaseValidationTest,
    REG_TRACE_LOG,
    REG_TRACE_LOG_RECORD_COUNT,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_RECORD_COUNT,
    REG_TRACE_NDJSON_ZST,
    REG_TRACE_NDJSON_ZST_RECORD_COUNT,
)


class ConsistencyTest(BaseValidationTest):
    """Core tests for consistency checker functions."""

    def test_compare_record_counts_exact_match(self):
        """Test that exact count match returns True."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 100}

        result = compare_record_counts(text_metadata, json_metadata)

        self.assertTrue(result)

    def test_compare_record_counts_outside_tolerance(self):
        """Test that counts outside tolerance return False."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 150}

        result = compare_record_counts(text_metadata, json_metadata, tolerance=0.1)

        self.assertFalse(result)

    def test_compare_record_counts_missing_field(self):
        """Test that missing record_count field raises ValueError."""
        with self.assertRaises(ValueError):
            compare_record_counts({}, {"record_count": 100})

        with self.assertRaises(ValueError):
            compare_record_counts({"record_count": 100}, {})

    def test_get_trace_statistics_real_ndjson(self):
        """Test statistics extraction from real NDJSON file."""
        stats = get_trace_statistics(REG_TRACE_NDJSON)

        self.assertEqual(stats["format"], "json")
        self.assertEqual(stats["record_count"], REG_TRACE_NDJSON_RECORD_COUNT)
        self.assertGreater(stats["file_size"], 0)
        self.assertIn("reg_trace", stats["message_types"])

    def test_get_trace_statistics_real_text(self):
        """Test statistics extraction from real text file."""
        stats = get_trace_statistics(REG_TRACE_LOG)

        self.assertEqual(stats["format"], "text")
        self.assertEqual(stats["record_count"], REG_TRACE_LOG_RECORD_COUNT)
        self.assertGreater(stats["file_size"], 0)

    def test_get_trace_statistics_file_not_found(self):
        """Test handling of non-existent file."""
        non_existent = self.temp_dir / "missing.ndjson"

        with self.assertRaises(FileNotFoundError):
            get_trace_statistics(non_existent)

    def test_compare_trace_formats_real_files(self):
        """Test comparison of real text and JSON trace files."""
        result = compare_trace_formats(REG_TRACE_LOG, REG_TRACE_NDJSON)

        self.assertIn("consistent", result)
        self.assertIn("record_count_match", result)
        self.assertIn("text_records", result)
        self.assertIn("json_records", result)
        self.assertEqual(result["text_records"], REG_TRACE_LOG_RECORD_COUNT)
        self.assertEqual(result["json_records"], REG_TRACE_NDJSON_RECORD_COUNT)


class ConsistencyCompressedTest(BaseValidationTest):
    """Tests for consistency checker with compressed files."""

    def test_get_trace_statistics_compressed(self):
        """Test statistics extraction from compressed NDJSON file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        stats = get_trace_statistics(REG_TRACE_NDJSON_ZST)

        self.assertEqual(stats["format"], "json")
        self.assertEqual(stats["compression"], "zstd")
        self.assertEqual(stats["record_count"], REG_TRACE_NDJSON_ZST_RECORD_COUNT)
        self.assertGreater(stats["file_size"], 0)
        self.assertIn("reg_trace", stats["message_types"])

    def test_get_trace_statistics_compression_field_uncompressed(self):
        """Test that uncompressed files have compression='none'."""
        stats = get_trace_statistics(REG_TRACE_NDJSON)

        self.assertEqual(stats["compression"], "none")

    def test_compare_trace_formats_with_compressed_json(self):
        """Test comparison of text and compressed JSON trace files."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = compare_trace_formats(REG_TRACE_LOG, REG_TRACE_NDJSON_ZST)

        self.assertIn("consistent", result)
        self.assertIn("record_count_match", result)
        self.assertEqual(result["text_records"], REG_TRACE_LOG_RECORD_COUNT)
        self.assertEqual(result["json_records"], REG_TRACE_NDJSON_ZST_RECORD_COUNT)


if __name__ == "__main__":
    unittest.main()
