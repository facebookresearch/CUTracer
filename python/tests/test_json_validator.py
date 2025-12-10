# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for JSON validator module.
"""

import unittest

from cutracer.validation.json_validator import (
    JsonValidationError,
    validate_json_schema,
    validate_json_syntax,
    validate_json_trace,
)
from tests.test_base import (
    BaseValidationTest,
    INVALID_SCHEMA_NDJSON,
    INVALID_SYNTAX_NDJSON,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_RECORD_COUNT,
    REG_TRACE_NDJSON_ZST,
    REG_TRACE_NDJSON_ZST_RECORD_COUNT,
)


class JsonValidatorTest(BaseValidationTest):
    """Core tests for JSON validator functions."""

    def test_validate_json_syntax_valid_file(self):
        """Test validation of real NDJSON trace file."""
        valid_count, errors = validate_json_syntax(REG_TRACE_NDJSON)

        self.assertEqual(valid_count, REG_TRACE_NDJSON_RECORD_COUNT)
        self.assertEqual(len(errors), 0)

    def test_validate_json_syntax_invalid_json(self):
        """Test detection of invalid JSON syntax."""
        valid_count, errors = validate_json_syntax(INVALID_SYNTAX_NDJSON)

        self.assertEqual(valid_count, 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("Line 2", errors[0])

    def test_validate_json_syntax_file_not_found(self):
        """Test handling of non-existent file."""
        non_existent = self.temp_dir / "missing.ndjson"

        with self.assertRaises(FileNotFoundError):
            validate_json_syntax(non_existent)

    def test_validate_json_schema_valid(self):
        """Test schema validation with real trace data."""
        result = validate_json_schema(REG_TRACE_NDJSON, message_type="reg_trace")

        self.assertTrue(result)

    def test_validate_json_schema_invalid(self):
        """Test schema validation with invalid records."""
        with self.assertRaises(JsonValidationError) as ctx:
            validate_json_schema(INVALID_SCHEMA_NDJSON, message_type="reg_trace")

        self.assertIn("Schema validation failed", str(ctx.exception))

    def test_validate_json_trace_complete(self):
        """Test complete validation of real trace file."""
        result = validate_json_trace(REG_TRACE_NDJSON)

        self.assertTrue(result["valid"])
        self.assertEqual(result["record_count"], REG_TRACE_NDJSON_RECORD_COUNT)
        self.assertEqual(result["message_type"], "reg_trace")
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_json_trace_empty_file(self):
        """Test validation of empty file."""
        empty_file = self.create_temp_file("empty.ndjson", "")

        result = validate_json_trace(empty_file)

        self.assertFalse(result["valid"])
        self.assertEqual(result["record_count"], 0)


class JsonValidatorCompressedTest(BaseValidationTest):
    """Tests for JSON validator with compressed files."""

    def test_validate_json_syntax_compressed(self):
        """Test validation of Zstd compressed NDJSON trace file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        valid_count, errors = validate_json_syntax(REG_TRACE_NDJSON_ZST)

        self.assertEqual(valid_count, REG_TRACE_NDJSON_ZST_RECORD_COUNT)
        self.assertEqual(len(errors), 0)

    def test_validate_json_schema_compressed(self):
        """Test schema validation with compressed trace data."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = validate_json_schema(REG_TRACE_NDJSON_ZST, message_type="reg_trace")

        self.assertTrue(result)

    def test_validate_json_trace_compressed(self):
        """Test complete validation of compressed trace file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = validate_json_trace(REG_TRACE_NDJSON_ZST)

        self.assertTrue(result["valid"])
        self.assertEqual(result["record_count"], REG_TRACE_NDJSON_ZST_RECORD_COUNT)
        self.assertEqual(result["message_type"], "reg_trace")
        self.assertEqual(result["compression"], "zstd")
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_json_trace_compression_field_uncompressed(self):
        """Test that uncompressed files have compression='none'."""
        result = validate_json_trace(REG_TRACE_NDJSON)

        self.assertEqual(result["compression"], "none")


if __name__ == "__main__":
    unittest.main()
