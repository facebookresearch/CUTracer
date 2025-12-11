# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for text validator module.
"""

import unittest

from cutracer.validation.text_validator import (
    parse_text_trace_record,
    REG_INFO_HEADER_PATTERN,
    validate_text_format,
    validate_text_trace,
)
from tests.test_base import (
    BaseValidationTest,
    REG_TRACE_LOG,
    REG_TRACE_LOG_RECORD_COUNT,
)


class TextValidatorTest(BaseValidationTest):
    """Core tests for text validator functions."""

    def test_reg_info_header_pattern_matches_real_data(self):
        """Test that REG_INFO_HEADER_PATTERN matches headers in real trace file."""
        with open(REG_TRACE_LOG, "r") as f:
            match_count = 0
            for line in f:
                if REG_INFO_HEADER_PATTERN.match(line):
                    match_count += 1
                    if match_count >= 10:
                        break
        self.assertEqual(match_count, 10)

    def test_validate_text_format_real_file(self):
        """Test validation of real text trace file."""
        result = validate_text_format(REG_TRACE_LOG)

        self.assertTrue(result)

    def test_validate_text_format_file_not_found(self):
        """Test handling of non-existent file."""
        non_existent = self.temp_dir / "missing.log"

        with self.assertRaises(FileNotFoundError):
            validate_text_format(non_existent)

    def test_validate_text_trace_real_file(self):
        """Test complete validation of real text trace file."""
        result = validate_text_trace(REG_TRACE_LOG)

        self.assertTrue(result["valid"])
        self.assertEqual(result["record_count"], REG_TRACE_LOG_RECORD_COUNT)
        self.assertGreater(result["file_size"], 0)
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_text_trace_empty_file(self):
        """Test validation of empty file."""
        empty_file = self.create_temp_file("empty.log", "")

        result = validate_text_trace(empty_file)

        self.assertFalse(result["valid"])
        self.assertEqual(result["record_count"], 0)
        self.assertIn("No trace records", str(result["errors"]))

    def test_parse_text_trace_record_from_real_file(self):
        """Test parsing of real trace record header."""
        with open(REG_TRACE_LOG, "r") as f:
            first_line = f.readline().strip()

        result = parse_text_trace_record([first_line])

        self.assertIn("ctx", result)
        self.assertIn("cta", result)
        self.assertIn("warp", result)
        self.assertIn("sass", result)
        self.assertEqual(result["record_type"], "reg_info")
        self.assertEqual(len(result["cta"]), 3)

    def test_parse_text_trace_record_invalid(self):
        """Test that invalid header raises ValueError."""
        with self.assertRaises(ValueError):
            parse_text_trace_record(["This is not a valid header"])

        with self.assertRaises(ValueError):
            parse_text_trace_record([])


if __name__ == "__main__":
    unittest.main()
