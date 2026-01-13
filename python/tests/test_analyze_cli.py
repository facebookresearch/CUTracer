# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for analyze CLI command.
"""

import unittest

from click.testing import CliRunner
from cutracer.cli import main
from tests.test_base import BaseValidationTest, REG_TRACE_NDJSON, REG_TRACE_NDJSON_ZST


class TestAnalyzeCommand(BaseValidationTest):
    """Tests for analyze CLI command."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()

    def test_analyze_help(self):
        """Test analyze --help shows usage information."""
        result = self.runner.invoke(main, ["analyze", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Analyze trace data", result.output)
        self.assertIn("--head", result.output)
        self.assertIn("--tail", result.output)
        self.assertIn("--filter", result.output)

    def test_analyze_default_head(self):
        """Test analyze with default head (10 records)."""
        result = self.runner.invoke(main, ["analyze", str(REG_TRACE_NDJSON)])
        self.assertEqual(result.exit_code, 0)
        # Should have header + 10 data rows = 11 lines
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 11)

    def test_analyze_custom_head(self):
        """Test analyze with custom head value."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON), "--head", "5"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 6)  # header + 5 data rows

    def test_analyze_tail(self):
        """Test analyze with tail option."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON), "--tail", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_zst_file(self):
        """Test analyze with Zstd-compressed file."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON_ZST), "--head", "5"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 6)

    def test_analyze_filter(self):
        """Test analyze with filter expression."""
        result = self.runner.invoke(
            main,
            ["analyze", str(REG_TRACE_NDJSON), "--filter", "warp=0", "--head", "5"],
        )
        self.assertEqual(result.exit_code, 0)
        # All displayed records should have warp=0
        # Check that output contains "0" in the warp column
        self.assertIn("0", result.output)

    def test_analyze_filter_invalid(self):
        """Test analyze with invalid filter expression."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON), "--filter", "invalid"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid filter expression", result.output)

    def test_analyze_nonexistent_file(self):
        """Test analyze with non-existent file."""
        result = self.runner.invoke(main, ["analyze", "/nonexistent/file.ndjson"])
        self.assertNotEqual(result.exit_code, 0)

    def test_analyze_empty_file(self):
        """Test analyze with empty file."""
        empty_file = self.create_temp_file("empty.ndjson", "")
        result = self.runner.invoke(main, ["analyze", str(empty_file)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No records found", result.output)

    def test_analyze_short_options(self):
        """Test analyze with short option names."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON), "-n", "3", "-f", "warp=0"]
        )
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
