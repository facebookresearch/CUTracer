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

    def test_analyze_format_json(self):
        """Test analyze with JSON output format."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON), "--format", "json", "--head", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        # JSON output should be parseable
        import json

        data = json.loads(result.output)
        self.assertEqual(len(data), 3)

    def test_analyze_format_csv(self):
        """Test analyze with CSV output format."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON), "--format", "csv", "--head", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.output.strip().split("\n")
        self.assertEqual(len(lines), 4)  # header + 3 data rows
        # CSV header should contain field names
        self.assertIn("warp", lines[0])

    def test_analyze_format_table_explicit(self):
        """Test analyze with explicit table format."""
        result = self.runner.invoke(
            main,
            ["analyze", str(REG_TRACE_NDJSON), "--format", "table", "--head", "3"],
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_fields_custom(self):
        """Test analyze with custom fields."""
        result = self.runner.invoke(
            main,
            [
                "analyze",
                str(REG_TRACE_NDJSON),
                "--fields",
                "warp,sass",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Output should contain WARP and SASS headers
        self.assertIn("WARP", result.output)
        self.assertIn("SASS", result.output)
        # But not PC (since we only asked for warp,sass)
        self.assertNotIn("PC", result.output.split("\n")[0])

    def test_analyze_no_header(self):
        """Test analyze with --no-header flag."""
        result = self.runner.invoke(
            main, ["analyze", str(REG_TRACE_NDJSON), "--no-header", "--head", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 3)  # 3 data rows, no header

    def test_analyze_format_csv_no_header(self):
        """Test analyze with CSV format and no header."""
        result = self.runner.invoke(
            main,
            [
                "analyze",
                str(REG_TRACE_NDJSON),
                "--format",
                "csv",
                "--no-header",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.output.strip().split("\n")
        self.assertEqual(len(lines), 3)  # 3 data rows, no header

    def test_analyze_group_by_basic(self):
        """Test analyze with --group-by option."""
        result = self.runner.invoke(
            main,
            ["analyze", str(REG_TRACE_NDJSON), "--group-by", "warp", "--head", "3"],
        )
        self.assertEqual(result.exit_code, 0)
        # Should have group headers
        self.assertIn("=== Group:", result.output)
        self.assertIn("records) ===", result.output)

    def test_analyze_group_by_with_tail(self):
        """Test analyze with --group-by and --tail."""
        result = self.runner.invoke(
            main,
            ["analyze", str(REG_TRACE_NDJSON), "--group-by", "warp", "--tail", "2"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)

    def test_analyze_group_by_with_filter(self):
        """Test analyze with --group-by and --filter."""
        result = self.runner.invoke(
            main,
            [
                "analyze",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--filter",
                "warp=0",
                "--head",
                "5",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should only have warp=0 group
        self.assertIn("=== Group: 0", result.output)

    def test_analyze_group_by_json_format(self):
        """Test analyze with --group-by and JSON format."""
        result = self.runner.invoke(
            main,
            [
                "analyze",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--format",
                "json",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)
        # JSON output should contain brackets
        self.assertIn("[", result.output)
        self.assertIn("]", result.output)

    def test_analyze_group_by_csv_format(self):
        """Test analyze with --group-by and CSV format."""
        result = self.runner.invoke(
            main,
            [
                "analyze",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--format",
                "csv",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)

    def test_analyze_group_by_short_option(self):
        """Test analyze with -g short option for --group-by."""
        result = self.runner.invoke(
            main,
            ["analyze", str(REG_TRACE_NDJSON), "-g", "warp", "-n", "2"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)


if __name__ == "__main__":
    unittest.main()
