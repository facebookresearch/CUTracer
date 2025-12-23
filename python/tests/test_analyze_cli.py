# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for analyze CLI command."""

import json
import unittest

from click.testing import CliRunner

from cutracer.cli import main
from tests.test_base import (
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_ZST,
)


class AnalyzeCommandBasicTest(unittest.TestCase):
    """Basic tests for analyze subcommand."""

    def setUp(self):
        self.runner = CliRunner()
        self.test_file = REG_TRACE_NDJSON

    def test_analyze_default(self):
        """Test analyze with default options (first 10 records)."""
        result = self.runner.invoke(main, ["analyze", str(self.test_file)])
        self.assertEqual(result.exit_code, 0)
        # Should have header + 10 data rows
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        self.assertEqual(len(lines), 11)  # 1 header + 10 records

    def test_analyze_help(self):
        """Test analyze --help."""
        result = self.runner.invoke(main, ["analyze", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--head", result.output)
        self.assertIn("--tail", result.output)
        self.assertIn("--filter", result.output)
        self.assertIn("--format", result.output)

    def test_analyze_file_not_found(self):
        """Test error handling for non-existent file."""
        result = self.runner.invoke(main, ["analyze", "/nonexistent/file.ndjson"])
        self.assertEqual(result.exit_code, 2)


class AnalyzeCommandHeadTailTest(unittest.TestCase):
    """Tests for --head and --tail options."""

    def setUp(self):
        self.runner = CliRunner()
        self.test_file = REG_TRACE_NDJSON

    def test_analyze_head(self):
        """Test --head option shows first N records."""
        result = self.runner.invoke(main, ["analyze", str(self.test_file), "-n", "5"])
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        self.assertEqual(len(lines), 6)  # 1 header + 5 records

    def test_analyze_tail(self):
        """Test --tail option shows last N records."""
        result = self.runner.invoke(main, ["analyze", str(self.test_file), "--tail", "5"])
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        self.assertEqual(len(lines), 6)  # 1 header + 5 records


class AnalyzeCommandFilterTest(unittest.TestCase):
    """Tests for --filter option."""

    def setUp(self):
        self.runner = CliRunner()
        self.test_file = REG_TRACE_NDJSON

    def test_analyze_filter_warp(self):
        """Test --filter option filters by warp."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "--filter", "warp=0", "-n", "100"]
        )
        self.assertEqual(result.exit_code, 0)
        # All output records should have warp=0
        lines = result.output.strip().split("\n")
        # Skip header, check data lines start with warp 0
        for line in lines[1:]:
            if line.strip():
                self.assertTrue(line.strip().startswith("0"))

    def test_analyze_filter_no_match(self):
        """Test --filter with no matching records."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "--filter", "warp=99999"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No records found", result.output)

    def test_analyze_filter_invalid(self):
        """Test --filter with invalid expression."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "--filter", "invalid"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid filter expression", result.output)


class AnalyzeCommandFieldsTest(unittest.TestCase):
    """Tests for --fields option."""

    def setUp(self):
        self.runner = CliRunner()
        self.test_file = REG_TRACE_NDJSON

    def test_analyze_default_fields(self):
        """Test default fields are warp, pc, sass."""
        result = self.runner.invoke(main, ["analyze", str(self.test_file), "-n", "1"])
        self.assertEqual(result.exit_code, 0)
        header_line = result.output.strip().split("\n")[0]
        self.assertIn("WARP", header_line)
        self.assertIn("PC", header_line)
        self.assertIn("SASS", header_line)

    def test_analyze_custom_fields(self):
        """Test custom --fields option."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "-n", "1", "--fields", "warp,cta,type"]
        )
        self.assertEqual(result.exit_code, 0)
        header_line = result.output.strip().split("\n")[0]
        self.assertIn("WARP", header_line)
        self.assertIn("CTA", header_line)
        self.assertIn("TYPE", header_line)


class AnalyzeCommandFormatTest(unittest.TestCase):
    """Tests for output format options."""

    def setUp(self):
        self.runner = CliRunner()
        self.test_file = REG_TRACE_NDJSON

    def test_analyze_format_json(self):
        """Test --format json output."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "-n", "2", "--format", "json"]
        )
        self.assertEqual(result.exit_code, 0)
        # Should be valid JSON
        data = json.loads(result.output)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertIn("warp", data[0])

    def test_analyze_format_csv(self):
        """Test --format csv output."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "-n", "2", "--format", "csv"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.output.strip().split("\n")
        # First line is header
        self.assertIn("warp", lines[0])
        # Should have header + 2 data rows
        self.assertEqual(len(lines), 3)

    def test_analyze_no_header(self):
        """Test --no-header option."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "-n", "3", "--no-header"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        self.assertEqual(len(lines), 3)  # Just 3 records, no header

    def test_analyze_zst_file(self):
        """Test analyzing Zstd-compressed file."""
        result = self.runner.invoke(main, ["analyze", str(REG_TRACE_NDJSON_ZST), "-n", "5"])
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        self.assertEqual(len(lines), 6)  # 1 header + 5 records


class AnalyzeInHelpTest(unittest.TestCase):
    """Test that analyze appears in main help."""

    def setUp(self):
        self.runner = CliRunner()

    def test_analyze_in_main_help(self):
        """Test that analyze command appears in main help."""
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("analyze", result.output)


class AnalyzeCommandGroupByTest(unittest.TestCase):
    """Tests for --group-by option."""

    def setUp(self):
        self.runner = CliRunner()
        self.test_file = REG_TRACE_NDJSON

    def test_analyze_group_by_warp_tail(self):
        """Test --group-by warp --tail option (core use case)."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "--group-by", "warp", "--tail", "2"]
        )
        self.assertEqual(result.exit_code, 0)
        # Should have group headers
        self.assertIn("warp=", result.output)
        self.assertIn("records", result.output)

    def test_analyze_group_by_warp_head(self):
        """Test --group-by warp --head option."""
        result = self.runner.invoke(
            main, ["analyze", str(self.test_file), "--group-by", "warp", "-n", "2"]
        )
        self.assertEqual(result.exit_code, 0)
        # Should have group headers
        self.assertIn("warp=", result.output)

    def test_analyze_group_by_json_format(self):
        """Test --group-by with --format json."""
        result = self.runner.invoke(
            main,
            ["analyze", str(self.test_file), "--group-by", "warp", "--tail", "2", "--format", "json"],
        )
        self.assertEqual(result.exit_code, 0)
        # Should be valid JSON with group keys
        data = json.loads(result.output)
        self.assertIsInstance(data, dict)
        # Keys should be warp values (as strings)
        self.assertTrue(len(data) > 0)

    def test_analyze_group_by_with_filter(self):
        """Test --group-by combined with --filter."""
        result = self.runner.invoke(
            main,
            ["analyze", str(self.test_file), "--filter", "warp=0", "--group-by", "warp", "--tail", "5"],
        )
        self.assertEqual(result.exit_code, 0)
        # Should only have warp=0 group
        self.assertIn("warp=0", result.output)
        # Should not have other warps
        self.assertNotIn("warp=1", result.output)


if __name__ == "__main__":
    unittest.main()
