# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for analyze CLI command."""

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
        self.assertIn("--fields", result.output)

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


if __name__ == "__main__":
    unittest.main()
