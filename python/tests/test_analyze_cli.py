# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the analyze command."""

import json
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner
from cutracer.cli import main


class AnalyzeWarpSummaryTest(unittest.TestCase):
    """Tests for analyze warp-summary command."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test trace file with warp data
        self.trace_file = Path(self.temp_dir) / "trace.ndjson"
        with open(self.trace_file, "w") as f:
            records = [
                {"warp": 0, "pc": 100, "sass": "MOV R1, R2;"},
                {"warp": 0, "pc": 200, "sass": "ADD R1, R2, R3;"},
                {"warp": 0, "pc": 300, "sass": "EXIT;"},
                {"warp": 1, "pc": 100, "sass": "MOV R1, R2;"},
                {"warp": 1, "pc": 200, "sass": "ADD R1, R2, R3;"},
                {"warp": 2, "pc": 100, "sass": "MOV R1, R2;"},
                {"warp": 2, "pc": 200, "sass": "EXIT;"},
            ]
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_warp_summary_basic(self):
        """Test basic warp-summary command."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", str(self.trace_file)],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Warp Summary", result.output)
        self.assertIn("Total warps observed", result.output)
        self.assertIn("Completed (EXIT)", result.output)
        self.assertIn("In-progress", result.output)

    def test_warp_summary_json_format(self):
        """Test warp-summary with JSON output."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", str(self.trace_file), "--format", "json"],
        )

        self.assertEqual(result.exit_code, 0)

        # Parse JSON output
        output = json.loads(result.output)
        self.assertIn("total_observed", output)
        self.assertIn("completed", output)
        self.assertIn("in_progress", output)
        self.assertIn("missing", output)

        # Verify counts
        self.assertEqual(output["total_observed"], 3)
        self.assertEqual(output["completed"]["count"], 2)  # warp 0 and 2
        self.assertEqual(output["in_progress"]["count"], 1)  # warp 1

    def test_warp_summary_output_file(self):
        """Test warp-summary with output file."""
        output_file = Path(self.temp_dir) / "summary.txt"

        result = self.runner.invoke(
            main,
            [
                "analyze",
                "warp-summary",
                str(self.trace_file),
                "-o",
                str(output_file),
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Output written to", result.output)
        self.assertTrue(output_file.exists())

        content = output_file.read_text()
        self.assertIn("Warp Summary", content)

    def test_warp_summary_json_output_file(self):
        """Test warp-summary with JSON output to file."""
        output_file = Path(self.temp_dir) / "summary.json"

        result = self.runner.invoke(
            main,
            [
                "analyze",
                "warp-summary",
                str(self.trace_file),
                "--format",
                "json",
                "-o",
                str(output_file),
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(output_file.exists())

        with open(output_file) as f:
            output = json.load(f)
        self.assertIn("total_observed", output)

    def test_warp_summary_nonexistent_file(self):
        """Test warp-summary with nonexistent file."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", "/nonexistent/file.ndjson"],
        )

        self.assertNotEqual(result.exit_code, 0)

    def test_warp_summary_help(self):
        """Test warp-summary --help."""
        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", "--help"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Analyze warp execution status", result.output)
        self.assertIn("--format", result.output)
        self.assertIn("--output", result.output)

    def test_analyze_help(self):
        """Test analyze --help shows subcommands."""
        result = self.runner.invoke(
            main,
            ["analyze", "--help"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Analyze trace data", result.output)
        self.assertIn("warp-summary", result.output)

    def test_analyze_no_subcommand(self):
        """Test analyze without subcommand shows help."""
        result = self.runner.invoke(
            main,
            ["analyze"],
        )

        # Click 8+ returns 0 and shows help; Click 7 returns 2
        self.assertIn(result.exit_code, (0, 2))
        self.assertIn("warp-summary", result.output)


class AnalyzeIntegrationTest(unittest.TestCase):
    """Integration tests for analyze command with compressed files."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def test_warp_summary_from_zst_file(self):
        """Test warp-summary from Zstd compressed file."""
        import zstandard as zstd

        # Create compressed test file
        trace_file = Path(self.temp_dir) / "trace.ndjson.zst"
        records = [
            {"warp": 0, "pc": 100, "sass": "EXIT;"},
            {"warp": 1, "pc": 100, "sass": "MOV R1, R2;"},
        ]

        cctx = zstd.ZstdCompressor()
        with open(trace_file, "wb") as f:
            with cctx.stream_writer(f) as writer:
                for record in records:
                    writer.write((json.dumps(record) + "\n").encode("utf-8"))

        result = self.runner.invoke(
            main,
            ["analyze", "warp-summary", str(trace_file)],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Warp Summary", result.output)


if __name__ == "__main__":
    unittest.main()
