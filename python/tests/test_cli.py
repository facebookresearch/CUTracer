# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for cutracer CLI."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from cutracer.cli import main
from cutracer.validation.cli import _detect_format, _format_size


class DetectFormatTest(unittest.TestCase):
    """Tests for _detect_format function."""

    def test_detect_ndjson(self):
        self.assertEqual(_detect_format(Path("trace.ndjson")), "json")

    def test_detect_ndjson_zst(self):
        self.assertEqual(_detect_format(Path("trace.ndjson.zst")), "json")

    def test_detect_log(self):
        self.assertEqual(_detect_format(Path("trace.log")), "text")

    def test_detect_unknown(self):
        self.assertEqual(_detect_format(Path("trace.txt")), "unknown")
        self.assertEqual(_detect_format(Path("trace.bin")), "unknown")


class FormatSizeTest(unittest.TestCase):
    """Tests for _format_size function."""

    def test_format_bytes(self):
        self.assertEqual(_format_size(500), "500 B")

    def test_format_kilobytes(self):
        self.assertEqual(_format_size(2048), "2.0 KB")

    def test_format_megabytes(self):
        self.assertEqual(_format_size(1048576), "1.0 MB")
        self.assertEqual(_format_size(2621440), "2.5 MB")


class ValidateCommandTest(unittest.TestCase):
    """Tests for validate subcommand."""

    def setUp(self):
        self.test_dir = Path(__file__).parent / "example_inputs"

    def test_validate_valid_json(self):
        """Test validating a valid NDJSON file."""
        with patch(
            "sys.argv",
            ["cutraceross", "validate", str(self.test_dir / "reg_trace_sample.ndjson")],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)

    def test_validate_valid_json_zst(self):
        """Test validating a valid Zstd-compressed NDJSON file."""
        with patch(
            "sys.argv",
            [
                "cutraceross",
                "validate",
                str(self.test_dir / "reg_trace_sample.ndjson.zst"),
            ],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)

    def test_validate_valid_text(self):
        """Test validating a valid text log file."""
        with patch(
            "sys.argv",
            ["cutraceross", "validate", str(self.test_dir / "reg_trace_sample.log")],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)

    def test_validate_invalid_syntax(self):
        """Test validating a file with invalid JSON syntax."""
        with patch(
            "sys.argv",
            ["cutraceross", "validate", str(self.test_dir / "invalid_syntax.ndjson")],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 1)

    def test_validate_invalid_schema(self):
        """Test validating a file with schema errors."""
        with patch(
            "sys.argv",
            ["cutraceross", "validate", str(self.test_dir / "invalid_schema.ndjson")],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 1)

    def test_validate_quiet_mode(self):
        """Test quiet mode returns only exit code."""
        with patch(
            "sys.argv",
            [
                "cutraceross",
                "validate",
                "--quiet",
                str(self.test_dir / "reg_trace_sample.ndjson"),
            ],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)

    def test_validate_json_output(self):
        """Test JSON output format."""
        with patch(
            "sys.argv",
            [
                "cutraceross",
                "validate",
                "--json",
                str(self.test_dir / "reg_trace_sample.ndjson"),
            ],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)

    def test_validate_file_not_found(self):
        """Test error handling for non-existent file."""
        with patch("sys.argv", ["cutraceross", "validate", "/nonexistent/file.ndjson"]):
            exit_code = main()
            self.assertEqual(exit_code, 2)

    def test_validate_unknown_format(self):
        """Test error handling for unknown format."""
        with patch(
            "sys.argv",
            [
                "cutraceross",
                "validate",
                str(self.test_dir / "reg_trace_sample.ndjson").replace(
                    ".ndjson", ".unknown"
                ),
            ],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 2)

    def test_validate_explicit_format_json(self):
        """Test explicit --format json option."""
        with patch(
            "sys.argv",
            [
                "cutraceross",
                "validate",
                "--format",
                "json",
                str(self.test_dir / "reg_trace_sample.ndjson"),
            ],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)

    def test_validate_explicit_format_text(self):
        """Test explicit --format text option."""
        with patch(
            "sys.argv",
            [
                "cutraceross",
                "validate",
                "--format",
                "text",
                str(self.test_dir / "reg_trace_sample.log"),
            ],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)


class MainEntryPointTest(unittest.TestCase):
    """Tests for main entry point."""

    def test_version_flag(self):
        """Test --version flag."""
        with patch("sys.argv", ["cutraceross", "--version"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)

    def test_help_flag(self):
        """Test --help flag."""
        with patch("sys.argv", ["cutraceross", "--help"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)

    def test_no_command(self):
        """Test error when no command is provided."""
        with patch("sys.argv", ["cutraceross"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertNotEqual(cm.exception.code, 0)


class ModuleEntryPointTest(unittest.TestCase):
    """Tests for python -m cutracer entry point."""

    def test_module_help(self):
        """Test python -m cutracer --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "cutracer", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("cutraceross", result.stdout)
        self.assertIn("validate", result.stdout)


if __name__ == "__main__":
    unittest.main()
