# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for cutracer.query.sass module.

Tests SASS extraction from cubin files using nvdisasm.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cutracer.query.sass import dump_sass, dump_sass_to_file, SassOutput


class TestSassOutput(unittest.TestCase):
    """Tests for SassOutput dataclass."""

    def test_save_creates_file(self) -> None:
        """SassOutput.save() writes content to file."""
        sass = SassOutput(
            raw_text="MOV R1, R2",
            cubin_path=Path("/tmp/test.cubin"),
            flags_used=["-g", "-c"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.sass"
            sass.save(output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_text(), "MOV R1, R2")

    def test_save_creates_parent_dirs(self) -> None:
        """SassOutput.save() creates parent directories if needed."""
        sass = SassOutput(
            raw_text="MOV R1, R2",
            cubin_path=Path("/tmp/test.cubin"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "output.sass"
            sass.save(output_path)
            self.assertTrue(output_path.exists())

    def test_len(self) -> None:
        """SassOutput __len__ returns text length."""
        sass = SassOutput(
            raw_text="MOV R1, R2",
            cubin_path=Path("/tmp/test.cubin"),
        )
        self.assertEqual(len(sass), 10)

    def test_line_count(self) -> None:
        """SassOutput.line_count returns number of lines."""
        sass = SassOutput(
            raw_text="MOV R1, R2\nADD R3, R4\nSUB R5, R6",
            cubin_path=Path("/tmp/test.cubin"),
        )
        self.assertEqual(sass.line_count, 3)

    def test_line_count_trailing_newline(self) -> None:
        """SassOutput.line_count handles trailing newline correctly."""
        sass = SassOutput(
            raw_text="MOV R1, R2\nADD R3, R4\n",
            cubin_path=Path("/tmp/test.cubin"),
        )
        self.assertEqual(sass.line_count, 2)

    def test_line_count_empty(self) -> None:
        """SassOutput.line_count handles empty text."""
        sass = SassOutput(
            raw_text="",
            cubin_path=Path("/tmp/test.cubin"),
        )
        self.assertEqual(sass.line_count, 0)


class TestDumpSass(unittest.TestCase):
    """Tests for dump_sass() function."""

    def test_nonexistent_file_returns_none(self) -> None:
        """Non-existent cubin file returns None."""
        result = dump_sass(Path("/nonexistent/file.cubin"))
        self.assertIsNone(result)

    @patch("cutracer.query.sass.subprocess.run")
    def test_nvdisasm_not_found_returns_none(self, mock_run: MagicMock) -> None:
        """nvdisasm not in PATH returns None."""
        mock_run.side_effect = FileNotFoundError()
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(Path(f.name))
            self.assertIsNone(result)

    @patch("cutracer.query.sass.subprocess.run")
    def test_nvdisasm_timeout_returns_none(self, mock_run: MagicMock) -> None:
        """nvdisasm timeout returns None."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvdisasm", timeout=60)
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(Path(f.name))
            self.assertIsNone(result)

    @patch("cutracer.query.sass.subprocess.run")
    def test_nvdisasm_nonzero_exit_returns_none(self, mock_run: MagicMock) -> None:
        """nvdisasm non-zero exit code returns None."""
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(Path(f.name))
            self.assertIsNone(result)

    @patch("cutracer.query.sass.subprocess.run")
    def test_success_returns_sass_output(self, mock_run: MagicMock) -> None:
        """Successful nvdisasm returns SassOutput."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/* 0x0000 */ MOV R1, R2;\n/* 0x0008 */ ADD R3, R4;",
            stderr="",
        )
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(Path(f.name))
            self.assertIsNotNone(result)
            self.assertIn("MOV R1, R2", result.raw_text)
            self.assertEqual(result.cubin_path, Path(f.name))

    @patch("cutracer.query.sass.subprocess.run")
    def test_default_flags(self, mock_run: MagicMock) -> None:
        """Default invocation includes -g and -c flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(Path(f.name))
            self.assertIsNotNone(result)
            self.assertEqual(result.flags_used, ["-g", "-c"])
            # Check that subprocess was called with correct flags
            call_args = mock_run.call_args
            self.assertIn("-g", call_args[0][0])
            self.assertIn("-c", call_args[0][0])

    @patch("cutracer.query.sass.subprocess.run")
    def test_no_source_info_flag(self, mock_run: MagicMock) -> None:
        """include_source_info=False omits -g flag."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(Path(f.name), include_source_info=False)
            self.assertIsNotNone(result)
            self.assertEqual(result.flags_used, ["-c"])
            call_args = mock_run.call_args
            self.assertNotIn("-g", call_args[0][0])

    @patch("cutracer.query.sass.subprocess.run")
    def test_no_line_info_flag(self, mock_run: MagicMock) -> None:
        """include_line_info=False omits -c flag."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(Path(f.name), include_line_info=False)
            self.assertIsNotNone(result)
            self.assertEqual(result.flags_used, ["-g"])
            call_args = mock_run.call_args
            self.assertNotIn("-c", call_args[0][0])

    @patch("cutracer.query.sass.subprocess.run")
    def test_minimal_flags(self, mock_run: MagicMock) -> None:
        """Both flags=False results in no extra flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            result = dump_sass(
                Path(f.name), include_source_info=False, include_line_info=False
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.flags_used, [])

    @patch("cutracer.query.sass.subprocess.run")
    def test_timeout_parameter(self, mock_run: MagicMock) -> None:
        """timeout parameter is passed to subprocess."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            dump_sass(Path(f.name), timeout=120)
            call_kwargs = mock_run.call_args[1]
            self.assertEqual(call_kwargs["timeout"], 120)


class TestDumpSassToFile(unittest.TestCase):
    """Tests for dump_sass_to_file() function."""

    @patch("cutracer.query.sass.subprocess.run")
    def test_default_output_path(self, mock_run: MagicMock) -> None:
        """Default output path replaces .cubin with .sass."""
        mock_run.return_value = MagicMock(returncode=0, stdout="MOV R1, R2", stderr="")
        with tempfile.TemporaryDirectory() as tmpdir:
            cubin_path = Path(tmpdir) / "kernel.cubin"
            cubin_path.write_bytes(b"")  # Create empty file

            result = dump_sass_to_file(cubin_path)

            expected_path = Path(tmpdir) / "kernel.sass"
            self.assertEqual(result, expected_path)
            self.assertTrue(expected_path.exists())
            self.assertEqual(expected_path.read_text(), "MOV R1, R2")

    @patch("cutracer.query.sass.subprocess.run")
    def test_explicit_output_path(self, mock_run: MagicMock) -> None:
        """Explicit output path is used."""
        mock_run.return_value = MagicMock(returncode=0, stdout="MOV R1, R2", stderr="")
        with tempfile.TemporaryDirectory() as tmpdir:
            cubin_path = Path(tmpdir) / "kernel.cubin"
            cubin_path.write_bytes(b"")
            output_path = Path(tmpdir) / "output" / "custom.sass"

            result = dump_sass_to_file(cubin_path, output_path=output_path)

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())

    def test_failure_returns_none(self) -> None:
        """Failed dump returns None."""
        result = dump_sass_to_file(Path("/nonexistent.cubin"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
