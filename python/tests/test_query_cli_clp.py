# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for query CLI command.
"""

import unittest

import yscope_clp_core
from click.testing import CliRunner
from cutracer.cli import main
from tests.test_base import BaseValidationTest, REG_TRACE_NDJSON


class TestQueryCommand(BaseValidationTest):
    """Tests for query CLI command."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()
        # create clp archive from ndjson
        self.clp_archive_path = self.temp_dir.joinpath("example_input.clp")
        with yscope_clp_core.open_archive(clp_archive_name, "w") as clp_archive:
            clp_archive.add(REG_TRACE_NDJSON)

    def test_analyze_clp_qrchive(self):
        """Test analyze with CLP-compressed file."""
        result = self.runner.invoke(
            main, ["query", self.clp_archive_path.absolute(), "--head", "5"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 6)

if __name__ == "__main__":
    unittest.main()
