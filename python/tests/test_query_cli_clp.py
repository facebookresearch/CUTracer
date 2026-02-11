# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for query CLI command.
"""

import unittest

from click.testing import CliRunner
from cutracer.cli import main
from tests.test_base import BaseValidationTest, REG_TRACE_CLP


class TestQueryCommand(BaseValidationTest):
    """Tests for query CLI command."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()
    
    def test_analyze_clp_qrchive(self):
        """Test analyze with CLP-compressed file."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_CLP), "--head", "5"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 6)


if __name__ == "__main__":
    unittest.main()
