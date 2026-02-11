# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Base test class and shared test data for CUTracer validation tests.
"""

import shutil
import tempfile
import unittest
from pathlib import Path


# Example inputs directory
EXAMPLE_INPUTS_DIR = Path(__file__).parent / "example_inputs"

# Real data file paths
REG_TRACE_NDJSON = EXAMPLE_INPUTS_DIR / "reg_trace_sample.ndjson"
REG_TRACE_NDJSON_ZST = EXAMPLE_INPUTS_DIR / "reg_trace_sample.ndjson.zst"
REG_TRACE_CLP = EXAMPLE_INPUTS_DIR / "reg_trace_sample.clp"
REG_TRACE_LOG = EXAMPLE_INPUTS_DIR / "reg_trace_sample.log"
INVALID_SYNTAX_NDJSON = EXAMPLE_INPUTS_DIR / "invalid_syntax.ndjson"
INVALID_SCHEMA_NDJSON = EXAMPLE_INPUTS_DIR / "invalid_schema.ndjson"

# Expected record counts for sample files
REG_TRACE_NDJSON_RECORD_COUNT = 100
REG_TRACE_NDJSON_ZST_RECORD_COUNT = 100
REG_TRACE_LOG_RECORD_COUNT = 67


class BaseValidationTest(unittest.TestCase):
    """Base class for validation tests with shared setup/teardown."""

    @classmethod
    def setUpClass(cls):
        """Verify example input files exist."""
        required_files = [
            REG_TRACE_NDJSON,
            REG_TRACE_LOG,
            INVALID_SYNTAX_NDJSON,
            INVALID_SCHEMA_NDJSON,
        ]
        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"Example input file not found: {path}")

    def setUp(self):
        """Create a temporary directory for test files that need to be generated."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def create_temp_file(self, filename: str, content: str) -> Path:
        """Create a temporary file with given content."""
        filepath = self.temp_dir / filename
        filepath.write_text(content)
        return filepath
