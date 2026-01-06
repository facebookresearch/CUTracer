# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for JSON schema definitions.
"""

import json
import unittest

import jsonschema
from cutracer.validation.schema_loader import (
    MEM_ACCESS_SCHEMA,
    OPCODE_ONLY_SCHEMA,
    REG_INFO_SCHEMA,
    SCHEMAS_BY_TYPE,
)
from tests.test_base import REG_TRACE_NDJSON


class SchemaTest(unittest.TestCase):
    """Core tests for JSON schema definitions."""

    def test_schemas_contain_all_types(self):
        """Test that SCHEMAS_BY_TYPE contains all expected message types."""
        expected_types = ["reg_trace", "mem_trace", "opcode_only"]
        for msg_type in expected_types:
            self.assertIn(msg_type, SCHEMAS_BY_TYPE)

    def test_schemas_are_valid_json_schema(self):
        """Test that all schemas are valid JSON Schema documents."""
        for schema in SCHEMAS_BY_TYPE.values():
            jsonschema.Draft7Validator.check_schema(schema)

    def test_real_reg_trace_record_passes(self):
        """Test that real reg_trace records from sample file pass schema validation."""
        with open(REG_TRACE_NDJSON, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:  # Test first 10 records
                    break
                record = json.loads(line.strip())
                # Should not raise any exception
                jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_mem_trace_schema_structure(self):
        """Test that mem_trace schema has required structure."""
        self.assertIn("properties", MEM_ACCESS_SCHEMA)
        self.assertIn("type", MEM_ACCESS_SCHEMA["properties"])
        self.assertIn("addrs", MEM_ACCESS_SCHEMA["properties"])

    def test_opcode_only_schema_structure(self):
        """Test that opcode_only schema has required structure."""
        self.assertIn("properties", OPCODE_ONLY_SCHEMA)
        self.assertIn("type", OPCODE_ONLY_SCHEMA["properties"])
        # opcode_only should NOT have regs or addrs
        self.assertNotIn("regs", OPCODE_ONLY_SCHEMA.get("required", []))
        self.assertNotIn("addrs", OPCODE_ONLY_SCHEMA.get("required", []))


if __name__ == "__main__":
    unittest.main()
