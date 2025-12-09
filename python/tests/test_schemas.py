# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for JSON schema definitions.
"""

import jsonschema
import pytest

from cutracer.validation.schemas import (
    MEM_ACCESS_SCHEMA,
    OPCODE_ONLY_SCHEMA,
    REG_INFO_SCHEMA,
    SCHEMAS_BY_TYPE,
)

from .conftest import (
    VALID_REG_TRACE_RECORD,
    VALID_MEM_TRACE_RECORD,
    VALID_OPCODE_ONLY_RECORD,
)


class TestSchemasByType:
    """Tests for SCHEMAS_BY_TYPE mapping."""

    def test_contains_all_types(self) -> None:
        """Test that mapping contains all expected message types."""
        expected_types = ["reg_trace", "mem_trace", "opcode_only"]

        for msg_type in expected_types:
            assert msg_type in SCHEMAS_BY_TYPE, \
                f"Missing message type: {msg_type}"

    def test_schemas_are_valid(self) -> None:
        """Test that all schemas are valid JSON Schema documents."""
        for msg_type, schema in SCHEMAS_BY_TYPE.items():
            # This will raise if schema is invalid
            jsonschema.Draft7Validator.check_schema(schema)


class TestRegInfoSchema:
    """Tests for REG_INFO_SCHEMA."""

    def test_valid_record_passes(self) -> None:
        """Test that a valid reg_trace record passes schema validation."""
        # Should not raise any exception
        jsonschema.validate(VALID_REG_TRACE_RECORD, REG_INFO_SCHEMA)

    def test_required_fields(self) -> None:
        """Test that all required fields must be present."""
        required_fields = [
            "type", "ctx", "sass", "trace_index", "timestamp",
            "grid_launch_id", "cta", "warp", "opcode_id", "pc", "regs"
        ]

        for field in required_fields:
            # Create a valid record
            record = dict(VALID_REG_TRACE_RECORD)

            # Remove one required field
            del record[field]

            # Should fail validation
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_invalid_ctx_format(self) -> None:
        """Test that invalid ctx format is rejected."""
        record = dict(VALID_REG_TRACE_RECORD)
        record["ctx"] = "invalid"  # Not a hex address

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_invalid_type_value(self) -> None:
        """Test that invalid type value is rejected."""
        record = dict(VALID_REG_TRACE_RECORD)
        record["type"] = "wrong_type"

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_cta_must_be_array_of_3(self) -> None:
        """Test that cta must be an array of exactly 3 integers."""
        record = dict(VALID_REG_TRACE_RECORD)

        # Too few elements
        record["cta"] = [0, 0]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

        # Too many elements
        record["cta"] = [0, 0, 0, 0]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_regs_must_be_2d_array(self) -> None:
        """Test that regs must be a 2D array."""
        record = dict(VALID_REG_TRACE_RECORD)

        # 1D array instead of 2D
        record["regs"] = [0, 1, 2, 3]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_optional_uregs_field(self) -> None:
        """Test that uregs is optional."""
        record = dict(VALID_REG_TRACE_RECORD)

        # Without uregs - should pass
        jsonschema.validate(record, REG_INFO_SCHEMA)

        # With uregs - should also pass
        record["uregs"] = [1, 2, 3, 4]
        jsonschema.validate(record, REG_INFO_SCHEMA)


class TestMemAccessSchema:
    """Tests for MEM_ACCESS_SCHEMA."""

    def test_valid_record_passes(self) -> None:
        """Test that a valid mem_trace record passes schema validation."""
        jsonschema.validate(VALID_MEM_TRACE_RECORD, MEM_ACCESS_SCHEMA)

    def test_required_fields(self) -> None:
        """Test that all required fields must be present."""
        required_fields = [
            "type", "ctx", "sass", "trace_index", "timestamp",
            "grid_launch_id", "cta", "warp", "opcode_id", "pc", "addrs"
        ]

        for field in required_fields:
            record = dict(VALID_MEM_TRACE_RECORD)
            del record[field]

            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(record, MEM_ACCESS_SCHEMA)

    def test_addrs_must_have_32_elements(self) -> None:
        """Test that addrs must have exactly 32 elements."""
        record = dict(VALID_MEM_TRACE_RECORD)

        # Too few addresses
        record["addrs"] = [0x7f0000000000] * 16
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, MEM_ACCESS_SCHEMA)

        # Too many addresses
        record["addrs"] = [0x7f0000000000] * 64
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, MEM_ACCESS_SCHEMA)

    def test_invalid_type_value(self) -> None:
        """Test that invalid type value is rejected."""
        record = dict(VALID_MEM_TRACE_RECORD)
        record["type"] = "reg_trace"  # Wrong type

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, MEM_ACCESS_SCHEMA)


class TestOpcodeOnlySchema:
    """Tests for OPCODE_ONLY_SCHEMA."""

    def test_valid_record_passes(self) -> None:
        """Test that a valid opcode_only record passes schema validation."""
        jsonschema.validate(VALID_OPCODE_ONLY_RECORD, OPCODE_ONLY_SCHEMA)

    def test_required_fields(self) -> None:
        """Test that all required fields must be present."""
        required_fields = [
            "type", "ctx", "sass", "trace_index", "timestamp",
            "grid_launch_id", "cta", "warp", "opcode_id", "pc"
        ]

        for field in required_fields:
            record = dict(VALID_OPCODE_ONLY_RECORD)
            del record[field]

            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(record, OPCODE_ONLY_SCHEMA)

    def test_does_not_require_regs_or_addrs(self) -> None:
        """Test that opcode_only doesn't require regs or addrs fields."""
        record = dict(VALID_OPCODE_ONLY_RECORD)

        # Ensure no regs or addrs
        assert "regs" not in record
        assert "addrs" not in record

        # Should still pass
        jsonschema.validate(record, OPCODE_ONLY_SCHEMA)

    def test_invalid_type_value(self) -> None:
        """Test that invalid type value is rejected."""
        record = dict(VALID_OPCODE_ONLY_RECORD)
        record["type"] = "reg_trace"  # Wrong type

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, OPCODE_ONLY_SCHEMA)


class TestSchemaFieldTypes:
    """Tests for field type validation across schemas."""

    def test_trace_index_must_be_integer(self) -> None:
        """Test that trace_index must be an integer."""
        record = dict(VALID_REG_TRACE_RECORD)
        record["trace_index"] = "0"  # String instead of int

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_timestamp_must_be_integer(self) -> None:
        """Test that timestamp must be an integer."""
        record = dict(VALID_REG_TRACE_RECORD)
        record["timestamp"] = "12345"  # String instead of int

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_warp_must_be_non_negative(self) -> None:
        """Test that warp must be non-negative."""
        record = dict(VALID_REG_TRACE_RECORD)
        record["warp"] = -1

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_pc_must_be_non_negative(self) -> None:
        """Test that pc must be non-negative."""
        record = dict(VALID_REG_TRACE_RECORD)
        record["pc"] = -1

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)

    def test_sass_must_be_non_empty(self) -> None:
        """Test that sass must be non-empty string."""
        record = dict(VALID_REG_TRACE_RECORD)
        record["sass"] = ""

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REG_INFO_SCHEMA)
