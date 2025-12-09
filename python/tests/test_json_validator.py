# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for JSON validator module.
"""

import json
from pathlib import Path

import pytest

from cutracer.validation.json_validator import (
    ValidationError,
    validate_json_schema,
    validate_json_syntax,
    validate_json_trace,
)


class TestValidateJsonSyntax:
    """Tests for validate_json_syntax function."""

    def test_valid_ndjson_file(self, valid_ndjson_file: Path) -> None:
        """Test validation of a valid NDJSON file."""
        valid_count, errors = validate_json_syntax(valid_ndjson_file)

        assert valid_count == 3
        assert len(errors) == 0

    def test_invalid_json_syntax(self, invalid_json_syntax_file: Path) -> None:
        """Test detection of invalid JSON syntax."""
        valid_count, errors = validate_json_syntax(invalid_json_syntax_file)

        assert valid_count == 2  # 2 valid lines
        assert len(errors) == 1  # 1 error
        assert "Line 2" in errors[0]

    def test_empty_file(self, empty_file: Path) -> None:
        """Test validation of an empty file."""
        valid_count, errors = validate_json_syntax(empty_file)

        assert valid_count == 0
        assert len(errors) == 0

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Test handling of non-existent file."""
        non_existent = temp_dir / "non_existent.ndjson"

        with pytest.raises(FileNotFoundError):
            validate_json_syntax(non_existent)

    def test_file_with_empty_lines(self, temp_dir: Path) -> None:
        """Test that empty lines are skipped."""
        filepath = temp_dir / "with_empty_lines.ndjson"
        content = '{"type": "reg_trace"}\n\n\n{"type": "reg_trace"}\n'
        filepath.write_text(content)

        valid_count, errors = validate_json_syntax(filepath)

        assert valid_count == 2
        assert len(errors) == 0

    def test_whitespace_only_lines(self, temp_dir: Path) -> None:
        """Test that whitespace-only lines are skipped."""
        filepath = temp_dir / "whitespace_lines.ndjson"
        content = '{"type": "reg_trace"}\n   \n\t\n{"type": "reg_trace"}\n'
        filepath.write_text(content)

        valid_count, errors = validate_json_syntax(filepath)

        assert valid_count == 2
        assert len(errors) == 0


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_valid_reg_trace_schema(self, valid_ndjson_file: Path) -> None:
        """Test validation of valid reg_trace records against schema."""
        result = validate_json_schema(valid_ndjson_file, message_type="reg_trace")

        assert result is True

    def test_invalid_schema_missing_fields(self, temp_dir: Path) -> None:
        """Test detection of missing required fields."""
        filepath = temp_dir / "missing_fields.ndjson"
        record = {"type": "reg_trace", "ctx": "0x5a1100"}  # Missing required fields
        filepath.write_text(json.dumps(record) + "\n")

        with pytest.raises(ValidationError) as exc_info:
            validate_json_schema(filepath, message_type="reg_trace")

        assert "Schema validation failed" in str(exc_info.value)

    def test_invalid_message_type(self, valid_ndjson_file: Path) -> None:
        """Test handling of unknown message type."""
        with pytest.raises(ValueError) as exc_info:
            validate_json_schema(valid_ndjson_file, message_type="unknown_type")

        assert "Unknown message type" in str(exc_info.value)

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Test handling of non-existent file."""
        non_existent = temp_dir / "non_existent.ndjson"

        with pytest.raises(FileNotFoundError):
            validate_json_schema(non_existent, message_type="reg_trace")

    def test_type_mismatch(self, temp_dir: Path) -> None:
        """Test detection of type field mismatch."""
        filepath = temp_dir / "type_mismatch.ndjson"
        # Create a mem_trace record but validate against reg_trace schema
        from .conftest import VALID_MEM_TRACE_RECORD

        filepath.write_text(json.dumps(VALID_MEM_TRACE_RECORD) + "\n")

        with pytest.raises(ValidationError) as exc_info:
            validate_json_schema(filepath, message_type="reg_trace")

        assert "Expected type 'reg_trace'" in str(exc_info.value)

    def test_valid_mem_trace_schema(self, temp_dir: Path) -> None:
        """Test validation of valid mem_trace records against schema."""
        filepath = temp_dir / "mem_trace.ndjson"
        from .conftest import VALID_MEM_TRACE_RECORD

        filepath.write_text(json.dumps(VALID_MEM_TRACE_RECORD) + "\n")

        result = validate_json_schema(filepath, message_type="mem_trace")

        assert result is True

    def test_valid_opcode_only_schema(self, temp_dir: Path) -> None:
        """Test validation of valid opcode_only records against schema."""
        filepath = temp_dir / "opcode_only.ndjson"
        from .conftest import VALID_OPCODE_ONLY_RECORD

        filepath.write_text(json.dumps(VALID_OPCODE_ONLY_RECORD) + "\n")

        result = validate_json_schema(filepath, message_type="opcode_only")

        assert result is True


class TestValidateJsonTrace:
    """Tests for validate_json_trace function."""

    def test_valid_trace_file(self, valid_ndjson_file: Path) -> None:
        """Test complete validation of a valid trace file."""
        result = validate_json_trace(valid_ndjson_file)

        assert result["valid"] is True
        assert result["record_count"] == 3
        assert result["message_type"] == "reg_trace"
        assert result["file_size"] > 0
        assert len(result["errors"]) == 0

    def test_empty_file(self, empty_file: Path) -> None:
        """Test validation of an empty file."""
        result = validate_json_trace(empty_file)

        assert result["valid"] is False
        assert result["record_count"] == 0
        assert "No valid JSON records" in str(result["errors"])

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Test handling of non-existent file."""
        non_existent = temp_dir / "non_existent.ndjson"

        with pytest.raises(FileNotFoundError):
            validate_json_trace(non_existent)

    def test_invalid_syntax_returns_errors(
        self, invalid_json_syntax_file: Path
    ) -> None:
        """Test that syntax errors are reported."""
        result = validate_json_trace(invalid_json_syntax_file)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_unknown_message_type(self, temp_dir: Path) -> None:
        """Test handling of unknown message type in file."""
        filepath = temp_dir / "unknown_type.ndjson"
        record = {
            "type": "invalid_type",
            "ctx": "0x5a1100",
            "sass": "NOP ;",
            "trace_index": 0,
            "timestamp": 123456789,
        }
        filepath.write_text(json.dumps(record) + "\n")

        result = validate_json_trace(filepath)

        assert result["valid"] is False
        assert "Unknown message type" in str(result["errors"])

    def test_result_contains_file_size(self, valid_ndjson_file: Path) -> None:
        """Test that result includes correct file size."""
        result = validate_json_trace(valid_ndjson_file)

        expected_size = valid_ndjson_file.stat().st_size
        assert result["file_size"] == expected_size

    def test_auto_detects_message_type(self, temp_dir: Path) -> None:
        """Test auto-detection of message type."""
        filepath = temp_dir / "mem_trace.ndjson"
        from .conftest import VALID_MEM_TRACE_RECORD

        filepath.write_text(json.dumps(VALID_MEM_TRACE_RECORD) + "\n")

        result = validate_json_trace(filepath)

        assert result["valid"] is True
        assert result["message_type"] == "mem_trace"


class TestRealTraceFiles:
    """Integration tests with real trace files."""

    def test_real_ndjson_trace_syntax(self, real_ndjson_trace: Path) -> None:
        """Test JSON syntax validation on real trace file."""
        valid_count, errors = validate_json_syntax(real_ndjson_trace)

        assert valid_count > 0
        assert len(errors) == 0

    def test_real_ndjson_trace_full(self, real_ndjson_trace: Path) -> None:
        """Test full validation on real trace file."""
        result = validate_json_trace(real_ndjson_trace)

        assert result["valid"] is True
        assert result["record_count"] > 0
        assert result["message_type"] in ["reg_trace", "mem_trace", "opcode_only"]
