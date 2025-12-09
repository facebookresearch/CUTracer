# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for text validator module.
"""

from pathlib import Path

import pytest

from cutracer.validation.text_validator import (
    REG_INFO_HEADER_PATTERN,
    REGISTER_VALUE_PATTERN,
    parse_text_trace_record,
    validate_text_format,
    validate_text_trace,
)


class TestRegexPatterns:
    """Tests for regex patterns used in text validation."""

    def test_reg_info_header_pattern_matches_valid(self) -> None:
        """Test that REG_INFO_HEADER_PATTERN matches valid headers."""
        valid_headers = [
            "CTX 0x5a46d0 - CTA 0,0,0 - warp 0 - LDC R1, c[0x0][0x28] ;:",
            "CTX 0xABCDEF - CTA 1,2,3 - warp 10 - ADD R2, R1, R0 ;:",
            "CTX 0x123456 - CTA 100,200,300 - warp 31 - EXIT ;:",
        ]
        for header in valid_headers:
            assert REG_INFO_HEADER_PATTERN.match(header) is not None, \
                f"Pattern should match: {header}"

    def test_reg_info_header_pattern_rejects_invalid(self) -> None:
        """Test that REG_INFO_HEADER_PATTERN rejects invalid headers."""
        invalid_headers = [
            "CTX 0x5a46d0 - CTA 0,0,0 - warp 0 - LDC R1",  # Missing colon
            "CTA 0,0,0 - warp 0 - LDC R1 ;:",  # Missing CTX
            "CTX invalid - CTA 0,0,0 - warp 0 - LDC R1 ;:",  # Invalid hex
            "",  # Empty string
        ]
        for header in invalid_headers:
            assert REG_INFO_HEADER_PATTERN.match(header) is None, \
                f"Pattern should not match: {header}"

    def test_register_value_pattern_matches_valid(self) -> None:
        """Test that REGISTER_VALUE_PATTERN matches valid register lines."""
        valid_lines = [
            "  * Reg0_T00: 0x00000000 Reg0_T01: 0x00000001",
            "  * Reg0_T00: 0xDEADBEEF Reg0_T01: 0xCAFEBABE",
        ]
        for line in valid_lines:
            assert REGISTER_VALUE_PATTERN.match(line) is not None, \
                f"Pattern should match: {line}"


class TestValidateTextFormat:
    """Tests for validate_text_format function."""

    def test_valid_text_trace(self, valid_text_trace_file: Path) -> None:
        """Test validation of a valid text trace file."""
        result = validate_text_format(valid_text_trace_file)
        assert result is True

    def test_empty_file(self, empty_text_file: Path) -> None:
        """Test validation of an empty file - should not raise."""
        # Empty file is technically valid (no format errors)
        result = validate_text_format(empty_text_file)
        assert result is True

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Test handling of non-existent file."""
        non_existent = temp_dir / "non_existent.log"
        with pytest.raises(FileNotFoundError):
            validate_text_format(non_existent)

    def test_valid_header_only_file(self, temp_dir: Path) -> None:
        """Test file with valid header but no register lines."""
        filepath = temp_dir / "header_only.log"
        content = "CTX 0x5a46d0 - CTA 0,0,0 - warp 0 - LDC R1, c[0x0][0x28] ;:\n"
        filepath.write_text(content)

        result = validate_text_format(filepath)
        assert result is True


class TestValidateTextTrace:
    """Tests for validate_text_trace function."""

    def test_valid_trace_file(self, valid_text_trace_file: Path) -> None:
        """Test complete validation of a valid text trace file."""
        result = validate_text_trace(valid_text_trace_file)

        assert result["valid"] is True
        assert result["record_count"] == 3  # 3 header lines
        assert result["file_size"] > 0
        assert len(result["errors"]) == 0

    def test_empty_file(self, empty_text_file: Path) -> None:
        """Test validation of an empty file."""
        result = validate_text_trace(empty_text_file)

        assert result["valid"] is False
        assert result["record_count"] == 0
        assert "No trace records" in str(result["errors"])

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Test handling of non-existent file."""
        non_existent = temp_dir / "non_existent.log"

        with pytest.raises(FileNotFoundError):
            validate_text_trace(non_existent)

    def test_result_contains_file_size(self, valid_text_trace_file: Path) -> None:
        """Test that result includes correct file size."""
        result = validate_text_trace(valid_text_trace_file)

        expected_size = valid_text_trace_file.stat().st_size
        assert result["file_size"] == expected_size

    def test_counts_multiple_records(self, temp_dir: Path) -> None:
        """Test that multiple records are counted correctly."""
        filepath = temp_dir / "multi_record.log"
        content = """CTX 0x5a46d0 - CTA 0,0,0 - warp 0 - LDC R1, c[0x0][0x28] ;:
  * Reg0_T00: 0x00000000

CTX 0x5a46d0 - CTA 0,0,0 - warp 1 - ADD R2, R1, R0 ;:
  * Reg0_T00: 0x00000001

CTX 0x5a46d0 - CTA 0,0,0 - warp 2 - EXIT ;:
  * Reg0_T00: 0x00000002

CTX 0x5a46d0 - CTA 1,0,0 - warp 3 - NOP ;:
  * Reg0_T00: 0x00000003

CTX 0x5a46d0 - CTA 1,0,0 - warp 4 - RET ;:
  * Reg0_T00: 0x00000004
"""
        filepath.write_text(content)

        result = validate_text_trace(filepath)

        assert result["valid"] is True
        assert result["record_count"] == 5


class TestParseTextTraceRecord:
    """Tests for parse_text_trace_record function."""

    def test_parse_reg_info_header(self) -> None:
        """Test parsing of reg_info header line."""
        lines = ["CTX 0x5a46d0 - CTA 1,2,3 - warp 5 - LDC R1, c[0x0][0x28] ;:"]

        result = parse_text_trace_record(lines)

        assert result["ctx"] == "0x5a46d0"
        assert result["cta"] == [1, 2, 3]
        assert result["warp"] == 5
        assert result["sass"] == "LDC R1, c[0x0][0x28] ;"
        assert result["record_type"] == "reg_info"

    def test_parse_empty_lines_raises(self) -> None:
        """Test that empty lines raise ValueError."""
        with pytest.raises(ValueError, match="Empty record"):
            parse_text_trace_record([])

    def test_parse_invalid_header_raises(self) -> None:
        """Test that invalid header format raises ValueError."""
        lines = ["This is not a valid header"]

        with pytest.raises(ValueError, match="Unrecognized header format"):
            parse_text_trace_record(lines)

    def test_parse_various_cta_coordinates(self) -> None:
        """Test parsing of various CTA coordinates."""
        test_cases = [
            ("CTX 0x123 - CTA 0,0,0 - warp 0 - NOP ;:", [0, 0, 0]),
            ("CTX 0x456 - CTA 10,20,30 - warp 1 - NOP ;:", [10, 20, 30]),
            ("CTX 0x789 - CTA 100,200,300 - warp 31 - NOP ;:", [100, 200, 300]),
        ]

        for header, expected_cta in test_cases:
            result = parse_text_trace_record([header])
            assert result["cta"] == expected_cta, f"Failed for header: {header}"

    def test_parse_various_warp_ids(self) -> None:
        """Test parsing of various warp IDs."""
        test_cases = [
            ("CTX 0x123 - CTA 0,0,0 - warp 0 - NOP ;:", 0),
            ("CTX 0x456 - CTA 0,0,0 - warp 15 - NOP ;:", 15),
            ("CTX 0x789 - CTA 0,0,0 - warp 31 - NOP ;:", 31),
        ]

        for header, expected_warp in test_cases:
            result = parse_text_trace_record([header])
            assert result["warp"] == expected_warp, f"Failed for header: {header}"


class TestRealTraceFiles:
    """Integration tests with real trace files."""

    def test_real_text_trace_format(self, real_text_trace: Path) -> None:
        """Test format validation on real text trace file."""
        result = validate_text_format(real_text_trace)
        assert result is True

    def test_real_text_trace_full(self, real_text_trace: Path) -> None:
        """Test full validation on real text trace file."""
        result = validate_text_trace(real_text_trace)

        assert result["valid"] is True
        assert result["record_count"] > 0
        assert result["file_size"] > 0
