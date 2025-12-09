# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for consistency checker module.
"""

import json
from pathlib import Path

import pytest

from cutracer.validation.consistency import (
    compare_record_counts,
    compare_trace_content,
    compare_trace_formats,
    get_trace_statistics,
)


class TestCompareRecordCounts:
    """Tests for compare_record_counts function."""

    def test_exact_match(self) -> None:
        """Test that exact count match returns True."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 100}

        result = compare_record_counts(text_metadata, json_metadata)

        assert result is True

    def test_within_tolerance(self) -> None:
        """Test that counts within tolerance return True."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 105}  # 5% difference

        result = compare_record_counts(text_metadata, json_metadata, tolerance=0.1)

        assert result is True

    def test_outside_tolerance(self) -> None:
        """Test that counts outside tolerance return False."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 150}  # 50% difference

        result = compare_record_counts(text_metadata, json_metadata, tolerance=0.1)

        assert result is False

    def test_both_zero(self) -> None:
        """Test that both zero counts return True."""
        text_metadata = {"record_count": 0}
        json_metadata = {"record_count": 0}

        result = compare_record_counts(text_metadata, json_metadata)

        assert result is True

    def test_one_zero(self) -> None:
        """Test that one zero count returns False."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 0}

        result = compare_record_counts(text_metadata, json_metadata)

        assert result is False

    def test_missing_record_count_raises(self) -> None:
        """Test that missing record_count field raises ValueError."""
        with pytest.raises(ValueError, match="text_metadata missing"):
            compare_record_counts({}, {"record_count": 100})

        with pytest.raises(ValueError, match="json_metadata missing"):
            compare_record_counts({"record_count": 100}, {})

    def test_custom_tolerance(self) -> None:
        """Test with custom tolerance values."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 120}  # 20% difference

        # Should fail with 10% tolerance
        assert compare_record_counts(
            text_metadata, json_metadata, tolerance=0.1
        ) is False

        # Should pass with 25% tolerance
        assert compare_record_counts(
            text_metadata, json_metadata, tolerance=0.25
        ) is True


class TestGetTraceStatistics:
    """Tests for get_trace_statistics function."""

    def test_ndjson_statistics(self, valid_ndjson_file: Path) -> None:
        """Test statistics extraction from NDJSON file."""
        stats = get_trace_statistics(valid_ndjson_file)

        assert stats["format"] == "json"
        assert stats["record_count"] == 3
        assert stats["file_size"] > 0
        assert "reg_trace" in stats["message_types"]
        assert stats["unique_ctxs"] >= 1
        assert stats["unique_warps"] >= 1

    def test_text_statistics(self, valid_text_trace_file: Path) -> None:
        """Test statistics extraction from text file."""
        stats = get_trace_statistics(valid_text_trace_file)

        assert stats["format"] == "text"
        assert stats["record_count"] == 3
        assert stats["file_size"] > 0
        assert "reg_info" in stats["message_types"]
        assert stats["unique_ctxs"] >= 1
        assert stats["unique_warps"] >= 1

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Test handling of non-existent file."""
        non_existent = temp_dir / "non_existent.ndjson"

        with pytest.raises(FileNotFoundError):
            get_trace_statistics(non_existent)

    def test_unknown_format_raises(self, temp_dir: Path) -> None:
        """Test that unknown file extension raises ValueError."""
        filepath = temp_dir / "unknown.xyz"
        filepath.write_text("some content")

        with pytest.raises(ValueError, match="Unknown file format"):
            get_trace_statistics(filepath)

    def test_message_type_counts(self, temp_dir: Path) -> None:
        """Test that message types are counted correctly."""
        filepath = temp_dir / "multi_type.ndjson"
        from .conftest import (
            VALID_REG_TRACE_RECORD,
            VALID_MEM_TRACE_RECORD,
            VALID_OPCODE_ONLY_RECORD,
        )

        records = [
            VALID_REG_TRACE_RECORD,
            VALID_REG_TRACE_RECORD,
            VALID_MEM_TRACE_RECORD,
            VALID_OPCODE_ONLY_RECORD,
        ]
        with open(filepath, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        stats = get_trace_statistics(filepath)

        assert stats["message_types"]["reg_trace"] == 2
        assert stats["message_types"]["mem_trace"] == 1
        assert stats["message_types"]["opcode_only"] == 1


class TestCompareTraceContent:
    """Tests for compare_trace_content function."""

    def test_file_not_found_text(self, temp_dir: Path, valid_ndjson_file: Path) -> None:
        """Test handling of non-existent text file."""
        non_existent = temp_dir / "non_existent.log"

        with pytest.raises(FileNotFoundError):
            compare_trace_content(non_existent, valid_ndjson_file)

    def test_file_not_found_json(self, temp_dir: Path, valid_text_trace_file: Path) -> None:
        """Test handling of non-existent JSON file."""
        non_existent = temp_dir / "non_existent.ndjson"

        with pytest.raises(FileNotFoundError):
            compare_trace_content(valid_text_trace_file, non_existent)

    def test_empty_files_inconsistent(self, temp_dir: Path) -> None:
        """Test that empty files are reported as inconsistent."""
        text_file = temp_dir / "empty.log"
        json_file = temp_dir / "empty.ndjson"
        text_file.touch()
        json_file.touch()

        result = compare_trace_content(text_file, json_file)

        assert result["consistent"] is False
        assert result["samples_compared"] == 0
        assert len(result["differences"]) > 0


class TestCompareTraceFormats:
    """Tests for compare_trace_formats function."""

    def test_file_not_found_text(self, temp_dir: Path, valid_ndjson_file: Path) -> None:
        """Test handling of non-existent text file."""
        non_existent = temp_dir / "non_existent.log"

        with pytest.raises(FileNotFoundError):
            compare_trace_formats(non_existent, valid_ndjson_file)

    def test_file_not_found_json(self, temp_dir: Path, valid_text_trace_file: Path) -> None:
        """Test handling of non-existent JSON file."""
        non_existent = temp_dir / "non_existent.ndjson"

        with pytest.raises(FileNotFoundError):
            compare_trace_formats(valid_text_trace_file, non_existent)

    def test_result_structure(
        self, valid_text_trace_file: Path, valid_ndjson_file: Path
    ) -> None:
        """Test that result contains all expected fields."""
        result = compare_trace_formats(valid_text_trace_file, valid_ndjson_file)

        assert "consistent" in result
        assert "record_count_match" in result
        assert "content_match" in result
        assert "text_records" in result
        assert "json_records" in result
        assert "samples_compared" in result
        assert "differences" in result

    def test_invalid_text_file(self, temp_dir: Path, valid_ndjson_file: Path) -> None:
        """Test handling of invalid text file."""
        text_file = temp_dir / "invalid.log"
        text_file.touch()  # Empty file

        result = compare_trace_formats(text_file, valid_ndjson_file)

        assert result["consistent"] is False
        assert "validation failed" in str(result["differences"]).lower()

    def test_invalid_json_file(
        self, temp_dir: Path, valid_text_trace_file: Path
    ) -> None:
        """Test handling of invalid JSON file."""
        json_file = temp_dir / "invalid.ndjson"
        json_file.touch()  # Empty file

        result = compare_trace_formats(valid_text_trace_file, json_file)

        assert result["consistent"] is False
        assert "validation failed" in str(result["differences"]).lower()


class TestRealTraceFiles:
    """Integration tests with real trace files."""

    def test_real_ndjson_statistics(self, real_ndjson_trace: Path) -> None:
        """Test statistics extraction from real NDJSON trace."""
        stats = get_trace_statistics(real_ndjson_trace)

        assert stats["format"] == "json"
        assert stats["record_count"] > 0
        assert stats["file_size"] > 0
        assert len(stats["message_types"]) >= 1

    def test_real_text_statistics(self, real_text_trace: Path) -> None:
        """Test statistics extraction from real text trace."""
        stats = get_trace_statistics(real_text_trace)

        assert stats["format"] == "text"
        assert stats["record_count"] > 0
        assert stats["file_size"] > 0
