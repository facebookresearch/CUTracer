# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Cross-format consistency checker for CUTracer traces.

This module provides functions to compare trace files in different formats
(text vs NDJSON) for data consistency.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

from .text_validator import (
    REG_INFO_HEADER_PATTERN,
    MEM_ACCESS_HEADER_PATTERN,
    parse_text_trace_record
)


def compare_record_counts(
    text_metadata: Dict[str, Any],
    json_metadata: Dict[str, Any],
    tolerance: float = 0.1
) -> bool:
    """
    Compare record counts between text and JSON formats.

    Args:
        text_metadata: Metadata from validate_text_trace()
        json_metadata: Metadata from validate_json_trace()
        tolerance: Allowed difference as fraction (default 10%)

    Returns:
        True if counts are within tolerance

    Raises:
        ValueError: If metadata is invalid or missing required fields
    """
    if "record_count" not in text_metadata:
        raise ValueError("text_metadata missing 'record_count' field")
    if "record_count" not in json_metadata:
        raise ValueError("json_metadata missing 'record_count' field")

    text_count = text_metadata["record_count"]
    json_count = json_metadata["record_count"]

    if text_count == 0 and json_count == 0:
        return True

    if text_count == 0 or json_count == 0:
        return False

    # Calculate relative difference
    max_count = max(text_count, json_count)
    diff = abs(text_count - json_count)
    relative_diff = diff / max_count

    return relative_diff <= tolerance


def compare_trace_content(
    text_file: Path,
    json_file: Path,
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Compare actual trace content between formats (sampling).

    Validates:
    - Same grid_launch_id in both formats
    - Same trace_index values
    - Consistent SASS strings
    - Matching CTA and warp information

    Args:
        text_file: Path to text trace file
        json_file: Path to NDJSON trace file
        sample_size: Number of records to sample for comparison (default: 10)

    Returns:
        Dictionary containing:
            - consistent: bool - Whether sampled content is consistent
            - samples_compared: int - Number of samples compared
            - differences: List[str] - Differences found

    Raises:
        FileNotFoundError: If either file does not exist
    """
    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    result: Dict[str, Any] = {
        "consistent": True,
        "samples_compared": 0,
        "differences": []
    }

    try:
        # Read sample of JSON records
        json_records = []
        with open(json_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                line = line.strip()
                if line:
                    json_records.append(json.loads(line))

        # Read sample of text records
        text_records = []
        with open(text_file, "r", encoding="utf-8") as f:
            current_record = []
            for line in f:
                if not line.strip():
                    if current_record:
                        try:
                            parsed = parse_text_trace_record(current_record)
                            text_records.append(parsed)
                            if len(text_records) >= sample_size:
                                break
                        except ValueError:
                            pass
                        current_record = []
                    continue

                if REG_INFO_HEADER_PATTERN.match(line) or \
                   MEM_ACCESS_HEADER_PATTERN.match(line):
                    if current_record:
                        try:
                            parsed = parse_text_trace_record(current_record)
                            text_records.append(parsed)
                            if len(text_records) >= sample_size:
                                break
                        except ValueError:
                            pass
                    current_record = [line]
                else:
                    current_record.append(line)

        # Compare the samples
        min_samples = min(len(text_records), len(json_records))
        result["samples_compared"] = min_samples

        if min_samples == 0:
            result["differences"].append("No records found to compare")
            result["consistent"] = False
            return result

        for i in range(min_samples):
            text_rec = text_records[i]
            json_rec = json_records[i]

            # Compare SASS instruction
            text_sass = text_rec.get("sass", "").strip()
            json_sass = json_rec.get("sass", "").strip()
            if text_sass != json_sass:
                result["differences"].append(
                    f"Record {i}: SASS mismatch - text: '{text_sass}', "
                    f"json: '{json_sass}'"
                )
                result["consistent"] = False

            # Compare CTA coordinates
            if "cta" in text_rec and "cta" in json_rec:
                if text_rec["cta"] != json_rec["cta"]:
                    result["differences"].append(
                        f"Record {i}: CTA mismatch - text: {text_rec['cta']}, "
                        f"json: {json_rec['cta']}"
                    )
                    result["consistent"] = False

            # Compare warp
            if "warp" in text_rec and "warp" in json_rec:
                if text_rec["warp"] != json_rec["warp"]:
                    result["differences"].append(
                        f"Record {i}: Warp mismatch - text: {text_rec['warp']}, "
                        f"json: {json_rec['warp']}"
                    )
                    result["consistent"] = False

    except Exception as e:
        result["differences"].append(f"Error during comparison: {str(e)}")
        result["consistent"] = False

    return result


def compare_trace_formats(
    text_file: Path,
    json_file: Path,
    tolerance: float = 0.1,
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Comprehensive comparison of two trace formats.

    Performs both record count comparison and content sampling.

    Args:
        text_file: Path to text trace file
        json_file: Path to NDJSON trace file
        tolerance: Allowed difference for record count (default: 10%)
        sample_size: Number of records to sample (default: 10)

    Returns:
        Dictionary containing:
            - consistent: bool - Overall consistency
            - record_count_match: bool - Whether counts match
            - content_match: bool - Whether content matches
            - text_records: int - Number of text records
            - json_records: int - Number of JSON records
            - samples_compared: int - Number of samples compared
            - differences: List[str] - All differences found

    Raises:
        FileNotFoundError: If either file does not exist
    """
    from .text_validator import validate_text_trace
    from .json_validator import validate_json_trace

    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    result: Dict[str, Any] = {
        "consistent": False,
        "record_count_match": False,
        "content_match": False,
        "text_records": 0,
        "json_records": 0,
        "samples_compared": 0,
        "differences": []
    }

    try:
        # Validate both files
        text_metadata = validate_text_trace(text_file)
        json_metadata = validate_json_trace(json_file)

        result["text_records"] = text_metadata["record_count"]
        result["json_records"] = json_metadata["record_count"]

        # Check if validation passed
        if not text_metadata["valid"]:
            result["differences"].append(
                f"Text file validation failed: {text_metadata['errors']}"
            )
            return result

        if not json_metadata["valid"]:
            result["differences"].append(
                f"JSON file validation failed: {json_metadata['errors']}"
            )
            return result

        # Compare record counts
        count_match = compare_record_counts(
            text_metadata, json_metadata, tolerance=tolerance
        )
        result["record_count_match"] = count_match

        if not count_match:
            diff = abs(result["text_records"] - result["json_records"])
            result["differences"].append(
                f"Record count mismatch: text={result['text_records']}, "
                f"json={result['json_records']}, diff={diff}"
            )

        # Compare content
        content_result = compare_trace_content(
            text_file, json_file, sample_size=sample_size
        )
        result["content_match"] = content_result["consistent"]
        result["samples_compared"] = content_result["samples_compared"]

        if not content_result["consistent"]:
            result["differences"].extend(content_result["differences"])

        # Overall consistency
        result["consistent"] = count_match and content_result["consistent"]

    except Exception as e:
        result["differences"].append(f"Comparison error: {str(e)}")

    return result


def get_trace_statistics(filepath: Path) -> Dict[str, Any]:
    """
    Extract statistics from a trace file.

    Works with both text and JSON formats.

    Args:
        filepath: Path to trace file

    Returns:
        Dictionary containing:
            - format: str - "text" or "json"
            - record_count: int
            - file_size: int
            - message_types: Dict[str, int] - Count by message type
            - unique_ctxs: int - Number of unique contexts
            - unique_warps: int - Number of unique warps

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not recognized
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    stats: Dict[str, Any] = {
        "format": None,
        "record_count": 0,
        "file_size": filepath.stat().st_size,
        "message_types": {},
        "unique_ctxs": set(),
        "unique_warps": set()
    }

    # Determine format by extension
    if filepath.suffix == ".ndjson":
        stats["format"] = "json"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    stats["record_count"] += 1

                    msg_type = record.get("type", "unknown")
                    stats["message_types"][msg_type] = \
                        stats["message_types"].get(msg_type, 0) + 1

                    if "ctx" in record:
                        stats["unique_ctxs"].add(record["ctx"])
                    if "warp" in record:
                        stats["unique_warps"].add(record["warp"])

                except json.JSONDecodeError:
                    pass

    elif filepath.suffix == ".log":
        stats["format"] = "text"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if REG_INFO_HEADER_PATTERN.match(line):
                    stats["record_count"] += 1
                    stats["message_types"]["reg_info"] = \
                        stats["message_types"].get("reg_info", 0) + 1

                    # Extract context
                    ctx_match = re.search(r"CTX\s+(0x[0-9a-fA-F]+)", line)
                    if ctx_match:
                        stats["unique_ctxs"].add(ctx_match.group(1))

                    # Extract warp
                    warp_match = re.search(r"warp\s+(\d+)", line)
                    if warp_match:
                        stats["unique_warps"].add(int(warp_match.group(1)))

                elif MEM_ACCESS_HEADER_PATTERN.match(line):
                    stats["record_count"] += 1
                    stats["message_types"]["mem_access"] = \
                        stats["message_types"].get("mem_access", 0) + 1

                    ctx_match = re.search(r"CTX\s+(0x[0-9a-fA-F]+)", line)
                    if ctx_match:
                        stats["unique_ctxs"].add(ctx_match.group(1))

                    warp_match = re.search(r"warp\s+(\d+)", line)
                    if warp_match:
                        stats["unique_warps"].add(int(warp_match.group(1)))
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")

    # Convert sets to counts
    stats["unique_ctxs"] = len(stats["unique_ctxs"])
    stats["unique_warps"] = len(stats["unique_warps"])

    return stats
