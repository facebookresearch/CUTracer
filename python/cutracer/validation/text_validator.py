# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Text format validator for CUTracer text trace files.

This module provides functions to validate text-format trace files (mode 0)
produced by CUTracer for format compliance.
"""

import re
from pathlib import Path
from typing import Any, Dict, List


class ValidationError(Exception):
    """Exception raised for validation errors."""

    pass


# Regex patterns for validating text trace format
# Pattern for header line (MSG_TYPE_REG_INFO)
REG_INFO_HEADER_PATTERN = re.compile(
    r"^CTX\s+0x[0-9a-fA-F]+\s+-\s+CTA\s+\d+,\d+,\d+\s+-\s+warp\s+\d+\s+-\s+.+:$"
)

# Pattern for register value line
REGISTER_VALUE_PATTERN = re.compile(r"^\s+\*\s+(Reg\d+_T\d+:\s+0x[0-9a-fA-F]+\s*)+$")

# Pattern for memory access header (MSG_TYPE_MEM_ACCESS)
MEM_ACCESS_HEADER_PATTERN = re.compile(
    r"^CTX\s+0x[0-9a-fA-F]+\s+-\s+kernel_launch_id\s+\d+\s+-\s+CTA\s+\d+,\d+,\d+\s+-\s+"
    r"warp\s+\d+\s+-\s+PC\s+\d+\s+-\s+.+:$"
)

# Pattern for memory addresses
MEMORY_ADDRESS_PATTERN = re.compile(r"T\d+:\s+0x[0-9a-fA-F]{16}")


def validate_text_format(filepath: Path) -> bool:
    """
    Validate text format of trace file.

    Checks for proper structure:
    - Header lines with CTX, CTA, warp, and SASS instruction
    - Register value lines or memory address lines
    - Proper hex formatting

    Args:
        filepath: Path to text trace file

    Returns:
        True if format is valid

    Raises:
        ValidationError: If format validation fails
        FileNotFoundError: If file does not exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    errors: List[str] = []
    line_num = 0
    last_was_header = False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                # Skip empty lines
                if not line.strip():
                    last_was_header = False
                    continue

                # Check if this is a header line
                is_reg_header = REG_INFO_HEADER_PATTERN.match(line)
                is_mem_header = MEM_ACCESS_HEADER_PATTERN.match(line)

                if is_reg_header or is_mem_header:
                    last_was_header = True
                    continue

                # Check if this is a register value line
                if REGISTER_VALUE_PATTERN.match(line):
                    if not last_was_header:
                        errors.append(
                            f"Line {line_num}: Register values without header"
                        )
                    continue

                # Check if this line contains memory addresses
                if "Memory Addresses:" in line:
                    continue

                if MEMORY_ADDRESS_PATTERN.search(line):
                    continue

                # If we get here, the line doesn't match any expected pattern
                # Only report if it's not whitespace-only or a separator
                if line.strip() and not line.strip().startswith("*"):
                    errors.append(f"Line {line_num}: Unrecognized format: {line[:50]}")

    except Exception as e:
        errors.append(f"File reading error: {str(e)}")

    if errors:
        max_errors = 10
        error_summary = "\n".join(errors[:max_errors])
        if len(errors) > max_errors:
            error_summary += (
                f"\n... (showing first {max_errors} of {len(errors)} errors)"
            )
        raise ValidationError(f"Text format validation failed:\n{error_summary}")

    return True


def validate_text_trace(filepath: Path) -> Dict[str, Any]:
    """
    Complete validation of text trace file.

    Performs format validation and collects file statistics.

    Args:
        filepath: Path to text trace file

    Returns:
        Dictionary containing:
            - valid: bool - Whether validation passed
            - record_count: int - Number of trace records (header lines)
            - file_size: int - File size in bytes
            - errors: List[str] - Error messages (empty if valid)

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    result: Dict[str, Any] = {
        "valid": False,
        "record_count": 0,
        "file_size": filepath.stat().st_size,
        "errors": [],
    }

    # Count records (header lines)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if REG_INFO_HEADER_PATTERN.match(
                    line
                ) or MEM_ACCESS_HEADER_PATTERN.match(line):
                    result["record_count"] += 1

        if result["record_count"] == 0:
            result["errors"].append("No trace records found in file")
            return result

    except Exception as e:
        result["errors"].append(f"Error counting records: {str(e)}")
        return result

    # Validate format
    try:
        validate_text_format(filepath)
        result["valid"] = True
    except ValidationError as e:
        result["errors"].append(str(e))
    except Exception as e:
        result["errors"].append(f"Format validation error: {str(e)}")

    return result


def parse_text_trace_record(lines: List[str]) -> Dict[str, Any]:
    """
    Parse a single trace record from text format.

    Args:
        lines: List of lines comprising a single trace record
              (header line + data lines)

    Returns:
        Dictionary containing parsed fields:
            - ctx: str - Context pointer
            - cta: List[int] - CTA coordinates [x, y, z]
            - warp: int - Warp ID
            - sass: str - SASS instruction
            - record_type: str - "reg_info" or "mem_access"
            - data: Dict[str, Any] - Type-specific data

    Raises:
        ValueError: If record format is invalid
    """
    if not lines:
        raise ValueError("Empty record")

    header = lines[0]

    # Try to match reg_info header
    reg_match = re.match(
        r"^CTX\s+(0x[0-9a-fA-F]+)\s+-\s+CTA\s+(\d+),(\d+),(\d+)\s+-\s+"
        r"warp\s+(\d+)\s+-\s+(.+):$",
        header,
    )

    if reg_match:
        ctx, cta_x, cta_y, cta_z, warp, sass = reg_match.groups()
        return {
            "ctx": ctx,
            "cta": [int(cta_x), int(cta_y), int(cta_z)],
            "warp": int(warp),
            "sass": sass,
            "record_type": "reg_info",
            "data": {},
        }

    # Try to match mem_access header
    mem_match = re.match(
        r"^CTX\s+(0x[0-9a-fA-F]+)\s+-\s+kernel_launch_id\s+(\d+)\s+-\s+"
        r"CTA\s+(\d+),(\d+),(\d+)\s+-\s+warp\s+(\d+)\s+-\s+PC\s+(\d+)\s+-\s+(.+):$",
        header,
    )

    if mem_match:
        ctx, kid, cta_x, cta_y, cta_z, warp, pc, sass = mem_match.groups()
        return {
            "ctx": ctx,
            "kernel_launch_id": int(kid),
            "cta": [int(cta_x), int(cta_y), int(cta_z)],
            "warp": int(warp),
            "pc": int(pc),
            "sass": sass,
            "record_type": "mem_access",
            "data": {},
        }

    raise ValueError(f"Unrecognized header format: {header}")
