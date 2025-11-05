#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Minimal trace validation tool for CUTracer CI testing.

Validates trace files in text (.log) and JSON (.ndjson) formats.
Designed for CI use with minimal dependencies and simple output.

Usage:
    validate_trace.py [--no-color] text <file.log>
    validate_trace.py [--no-color] json <file.ndjson>
    validate_trace.py [--no-color] compare <file.log> <file.ndjson>

Exit Codes:
    0 - Validation passed
    1 - Validation failed
    2 - File not found or error
"""

import json
import sys
from pathlib import Path


def validate_text(filepath):
    """
    Validate text trace file.

    Args:
        filepath: Path to text trace file

    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        if not filepath.exists():
            return False, f"File not found: {filepath}"

        content = filepath.read_text()
        if not content:
            return False, "Text trace file is empty"

        return True, None

    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_json(filepath):
    """
    Validate JSON trace file (NDJSON format).
    Checks that each line is valid JSON and validates trace record structure.

    Args:
        filepath: Path to NDJSON trace file

    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        if not filepath.exists():
            return False, f"File not found: {filepath}"

        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON at line {line_num}: {e}"

                # Validate trace record structure
                if "type" not in record:
                    return False, f"Missing 'type' field at line {line_num}"

                trace_type = record["type"]
                if trace_type not in ["reg_trace", "mem_trace", "opcode_only"]:
                    return (
                        False,
                        f"Invalid trace type '{trace_type}' at line {line_num}",
                    )

                # Validate common fields
                required_common = ["ctx", "trace_index", "timestamp"]
                for field in required_common:
                    if field not in record:
                        return (
                            False,
                            f"Missing '{field}' field in {trace_type} at line {line_num}",
                        )

                # Type-specific validation
                if trace_type == "reg_trace":
                    required = [
                        "grid_launch_id",
                        "cta",
                        "warp",
                        "opcode_id",
                        "pc",
                        "regs",
                    ]
                    for field in required:
                        if field not in record:
                            return (
                                False,
                                f"Missing '{field}' field in reg_trace at line {line_num}",
                            )

                    # Validate regs array structure (2D array)
                    if not isinstance(record["regs"], list):
                        return (
                            False,
                            f"'regs' must be an array in reg_trace at line {line_num}",
                        )
                    if record["regs"] and not isinstance(record["regs"][0], list):
                        return (
                            False,
                            f"'regs' must be 2D array in reg_trace at line {line_num}",
                        )

                    # Validate uregs if present (1D array)
                    if "uregs" in record:
                        if not isinstance(record["uregs"], list):
                            return (
                                False,
                                f"'uregs' must be an array in reg_trace at line {line_num}",
                            )

                elif trace_type == "mem_trace":
                    required = [
                        "grid_launch_id",
                        "cta",
                        "warp",
                        "opcode_id",
                        "pc",
                        "addrs",
                    ]
                    for field in required:
                        if field not in record:
                            return (
                                False,
                                f"Missing '{field}' field in mem_trace at line {line_num}",
                            )

                    # Validate addrs array (must be 32 addresses)
                    if not isinstance(record["addrs"], list):
                        return (
                            False,
                            f"'addrs' must be an array in mem_trace at line {line_num}",
                        )
                    if len(record["addrs"]) != 32:
                        return (
                            False,
                            f"'addrs' must have 32 elements in mem_trace at line {line_num}, got {len(record['addrs'])}",
                        )

                elif trace_type == "opcode_only":
                    required = ["grid_launch_id", "cta", "warp", "opcode_id", "pc"]
                    for field in required:
                        if field not in record:
                            return (
                                False,
                                f"Missing '{field}' field in opcode_only at line {line_num}",
                            )

        return True, None

    except Exception as e:
        return False, f"Error reading file: {e}"


def compare_formats(text_file, json_file):
    """
    Compare text and JSON trace formats.

    Args:
        text_file: Path to text trace file
        json_file: Path to JSON trace file

    Returns:
        (success: bool, error_message: str or None)
    """
    # Validate both files
    text_ok, text_err = validate_text(text_file)
    json_ok, json_err = validate_json(json_file)

    errors = []
    if not text_ok:
        errors.append(f"Text validation failed: {text_err}")
    if not json_ok:
        errors.append(f"JSON validation failed: {json_err}")

    if errors:
        return False, "; ".join(errors)

    return True, None


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: validate_trace.py [--no-color] {text|json|compare} <files...>")
        return 2

    # Parse arguments (skip --no-color flag if present)
    args = sys.argv[1:]
    if args[0] == "--no-color":
        args = args[1:]

    if len(args) < 2:
        print("Usage: validate_trace.py [--no-color] {text|json|compare} <files...>")
        return 2

    command = args[0]

    try:
        # Execute command
        if command == "text":
            filepath = Path(args[1])
            success, error = validate_text(filepath)

            if success:
                size = filepath.stat().st_size
                print(f"\n{'='*70}")
                print(f"Validating Text Trace: {filepath.name}")
                print(f"{'='*70}\n")
                print("✓ Text trace validation passed")
                print(f"ℹ File size: {format_size(size)}\n")
                return 0
            else:
                print(f"✗ Text validation failed: {error}")
                return 1

        elif command == "json":
            filepath = Path(args[1])
            success, error = validate_json(filepath)

            if success:
                size = filepath.stat().st_size
                # Count records
                with open(filepath) as f:
                    record_count = sum(1 for line in f if line.strip())

                print(f"\n{'='*70}")
                print(f"Validating JSON Trace: {filepath.name}")
                print(f"{'='*70}\n")
                print("✓ JSON trace validation passed")
                print(f"ℹ Record count: {record_count:,}")
                print(f"ℹ File size: {format_size(size)}\n")
                return 0
            else:
                print(f"✗ JSON validation failed: {error}")
                return 1

        elif command == "compare":
            if len(args) < 3:
                print("Usage: validate_trace.py compare <text_file> <json_file>")
                return 2

            text_file = Path(args[1])
            json_file = Path(args[2])

            success, error = compare_formats(text_file, json_file)

            print(f"\n{'='*70}")
            print("Comparing Trace Formats")
            print(f"{'='*70}\n")
            print(f"ℹ Text: {text_file.name}")
            print(f"ℹ JSON: {json_file.name}\n")

            # Show individual results
            print("Results:")
            text_ok, _ = validate_text(text_file)
            json_ok, _ = validate_json(json_file)

            if text_ok:
                print("✓ Text format valid")
            else:
                print("✗ Text format invalid")

            if json_ok:
                with open(json_file) as f:
                    record_count = sum(1 for line in f if line.strip())
                print(f"✓ JSON format valid (records: {record_count:,})")
            else:
                print("✗ JSON format invalid")

            if text_ok and json_ok:
                print(f"ℹ Text size: {format_size(text_file.stat().st_size)}")
                print(f"ℹ JSON size: {format_size(json_file.stat().st_size)}")

            print()
            if success:
                print("✓ Formats are consistent\n")
                return 0
            else:
                print("✗ Inconsistencies found:")
                print(f"✗   {error}\n")
                return 1

        else:
            print(f"Unknown command: {command}")
            print("Valid commands: text, json, compare")
            return 2

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
