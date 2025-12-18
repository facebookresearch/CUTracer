# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the validate subcommand.

This module provides command-line interface for validating CUTracer trace files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .json_validator import validate_json_trace
from .text_validator import validate_text_trace


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the validate subcommand."""
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the trace file to validate",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "text", "auto"],
        default="auto",
        help="File format. Default: auto-detect from extension.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode. Only return exit code.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output results in JSON format.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with additional details.",
    )


def _detect_format(file_path: Path) -> str:
    """Auto-detect file format from extension."""
    suffixes = "".join(file_path.suffixes).lower()
    if ".ndjson" in suffixes:
        return "json"
    elif file_path.suffix == ".log":
        return "text"
    else:
        return "unknown"


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_trace_format(result: dict[str, Any]) -> str:
    """Format trace format for display based on result."""
    compression = result.get("compression", "none")
    message_type = result.get("message_type")

    if compression == "zstd":
        return "NDJSON + Zstd"
    elif message_type:
        return "NDJSON"
    else:
        return "Text"


def _print_validation_result(result: dict[str, Any], verbose: bool = False) -> None:
    """Print validation result in human-readable format."""
    if result["valid"]:
        print("\u2705 Valid trace file")
        print(f"   Format:       {_format_trace_format(result)}")
        print(f"   Records:      {result['record_count']}")
        if result.get("message_type"):
            print(f"   Message type: {result['message_type']}")
        if result.get("file_size"):
            print(f"   File size:    {_format_size(result['file_size'])}")
        if verbose and result.get("compression") == "zstd":
            print("   Compression:  zstd")
    else:
        print("\u274c Validation failed")
        for error in result.get("errors", []):
            print(f"   {error}")


def validate_command(args: argparse.Namespace) -> int:
    """Execute the validate subcommand."""
    file_path: Path = args.file

    # Check file exists
    if not file_path.exists():
        if not args.quiet:
            print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 2

    # Detect format
    file_format = args.format
    if file_format == "auto":
        file_format = _detect_format(file_path)
        if file_format == "unknown":
            if not args.quiet:
                print(
                    f"Error: Cannot auto-detect format for {file_path}. "
                    "Use --format to specify.",
                    file=sys.stderr,
                )
            return 2

    # Run validation
    if file_format == "json":
        result = validate_json_trace(file_path)
    else:
        result = validate_text_trace(file_path)

    # Handle quiet mode
    if args.quiet:
        return 0 if result["valid"] else 1

    # Handle JSON output
    if args.json_output:
        # Convert Path objects to strings for JSON serialization
        output = {k: str(v) if isinstance(v, Path) else v for k, v in result.items()}
        print(json.dumps(output, indent=2))
        return 0 if result["valid"] else 1

    # Human-readable output
    _print_validation_result(result, args.verbose)

    return 0 if result["valid"] else 1
