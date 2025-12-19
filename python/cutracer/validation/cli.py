# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the validate subcommand.

This module provides command-line interface for validating CUTracer trace files.
"""

import json
import sys
from pathlib import Path
from typing import Any

import click

from .json_validator import validate_json_trace
from .text_validator import validate_text_trace


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
        click.echo("\u2705 Valid trace file")
        click.echo(f"   Format:       {_format_trace_format(result)}")
        click.echo(f"   Records:      {result['record_count']}")
        if result.get("message_type"):
            click.echo(f"   Message type: {result['message_type']}")
        if result.get("file_size"):
            click.echo(f"   File size:    {_format_size(result['file_size'])}")
        if verbose and result.get("compression") == "zstd":
            click.echo("   Compression:  zstd")
    else:
        click.echo("\u274c Validation failed")
        for error in result.get("errors", []):
            click.echo(f"   {error}")


@click.command(name="validate")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "-f",
    "file_format",
    type=click.Choice(["json", "text", "auto"]),
    default="auto",
    help="File format. Default: auto-detect from extension.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Quiet mode. Only return exit code.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results in JSON format.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output with additional details.",
)
def validate_command(
    file: Path,
    file_format: str,
    quiet: bool,
    json_output: bool,
    verbose: bool,
) -> None:
    """Validate a CUTracer trace file.

    Checks syntax and schema compliance for NDJSON, Zstd-compressed,
    and text format trace files.

    FILE is the path to the trace file to validate.
    """
    file_path = file

    # Detect format
    if file_format == "auto":
        file_format = _detect_format(file_path)
        if file_format == "unknown":
            if not quiet:
                click.echo(
                    f"Error: Cannot auto-detect format for {file_path}. "
                    "Use --format to specify.",
                    err=True,
                )
            sys.exit(2)

    # Run validation
    if file_format == "json":
        result = validate_json_trace(file_path)
    else:
        result = validate_text_trace(file_path)

    # Handle quiet mode
    if quiet:
        sys.exit(0 if result["valid"] else 1)

    # Handle JSON output
    if json_output:
        # Convert Path objects to strings for JSON serialization
        output = {k: str(v) if isinstance(v, Path) else v for k, v in result.items()}
        click.echo(json.dumps(output, indent=2))
        sys.exit(0 if result["valid"] else 1)

    # Human-readable output
    _print_validation_result(result, verbose)

    sys.exit(0 if result["valid"] else 1)
