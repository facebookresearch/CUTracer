# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the analyze subcommand.

This module provides command-line interface for analyzing CUTracer trace files.
"""

from pathlib import Path
from typing import Optional

import click
from tabulate import tabulate

from cutracer.analysis.reader import TraceReader


# Default fields to display when --fields is not specified
DEFAULT_FIELDS = ["warp", "pc", "sass"]


def _format_value(value) -> str:
    """Format a value for display."""
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def _apply_record_selection(
    records: list[dict],
    head: Optional[int] = None,
    tail: Optional[int] = None,
) -> list[dict]:
    """
    Apply record selection based on options.

    Currently supports head and tail. Designed for future extension
    with options like --range, --skip, etc.

    Args:
        records: All trace records
        head: Number of records from the beginning (default: 10)
        tail: Number of records from the end (overrides head)

    Returns:
        Selected subset of records
    """
    if tail is not None:
        return records[-tail:] if tail > 0 else []

    # Default head to 10 if not specified
    head = head if head is not None else 10
    return records[:head] if head > 0 else []


def _get_display_fields(records: list[dict], requested_fields: Optional[str]) -> list[str]:
    """
    Determine which fields to display.

    Args:
        records: List of trace records
        requested_fields: Comma-separated field names or None

    Returns:
        List of field names to display
    """
    if requested_fields:
        return [f.strip() for f in requested_fields.split(",")]

    # Use default fields, but only if they exist in the records
    if records:
        available_fields = set(records[0].keys())
        return [f for f in DEFAULT_FIELDS if f in available_fields]

    return DEFAULT_FIELDS


def _format_records_table(
    records: list[dict],
    fields: list[str],
    show_header: bool,
) -> str:
    """
    Format records as a table.

    Args:
        records: List of trace records
        fields: List of field names to display
        show_header: Whether to show the table header

    Returns:
        Formatted table string
    """
    if not records:
        return "No records found."

    # Build table data
    table_data = []
    for record in records:
        row = [_format_value(record.get(f, "")) for f in fields]
        table_data.append(row)

    headers = [f.upper() for f in fields] if show_header else []

    return tabulate(
        table_data,
        headers=headers,
        tablefmt="plain",
    )


@click.command(name="analyze")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--head",
    "-n",
    type=int,
    default=10,
    show_default=True,
    help="Show first N records.",
)
@click.option(
    "--tail",
    type=int,
    default=None,
    help="Show last N records (overrides --head).",
)
@click.option(
    "--fields",
    type=str,
    default=None,
    help="Fields to display (comma-separated). Default: warp,pc,sass",
)
@click.option(
    "--no-header",
    is_flag=True,
    help="Don't show table header.",
)
def analyze_command(
    file: Path,
    head: int,
    tail: Optional[int],
    fields: Optional[str],
    no_header: bool,
) -> None:
    """Analyze trace data.

    Read and display trace records from NDJSON files.
    Supports plain and Zstd-compressed (.ndjson.zst) files.

    \b
    Examples:
      cutraceross analyze trace.ndjson              # Show first 10 records
      cutraceross analyze trace.ndjson -n 20        # Show first 20 records
      cutraceross analyze trace.ndjson --tail 5     # Show last 5 records
      cutraceross analyze trace.ndjson --fields warp,pc,sass,cta
    """
    reader = TraceReader(file)

    # Read all records (needed for tail functionality)
    records = list(reader.iter_records())

    # Apply record selection
    records = _apply_record_selection(records, head=head, tail=tail)

    # Determine fields to display
    display_fields = _get_display_fields(records, fields)

    # Format and output
    output = _format_records_table(records, display_fields, show_header=not no_header)
    click.echo(output)
