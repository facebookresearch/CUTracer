# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the analyze subcommand.

This module provides command-line interface for analyzing CUTracer trace files.
"""

import csv
import io
import json
from pathlib import Path
from typing import Optional

import click
from tabulate import tabulate

from cutracer.analysis.reader import parse_filter_expr, TraceReader


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


def _format_records_json(records: list[dict], fields: list[str]) -> str:
    """
    Format records as JSON.

    Args:
        records: List of trace records
        fields: List of field names to include

    Returns:
        JSON formatted string
    """
    if not records:
        return "[]"

    # Filter to only requested fields
    filtered_records = []
    for record in records:
        filtered_record = {f: record.get(f) for f in fields if f in record}
        filtered_records.append(filtered_record)

    return json.dumps(filtered_records, indent=2)


def _format_records_csv(records: list[dict], fields: list[str], show_header: bool) -> str:
    """
    Format records as CSV.

    Args:
        records: List of trace records
        fields: List of field names to include
        show_header: Whether to include the header row

    Returns:
        CSV formatted string
    """
    if not records:
        return ""

    output = io.StringIO()
    writer = csv.writer(output)

    if show_header:
        writer.writerow(fields)

    for record in records:
        row = [_format_value(record.get(f, "")) for f in fields]
        writer.writerow(row)

    return output.getvalue().rstrip("\n")


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
    "--filter",
    "filter_expr",
    type=str,
    default=None,
    help="Filter expression (e.g., 'warp=24', 'type=mem_trace').",
)
@click.option(
    "--fields",
    type=str,
    default=None,
    help="Fields to display (comma-separated). Default: warp,pc,sass",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--no-header",
    is_flag=True,
    help="Don't show table/CSV header.",
)
def analyze_command(
    file: Path,
    head: int,
    tail: Optional[int],
    filter_expr: Optional[str],
    fields: Optional[str],
    output_format: str,
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
      cutraceross analyze trace.ndjson --filter "warp=24"
      cutraceross analyze trace.ndjson --format json
      cutraceross analyze trace.ndjson --format csv > output.csv
    """
    reader = TraceReader(file)

    # Read all records
    records = list(reader.iter_records())

    # Apply filter if specified
    if filter_expr:
        try:
            predicate = parse_filter_expr(filter_expr)
            records = [r for r in records if predicate(r)]
        except ValueError as e:
            raise click.ClickException(str(e))

    # Apply record selection (head/tail)
    records = _apply_record_selection(records, head=head, tail=tail)

    # Determine fields to display
    display_fields = _get_display_fields(records, fields)

    # Format and output based on format option
    if output_format == "json":
        output = _format_records_json(records, display_fields)
    elif output_format == "csv":
        output = _format_records_csv(records, display_fields, show_header=not no_header)
    else:  # table
        output = _format_records_table(records, display_fields, show_header=not no_header)

    click.echo(output)
