# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the analyze subcommand.

This module provides command-line interface for analyzing CUTracer trace files.
"""

import csv
import io
import json
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Iterator, Optional

import click
from tabulate import tabulate

from cutracer.analysis.grouper import StreamingGrouper
from cutracer.analysis.reader import parse_filter_expr, TraceReader


# Default fields to display when --fields is not specified
DEFAULT_FIELDS = ["warp", "pc", "sass"]


def _format_value(value) -> str:
    """Format a value for display."""
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def _apply_record_selection_streaming(
    records: Iterator[dict],
    head: Optional[int] = None,
    tail: Optional[int] = None,
) -> list[dict]:
    """
    Apply record selection using streaming (memory-efficient).

    This function processes records in a streaming fashion:
    - For --head: Uses itertools.islice to stop early after N records
    - For --tail: Uses collections.deque(maxlen=N) to keep only last N records

    Memory complexity:
    - --head N: O(N) - only stores N records
    - --tail N: O(N) - deque automatically discards older records

    Args:
        records: Iterator of trace records (not a list!)
        head: Number of records from the beginning (default: 10)
        tail: Number of records from the end (overrides head)

    Returns:
        Selected subset of records as a list
    """
    if tail is not None:
        if tail <= 0:
            return []
        # deque with maxlen automatically discards oldest items
        # Memory: O(tail) regardless of total record count
        return list(deque(records, maxlen=tail))

    # Default head to 10 if not specified
    head = head if head is not None else 10
    if head <= 0:
        return []
    # islice stops iteration after head items - no need to read entire file
    # Memory: O(head) regardless of total record count
    return list(islice(records, head))


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
    "--group-by",
    "-g",
    type=str,
    default=None,
    help="Group by field (e.g., 'warp', 'cta', 'sass').",
)
@click.option(
    "--count",
    is_flag=True,
    help="Show record count per group (requires --group-by).",
)
@click.option(
    "--top",
    type=int,
    default=None,
    help="Show only top N groups by count (requires --group-by --count).",
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
    group_by: Optional[str],
    count: bool,
    top: Optional[int],
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
      cutraceross analyze trace.ndjson --group-by warp --tail 10
      cutraceross analyze trace.ndjson --group-by warp --count
      cutraceross analyze trace.ndjson --group-by sass --count --top 20
      cutraceross analyze trace.ndjson --format json
    """
    # Validate option combinations
    if count and not group_by:
        raise click.ClickException("--count requires --group-by")
    if top is not None and not count:
        raise click.ClickException("--top requires --count")

    reader = TraceReader(file)

    # Get iterator (lazy - does not load all records)
    records_iter = reader.iter_records()

    # Apply filter if specified (still streaming)
    if filter_expr:
        try:
            predicate = parse_filter_expr(filter_expr)
            # Generator expression - maintains streaming
            records_iter = (r for r in records_iter if predicate(r))
        except ValueError as e:
            raise click.ClickException(str(e))

    # Determine fields to display
    field_list = [f.strip() for f in fields.split(",")] if fields else None

    # Group mode
    if group_by:
        grouper = StreamingGrouper(records_iter, group_by)

        if count:
            # Count mode - show record count per group
            counts = grouper.count_per_group()
            _output_counts(counts, group_by, top, output_format, not no_header)
        elif tail is not None:
            # Get last N records per group
            groups = grouper.tail_per_group(tail)
            _output_groups(groups, group_by, field_list, output_format, not no_header)
        else:
            # Get first N records per group (default: 10)
            groups = grouper.head_per_group(head)
            _output_groups(groups, group_by, field_list, output_format, not no_header)
    else:
        # Simple view mode - streaming
        records = _apply_record_selection_streaming(records_iter, head=head, tail=tail)

        # Determine fields from first record if not specified
        display_fields = _get_display_fields(records, fields)

        # Format and output based on format option
        if output_format == "json":
            output = _format_records_json(records, display_fields)
        elif output_format == "csv":
            output = _format_records_csv(records, display_fields, show_header=not no_header)
        else:  # table
            output = _format_records_table(records, display_fields, show_header=not no_header)

        click.echo(output)


def _output_groups(
    groups: dict,
    group_field: str,
    field_list: Optional[list[str]],
    output_format: str,
    show_header: bool,
) -> None:
    """
    Output grouped records.

    Args:
        groups: Dict mapping group key to list of records
        group_field: Name of the field used for grouping
        field_list: List of fields to display (or None for defaults)
        output_format: Output format (table, json, csv)
        show_header: Whether to show headers
    """
    if not groups:
        click.echo("No records found.")
        return

    if output_format == "json":
        # JSON output: nested structure
        output_data = {}
        for key, records in sorted(groups.items(), key=lambda x: str(x[0])):
            key_str = str(key) if not isinstance(key, str) else key
            # Filter fields if specified
            if field_list:
                records = [{f: r.get(f) for f in field_list if f in r} for r in records]
            output_data[key_str] = records
        click.echo(json.dumps(output_data, indent=2))
    else:
        # Table or CSV output: show each group with header
        first_group = True
        for key, records in sorted(groups.items(), key=lambda x: str(x[0])):
            if not records:
                continue

            # Determine fields for this group
            display_fields = field_list if field_list else _get_display_fields(records, None)

            # Group header
            if output_format == "table":
                if not first_group:
                    click.echo()  # Blank line between groups
                click.echo(f"=== {group_field}={key} ({len(records)} records) ===")

            # Format records
            if output_format == "csv":
                output = _format_records_csv(records, display_fields, show_header=show_header and first_group)
            else:  # table
                output = _format_records_table(records, display_fields, show_header=show_header)

            click.echo(output)
            first_group = False


def _output_counts(
    counts: dict,
    group_field: str,
    top_n: Optional[int],
    output_format: str,
    show_header: bool,
) -> None:
    """
    Output record counts per group.

    Args:
        counts: Dict mapping group key to count
        group_field: Name of the field used for grouping
        top_n: Only show top N groups by count (None for all)
        output_format: Output format (table, json, csv)
        show_header: Whether to show headers
    """
    if not counts:
        click.echo("No records found.")
        return

    # Sort by count (descending), then by key (ascending) for stability
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))

    # Apply top N filter if specified
    if top_n is not None and top_n > 0:
        sorted_counts = sorted_counts[:top_n]

    if output_format == "json":
        # JSON output: object with group key -> count
        output_data = {str(k): v for k, v in sorted_counts}
        click.echo(json.dumps(output_data, indent=2))
    elif output_format == "csv":
        # CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        if show_header:
            writer.writerow([group_field, "count"])
        for key, count in sorted_counts:
            writer.writerow([key, count])
        click.echo(output.getvalue().rstrip("\n"))
    else:
        # Table output
        table_data = [[key, count] for key, count in sorted_counts]
        headers = [group_field.upper(), "COUNT"] if show_header else []
        click.echo(tabulate(table_data, headers=headers, tablefmt="plain"))
