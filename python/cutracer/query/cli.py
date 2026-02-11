# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the query subcommand.

Provides command-line interface for querying and viewing CUTracer trace files.
"""

import csv
import io
import json
from pathlib import Path
from typing import Any, Optional

import click
from cutracer.query.formatters import (
    format_records_csv,
    format_records_json,
    format_records_table,
    get_display_fields,
)
from cutracer.query.grouper import StreamingGrouper
from cutracer.query.reader import parse_filter_expr, select_records, TraceReader
from cutracer.query.warp_summary import (
    compute_warp_summary,
    format_warp_summary_text,
    warp_summary_to_dict,
)
from tabulate import tabulate


def _output_groups(
    groups: dict[Any, list[dict]],
    group_by: str,
    output_format: str,
    fields: Optional[str],
    no_header: bool,
) -> None:
    """Output grouped records with group headers."""
    if not groups:
        click.echo("No records found.")
        return

    # Compute warp summary if grouping by warp
    warp_summary = None
    if group_by == "warp":
        warp_summary = compute_warp_summary(groups)

    if output_format == "json":
        # JSON output with optional warp_summary
        groups_data = {}
        for group_key, records in sorted(groups.items(), key=lambda x: str(x[0])):
            if not records:
                continue
            display_fields = get_display_fields(records, fields)
            filtered_records = [
                {f: r.get(f) for f in display_fields if f in r} for r in records
            ]
            groups_data[str(group_key)] = filtered_records

        if warp_summary:
            output_data = {
                "groups": groups_data,
                "warp_summary": warp_summary_to_dict(warp_summary),
            }
        else:
            output_data = groups_data

        click.echo(json.dumps(output_data, indent=2))
    else:
        # Table or CSV output
        for group_key, records in sorted(groups.items(), key=lambda x: str(x[0])):
            if not records:
                continue
            display_fields = get_display_fields(records, fields)
            click.echo(f"\n=== Group: {group_key} ({len(records)} records) ===")
            if output_format == "csv":
                output = format_records_csv(
                    records, display_fields, show_header=not no_header
                )
            else:  # table
                output = format_records_table(
                    records, display_fields, show_header=not no_header
                )
            click.echo(output)

        # Print warp summary for table format only (not CSV)
        if output_format == "table" and warp_summary:
            click.echo(format_warp_summary_text(warp_summary))


@click.command(name="query")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--head",
    "-n",
    type=int,
    default=10,
    show_default=True,
    help="Number of records to show from the beginning.",
)
@click.option(
    "--tail",
    "-t",
    type=int,
    default=None,
    help="Number of records to show from the end (overrides --head).",
)
@click.option(
    "--filter",
    "-f",
    "filter_expr",
    type=str,
    default=None,
    help="Filter expression (e.g., 'warp=24').",
)
@click.option(
    "--format",
    "-o",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--fields",
    type=str,
    default=None,
    help="Comma-separated list of fields to display (e.g., 'warp,pc,sass').",
)
@click.option(
    "--no-header",
    is_flag=True,
    default=False,
    help="Hide the table/CSV header row.",
)
@click.option(
    "--group-by",
    "-g",
    "group_by",
    type=str,
    default=None,
    help="Group records by field (e.g., 'warp').",
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
def query_command(
    file: Path,
    head: int,
    tail: Optional[int],
    filter_expr: Optional[str],
    output_format: str,
    fields: Optional[str],
    no_header: bool,
    group_by: Optional[str],
    count: bool,
    top: Optional[int],
) -> None:
    """
    Query and view trace data from FILE.

    Reads NDJSON trace files (plain or Zstd-compressed) and displays
    records in a formatted table.

    \b
    Examples:
      cutracer query trace.ndjson
      cutracer query trace.ndjson.zst --head 20
      cutracer query trace.ndjson --tail 5
      cutracer query trace.ndjson --filter "warp=24"
      cutracer query trace.ndjson --group-by warp
      cutracer query trace.ndjson --group-by warp --count
      cutracer query trace.ndjson --group-by sass --count --top 20
    """
    # Validate option combinations
    if count and not group_by:
        raise click.ClickException("--count requires --group-by")
    if top is not None and not count:
        raise click.ClickException("--top requires --count")

    # Create reader
    try:
        reader = TraceReader(file)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    # Get records iterator
    records = reader.iter_records()

    # Apply filter if specified
    if filter_expr:
        try:
            predicate = parse_filter_expr(filter_expr)
        except ValueError as e:
            raise click.ClickException(str(e))
        records = (r for r in records if predicate(r))

    # Handle grouped output
    if group_by:
        grouper = StreamingGrouper(records, group_by)

        if count:
            # Count mode - show record count per group
            counts = grouper.count_per_group()
            _output_counts(counts, group_by, top, output_format, not no_header)
        elif tail is not None:
            groups = grouper.tail_per_group(tail)
            _output_groups(groups, group_by, output_format, fields, no_header)
        else:
            groups = grouper.head_per_group(head)
            _output_groups(groups, group_by, output_format, fields, no_header)
        return

    # Apply head/tail selection
    selected = select_records(records, head=head, tail=tail)

    # Determine fields to display
    display_fields = get_display_fields(selected, fields)

    # Format output based on format option
    if output_format == "json":
        output = format_records_json(selected, display_fields)
    elif output_format == "csv":
        output = format_records_csv(selected, display_fields, show_header=not no_header)
    else:  # table
        output = format_records_table(
            selected, display_fields, show_header=not no_header
        )

    click.echo(output)


def _output_counts(
    counts: dict[Any, int],
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
        output_data = {str(k): v for k, v in sorted_counts}
        click.echo(json.dumps(output_data, indent=2))
    elif output_format == "csv":
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        if show_header:
            writer.writerow([group_field, "count"])
        for key, cnt in sorted_counts:
            writer.writerow([key, cnt])
        click.echo(output.getvalue().rstrip("\n").replace("\r", ""))
    else:  # table
        table_data = [[key, cnt] for key, cnt in sorted_counts]
        headers = [group_field.upper(), "COUNT"] if show_header else []
        click.echo(tabulate(table_data, headers=headers, tablefmt="plain"))
