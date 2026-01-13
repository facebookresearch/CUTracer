# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the analyze subcommand.

Provides command-line interface for analyzing CUTracer trace files.
"""

from pathlib import Path
from typing import Any, Optional

import click
from cutracer.analysis.formatters import (
    format_records_csv,
    format_records_json,
    format_records_table,
    get_display_fields,
)
from cutracer.analysis.grouper import StreamingGrouper
from cutracer.analysis.reader import parse_filter_expr, select_records, TraceReader


def _output_groups(
    groups: dict[Any, list[dict]],
    output_format: str,
    fields: Optional[str],
    no_header: bool,
) -> None:
    """Output grouped records with group headers."""
    for group_key, records in groups.items():
        if not records:
            continue
        display_fields = get_display_fields(records, fields)
        click.echo(f"\n=== Group: {group_key} ({len(records)} records) ===")
        if output_format == "json":
            output = format_records_json(records, display_fields)
        elif output_format == "csv":
            output = format_records_csv(
                records, display_fields, show_header=not no_header
            )
        else:  # table
            output = format_records_table(
                records, display_fields, show_header=not no_header
            )
        click.echo(output)


@click.command(name="analyze")
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
def analyze_command(
    file: Path,
    head: int,
    tail: Optional[int],
    filter_expr: Optional[str],
    output_format: str,
    fields: Optional[str],
    no_header: bool,
    group_by: Optional[str],
) -> None:
    """
    Analyze trace data from FILE.

    Reads NDJSON trace files (plain or Zstd-compressed) and displays
    records in a formatted table.

    \b
    Examples:
      cutracer analyze trace.ndjson
      cutracer analyze trace.ndjson.zst --head 20
      cutracer analyze trace.ndjson --tail 5
      cutracer analyze trace.ndjson --filter "warp=24"
      cutracer analyze trace.ndjson --group-by warp
    """
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
        if tail is not None:
            groups = grouper.tail_per_group(tail)
        else:
            groups = grouper.head_per_group(head)
        _output_groups(groups, output_format, fields, no_header)
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
