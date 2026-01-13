# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the analyze subcommand.

Provides command-line interface for analyzing CUTracer trace files.
"""

from pathlib import Path
from typing import Optional

import click
from cutracer.analysis.formatters import format_records_table, get_display_fields
from cutracer.analysis.reader import parse_filter_expr, select_records, TraceReader


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
def analyze_command(
    file: Path,
    head: int,
    tail: Optional[int],
    filter_expr: Optional[str],
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

    # Apply head/tail selection
    selected = select_records(records, head=head, tail=tail)

    # Determine fields to display
    fields = get_display_fields(selected)

    # Format and output
    output = format_records_table(selected, fields)
    click.echo(output)
