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
    format_records_ndjson,
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


def _write_output(
    output: str,
    output_file: Optional[Path],
    compress: bool = False,
    record_count: Optional[int] = None,
) -> None:
    """Write output to file or stdout."""
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            import zstandard as zstd

            cctx = zstd.ZstdCompressor()
            output_file.write_bytes(cctx.compress((output + "\n").encode("utf-8")))
        else:
            output_file.write_text(output + "\n")
        if record_count is not None:
            click.echo(f"{record_count} records written to {output_file}", err=True)
        else:
            click.echo(f"Output written to {output_file}", err=True)
    else:
        click.echo(output)


def _format_counts(
    counts: dict[Any, int],
    group_field: str,
    top_n: Optional[int],
    output_format: str,
    show_header: bool,
) -> str:
    """
    Format record counts per group.

    Args:
        counts: Dict mapping group key to count
        group_field: Name of the field used for grouping
        top_n: Only show top N groups by count (None for all)
        output_format: Output format (table, json, csv, ndjson)
        show_header: Whether to show headers

    Returns:
        Formatted output string
    """
    if not counts:
        return "No records found."

    # Sort by count (descending), then by key (ascending) for stability
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))

    # Apply top N filter if specified
    if top_n is not None and top_n > 0:
        sorted_counts = sorted_counts[:top_n]

    if output_format == "json":
        output_data = {str(k): v for k, v in sorted_counts}
        return json.dumps(output_data, indent=2)
    elif output_format == "csv":
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        if show_header:
            writer.writerow([group_field, "count"])
        for key, cnt in sorted_counts:
            writer.writerow([key, cnt])
        return output.getvalue().rstrip("\n").replace("\r", "")
    elif output_format == "ndjson":
        lines = [json.dumps({group_field: k, "count": v}) for k, v in sorted_counts]
        return "\n".join(lines)
    else:  # table
        table_data = [[key, cnt] for key, cnt in sorted_counts]
        headers = [group_field.upper(), "COUNT"] if show_header else []
        return tabulate(table_data, headers=headers, tablefmt="plain")


def _format_groups(
    groups: dict[Any, list[dict]],
    group_by: str,
    output_format: str,
    fields: Optional[str],
    no_header: bool,
) -> str:
    """Format grouped records with group headers."""
    if not groups:
        return "No records found."

    # Compute warp summary if grouping by warp
    warp_summary = None
    if group_by == "warp":
        warp_summary = compute_warp_summary(groups)

    output_parts = []

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

        return json.dumps(output_data, indent=2)

    elif output_format == "ndjson":
        # NDJSON output: each record on a line, with group field added
        is_all_fields = fields and fields.strip() in ("*", "all")
        for group_key, records in sorted(groups.items(), key=lambda x: str(x[0])):
            if not records:
                continue
            for record in records:
                if is_all_fields:
                    # All fields: output each record as-is, preserving all fields
                    out_record = dict(record)
                else:
                    display_fields = get_display_fields(records, fields)
                    out_record = {
                        f: record.get(f) for f in display_fields if f in record
                    }
                out_record["_group"] = group_key
                output_parts.append(json.dumps(out_record))
        return "\n".join(output_parts)

    else:
        # Table or CSV output
        for group_key, records in sorted(groups.items(), key=lambda x: str(x[0])):
            if not records:
                continue
            display_fields = get_display_fields(records, fields)
            output_parts.append(
                f"\n=== Group: {group_key} ({len(records)} records) ==="
            )
            if output_format == "csv":
                output_parts.append(
                    format_records_csv(
                        records, display_fields, show_header=not no_header
                    )
                )
            else:  # table
                output_parts.append(
                    format_records_table(
                        records, display_fields, show_header=not no_header
                    )
                )

        # Print warp summary for table format only (not CSV)
        if output_format == "table" and warp_summary:
            output_parts.append(format_warp_summary_text(warp_summary))

        return "\n".join(output_parts)


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
    "--all",
    "-a",
    "all_records",
    is_flag=True,
    help="Show all records (overrides --head and --tail).",
)
@click.option(
    "--filter",
    "-f",
    "filter_expr",
    type=str,
    default=None,
    help="Filter expression (e.g., 'warp=24', 'pc=0x43d0').",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv", "ndjson"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--fields",
    type=str,
    default=None,
    help="Comma-separated list of fields to display (e.g., 'warp,pc,sass'), or '*'/'all' for all fields.",
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
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: stdout).",
)
@click.option(
    "--compress",
    is_flag=True,
    default=False,
    help="Compress output with Zstd (requires --output).",
)
def query_command(
    file: Path,
    head: int,
    tail: Optional[int],
    all_records: bool,
    filter_expr: Optional[str],
    output_format: str,
    fields: Optional[str],
    no_header: bool,
    group_by: Optional[str],
    count: bool,
    top: Optional[int],
    output_file: Optional[Path],
    compress: bool,
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
      cutracer query trace.ndjson --all --format ndjson
      cutracer query trace.ndjson --filter "warp=24"
      cutracer query trace.ndjson --filter "pc=0x43d0" --all --output filtered.ndjson
      cutracer query trace.ndjson --filter "pc=0x43d0" --all --format ndjson -o out.zst --compress
      cutracer query trace.ndjson --group-by warp
      cutracer query trace.ndjson --group-by warp --count
      cutracer query trace.ndjson --group-by sass --count --top 20
    """
    # Validate option combinations
    if count and not group_by:
        raise click.ClickException("--count requires --group-by")
    if top is not None and not count:
        raise click.ClickException("--top requires --count")
    if compress and not output_file:
        raise click.ClickException("--compress requires --output")

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
            output = _format_counts(counts, group_by, top, output_format, not no_header)
        elif all_records:
            groups = grouper.all_per_group()
            output = _format_groups(groups, group_by, output_format, fields, no_header)
        elif tail is not None:
            groups = grouper.tail_per_group(tail)
            output = _format_groups(groups, group_by, output_format, fields, no_header)
        else:
            groups = grouper.head_per_group(head)
            output = _format_groups(groups, group_by, output_format, fields, no_header)

        _write_output(output, output_file, compress)
        return

    # Apply head/tail/all selection
    if all_records:
        selected = list(records)
    else:
        selected = select_records(records, head=head, tail=tail)

    # Check if user requested all fields
    is_all_fields = fields and fields.strip() in ("*", "all")

    # Format output based on format option
    if output_format == "ndjson" and is_all_fields:
        # NDJSON + all fields: output each record as-is without field filtering.
        # This preserves all fields including those that only appear in some records
        # (e.g., 'uregs' in UREG instructions, 'addrs'/'values' in mem_value_trace).
        output = format_records_ndjson(selected, fields=None)
    else:
        # Determine fields to display
        display_fields = get_display_fields(selected, fields)

        if output_format == "json":
            output = format_records_json(selected, display_fields)
        elif output_format == "csv":
            output = format_records_csv(
                selected, display_fields, show_header=not no_header
            )
        elif output_format == "ndjson":
            output = format_records_ndjson(selected, display_fields)
        else:  # table
            output = format_records_table(
                selected, display_fields, show_header=not no_header
            )

    _write_output(output, output_file, compress, record_count=len(selected))
