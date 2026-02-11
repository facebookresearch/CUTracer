# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Record formatting utilities for different output formats.

Provides functions to format trace records as table, JSON, or CSV.
All functions are pure (no side effects) and return strings.
"""

import csv
import io
import json
from typing import Optional

# Default fields to display when --fields is not specified
DEFAULT_FIELDS = ["warp", "pc", "sass"]


def format_value(value) -> str:
    """
    Format a value for display.

    Handles special cases:
    - None -> empty string
    - bool -> lowercase "true"/"false"
    - list -> comma-separated values in brackets
    - dict -> JSON string
    - other -> str()

    Args:
        value: Any value to format

    Returns:
        String representation suitable for display
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def get_display_fields(
    records: list[dict], requested_fields: Optional[str] = None
) -> list[str]:
    """
    Determine which fields to display.

    Args:
        records: List of trace records
        requested_fields: Comma-separated field names, "*" or "all" for all fields, or None

    Returns:
        List of field names to display
    """
    if requested_fields:
        # Handle "*" or "all" to show all available fields
        if requested_fields.strip() in ("*", "all"):
            if records:
                # Collect union of all fields across all records.
                # Use dict.fromkeys to preserve insertion order (first record's order first).
                # This ensures fields like 'uregs' that only appear in some records
                # are included in the output.
                seen = dict.fromkeys(records[0].keys())
                for record in records[1:]:
                    for key in record:
                        if key not in seen:
                            seen[key] = None
                return list(seen)
            return DEFAULT_FIELDS
        return [f.strip() for f in requested_fields.split(",")]

    # Use default fields, but only if they exist in the records
    if records:
        available_fields = set(records[0].keys())
        return [f for f in DEFAULT_FIELDS if f in available_fields]

    return DEFAULT_FIELDS


def format_records_table(
    records: list[dict],
    fields: list[str],
    show_header: bool = True,
) -> str:
    """
    Format records as a plain text table with aligned columns.

    Args:
        records: List of trace records
        fields: List of field names to display
        show_header: Whether to show the table header

    Returns:
        Formatted table string
    """
    if not records:
        return "No records found."

    # Calculate column widths
    col_widths = {}
    for field in fields:
        # Start with header width if showing header
        max_width = len(field) if show_header else 0
        for record in records:
            val_str = format_value(record.get(field, ""))
            max_width = max(max_width, len(val_str))
        col_widths[field] = max_width

    lines = []

    # Header row
    if show_header:
        header_parts = [field.upper().ljust(col_widths[field]) for field in fields]
        lines.append("  ".join(header_parts))

    # Data rows
    for record in records:
        row_parts = []
        for field in fields:
            val_str = format_value(record.get(field, ""))
            row_parts.append(val_str.ljust(col_widths[field]))
        lines.append("  ".join(row_parts))

    return "\n".join(lines)


def format_records_json(records: list[dict], fields: list[str]) -> str:
    """
    Format records as JSON.

    Args:
        records: List of trace records
        fields: List of field names to include

    Returns:
        JSON formatted string (pretty-printed array)
    """
    if not records:
        return "[]"

    # Filter to only requested fields
    filtered_records = []
    for record in records:
        filtered_record = {f: record.get(f) for f in fields if f in record}
        filtered_records.append(filtered_record)

    return json.dumps(filtered_records, indent=2)


def format_records_csv(
    records: list[dict],
    fields: list[str],
    show_header: bool = True,
) -> str:
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

    output = io.StringIO(newline="")
    writer = csv.writer(output)

    if show_header:
        writer.writerow(fields)

    for record in records:
        row = [format_value(record.get(f, "")) for f in fields]
        writer.writerow(row)

    return output.getvalue().rstrip("\r\n").replace("\r\n", "\n").replace("\r", "")


def format_records_ndjson(
    records: list[dict], fields: Optional[list[str]] = None
) -> str:
    """
    Format records as NDJSON (Newline-Delimited JSON).

    Each record is output as a single JSON object on its own line.
    This format is ideal for streaming and line-by-line processing.

    Args:
        records: List of trace records
        fields: Optional list of field names to include. If None, all fields are included.

    Returns:
        NDJSON formatted string (one JSON object per line)
    """
    if not records:
        return ""

    lines = []
    for record in records:
        if fields:
            filtered_record = {f: record.get(f) for f in fields if f in record}
            lines.append(json.dumps(filtered_record))
        else:
            lines.append(json.dumps(record))

    return "\n".join(lines)
