# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
JSON validator for CUTracer NDJSON trace files.

This module provides functions to validate NDJSON trace files produced by
CUTracer for syntax correctness and schema compliance.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonschema
from jsonschema import ValidationError as JsonSchemaValidationError

from .schemas import SCHEMAS_BY_TYPE


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def validate_json_syntax(filepath: Path) -> Tuple[int, List[str]]:
    """
    Validate JSON syntax line-by-line for NDJSON file.

    Args:
        filepath: Path to NDJSON trace file

    Returns:
        Tuple of (valid_count, errors) where:
            - valid_count: Number of valid JSON lines parsed
            - errors: List of error messages with line numbers

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    valid_count = 0
    errors: List[str] = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    json.loads(line)
                    valid_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON decode error - {e.msg}")

    except Exception as e:
        errors.append(f"File reading error: {str(e)}")

    return valid_count, errors


def validate_json_schema(
    filepath: Path,
    message_type: str = "reg_trace",
    max_errors: int = 10
) -> bool:
    """
    Validate JSON schema against TraceRecord definition.

    Args:
        filepath: Path to NDJSON trace file
        message_type: Expected message type (default: "reg_trace")
        max_errors: Maximum number of schema errors to collect (default: 10)

    Returns:
        True if all records pass schema validation

    Raises:
        ValidationError: If schema validation fails (includes detailed errors)
        FileNotFoundError: If file does not exist
        ValueError: If message_type is not recognized
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if message_type not in SCHEMAS_BY_TYPE:
        valid_types = ", ".join(SCHEMAS_BY_TYPE.keys())
        raise ValueError(
            f"Unknown message type: {message_type}. Valid types: {valid_types}"
        )

    schema = SCHEMAS_BY_TYPE[message_type]
    validator = jsonschema.Draft7Validator(schema)

    errors: List[str] = []
    line_num = 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON syntax error - {e.msg}")
                    if len(errors) >= max_errors:
                        break
                    continue

                # Check if type field matches expected type
                record_type = record.get("type", "")
                if record_type != message_type:
                    errors.append(
                        f"Line {line_num}: Expected type '{message_type}', "
                        f"got '{record_type}'"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                # Validate against schema
                validation_errors = list(validator.iter_errors(record))
                if validation_errors:
                    for error in validation_errors[:3]:  # Show first 3 errors per record
                        field_path = ".".join(str(p) for p in error.path)
                        errors.append(
                            f"Line {line_num}: Schema error at '{field_path}': "
                            f"{error.message}"
                        )
                    if len(errors) >= max_errors:
                        break

    except Exception as e:
        errors.append(f"File reading error: {str(e)}")

    if errors:
        error_summary = "\n".join(errors)
        if len(errors) >= max_errors:
            error_summary += f"\n... (showing first {max_errors} errors)"
        raise ValidationError(f"Schema validation failed:\n{error_summary}")

    return True


def validate_json_trace(filepath: Path) -> Dict[str, Any]:
    """
    Complete validation of NDJSON trace file.

    Performs both syntax and schema validation, with auto-detection of
    message type from the first record.

    Args:
        filepath: Path to NDJSON trace file

    Returns:
        Dictionary containing:
            - valid: bool - Whether validation passed
            - record_count: int - Number of valid records
            - file_size: int - File size in bytes
            - message_type: str - Detected message type
            - errors: List[str] - Error messages (empty if valid)

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    result: Dict[str, Any] = {
        "valid": False,
        "record_count": 0,
        "file_size": filepath.stat().st_size,
        "message_type": None,
        "errors": []
    }

    # Step 1: Validate syntax
    try:
        valid_count, syntax_errors = validate_json_syntax(filepath)
        result["record_count"] = valid_count

        if syntax_errors:
            result["errors"].extend(syntax_errors)
            return result

        if valid_count == 0:
            result["errors"].append("No valid JSON records found in file")
            return result

    except Exception as e:
        result["errors"].append(f"Syntax validation error: {str(e)}")
        return result

    # Step 2: Auto-detect message type from first record
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    first_record = json.loads(line)
                    message_type = first_record.get("type")
                    if message_type not in SCHEMAS_BY_TYPE:
                        result["errors"].append(
                            f"Unknown message type in file: {message_type}"
                        )
                        return result
                    result["message_type"] = message_type
                    break
    except Exception as e:
        result["errors"].append(f"Failed to detect message type: {str(e)}")
        return result

    # Step 3: Validate schema
    try:
        validate_json_schema(filepath, message_type=result["message_type"])
        result["valid"] = True
    except ValidationError as e:
        result["errors"].append(str(e))
    except Exception as e:
        result["errors"].append(f"Schema validation error: {str(e)}")

    return result
