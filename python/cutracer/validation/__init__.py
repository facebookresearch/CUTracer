# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer validation module.

This module provides validation functions for trace files in different formats:
- JSON validation (syntax and schema)
- Text format validation
- Cross-format consistency checking
"""

from .json_validator import (
    validate_json_syntax,
    validate_json_schema,
    validate_json_trace,
    ValidationError as JsonValidationError,
)

from .text_validator import (
    validate_text_format,
    validate_text_trace,
    parse_text_trace_record,
    ValidationError as TextValidationError,
)

from .consistency import (
    compare_record_counts,
    compare_trace_content,
    compare_trace_formats,
    get_trace_statistics,
)

from .schema_loader import (
    REG_INFO_SCHEMA,
    MEM_ACCESS_SCHEMA,
    OPCODE_ONLY_SCHEMA,
    SCHEMAS_BY_TYPE,
)

__all__ = [
    # JSON validation
    "validate_json_syntax",
    "validate_json_schema",
    "validate_json_trace",
    "JsonValidationError",
    
    # Text validation
    "validate_text_format",
    "validate_text_trace",
    "parse_text_trace_record",
    "TextValidationError",
    
    # Cross-format consistency
    "compare_record_counts",
    "compare_trace_content",
    "compare_trace_formats",
    "get_trace_statistics",
    
    # Schemas
    "REG_INFO_SCHEMA",
    "MEM_ACCESS_SCHEMA",
    "OPCODE_ONLY_SCHEMA",
    "SCHEMAS_BY_TYPE",
]
