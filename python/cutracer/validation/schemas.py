# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
JSON Schema definitions for CUTracer trace formats.

This module contains JSON Schema definitions for validating NDJSON trace files
produced by CUTracer. Each schema corresponds to a specific message type.
"""

from typing import Dict, Any

# Schema for MSG_TYPE_REG_INFO (type="reg_trace")
REG_INFO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "type",
        "ctx",
        "sass",
        "trace_index",
        "timestamp",
        "grid_launch_id",
        "cta",
        "warp",
        "opcode_id",
        "pc",
        "regs",
    ],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["reg_trace"]
        },
        "ctx": {
            "type": "string",
            "pattern": "^0x[0-9a-fA-F]+$"
        },
        "sass": {
            "type": "string",
            "minLength": 1
        },
        "trace_index": {
            "type": "integer",
            "minimum": 0
        },
        "timestamp": {
            "type": "integer",
            "minimum": 0
        },
        "grid_launch_id": {
            "type": "integer",
            "minimum": 0
        },
        "cta": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 3,
            "maxItems": 3
        },
        "warp": {
            "type": "integer",
            "minimum": 0
        },
        "opcode_id": {
            "type": "integer",
            "minimum": 0
        },
        "pc": {
            "type": "integer",
            "minimum": 0
        },
        "regs": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 32,
                "maxItems": 32
            }
        },
        "uregs": {
            "type": "array",
            "items": {"type": "integer"}
        }
    },
    "additionalProperties": False
}

# Schema for MSG_TYPE_MEM_ACCESS (type="mem_trace")
MEM_ACCESS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "type",
        "ctx",
        "sass",
        "trace_index",
        "timestamp",
        "grid_launch_id",
        "cta",
        "warp",
        "opcode_id",
        "pc",
        "addrs",
    ],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["mem_trace"]
        },
        "ctx": {
            "type": "string",
            "pattern": "^0x[0-9a-fA-F]+$"
        },
        "sass": {
            "type": "string",
            "minLength": 1
        },
        "trace_index": {
            "type": "integer",
            "minimum": 0
        },
        "timestamp": {
            "type": "integer",
            "minimum": 0
        },
        "grid_launch_id": {
            "type": "integer",
            "minimum": 0
        },
        "cta": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 3,
            "maxItems": 3
        },
        "warp": {
            "type": "integer",
            "minimum": 0
        },
        "opcode_id": {
            "type": "integer",
            "minimum": 0
        },
        "pc": {
            "type": "integer",
            "minimum": 0
        },
        "addrs": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 32,
            "maxItems": 32
        }
    },
    "additionalProperties": False
}

# Schema for MSG_TYPE_OPCODE_ONLY (type="opcode_only")
OPCODE_ONLY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "type",
        "ctx",
        "sass",
        "trace_index",
        "timestamp",
        "grid_launch_id",
        "cta",
        "warp",
        "opcode_id",
        "pc",
    ],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["opcode_only"]
        },
        "ctx": {
            "type": "string",
            "pattern": "^0x[0-9a-fA-F]+$"
        },
        "sass": {
            "type": "string",
            "minLength": 1
        },
        "trace_index": {
            "type": "integer",
            "minimum": 0
        },
        "timestamp": {
            "type": "integer",
            "minimum": 0
        },
        "grid_launch_id": {
            "type": "integer",
            "minimum": 0
        },
        "cta": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 3,
            "maxItems": 3
        },
        "warp": {
            "type": "integer",
            "minimum": 0
        },
        "opcode_id": {
            "type": "integer",
            "minimum": 0
        },
        "pc": {
            "type": "integer",
            "minimum": 0
        }
    },
    "additionalProperties": False
}

# Mapping from type field to schema
SCHEMAS_BY_TYPE: Dict[str, Dict[str, Any]] = {
    "reg_trace": REG_INFO_SCHEMA,
    "mem_trace": MEM_ACCESS_SCHEMA,
    "opcode_only": OPCODE_ONLY_SCHEMA,
}
