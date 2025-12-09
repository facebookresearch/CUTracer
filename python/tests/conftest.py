# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pytest configuration and shared fixtures for CUTracer tests.
"""

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# Sample valid reg_trace record
VALID_REG_TRACE_RECORD = {
    "type": "reg_trace",
    "ctx": "0x5a1100",
    "sass": "LDC R1, c[0x0][0x28] ;",
    "trace_index": 0,
    "timestamp": 1763069214784647489,
    "grid_launch_id": 0,
    "cta": [0, 0, 0],
    "warp": 0,
    "opcode_id": 0,
    "pc": 0,
    "regs": [[0] * 32],
}

# Sample valid mem_trace record
VALID_MEM_TRACE_RECORD = {
    "type": "mem_trace",
    "ctx": "0x5a1100",
    "sass": "LD.E.64 R2, [R8] ;",
    "trace_index": 1,
    "timestamp": 1763069214784700000,
    "grid_launch_id": 0,
    "cta": [0, 0, 0],
    "warp": 0,
    "opcode_id": 1,
    "pc": 16,
    "addrs": [0x7f0000000000 + i * 8 for i in range(32)],
}

# Sample valid opcode_only record
VALID_OPCODE_ONLY_RECORD = {
    "type": "opcode_only",
    "ctx": "0x5a1100",
    "sass": "EXIT ;",
    "trace_index": 2,
    "timestamp": 1763069214784800000,
    "grid_launch_id": 0,
    "cta": [0, 0, 0],
    "warp": 0,
    "opcode_id": 2,
    "pc": 32,
}

# Sample valid text trace header line
VALID_TEXT_HEADER = "CTX 0x5a46d0 - CTA 0,0,0 - warp 0 - LDC R1, c[0x0][0x28] ;:"

# Sample valid text trace register line
VALID_TEXT_REGISTER_LINE = "  * Reg0_T00: 0x00000000 Reg0_T01: 0x00000000 Reg0_T02: 0x00000000 Reg0_T03: 0x00000000"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_ndjson_file(temp_dir: Path) -> Path:
    """Create a valid NDJSON trace file with reg_trace records."""
    filepath = temp_dir / "valid_trace.ndjson"
    records = [
        VALID_REG_TRACE_RECORD,
        {**VALID_REG_TRACE_RECORD, "trace_index": 1, "warp": 1},
        {**VALID_REG_TRACE_RECORD, "trace_index": 2, "warp": 2},
    ]
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return filepath


@pytest.fixture
def valid_ndjson_file_mixed(temp_dir: Path) -> Path:
    """Create a valid NDJSON trace file with mixed record types."""
    filepath = temp_dir / "mixed_trace.ndjson"
    records = [
        VALID_REG_TRACE_RECORD,
        VALID_MEM_TRACE_RECORD,
        VALID_OPCODE_ONLY_RECORD,
    ]
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return filepath


@pytest.fixture
def invalid_json_syntax_file(temp_dir: Path) -> Path:
    """Create a file with invalid JSON syntax."""
    filepath = temp_dir / "invalid_syntax.ndjson"
    with open(filepath, "w") as f:
        f.write('{"type": "reg_trace", "ctx": "0x5a1100"}\n')
        f.write('{"type": "reg_trace", "ctx": invalid}\n')  # Invalid JSON
        f.write('{"type": "reg_trace", "ctx": "0x5a1100"}\n')
    return filepath


@pytest.fixture
def invalid_schema_file(temp_dir: Path) -> Path:
    """Create a file with valid JSON but invalid schema."""
    filepath = temp_dir / "invalid_schema.ndjson"
    records = [
        {"type": "reg_trace", "ctx": "0x5a1100"},  # Missing required fields
        {"type": "unknown_type", "ctx": "0x5a1100"},  # Invalid type
    ]
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return filepath


@pytest.fixture
def empty_file(temp_dir: Path) -> Path:
    """Create an empty file."""
    filepath = temp_dir / "empty.ndjson"
    filepath.touch()
    return filepath


@pytest.fixture
def valid_text_trace_file(temp_dir: Path) -> Path:
    """Create a valid text trace file."""
    filepath = temp_dir / "valid_trace.log"
    content = """CTX 0x5a46d0 - CTA 0,0,0 - warp 0 - LDC R1, c[0x0][0x28] ;:
  * Reg0_T00: 0x00000000 Reg0_T01: 0x00000000 Reg0_T02: 0x00000000 Reg0_T03: 0x00000000 Reg0_T04: 0x00000000 Reg0_T05: 0x00000000 Reg0_T06: 0x00000000 Reg0_T07: 0x00000000 Reg0_T08: 0x00000000 Reg0_T09: 0x00000000 Reg0_T10: 0x00000000 Reg0_T11: 0x00000000 Reg0_T12: 0x00000000 Reg0_T13: 0x00000000 Reg0_T14: 0x00000000 Reg0_T15: 0x00000000 Reg0_T16: 0x00000000 Reg0_T17: 0x00000000 Reg0_T18: 0x00000000 Reg0_T19: 0x00000000 Reg0_T20: 0x00000000 Reg0_T21: 0x00000000 Reg0_T22: 0x00000000 Reg0_T23: 0x00000000 Reg0_T24: 0x00000000 Reg0_T25: 0x00000000 Reg0_T26: 0x00000000 Reg0_T27: 0x00000000 Reg0_T28: 0x00000000 Reg0_T29: 0x00000000 Reg0_T30: 0x00000000 Reg0_T31: 0x00000000

CTX 0x5a46d0 - CTA 1,0,0 - warp 1 - LDC R1, c[0x0][0x28] ;:
  * Reg0_T00: 0x00000001 Reg0_T01: 0x00000001 Reg0_T02: 0x00000001 Reg0_T03: 0x00000001 Reg0_T04: 0x00000001 Reg0_T05: 0x00000001 Reg0_T06: 0x00000001 Reg0_T07: 0x00000001 Reg0_T08: 0x00000001 Reg0_T09: 0x00000001 Reg0_T10: 0x00000001 Reg0_T11: 0x00000001 Reg0_T12: 0x00000001 Reg0_T13: 0x00000001 Reg0_T14: 0x00000001 Reg0_T15: 0x00000001 Reg0_T16: 0x00000001 Reg0_T17: 0x00000001 Reg0_T18: 0x00000001 Reg0_T19: 0x00000001 Reg0_T20: 0x00000001 Reg0_T21: 0x00000001 Reg0_T22: 0x00000001 Reg0_T23: 0x00000001 Reg0_T24: 0x00000001 Reg0_T25: 0x00000001 Reg0_T26: 0x00000001 Reg0_T27: 0x00000001 Reg0_T28: 0x00000001 Reg0_T29: 0x00000001 Reg0_T30: 0x00000001 Reg0_T31: 0x00000001

CTX 0x5a46d0 - CTA 0,0,0 - warp 2 - ADD R2, R1, R0 ;:
  * Reg0_T00: 0x00000002 Reg0_T01: 0x00000002 Reg0_T02: 0x00000002 Reg0_T03: 0x00000002 Reg0_T04: 0x00000002 Reg0_T05: 0x00000002 Reg0_T06: 0x00000002 Reg0_T07: 0x00000002 Reg0_T08: 0x00000002 Reg0_T09: 0x00000002 Reg0_T10: 0x00000002 Reg0_T11: 0x00000002 Reg0_T12: 0x00000002 Reg0_T13: 0x00000002 Reg0_T14: 0x00000002 Reg0_T15: 0x00000002 Reg0_T16: 0x00000002 Reg0_T17: 0x00000002 Reg0_T18: 0x00000002 Reg0_T19: 0x00000002 Reg0_T20: 0x00000002 Reg0_T21: 0x00000002 Reg0_T22: 0x00000002 Reg0_T23: 0x00000002 Reg0_T24: 0x00000002 Reg0_T25: 0x00000002 Reg0_T26: 0x00000002 Reg0_T27: 0x00000002 Reg0_T28: 0x00000002 Reg0_T29: 0x00000002 Reg0_T30: 0x00000002 Reg0_T31: 0x00000002
"""
    filepath.write_text(content)
    return filepath


@pytest.fixture
def empty_text_file(temp_dir: Path) -> Path:
    """Create an empty text file."""
    filepath = temp_dir / "empty.log"
    filepath.touch()
    return filepath


@pytest.fixture
def real_ndjson_trace() -> Path:
    """Return path to real NDJSON trace file if it exists."""
    filepath = Path("/home/yhao/CUTracer_oss/tests/vectoradd/kernel_dcd76e64b30810e4_iter0__Z6vecAddPdS_S_i.ndjson")
    if filepath.exists():
        return filepath
    pytest.skip("Real NDJSON trace file not found")


@pytest.fixture
def real_text_trace() -> Path:
    """Return path to real text trace file if it exists."""
    filepath = Path("/home/yhao/CUTracer_oss/tests/vectoradd/kernel_dcd76e64b30810e4_iter0__Z6vecAddPdS_S_i.log")
    if filepath.exists():
        return filepath
    pytest.skip("Real text trace file not found")
