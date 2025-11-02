# CUTracer Python Module

Python tools for CUTracer trace validation, parsing, and analysis.

## Overview

The `cutracer` Python package provides a comprehensive framework for working with CUTracer trace files. This module is designed to be:

- **Reusable**: Import and use in your own Python scripts
- **Testable**: Full pytest suite with high code coverage
- **Type-safe**: Type hints and mypy compatibility
- **Extensible**: Plugin architecture for future enhancements

## Installation

### For Development

```bash
cd /home/yhao/CUTracer_oss/python
pip install -e ".[dev]"
```

### For Production Use

```bash
cd /home/yhao/CUTracer_oss/python
pip install .
```

## Features

### Trace Validation (Current)

- **JSON Validation**: Validate NDJSON trace files (mode 2) for syntax and schema compliance
- **Text Validation**: Validate text-format trace files (mode 0) for format compliance
- **Cross-Format Consistency**: Compare different trace formats for data consistency

### Planned Features

- **Trace Parsing**: Parse trace files into structured Python objects
- **Analysis Tools**: Instruction histograms, performance metrics, trace comparison
- **Format Conversion**: Convert between different trace formats
- **Compression Support**: Handle zstd-compressed traces (mode 1)

## Usage

### Python API

```python
from cutracer.validation import validate_json_trace, validate_text_trace, compare_trace_formats

# Validate JSON trace
result = validate_json_trace("kernel_trace.ndjson")
if result["valid"]:
    print(f"✓ Valid JSON trace with {result['record_count']} records")
else:
    print(f"✗ Validation failed: {result['errors']}")

# Validate text trace
result = validate_text_trace("kernel_trace.log")
if result["valid"]:
    print(f"✓ Valid text trace")
else:
    print(f"✗ Validation failed: {result['errors']}")

# Compare two formats
result = compare_trace_formats("kernel_trace.log", "kernel_trace.ndjson")
if result["consistent"]:
    print("✓ Formats are consistent")
else:
    print(f"✗ Inconsistencies found: {result['differences']}")
```

### Command-Line Interface

The package includes a CLI tool for trace validation:

```bash
# Validate text format
python scripts/validate_trace.py text kernel_trace.log

# Validate JSON format
python scripts/validate_trace.py json kernel_trace.ndjson

# Compare two formats
python scripts/validate_trace.py compare kernel_trace.log kernel_trace.ndjson

# Verbose output
python scripts/validate_trace.py json kernel_trace.ndjson --verbose

# JSON output (for CI integration)
python scripts/validate_trace.py json kernel_trace.ndjson --output json
```

#### Exit Codes

- `0`: Success - validation passed
- `1`: Validation failed - errors found
- `2`: File not found or I/O error

## Module Structure

```
python/
├── cutracer/                    # Main package
│   ├── __init__.py              # Package entry point
│   └── validation/              # Validation framework
│       ├── __init__.py          # Validation API exports
│       ├── json_validator.py    # JSON syntax & schema validation
│       ├── text_validator.py    # Text format validation
│       ├── consistency.py       # Cross-format consistency checks
│       └── schemas.py           # JSON Schema definitions
├── tests/                       # Unit tests
│   ├── test_json_validator.py
│   ├── test_text_validator.py
│   └── test_consistency.py
├── setup.py                     # setuptools config
├── pyproject.toml               # Modern Python project config
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
└── README.md                    # This file
```

## Development

### Running Tests

```bash
cd python/
pytest tests/ -v

# With coverage
pytest tests/ --cov=cutracer --cov-report=html
```

### Type Checking

```bash
cd python/
mypy cutracer/
```

### Code Formatting

```bash
cd python/
black cutracer/ tests/
ruff check cutracer/ tests/
```

### Running All Checks

```bash
# Format code
black cutracer/ tests/

# Lint
ruff check cutracer/ tests/

# Type check
mypy cutracer/

# Run tests
pytest tests/ -v --cov=cutracer
```

## Validation Details

### JSON Trace Validation

The JSON validator checks:

- **Syntax**: Valid JSON format on each line (NDJSON)
- **Schema**: Correct field types and structure
- **Required Fields**: `type`, `ctx`, `grid_launch_id`, `trace_index`, `timestamp`, `sass`, etc.
- **Register Values**: Arrays of integers
- **CTA/Warp IDs**: Valid integer ranges

### Text Trace Validation

The text validator checks:

- **Format Patterns**: Correct field patterns (e.g., `kernel_launch_id:`, `trace_index:`)
- **Register Output**: Proper hex format (e.g., `Reg0_T00: 0x...`)
- **CTA Exit Pattern**: Valid exit messages

### Consistency Validation

The consistency validator compares:

- **Record Counts**: Same number of records in both formats
- **Content Matching**: Same kernel IDs, trace indices, SASS strings
- **Timestamp Order**: Consistent ordering between formats

## Trace Format Reference

### JSON Format (NDJSON - Mode 2)

Each line is a JSON object with the following structure:

```json
{
  "type": "reg_trace",
  "ctx": "0x58a0c0",
  "grid_launch_id": 0,
  "trace_index": 0,
  "timestamp": 1762026820167834792,
  "sass": "LDC R1, c[0x0][0x28] ;",
  "pc": 0,
  "opcode_id": 0,
  "warp": 0,
  "cta": [0, 0, 0],
  "regs": [[0, 0, 0, ...]]
}
```

### Text Format (Mode 0)

Human-readable format with fields on separate lines:

```
kernel_launch_id: 0
trace_index: 0
timestamp_ns: 1762026820167834792
sass: LDC R1, c[0x0][0x28] ;
...
```

## Contributing

1. Install development dependencies: `pip install -e ".[dev]"`
2. Make your changes
3. Run tests: `pytest tests/`
4. Run type checker: `mypy cutracer/`
5. Format code: `black cutracer/ tests/`
6. Submit a pull request

## License

BSD-3-Clause - See LICENSE file for details

## Support

For issues and questions, please open an issue on the CUTracer GitHub repository.
