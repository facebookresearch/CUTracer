# CUTracer Python Module - Quick Start Guide

This guide will help you get started with the CUTracer Python validation tools.

## Installation

### Option 1: Quick Installation (Development Mode)

```bash
cd /home/yhao/CUTracer_oss/python
pip install -e .
```

### Option 2: Install with Development Dependencies

```bash
cd /home/yhao/CUTracer_oss/python
pip install -e ".[dev]"
```

### Option 3: Install Only Runtime Dependencies

```bash
cd /home/yhao/CUTracer_oss/python
pip install -r requirements.txt
```

## Quick Test

### Test the CLI Tool

After installation, test the CLI tool with an existing trace file:

```bash
# Test JSON validation
python /home/yhao/CUTracer_oss/scripts/validate_trace.py json \
    /home/yhao/CUTracer_oss/tests/vectoradd/kernel_dcd76e64b30810e4_iter0__Z6vecAddPdS_S_i.ndjson

# Test with verbose output
python /home/yhao/CUTracer_oss/scripts/validate_trace.py json \
    /home/yhao/CUTracer_oss/tests/vectoradd/kernel_dcd76e64b30810e4_iter0__Z6vecAddPdS_S_i.ndjson \
    --verbose

# Get JSON output (for CI integration)
python /home/yhao/CUTracer_oss/scripts/validate_trace.py json \
    /home/yhao/CUTracer_oss/tests/vectoradd/kernel_dcd76e64b30810e4_iter0__Z6vecAddPdS_S_i.ndjson \
    --output json
```

### Test the Python API

```python
import sys
sys.path.insert(0, '/home/yhao/CUTracer_oss/python')

from cutracer.validation import validate_json_trace

# Validate a JSON trace file
result = validate_json_trace(
    "/home/yhao/CUTracer_oss/tests/vectoradd/kernel_dcd76e64b30810e4_iter0__Z6vecAddPdS_S_i.ndjson"
)

print(f"Valid: {result['valid']}")
print(f"Records: {result['record_count']}")
print(f"File size: {result['file_size']} bytes")

if not result['valid']:
    print(f"Errors: {result['errors']}")
```

## Usage Examples

### 1. Validate JSON Trace

```bash
python scripts/validate_trace.py json trace.ndjson
```

Expected output:
```
======================================================================
Validating JSON Trace: trace.ndjson
======================================================================

✓ JSON trace validation passed
ℹ Record count: 609
ℹ File size: 155.31 KB
```

### 2. Validate Text Trace

```bash
python scripts/validate_trace.py text trace.log
```

### 3. Compare Formats

```bash
python scripts/validate_trace.py compare trace.log trace.ndjson
```

Expected output:
```
======================================================================
Comparing Trace Formats
======================================================================

ℹ Text: trace.log
ℹ JSON: trace.ndjson

Results:
✓ Text format valid
✓ JSON format valid (records: 609)
ℹ Text size: 45.23 KB
ℹ JSON size: 155.31 KB

✓ Formats are consistent
```

### 4. Verbose Output

```bash
python scripts/validate_trace.py json trace.ndjson --verbose
```

### 5. CI-Friendly JSON Output

```bash
python scripts/validate_trace.py json trace.ndjson --output json
```

Example output:
```json
{
  "valid": true,
  "record_count": 609,
  "file_size": 159040,
  "errors": []
}
```

## Python API Usage

### Import the Module

```python
from cutracer.validation import (
    validate_json_trace,
    validate_text_trace,
    compare_trace_formats
)
```

### Validate JSON Trace

```python
result = validate_json_trace("kernel_trace.ndjson")

if result["valid"]:
    print(f"✓ Valid trace with {result['record_count']} records")
else:
    print(f"✗ Validation failed:")
    for error in result["errors"]:
        print(f"  - {error}")
```

### Validate Text Trace

```python
result = validate_text_trace("kernel_trace.log")

if result["valid"]:
    print(f"✓ Valid text trace")
    print(f"  File size: {result['file_size']} bytes")
else:
    print(f"✗ Validation failed:")
    for error in result["errors"]:
        print(f"  - {error}")
```

### Compare Two Formats

```python
from pathlib import Path

result = compare_trace_formats(
    Path("kernel_trace.log"),
    Path("kernel_trace.ndjson")
)

print(f"Text valid: {result['text_valid']}")
print(f"JSON valid: {result['json_valid']}")
print(f"Consistent: {result['consistent']}")

if not result["consistent"]:
    print("Differences:")
    for diff in result["differences"]:
        print(f"  - {diff}")
```

## Exit Codes

The CLI tool uses standard exit codes:

- **0**: Success - validation passed
- **1**: Validation failed - errors found
- **2**: File not found or I/O error
- **130**: Interrupted by user (Ctrl+C)

Example usage in a bash script:

```bash
#!/bin/bash

if python scripts/validate_trace.py json trace.ndjson; then
    echo "Validation passed!"
else
    exit_code=$?
    echo "Validation failed with exit code: $exit_code"
    exit $exit_code
fi
```

## Integration with CI/CD

### Shell Script Integration

```bash
#!/bin/bash
set -e

# Run validation
python scripts/validate_trace.py json trace.ndjson --output json > validation_result.json

# Check results
if python -c "import json; result = json.load(open('validation_result.json')); exit(0 if result['valid'] else 1)"; then
    echo "✓ Trace validation passed"
else
    echo "✗ Trace validation failed"
    cat validation_result.json
    exit 1
fi
```

### Python Integration

```python
import subprocess
import json

def validate_trace_file(filepath):
    """Validate a trace file and return results"""
    result = subprocess.run(
        ["python", "scripts/validate_trace.py", "json", filepath, "--output", "json"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        return None

# Usage
result = validate_trace_file("trace.ndjson")
if result and result["valid"]:
    print(f"✓ Validation passed: {result['record_count']} records")
else:
    print("✗ Validation failed")
```

## Troubleshooting

### Import Error: No module named 'cutracer'

Make sure you've installed the package:

```bash
cd /home/yhao/CUTracer_oss/python
pip install -e .
```

Or add the path to your PYTHONPATH:

```bash
export PYTHONPATH="/home/yhao/CUTracer_oss/python:$PYTHONPATH"
```

### Import Error: No module named 'jsonschema'

Install the required dependencies:

```bash
pip install jsonschema
```

Or install all requirements:

```bash
cd /home/yhao/CUTracer_oss/python
pip install -r requirements.txt
```

### File Not Found Error

Make sure to use absolute paths or correct relative paths:

```bash
# Absolute path (recommended)
python scripts/validate_trace.py json /full/path/to/trace.ndjson

# Relative path (from project root)
cd /home/yhao/CUTracer_oss
python scripts/validate_trace.py json tests/vectoradd/kernel_*.ndjson
```

## Next Steps

1. **Run Tests**: See `python/README.md` for testing instructions
2. **Explore API**: Check the module docstrings for detailed API documentation
3. **Integrate with CI**: Add validation to your CI pipeline
4. **Contribute**: See `CONTRIBUTING.md` for contribution guidelines

## Support

For issues or questions:
- Check `python/README.md` for detailed documentation
- Open an issue on GitHub
- See design document: `ai_discussions/cutracer/pr2.5_python_module_design.md`
