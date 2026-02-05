#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Main test script for CUTracer
# This script builds CUTracer and runs the vectoradd test with validation

set -e

# Default values
DEBUG=${DEBUG:-"0"}
TEST_TYPE=${TEST_TYPE:-"all"}
TIMEOUT=${TIMEOUT:-"60"}
INSTALL_THIRD_PARTY=${INSTALL_THIRD_PARTY:-"0"} # Set to 1 to force installation
CONDA_ENV=${CONDA_ENV:-"cutracer"}
SKIP_BUILD=${SKIP_BUILD:-"0"}                   # Set to 1 to skip ALL builds
SKIP_CUTRACER_BUILD=${SKIP_CUTRACER_BUILD:-"0"} # Set to 1 to skip CUTracer build only

echo "Running CUTracer tests..."
echo "DEBUG: $DEBUG"
echo "TEST_TYPE: $TEST_TYPE"
echo "TIMEOUT: $TIMEOUT"
echo "INSTALL_THIRD_PARTY: $INSTALL_THIRD_PARTY"
echo "CONDA_ENV: $CONDA_ENV"
echo "SKIP_BUILD: $SKIP_BUILD"
echo "SKIP_CUTRACER_BUILD: $SKIP_CUTRACER_BUILD"

# Define project root path (absolute path)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Setup CUDA environment
# Use existing CUDA_HOME if set, otherwise default to /usr/local/cuda
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# Setup Python environment
export PYTHONPATH="$PROJECT_ROOT/python:$PYTHONPATH"

# Activate conda environment
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "🐍 Activating conda environment..."
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate $CONDA_ENV
  conda install -y -c conda-forge libstdcxx-ng=15.1.0
  pip install pandas

  # Install cutracer Python package in editable mode (includes all dependencies)
  echo "📦 Installing cutracer Python package..."
  pip install -e "$PROJECT_ROOT/python"
else
  echo "⚠️ Conda activation script not found, skipping."
fi

# Function to install third-party dependencies
install_third_party() {
  echo "📦 Installing third-party dependencies..."

  cd "$PROJECT_ROOT"

  # Check if install script exists
  if [ ! -f "install_third_party.sh" ]; then
    echo "❌ install_third_party.sh not found!"
    return 1
  fi

  # Make script executable and run it
  chmod +x install_third_party.sh

  if ./install_third_party.sh; then
    echo "✅ Third-party dependencies installed successfully"

    # Verify NVBit was installed
    if [ -d "third_party/nvbit" ]; then
      echo "✅ NVBit found in third_party/nvbit"
      ls -la third_party/nvbit/
    else
      echo "❌ NVBit directory not found after installation"
      return 1
    fi

    return 0
  else
    echo "❌ Failed to install third-party dependencies"
    return 1
  fi
}

# Function to build CUTracer
build_cutracer() {
  echo "🔨 Building CUTracer..."

  cd "$PROJECT_ROOT"

  # Install third-party dependencies first
  if [ "$INSTALL_THIRD_PARTY" = "1" ]; then
    if ! install_third_party; then
      echo "❌ Failed to install third-party dependencies"
      return 1
    fi
  else
    echo "⏩ Skipping installation of third-party dependencies."
  fi

  # Clean previous build
  make clean

  # Build based on debug flag
  if [ "$DEBUG" = "1" ]; then
    echo "Building in DEBUG mode..."
    DEBUG=1 make -j$(nproc)
  else
    echo "Building in RELEASE mode..."
    make -j$(nproc)
  fi

  # Verify the library was built
  if [ ! -f "$PROJECT_ROOT/lib/cutracer.so" ]; then
    echo "❌ CUTracer library not found!"
    return 1
  fi

  echo "✅ CUTracer library built successfully"
  ls -la "$PROJECT_ROOT/lib/"
  return 0
}

# Function to build vectoradd test
build_vectoradd() {
  echo "🔨 Building vectoradd test..."

  cd "$PROJECT_ROOT/tests/vectoradd"

  # Clean and build
  make clean
  make -j$(nproc)

  # Verify the test executable was built
  if [ ! -f "$PROJECT_ROOT/tests/vectoradd/vectoradd" ]; then
    echo "❌ vectoradd test not found!"
    return 1
  fi

  echo "✅ vectoradd test built successfully"
  ls -la "$PROJECT_ROOT/tests/vectoradd/vectoradd"

  return 0
}

# Function to build vectoradd_smem test (shared memory version)
build_vectoradd_smem() {
  echo "🔨 Building vectoradd_smem test..."

  cd "$PROJECT_ROOT/tests/vectoradd_smem"

  # Clean and build
  make clean
  make -j$(nproc)

  # Verify the test executable was built
  if [ ! -f "$PROJECT_ROOT/tests/vectoradd_smem/vectoradd_smem" ]; then
    echo "❌ vectoradd_smem test not found!"
    return 1
  fi

  echo "✅ vectoradd_smem test built successfully"
  ls -la "$PROJECT_ROOT/tests/vectoradd_smem/vectoradd_smem"

  return 0
}

# Function to run vectoradd_smem with a specific instrument mode
run_vectoradd_smem() {
  local instrument_mode=$1
  echo "  -> Running vectoradd_smem with CUTRACER_INSTRUMENT=$instrument_mode..."

  cd "$PROJECT_ROOT/tests/vectoradd_smem"

  # Clean up old trace files
  rm -f *.ndjson *.ndjson.zst *.log

  if ! CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_INSTRUMENT="$instrument_mode" \
       ./vectoradd_smem >cutracer_output.log 2>&1; then
    exit_code=$?
    echo "❌ vectoradd_smem with $instrument_mode failed with exit code: $exit_code"
    echo "     === CUTracer Output ==="
    cat cutracer_output.log
    return 1
  fi

  echo "  ✅ vectoradd_smem with $instrument_mode completed successfully"
  return 0
}

# Function to get the trace file from a directory
get_smem_trace_file() {
  local trace_dir=$1
  # Find the ndjson file (prefer uncompressed, fall back to zstd)
  local trace_file=$(ls -1 "$trace_dir"/kernel_*.ndjson 2>/dev/null | head -n 1)
  if [ -z "$trace_file" ]; then
    trace_file=$(ls -1 "$trace_dir"/kernel_*.ndjson.zst 2>/dev/null | head -n 1)
  fi
  echo "$trace_file"
}

# Function to verify memory values in trace
verify_mem_values() {
  local trace_file=$1
  echo "  🔍 Verifying memory values in trace..."

  if [ -z "$trace_file" ] || [ ! -f "$trace_file" ]; then
    echo "❌ Trace file not found: $trace_file"
    return 1
  fi

  # Decompress if needed
  local cat_cmd="cat"
  if [[ "$trace_file" == *.zst ]]; then
    cat_cmd="zstd -dc"
  fi

  # Check 1: trace contains 'values' field
  local values_count=$($cat_cmd "$trace_file" | jq -c 'select(.values != null)' 2>/dev/null | wc -l)

  if [ "$values_count" -lt 1 ]; then
    echo "❌ FAILED: No 'values' field found in trace"
    echo "     First few lines of trace:"
    $cat_cmd "$trace_file" | head -5
    return 1
  fi
  echo "  ✅ Found $values_count lines with values"

  # Check 2: values arrays are not empty
  local empty_values=$($cat_cmd "$trace_file" | jq -c 'select(.values != null and (.values | length) == 0)' 2>/dev/null | wc -l)

  if [ "$empty_values" -gt 0 ]; then
    echo "❌ FAILED: Found $empty_values lines with empty values array"
    return 1
  fi
  echo "  ✅ All values arrays are non-empty"

  # Check 3: values format is correct [low, high] pairs
  local valid_format=$($cat_cmd "$trace_file" | jq -c 'select(.values != null) | .values[0] | length' 2>/dev/null | head -1)

  if [ "$valid_format" != "2" ]; then
    echo "❌ FAILED: values format incorrect, expected [low, high] pairs, got length: $valid_format"
    return 1
  fi
  echo "  ✅ Values format is correct [low_32bit, high_32bit]"

  # Check 4: Verify computation results (sin²(i) + cos²(i) = 1.0)
  # 1.0 in IEEE 754 double = [0, 1072693248] or near [4294967295, 1072693247] (precision error)
  local result_ones=$($cat_cmd "$trace_file" | jq -c 'select(.sass_str | test("STG")) | .values[]? | select(.[1] == 1072693248 or .[1] == 1072693247)' 2>/dev/null | wc -l)

  if [ "$result_ones" -lt 1 ]; then
    echo "⚠️  WARNING: Expected some result values ≈ 1.0, found $result_ones"
    echo "     This may indicate incorrect value capture. Checking STG values..."
    $cat_cmd "$trace_file" | jq -c 'select(.sass_str | test("STG")) | .values[0]' 2>/dev/null | head -3
  else
    echo "  ✅ Found $result_ones values ≈ 1.0 (sin²+cos²=1 verified)"
  fi

  # Check 5: Verify addrs field is also present (mem_value_trace should have both)
  local addrs_count=$($cat_cmd "$trace_file" | jq -c 'select(.addrs != null)' 2>/dev/null | wc -l)

  if [ "$addrs_count" -lt 1 ]; then
    echo "⚠️  WARNING: No 'addrs' field found in trace (mem_value_trace should include addresses)"
  else
    echo "  ✅ Found $addrs_count lines with addrs (addresses also captured)"
  fi

  echo "  ✅ Memory value verification passed!"
  return 0
}

# Function to run the complete vectoradd test suite
test_vectoradd() {
  echo "🧪 Testing vectoradd (baseline, with CUTracer, and validation)..."
  cd "$PROJECT_ROOT/tests/vectoradd"

  # 1. Baseline test
  echo "  -> Running baseline test (without CUTracer)..."
  if ! ./vectoradd; then
    exit_code=$?
    echo "❌ vectoradd baseline test failed with exit code $exit_code"
    return 1
  fi
  echo "  ✅ vectoradd runs successfully without CUTracer"

  # 2. Test injection without instrumentation (should produce no trace files)
  echo "  -> Testing injection without CUTRACER_INSTRUMENT (should produce no trace files)..."
  # Clean up any existing trace files
  rm -f cutracer_*.zst kernel_*.log kernel_*.ndjson kernel_*.ndjson.zst

  if ! CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       ./vectoradd >injection_only_output.log 2>&1; then
    exit_code=$?
    echo "❌ vectoradd with injection only failed with exit code: $exit_code"
    cat injection_only_output.log
    return 1
  fi
  echo "  ✅ vectoradd with injection only completed successfully"

  # Verify no trace files were generated
  ZST_FILES=$(ls cutracer_*.zst 2>/dev/null || true)
  LOG_FILES=$(ls kernel_*.log 2>/dev/null || true)
  NDJSON_FILES=$(ls kernel_*.ndjson* 2>/dev/null || true)

  if [ -n "$ZST_FILES" ] || [ -n "$LOG_FILES" ] || [ -n "$NDJSON_FILES" ]; then
    echo "❌ FAILED: Trace files should NOT be generated when CUTRACER_INSTRUMENT is not set"
    echo "     Found zst files: $ZST_FILES"
    echo "     Found log files: $LOG_FILES"
    echo "     Found ndjson files: $NDJSON_FILES"
    return 1
  fi
  echo "  ✅ No trace files generated (as expected when CUTRACER_INSTRUMENT is not set)"

  # 3. Test with CUTracer
  echo "  -> Running test with CUTracer..."
  # Clean up old logs to ensure a fresh run
  rm -f *.log

  if ! TRACE_FORMAT_NDJSON=0 \
       CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_INSTRUMENT=reg_trace \
       ./vectoradd >cutracer_output.log 2>&1; then
    exit_code=$?
    echo "❌ CUTracer test failed with exit code: $exit_code"
    echo "     === CUTracer Output ==="
    cat cutracer_output.log
    return 1
  fi
  echo "  ✅ vectoradd with CUTracer completed successfully"
  echo "     === CUTracer Output ==="
  cat cutracer_output.log

  # 3. Validate output
  echo "  -> Validating CUTracer output..."
  kernel_log_file=$(ls -1 kernel_*vecAdd*.log 2>/dev/null | head -n 1)

  if [ -z "$kernel_log_file" ] || [ ! -f "$kernel_log_file" ]; then
    echo "❌ Kernel trace log file (kernel_*vecAdd*.log) not found!"
    echo "     Listing current directory contents:"
    ls -la
    return 1
  fi
  echo "  ✅ Found kernel trace log: $kernel_log_file"

  # Check for register trace output in the kernel log
  if grep -q "Reg0_T00: 0x00000000" "$kernel_log_file"; then
    echo "  ✅ Found expected register trace: Reg0_T00: 0x00000000"
  else
    echo "❌ Missing expected register trace: Reg0_T00: 0x00000000"
    echo "     === Searching for similar patterns in $kernel_log_file ==="
    grep -i "reg.*t0" "$kernel_log_file" || echo "No register patterns found"
    return 1
  fi

  # Check for CTA exit pattern in the kernel log
  new_exit_pattern="CTX 0x[0-9a-f]+ - CTA [0-9]+,[0-9]+,[0-9]+ - warp [0-9]+ - EXIT ;:"
  if grep -qE "$new_exit_pattern" "$kernel_log_file"; then
    echo "  ✅ Found expected CTA exit pattern"
    echo "     Matching line:"
    grep -m 1 -E "$new_exit_pattern" "$kernel_log_file"
  else
    echo "❌ Missing expected CTA exit pattern"
    echo "     === Searching for similar patterns in $kernel_log_file ==="
    grep -i "cta.*warp.*exit" "$kernel_log_file" || echo "No CTA exit patterns found"
    return 1
  fi

  echo "✅ All CUTracer output validation passed!"
  return 0
}

# Function to test trace formats (mode 0 and mode 2)
test_trace_formats() {
  echo "🧪 Testing all trace formats (Unified TraceWriter Implementation)..."
  echo "   Testing with combined instrumentation: reg_trace + mem_trace"
  echo "   Testing with PT2 compiled kernel filter: triton_poi_fused"
  cd "$PROJECT_ROOT/tests/py_add"

  # Initialize result tracking variables
  local mode0_status="pending"
  local mode1_status="pending"
  local mode2_status="pending"
  local mode0_file=""
  local mode1_file=""
  local mode2_file=""
  local mode0_reg_count=0
  local mode0_mem_count=0
  local mode0_total_count=0
  local mode1_reg_count=0
  local mode1_mem_count=0
  local mode1_total_count=0
  local mode2_reg_count=0
  local mode2_mem_count=0
  local mode2_total_count=0
  local cross_validation_status="pending"

  # Clean up old trace files
  rm -f *.log *.ndjson *.ndjson.zst

  # ===== Test Mode 0 (Text Format) =====
  echo ""
  echo "  📝 Testing Mode 0 (Text Format)..."

  if ! TRACE_FORMAT_NDJSON=0 \
       CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_INSTRUMENT=reg_trace,mem_trace \
       KERNEL_FILTERS=triton_poi_fused \
       python ./test_add.py >mode0_run.log 2>&1; then
    echo "    ❌ Mode 0 execution failed"
    mode0_status="failed"
  else
    # Find generated .log file (PT2 compiled Triton kernel)
    mode0_file=$(ls -1t kernel_*triton_poi_fused*.log 2>/dev/null | head -n 1)
    if [ -z "$mode0_file" ]; then
      echo "    ❌ No .log file generated"
      mode0_status="failed"
    else
      echo "    ✅ Found: $mode0_file"

      # Validate using Python module
      echo "    🔍 Validating text format..."
      if python3 "$PROJECT_ROOT/scripts/validate_trace.py" --no-color text "$mode0_file" >mode0_validation.log 2>&1; then
        mode0_status="passed"

        # Count reg_trace records (CTX lines without "kernel_launch_id")
        mode0_reg_count=$(grep "^CTX" "$mode0_file" | grep -v "kernel_launch_id" | wc -l | tr -d '[:space:]')

        # Count mem_trace records (lines with "Memory Addresses:")
        mode0_mem_count=$(grep -ac "Memory Addresses:" "$mode0_file" 2>/dev/null | tr -d '[:space:]')

        # Total count
        mode0_total_count=$((mode0_reg_count + mode0_mem_count))

        echo "    ✅ Mode 0 validation passed"
        echo "       📊 Record breakdown:"
        echo "          reg_trace:  $mode0_reg_count records"
        echo "          mem_trace:  $mode0_mem_count records"
        echo "          Total:      $mode0_total_count records"
        echo "       File size: $(stat -f%z "$mode0_file" 2>/dev/null || stat -c%s "$mode0_file") bytes"
      else
        echo "    ❌ Mode 0 validation failed"
        echo "    === Validation errors ==="
        cat mode0_validation.log
        mode0_status="failed"
      fi
    fi
  fi

  # ===== Test Mode 2 (NDJSON Format) =====
  echo ""
  echo "  📊 Testing Mode 2 (NDJSON Format)..."

  # Clean old ndjson files
  rm -f *.ndjson

  if ! TRACE_FORMAT_NDJSON=2 \
       CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_INSTRUMENT=reg_trace,mem_trace \
       KERNEL_FILTERS=triton_poi_fused \
       python ./test_add.py >mode2_run.log 2>&1; then
    echo "    ❌ Mode 2 execution failed"
    mode2_status="failed"
  else
    # Find generated .ndjson file (PT2 compiled Triton kernel)
    mode2_file=$(ls -1t kernel_*triton_poi_fused*.ndjson 2>/dev/null | head -n 1)
    if [ -z "$mode2_file" ]; then
      echo "    ❌ No .ndjson file generated"
      mode2_status="failed"
    else
      echo "    ✅ Found: $mode2_file"

      # Validate using Python module
      echo "    🔍 Validating JSON format..."
      if python3 "$PROJECT_ROOT/scripts/validate_trace.py" --no-color json "$mode2_file" >mode2_validation.log 2>&1; then
        mode2_status="passed"

        # Count each trace type separately (note: JSON has no spaces after colons)
        mode2_reg_count=$(grep -ac '"type":"reg_trace"' "$mode2_file" 2>/dev/null | tr -d '[:space:]')
        mode2_mem_count=$(grep -ac '"type":"mem_trace"' "$mode2_file" 2>/dev/null | tr -d '[:space:]')
        mode2_total_count=$(wc -l < "$mode2_file" 2>/dev/null | tr -d '[:space:]')

        echo "    ✅ Mode 2 validation passed"
        echo "       📊 Record breakdown:"
        echo "          reg_trace:  $mode2_reg_count records"
        echo "          mem_trace:  $mode2_mem_count records"
        echo "          Total:      $mode2_total_count records"
        echo "       File size: $(stat -f%z "$mode2_file" 2>/dev/null || stat -c%s "$mode2_file") bytes"

        # Show first record of each type (formatted)
        echo "       First reg_trace record (formatted):"
        grep -a '"type":"reg_trace"' "$mode2_file" | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'

        if [ "$mode2_mem_count" -gt 0 ]; then
          echo "       First mem_trace record (formatted):"
          grep -a '"type":"mem_trace"' "$mode2_file" | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'
        fi
      else
        echo "    ❌ Mode 2 validation failed"
        echo "    === Validation errors ==="
        cat mode2_validation.log
        mode2_status="failed"
      fi
    fi
  fi

  # ===== Test Mode 1 (NDJSON + Zstd Compression) =====
  echo ""
  echo "  📦 Testing Mode 1 (NDJSON + Zstd)..."

  # Clean old zst files
  rm -f *.ndjson.zst mode1_decompressed.ndjson

  if ! TRACE_FORMAT_NDJSON=1 \
       CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_INSTRUMENT=reg_trace,mem_trace \
       KERNEL_FILTERS=triton_poi_fused \
       python ./test_add.py >mode1_run.log 2>&1; then
    echo "    ❌ Mode 1 execution failed"
    mode1_status="failed"
  else
    # Find generated .ndjson.zst file (PT2 compiled Triton kernel)
    mode1_file=$(ls -1t kernel_*triton_poi_fused*.ndjson.zst 2>/dev/null | head -n 1)
    if [ -z "$mode1_file" ]; then
      echo "    ❌ No .ndjson.zst file generated"
      mode1_status="failed"
    else
      echo "    ✅ Found: $mode1_file"

      # Get compressed file size
      local compressed_size=$(stat -c%s "$mode1_file" 2>/dev/null || stat -f%z "$mode1_file")
      echo "       Compressed file size: $compressed_size bytes"

      # Decompress for validation
      echo "    🔓 Decompressing..."
      if ! zstd -d "$mode1_file" -o mode1_decompressed.ndjson --force >mode1_decomp.log 2>&1; then
        echo "    ❌ Decompression failed"
        echo "    === Decompression errors ==="
        cat mode1_decomp.log
        mode1_status="failed"
      else
        # Get decompressed file size
        local decompressed_size=$(stat -c%s mode1_decompressed.ndjson 2>/dev/null || stat -f%z mode1_decompressed.ndjson)

        # Calculate compression ratio
        local ratio=$(awk "BEGIN {printf \"%.1f\", ($decompressed_size / $compressed_size)}")

        echo "    ✅ Decompression successful"
        echo "       Decompressed size: $decompressed_size bytes"
        echo "       Compression ratio: ${ratio}x"

        # Validate decompressed JSON
        echo "    🔍 Validating JSON format..."
        if python3 "$PROJECT_ROOT/scripts/validate_trace.py" --no-color json mode1_decompressed.ndjson >mode1_validation.log 2>&1; then
          mode1_status="passed"

          # Count each trace type separately
          mode1_reg_count=$(grep -ac '"type":"reg_trace"' mode1_decompressed.ndjson 2>/dev/null | tr -d '[:space:]')
          mode1_mem_count=$(grep -ac '"type":"mem_trace"' mode1_decompressed.ndjson 2>/dev/null | tr -d '[:space:]')
          mode1_total_count=$(wc -l < mode1_decompressed.ndjson 2>/dev/null | tr -d '[:space:]')

          echo "    ✅ Mode 1 validation passed"
          echo "       📊 Record breakdown:"
          echo "          reg_trace:  $mode1_reg_count records"
          echo "          mem_trace:  $mode1_mem_count records"
          echo "          Total:      $mode1_total_count records"

          # Show first record of each type (formatted)
          echo "       First reg_trace record (formatted):"
          grep -a '"type":"reg_trace"' mode1_decompressed.ndjson | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'

          if [ "$mode1_mem_count" -gt 0 ]; then
            echo "       First mem_trace record (formatted):"
            grep -a '"type":"mem_trace"' mode1_decompressed.ndjson | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'
          fi
        else
          echo "    ❌ Mode 1 validation failed"
          echo "    === Validation errors ==="
          cat mode1_validation.log
          mode1_status="failed"
        fi
      fi
    fi
  fi

  # ===== Cross-Format Validation =====
  echo ""
  echo "  🔄 Validating cross-format consistency..."

  if [ "$mode0_status" = "passed" ] && [ "$mode2_status" = "passed" ]; then
    # Assume cross-validation will pass, set to failed if any check fails
    cross_validation_status="passed"

    # Compare record counts by type
    echo "    📊 Comparing record counts..."
    echo "       Mode 0 breakdown: reg=$mode0_reg_count, mem=$mode0_mem_count, total=$mode0_total_count"
    echo "       Mode 2 breakdown: reg=$mode2_reg_count, mem=$mode2_mem_count, total=$mode2_total_count"
    echo ""

    # Compare total counts (no tolerance - must be exact match)
    if [ "$mode0_total_count" -gt 0 ]; then
      local diff=$((mode2_total_count - mode0_total_count))

      echo "       Total difference: $diff (no tolerance - exact match required)"

      if [ "$diff" -eq 0 ]; then
        echo "    ✅ Total record counts are consistent"
      else
        echo "    ❌ ERROR: Total record count difference ($diff) - exact match required"
        cross_validation_status="failed"
      fi
    else
      echo "    ❌ ERROR: Mode 0 has zero records, cannot compare"
      cross_validation_status="failed"
    fi

    # Compare reg_trace counts separately (no tolerance - exact match)
    echo ""
    echo "    🔍 Detailed comparison by trace type:"
    if [ "$mode0_reg_count" -gt 0 ] && [ "$mode2_reg_count" -gt 0 ]; then
      local reg_diff=$((mode2_reg_count - mode0_reg_count))

      echo "       reg_trace: mode0=$mode0_reg_count, mode2=$mode2_reg_count, diff=$reg_diff"
      if [ "$reg_diff" -eq 0 ]; then
        echo "       ✅ reg_trace counts consistent"
      else
        echo "       ❌ ERROR: reg_trace difference ($reg_diff) - exact match required"
        cross_validation_status="failed"
      fi
    else
      echo "       ⚠️  reg_trace: Cannot compare (one or both modes have 0 records)"
    fi

    # Compare mem_trace counts separately (no tolerance - exact match)
    if [ "$mode0_mem_count" -gt 0 ] && [ "$mode2_mem_count" -gt 0 ]; then
      local mem_diff=$((mode2_mem_count - mode0_mem_count))

      echo "       mem_trace: mode0=$mode0_mem_count, mode2=$mode2_mem_count, diff=$mem_diff"
      if [ "$mem_diff" -eq 0 ]; then
        echo "       ✅ mem_trace counts consistent"
      else
        echo "       ❌ ERROR: mem_trace difference ($mem_diff) - exact match required"
        cross_validation_status="failed"
      fi
    elif [ "$mode0_mem_count" -eq 0 ] && [ "$mode2_mem_count" -eq 0 ]; then
      echo "       ℹ️  mem_trace: Both modes have 0 records (no mem_trace data)"
    else
      echo "       ❌ ERROR: mem_trace: Inconsistent (mode0=$mode0_mem_count, mode2=$mode2_mem_count)"
      cross_validation_status="failed"
    fi

    # Run comprehensive comparison using Python module
    echo ""
    echo "    🔍 Running comprehensive format comparison..."
    if python3 "$PROJECT_ROOT/scripts/validate_trace.py" --no-color compare "$mode0_file" "$mode2_file" >compare_result.log 2>&1; then
      echo "    ✅ Format comparison passed"
    else
      echo "    ❌ ERROR: Format comparison failed:"
      cat compare_result.log
      cross_validation_status="failed"
    fi
  else
    echo "    ⏭️  Skipping cross-format validation (one or more formats failed)"
    cross_validation_status="skipped"
  fi

  # Add Mode 1 to cross-validation if passed
  if [ "$mode1_status" = "passed" ]; then
    echo ""
    echo "  🔄 Validating Mode 1 vs Mode 2 consistency..."
    echo "     Mode 1 breakdown: reg=$mode1_reg_count, mem=$mode1_mem_count, total=$mode1_total_count"
    echo "     Mode 2 breakdown: reg=$mode2_reg_count, mem=$mode2_mem_count, total=$mode2_total_count"

    if [ "$mode1_total_count" -eq "$mode2_total_count" ]; then
      echo "     ✅ Mode 1 and Mode 2 record counts match"
    else
      echo "     ❌ ERROR: Mode 1 vs Mode 2 record count mismatch (mode1=$mode1_total_count, mode2=$mode2_total_count) - exact match required"
      cross_validation_status="failed"
    fi

    # Compare by type
    if [ "$mode1_reg_count" -eq "$mode2_reg_count" ] && [ "$mode1_mem_count" -eq "$mode2_mem_count" ]; then
      echo "     ✅ reg_trace and mem_trace counts match between Mode 1 and Mode 2"
    else
      echo "     ❌ ERROR: Type-specific counts differ between Mode 1 and Mode 2 (reg: mode1=$mode1_reg_count vs mode2=$mode2_reg_count, mem: mode1=$mode1_mem_count vs mode2=$mode2_mem_count) - exact match required"
      cross_validation_status="failed"
    fi

    # Calculate and display compression statistics
    if [ -n "$mode1_file" ] && [ -n "$mode2_file" ]; then
      local mode1_size=$(stat -c%s "$mode1_file" 2>/dev/null || stat -f%z "$mode1_file")
      local mode2_size=$(stat -c%s "$mode2_file" 2>/dev/null || stat -f%z "$mode2_file")
      local ratio=$(awk "BEGIN {printf \"%.1f\", ($mode2_size / $mode1_size)}")
      local saved_bytes=$((mode2_size - mode1_size))
      local saved_percent=$(awk "BEGIN {printf \"%.1f\", (100.0 * $saved_bytes / $mode2_size)}")

      echo ""
      echo "  📊 Compression Statistics:"
      echo "     Mode 2 (uncompressed): $(numfmt --to=iec-i --suffix=B $mode2_size 2>/dev/null || echo "$mode2_size bytes")"
      echo "     Mode 1 (compressed):   $(numfmt --to=iec-i --suffix=B $mode1_size 2>/dev/null || echo "$mode1_size bytes")"
      echo "     Compression ratio:     ${ratio}x"
      echo "     Space saved:           $(numfmt --to=iec-i --suffix=B $saved_bytes 2>/dev/null || echo "$saved_bytes bytes") (${saved_percent}%)"
    fi
  fi

  # ===== Final Report =====
  echo ""
  echo "  ═══════════════════════════════════════════"
  echo "  📋 Test Summary"
  echo "  ═══════════════════════════════════════════"
  echo "  Mode 0 (Text):          $mode0_status"
  echo "  Mode 1 (NDJSON+Zstd):   $mode1_status"
  echo "  Mode 2 (NDJSON):        $mode2_status"
  echo "  Cross-Validation:       $cross_validation_status"
  echo "  ═══════════════════════════════════════════"

  # Determine overall result - all four must pass
  if [ "$mode0_status" = "passed" ] && \
     [ "$mode1_status" = "passed" ] && \
     [ "$mode2_status" = "passed" ] && \
     [ "$cross_validation_status" = "passed" ]; then
    echo "  ✅ All trace format tests passed!"
    return 0
  else
    echo "  ❌ Some trace format tests failed"
    return 1
  fi
}

test_py_add_with_kernel_filters() {
  echo "🧪 Testing py_add with kernel filters..."
  cd "$PROJECT_ROOT/tests/py_add"

  # Clean up old logs to ensure a fresh run
  rm -f *.log

  # Run the test with KERNEL_FILTERS enabled
  if ! TRACE_FORMAT_NDJSON=0 \
       CUDA_INJECTION64_PATH=$PROJECT_ROOT/lib/cutracer.so \
       CUTRACER_INSTRUMENT=reg_trace \
       KERNEL_FILTERS=triton_poi_fused \
       python ./test_add.py >py_add_output.log 2>&1; then
    echo "❌ Python script test_add.py failed to execute."
    echo "     === Python script output ==="
    cat py_add_output.log
    cd "$PROJECT_ROOT"
    return 1
  fi
  echo "     === Python script output ==="
  cat py_add_output.log

  # Find logs that match the kernel filter
  matching_logs=$(ls kernel*triton_poi_fused*.log 2>/dev/null)
  if [ -z "$matching_logs" ]; then
    echo "❌ Test failed: No log file generated for kernel containing 'triton_poi_fused'."
    echo "     Listing current directory contents:"
    ls -la
    cd "$PROJECT_ROOT"
    return 1
  fi

  # Ensure ALL generated kernel logs match the filter
  all_kernel_logs=$(ls kernel*.log 2>/dev/null)
  unmatched_logs=$(echo "$all_kernel_logs" | grep -v "triton_poi_fused")

  if [ -n "$unmatched_logs" ]; then
    echo "❌ Test failed: Found kernel logs that should have been filtered out:"
    echo "$unmatched_logs"
    cd "$PROJECT_ROOT"
    return 1
  fi

  echo "✅ All generated kernel logs correctly match the filter."

  # Check the content of the first matching log
  first_log=$(echo "$matching_logs" | head -n 1)
  echo "🔎 Inspecting log file: $first_log"

  # Use same pattern as vectoradd test for consistency
  # Triton kernels use "EXIT ;:" format (no predicate prefix like @P0)
  exit_pattern="CTA [0-9]+,[0-9]+,[0-9]+ - warp [0-9]+ - .*EXIT"
  if grep -qE "$exit_pattern" "$first_log"; then
    echo "✅ Test successful: Found EXIT pattern in the log."
    echo "   Matching line:"
    grep -m 1 -E "$exit_pattern" "$first_log"
  else
    echo "❌ Test failed: Did not find EXIT pattern in $first_log."
    echo "   Expected pattern: $exit_pattern"
    echo "   === Searching for similar patterns ==="
    grep -i "exit" "$first_log" | head -5 || echo "No EXIT patterns found"
    cd "$PROJECT_ROOT"
    return 1
  fi

  cd "$PROJECT_ROOT"
}

# Function to run proton tests
test_proton() {
  echo "🧪 Testing proton..."
  cd "$PROJECT_ROOT/tests/proton_tests"

  # Clean up old logs, traces, and CSVs to ensure a fresh run
  rm -f *.log *.csv *.chrome_trace

  # --- Start Combined Trace Analysis Test ---
  echo "  -> Step 1: Generating instruction histogram with kernel filters..."

  # First run: Execute with CUTracer to generate instruction histogram and tracer log.
  # We are using KERNEL_FILTERS to only trace 'add_kernel'.
  if ! TRACE_FORMAT_NDJSON=0 \
       CUDA_INJECTION64_PATH=$PROJECT_ROOT/lib/cutracer.so \
       CUTRACER_ANALYSIS=proton_instr_histogram \
       KERNEL_FILTERS=add_kernel \
       python ./vector-add-instrumented.py >proton_instr_output.log 2>&1; then
    echo "❌ Proton test (step 1: kernel filter run) failed to execute."
    echo "     === Proton test (kernel filter) output ==="
    cat proton_instr_output.log
    cd "$PROJECT_ROOT"
    return 1
  fi

  # Find the generated files from the first run. We expect one of each.
  instr_hist_csv=$(ls -1 kernel_*_add_kernel_hist.csv 2>/dev/null | head -n 1)
  cutracer_log=$(ls -1 cutracer_main_*.log 2>/dev/null | head -n 1)

  if [ -z "$instr_hist_csv" ] || [ ! -f "$instr_hist_csv" ]; then
    echo "❌ Instruction histogram CSV file (kernel_*_add_kernel_hist.csv) not found!"
    echo "     Listing current directory contents:"
    ls -la
    cd "$PROJECT_ROOT"
    return 1
  fi
  echo "  ✅ Found instruction histogram: $instr_hist_csv"

  if [ -z "$cutracer_log" ] || [ ! -f "$cutracer_log" ]; then
    echo "❌ CUTracer log file (cutracer_main_*.log) not found!"
    echo "     Listing current directory contents:"
    ls -la
    cd "$PROJECT_ROOT"
    return 1
  fi
  echo "  ✅ Found CUTracer log: $cutracer_log"

  # --- CSV Validation ---
  echo "  -> Validating histogram CSV header..."
  expected_header="warp_id,region_id,instruction,count"
  actual_header=$(head -n 1 "$instr_hist_csv")
  if [ "$actual_header" != "$expected_header" ]; then
    echo "❌ CSV header does not match expected header."
    echo "     Expected: $expected_header"
    echo "     Actual:   $actual_header"
    cd "$PROJECT_ROOT"
    return 1
  fi
  echo "  ✅ CSV header is correct."
  # --- End CSV Validation ---

  # --- Step 2: Generate clean Chrome Trace ---
  echo "  -> Step 2: Generating clean Chrome Trace..."

  # Second run: Execute without CUTracer to get a clean trace with accurate timing.
  # This is critical because the tracer can interfere with cycle counts.
  if ! python ./vector-add-instrumented.py >python_runner.log 2>&1; then
    echo "❌ Python runner (step 2: chrome trace generation) failed to execute."
    echo "     === Python runner output ==="
    cat python_runner.log
    cd "$PROJECT_ROOT"
    return 1
  fi

  if [ ! -f "vector.chrome_trace" ]; then
    echo "❌ Chrome trace file (vector.chrome_trace) not found!"
    echo "     Listing current directory contents:"
    ls -la
    cd "$PROJECT_ROOT"
    return 1
  fi
  echo "  ✅ Found vector.chrome_trace"

  # --- Step 3: Run the parsing script ---
  echo "  -> Step 3: Running parse_instr_hist_trace.py..."
  if ! python "$PROJECT_ROOT/scripts/parse_instr_hist_trace.py" --chrome-trace ./vector.chrome_trace --cutracer-trace "$instr_hist_csv" --cutracer-log "$cutracer_log" --output vectoradd_ipc.csv >parse_script_output.log 2>&1; then
    echo "❌ Parse script (step 3) failed to execute."
    echo "     === Parse script output ==="
    cat parse_script_output.log
    cd "$PROJECT_ROOT"
    return 1
  fi
  echo "  ✅ Parsing script executed successfully."

  # --- Step 4: Validate the output ---
  echo "  -> Step 4: Validating output file 'vectoradd_ipc.csv'..."
  if [ ! -f "vectoradd_ipc.csv" ]; then
    echo "❌ Output file 'vectoradd_ipc.csv' not found!"
    echo "     Listing current directory contents:"
    ls -la
    cd "$PROJECT_ROOT"
    return 1
  fi

  line_count=$(wc -l <"vectoradd_ipc.csv")
  if [ "$line_count" -le 5 ]; then
    echo "❌ Output file 'vectoradd_ipc.csv' has $line_count lines, which is not more than 5."
    echo "     --- Full content of vectoradd_ipc.csv ---"
    cat vectoradd_ipc.csv
    echo "     -----------------------------"
    cd "$PROJECT_ROOT"
    return 1
  fi

  echo "  ✅ Output file 'vectoradd_ipc.csv' has more than 5 lines ($line_count lines)."
  echo "     --- First 5 lines of vectoradd_ipc.csv ---"
  head -n 5 "vectoradd_ipc.csv"
  echo "     ------------------------------"
  echo "✅ Combined trace analysis test passed!"

  cd "$PROJECT_ROOT"
  return 0
}

# Function to run hang detection test
test_hang_test() {
  echo "🧪 Testing hang detection..."
  cd "$PROJECT_ROOT/tests/hang_test"

  # Clean up old logs to ensure a fresh run
  rm -f *.log *.csv *.chrome_trace

  # Require test_hang.py to exist
  if [ ! -f "test_hang.py" ]; then
    echo "❌ test_hang.py not found."
    echo "     Listing current directory contents:"
    ls -la
    cd "$PROJECT_ROOT"
    return 1
  fi

  # Run with CUTracer deadlock detection and a timeout guard
  if ! timeout "$TIMEOUT" env \
       TRACE_FORMAT_NDJSON=0 \
       CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_ANALYSIS=deadlock_detection \
       KERNEL_FILTERS=add_kernel \
       python "./test_hang.py" >hang_output.log 2>&1; then
    exit_code=$?
    echo "     === Hang test output ==="
    cat hang_output.log
    if [ "$exit_code" -eq 124 ]; then
      echo "❌ Hang test timed out (no detection within $TIMEOUT s)."
      cd "$PROJECT_ROOT"
      return 1
    fi
    if grep -q "Deadlock sustained" hang_output.log; then
      echo "  ✅ Hang detection confirmed (process terminated by tracer)."
    else
      echo "❌ Hang test failed without detection (exit $exit_code)."
      cd "$PROJECT_ROOT"
      return 1
    fi
  fi

  cd "$PROJECT_ROOT"
  return 0
}

# Function to run Python module unit tests
test_python_module() {
  echo "🧪 Running Python module unit tests..."
  cd "$PROJECT_ROOT/python"

  if ! python -m unittest discover -s tests -v; then
    echo "❌ Python module unit tests failed"
    cd "$PROJECT_ROOT"
    return 1
  fi

  echo "✅ Python module unit tests passed!"
  cd "$PROJECT_ROOT"
  return 0
}

# Function to test mem_value_trace (memory value tracing)
test_mem_value_trace() {
  echo "🧪 Testing memory value tracing (mem_value_trace)..."

  # Build vectoradd_smem test (uses shared memory, good for value verification)
  if ! build_vectoradd_smem; then
    echo "❌ Failed to build vectoradd_smem"
    return 1
  fi

  # Run with mem_value_trace mode
  if ! run_vectoradd_smem "mem_value_trace"; then
    echo "❌ Failed to run vectoradd_smem with mem_value_trace"
    return 1
  fi

  # Get the trace file
  local trace_file=$(get_smem_trace_file "$PROJECT_ROOT/tests/vectoradd_smem")
  if [ -z "$trace_file" ]; then
    echo "❌ No trace file found in vectoradd_smem directory"
    ls -la "$PROJECT_ROOT/tests/vectoradd_smem"
    return 1
  fi
  echo "  📄 Found trace file: $trace_file"

  # Verify memory values
  if ! verify_mem_values "$trace_file"; then
    echo "❌ Memory value verification failed"
    return 1
  fi

  echo "✅ mem_value_trace test passed!"
  cd "$PROJECT_ROOT"
  return 0
}

# Function to run all tests
run_all_tests() {
  echo "🚀 Running all CUTracer tests..."

  # Run Python module tests first (no build required)
  if ! test_python_module; then
    echo "❌ Python module tests failed"
    return 1
  fi

  # Test phase
  build_vectoradd
  if ! test_vectoradd; then
    echo "❌ vectoradd test failed"
    return 1
  fi

  if ! test_py_add_with_kernel_filters; then
    echo "❌ Python script test_add.py failed to execute."
    return 1
  fi

  if ! test_hang_test; then
    echo "❌ hang test failed"
    return 1
  fi

  if ! test_trace_formats; then
    echo "❌ trace format tests failed"
    return 1
  fi

  build_vectoradd_smem
  if ! test_mem_value_trace; then
    echo "❌ mem_value_trace test failed"
    return 1
  fi

  echo "🎉 All tests passed successfully!"
  return 0
}

# Build CUTracer (can be skipped with SKIP_BUILD=1 or SKIP_CUTRACER_BUILD=1)
if [ "$SKIP_BUILD" = "1" ]; then
  echo "⏭️ Skipping ALL builds (SKIP_BUILD=1)"
elif [ "$SKIP_CUTRACER_BUILD" = "1" ]; then
  echo "⏭️ Skipping CUTracer build (SKIP_CUTRACER_BUILD=1)"
else
  build_cutracer
fi

# Main execution
# Helper function to conditionally run build_vectoradd
maybe_build_vectoradd() {
  if [ "$SKIP_BUILD" = "1" ]; then
    echo "⏭️ Skipping vectoradd build (SKIP_BUILD=1)"
    return 0
  fi
  build_vectoradd
}

# Helper function to conditionally run build_vectoradd_smem
maybe_build_vectoradd_smem() {
  if [ "$SKIP_BUILD" = "1" ]; then
    echo "⏭️ Skipping vectoradd_smem build (SKIP_BUILD=1)"
    return 0
  fi
  build_vectoradd_smem
}

case "$TEST_TYPE" in
"build-only")
  build_cutracer && build_vectoradd && build_vectoradd_smem
  ;;
"vectoradd")
  maybe_build_vectoradd && test_vectoradd && test_py_add_with_kernel_filters
  ;;
"trace-formats")
  test_trace_formats
  ;;
"mem-value")
  maybe_build_vectoradd_smem && test_mem_value_trace
  ;;
"proton")
  test_proton
  ;;
"hang")
  test_hang_test
  ;;
"python-module")
  test_python_module
  ;;
"all" | *)
  run_all_tests
  ;;
esac

exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "✅ Test suite completed successfully!"
else
  echo "❌ Test suite failed with exit code $exit_code"
fi

exit $exit_code
