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

echo "Running CUTracer tests..."
echo "DEBUG: $DEBUG"
echo "TEST_TYPE: $TEST_TYPE"
echo "TIMEOUT: $TIMEOUT"
echo "INSTALL_THIRD_PARTY: $INSTALL_THIRD_PARTY"
echo "CONDA_ENV: $CONDA_ENV"

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

  # Install cutracer Python package dependencies
  echo "📦 Installing cutracer Python package dependencies..."
  pip install jsonschema>=4.0.0
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

  # 2. Test with CUTracer
  echo "  -> Running test with CUTracer..."
  # Clean up old logs to ensure a fresh run
  rm -f *.log

  if ! CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" CUTRACER_INSTRUMENT=reg_trace ./vectoradd >cutracer_output.log 2>&1; then
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
  cd "$PROJECT_ROOT/tests/vectoradd"

  # Initialize result tracking variables
  local mode0_status="pending"
  local mode2_status="pending"
  local mode0_file=""
  local mode2_file=""
  local mode0_reg_count=0
  local mode0_mem_count=0
  local mode0_total_count=0
  local mode2_reg_count=0
  local mode2_mem_count=0
  local mode2_total_count=0

  # Clean up old trace files
  rm -f *.log *.ndjson

  # ===== Test Mode 0 (Text Format) =====
  echo ""
  echo "  📝 Testing Mode 0 (Text Format)..."

  if ! TRACE_FORMAT_NDJSON=0 \
       CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_INSTRUMENT=reg_trace,mem_trace \
       ./vectoradd >mode0_run.log 2>&1; then
    echo "    ❌ Mode 0 execution failed"
    mode0_status="failed"
  else
    # Find generated .log file
    mode0_file=$(ls -1t kernel_*vecAdd*.log 2>/dev/null | head -n 1)
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
        mode0_mem_count=$(grep -c "Memory Addresses:" "$mode0_file" 2>/dev/null | tr -d '[:space:]')

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
       ./vectoradd >mode2_run.log 2>&1; then
    echo "    ❌ Mode 2 execution failed"
    mode2_status="failed"
  else
    # Find generated .ndjson file
    mode2_file=$(ls -1t kernel_*vecAdd*.ndjson 2>/dev/null | head -n 1)
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
        mode2_reg_count=$(grep -c '"type":"reg_trace"' "$mode2_file" 2>/dev/null | tr -d '[:space:]')
        mode2_mem_count=$(grep -c '"type":"mem_trace"' "$mode2_file" 2>/dev/null | tr -d '[:space:]')
        mode2_total_count=$(wc -l < "$mode2_file" 2>/dev/null | tr -d '[:space:]')

        echo "    ✅ Mode 2 validation passed"
        echo "       📊 Record breakdown:"
        echo "          reg_trace:  $mode2_reg_count records"
        echo "          mem_trace:  $mode2_mem_count records"
        echo "          Total:      $mode2_total_count records"
        echo "       File size: $(stat -f%z "$mode2_file" 2>/dev/null || stat -c%s "$mode2_file") bytes"

        # Show first record of each type (formatted)
        echo "       First reg_trace record (formatted):"
        grep '"type":"reg_trace"' "$mode2_file" | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'

        if [ "$mode2_mem_count" -gt 0 ]; then
          echo "       First mem_trace record (formatted):"
          grep '"type":"mem_trace"' "$mode2_file" | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'
        fi
      else
        echo "    ❌ Mode 2 validation failed"
        echo "    === Validation errors ==="
        cat mode2_validation.log
        mode2_status="failed"
      fi
    fi
  fi

  # ===== Cross-Format Validation =====
  echo ""
  echo "  🔄 Validating cross-format consistency..."

  if [ "$mode0_status" = "passed" ] && [ "$mode2_status" = "passed" ]; then
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
      fi
    else
      echo "    ❌ ERROR: Mode 0 has zero records, cannot compare"
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
      fi
    elif [ "$mode0_mem_count" -eq 0 ] && [ "$mode2_mem_count" -eq 0 ]; then
      echo "       ℹ️  mem_trace: Both modes have 0 records (no mem_trace data)"
    else
      echo "       ❌ ERROR: mem_trace: Inconsistent (mode0=$mode0_mem_count, mode2=$mode2_mem_count)"
    fi

    # Run comprehensive comparison using Python module
    echo ""
    echo "    🔍 Running comprehensive format comparison..."
    if python3 "$PROJECT_ROOT/scripts/validate_trace.py" --no-color compare "$mode0_file" "$mode2_file" >compare_result.log 2>&1; then
      echo "    ✅ Format comparison passed"
    else
      echo "    ⚠️  Format comparison found differences:"
      cat compare_result.log
    fi
  else
    echo "    ⏭️  Skipping cross-format validation (one or more formats failed)"
  fi

  # ===== Final Report =====
  echo ""
  echo "  ═══════════════════════════════════════════"
  echo "  📋 Test Summary"
  echo "  ═══════════════════════════════════════════"
  echo "  Mode 0 (Text):   $mode0_status"
  echo "  Mode 2 (NDJSON): $mode2_status"
  echo "  ═══════════════════════════════════════════"

  # Determine overall result
  if [ "$mode0_status" = "passed" ] && [ "$mode2_status" = "passed" ]; then
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
  if ! CUDA_INJECTION64_PATH=$PROJECT_ROOT/lib/cutracer.so CUTRACER_INSTRUMENT=reg_trace KERNEL_FILTERS=vectorized_elementwise_kernel python ./test_add.py >py_add_output.log 2>&1; then
    echo "❌ Python script test_add.py failed to execute."
    echo "     === Python script output ==="
    cat py_add_output.log
    cd "$PROJECT_ROOT"
    return 1
  fi
  echo "     === Python script output ==="
  cat py_add_output.log

  # Find logs that match the kernel filter
  matching_logs=$(ls kernel*vectorized_elementwise_kernel*.log 2>/dev/null)
  if [ -z "$matching_logs" ]; then
    echo "❌ Test failed: No log file generated for kernel containing 'vectorized_elementwise_kernel'."
    echo "     Listing current directory contents:"
    ls -la
    cd "$PROJECT_ROOT"
    return 1
  fi

  # Ensure ALL generated kernel logs match the filter
  all_kernel_logs=$(ls kernel*.log 2>/dev/null)
  unmatched_logs=$(echo "$all_kernel_logs" | grep -v "vectorized_elementwise_kernel")

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

  if grep -q "CTA 0,0,0 - warp 0 - @P0 EXIT" "$first_log"; then
    echo "✅ Test successful: Found 'CTA 0,0,0 - warp 0 - @P0 EXIT' in the log."
  else
    echo "❌ Test failed: Did not find 'CTA 0,0,0 - warp 0 - @P0 EXIT' in $first_log."
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
  if ! CUDA_INJECTION64_PATH=$PROJECT_ROOT/lib/cutracer.so CUTRACER_ANALYSIS=proton_instr_histogram KERNEL_FILTERS=add_kernel python ./vector-add-instrumented.py >proton_instr_output.log 2>&1; then
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
  if ! timeout "$TIMEOUT" env CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" CUTRACER_ANALYSIS=deadlock_detection KERNEL_FILTERS=add_kernel python "./test_hang.py" >hang_output.log 2>&1; then
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

# Function to run all tests
run_all_tests() {
  echo "🚀 Running all CUTracer tests..."
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

  echo "🎉 All tests passed successfully!"
  return 0
}

build_cutracer

# Main execution
case "$TEST_TYPE" in
"build-only")
  build_cutracer && build_vectoradd
  ;;
"vectoradd")
  build_vectoradd && test_vectoradd && test_py_add_with_kernel_filters
  ;;
"trace-formats")
  build_vectoradd && test_trace_formats
  ;;
"proton")
  test_proton
  ;;
"hang")
  test_hang_test
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
