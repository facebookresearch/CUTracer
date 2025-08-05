#!/bin/bash

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

# Activate conda environment
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "🐍 Activating conda environment..."
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate $CONDA_ENV
  conda install -y -c conda-forge libstdcxx-ng=15.1.0
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

# Function to run all tests
run_all_tests() {
  echo "🚀 Running all CUTracer tests..."

  # Build phase
  if ! build_cutracer; then
    echo "❌ CUTracer build failed"
    return 1
  fi

  if ! build_vectoradd; then
    echo "❌ vectoradd build failed"
    return 1
  fi

  # Test phase
  if ! test_vectoradd; then
    echo "❌ vectoradd test failed"
    return 1
  fi

  if ! test_py_add_with_kernel_filters; then
    echo "❌ Python script test_add.py failed to execute."
    return 1
  fi

  echo "🎉 All tests passed successfully!"
  return 0
}

# Main execution
case "$TEST_TYPE" in
"build-only")
  build_cutracer && build_vectoradd
  ;;
"vectoradd")
  test_vectoradd && test_py_add_with_kernel_filters
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
