#!/bin/bash

# Main test script for CUTracer
# This script builds CUTracer and runs the vectoradd test with validation

set -e

# Default values
DEBUG=${DEBUG:-"0"}
TEST_TYPE=${TEST_TYPE:-"all"}
TIMEOUT=${TIMEOUT:-"60"}

echo "Running CUTracer tests..."
echo "DEBUG: $DEBUG"
echo "TEST_TYPE: $TEST_TYPE"
echo "TIMEOUT: $TIMEOUT"

# Define project root path (absolute path)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Set up CUDA environment variables
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

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
    if ! install_third_party; then
        echo "❌ Failed to install third-party dependencies"
        return 1
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

# Function to test vectoradd without CUTracer
test_vectoradd_baseline() {
    echo "🧪 Testing vectoradd without CUTracer..."
    
    cd "$PROJECT_ROOT/tests/vectoradd"
    
    # Run with timeout
    if timeout ${TIMEOUT}s ./vectoradd; then
        echo "✅ vectoradd runs successfully without CUTracer"
        return 0
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "❌ vectoradd test timed out"
        else
            echo "❌ vectoradd test failed with exit code $exit_code"
        fi
        return 1
    fi
}

# Function to test vectoradd with CUTracer
test_vectoradd_with_cutracer() {
    echo "🧪 Testing vectoradd with CUTracer..."
    
    cd "$PROJECT_ROOT/tests/vectoradd"
    
    # Set up CUTracer environment
    export CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so"
    export DEADLOCK_TIMEOUT=30
    export LOG_TO_STDOUT=1
    
    echo "CUDA_INJECTION64_PATH=$CUDA_INJECTION64_PATH"
    
    # Run with timeout and capture output
    if timeout ${TIMEOUT}s ./vectoradd > cutracer_output.log 2>&1; then
        echo "✅ vectoradd with CUTracer completed successfully"
    else
        exit_code=$?
        echo "❌ CUTracer test failed or timed out (exit code: $exit_code)"
        echo "=== CUTracer Output ==="
        cat cutracer_output.log
        return 1
    fi
    
    echo "=== CUTracer Output ==="
    cat cutracer_output.log
    
    return 0
}

# Function to validate CUTracer output
validate_cutracer_output() {
    echo "🔍 Validating CUTracer output..."
    
    cd "$PROJECT_ROOT/tests/vectoradd"
    
    if [ ! -f cutracer_output.log ]; then
        echo "❌ CUTracer output log not found!"
        return 1
    fi
    
    # Check for register trace output
    if grep -q "Reg0_T0: 0x00000000" cutracer_output.log; then
        echo "✅ Found expected register trace: Reg0_T0: 0x00000000"
    else
        echo "❌ Missing expected register trace: Reg0_T0: 0x00000000"
        echo "=== Searching for similar patterns ==="
        grep -i "reg.*t0" cutracer_output.log || echo "No register patterns found"
        return 1
    fi
    
    # Check for CTA exit pattern with flexible PC value
    if grep -E "CTA [0-9]+,0,0 - warp [0-9]+ - PC [0-9]+ - EXIT" cutracer_output.log; then
        echo "✅ Found expected CTA exit pattern"
        echo "Matching lines:"
        grep -E "CTA [0-9]+,0,0 - warp [0-9]+ - PC [0-9]+ - EXIT" cutracer_output.log
    else
        echo "❌ Missing expected CTA exit pattern"
        echo "=== Searching for similar patterns ==="
        grep -i "cta.*warp.*pc.*exit" cutracer_output.log || echo "No CTA exit patterns found"
        grep -i "exit" cutracer_output.log || echo "No exit patterns found"
        return 1
    fi
    
    echo "✅ All CUTracer output validation passed!"
    return 0
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
    if ! test_vectoradd_baseline; then
        echo "❌ Baseline vectoradd test failed"
        return 1
    fi
    
    if ! test_vectoradd_with_cutracer; then
        echo "❌ CUTracer vectoradd test failed"
        return 1
    fi
    
    if ! validate_cutracer_output; then
        echo "❌ CUTracer output validation failed"
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
        test_vectoradd_baseline && test_vectoradd_with_cutracer && validate_cutracer_output
        ;;
    "all"|*)
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