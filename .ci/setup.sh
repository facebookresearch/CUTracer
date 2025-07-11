#!/bin/bash

# Setup script for CUTracer CI environment
# This script sets up CUDA and system dependencies for building and testing CUTracer

set -e

# Default values
CUDA_VERSION=${CUDA_VERSION:-"12.8"}
DEBUG=${DEBUG:-"0"}

echo "Setting up CUTracer environment..."
echo "CUDA_VERSION: $CUDA_VERSION"
echo "DEBUG: $DEBUG"

# Install system dependencies
echo "Installing system dependencies..."

# Update package lists
echo "🔄 Updating package lists..."
sudo apt-get update

# Install CUDA and development libraries
echo "Installing CUDA and development libraries..."

echo "📦 Installing CUDA $CUDA_VERSION and development libraries..."
# Install all packages including CUDA toolkit
sudo apt-get install -y cuda-toolkit-$(echo $CUDA_VERSION | tr '.' '-') build-essential cmake git bc gdb

# Set up CUDA environment variables
echo "Setting up CUDA environment..."
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Verify CUDA installation
echo "Verifying CUDA installation..."
if command -v nvcc &>/dev/null; then
    nvcc --version
    echo "✅ CUDA installation verified"
else
    echo "❌ CUDA installation failed"
    exit 1
fi

# Check NVIDIA GPU information
echo "Checking NVIDIA GPU information..."
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi
else
    echo "⚠️ nvidia-smi not found (this is expected in CPU-only environments)"
fi

echo "Setup completed successfully!" 