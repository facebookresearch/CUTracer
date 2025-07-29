#!/bin/bash

# Setup script for CUTracer CI environment
# This script sets up CUDA and system dependencies for building and testing CUTracer

set -e

# Default values
export CUDA_VERSION=${CUDA_VERSION:-"12.8"}
export DEBUG=${DEBUG:-"0"}
export CONDA_ENV=${CONDA_ENV:-"cutracer"}

echo "Setting up CUTracer environment..."
echo "CUDA_VERSION: $CUDA_VERSION"
echo "DEBUG: $DEBUG"
echo "CONDA_ENV: $CONDA_ENV"

# Setup CUDA, conda, and pytorch
echo "⬇️ Setting up dependencies using tritonparse setup script..."
if curl -sL https://raw.githubusercontent.com/pytorch-labs/tritonparse/main/.ci/setup.sh | bash; then
    echo "✅ Dependencies setup complete."
else
    echo "❌ Dependencies setup failed."
    exit 1
fi

# The tritonparse setup also installs CUDA, so we can verify it here.
# Verify CUDA installation
echo "Verifying CUDA installation..."
if command -v nvcc &>/dev/null; then
  nvcc --version
  echo "✅ CUDA installation verified"
else
  echo "❌ CUDA installation failed"
  exit 1
fi

echo "Setup completed successfully!"
