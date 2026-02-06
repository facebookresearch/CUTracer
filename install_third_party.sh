#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Script to download and install third-party dependencies for CUTracer
#
# Environment Variables:
#   NVBIT_VERSION  - NVBit version (default: 1.7.6, or "latest" for latest release)
#   JSON_VERSION   - nlohmann/json version (default: 3.11.3)
#
# Usage:
#   ./install_third_party.sh                        # Use defaults
#   NVBIT_VERSION=latest ./install_third_party.sh   # Use latest NVBit
#   NVBIT_VERSION=1.7.5 JSON_VERSION=3.10.0 ./install_third_party.sh

# ============================================================
# Configuration: Set default values if not provided
# ============================================================
NVBIT_VERSION="${NVBIT_VERSION:-1.7.6}"
JSON_VERSION="${JSON_VERSION:-3.11.3}"

# ============================================================
# Install NVBit
# ============================================================

# Create third_party directory (if it doesn't exist)
mkdir -p third_party

# Handle version: fetch latest or use specified version
if [ "$NVBIT_VERSION" = "latest" ]; then
  echo "Getting latest NVBit version information..."
  RELEASE_INFO=$(curl -s https://api.github.com/repos/NVlabs/NVBit/releases/latest)
  NVBIT_VERSION=$(echo "$RELEASE_INFO" | grep -o '"tag_name": "[^"]*' | cut -d'"' -f4)
  echo "Latest version: $NVBIT_VERSION"
else
  # Strip 'v' prefix if present for consistency
  [[ "$NVBIT_VERSION" =~ ^v ]] && NVBIT_VERSION="${NVBIT_VERSION#v}"
  echo "Using NVBit version: $NVBIT_VERSION"
  RELEASE_INFO=$(curl -s "https://api.github.com/repos/NVlabs/NVBit/releases/tags/v${NVBIT_VERSION}")
fi

# Check if API call was successful
if [ $? -ne 0 ]; then
  echo "Error: Unable to get NVBit release information. Please check your network connection or GitHub API access."
  exit 1
fi

# Verify version exists
TAG_CHECK=$(echo "$RELEASE_INFO" | grep -o '"tag_name": "[^"]*' | cut -d'"' -f4)
if [ -z "$TAG_CHECK" ]; then
  echo "Error: Specified version $NVBIT_VERSION not found."
  exit 1
fi

# Find the download link for the x86_64 version
DOWNLOAD_URL=$(echo "$RELEASE_INFO" | grep -o '"browser_download_url": "[^"]*x86_64[^"]*\.tar\.bz2"' | cut -d'"' -f4)

# Check if download link was found
if [ -z "$DOWNLOAD_URL" ]; then
  echo "Error: Unable to find download link for x86_64 version."
  exit 1
fi

echo "Download link: $DOWNLOAD_URL"

# Download NVBit package
echo "Downloading NVBit..."
TEMP_FILE=$(mktemp)
curl -L -o "$TEMP_FILE" "$DOWNLOAD_URL"

# Check if download was successful
if [ $? -ne 0 ]; then
  echo "Error: Download failed."
  rm -f "$TEMP_FILE"
  exit 1
fi

# Clean up old version (if exists)
echo "Cleaning up old version..."
rm -rf third_party/nvbit

# Extract to temporary directory
echo "Extracting NVBit..."
TEMP_DIR=$(mktemp -d)
tar -xjf "$TEMP_FILE" -C "$TEMP_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
  echo "Error: Extraction failed."
  rm -f "$TEMP_FILE"
  rm -rf "$TEMP_DIR"
  exit 1
fi

# Find the extracted directory
EXTRACTED_DIR=$(find "$TEMP_DIR" -maxdepth 1 -name "nvbit*" -type d | head -1)
if [ -z "$EXTRACTED_DIR" ]; then
  echo "Error: Unable to find extracted NVBit directory."
  rm -f "$TEMP_FILE"
  rm -rf "$TEMP_DIR"
  exit 1
fi

# Move the extracted directory to third_party/nvbit
echo "Installing NVBit to third_party/nvbit..."
mv "$EXTRACTED_DIR" third_party/nvbit

# Clean up temporary files and directories
rm -f "$TEMP_FILE"
rm -rf "$TEMP_DIR"

echo "NVBit $NVBIT_VERSION has been successfully installed to third_party/nvbit directory."

# ============================================================
# Install nlohmann/json
# ============================================================
echo ""
echo "Downloading nlohmann/json..."
JSON_URL="https://github.com/nlohmann/json/releases/download/v${JSON_VERSION}/json.hpp"

mkdir -p third_party/nlohmann
curl -L -o third_party/nlohmann/json.hpp "$JSON_URL"

if [ $? -eq 0 ]; then
    echo "nlohmann/json ${JSON_VERSION} has been successfully installed."
else
    echo "Error: Failed to download nlohmann/json."
    exit 1
fi

echo ""
echo "All third-party dependencies have been successfully installed."
