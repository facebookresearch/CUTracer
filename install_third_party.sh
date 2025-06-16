#!/bin/bash

# Script to automatically download and install the latest version of NVBit to the third_party directory

# Create third_party directory (if it doesn't exist)
mkdir -p third_party

# Use GitHub API to get the latest release information
echo "Getting latest NVBit version information..."
RELEASE_INFO=$(curl -s https://api.github.com/repos/NVlabs/NVBit/releases/latest)

# Check if API call was successful
if [ $? -ne 0 ]; then
    echo "Error: Unable to get NVBit release information. Please check your network connection or GitHub API access."
    exit 1
fi

# Get the latest version number
VERSION=$(echo $RELEASE_INFO | grep -o '"tag_name": "[^"]*' | cut -d'"' -f4)
echo "Latest version: $VERSION"

# Find the download link for the x86_64 version
DOWNLOAD_URL=$(echo $RELEASE_INFO | grep -o '"browser_download_url": "[^"]*x86_64[^"]*\.tar\.bz2"' | cut -d'"' -f4)

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

echo "NVBit $VERSION has been successfully installed to third_party/nvbit directory."
