#!/bin/bash
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# See LICENSE file for details.
# A script to check or apply code formatting using clang-format.
# It automatically finds all relevant source files in the project.

# --- Go to script's directory ---
# This ensures that the script can be called from any location and that all
# subsequent paths are relative to the project root.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd -- "$SCRIPT_DIR" || exit

# --- Check for dependencies ---
if ! command -v clang-format &> /dev/null; then
    echo "❌ Error: clang-format is not installed or not in your PATH." >&2
    echo "Please install clang-format to use this script." >&2
    echo "  - On Debian/Ubuntu: sudo apt install clang-format" >&2
    echo "  - On Fedora/CentOS: sudo dnf install clang-tools-extra" >&2
    echo "  - On macOS (Homebrew): brew install clang-format" >&2
    exit 1
fi

# --- Configuration ---
# Define the directories to be processed.
# Add or remove directories as needed for your project.
DIRECTORIES_TO_SCAN=(src include tests)

# Define the file extensions to be processed.
# Add or remove extensions as needed for your project.
FILE_PATTERNS=(-name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh")

# --- Helper Functions ---

# Function to print usage instructions
usage() {
  echo "Usage: $0 {check|format}"
  echo "  check      Check for formatting issues without modifying files."
  echo "             Exits with an error code if issues are found."
  echo "  format     Apply formatting to files in-place."
  exit 1
}

# --- Main Script Logic ---

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
  echo "Error: No mode specified."
  usage
fi

MODE=$1

# Filter for directories that actually exist to prevent 'find' errors.
EXISTING_DIRS=()
for dir in "${DIRECTORIES_TO_SCAN[@]}"; do
    if [ -d "$dir" ]; then
        EXISTING_DIRS+=("$dir")
    fi
done

if [ ${#EXISTING_DIRS[@]} -eq 0 ]; then
    echo "⏩  No source directories found to process. Searched for: ${DIRECTORIES_TO_SCAN[*]}. Exiting."
    exit 0
fi

# We pipe the output of find directly to xargs.
# This avoids issues with storing null-delimited strings in shell variables.
# The -r flag for xargs prevents it from running the command if find returns no files.

case "$MODE" in
  check)
    echo "🔎  Checking code formatting in: ${EXISTING_DIRS[*]}..."

    # Step 1: Run with --Werror to get a reliable exit code.
    # We silence the output because we only care about the exit code.
    find "${EXISTING_DIRS[@]}" -type f \( "${FILE_PATTERNS[@]}" \) -print0 | xargs -0 -r clang-format --dry-run --Werror
    
    # Step 2: If there was an error, run again to get the detailed diff for the user.
    if [ $? -ne 0 ]; then
      echo "----------------------------------------------------"
      echo "Please run './format.sh format' to fix them."
      exit 1
    else
      echo "✅  All files are correctly formatted (or no files were found to check)."
      exit 0
    fi
    ;;
    
  format)
    echo "🎨  Applying code formatting to: ${EXISTING_DIRS[*]}..."
    # The -i flag formats the files in-place.
    find "${EXISTING_DIRS[@]}" -type f \( "${FILE_PATTERNS[@]}" \) -print0 | xargs -0 -r clang-format -i
    echo "✅  Formatting complete."
    ;;
    
  *)
    echo "Error: Invalid mode '$MODE'."
    usage
    ;;
esac

exit 0 