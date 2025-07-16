#!/bin/bash
# Format all shell scripts in the project

set -e

# Create a temporary directory for tools
TEMP_BIN_DIR="/tmp/shell_tools_bin_$$"
mkdir -p "$TEMP_BIN_DIR"

# Cleanup function
cleanup() {
  echo "Cleaning up temporary directory..."
  rm -rf "$TEMP_BIN_DIR"
}
trap cleanup EXIT

echo "Installing shfmt if not available..."
if ! command -v shfmt &>/dev/null; then
  echo "shfmt not found. Installing..."

  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Get the latest version dynamically
    LATEST_VERSION=$(curl -s https://api.github.com/repos/mvdan/sh/releases/latest | grep '"tag_name"' | cut -d '"' -f 4)
    echo "Installing shfmt ${LATEST_VERSION}..."
    wget -O "$TEMP_BIN_DIR/shfmt" "https://github.com/mvdan/sh/releases/download/${LATEST_VERSION}/shfmt_${LATEST_VERSION}_linux_amd64"
    chmod +x "$TEMP_BIN_DIR/shfmt"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Get the latest version dynamically
    LATEST_VERSION=$(curl -s https://api.github.com/repos/mvdan/sh/releases/latest | grep '"tag_name"' | cut -d '"' -f 4)
    echo "Installing shfmt ${LATEST_VERSION}..."
    wget -O "$TEMP_BIN_DIR/shfmt" "https://github.com/mvdan/sh/releases/download/${LATEST_VERSION}/shfmt_${LATEST_VERSION}_darwin_amd64"
    chmod +x "$TEMP_BIN_DIR/shfmt"
  else
    echo "Please install shfmt manually from https://github.com/mvdan/sh"
    exit 1
  fi

  echo "shfmt installed to $TEMP_BIN_DIR"
fi

echo "Installing ShellCheck if not available..."
if ! command -v shellcheck &>/dev/null; then
  echo "ShellCheck not found. Installing..."

  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Get the latest version dynamically
    LATEST_VERSION=$(curl -s https://api.github.com/repos/koalaman/shellcheck/releases/latest | grep '"tag_name"' | cut -d '"' -f 4)
    echo "Installing ShellCheck ${LATEST_VERSION}..."

    # Download and extract ShellCheck
    wget -O "$TEMP_BIN_DIR/shellcheck.tar.xz" "https://github.com/koalaman/shellcheck/releases/download/${LATEST_VERSION}/shellcheck-${LATEST_VERSION}.linux.x86_64.tar.xz"
    cd "$TEMP_BIN_DIR"
    tar -xf shellcheck.tar.xz
    mv "shellcheck-${LATEST_VERSION}/shellcheck" ./
    rm -rf "shellcheck-${LATEST_VERSION}" shellcheck.tar.xz
    chmod +x shellcheck
    cd - >/dev/null
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Get the latest version dynamically
    LATEST_VERSION=$(curl -s https://api.github.com/repos/koalaman/shellcheck/releases/latest | grep '"tag_name"' | cut -d '"' -f 4)
    echo "Installing ShellCheck ${LATEST_VERSION}..."

    # Download and extract ShellCheck for macOS
    wget -O "$TEMP_BIN_DIR/shellcheck.tar.xz" "https://github.com/koalaman/shellcheck/releases/download/${LATEST_VERSION}/shellcheck-${LATEST_VERSION}.darwin.x86_64.tar.xz"
    cd "$TEMP_BIN_DIR"
    tar -xf shellcheck.tar.xz
    mv "shellcheck-${LATEST_VERSION}/shellcheck" ./
    rm -rf "shellcheck-${LATEST_VERSION}" shellcheck.tar.xz
    chmod +x shellcheck
    cd - >/dev/null
  else
    echo "Please install ShellCheck manually from https://github.com/koalaman/shellcheck"
    exit 1
  fi

  echo "ShellCheck installed to $TEMP_BIN_DIR"
fi

# Add to PATH for this session
export PATH="$TEMP_BIN_DIR:$PATH"

echo "Formatting shell scripts..."
find . -name "*.sh" -type f | while read -r file; do
  echo "Formatting: $file"
  shfmt -i 2 -ci -w "$file"
done

echo "Running ShellCheck..."
find . -name "*.sh" -type f | xargs shellcheck

echo "Done!"
