## Overview 🧹

This project uses a simple, script-driven formatting workflow:
- C/C++/CUDA files are formatted with clang-format.
- Python uses a three-step pipeline: usort (imports) → ruff (lint autofix) → black (code style).

## What gets formatted 📂

- Directories scanned (if present): `src`, `include`, `tests`, `tutorial`, `.ci`.
- File patterns for C/C++/CUDA: `*.h`, `*.hpp`, `*.cpp`, `*.cu`, `*.cuh`.

## Dependencies 📦

- clang-format (required for C/C++/CUDA)
  - Debian/Ubuntu: `sudo apt install clang-format`
  - Fedora/CentOS: `sudo dnf install clang-tools-extra`
  - macOS (Homebrew): `brew install clang-format`
- Python tools (for Python files): `usort`, `ruff`, `black`
  - Install: `pip install usort ruff black`

If a tool is missing, `format.sh` prints a clear message and exits non‑zero for “check”.

## Usage 🔧

From the repo root:

```bash
./format.sh check
```
- Runs clang-format in dry-run mode with `--Werror` and checks Python with `usort/ruff/black` (no changes).
- Exits with non-zero if any file would change or Python checks fail.

```bash
./format.sh format
```
- Applies clang-format in-place to all matched C/C++/CUDA files and lists changed files.
- Runs Python formatting: `usort format .` → `ruff check . --fix` → `black .`.

Notes 📝:
- The script only processes directories that actually exist.
- Third-party content is not included unless it lives under the scanned directories.
- clang-format style is whatever your local `clang-format` resolves to (no custom config in this repo unless you add one).

## Recommendations ✅

- Run `./format.sh format` before committing.
- If adding new source folders or file types, update `DIRECTORIES_TO_SCAN` or `FILE_PATTERNS` inside `format.sh`.

