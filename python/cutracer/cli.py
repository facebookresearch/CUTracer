# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer CLI entry point.

Provides command-line interface for trace validation and analysis.
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from cutracer.validation.cli import _add_validate_args, validate_command


def _get_package_version() -> str:
    """Get package version from metadata."""
    try:
        return version("cutracer")
    except PackageNotFoundError:
        return "0+unknown"


def main() -> int:
    """Main CLI entry point."""
    pkg_version = _get_package_version()

    parser = argparse.ArgumentParser(
        prog="cutraceross",
        description="CUTracer: CUDA trace validation and analysis tools",
        epilog=(
            "Examples:\n"
            "  cutraceross validate kernel_trace.ndjson\n"
            "  cutraceross validate kernel_trace.ndjson.zst --verbose\n"
            "  cutraceross validate trace.log --format text\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {pkg_version}",
        help="Show program's version number and exit",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a CUTracer trace file",
        description=(
            "Validate a CUTracer trace file.\n\n"
            "Checks syntax and schema compliance for NDJSON, Zstd-compressed,\n"
            "and text format trace files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_validate_args(validate_parser)
    validate_parser.set_defaults(func=validate_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
