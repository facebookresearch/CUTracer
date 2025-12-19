# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer CLI entry point.

Provides command-line interface for trace validation and analysis.
"""

import sys
from importlib.metadata import PackageNotFoundError, version

import click
from cutracer.validation.cli import validate_command


def _get_package_version() -> str:
    """Get package version from metadata."""
    try:
        return version("cutracer")
    except PackageNotFoundError:
        return "0+unknown"


EXAMPLES = """
Examples:
  cutraceross validate kernel_trace.ndjson
  cutraceross validate kernel_trace.ndjson.zst --verbose
  cutraceross validate trace.log --format text
"""


@click.group(epilog=EXAMPLES)
@click.version_option(version=_get_package_version(), prog_name="cutraceross")
def main() -> None:
    """CUTracer: CUDA trace validation and analysis tools."""
    pass


# Register subcommands
main.add_command(validate_command)


if __name__ == "__main__":
    sys.exit(main())
