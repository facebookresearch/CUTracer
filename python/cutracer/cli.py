# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer CLI entry point.

Provides command-line interface for trace validation, query, and analysis.
"""

import sys
from importlib.metadata import PackageNotFoundError, version

import click
from cutracer.analyze.cli import analyze_command
from cutracer.query.cli import query_command
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
  cutraceross query trace.ndjson --filter "warp=24"
  cutraceross query trace.ndjson -f "pc=0x43d0;warp=24"
  cutraceross query trace.ndjson --group-by warp --count
  cutraceross analyze warp-summary trace.ndjson
"""


@click.group(epilog=EXAMPLES)
@click.version_option(version=_get_package_version(), prog_name="cutraceross")
def main() -> None:
    """CUTracer: CUDA trace validation, query, and analysis tools."""
    pass


# Register subcommands
main.add_command(analyze_command)
main.add_command(query_command)
main.add_command(validate_command)


if __name__ == "__main__":
    sys.exit(main())
