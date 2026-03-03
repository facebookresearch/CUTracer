# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer trace runner.

Wraps user commands with CUTracer environment variables for trace collection.
Resolves cutracer.so automatically via buck resource or explicit path.

Usage:
    cutracer trace --instrument=tma_trace -- ./vectoradd
    cutracer trace --instrument=tma_trace --instr-categories=tma -- python -m pytest test.py
"""

import os
from pathlib import Path
from typing import Optional

import click


def resolve_cutracer_so(explicit_path: Optional[str] = None) -> str:
    """Resolve cutracer.so path.

    Resolution order:
    1. Explicit --cutracer-so path
    2. CUDA_INJECTION64_PATH env var → error (must unset or use --cutracer-so)
    3. Buck resource (bundled with python_library via resources = {})

    If all fail, raises ClickException with clear instructions.
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_file():
            raise click.ClickException(f"cutracer.so not found at: {explicit_path}")
        return str(p.resolve())

    env_path = os.environ.get("CUDA_INJECTION64_PATH")
    if env_path:
        raise click.ClickException(
            f"CUDA_INJECTION64_PATH is set in your environment:\n"
            f"  Path: {env_path}\n\n"
            f"This conflicts with CUTracer's bundled cutracer.so and may cause\n"
            f"stale or mismatched behavior. Please either:\n"
            f"  1. Unset it:  unset CUDA_INJECTION64_PATH\n"
            f"  2. Use --cutracer-so to explicitly specify a path:\n"
            f"     cutracer trace --cutracer-so {env_path} -i tma_trace -- ./app"
        )

    # Default: use cutracer.so bundled as a buck resource.
    try:
        import pkg_resources

        so_path = pkg_resources.resource_filename("cutracer", "cutracer.so")
        if os.path.isfile(so_path):
            click.echo(f"Using bundled cutracer.so: {so_path}")
            return so_path
    except Exception:
        pass

    raise click.ClickException(
        "Could not find cutracer.so. Options:\n"
        "  1. Use --cutracer-so /path/to/cutracer.so\n"
        "  2. Build via buck2 run (cutracer.so is bundled automatically):\n"
        "     buck2 run fbcode//triton/tools/CUTracer:cutracer -- trace ..."
    )


def _build_cutracer_env(
    cutracer_so: str,
    instrument: str,
    analysis: Optional[str],
    kernel_filters: Optional[str],
    instr_categories: Optional[str],
    trace_format: Optional[int],
    trace_output_dir: Optional[str],
    verbose: Optional[int],
    zstd_level: Optional[int],
    delay_ns: Optional[int],
    delay_dump_path: Optional[str],
    delay_load_path: Optional[str],
) -> dict:
    """Build environment dict with CUTracer variables."""
    env = os.environ.copy()
    env["CUDA_INJECTION64_PATH"] = cutracer_so
    env["CUTRACER_INSTRUMENT"] = instrument

    if analysis is not None:
        env["CUTRACER_ANALYSIS"] = analysis
    if kernel_filters is not None:
        env["KERNEL_FILTERS"] = kernel_filters
    if instr_categories is not None:
        env["CUTRACER_INSTR_CATEGORIES"] = instr_categories
    if trace_format is not None:
        env["TRACE_FORMAT_NDJSON"] = str(trace_format)
    if trace_output_dir is not None:
        env["CUTRACER_TRACE_OUTPUT_DIR"] = trace_output_dir
    if verbose is not None:
        env["TOOL_VERBOSE"] = str(verbose)
    if zstd_level is not None:
        env["CUTRACER_ZSTD_LEVEL"] = str(zstd_level)
    if delay_ns is not None:
        env["CUTRACER_DELAY_NS"] = str(delay_ns)
    if delay_dump_path is not None:
        env["CUTRACER_DELAY_DUMP_PATH"] = delay_dump_path
    if delay_load_path is not None:
        env["CUTRACER_DELAY_LOAD_PATH"] = delay_load_path

    # Add bundled CUDA tools (nvdisasm, cuobjdump) to PATH so NVBit can find them.
    # These are bundled as buck resources from fbsource's third-party CUDA,
    # matching the CUDA version specified via -c fbcode.platform010_cuda_version.
    try:
        import pkg_resources

        bin_dir = os.path.dirname(
            pkg_resources.resource_filename("cutracer", "bin/nvdisasm")
        )
        if os.path.isdir(bin_dir):
            env["PATH"] = bin_dir + ":" + env.get("PATH", "")
    except Exception:
        pass

    return env


def _print_config_summary(env: dict) -> None:
    """Print a summary of the active CUTracer configuration."""
    cutracer_keys = [
        "CUDA_INJECTION64_PATH",
        "CUTRACER_INSTRUMENT",
        "CUTRACER_ANALYSIS",
        "KERNEL_FILTERS",
        "CUTRACER_INSTR_CATEGORIES",
        "TRACE_FORMAT_NDJSON",
        "CUTRACER_TRACE_OUTPUT_DIR",
        "TOOL_VERBOSE",
        "CUTRACER_ZSTD_LEVEL",
        "CUTRACER_DELAY_NS",
        "CUTRACER_DELAY_DUMP_PATH",
        "CUTRACER_DELAY_LOAD_PATH",
    ]
    click.echo("=" * 60)
    click.echo("CUTracer Configuration:")
    for key in cutracer_keys:
        if key in env:
            click.echo(f"  {key} = {env[key]}")
    click.echo("=" * 60)


# Common options shared between trace and report commands
_CUTRACER_OPTIONS = [
    click.option(
        "--instrument",
        "-i",
        required=True,
        help="Instrumentation type(s): opcode_only, reg_trace, mem_addr_trace, "
        "mem_value_trace, tma_trace, random_delay",
    ),
    click.option(
        "--analysis",
        "-a",
        default=None,
        help="Analysis type(s): proton_instr_histogram, deadlock_detection, random_delay",
    ),
    click.option(
        "--kernel-filters",
        "-k",
        default=None,
        help="Comma-separated kernel name substring filters",
    ),
    click.option(
        "--instr-categories",
        default=None,
        help="Instruction category filters: mma, tma, sync",
    ),
    click.option(
        "--trace-format",
        type=int,
        default=None,
        help="Trace format: 0=text, 1=ndjson+zstd (default), 2=ndjson-only",
    ),
    click.option(
        "--trace-output-dir",
        default=None,
        help="Output directory for trace files",
    ),
    click.option(
        "--verbose",
        "-v",
        type=int,
        default=None,
        help="Verbosity level (0/1/2)",
    ),
    click.option(
        "--cutracer-so",
        default=None,
        help="Explicit path to cutracer.so (overrides buck2 build)",
    ),
    click.option(
        "--zstd-level",
        type=int,
        default=None,
        help="Zstd compression level (1-22)",
    ),
    click.option(
        "--delay-ns",
        type=int,
        default=None,
        help="Delay in nanoseconds for random_delay instrumentation",
    ),
    click.option(
        "--delay-dump-path",
        default=None,
        help="Output path to dump delay config JSON for replay",
    ),
    click.option(
        "--delay-load-path",
        default=None,
        help="Load delay config JSON for replay mode",
    ),
]


def cutracer_options(func):
    """Decorator to apply all common CUTracer options to a click command."""
    for option in reversed(_CUTRACER_OPTIONS):
        func = option(func)
    return func


@click.command(
    name="trace",
    context_settings={"ignore_unknown_options": True},
)
@cutracer_options
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=True)
def trace_command(
    instrument: str,
    analysis: Optional[str],
    kernel_filters: Optional[str],
    instr_categories: Optional[str],
    trace_format: Optional[int],
    trace_output_dir: Optional[str],
    verbose: Optional[int],
    cutracer_so: Optional[str],
    zstd_level: Optional[int],
    delay_ns: Optional[int],
    delay_dump_path: Optional[str],
    delay_load_path: Optional[str],
    cmd: tuple,
) -> None:
    """Trace a CUDA application with CUTracer instrumentation.

    Sets up CUTracer environment variables and runs the specified command.
    The cutracer.so shared library is bundled via buck2 resources if not
    explicitly provided.

    \b
    Examples:
      cutracer trace --instrument=tma_trace -- ./vectoradd
      cutracer trace -i tma_trace --instr-categories=tma --trace-format=2 -- ./my_app
      cutracer trace -i reg_trace --kernel-filters=matmul_kernel -- python -m pytest test.py
      cutracer trace -i tma_trace --cutracer-so=/path/to/cutracer.so -- ./app
      cutracer trace -i tma_trace -- CUDA_VISIBLE_DEVICES=6 TRITON_PRINT_AUTOTUNING=1 python3 test.py
    """
    if not cmd:
        raise click.UsageError(
            "No command specified. Usage: cutracer trace [OPTIONS] -- COMMAND"
        )

    so_path = resolve_cutracer_so(cutracer_so)

    run_env = _build_cutracer_env(
        cutracer_so=so_path,
        instrument=instrument,
        analysis=analysis,
        kernel_filters=kernel_filters,
        instr_categories=instr_categories,
        trace_format=trace_format,
        trace_output_dir=trace_output_dir,
        verbose=verbose,
        zstd_level=zstd_level,
        delay_ns=delay_ns,
        delay_dump_path=delay_dump_path,
        delay_load_path=delay_load_path,
    )

    _print_config_summary(run_env)

    # Join all args into a single shell command string and execute via bash.
    # This lets shell-style syntax work naturally:
    #   VAR=value cmd args    (env var assignment)
    #   cmd1 | cmd2           (pipes)
    #   cmd1 && cmd2          (chaining)
    import shlex
    import subprocess
    import sys

    cmd_string = shlex.join(cmd)
    click.echo(f"Running: {cmd_string}")
    click.echo("=" * 60)

    result = subprocess.run(cmd_string, shell=True, env=run_env)
    sys.exit(result.returncode)
