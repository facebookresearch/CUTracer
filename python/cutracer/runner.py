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
    instrument: Optional[str],
    analysis: Optional[str],
    kernel_filters: Optional[str],
    instr_categories: Optional[str],
    trace_format: Optional[int],
    output_dir: Optional[str],
    verbose: Optional[int],
    zstd_level: Optional[int],
    delay_ns: Optional[int],
    delay_min_ns: Optional[int] = None,
    delay_mode: Optional[str] = None,
    delay_dump_path: Optional[str] = None,
    delay_load_path: Optional[str] = None,
    cpu_callstack: Optional[int] = None,
    channel_records: Optional[int] = None,
    dump_cubin: bool = False,
    trace_size_limit_mb: int = 0,
    kernel_timeout_s: int = 0,
    no_data_timeout_s: int = 15,
) -> dict:
    """Build environment dict with CUTracer variables."""
    env = os.environ.copy()
    env["CUDA_INJECTION64_PATH"] = cutracer_so

    if instrument is not None:
        env["CUTRACER_INSTRUMENT"] = instrument

    if analysis is not None:
        env["CUTRACER_ANALYSIS"] = analysis
    if kernel_filters is not None:
        env["KERNEL_FILTERS"] = kernel_filters
    if instr_categories is not None:
        env["CUTRACER_INSTR_CATEGORIES"] = instr_categories
    if trace_format is not None:
        env["CUTRACER_TRACE_FORMAT"] = str(trace_format)
    if output_dir is not None:
        env["CUTRACER_OUTPUT_DIR"] = output_dir
    if verbose is not None:
        env["TOOL_VERBOSE"] = str(verbose)
    if zstd_level is not None:
        env["CUTRACER_ZSTD_LEVEL"] = str(zstd_level)
    if delay_ns is not None:
        env["CUTRACER_DELAY_NS"] = str(delay_ns)
    if delay_min_ns is not None:
        env["CUTRACER_DELAY_MIN_NS"] = str(delay_min_ns)
    if delay_mode is not None:
        env["CUTRACER_DELAY_MODE"] = delay_mode
    if delay_dump_path is not None:
        env["CUTRACER_DELAY_DUMP_PATH"] = delay_dump_path
    if delay_load_path is not None:
        env["CUTRACER_DELAY_LOAD_PATH"] = delay_load_path
    if cpu_callstack is not None:
        env["CUTRACER_CPU_CALLSTACK"] = str(cpu_callstack)
    if channel_records is not None:
        env["CUTRACER_CHANNEL_RECORDS"] = str(channel_records)
    if dump_cubin:
        env["CUTRACER_DUMP_CUBIN"] = "1"
    env["CUTRACER_TRACE_SIZE_LIMIT_MB"] = str(trace_size_limit_mb)
    env["CUTRACER_KERNEL_TIMEOUT_S"] = str(kernel_timeout_s)
    env["CUTRACER_NO_DATA_TIMEOUT_S"] = str(no_data_timeout_s)

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
        "CUTRACER_TRACE_FORMAT",
        "CUTRACER_OUTPUT_DIR",
        "CUTRACER_DUMP_CUBIN",
        "TOOL_VERBOSE",
        "CUTRACER_ZSTD_LEVEL",
        "CUTRACER_DELAY_NS",
        "CUTRACER_DELAY_DUMP_PATH",
        "CUTRACER_DELAY_LOAD_PATH",
        "CUTRACER_CPU_CALLSTACK",
        "CUTRACER_CHANNEL_RECORDS",
        "CUTRACER_TRACE_SIZE_LIMIT_MB",
        "CUTRACER_KERNEL_TIMEOUT_S",
        "CUTRACER_NO_DATA_TIMEOUT_S",
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
        default=None,
        help="Instrumentation type(s): opcode_only, reg_trace, mem_addr_trace, "
        "mem_value_trace, tma_trace, random_delay. "
        "If omitted, CUTracer acts as a kernel launch logger (no trace files).",
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
        help="Trace format: 0=text, 1=ndjson+zstd, 2=ndjson (default)",
    ),
    click.option(
        "--output-dir",
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
        help="Max delay in nanoseconds for random_delay instrumentation",
    ),
    click.option(
        "--delay-min-ns",
        type=int,
        default=None,
        help="Min delay in nanoseconds (floor for random mode, default: 0). "
        "Setting min > 0 ensures every thread gets at least this much delay.",
    ),
    click.option(
        "--delay-mode",
        type=click.Choice(["random", "fixed"]),
        default=None,
        help="Delay mode: 'random' (per-thread random delay, default) or "
        "'fixed' (same delay for all threads, often masks races)",
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
    click.option(
        "--cpu-callstack",
        type=int,
        default=None,
        help="CPU call stack capture at kernel launch (1=enabled, 0=disabled, default: 1)",
    ),
    click.option(
        "--channel-records",
        type=int,
        default=None,
        help="Channel buffer capacity in records (default: auto/4MB). "
        "Set to 1 for per-record flush (useful for hang debugging)",
    ),
    click.option(
        "--dump-cubin/--no-dump-cubin",
        default=None,
        help="Dump cubin files for instrumented kernels (for SASS disassembly via nvdisasm). "
        "Auto-enabled when --instrument is set; use --no-dump-cubin to override.",
    ),
    click.option(
        "--trace-size-limit-mb",
        type=int,
        default=0,
        show_default=True,
        help="Maximum trace file size in MB (0 = disabled, default: 0). "
        "Stops tracing when any file exceeds this limit; "
        "kernel execution continues normally.",
    ),
    click.option(
        "--kernel-timeout-s",
        type=int,
        default=0,
        show_default=True,
        help="Kernel execution timeout in seconds (0 = disabled, default: 0). "
        "Auto-terminates any kernel running longer than this value. "
        "Independent of deadlock detection.",
    ),
    click.option(
        "--no-data-timeout-s",
        type=int,
        default=15,
        show_default=True,
        help="No-data timeout in seconds for silent hang detection (default: 15). "
        "Terminates the process when no trace data arrives for this duration. "
        "Independent of deadlock detection (does not require -a deadlock_detection). "
        "Set to 0 to disable.",
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
    instrument: Optional[str],
    analysis: Optional[str],
    kernel_filters: Optional[str],
    instr_categories: Optional[str],
    trace_format: Optional[int],
    output_dir: Optional[str],
    verbose: Optional[int],
    cutracer_so: Optional[str],
    zstd_level: Optional[int],
    delay_ns: Optional[int],
    delay_min_ns: Optional[int],
    delay_mode: Optional[str],
    delay_dump_path: Optional[str],
    delay_load_path: Optional[str],
    cpu_callstack: Optional[int],
    channel_records: Optional[int],
    dump_cubin: Optional[bool],
    trace_size_limit_mb: int,
    kernel_timeout_s: int,
    no_data_timeout_s: int,
    cmd: tuple,
) -> None:
    """Trace a CUDA application with CUTracer instrumentation.

    Sets up CUTracer environment variables and runs the specified command.
    The cutracer.so shared library is bundled via buck2 resources if not
    explicitly provided.

    When --instrument is omitted, CUTracer acts as a lightweight kernel launch
    logger: kernel names, grid/block dims, and shared memory usage are printed
    but no trace files are created and no instrumentation overhead is added.

    \b
    Examples:
      cutracer trace --instrument=tma_trace -- ./vectoradd
      cutracer trace -i tma_trace --instr-categories=tma --trace-format=2 -- ./my_app
      cutracer trace -i reg_trace --kernel-filters=matmul_kernel -- python -m pytest test.py
      cutracer trace -i tma_trace --cutracer-so=/path/to/cutracer.so -- ./app
      cutracer trace -- ./my_app  # kernel launch logger only (no instrumentation)
    """
    if not cmd:
        raise click.UsageError(
            "No command specified. Usage: cutracer trace [OPTIONS] -- COMMAND"
        )

    so_path = resolve_cutracer_so(cutracer_so)

    # Auto-enable cubin dumping when instrumentation is active, unless
    # the user explicitly passed --no-dump-cubin.
    if dump_cubin is None:
        dump_cubin = instrument is not None

    run_env = _build_cutracer_env(
        cutracer_so=so_path,
        instrument=instrument,
        analysis=analysis,
        kernel_filters=kernel_filters,
        instr_categories=instr_categories,
        trace_format=trace_format,
        output_dir=output_dir,
        verbose=verbose,
        zstd_level=zstd_level,
        delay_ns=delay_ns,
        delay_min_ns=delay_min_ns,
        delay_mode=delay_mode,
        delay_dump_path=delay_dump_path,
        delay_load_path=delay_load_path,
        cpu_callstack=cpu_callstack,
        channel_records=channel_records,
        dump_cubin=dump_cubin,
        trace_size_limit_mb=trace_size_limit_mb,
        kernel_timeout_s=kernel_timeout_s,
        no_data_timeout_s=no_data_timeout_s,
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
