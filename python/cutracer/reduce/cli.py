# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI for CUTracer reduce command.

Provides command-line interface for reducing delay injection configurations.
"""

import sys
from typing import NoReturn

import click
from cutracer.reduce.config_mutator import DelayPoint
from cutracer.reduce.reduce import reduce_delay_points, ReduceConfig, ReduceResult
from cutracer.reduce.report import generate_report, save_report


def progress_callback(
    current: int, total: int, point: DelayPoint, is_essential: bool
) -> None:
    """Print progress during reduction."""
    status = "✓ ESSENTIAL" if is_essential else "  not essential"
    click.echo(f"[{current}/{total}] {point.sass[:50]:<50} {status}")


def _exit_with_error(
    message: str, verbose: bool = False, exception: Exception | None = None
) -> NoReturn:
    """Print error message and exit."""
    click.echo(f"\n❌ Error: {message}", err=True)
    if verbose and exception:
        import traceback

        traceback.print_exc()
    sys.exit(1)


def _run_reduction(reduce_config: ReduceConfig, verbose: bool) -> ReduceResult:
    """Run the reduction algorithm and handle errors."""
    try:
        return reduce_delay_points(
            config=reduce_config,
            progress_callback=progress_callback if not verbose else None,
        )
    except ValueError as e:
        _exit_with_error(str(e))
    except Exception as e:
        _exit_with_error(f"Unexpected error: {e}", verbose=verbose, exception=e)


@click.command(name="reduce")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to delay config JSON file (repro config).",
)
@click.option(
    "--test",
    "-t",
    required=True,
    type=str,
    help="Test script. Returns 0 if race occurs, non-zero if no race.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="reduce_report.json",
    help="Output path for the report JSON.",
)
@click.option(
    "--minimal-config",
    "-m",
    type=click.Path(),
    default="minimal_config.json",
    help="Output path for the minimal config.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output.",
)
@click.option(
    "--no-validate-schema",
    is_flag=True,
    help="Skip JSON schema validation of the config file.",
)
def reduce_command(
    config: str,
    test: str,
    output: str,
    minimal_config: str,
    verbose: bool,
    no_validate_schema: bool,
) -> None:
    """
    Reduce delay injection config to find minimal race trigger.

    Uses delta debugging to find the minimal set of delay points
    that trigger a data race.

    Test script convention (same as llvm-reduce):

    \b
      - Exit 0: Interesting (race occurred)
      - Exit 1+: Not interesting (no race)

    Example:

    \b
      cutraceross reduce --config repro.json --test ./test_race.sh
    """
    click.echo("=" * 60)
    click.echo(" CUTRACER REDUCE")
    click.echo("=" * 60)
    click.echo(f"Config: {config}")
    click.echo(f"Test script: {test}")
    click.echo("")

    # Create reduce config
    reduce_config = ReduceConfig(
        config_path=config,
        test_script=test,
        output_path=minimal_config,
        verbose=verbose,
        validate_schema=not no_validate_schema,
    )

    # Run reduction
    click.echo("\n🔍 Starting reduction...\n")
    result = _run_reduction(reduce_config, verbose)

    # Generate report
    click.echo("\n" + "=" * 60)
    click.echo(" REDUCTION COMPLETE")
    click.echo("=" * 60)

    if result.essential_points:
        click.echo(f"\n✅ Found {len(result.essential_points)} essential point(s):\n")
        for i, point in enumerate(result.essential_points, 1):
            click.echo(f"  {i}. {point.sass}")
            click.echo(f"     PC: {point.pc_offset}")
            click.echo(f"     Kernel: {point.kernel_name}")
            click.echo("")
    else:
        click.echo("\n⚠️  No essential points found.")
        click.echo("    This may indicate the race is not deterministically triggered")
        click.echo("    by the delay injection points in this config.")

    # Generate and save report
    report = generate_report(
        result=result,
        config_path=config,
        test_script=test,
    )
    save_report(report, output)
    click.echo(f"\n📊 Report saved to: {output}")

    if result.minimal_config_path:
        click.echo(f"📋 Minimal config saved to: {result.minimal_config_path}")

    # Print summary stats
    click.echo("\n📈 Stats:")
    click.echo(f"   Total points: {result.total_points}")
    click.echo(f"   Essential: {len(result.essential_points)}")
    click.echo(f"   Iterations: {result.iterations}")

    click.echo("")
