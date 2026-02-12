# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core reduction algorithm for finding minimal race-triggering delay configurations.

Implements delta debugging to find the minimal set of delay injection points
that trigger a data race.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from cutracer.reduce.config_mutator import DelayConfigMutator, DelayPoint


@dataclass
class ReduceResult:
    """Result of a reduction run."""

    total_points: int
    essential_points: list[DelayPoint]
    iterations: int
    minimal_config_path: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether reduction found essential points."""
        return len(self.essential_points) > 0

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "Reduction complete!",
            f"  Total points tested: {self.total_points}",
            f"  Essential points found: {len(self.essential_points)}",
            f"  Iterations: {self.iterations}",
        ]
        if self.essential_points:
            lines.append("\nEssential delay points:")
            for i, point in enumerate(self.essential_points, 1):
                lines.append(f"  {i}. {point.sass}")
                lines.append(f"     PC: {point.pc_offset}, Kernel: {point.kernel_name}")
        return "\n".join(lines)


@dataclass
class ReduceConfig:
    """Configuration for reduction."""

    config_path: str
    test_script: str
    output_path: Optional[str] = None
    verbose: bool = False
    validate_schema: bool = True


def run_test(
    test_script: str,
    config_path: str,
    verbose: bool = False,
) -> bool:
    """
    Run the test script with a given delay config.

    Args:
        test_script: Path to the test script.
        config_path: Path to the delay config JSON file.
        verbose: Whether to print test output.

    Returns:
        True if data race occurred (exit code 0), False otherwise.
    """
    env = os.environ.copy()
    env["CUTRACER_DELAY_LOAD_PATH"] = config_path

    try:
        result = subprocess.run(
            test_script,
            shell=True,
            env=env,
            capture_output=not verbose,
            timeout=300,  # 5 minute timeout
        )
        return result.returncode == 0  # 0 = bad (race), non-zero = good
    except subprocess.TimeoutExpired:
        if verbose:
            print("  Test timed out (treating as no race)")
        return False
    except Exception as e:
        if verbose:
            print(f"  Test error: {e}")
        return False


def _cleanup_temp_file(path: str) -> None:
    """Safely remove a temporary file."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _find_point_in_mutator(
    mutator: DelayConfigMutator, kernel_key: str, pc_offset: str
) -> Optional[DelayPoint]:
    """Find a matching point in a mutator by kernel_key and pc_offset."""
    for p in mutator.enabled_points:
        if p.kernel_key == kernel_key and p.pc_offset == pc_offset:
            return p
    return None


def _test_point_essentiality(
    point: DelayPoint,
    mutator: DelayConfigMutator,
    test_script: str,
    verbose: bool,
) -> tuple[bool, str]:
    """
    Test if a point is essential by disabling it and running the test.

    Returns:
        Tuple of (is_essential, temp_config_path).
    """
    test_mutator = mutator.clone()
    cloned_point = _find_point_in_mutator(
        test_mutator, point.kernel_key, point.pc_offset
    )
    if cloned_point:
        test_mutator.set_point_enabled(cloned_point, False)

    test_config_path = test_mutator.save()
    race_occurs = run_test(test_script, test_config_path, verbose)
    is_essential = not race_occurs

    return is_essential, test_config_path


def _validate_initial_config(
    mutator: DelayConfigMutator,
    config: ReduceConfig,
) -> str:
    """
    Validate that the initial config triggers a race.

    Returns:
        Path to the saved initial config.

    Raises:
        ValueError: If no enabled points or initial config doesn't trigger race.
    """
    enabled_points = mutator.enabled_points

    if not enabled_points:
        raise ValueError("No enabled delay points in config")

    if config.verbose:
        print(f"Loaded {len(enabled_points)} enabled delay points")

    initial_config_path = mutator.save()
    if not run_test(config.test_script, initial_config_path, config.verbose):
        raise ValueError(
            "Initial config does not trigger the race! "
            "Test script must return 0 (race) with the original config."
        )

    if config.verbose:
        print("✓ Initial config triggers the race")

    return initial_config_path


def reduce_delay_points(
    config: ReduceConfig,
    progress_callback: Optional[Callable[[int, int, DelayPoint, bool], None]] = None,
) -> ReduceResult:
    """
    Find minimal set of delay points that trigger a data race.

    Uses delta debugging algorithm:
    1. Start with all enabled points
    2. For each point, disable it and test
    3. If race still occurs, the point is not essential
    4. If race disappears, the point is essential

    Args:
        config: Reduction configuration.
        progress_callback: Optional callback for progress updates.
            Called with (current_iteration, total, point, is_essential).

    Returns:
        ReduceResult with essential points and statistics.
    """
    mutator = DelayConfigMutator(config.config_path, validate=config.validate_schema)
    initial_config_path = _validate_initial_config(mutator, config)

    points_to_test = list(mutator.enabled_points)
    essential_points: list[DelayPoint] = []
    iteration = 0

    for point in points_to_test:
        iteration += 1

        if config.verbose:
            print(f"\n[{iteration}/{len(points_to_test)}] Testing: {point.sass}")

        is_essential, test_config_path = _test_point_essentiality(
            point, mutator, config.test_script, config.verbose
        )

        if is_essential:
            if config.verbose:
                print("   → Race disappears (point IS essential)")
            essential_points.append(point)
        else:
            if config.verbose:
                print("   → Race still occurs (point NOT essential)")
            # Update mutator to keep this point disabled
            original_point = _find_point_in_mutator(
                mutator, point.kernel_key, point.pc_offset
            )
            if original_point:
                mutator.set_point_enabled(original_point, False)

        if progress_callback:
            progress_callback(iteration, len(points_to_test), point, is_essential)

        _cleanup_temp_file(test_config_path)

    # Save minimal config if output path specified
    minimal_config_path = None
    if config.output_path:
        minimal_config_path = mutator.save(config.output_path)
        if config.verbose:
            print(f"\nMinimal config saved to: {minimal_config_path}")

    _cleanup_temp_file(initial_config_path)

    return ReduceResult(
        total_points=len(points_to_test),
        essential_points=essential_points,
        iterations=iteration,
        minimal_config_path=minimal_config_path,
    )
