# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Delay config mutator for reduction.

Provides utilities to load, modify, and save delay injection configurations.
Uses the JSON schema from cutracer.validation for config validation.
"""

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jsonschema
from cutracer.validation import DELAY_CONFIG_SCHEMA


@dataclass
class DelayPoint:
    """Represents a single delay injection point."""

    kernel_key: str
    kernel_name: str
    pc_offset: str
    sass: str
    delay_ns: int
    enabled: bool

    def __repr__(self) -> str:
        status = "ON" if self.enabled else "OFF"
        return (
            f"[{status}] {self.sass} (PC: {self.pc_offset}, kernel: {self.kernel_name})"
        )


class DelayConfigMutator:
    """
    Mutator for delay injection configurations.

    Provides methods to load, modify, and save delay configs for reduction.
    Validates configs against the DELAY_CONFIG_SCHEMA on load.
    """

    def __init__(self, config_path: str, validate: bool = True):
        """
        Load a delay config from file.

        Args:
            config_path: Path to the delay config JSON file.
            validate: Whether to validate config against JSON schema (default: True).

        Raises:
            ValueError: If config fails schema validation.
            FileNotFoundError: If config file does not exist.
            json.JSONDecodeError: If config file contains invalid JSON.
        """
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = json.load(f)

        # Validate against schema
        if validate:
            try:
                jsonschema.validate(self.config, DELAY_CONFIG_SCHEMA)
            except jsonschema.ValidationError as e:
                raise ValueError(
                    f"Invalid delay config '{config_path}': {e.message}"
                ) from e

        self._delay_points: list[DelayPoint] = []
        self._parse_delay_points()

    def _parse_delay_points(self) -> None:
        """Parse delay points from the config."""
        self._delay_points = []
        for kernel_key, kernel_config in self.config.get("kernels", {}).items():
            kernel_name = kernel_config.get("kernel_name", kernel_key)
            for pc_offset, point in kernel_config.get(
                "instrumentation_points", {}
            ).items():
                self._delay_points.append(
                    DelayPoint(
                        kernel_key=kernel_key,
                        kernel_name=kernel_name,
                        pc_offset=pc_offset,
                        sass=point.get("sass", ""),
                        delay_ns=point.get("delay", 0),
                        enabled=point.get("on", False),
                    )
                )

    @property
    def delay_points(self) -> list[DelayPoint]:
        """Get all delay points."""
        return self._delay_points

    @property
    def enabled_points(self) -> list[DelayPoint]:
        """Get only enabled delay points."""
        return [p for p in self._delay_points if p.enabled]

    def set_point_enabled(self, point: DelayPoint, enabled: bool) -> None:
        """
        Enable or disable a delay point.

        Args:
            point: The delay point to modify.
            enabled: Whether to enable or disable the point.
        """
        point.enabled = enabled
        self.config["kernels"][point.kernel_key]["instrumentation_points"][
            point.pc_offset
        ]["on"] = enabled

    def set_all_enabled(self, enabled: bool) -> None:
        """Enable or disable all delay points."""
        for point in self._delay_points:
            self.set_point_enabled(point, enabled)

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the config to a file.

        Args:
            path: Optional path to save to. If None, creates a temp file.

        Returns:
            Path to the saved file.
        """
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".json", prefix="cutracer_bisect_")
            with open(fd, "w") as f:
                json.dump(self.config, f, indent=2)
        else:
            with open(path, "w") as f:
                json.dump(self.config, f, indent=2)
        return path

    def clone(self) -> "DelayConfigMutator":
        """Create a deep copy of this mutator."""
        new_mutator = DelayConfigMutator.__new__(DelayConfigMutator)
        new_mutator.config_path = self.config_path
        new_mutator.config = copy.deepcopy(self.config)
        new_mutator._delay_points = []
        new_mutator._parse_delay_points()
        return new_mutator

    def __len__(self) -> int:
        return len(self._delay_points)

    def __repr__(self) -> str:
        enabled = len(self.enabled_points)
        total = len(self._delay_points)
        return f"DelayConfigMutator({enabled}/{total} points enabled)"
