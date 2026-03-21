# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
SASS code extraction from cubin files.

Provides utilities to disassemble NVIDIA cubin files and extract
SASS assembly code using nvdisasm. This module can be used standalone
via CLI or programmatically by other modules (e.g., deadlock analysis).

Usage (CLI):
    cutracer query sass kernel.cubin
    cutracer query sass kernel.cubin -o kernel.sass

Usage (programmatic):
    from cutracer.query.sass import dump_sass, dump_sass_to_file

    # Get SASS text in memory
    output = dump_sass(Path("kernel.cubin"))
    if output:
        print(output.raw_text)

    # Save SASS to file
    sass_path = dump_sass_to_file(Path("kernel.cubin"))
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SassOutput:
    """Container for nvdisasm output.

    Attributes:
        raw_text: The disassembled SASS text.
        cubin_path: Path to the source cubin file.
        flags_used: List of nvdisasm flags that were used.
    """

    raw_text: str
    cubin_path: Path
    flags_used: list[str] = field(default_factory=list)

    def save(self, output_path: Path) -> None:
        """Save SASS text to file.

        Args:
            output_path: Path where SASS text will be written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.raw_text)

    def __len__(self) -> int:
        """Return length of SASS text."""
        return len(self.raw_text)

    @property
    def line_count(self) -> int:
        """Return number of lines in SASS text."""
        if not self.raw_text:
            return 0
        return len(self.raw_text.rstrip("\n").split("\n"))


def dump_sass(
    cubin_path: Path,
    *,
    include_source_info: bool = True,
    include_line_info: bool = True,
    timeout: int = 60,
) -> SassOutput | None:
    """
    Dump SASS assembly from cubin file using nvdisasm.

    This is the core function for extracting SASS code. It runs:
        nvdisasm -g -c <cubin_path>

    Flags:
        -g: Include source-level debug info (file/line annotations)
        -c: Output assembly with //## File comments for source mapping

    The combination of -g and -c provides:
    - Full instruction listing with PC offsets
    - Source file/line annotations via //## comments
    - Function boundaries and labels

    Args:
        cubin_path: Absolute path to cubin file.
        include_source_info: If True, add -g flag for debug info.
        include_line_info: If True, add -c flag for //## comments.
        timeout: Timeout in seconds (default: 60).

    Returns:
        SassOutput containing raw text and metadata, or None if failed.

    Example:
        >>> output = dump_sass(Path("/path/to/kernel.cubin"))
        >>> if output:
        ...     print(f"Got {output.line_count} lines of SASS")
        ...     output.save(Path("kernel.sass"))
    """
    if not cubin_path.exists():
        logger.warning("cubin file not found: %s", cubin_path)
        return None

    flags: list[str] = []
    if include_source_info:
        flags.append("-g")
    if include_line_info:
        flags.append("-c")

    cmd = ["nvdisasm", *flags, str(cubin_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        logger.warning("nvdisasm not found in PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("nvdisasm timed out after %ds", timeout)
        return None

    if result.returncode != 0:
        logger.warning(
            "nvdisasm failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return None

    return SassOutput(
        raw_text=result.stdout,
        cubin_path=cubin_path,
        flags_used=flags,
    )


def dump_sass_to_file(
    cubin_path: Path,
    output_path: Path | None = None,
    *,
    include_source_info: bool = True,
    include_line_info: bool = True,
    timeout: int = 60,
) -> Path | None:
    """
    Dump SASS to file. Convenience wrapper around dump_sass().

    Args:
        cubin_path: Path to cubin file.
        output_path: Output .sass file path. If None, derives from cubin path
                     by replacing extension with .sass.
        include_source_info: If True, add -g flag for debug info.
        include_line_info: If True, add -c flag for //## comments.
        timeout: Timeout in seconds (default: 60).

    Returns:
        Path to output file, or None if dump failed.

    Example:
        >>> # Creates kernel.sass next to kernel.cubin
        >>> sass_path = dump_sass_to_file(Path("kernel.cubin"))
        >>> # Or specify explicit output path
        >>> sass_path = dump_sass_to_file(Path("kernel.cubin"), Path("output/kernel.sass"))
    """
    sass_output = dump_sass(
        cubin_path,
        include_source_info=include_source_info,
        include_line_info=include_line_info,
        timeout=timeout,
    )
    if sass_output is None:
        return None

    if output_path is None:
        output_path = cubin_path.with_suffix(".sass")

    sass_output.save(output_path)
    return output_path
