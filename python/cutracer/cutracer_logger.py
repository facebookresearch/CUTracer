# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified logger module for CUTracer.

Provides a consistent logging interface across all CUTracer modules.
"""

import logging
import os
from typing import Optional

# Check if debug mode is enabled via environment variable
CUTRACER_DEBUG = os.getenv("CUTRACER_DEBUG", None) in ["1", "true", "True"]

# Main logger for cutracer
logger = logging.getLogger("cutracer")

# Set default level based on CUTRACER_DEBUG environment variable
logger.setLevel(logging.DEBUG if CUTRACER_DEBUG else logging.INFO)

# Add a default StreamHandler if no handlers are configured
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
    logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a cutracer logger instance.

    Args:
        name: Sub-module name. If provided, creates a "cutracer.{name}" child logger.
              If None, returns the main "cutracer" logger.

    Returns:
        logging.Logger: Logger instance

    Examples:
        >>> from cutracer.cutracer_logger import get_logger
        >>> logger = get_logger()  # Returns "cutracer" logger
        >>> logger = get_logger("data_race")  # Returns "cutracer.data_race" logger
    """
    if name:
        return logging.getLogger(f"cutracer.{name}")
    return logger
