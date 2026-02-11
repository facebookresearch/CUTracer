# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer Reduce Module.

Provides tools for reducing delay injection configurations to find minimal
sets that trigger data races.
"""

from cutracer.reduce.config_mutator import DelayConfigMutator
from cutracer.reduce.reduce import reduce_delay_points

__all__ = ["reduce_delay_points", "DelayConfigMutator"]
