"""Opaque public wrapper for the current R01 generation."""

from .rlb_r01_core import R01Core


class R01Optimizer(R01Core):
    """Global cross-layer RLB loss-metric optimizer."""
