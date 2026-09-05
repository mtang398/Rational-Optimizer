"""Opaque public wrapper for the reusable-slot R09 generation."""

from .rlb_r09_core import R09LossMetricCore


class R09Optimizer(R09LossMetricCore):
    """Downstream-loss geometry in current R02's exact RLB group span."""


__all__ = ("R09Optimizer",)
