"""Opaque public wrappers for the current R03 candidate slot."""

from .rlb_r02 import R02AttentionOptimizer
from .rlb_r03_core import R03Core


class R03Optimizer(R03Core):
    """Persistent exact-P5/Q4 score geometry on complete R08."""


class R03AttentionOptimizer(R02AttentionOptimizer):
    """Literal complete-R08 attention transaction."""


__all__ = ("R03Optimizer", "R03AttentionOptimizer")
