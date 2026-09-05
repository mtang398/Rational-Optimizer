"""Opaque public wrappers for the current R08 candidate slot."""

from .rlb_r02 import R02AttentionOptimizer
from .rlb_r08_next_core import R08NextCore


class R08Optimizer(R08NextCore):
    """Current-P5/Q4 two-role radial natural geometry on complete R01."""


class R08AttentionOptimizer(R02AttentionOptimizer):
    """Literal complete-R01/current-R02 attention transaction."""


__all__ = ("R08Optimizer", "R08AttentionOptimizer")
