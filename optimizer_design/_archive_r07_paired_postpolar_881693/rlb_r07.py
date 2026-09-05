"""Opaque public wrapper for the current R07 candidate slot."""

from .rlb_r02 import R02AttentionOptimizer
from .rlb_r07_frame_core import R07PairedAdaptiveFrameCore


class R07Optimizer(R07PairedAdaptiveFrameCore):
    """Paired-adaptive cross-role RLB frame polar."""


class R07AttentionOptimizer(R02AttentionOptimizer):
    """Literal complete-R03 attention transaction."""


__all__ = ("R07Optimizer", "R07AttentionOptimizer")
