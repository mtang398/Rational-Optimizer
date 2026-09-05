"""Opaque public wrapper for the current R10 candidate slot."""

from .rlb_r10_core import R10AttentionCore


class R10AttentionOptimizer(R10AttentionCore):
    """Row-product coordinates around complete RLB-conditioned attention."""


__all__ = ("R10AttentionOptimizer",)
