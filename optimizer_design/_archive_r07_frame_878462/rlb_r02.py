"""Opaque production wrappers for the current R02 generation."""

from .rlb_r02_core import R02AttentionCore, R02Core


class R02Optimizer(R02Core):
    pass


class R02AttentionOptimizer(R02AttentionCore):
    pass
