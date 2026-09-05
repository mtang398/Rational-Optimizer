"""Attention-only NS5 batching bridge for recovered Method2/job 881693_0."""

from __future__ import annotations

from optimizer_design.rlb_r01_9150_fast import R02FastAttentionOptimizer


class R07PairedPostpolar881693AttentionBatchOptimizer(
    R02FastAttentionOptimizer
):
    """Exact R02 attention equations with only independent NS5 calls batched."""


__all__ = ("R07PairedPostpolar881693AttentionBatchOptimizer",)
