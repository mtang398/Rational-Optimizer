"""Opt-in Phase-3 selector-reduction shard for archived R01."""

from .core import (
    FULL_SHARD_PHASE3_ID,
    SELECTOR_FUSION_ONLY_ID,
    R01FullShardPhase3Optimizer,
    R01SelectorFusionOnlyOptimizer,
    R02FullShardPhase3AttentionOptimizer,
    R02SelectorFusionOnlyAttentionOptimizer,
)


__all__ = (
    "FULL_SHARD_PHASE3_ID",
    "SELECTOR_FUSION_ONLY_ID",
    "R01FullShardPhase3Optimizer",
    "R01SelectorFusionOnlyOptimizer",
    "R02FullShardPhase3AttentionOptimizer",
    "R02SelectorFusionOnlyAttentionOptimizer",
)
