#!/usr/bin/env python3
"""Locked DCLM entrypoint for global response-transaction Muon."""

from __future__ import annotations

from experiments.rlb_300m_4000_design_20260731 import (
    candidate_entrypoint_loss_weighted_four_role_response_homotopy_batched_muon_v2 as base,
)
from optimizer_design.rlb_global_response_transaction_muon import (
    FAMILY_ID,
    GlobalResponseTransactionAttentionOptimizer,
    GlobalResponseTransactionRouter,
)


OPTIMIZER_ID = FAMILY_ID
CANDIDATES = {OPTIMIZER_ID: GlobalResponseTransactionRouter}


class GlobalResponseTransactionCompositeOptimizer(
    base.BatchedFourRoleResponseHomotopyCompositeOptimizer
):
    _ROLES = (
        "global_response_transaction_router",
        "global_response_transaction_attention",
        "ordinary_adamw",
    )
    _SCHEMA = "owner_free_global_response_transaction_composite_v1"


R01CompositeOptimizer = GlobalResponseTransactionCompositeOptimizer

base.OPTIMIZER_ID = OPTIMIZER_ID
base.CANDIDATES = CANDIDATES
base.BatchedFourRoleResponseHomotopyRouter = GlobalResponseTransactionRouter
base.BatchedFourRoleResponseHomotopyAttentionOptimizer = (
    GlobalResponseTransactionAttentionOptimizer
)
base.BatchedFourRoleResponseHomotopyCompositeOptimizer = (
    GlobalResponseTransactionCompositeOptimizer
)
base.R01CompositeOptimizer = R01CompositeOptimizer


def configure_candidate_optimizer(model, args):
    return base.configure_candidate_optimizer(model, args)


def clip_candidate_gradients(model, grad_clip, capture_norm):
    return base.clip_candidate_gradients(model, grad_clip, capture_norm)


def main():
    base.main()


if __name__ == "__main__":
    main()
