#!/usr/bin/env python3
"""Locked DCLM entrypoint for factorized every-step RFD gradient ledger."""

from experiments.rlb_300m_4000_design_20260731 import (
    candidate_entrypoint_global_response_transaction_muon_v1 as base,
)
from optimizer_design.rlb_factorized_every_step_rfd_gradient_ledger_muon import (
    FAMILY_ID,
    FactorizedEveryStepRFDGradientLedgerAttentionOptimizer,
    FactorizedEveryStepRFDGradientLedgerRouter,
)


OPTIMIZER_ID = FAMILY_ID
CANDIDATES = {OPTIMIZER_ID: FactorizedEveryStepRFDGradientLedgerRouter}


class FactorizedEveryStepRFDGradientLedgerCompositeOptimizer(
    base.GlobalResponseTransactionCompositeOptimizer
):
    _ROLES = (
        "factorized_every_step_rfd_gradient_ledger_router",
        "unchanged_compiled_factorized_attention",
        "ordinary_adamw",
    )
    _SCHEMA = "owner_free_factorized_every_step_rfd_gradient_ledger_v1"


R01CompositeOptimizer = FactorizedEveryStepRFDGradientLedgerCompositeOptimizer
base.OPTIMIZER_ID = OPTIMIZER_ID
base.CANDIDATES = CANDIDATES
base.GlobalResponseTransactionRouter = FactorizedEveryStepRFDGradientLedgerRouter
base.GlobalResponseTransactionAttentionOptimizer = (
    FactorizedEveryStepRFDGradientLedgerAttentionOptimizer
)
base.GlobalResponseTransactionCompositeOptimizer = (
    FactorizedEveryStepRFDGradientLedgerCompositeOptimizer
)
base.R01CompositeOptimizer = R01CompositeOptimizer
base.base.OPTIMIZER_ID = OPTIMIZER_ID
base.base.CANDIDATES = CANDIDATES
base.base.BatchedFourRoleResponseHomotopyRouter = (
    FactorizedEveryStepRFDGradientLedgerRouter
)
base.base.BatchedFourRoleResponseHomotopyAttentionOptimizer = (
    FactorizedEveryStepRFDGradientLedgerAttentionOptimizer
)
base.base.BatchedFourRoleResponseHomotopyCompositeOptimizer = (
    FactorizedEveryStepRFDGradientLedgerCompositeOptimizer
)
base.base.R01CompositeOptimizer = R01CompositeOptimizer


def configure_candidate_optimizer(model, args):
    return base.configure_candidate_optimizer(model, args)


def clip_candidate_gradients(model, grad_clip, capture_norm):
    return base.clip_candidate_gradients(model, grad_clip, capture_norm)


def main():
    base.main()


if __name__ == "__main__":
    main()


