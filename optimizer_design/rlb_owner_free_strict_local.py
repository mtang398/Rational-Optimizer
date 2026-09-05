"""Strict local execution stack for the owner-free R01 frame family.

The owner-free frame optimizers originally inherited the archived eager R01
router directly.  The quality-qualified R01 lineage already contains several
strict execution improvements that do not depend on layer ownership or update
publication:

* one explicit FP32 inverse per response metric factor, reused by matmuls;
* one fixed-shape compiled P5/Q4 response-statistics program;
* one compiled implementation of the literal 64-round secular bisection;
* ordinary-step elision of endpoint certificates that the selector does not
  consume, with the historical path retained for telemetry transitions; and
* direct scalar loss-score contraction without residual-width tangent images.

This class composes those local operations through cooperative multiple
inheritance.  It deliberately imports no owner composite and performs no
selected-update publication.  Every final direction equation, LR/WD scale,
and all five Newton--Schulz iterations remain inherited unchanged.
"""

from __future__ import annotations

from .rlb_r01_9150_current_lean_router_attention_int4 import (
    _R01CurrentLeanRouter,
)
from .rlb_r01_9150_direct_scores import R01DirectScoreOptimizer


STRICT_LOCAL_EXECUTION_ID = "owner_free_r01_strict_local_execution_v1"


class OwnerFreeStrictLocalR01Optimizer(
    R01DirectScoreOptimizer,
    _R01CurrentLeanRouter,
):
    """Exact R01 router equations with the proven strict local executors."""

    execution_variant = STRICT_LOCAL_EXECUTION_ID

    def strict_local_execution_audit(self):
        return {
            "explicit_response_metric_inverse_reuse": True,
            "compiled_batched_response_statistics": True,
            "compiled_literal_span64": True,
            "ordinary_router_certificate_elision_only": True,
            "direct_loss_score_contraction": True,
            "complete_layer_owners": 0,
            "selected_update_elements_published": 0,
            "optimizer_equations_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.ns_steps),
        }


__all__ = (
    "OwnerFreeStrictLocalR01Optimizer",
    "STRICT_LOCAL_EXECUTION_ID",
)
