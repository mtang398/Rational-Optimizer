"""Grouped-collective routers for qualified Method2, Method3, and R01.

Each class is layered directly on its established complete-4,000-step fast
parent.  Only the shared Fisher/cross/count execution is grouped; every
scientific equation, attention implementation, and NS5 call remains unchanged.
"""

from __future__ import annotations

import threading

from ._archive_r01_9150.rlb_r01_core import R01Core as _R01ExactCore
from .rlb_method1_grouped_collectives import (
    _Method1GroupedCollectiveMixin,
)
from .rlb_r07_paired_postpolar_881693_lean import (
    Method2LeanOuter4Optimizer,
)
from .rlb_r10_row_product_881377_metric2 import Method3Metric2Outer4Router
from .rlb_r01_9150_metric2_response_stagger import (
    R01Metric2FusedAllocationResponseStaggerOptimizer,
)


METHOD2_FAMILY_ID = "method2_qualified_outer4_grouped_collectives_v1"
METHOD3_FAMILY_ID = "method3_qualified_outer4_grouped_collectives_v1"
R01_FAMILY_ID = "r01_qualified_metric2_grouped_collectives_v1"
_R01_PATCH_LOCK = threading.RLock()


class Method2GroupedQualifiedRouter(
    _Method1GroupedCollectiveMixin,
    Method2LeanOuter4Optimizer,
):
    """Grouped reductions on quality-qualified fast Method2."""

    _grouped_family_id = METHOD2_FAMILY_ID
    checkpoint_schema = METHOD2_FAMILY_ID


class Method3GroupedQualifiedRouter(
    _Method1GroupedCollectiveMixin,
    Method3Metric2Outer4Router,
):
    """Grouped reductions on quality-qualified fast Method3."""

    _grouped_family_id = METHOD3_FAMILY_ID
    checkpoint_schema = METHOD3_FAMILY_ID


class _R01GroupedCollectiveMixin(_Method1GroupedCollectiveMixin):
    _grouped_family_id = R01_FAMILY_ID
    _grouped_exact_r01_core = _R01ExactCore
    _grouped_exact_global_reducer = staticmethod(
        _R01ExactCore._reduce_global_loss_metric
    )
    _grouped_patch_lock = _R01_PATCH_LOCK


class R01GroupedQualifiedRouter(
    _R01GroupedCollectiveMixin,
    R01Metric2FusedAllocationResponseStaggerOptimizer,
):
    """Grouped reductions on quality-qualified transfer R01."""

    checkpoint_schema = R01_FAMILY_ID


def grouped_other_report():
    return {
        "method2": {
            "family_id": METHOD2_FAMILY_ID,
            "router_parent": Method2LeanOuter4Optimizer.__name__,
            "attention_parent": "Method2LeanAttentionOptimizer",
        },
        "method3": {
            "family_id": METHOD3_FAMILY_ID,
            "router_parent": Method3Metric2Outer4Router.__name__,
            "attention_parent": "Method3LeanAttentionOptimizer",
        },
        "r01": {
            "family_id": R01_FAMILY_ID,
            "router_parent": (
                R01Metric2FusedAllocationResponseStaggerOptimizer.__name__
            ),
            "attention_parent": "R01Metric2ExactLeanAttentionOptimizer",
        },
        "scientific_equations_changed": False,
        "refresh_cadence_changed": False,
        "ns5_changed": False,
        "lr_or_wd_changed": False,
        "production_four_rank_bitwise_gate_required": True,
        "fresh_quality_required_if_any_bitwise_drift": True,
        "all_parents_have_complete_4000_step_quality": True,
    }


__all__ = (
    "METHOD2_FAMILY_ID",
    "METHOD3_FAMILY_ID",
    "R01_FAMILY_ID",
    "Method2GroupedQualifiedRouter",
    "Method3GroupedQualifiedRouter",
    "R01GroupedQualifiedRouter",
    "grouped_other_report",
)
