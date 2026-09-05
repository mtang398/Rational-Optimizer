"""R01 metric-2 candidate with telemetry-only attention certificates.

R01's primary numerical approximation remains its two-step exact covariance
refresh cadence.  This module independently removes only ordinary-step R02
attention certificates after the update direction is fixed.  Telemetry
transitions retain the complete historical attention path, and NS5 remains
exact on every role and transition.
"""

from __future__ import annotations

from .rlb_r01_9150_archive import R02AttentionOptimizer
from .rlb_r01_9150_metric2_exact import R01Metric2ExactOptimizer
from .rlb_r07_frame_878462_lean_attention import _LeanR02AttentionMixin


FAMILY_ID = "r01_9150_metric2_exact_ns5_lean_attention_v1"


class R01Metric2ExactLeanAttentionOptimizer(
    _LeanR02AttentionMixin, R02AttentionOptimizer
):
    """Exact R01 attention update without ordinary telemetry products."""


def runtime_report():
    return {
        "family_id": FAMILY_ID,
        "router": R01Metric2ExactOptimizer.__name__,
        "covariance_refresh_interval": 2,
        "attention_update_changed": False,
        "ns5_changed": False,
        "lr_or_wd_changed": False,
        "telemetry_transitions_use_historical_path": True,
        "ordinary_attention_gate": "bitwise_parameters_and_state",
    }


__all__ = (
    "FAMILY_ID",
    "R01Metric2ExactLeanAttentionOptimizer",
    "runtime_report",
)
