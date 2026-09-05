"""Exact lean attention on the current quality-qualified R01 executor.

The current batched-response, compiled-span, compiled-transport R01 path still
uses the archived attention implementation on ordinary transitions.  That
implementation forms condition numbers, branch angles, descent summaries,
and residual certificates after its update direction has already been fixed;
those tensors are consumed only when telemetry is published.  The established
lean R02 attention implementation stops at the same update on ordinary steps
and delegates to the complete historical implementation on telemetry steps.

This composition changes no router equation, response statistic, coordinate
map, span solve, attention update, INT4 publication, LR/WD, or Newton--Schulz
operation.  Quality inheritance requires complete four-rank bitwise equality
to the current R01 parent.
"""

from __future__ import annotations

from . import rlb_r01_9150_local_layer_owner as _owner_module
from .rlb_int4_owner_compiled_transport import (
    R01Int4PaddedCompiledTransportComposite,
)
from .rlb_r01_9150_archive import R02AttentionOptimizer
from .rlb_r01_9150_compiled_response_inverse_int4 import _CONSTRUCTION_LOCK
from .rlb_r01_9150_metric2_exact_lean import (
    R01Metric2ExactLeanAttentionOptimizer,
)


FAMILY_ID = "r01_9150_current_exact_lean_attention_int4_v1"


class R019150CurrentLeanAttentionInt4PaddedComposite(
    R01Int4PaddedCompiledTransportComposite
):
    """Install exact certificate-eliding attention on current R01."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        with _CONSTRUCTION_LOCK:
            original = _owner_module.R02AttentionOptimizer
            if original is not R02AttentionOptimizer:
                raise RuntimeError("R01 owner attention constructor was already patched")
            _owner_module.R02AttentionOptimizer = (
                R01Metric2ExactLeanAttentionOptimizer
            )
            try:
                super().__init__(blocks, adamw, **kwargs)
            finally:
                _owner_module.R02AttentionOptimizer = original
        if not isinstance(
            self.attention, R01Metric2ExactLeanAttentionOptimizer
        ):
            raise RuntimeError("current R01 lean attention was not installed")

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "ordinary_attention_executor": (
                "exact_update_with_telemetry_certificate_elision"
            ),
            "telemetry_attention_executor": "complete_historical_path",
            "attention_update_changed_vs_parent": False,
            "router_equations_changed": False,
            "optimizer_equations_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "quality_inheritance_requires_complete_cuda_bitwise_preflight": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "R019150CurrentLeanAttentionInt4PaddedComposite",
)
