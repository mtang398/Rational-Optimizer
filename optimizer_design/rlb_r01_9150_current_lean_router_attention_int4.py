"""Exact ordinary-step router elision on the current lean R01 executor.

The current R01 router constructs its R02 endpoint with a budget-normalized
chord, then computes full-matrix descent, residual, and branch-angle
certificates.  R01's global selector does not consume those endpoint
certificates when telemetry is disabled; only the already-fixed chord tensor
is used.  This executor evaluates the literal chord operation sequence and
omits only the trailing ordinary-step certificate reductions.  Telemetry
transitions delegate to the complete historical implementation.

Attention, response statistics, coordinates, global allocation, packed INT4,
LR/WD, and all five Newton--Schulz iterations remain unchanged.  Quality can
be inherited only after complete four-rank bitwise equivalence.
"""

from __future__ import annotations

import torch

from . import rlb_r01_9150_batched_response_inverse_compiled_span_int4 as _batched_module
from ._archive_r01_9150.rlb_r05_revision_core import (
    R05RevisionRouterCore,
)
from .rlb_r01_9150_batched_response_inverse_compiled_span_int4 import (
    _R01BatchedResponseInverseCompiledSpanRouter,
)
from .rlb_r01_9150_compiled_response_inverse_int4 import _CONSTRUCTION_LOCK
from .rlb_r01_9150_current_lean_attention_int4 import (
    R019150CurrentLeanAttentionInt4PaddedComposite,
)
from .rlb_r07_frame_878462_lean_attention import _LeanR02AttentionMixin


FAMILY_ID = "r01_9150_current_exact_lean_router_attention_int4_v1"


class _R01CurrentLeanRouter(_R01BatchedResponseInverseCompiledSpanRouter):
    """Return the exact ordinary R02 chord before telemetry-only products."""

    # The persistent state layout is byte-for-byte the parent layout; retain
    # its schema so checkpoints remain directly interchangeable.
    checkpoint_schema = _R01BatchedResponseInverseCompiledSpanRouter.checkpoint_schema

    def _descent_safe_endpoint(self, parent, adaptive, momentum, alignment):
        if bool(self._capture_telemetry_next_step):
            return super()._descent_safe_endpoint(
                parent, adaptive, momentum, alignment
            )
        del adaptive, alignment
        role = self._r02_endpoint_role
        if (
            role not in (0, 1)
            or len(self._r02_blend_records) != 2
            or self._r02_group_participation is None
            or self._r02_congruences is None
        ):
            raise RuntimeError("R01 lean endpoint construction order changed")
        literal_parent, u5_parent = self._r02_blend_records[role]
        u5, _ = R05RevisionRouterCore._family_route(
            u5_parent,
            momentum,
            self._r02_group_participation[:, role],
            groups=self.groups,
            width=self.width,
        )
        direction = _LeanR02AttentionMixin._ordinary_chord(
            parent,
            u5,
            literal_parent,
            momentum,
            self._r02_congruences[:, role],
            groups=self.groups,
            width=self.width,
        )
        # R01's global selector deletes these inherited scalar arguments.  A
        # shape-correct placeholder preserves the call contract without
        # evaluating certificate-only full-matrix reductions.
        zero = torch.zeros(
            direction.shape[0], device=direction.device, dtype=direction.dtype
        )
        metadata = {
            "parent_descent": zero,
            "endpoint_descent": zero,
        }
        self._r02_endpoint_role += 1
        self._r02_endpoint_records.append(metadata)
        return direction, metadata


class R019150CurrentLeanRouterAttentionInt4PaddedComposite(
    R019150CurrentLeanAttentionInt4PaddedComposite
):
    """Compose exact router and attention certificate elision."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        with _CONSTRUCTION_LOCK:
            original = _batched_module._R01BatchedResponseInverseCompiledSpanRouter
            if original is not _R01BatchedResponseInverseCompiledSpanRouter:
                raise RuntimeError("current R01 router constructor was already patched")
            _batched_module._R01BatchedResponseInverseCompiledSpanRouter = (
                _R01CurrentLeanRouter
            )
            try:
                super().__init__(blocks, adamw, **kwargs)
            finally:
                _batched_module._R01BatchedResponseInverseCompiledSpanRouter = (
                    original
                )
        if not isinstance(self.router, _R01CurrentLeanRouter):
            raise RuntimeError("current R01 lean router was not installed")

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "ordinary_router_executor": (
                "exact_chord_with_telemetry_certificate_elision"
            ),
            "telemetry_router_executor": "complete_historical_path",
            "router_update_changed_vs_parent": False,
            "attention_update_changed_vs_parent": False,
            "optimizer_equations_changed": False,
            "floating_point_update_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "quality_inheritance_requires_complete_cuda_bitwise_preflight": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "R019150CurrentLeanRouterAttentionInt4PaddedComposite",
    "_R01CurrentLeanRouter",
)
