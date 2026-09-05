"""Reuse explicit FP32 metric inverses in qualified packed-INT4 R01.

Each R01 transition applies the same three Cholesky factors through repeated
left/right coordinate and adjoint triangular solves. This implementation
solves each factor against the identity once, then applies the mathematically
identical inverse maps with FP32 matrix multiplication. All response,
allocation, attention, owner-publication, and Newton--Schulz equations remain
unchanged. Explicit inversion changes floating-point realization and requires
fresh complete quality validation if it improves matched runtime.
"""

from __future__ import annotations

import threading

import torch

from . import rlb_r01_9150_local_layer_owner as _owner_module
from .rlb_r01_9150_archive import R01Optimizer
from .rlb_r01_9150_local_owner_int4_direct import (
    R019150LocalLayerOwnerInt4DirectComposite,
)


FAMILY_ID = "r01_9150_inverse_reuse_block256_int4_v1"
_CONSTRUCTION_LOCK = threading.RLock()


def _form_lower_inverse(lower: torch.Tensor) -> torch.Tensor:
    dimension = int(lower.shape[-1])
    identity = torch.eye(
        dimension, device=lower.device, dtype=lower.dtype
    ).expand(*lower.shape[:-2], dimension, dimension)
    return torch.linalg.solve_triangular(lower, identity, upper=False)


class _R01InverseReuseRouter(R01Optimizer):
    checkpoint_schema = FAMILY_ID + "_router"

    def __init__(self, pairs, **kwargs):
        self._inverse_reuse_cache = {}
        super().__init__(pairs, **kwargs)

    def _unit_volume_cholesky(self, metric, *, capture_spectrum=False):
        result = super()._unit_volume_cholesky(
            metric, capture_spectrum=capture_spectrum
        )
        if not capture_spectrum:
            lower = result[0]
            if lower is not self._r02_identity_lower:
                self._inverse_reuse_cache[id(lower)] = (
                    lower, _form_lower_inverse(lower)
                )
        return result

    def _inverse_for(self, lower):
        cached = self._inverse_reuse_cache.get(id(lower))
        if cached is None or cached[0] is not lower:
            raise RuntimeError("R01 inverse-reuse factor was not initialized")
        return cached[1]

    @torch.no_grad()
    def step(self, closure=None):
        self._inverse_reuse_cache.clear()
        return super().step(closure)

    def _left_coordinate(self, lower, volume, value):
        if lower is self._r02_identity_lower:
            return value
        inverse = self._inverse_for(lower)
        result = torch.matmul(inverse, value)
        return result * volume[..., None, None]

    def _left_adjoint(self, lower, volume, value):
        if lower is self._r02_identity_lower:
            return value
        inverse_transpose = self._inverse_for(lower).transpose(-2, -1)
        result = torch.matmul(inverse_transpose, value)
        return result * volume[..., None, None]

    def _right_coordinate(self, lower, volume, value):
        inverse_transpose = self._inverse_for(lower).transpose(-2, -1)
        result = torch.matmul(value, inverse_transpose)
        return result * volume[..., None, None]

    def _right_adjoint(self, lower, volume, value):
        inverse = self._inverse_for(lower)
        result = torch.matmul(value, inverse)
        return result * volume[..., None, None]


class R019150InverseReuseInt4Composite(
    R019150LocalLayerOwnerInt4DirectComposite
):
    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        with _CONSTRUCTION_LOCK:
            original = _owner_module.R01Optimizer
            if original is not R01Optimizer:
                raise RuntimeError("R01 owner router was already patched")
            _owner_module.R01Optimizer = _R01InverseReuseRouter
            try:
                super().__init__(blocks, adamw, **kwargs)
            finally:
                _owner_module.R01Optimizer = original
        if not isinstance(self.router, _R01InverseReuseRouter):
            raise RuntimeError("R01 inverse-reuse router was not installed")

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "scientific_parent_family_id": (
                "r01_9150_local_owner_block256_int4_direct_apply_v1"
            ),
            "metric_factorization": "unchanged_fp32_cholesky",
            "coordinate_executor": "one_fp32_inverse_per_factor_then_matmul",
            "iterative_refinement_steps": 0,
            "coordinate_linear_map_changed": False,
            "r01_equations_changed": False,
            "floating_point_realization_changed": True,
            "owner_publication_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "R019150InverseReuseInt4Composite",
    "_R01InverseReuseRouter",
    "_form_lower_inverse",
)
