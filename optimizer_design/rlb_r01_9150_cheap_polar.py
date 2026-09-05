"""Linear-cost polar surrogates for the archived R01 execution path.

The historical optimizer uses five Newton--Schulz iterations to approximate
the polar factor of each structural momentum batch.  These opt-in quality
experiments retain the same source matrices and target Frobenius norm but
replace the cubic matrix products with either row normalization or one global
Frobenius normalization.  They preserve the RLB functional allocation,
budgets, momentum, LR, and WD, but they are not the exact historical method.
"""

from __future__ import annotations

import math
import threading

import torch

from .rlb_r01_9150_archive import (
    R01Optimizer as _ExactR01Optimizer,
    R02AttentionOptimizer as _ExactR02AttentionOptimizer,
    verify_r01_9150_archive,
)
from .rlb_r01_9150_diagonal_metric import _DiagonalMetricMixin
from .rlb_r01_9150_identity_metric import _IdentityMetricMixin


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
CHEAP_POLAR_FAMILY_ID = "r01_9150_linear_cost_polar_surrogate_v1"

from ._archive_r01_9150 import rlb_r02_core as _attention_module  # noqa: E402
from ._archive_r01_9150 import rlb_r05_core as _router_module  # noqa: E402


_ROUTER_ZERO_POWER = _router_module._batched_zero_power
_ATTENTION_ZERO_POWER = _attention_module._batched_zero_power
_PATCH_LOCK = threading.RLock()


def _cheap_polar(source: torch.Tensor, requested: int, mode: str):
    if int(requested) != 5:
        raise RuntimeError("archived polar iteration inventory changed")
    if source.ndim != 3:
        raise RuntimeError("cheap polar expects a matrix batch")
    transposed = source.shape[-2] > source.shape[-1]
    work = source.transpose(-2, -1) if transposed else source
    work = work.to(dtype=torch.bfloat16)
    eps = 1.0e-7
    if mode == "row":
        norm = torch.linalg.vector_norm(
            work.float(), dim=-1, keepdim=True
        ).clamp_min(eps)
        result = work / norm.to(dtype=work.dtype)
    elif mode == "frobenius":
        norm = torch.linalg.vector_norm(
            work.float(), dim=(-2, -1), keepdim=True
        ).clamp_min(eps)
        target = math.sqrt(float(work.shape[-2]))
        result = work * (target / norm).to(dtype=work.dtype)
    else:
        raise RuntimeError("unknown cheap polar mode")
    return result.transpose(-2, -1) if transposed else result


class _CheapPolarMixin:
    cheap_polar_mode: str
    expected_polar_calls: int
    polar_module = None

    def _evaluate_cheap_polar(self, source, requested):
        self._cheap_polar_call_count += 1
        return _cheap_polar(source, requested, self.cheap_polar_mode)

    @torch.no_grad()
    def step(self, closure=None):
        module = self.polar_module
        if module is None:
            raise RuntimeError("cheap polar module is absent")
        with _PATCH_LOCK:
            expected = (
                _ROUTER_ZERO_POWER
                if module is _router_module
                else _ATTENTION_ZERO_POWER
            )
            if module._batched_zero_power is not expected:
                raise RuntimeError("archived polar kernel was already patched")
            self._cheap_polar_call_count = 0
            module._batched_zero_power = self._evaluate_cheap_polar
            try:
                result = super().step(closure)
                if self._cheap_polar_call_count != self.expected_polar_calls:
                    raise RuntimeError("archived polar call inventory changed")
                return result
            finally:
                module._batched_zero_power = expected

    def cheap_polar_runtime_report(self):
        return {
            "family_id": CHEAP_POLAR_FAMILY_ID,
            "mode": self.cheap_polar_mode,
            "requested_newton_schulz_steps": 5,
            "executed_newton_schulz_steps": 0,
            "target_frobenius_norm_squared": "wide_row_count",
            "rlb_geometry_or_fisher_changed": False,
            "lr_or_wd_changed": False,
            "polar_accuracy_changed": True,
        }


class _CheapRouter(_CheapPolarMixin, _ExactR01Optimizer):
    expected_polar_calls = 3
    polar_module = _router_module


class _CheapAttention(_CheapPolarMixin, _ExactR02AttentionOptimizer):
    expected_polar_calls = 4
    polar_module = _attention_module


class R01RowPolarOptimizer(_CheapRouter):
    cheap_polar_mode = "row"


class R02RowPolarAttentionOptimizer(_CheapAttention):
    cheap_polar_mode = "row"


class R01FrobeniusPolarOptimizer(_CheapRouter):
    cheap_polar_mode = "frobenius"


class R02FrobeniusPolarAttentionOptimizer(_CheapAttention):
    cheap_polar_mode = "frobenius"


class R01DiagonalRowPolarOptimizer(_DiagonalMetricMixin, R01RowPolarOptimizer):
    pass


class R01IdentityRowPolarOptimizer(_IdentityMetricMixin, R01RowPolarOptimizer):
    pass


class R01DiagonalFrobeniusPolarOptimizer(
    _DiagonalMetricMixin, R01FrobeniusPolarOptimizer
):
    pass


class R01IdentityFrobeniusPolarOptimizer(
    _IdentityMetricMixin, R01FrobeniusPolarOptimizer
):
    pass


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "CHEAP_POLAR_FAMILY_ID",
    "R01DiagonalFrobeniusPolarOptimizer",
    "R01DiagonalRowPolarOptimizer",
    "R01FrobeniusPolarOptimizer",
    "R01IdentityFrobeniusPolarOptimizer",
    "R01IdentityRowPolarOptimizer",
    "R01RowPolarOptimizer",
    "R02FrobeniusPolarAttentionOptimizer",
    "R02RowPolarAttentionOptimizer",
)
