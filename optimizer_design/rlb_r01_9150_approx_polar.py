"""Opt-in lower-cost polar execution for the exact archived R01 method.

This module changes only the numerical accuracy used to evaluate the fixed
Muon polar map.  The RLB response metrics, loss-conditioned Fisher selector,
same-budget constraints, momentum, LR, WD, parameter ownership, and optimizer
state are inherited unchanged from the hash-gated 9,150-step R01 archive.

The archive requests five Newton--Schulz iterations at each of its three
router and four attention polar call sites.  The classes below evaluate the
same quintic recurrence for two or three iterations.  They are runtime/quality
experiments, not claims of bitwise equivalence.
"""

from __future__ import annotations

import threading

import torch

from .rlb_r01_9150_archive import (
    R01Optimizer as _ExactR01Optimizer,
    R02AttentionOptimizer as _ExactR02AttentionOptimizer,
    verify_r01_9150_archive,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()

from ._archive_r01_9150 import rlb_group_muon_core as _kernel_module  # noqa: E402
from ._archive_r01_9150 import rlb_r02_core as _attention_module  # noqa: E402
from ._archive_r01_9150 import rlb_r05_core as _router_module  # noqa: E402


APPROX_POLAR_FAMILY_ID = "r01_9150_same_geometry_approximate_polar_v1"
_EXACT_ZERO_POWER = _kernel_module._batched_zero_power
_ROUTER_ZERO_POWER = _router_module._batched_zero_power
_ATTENTION_ZERO_POWER = _attention_module._batched_zero_power
_PATCH_LOCK = threading.RLock()


def _approximate_zero_power(source: torch.Tensor, requested: int, used: int):
    if int(requested) != 5:
        raise RuntimeError("archived polar iteration inventory changed")
    if int(used) not in (1, 2, 3):
        raise RuntimeError("approximate polar iteration count is invalid")
    return _EXACT_ZERO_POWER(source, int(used))


class _ApproximatePolarMixin:
    approximate_polar_steps: int
    expected_polar_calls: int
    polar_module = None

    def _approximate_polar(self, source, requested):
        self._approximate_polar_call_count += 1
        return _approximate_zero_power(
            source, requested, self.approximate_polar_steps
        )

    @torch.no_grad()
    def step(self, closure=None):
        module = self.polar_module
        if module is None:
            raise RuntimeError("approximate polar module is absent")
        with _PATCH_LOCK:
            expected = (
                _ROUTER_ZERO_POWER
                if module is _router_module
                else _ATTENTION_ZERO_POWER
            )
            if module._batched_zero_power is not expected:
                raise RuntimeError("archived polar kernel was already patched")
            self._approximate_polar_call_count = 0
            module._batched_zero_power = self._approximate_polar
            try:
                result = super().step(closure)
                if self._approximate_polar_call_count != self.expected_polar_calls:
                    raise RuntimeError("archived polar call inventory changed")
                return result
            finally:
                module._batched_zero_power = expected

    def runtime_approximation_report(self):
        return {
            "family_id": APPROX_POLAR_FAMILY_ID,
            "requested_newton_schulz_steps": 5,
            "executed_newton_schulz_steps": int(
                self.approximate_polar_steps
            ),
            "expected_polar_calls": int(self.expected_polar_calls),
            "rlb_geometry_changed": False,
            "loss_selector_changed": False,
            "lr_or_wd_changed": False,
            "numerical_polar_accuracy_changed": True,
        }


class _ApproximateRouter(_ApproximatePolarMixin, _ExactR01Optimizer):
    expected_polar_calls = 3
    polar_module = _router_module


class _ApproximateAttention(
    _ApproximatePolarMixin, _ExactR02AttentionOptimizer
):
    expected_polar_calls = 4
    polar_module = _attention_module


class R01Polar3Optimizer(_ApproximateRouter):
    approximate_polar_steps = 3


class R02Polar3AttentionOptimizer(_ApproximateAttention):
    approximate_polar_steps = 3


class R01Polar2Optimizer(_ApproximateRouter):
    approximate_polar_steps = 2


class R02Polar2AttentionOptimizer(_ApproximateAttention):
    approximate_polar_steps = 2


class R01Polar1Optimizer(_ApproximateRouter):
    approximate_polar_steps = 1


class R02Polar1AttentionOptimizer(_ApproximateAttention):
    approximate_polar_steps = 1


__all__ = (
    "APPROX_POLAR_FAMILY_ID",
    "ARCHIVE_CERTIFICATE",
    "R01Polar1Optimizer",
    "R01Polar2Optimizer",
    "R01Polar3Optimizer",
    "R02Polar1AttentionOptimizer",
    "R02Polar2AttentionOptimizer",
    "R02Polar3AttentionOptimizer",
)
