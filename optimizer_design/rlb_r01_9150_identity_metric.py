"""Identity-coordinate execution approximation for archived R01.

This is an intentionally aggressive runtime/quality experiment.  The exact
R01 functional samples, learned P5/Q4 groups, current-gradient Fisher
allocation, Frobenius budgets, momentum, LR, and WD remain unchanged.  Only
the two observed nontrivial coordinate metrics are approximated by identity,
removing their Cholesky factorizations and all triangular coordinate maps.

The historical archive is never modified.  These classes require fresh
4,000-step and 9,150-step quality runs before they may be retained.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import (
    R01Optimizer as _ExactR01Optimizer,
    verify_r01_9150_archive,
)
from .rlb_r01_9150_approx_polar import (
    R01Polar1Optimizer,
    R01Polar2Optimizer,
    R01Polar3Optimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
IDENTITY_METRIC_FAMILY_ID = "r01_9150_identity_coordinate_execution_v1"


class _IdentityMetricMixin:
    _identity_metric_calls: int

    def _unit_volume_cholesky(self, metric, *, capture_spectrum=False):
        call = self._r02_metric_factor_call
        if call in (None, 0):
            # Keep the parent's exact symbolic incoming-identity path.
            return super()._unit_volume_cholesky(
                metric, capture_spectrum=capture_spectrum
            )
        if capture_spectrum:
            raise RuntimeError(
                "identity runtime approximation does not capture a spectrum"
            )
        self._r02_metric_factor_call = int(call) + 1
        torch._assert_async(torch.isfinite(metric).all())
        volume = torch.ones(
            metric.shape[:-2], device=metric.device, dtype=metric.dtype
        )
        # A factor with the same rank/shape as ``volume`` is an unambiguous
        # private marker: exact full factors have two extra dimensions and
        # the diagonal approximation has one extra dimension.
        factor = torch.ones_like(volume)
        residual = torch.zeros_like(volume)
        self._identity_metric_calls += 1
        return factor, volume, 0.0, None, residual

    @staticmethod
    def _is_identity_factor(lower, volume):
        return lower.shape == volume.shape

    def _left_coordinate(self, lower, volume, value):
        if self._is_identity_factor(lower, volume):
            return value
        return super()._left_coordinate(lower, volume, value)

    def _left_adjoint(self, lower, volume, value):
        if self._is_identity_factor(lower, volume):
            return value
        return super()._left_adjoint(lower, volume, value)

    def _right_coordinate(self, lower, volume, value):
        if self._is_identity_factor(lower, volume):
            return value
        return super()._right_coordinate(lower, volume, value)

    def _right_adjoint(self, lower, volume, value):
        if self._is_identity_factor(lower, volume):
            return value
        return super()._right_adjoint(lower, volume, value)

    @torch.no_grad()
    def step(self, closure=None):
        self._identity_metric_calls = 0
        result = super().step(closure)
        if self._identity_metric_calls != 2:
            raise RuntimeError("identity metric call inventory changed")
        return result

    def identity_metric_runtime_report(self):
        return {
            "family_id": IDENTITY_METRIC_FAMILY_ID,
            "full_metrics_observed": True,
            "incoming_symbolic_identity_preserved": True,
            "outgoing_and_input_metrics_approximated_by_identity": True,
            "expected_identity_metric_calls": 2,
            "rlb_groups_or_fisher_changed": False,
            "lr_or_wd_changed": False,
        }


class R01IdentityMetricOptimizer(_IdentityMetricMixin, _ExactR01Optimizer):
    pass


class R01IdentityMetricPolar3Optimizer(
    _IdentityMetricMixin, R01Polar3Optimizer
):
    pass


class R01IdentityMetricPolar2Optimizer(
    _IdentityMetricMixin, R01Polar2Optimizer
):
    pass


class R01IdentityMetricPolar1Optimizer(
    _IdentityMetricMixin, R01Polar1Optimizer
):
    pass


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "IDENTITY_METRIC_FAMILY_ID",
    "R01IdentityMetricOptimizer",
    "R01IdentityMetricPolar1Optimizer",
    "R01IdentityMetricPolar2Optimizer",
    "R01IdentityMetricPolar3Optimizer",
)
