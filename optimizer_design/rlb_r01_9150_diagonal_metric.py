"""Diagonal unit-volume execution approximation for archived R01.

The exact method whitens the outgoing feature and residual-input coordinates
with full Cholesky factors.  This runtime variant keeps the identical observed
metrics and unit-volume normalization but uses their positive diagonals.  It
therefore replaces batched Cholesky/triangular solves by elementwise scaling.
All RLB groups, loss-conditioned coefficient selection, budgets, momentum,
LR, and WD remain unchanged.  A retained variant requires a fresh quality run.
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
DIAGONAL_METRIC_FAMILY_ID = "r01_9150_diagonal_unit_volume_execution_v1"


class _DiagonalMetricMixin:
    _diagonal_metric_calls: int

    def _unit_volume_cholesky(self, metric, *, capture_spectrum=False):
        call = self._r02_metric_factor_call
        if call in (None, 0):
            # Preserve R02's exact symbolic-identity fast path.
            return super()._unit_volume_cholesky(
                metric, capture_spectrum=capture_spectrum
            )
        if capture_spectrum:
            raise RuntimeError(
                "diagonal runtime approximation does not capture a spectrum"
            )
        self._r02_metric_factor_call = int(call) + 1
        diagonal = metric.diagonal(dim1=-2, dim2=-1)
        mean_diagonal = diagonal.mean(dim=-1)
        if bool(
            torch.any(mean_diagonal <= 0.0)
            or torch.any(~torch.isfinite(diagonal))
        ):
            raise RuntimeError("diagonal structural metric is not positive")
        relative_shift = torch.finfo(metric.dtype).eps * metric.shape[-1]
        shifted = diagonal + relative_shift * mean_diagonal.unsqueeze(-1)
        lower_diagonal = torch.sqrt(shifted)
        log_volume = lower_diagonal.log().mean(dim=-1)
        volume = log_volume.exp()
        residual = 2.0 * (
            lower_diagonal.log() - log_volume.unsqueeze(-1)
        ).mean(dim=-1).abs()
        self._diagonal_metric_calls += 1
        return (
            lower_diagonal,
            volume,
            relative_shift,
            None,
            residual,
        )

    @staticmethod
    def _is_diagonal_factor(lower, volume):
        return lower.ndim == volume.ndim + 1

    def _left_coordinate(self, lower, volume, value):
        if self._is_diagonal_factor(lower, volume):
            return (
                value / lower.unsqueeze(-1)
            ) * volume[..., None, None]
        return super()._left_coordinate(lower, volume, value)

    def _left_adjoint(self, lower, volume, value):
        if self._is_diagonal_factor(lower, volume):
            return (
                value / lower.unsqueeze(-1)
            ) * volume[..., None, None]
        return super()._left_adjoint(lower, volume, value)

    def _right_coordinate(self, lower, volume, value):
        if self._is_diagonal_factor(lower, volume):
            return (
                value / lower.unsqueeze(-2)
            ) * volume[..., None, None]
        return super()._right_coordinate(lower, volume, value)

    def _right_adjoint(self, lower, volume, value):
        if self._is_diagonal_factor(lower, volume):
            return (
                value / lower.unsqueeze(-2)
            ) * volume[..., None, None]
        return super()._right_adjoint(lower, volume, value)

    @torch.no_grad()
    def step(self, closure=None):
        self._diagonal_metric_calls = 0
        result = super().step(closure)
        if self._diagonal_metric_calls != 2:
            raise RuntimeError("diagonal metric call inventory changed")
        return result

    def diagonal_metric_runtime_report(self):
        return {
            "family_id": DIAGONAL_METRIC_FAMILY_ID,
            "full_metrics_observed": True,
            "unit_volume_normalization_preserved": True,
            "off_diagonal_coordinate_coupling_approximated": True,
            "expected_diagonal_metric_calls": 2,
            "lr_or_wd_changed": False,
        }


class R01DiagonalMetricOptimizer(_DiagonalMetricMixin, _ExactR01Optimizer):
    pass


class R01DiagonalMetricPolar3Optimizer(
    _DiagonalMetricMixin, R01Polar3Optimizer
):
    pass


class R01DiagonalMetricPolar2Optimizer(
    _DiagonalMetricMixin, R01Polar2Optimizer
):
    pass


class R01DiagonalMetricPolar1Optimizer(
    _DiagonalMetricMixin, R01Polar1Optimizer
):
    pass


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "DIAGONAL_METRIC_FAMILY_ID",
    "R01DiagonalMetricOptimizer",
    "R01DiagonalMetricPolar1Optimizer",
    "R01DiagonalMetricPolar2Optimizer",
    "R01DiagonalMetricPolar3Optimizer",
)
