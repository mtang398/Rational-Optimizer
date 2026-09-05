"""BF16 inverse-matmul stale coordinate approximation for fast R01.

Refresh transitions execute the complete FP32 triangular coordinate maps and
form BF16 inverses of the same cached Cholesky factors.  The seven intervening
steps apply those full, cross-channel inverse maps through Tensor Core matrix
multiplication.  This changes numerical precision, not the represented metric
coupling, and therefore requires its own quality trajectory.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_stale_allocation import (
    R01StaleMetricAllocation8RowOptimizer,
    R02StaleMetricAllocation8RowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
INVERSE_COORDINATE_FAMILY_ID = "r01_stale8_allocation_bf16_inverse_matmul_v1"


class _BF16InverseCoordinateMixin:
    checkpoint_schema = "r01_stale8_allocation_bf16_inverse_matmul_v1"

    def __init__(self, pairs, **kwargs):
        self._cached_bf16_metric_inverses = {}
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["r01_stale_coordinate_mode"] = (
            "full_bf16_inverse_matmul"
        )
        self.param_groups[0]["r01_inverse_coordinate_family_id"] = (
            INVERSE_COORDINATE_FAMILY_ID
        )

    def _unit_volume_cholesky(self, metric, *, capture_spectrum=False):
        call = self._r02_metric_factor_call
        result = super()._unit_volume_cholesky(
            metric, capture_spectrum=capture_spectrum
        )
        if (
            call in (1, 2)
            and self._capture_full_metric_this_step
            and not capture_spectrum
        ):
            lower = result[0]
            dimension = int(lower.shape[-1])
            identity = torch.eye(
                dimension, device=lower.device, dtype=lower.dtype
            ).expand(*lower.shape[:-2], dimension, dimension)
            inverse = torch.linalg.solve_triangular(
                lower, identity, upper=False
            )
            torch._assert_async(torch.isfinite(inverse).all())
            self._cached_bf16_metric_inverses[int(call)] = (
                lower,
                inverse.to(dtype=torch.bfloat16),
            )
        return result

    def _inverse_for(self, lower):
        for call in (1, 2):
            cached = self._cached_bf16_metric_inverses.get(call)
            if cached is not None and lower is cached[0]:
                return cached[1]
        raise RuntimeError("stale BF16 inverse coordinate was not initialized")

    @staticmethod
    def _bf16_matmul(left, right):
        result = torch.matmul(
            left.to(dtype=torch.bfloat16),
            right.to(dtype=torch.bfloat16),
        )
        return result.float()

    def _left_coordinate(self, lower, volume, value):
        if self._capture_full_metric_this_step:
            return super()._left_coordinate(lower, volume, value)
        if lower is self._r02_identity_lower:
            return value
        result = self._bf16_matmul(self._inverse_for(lower), value)
        return result * volume[..., None, None]

    def _left_adjoint(self, lower, volume, value):
        if self._capture_full_metric_this_step:
            return super()._left_adjoint(lower, volume, value)
        if lower is self._r02_identity_lower:
            return value
        inverse_transpose = self._inverse_for(lower).transpose(-2, -1)
        result = self._bf16_matmul(inverse_transpose, value)
        return result * volume[..., None, None]

    def _right_coordinate(self, lower, volume, value):
        if self._capture_full_metric_this_step:
            return super()._right_coordinate(lower, volume, value)
        inverse_transpose = self._inverse_for(lower).transpose(-2, -1)
        result = self._bf16_matmul(value, inverse_transpose)
        return result * volume[..., None, None]

    def _right_adjoint(self, lower, volume, value):
        if self._capture_full_metric_this_step:
            return super()._right_adjoint(lower, volume, value)
        result = self._bf16_matmul(value, self._inverse_for(lower))
        return result * volume[..., None, None]

    def inverse_coordinate_runtime_report(self):
        return {
            "family_id": INVERSE_COORDINATE_FAMILY_ID,
            "refresh_coordinate_map": "fp32_triangular_solve",
            "stale_coordinate_map": "full_bf16_inverse_matmul",
            "cross_channel_metric_coupling_preserved": True,
            "current_budget_and_descent_gates_preserved": True,
            "lr_or_wd_changed": False,
        }


class R01StaleMetricAllocation8BF16InverseRowOptimizer(
    _BF16InverseCoordinateMixin,
    R01StaleMetricAllocation8RowOptimizer,
):
    pass


R02StaleMetricAllocation8BF16InverseRowAttentionOptimizer = (
    R02StaleMetricAllocation8RowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "INVERSE_COORDINATE_FAMILY_ID",
    "R01StaleMetricAllocation8BF16InverseRowOptimizer",
    "R02StaleMetricAllocation8BF16InverseRowAttentionOptimizer",
)
