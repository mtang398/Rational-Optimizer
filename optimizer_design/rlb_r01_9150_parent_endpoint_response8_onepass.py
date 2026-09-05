"""One-pass finite-precision adjoint correction for response8 U6-parent R01.

The archived coordinate adjoint uses two rank-one FP32 correction passes to
restore a bilinear identity.  This approximation retains the same target,
energy, correction axis, and first correction, and deletes only the second
roundoff cleanup pass.  It therefore needs its own timing and quality gates.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_parent_endpoint_response8 import (
    R01StaleMetricAllocation8BF16InverseParentEndpointResponse8RowOptimizer,
    R02StaleMetricAllocation8BF16InverseParentEndpointResponse8RowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
ONEPASS_ADJOINT_FAMILY_ID = (
    "r01_stale8_bf16_inverse_u6_parent_response8_onepass_adjoint_v1"
)


class _OnePassAdjointMixin:
    checkpoint_schema = (
        "r01_stale8_bf16_inverse_u6_parent_response8_onepass_adjoint_v1"
    )

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_adjoint_correction_passes"] = 1
        group["r01_adjoint_correction_family_id"] = ONEPASS_ADJOINT_FAMILY_ID

    @staticmethod
    def _roundoff_compensated_adjoint(
        primal, coordinate_primal, coordinate_dual, adjoint
    ):
        target = (coordinate_primal * coordinate_dual).sum(dim=(-2, -1))
        energy = primal.square().sum(dim=(-2, -1)).clamp_min(
            torch.finfo(primal.dtype).tiny
        )
        observed = (primal * adjoint).sum(dim=(-2, -1))
        coefficient = (target - observed) / energy
        corrected = adjoint + coefficient[:, None, None] * primal
        return corrected, coefficient

    def onepass_adjoint_report(self):
        return {
            "family_id": ONEPASS_ADJOINT_FAMILY_ID,
            "adjoint_correction_passes": 1,
            "correction_target_unchanged": True,
            "correction_axis_unchanged": True,
            "lr_or_wd_changed": False,
            "fresh_quality_trajectory_required": True,
        }


class R01StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowOptimizer(
    _OnePassAdjointMixin,
    R01StaleMetricAllocation8BF16InverseParentEndpointResponse8RowOptimizer,
):
    pass


R02StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer = (
    R02StaleMetricAllocation8BF16InverseParentEndpointResponse8RowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "ONEPASS_ADJOINT_FAMILY_ID",
    "R01StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowOptimizer",
    "R02StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer",
)
