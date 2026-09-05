"""Direct deletion of the historically negative paired endpoint component.

The current fast R01 path already constructs a descent-positive equal-budget
parent direction before ``_descent_safe_endpoint``.  This numerical/scientific
variant selects that parent directly, retaining the response geometry,
functional allocation, momentum, budget, LR, and WD.  It avoids the expensive
two-vector geodesic endpoint whose completed generation was slightly worse
than its own parent.  A clone is required because the archived caller scales
the parent and endpoint tensors independently in place.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_inverse_coordinate import (
    R01StaleMetricAllocation8BF16InverseRowOptimizer,
    R02StaleMetricAllocation8BF16InverseRowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
PARENT_ENDPOINT_FAMILY_ID = "r01_stale8_bf16_inverse_parent_endpoint_v1"


class _ParentEndpointMixin:
    checkpoint_schema = "r01_stale8_bf16_inverse_parent_endpoint_v1"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_parent_endpoint_family_id"] = PARENT_ENDPOINT_FAMILY_ID
        group["r01_paired_geodesic_endpoint_enabled"] = 0.0

    def _descent_safe_endpoint(self, parent, adaptive, momentum, alignment):
        del adaptive, alignment
        role = self._r02_endpoint_role
        if (
            role not in (0, 1)
            or len(self._r02_blend_records) != 2
            or self._r02_congruences is None
        ):
            raise RuntimeError("R01 parent endpoint role state changed")
        literal_parent, _unused_u5_parent = self._r02_blend_records[role]
        if (
            parent.shape != momentum.shape
            or parent.shape != literal_parent.shape
            or parent.ndim != 3
        ):
            raise RuntimeError("R01 parent endpoint geometry changed")
        layers, hidden, external = parent.shape
        if hidden != self.groups * self.width:
            raise RuntimeError("R01 parent endpoint group inventory changed")
        shape = (layers, self.groups, self.width, external)
        source = parent.reshape(shape)
        target = literal_parent.reshape(shape)
        moment = momentum.reshape(shape)
        dims = (-2, -1)
        tiny = torch.finfo(parent.dtype).tiny
        machine = torch.finfo(parent.dtype).eps
        source_norm = torch.linalg.vector_norm(
            source, dim=dims, keepdim=True
        )
        target_norm = torch.linalg.vector_norm(
            target, dim=dims, keepdim=True
        )
        norm_valid = (
            torch.isfinite(source_norm)
            & torch.isfinite(target_norm)
            & (source_norm > 0.0)
            & (target_norm > 0.0)
        )
        torch._assert_async(norm_valid.all())
        direction_blocks = source * (
            target_norm / source_norm.clamp_min(tiny)
        )
        direction_norm = torch.linalg.vector_norm(
            direction_blocks, dim=dims, keepdim=True
        )
        budget_residual = (
            (direction_norm - target_norm).abs()
            / target_norm.clamp_min(1.0)
        )
        group_descent = (moment * direction_blocks).sum(dim=dims)
        parent_descent = group_descent.sum(dim=-1)
        valid = (
            torch.isfinite(group_descent).all(dim=-1)
            & (group_descent > 0.0).all(dim=-1)
            & torch.isfinite(parent_descent)
            & (parent_descent > 0.0)
        )
        torch._assert_async(valid.all())
        torch._assert_async(
            (budget_residual <= 1024.0 * machine).all()
        )
        endpoint = direction_blocks.reshape_as(parent)
        zeros = torch.zeros_like(parent_descent)
        group_zeros = torch.zeros_like(group_descent).flatten()
        group_ones = torch.ones_like(group_descent).flatten()
        congruence = self._r02_congruences[:, role]
        metadata = {
            "congruence": congruence,
            "delta": torch.zeros_like(congruence),
            "pythagorean_residual": zeros,
            "budget_residual": budget_residual.flatten(),
            "u6_descent": parent_descent,
            "u5_descent": parent_descent,
            "chord_descent": parent_descent,
            "u6_group_descent": group_descent.flatten(),
            "u5_group_descent": group_descent.flatten(),
            "chord_group_descent": group_descent.flatten(),
            "branch_cosine": torch.ones_like(group_descent).flatten(),
            "half_angle": group_zeros,
            "response_cap": zeros,
            "branch_cap": group_zeros,
            "descent_cap": group_zeros,
            "gamma": group_ones,
            "budget_residual": zeros,
            "descent_margin": zeros,
            "parent_descent": parent_descent,
            "endpoint_descent": parent_descent,
            "u6_direction": endpoint,
            "u5_direction": endpoint,
        }
        self._r02_endpoint_role += 1
        self._r02_endpoint_records.append(metadata)
        return endpoint, metadata

    def parent_endpoint_runtime_report(self):
        return {
            "family_id": PARENT_ENDPOINT_FAMILY_ID,
            "paired_geodesic_endpoint_enabled": False,
            "selected_direction": "group_budget_normalized_U6_parent",
            "parent_descent_checked_every_step": True,
            "response_geometry_preserved": True,
            "functional_allocation_preserved": True,
            "lr_or_wd_changed": False,
            "fresh_quality_trajectory_required": True,
        }


class R01StaleMetricAllocation8BF16InverseParentEndpointRowOptimizer(
    _ParentEndpointMixin,
    R01StaleMetricAllocation8BF16InverseRowOptimizer,
):
    pass


R02StaleMetricAllocation8BF16InverseParentEndpointRowAttentionOptimizer = (
    R02StaleMetricAllocation8BF16InverseRowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "PARENT_ENDPOINT_FAMILY_ID",
    "R01StaleMetricAllocation8BF16InverseParentEndpointRowOptimizer",
    "R02StaleMetricAllocation8BF16InverseParentEndpointRowAttentionOptimizer",
)
