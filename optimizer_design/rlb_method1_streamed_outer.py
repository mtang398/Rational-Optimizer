"""Fused/streamed numerical execution for qualified Method1's outer path.

The scientific R01/R05-next/R08/R03/R07 equations are unchanged. The R05
paired-role magnitude construction, feasibility certificates and all four
linear-coordinate projections are one compiled tensor program. Dead full
plus/minus atlas copies are streamed away. The frame axis statistics and
combination use the existing compiled exact reductions. Floating-point
association may differ, so promotion requires a fresh 4,000-step trajectory.
"""

from __future__ import annotations

import torch

from ._method1_streamed_outer.rlb_r07_frame_core import R07FrameCore
from .rlb_method1_grouped_collectives import _Method1GroupedCollectiveMixin
from .rlb_r07_frame_878462_metric2 import _PeriodicOuterMethod1Mixin
from .rlb_recursive_inverse_numerics import _RecursiveAllStepInverseMixin


FAMILY_ID = "method1_streamed_compiled_outer_recursive_inverse_v1"


def _complete_magnitude_atlas_program(
    parent_incoming,
    parent_outgoing,
    unit_incoming,
    unit_outgoing,
    role_direction,
    incoming_gradients,
    outgoing_gradients,
    incoming_momentum,
    outgoing_momentum,
):
    raw_incoming = unit_incoming * role_direction[..., 0, None]
    raw_outgoing = unit_outgoing * role_direction[..., 1, None]
    parent_budget = (
        parent_incoming.square().sum(dim=(-2, -1))
        + parent_outgoing.square().sum(dim=(-2, -1))
    )
    cross = (
        (raw_incoming * parent_incoming).sum(dim=(-2, -1))
        + (raw_outgoing * parent_outgoing).sum(dim=(-2, -1))
    )
    tiny = torch.finfo(parent_incoming.dtype).tiny
    projection = cross / parent_budget.clamp_min(tiny)
    magnitude_incoming = (
        raw_incoming - projection[..., None, None] * parent_incoming
    )
    magnitude_outgoing = (
        raw_outgoing - projection[..., None, None] * parent_outgoing
    )
    magnitude_budget = (
        magnitude_incoming.square().sum(dim=(-2, -1))
        + magnitude_outgoing.square().sum(dim=(-2, -1))
    )
    scale = torch.sqrt(parent_budget / magnitude_budget.clamp_min(tiny))
    magnitude_incoming = magnitude_incoming * scale[..., None, None]
    magnitude_outgoing = magnitude_outgoing * scale[..., None, None]
    closed_budget = (
        magnitude_incoming.square().sum(dim=(-2, -1))
        + magnitude_outgoing.square().sum(dim=(-2, -1))
    )
    closed_cross = (
        (magnitude_incoming * parent_incoming).sum(dim=(-2, -1))
        + (magnitude_outgoing * parent_outgoing).sum(dim=(-2, -1))
    )
    budget_residual = (
        (closed_budget - parent_budget).abs() / parent_budget.clamp_min(1.0)
    )
    orthogonality_residual = closed_cross.abs() / torch.sqrt(
        closed_budget * parent_budget
    ).clamp_min(1.0)
    machine = torch.finfo(parent_incoming.dtype).eps
    valid = (
        torch.isfinite(magnitude_incoming).all(dim=(-2, -1))
        & torch.isfinite(magnitude_outgoing).all(dim=(-2, -1))
        & torch.isfinite(parent_budget)
        & torch.isfinite(magnitude_budget)
        & (parent_budget > 0.0)
        & (magnitude_budget > machine * parent_budget)
        & (budget_residual <= 4096.0 * machine)
        & (orthogonality_residual <= 4096.0 * machine)
    )
    plus_incoming = 0.5 * (parent_incoming + magnitude_incoming)
    minus_incoming = 0.5 * (parent_incoming - magnitude_incoming)
    plus_outgoing = 0.5 * (parent_outgoing + magnitude_outgoing)
    minus_outgoing = 0.5 * (parent_outgoing - magnitude_outgoing)

    incoming_axes = torch.stack((plus_incoming, minus_incoming), dim=2)
    outgoing_axes = torch.stack((plus_outgoing, minus_outgoing), dim=2)
    incoming_exact = (
        incoming_gradients[:, :, None] * incoming_axes
    ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
    outgoing_exact = (
        outgoing_gradients[:, :, None] * outgoing_axes
    ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
    incoming_nesterov = (
        incoming_momentum[:, :, None] * incoming_axes
    ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
    outgoing_nesterov = (
        outgoing_momentum[:, :, None] * outgoing_axes
    ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
    atlas_budget = torch.cat((
        plus_incoming.square().sum(dim=(-2, -1))
        + plus_outgoing.square().sum(dim=(-2, -1)),
        minus_incoming.square().sum(dim=(-2, -1))
        + minus_outgoing.square().sum(dim=(-2, -1)),
    ), dim=1).reshape(1, -1)
    return (
        magnitude_incoming,
        magnitude_outgoing,
        valid,
        parent_budget,
        budget_residual,
        orthogonality_residual,
        plus_incoming,
        minus_incoming,
        plus_outgoing,
        minus_outgoing,
        incoming_exact,
        outgoing_exact,
        incoming_nesterov,
        outgoing_nesterov,
        atlas_budget,
    )


_compiled_complete_magnitude_atlas = torch.compile(
    _complete_magnitude_atlas_program, fullgraph=True, dynamic=False
)


class _StreamedCompiledOuterMixin:
    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self._r07_stream_r05_atlas_enabled = True
        self._r07_axis_materialization_reference = False
        self._r07_compile_axis_statistics = True
        self._r07_compile_axis_combination = True
        group = self.param_groups[0]
        group["rlb_streamed_compiled_outer_family_id"] = FAMILY_ID

    def _complete_magnitude_atlas(self, **packet):
        role = self._r08_role_direction
        if role is None:
            return None
        program = (
            _compiled_complete_magnitude_atlas
            if packet["parent_incoming"].is_cuda
            else _complete_magnitude_atlas_program
        )
        values = program(
            packet["parent_incoming"],
            packet["parent_outgoing"],
            packet["unit_incoming"],
            packet["unit_outgoing"],
            role,
            packet["incoming_gradients"],
            packet["outgoing_gradients"],
            packet["incoming_momentum"],
            packet["outgoing_momentum"],
        )
        (
            magnitude_incoming,
            magnitude_outgoing,
            valid,
            parent_budget,
            budget_residual,
            orthogonality_residual,
            plus_incoming,
            minus_incoming,
            plus_outgoing,
            minus_outgoing,
            incoming_exact,
            outgoing_exact,
            incoming_nesterov,
            outgoing_nesterov,
            atlas_budget,
        ) = values
        metadata = {
            "valid": valid,
            "parent_budget": parent_budget,
            "budget_residual": budget_residual,
            "orthogonality_residual": orthogonality_residual,
        }
        coordinates = (
            incoming_exact,
            outgoing_exact,
            incoming_nesterov,
            outgoing_nesterov,
            atlas_budget,
        )
        return (
            magnitude_incoming,
            magnitude_outgoing,
            metadata,
            plus_incoming,
            minus_incoming,
            plus_outgoing,
            minus_outgoing,
            coordinates,
        )

    def streamed_outer_runtime_report(self):
        return {
            "family_id": FAMILY_ID,
            "r05_magnitude_and_coordinates": "one_compiled_tensor_program",
            "persistent_plus_minus_atlas_copies": False,
            "frame_axis_statistics": "compiled_streamed",
            "frame_axis_combination": "compiled_streamed",
            "newton_schulz_changed": False,
            "scientific_equations_changed": False,
            "floating_point_association_may_change": True,
            "fresh_quality_trajectory_required": True,
        }


class _StreamedOuter4(
    _StreamedCompiledOuterMixin, _PeriodicOuterMethod1Mixin, R07FrameCore
):
    outer_refresh_interval = 4
    checkpoint_schema = FAMILY_ID + "_outer4"


class Method1StreamedOuterRecursiveRouter(
    _RecursiveAllStepInverseMixin,
    _Method1GroupedCollectiveMixin,
    _StreamedOuter4,
):
    checkpoint_schema = FAMILY_ID


__all__ = (
    "FAMILY_ID",
    "Method1StreamedOuterRecursiveRouter",
    "_complete_magnitude_atlas_program",
)
