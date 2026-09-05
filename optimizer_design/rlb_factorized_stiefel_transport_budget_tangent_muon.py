"""Exact factorized adaptive transport inside an owner-free Global-RLB route.

The strongest completed positive endpoint retained two genuinely different
matrix sources: a factorized adaptive source and a sign-family source.  The
compact descendants kept the loss-aware coefficient transaction but replaced
the adaptive source by directions very close to the already-polarized Muon
parent.  That produced a large step-1,000 lead followed by rapid erosion.

This method restores exact matched-beta2 row/column second moments without a
second Newton--Schulz map.  It transports the factorized adaptive residual
through a diagonal approximation to the Stiefel tangent projector of the
already-computed parent polar, retracts by exact Frobenius normalization, and
keeps that adaptive branch separate from the signed branch.  Global-RLB
participation controls both inner routes, response congruence closes the
outer chord, and the fixed-32 budget-tangent transaction selects fresh signed
layer/group coefficients.

The extra state is O(L H + L G d), independent of activation positions N and
sublinear in parameter count.  Arbitrary shards reduce row/column sufficient
statistics; they never publish a selected parameter-sized update.  There are
no complete-layer owners, dense (LG)x(LG) metrics, dense tangent projectors,
or additional polar maps.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from . import rlb_sketch32_adaptive_bifurcation_budget_tangent_muon as _sketch
from .rlb_budget_tangent_window32_cadence8_group_polar_muon import (
    budget_tangent_window32_scaling_formula,
)


FAMILY_ID = "factorized_stiefel_transport_budget_tangent_muon_v1"
PREFIX = "factorized_stiefel_transport_budget_tangent_"
PARENT_FAMILY_ID = _sketch.FAMILY_ID
PARENT_PREFIX = _sketch.PREFIX
MATCHED_BETA2 = 0.95
LOCKED_EPS = 1.0e-8


def _group_view(
    value: torch.Tensor,
    *,
    groups: int,
    width: int | None,
    grouped_axis: str,
):
    return _sketch._group_view(
        value, groups=groups, width=width, grouped_axis=grouped_axis
    )


def factorized_stiefel_transport_direction(
    parent: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    row_second_moment: torch.Tensor,
    column_second_moment: torch.Tensor,
    participation: torch.Tensor,
    congruence: torch.Tensor,
    *,
    groups: int,
    width: int | None,
    grouped_axis: str,
    beta2: float,
    step: int,
    eps: float = LOCKED_EPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Retract an exact factorized adaptive residual through one polar map."""

    if not (
        parent.shape == momentum.shape == gradient.shape and parent.ndim == 3
    ):
        raise RuntimeError("factorized transport tensor inventory changed")
    if (
        float(beta2) != MATCHED_BETA2
        or int(step) < 1
        or float(eps) != LOCKED_EPS
    ):
        raise ValueError("factorized transport locked numerics changed")

    p, restore = _group_view(
        parent, groups=groups, width=width, grouped_axis=grouped_axis
    )
    m, _ = _group_view(
        momentum, groups=groups, width=width, grouped_axis=grouped_axis
    )
    g, _ = _group_view(
        gradient, groups=groups, width=width, grouped_axis=grouped_axis
    )
    expected = p.shape[:2]
    if (
        participation.shape != expected
        or congruence.shape != expected
        or row_second_moment.shape != p.shape[:-1]
        or column_second_moment.shape != p.shape[:2] + p.shape[-1:]
    ):
        raise RuntimeError("factorized transport state inventory changed")

    squared = g.square()
    row_second_moment.mul_(beta2).add_(
        squared.sum(dim=-1), alpha=1.0 - beta2
    )
    column_second_moment.mul_(beta2).add_(
        squared.sum(dim=-2), alpha=1.0 - beta2
    )
    correction = 1.0 - float(beta2) ** int(step)
    tiny = torch.finfo(p.dtype).tiny
    row_total = row_second_moment.sum(dim=-1, keepdim=True).clamp_min(tiny)
    variance = (
        row_second_moment[..., :, None]
        * column_second_moment[..., None, :]
        / row_total[..., :, None]
    )
    variance.div_(correction)
    inverse_root = torch.reciprocal(variance.sqrt() + float(eps))
    adaptive = m * inverse_root

    dims = (-2, -1)
    parent_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
    adaptive_norm = torch.linalg.vector_norm(
        adaptive, dim=dims, keepdim=True
    )
    sign = torch.sign(m)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = (
        torch.isfinite(p).all(dim=dims, keepdim=True)
        & torch.isfinite(adaptive).all(dim=dims, keepdim=True)
        & torch.isfinite(sign).all(dim=dims, keepdim=True)
        & (parent_norm > 0.0)
        & (adaptive_norm > 0.0)
        & (sign_norm > 0.0)
    )
    adaptive_equal = adaptive * (
        parent_norm / adaptive_norm.clamp_min(tiny)
    )
    sign_equal = sign * (parent_norm / sign_norm.clamp_min(tiny))

    # Remove the common radial mode first, then remove diagonal row/column
    # Gram components.  This is the diagonal Sylvester approximation to the
    # exact rectangular-polar Frechet derivative; it needs only vector
    # reductions and never materializes a dense tangent projector.
    parent_energy = p.square().sum(dim=dims, keepdim=True).clamp_min(tiny)
    residual = adaptive_equal - p * (
        (adaptive_equal * p).sum(dim=dims, keepdim=True) / parent_energy
    )
    row_energy = p.square().sum(dim=-1, keepdim=True).clamp_min(tiny)
    column_energy = p.square().sum(dim=-2, keepdim=True).clamp_min(tiny)
    row_radial = (residual * p).sum(dim=-1, keepdim=True) / row_energy
    column_radial = (residual * p).sum(dim=-2, keepdim=True) / column_energy
    tangent = residual - 0.5 * p * (row_radial + column_radial)
    tangent = tangent - p * (
        (tangent * p).sum(dim=dims, keepdim=True) / parent_energy
    )
    tangent_norm = torch.linalg.vector_norm(tangent, dim=dims, keepdim=True)
    tangent_valid = torch.isfinite(tangent).all(
        dim=dims, keepdim=True
    ) & (tangent_norm > 0.0)
    tangent_equal = tangent * (
        parent_norm / tangent_norm.clamp_min(tiny)
    )

    # A normalized first-order retraction keeps the adaptive route inside a
    # stable 45-degree trust region around the native polar while retaining a
    # direction unavailable to the polar/sign span.
    transported_source = p + tangent_equal
    transported = transported_source * (
        parent_norm
        / torch.linalg.vector_norm(
            transported_source, dim=dims, keepdim=True
        ).clamp_min(tiny)
    )
    transported = torch.where(tangent_valid, transported, p)

    c = participation.float().clamp(0.0, 1.0)[..., None, None]
    root_c = torch.sqrt(c)
    root_residual = torch.sqrt((1.0 - c).clamp_min(0.0))
    u6_source = root_c * p + root_residual * transported
    u5_source = root_c * p + root_residual * sign_equal
    u6 = u6_source * (
        parent_norm
        / torch.linalg.vector_norm(u6_source, dim=dims, keepdim=True).clamp_min(
            tiny
        )
    )
    u5 = u5_source * (
        parent_norm
        / torch.linalg.vector_norm(u5_source, dim=dims, keepdim=True).clamp_min(
            tiny
        )
    )
    u6 = torch.where(c == 1.0, p, u6)
    u5 = torch.where(c == 1.0, p, u5)

    a = congruence.float().clamp(0.0, 1.0)[..., None, None]
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    candidate_source = a * u6 + delta * u5
    candidate = candidate_source * (
        parent_norm
        / torch.linalg.vector_norm(
            candidate_source, dim=dims, keepdim=True
        ).clamp_min(tiny)
    )
    candidate = torch.where(a == 1.0, u6, candidate)
    candidate = torch.where(a == 0.0, u5, candidate)
    candidate = torch.where(valid & tangent_valid, candidate, p)

    descents = torch.stack(
        tuple(
            (g * value).sum(dim=dims)
            for value in (p, transported, u6, u5, candidate)
        ),
        dim=-1,
    )
    safe = (
        valid[..., 0, 0]
        & tangent_valid[..., 0, 0]
        & torch.isfinite(descents).all(dim=-1)
        & (descents[..., 0] > 0.0)
        & (descents[..., 4] > 0.0)
    )
    selected = torch.where(safe[..., None, None], candidate, p)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    parent_norm_flat = parent_norm[..., 0, 0]
    parent_cosine = (
        (p * selected).sum(dim=dims)
        / (parent_norm_flat * selected_norm).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    branch_cosine = (
        (u6 * u5).sum(dim=dims)
        / (
            torch.linalg.vector_norm(u6, dim=dims)
            * torch.linalg.vector_norm(u5, dim=dims)
        ).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    adaptive_parent_cosine = (
        (p * transported).sum(dim=dims)
        / (
            parent_norm_flat
            * torch.linalg.vector_norm(transported, dim=dims)
        ).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    tangent_parent_inner = (
        (p * tangent_equal).sum(dim=dims) / parent_energy[..., 0, 0]
    )
    budget = (
        (selected_norm - parent_norm_flat).abs()
        / parent_norm_flat.clamp_min(1.0)
    )
    torch._assert_async(torch.isfinite(selected).all())
    return restore(selected), {
        "active": valid[..., 0, 0] & tangent_valid[..., 0, 0],
        "safe": safe,
        "parent_cosine": parent_cosine,
        "branch_cosine": branch_cosine,
        "branch_disagreement": 1.0 - branch_cosine,
        "adaptive_parent_cosine": adaptive_parent_cosine,
        "tangent_parent_inner": tangent_parent_inner,
        "budget_residual": budget,
        "branch_descents": descents,
        # The parent wrapper records these but this child removes the bucket
        # labels from externally reported telemetry.
        "row_bins": torch.tensor(int(p.shape[-2]), device=p.device),
        "column_bins": torch.tensor(int(p.shape[-1]), device=p.device),
    }


def factorized_stiefel_transport_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
) -> dict[str, int | float]:
    result = dict(budget_tangent_window32_scaling_formula(
        total_positions=total_positions,
        total_layers=total_layers,
        total_groups=total_groups,
        intermediate_width=intermediate_width,
        model_width=model_width,
    ))
    layers = int(total_layers)
    groups = int(total_groups)
    hidden = int(intermediate_width)
    external = int(model_width)
    # Two MLP roles plus QKV [3d,d] and output [d,d] attention roles.
    factor_state = (
        2 * layers * (hidden + groups * external)
        + 6 * layers * external
    )
    branch_scalars = 20 * layers * (groups + 1)
    # Gradient-square marginals plus two row/column diagonal-tangent packets.
    arbitrary_shard_direction_summary = 3 * factor_state + branch_scalars
    result.update({
        # Compatibility aliases consumed only inside the inherited fused
        # wrapper; child telemetry removes them before publication.
        "sketch_bins_per_axis": 0,
        "factorized_bucket_state_elements": factor_state,
        "factorized_row_column_state_elements": factor_state,
        "persistent_state_elements": (
            int(result["persistent_state_elements"]) + factor_state
        ),
        "communicated_summary_elements": (
            int(result["communicated_summary_elements"])
            + arbitrary_shard_direction_summary
        ),
        "ordinary_communicated_summary_elements": (
            int(result["ordinary_communicated_summary_elements"])
            + arbitrary_shard_direction_summary
        ),
        "arbitrary_shard_direction_summary_elements": (
            arbitrary_shard_direction_summary
        ),
        "additional_persistent_state_elements": factor_state,
        "state_depends_on_total_activation_positions": 0,
        "state_scales_linearly_with_intermediate_and_model_width": 1,
        "additional_dense_solve_dimension": 0,
        "dense_stiefel_projector_elements": 0,
        "additional_native_polar_maps_per_role": 0,
    })
    return result


def _state_for_direction(
    optimizer,
    *,
    parent: torch.Tensor,
    groups: int,
    width: int | None,
    grouped_axis: str,
    key_prefix: str,
):
    grouped, _ = _group_view(
        parent, groups=groups, width=width, grouped_axis=grouped_axis
    )
    row_shape = grouped.shape[:-1]
    column_shape = grouped.shape[:2] + grouped.shape[-1:]
    anchor = optimizer.state[
        optimizer.pairs[0]["in_weight"]
        if hasattr(optimizer, "pairs")
        else optimizer.role_parameters["qkv"][0]
    ]
    row_key = key_prefix + "_exact_row_second_moment"
    column_key = key_prefix + "_exact_column_second_moment"
    row = anchor.get(row_key)
    column = anchor.get(column_key)
    if row is None:
        row = torch.zeros(
            row_shape, device=grouped.device, dtype=torch.float32
        )
        anchor[row_key] = row
    if column is None:
        column = torch.zeros(
            column_shape, device=grouped.device, dtype=torch.float32
        )
        anchor[column_key] = column
    if row.shape != row_shape or column.shape != column_shape:
        raise RuntimeError("factorized transport checkpoint inventory changed")
    return anchor, row, column


@contextmanager
def _installed_parent_hooks():
    """Install child math only while the inherited fused step is active."""

    with _sketch._PATCH_LOCK:
        old_direction = _sketch.sketch32_adaptive_bifurcation_direction
        old_state = _sketch._state_for_direction
        old_scaling = _sketch.sketch32_adaptive_bifurcation_scaling_formula
        _sketch.sketch32_adaptive_bifurcation_direction = (
            factorized_stiefel_transport_direction
        )
        _sketch._state_for_direction = _state_for_direction
        _sketch.sketch32_adaptive_bifurcation_scaling_formula = (
            factorized_stiefel_transport_scaling_formula
        )
        try:
            yield
        finally:
            _sketch.sketch32_adaptive_bifurcation_direction = old_direction
            _sketch._state_for_direction = old_state
            _sketch.sketch32_adaptive_bifurcation_scaling_formula = old_scaling


def _retag_telemetry(report: dict) -> dict:
    renamed = {}
    for key, value in report.items():
        if key.startswith(PARENT_PREFIX):
            key = PREFIX + key[len(PARENT_PREFIX):]
        if value == PARENT_FAMILY_ID:
            value = FAMILY_ID
        renamed[key] = value
    for suffix in (
        "sketch_bins_per_axis",
        "factorized_bucket_state_elements",
        "state_depends_on_intermediate_or_model_width",
        "attention_sketch_bins_per_axis",
    ):
        renamed.pop(PREFIX + suffix, None)
    return renamed


class FactorizedStiefelTransportBudgetTangentRouter(
    _sketch.Sketch32AdaptiveBifurcationBudgetTangentRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = "factorized_stiefel_transport_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0].pop(PARENT_PREFIX + "family_id", None)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "exact_factorized_second_moment_lr_scale": 1.0,
            "diagonal_stiefel_transport_lr_scale": 1.0,
            "adaptive_sign_branch_credit_lr_scale": 1.0,
        })
        return result

    @torch.no_grad()
    def step(self, closure=None):
        with _installed_parent_hooks():
            loss = super().step(closure)
        self._last_telemetry = _retag_telemetry(self._last_telemetry)
        if self._last_telemetry:
            scaling = factorized_stiefel_transport_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            self._last_telemetry.update({
                PREFIX + "family_id": FAMILY_ID,
                PREFIX + "factorized_row_column_state_elements": scaling[
                    "factorized_row_column_state_elements"
                ],
                PREFIX + "state_depends_on_total_activation_positions": 0,
                PREFIX + "state_scales_linearly_with_width": 1,
                PREFIX + "dense_stiefel_projector_elements": 0,
                PREFIX + "additional_native_polar_maps_per_role": 0,
            })
        return loss


class FactorizedStiefelTransportBudgetTangentAttentionOptimizer(
    _sketch.Sketch32AdaptiveBifurcationBudgetTangentAttentionOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0].pop(PARENT_PREFIX + "family_id", None)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "attention_exact_factorized_second_moment_lr_scale": 1.0,
            "attention_diagonal_stiefel_transport_lr_scale": 1.0,
        })
        return result

    @torch.no_grad()
    def step(self, closure=None):
        with _installed_parent_hooks():
            loss = super().step(closure)
        self._last_telemetry = _retag_telemetry(self._last_telemetry)
        if self._last_telemetry:
            self._last_telemetry.update({
                PREFIX + "attention_family_id": FAMILY_ID,
                PREFIX + "attention_owner_count": 0,
                PREFIX + "attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "FAMILY_ID",
    "LOCKED_EPS",
    "MATCHED_BETA2",
    "PREFIX",
    "FactorizedStiefelTransportBudgetTangentAttentionOptimizer",
    "FactorizedStiefelTransportBudgetTangentRouter",
    "factorized_stiefel_transport_direction",
    "factorized_stiefel_transport_scaling_formula",
)
