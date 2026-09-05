"""Stable four-role factorized adaptive tangent chords for Global-RLB.

The strongest later-positive inexpensive direction used a response-funded
sign tangent whose median parent cosine remained near 0.96.  In contrast, the
factorized Stiefel candidate normalizes its adaptive tangent to the full parent
norm before nested routing, producing much larger early angular departures.
This method keeps the proven conservative angular budget but replaces the
coarse sign source by an exact matched-beta2 row/column adaptive source.

For every MLP rational group and attention role, the factorized adaptive
residual is projected off the native polar, oriented by the current gradient,
and assigned energy ``participation * (1-congruence**2)``.  The chord has the
exact native norm and signed span coefficients.  There is one native NS5 map,
no dense projector or solve, no owner, and no selected-update publication.
Persistent state is O(LH + LGd), independent of activation positions N.
"""

from __future__ import annotations

import threading

import torch

from . import rlb_compact_four_role_response_homotopy_muon as _compact
from . import rlb_lagged_predictive_response_transaction_muon as _predictive
from . import rlb_rank64_cadence8_response_group_polar_muon as _cadence
from .rlb_factorized_stiefel_transport_budget_tangent_muon import _group_view
from .rlb_window32_cadence8_response_group_polar_muon import (
    FAMILY_ID as WINDOW32_FAMILY_ID,
    PREFIX as WINDOW32_PREFIX,
    Window32Cadence8HeadPolarAttentionOptimizer,
    Window32Cadence8ResponseGroupPolarRouter,
    window32_cadence8_scaling_formula,
)


FAMILY_ID = "factorized_adaptive_tangent_chord_muon_v1"
PREFIX = "factorized_adaptive_tangent_chord_"
LOCKED_EPS = 1.0e-8
MATCHED_BETA2 = 0.95
_PATCH_LOCK = threading.RLock()
_EXPECTED_REFRESH_DIRECTION = _predictive.compact_postpolar_group_response_homotopy
_EXPECTED_ORDINARY_DIRECTION = _cadence.compact_postpolar_group_response_homotopy
_EXPECTED_ATTENTION_DIRECTION = _compact.compact_postpolar_group_response_homotopy


def factorized_adaptive_tangent_chord_direction(
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
):
    """Spend the stable response energy on a factorized adaptive tangent."""

    if not (
        parent.shape == momentum.shape == gradient.shape and parent.ndim == 3
    ):
        raise RuntimeError("factorized adaptive tangent inventory changed")
    if float(beta2) != MATCHED_BETA2 or int(step) < 1 or float(eps) != LOCKED_EPS:
        raise ValueError("factorized adaptive tangent locked numerics changed")
    p, restore = _group_view(
        parent.float(), groups=groups, width=width, grouped_axis=grouped_axis
    )
    m, _ = _group_view(
        momentum.float(), groups=groups, width=width, grouped_axis=grouped_axis
    )
    g, _ = _group_view(
        gradient.float(), groups=groups, width=width, grouped_axis=grouped_axis
    )
    if (
        participation.shape != p.shape[:2]
        or congruence.shape != p.shape[:2]
        or row_second_moment.shape != p.shape[:-1]
        or column_second_moment.shape != p.shape[:2] + p.shape[-1:]
    ):
        raise RuntimeError("factorized adaptive tangent state inventory changed")

    squared = g.square()
    row_second_moment.mul_(beta2).add_(
        squared.sum(dim=-1), alpha=1.0 - beta2
    )
    column_second_moment.mul_(beta2).add_(
        squared.sum(dim=-2), alpha=1.0 - beta2
    )
    correction = 1.0 - float(beta2) ** int(step)
    tiny = torch.finfo(torch.float32).tiny
    row_total = row_second_moment.sum(dim=-1, keepdim=True).clamp_min(tiny)
    variance = (
        row_second_moment[..., :, None]
        * column_second_moment[..., None, :]
        / row_total[..., :, None]
    ) / correction
    adaptive = m / (variance.sqrt() + float(eps))

    dims = (-2, -1)
    parent2 = p.square().sum(dim=dims)
    adaptive2 = adaptive.square().sum(dim=dims)
    parent_adaptive = (p * adaptive).sum(dim=dims)
    parent_descent = (g * p).sum(dim=dims)
    adaptive_descent = (g * adaptive).sum(dim=dims)
    projection = parent_adaptive / parent2.clamp_min(tiny)
    tangent2 = (
        adaptive2 - parent_adaptive.square() / parent2.clamp_min(tiny)
    ).clamp_min(0.0)
    tangent_descent_signed = adaptive_descent - projection * parent_descent
    orientation = torch.where(
        tangent_descent_signed >= 0.0,
        torch.ones_like(tangent_descent_signed),
        -torch.ones_like(tangent_descent_signed),
    )

    information = participation.float().clamp(0.0, 1.0)
    alignment = congruence.float().clamp(0.0, 1.0)
    departure_energy = (
        information * (1.0 - alignment.square()).clamp_min(0.0)
    ).clamp(0.0, 1.0)
    parent_weight = torch.sqrt((1.0 - departure_energy).clamp_min(0.0))
    tangent_weight = torch.sqrt(departure_energy.clamp_min(0.0))
    adaptive_coefficient = (
        tangent_weight
        * torch.sqrt(parent2 / tangent2.clamp_min(tiny))
        * orientation
    )
    parent_coefficient = parent_weight - adaptive_coefficient * projection
    candidate_descent = (
        parent_coefficient * parent_descent
        + adaptive_coefficient * adaptive_descent
    )
    finite = (
        torch.isfinite(p).all(dim=dims)
        & torch.isfinite(m).all(dim=dims)
        & torch.isfinite(g).all(dim=dims)
        & torch.isfinite(adaptive).all(dim=dims)
        & torch.isfinite(parent_coefficient)
        & torch.isfinite(adaptive_coefficient)
        & torch.isfinite(candidate_descent)
    )
    active = (
        finite
        & (parent2 > 0.0)
        & (adaptive2 > 0.0)
        & (tangent2 > tiny)
    )
    safe = active & (parent_descent > 0.0) & (candidate_descent > 0.0)
    parent_coefficient = torch.where(
        safe, parent_coefficient, torch.ones_like(parent_coefficient)
    )
    adaptive_coefficient = torch.where(
        safe, adaptive_coefficient, torch.zeros_like(adaptive_coefficient)
    )

    selected2 = (
        parent_coefficient.square() * parent2
        + adaptive_coefficient.square() * adaptive2
        + 2.0 * parent_coefficient * adaptive_coefficient * parent_adaptive
    ).clamp_min(0.0)
    selected_norm = torch.sqrt(selected2)
    parent_norm = torch.sqrt(parent2.clamp_min(0.0))
    parent_cosine = (
        (parent_coefficient * parent2 + adaptive_coefficient * parent_adaptive)
        / (selected_norm * parent_norm).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    budget = (
        (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)
    )
    shape = (*parent_coefficient.shape, 1, 1)
    p.mul_(parent_coefficient.view(shape))
    p.addcmul_(adaptive, adaptive_coefficient.view(shape))
    selected = restore(p).to(parent.dtype)
    torch._assert_async(torch.isfinite(selected).all())
    return selected, {
        "active": active,
        "safe": safe,
        "parent_cosine": parent_cosine,
        "budget_residual": budget,
        "departure_energy": departure_energy,
        "parent_descent": parent_descent,
        "candidate_descent": candidate_descent,
        "parent_coefficient": parent_coefficient,
        "adaptive_coefficient": adaptive_coefficient,
        "tangent_orientation": orientation,
    }


def _factor_state_elements(*, layers: int, groups: int, hidden: int, external: int):
    # Two MLP roles plus QKV [3d,d] and output [d,d] attention roles.
    return 2 * layers * (hidden + groups * external) + 6 * layers * external


def factorized_adaptive_tangent_chord_scaling_formula(**kwargs):
    result = dict(window32_cadence8_scaling_formula(**kwargs))
    layers = int(kwargs["total_layers"])
    groups = int(kwargs["total_groups"])
    hidden = int(kwargs["intermediate_width"])
    external = int(kwargs["model_width"])
    factor_state = _factor_state_elements(
        layers=layers, groups=groups, hidden=hidden, external=external
    )
    # Conservative arbitrary-rectangle upper bound: gradient-square
    # marginals, two factor packets, and five scalar tangent moments.
    direction_summary = 3 * factor_state + 5 * layers * (groups + 1)
    result.update({
        "factorized_row_column_state_elements": factor_state,
        "persistent_state_elements": int(result["persistent_state_elements"])
        + factor_state,
        "arbitrary_shard_direction_summary_elements": direction_summary,
        "communicated_summary_elements": int(result["communicated_summary_elements"])
        + direction_summary,
        "ordinary_communicated_summary_elements": int(
            result["ordinary_communicated_summary_elements"]
        ) + direction_summary,
        "additional_persistent_state_elements": factor_state,
        "state_depends_on_total_activation_positions": 0,
        "state_scales_linearly_with_width": 1,
        "additional_native_polar_maps_per_role": 0,
        "dense_tangent_projector_elements": 0,
        "additional_dense_solve_dimension": 0,
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
    if hasattr(optimizer, "pairs"):
        anchor = optimizer.state[optimizer.pairs[0]["in_weight"]]
    else:
        anchor = optimizer.state[optimizer.role_parameters["qkv"][0]]
    row_key = key_prefix + "_row_second_moment"
    column_key = key_prefix + "_column_second_moment"
    row = anchor.get(row_key)
    column = anchor.get(column_key)
    if row is None:
        row = torch.zeros(
            grouped.shape[:-1], device=grouped.device, dtype=torch.float32
        )
        anchor[row_key] = row
    if column is None:
        column = torch.zeros(
            grouped.shape[:2] + grouped.shape[-1:],
            device=grouped.device,
            dtype=torch.float32,
        )
        anchor[column_key] = column
    if row.shape != grouped.shape[:-1] or column.shape != (
        grouped.shape[:2] + grouped.shape[-1:]
    ):
        raise RuntimeError("factorized adaptive tangent checkpoint inventory changed")
    return anchor, row, column


def _retag(report: dict) -> dict:
    return {
        key.replace(WINDOW32_PREFIX, PREFIX, 1): (
            FAMILY_ID if value == WINDOW32_FAMILY_ID else value
        )
        for key, value in report.items()
    }


class FactorizedAdaptiveTangentChordRouter(
    Window32Cadence8ResponseGroupPolarRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = "factorized_adaptive_tangent_chord_lr_scale"

    def __init__(self, pairs, **kwargs):
        if float(kwargs.get("beta2", -1.0)) != MATCHED_BETA2:
            raise ValueError("factorized adaptive tangent requires beta2=.95")
        if float(kwargs.get("eps", -1.0)) != LOCKED_EPS:
            raise ValueError("factorized adaptive tangent requires eps=1e-8")
        super().__init__(pairs, **kwargs)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "matched_beta2_factorized_tangent_lr_scale": 1.0,
            "response_energy_chord_lr_scale": 1.0,
            "signed_current_gradient_orientation_lr_scale": 1.0,
            "exact_native_budget_lr_scale": 1.0,
        })
        return result

    @torch.no_grad()
    def step(self, closure=None):
        records = []

        def direction(
            parent, momentum, gradient, participation, congruence,
            *, groups, width, grouped_axis, eps=LOCKED_EPS,
        ):
            anchor, rows, columns = _state_for_direction(
                self,
                parent=parent,
                groups=groups,
                width=width,
                grouped_axis=grouped_axis,
                key_prefix="factorized_adaptive_tangent_" + grouped_axis,
            )
            update = int(anchor.get("predictive_response_transaction_updates", 0)) + 1
            selected, metadata = factorized_adaptive_tangent_chord_direction(
                parent, momentum, gradient, rows, columns,
                participation, congruence,
                groups=groups, width=width, grouped_axis=grouped_axis,
                beta2=MATCHED_BETA2, step=update, eps=eps,
            )
            records.append(metadata)
            return selected, metadata

        with _PATCH_LOCK:
            if (
                _predictive.compact_postpolar_group_response_homotopy
                is not _EXPECTED_REFRESH_DIRECTION
                or _cadence.compact_postpolar_group_response_homotopy
                is not _EXPECTED_ORDINARY_DIRECTION
            ):
                raise RuntimeError("factorized adaptive router binding changed")
            _predictive.compact_postpolar_group_response_homotopy = direction
            _cadence.compact_postpolar_group_response_homotopy = direction
            try:
                loss = super().step(closure)
            finally:
                _predictive.compact_postpolar_group_response_homotopy = (
                    _EXPECTED_REFRESH_DIRECTION
                )
                _cadence.compact_postpolar_group_response_homotopy = (
                    _EXPECTED_ORDINARY_DIRECTION
                )
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry and records:
            safe = torch.cat([x["safe"].reshape(-1) for x in records])
            cosine = torch.cat([x["parent_cosine"].reshape(-1) for x in records])
            budget = torch.cat([x["budget_residual"].reshape(-1) for x in records])
            energy = torch.cat([x["departure_energy"].reshape(-1) for x in records])
            signed = torch.cat([x["adaptive_coefficient"].reshape(-1) for x in records])
            scaling = factorized_adaptive_tangent_chord_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            self._last_telemetry.update({
                PREFIX + "family_id": FAMILY_ID,
                PREFIX + "adaptive_tangent_safe_fraction": float(safe.float().mean()),
                PREFIX + "adaptive_tangent_parent_cosine_median": float(cosine.median()),
                PREFIX + "adaptive_tangent_budget_residual_max": float(budget.amax()),
                PREFIX + "departure_energy_median": float(energy.median()),
                PREFIX + "adaptive_coefficient_min": float(signed.amin()),
                PREFIX + "adaptive_coefficient_max": float(signed.amax()),
                PREFIX + "factorized_row_column_state_elements": scaling[
                    "factorized_row_column_state_elements"
                ],
                PREFIX + "arbitrary_shard_direction_summary_elements": scaling[
                    "arbitrary_shard_direction_summary_elements"
                ],
                PREFIX + "state_depends_on_total_activation_positions": 0,
                PREFIX + "state_scales_linearly_with_width": 1,
                PREFIX + "additional_native_polar_maps_per_role": 0,
                PREFIX + "dense_tangent_projector_elements": 0,
            })
        return loss


class FactorizedAdaptiveTangentChordAttentionOptimizer(
    Window32Cadence8HeadPolarAttentionOptimizer
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "attention_matched_beta2_factorized_tangent_lr_scale": 1.0,
            "attention_response_energy_chord_lr_scale": 1.0,
            "attention_signed_current_gradient_orientation_lr_scale": 1.0,
        })
        return result

    @torch.no_grad()
    def step(self, closure=None):
        records = []
        calls = 0
        anchor = self.state[self.role_parameters["qkv"][0]]
        update = int(anchor.get("factorized_adaptive_tangent_updates", 0)) + 1

        def direction(
            parent, momentum, gradient, participation, congruence,
            *, groups, width, grouped_axis, eps=LOCKED_EPS,
        ):
            nonlocal calls
            if calls >= 2:
                raise RuntimeError("factorized adaptive attention role count changed")
            role = "qkv" if calls == 0 else "attention_output"
            calls += 1
            _anchor, rows, columns = _state_for_direction(
                self,
                parent=parent,
                groups=groups,
                width=width,
                grouped_axis=grouped_axis,
                key_prefix="factorized_adaptive_tangent_" + role,
            )
            selected, metadata = factorized_adaptive_tangent_chord_direction(
                parent, momentum, gradient, rows, columns,
                participation, congruence,
                groups=groups, width=width, grouped_axis=grouped_axis,
                beta2=MATCHED_BETA2, step=update, eps=eps,
            )
            records.append(metadata)
            return selected, metadata

        with _PATCH_LOCK:
            if _compact.compact_postpolar_group_response_homotopy is not (
                _EXPECTED_ATTENTION_DIRECTION
            ):
                raise RuntimeError("factorized adaptive attention binding changed")
            _compact.compact_postpolar_group_response_homotopy = direction
            try:
                loss = super().step(closure)
            finally:
                _compact.compact_postpolar_group_response_homotopy = (
                    _EXPECTED_ATTENTION_DIRECTION
                )
        if calls != 2:
            raise RuntimeError("factorized adaptive attention omitted a role")
        anchor["factorized_adaptive_tangent_updates"] = update
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry:
            safe = torch.cat([x["safe"].reshape(-1) for x in records])
            cosine = torch.cat([x["parent_cosine"].reshape(-1) for x in records])
            self._last_telemetry.update({
                PREFIX + "attention_family_id": FAMILY_ID,
                PREFIX + "attention_owner_count": 0,
                PREFIX + "attention_selected_update_elements_published": 0,
                PREFIX + "attention_adaptive_tangent_safe_fraction": float(
                    safe.float().mean()
                ),
                PREFIX + "attention_adaptive_tangent_parent_cosine_median": float(
                    cosine.median()
                ),
            })
        return loss


__all__ = (
    "FAMILY_ID",
    "PREFIX",
    "FactorizedAdaptiveTangentChordAttentionOptimizer",
    "FactorizedAdaptiveTangentChordRouter",
    "factorized_adaptive_tangent_chord_direction",
    "factorized_adaptive_tangent_chord_scaling_formula",
)
