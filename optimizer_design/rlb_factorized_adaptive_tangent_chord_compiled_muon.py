"""Compiled execution of the unchanged adaptive tangent chord equation."""

from __future__ import annotations

from contextlib import contextmanager
import threading

import torch

from . import rlb_factorized_adaptive_tangent_chord_muon as _parent


FAMILY_ID = "factorized_adaptive_tangent_chord_compiled_muon_v2"
PREFIX = "factorized_adaptive_tangent_chord_compiled_"
PARENT_FAMILY_ID = _parent.FAMILY_ID
PARENT_PREFIX = _parent.PREFIX
LOCKED_EPS = _parent.LOCKED_EPS
MATCHED_BETA2 = _parent.MATCHED_BETA2
_PATCH_LOCK = threading.RLock()
_TINY = torch.finfo(torch.float32).tiny


def _adaptive_tangent_chord_fullgraph(
    p: torch.Tensor,
    m: torch.Tensor,
    g: torch.Tensor,
    row_second_moment: torch.Tensor,
    column_second_moment: torch.Tensor,
    participation: torch.Tensor,
    congruence: torch.Tensor,
    correction: torch.Tensor,
):
    squared = g.square()
    row_second_moment.mul_(0.95).add_(squared.sum(dim=-1), alpha=0.05)
    column_second_moment.mul_(0.95).add_(squared.sum(dim=-2), alpha=0.05)
    row_total = row_second_moment.sum(dim=-1, keepdim=True).clamp_min(_TINY)
    variance = (
        row_second_moment[..., :, None]
        * column_second_moment[..., None, :]
        / row_total[..., :, None]
    )
    variance.div_(correction)
    adaptive = m / (variance.sqrt() + 1.0e-8)

    dims = (-2, -1)
    parent2 = p.square().sum(dim=dims)
    adaptive2 = adaptive.square().sum(dim=dims)
    parent_adaptive = (p * adaptive).sum(dim=dims)
    parent_descent = (g * p).sum(dim=dims)
    adaptive_descent = (g * adaptive).sum(dim=dims)
    projection = parent_adaptive / parent2.clamp_min(_TINY)
    tangent2 = (
        adaptive2 - parent_adaptive.square() / parent2.clamp_min(_TINY)
    ).clamp_min(0.0)
    tangent_descent = adaptive_descent - projection * parent_descent
    orientation = torch.where(
        tangent_descent >= 0.0,
        torch.ones_like(tangent_descent),
        -torch.ones_like(tangent_descent),
    )
    information = participation.clamp(0.0, 1.0)
    alignment = congruence.clamp(0.0, 1.0)
    departure_energy = (
        information * (1.0 - alignment.square()).clamp_min(0.0)
    ).clamp(0.0, 1.0)
    parent_weight = torch.sqrt((1.0 - departure_energy).clamp_min(0.0))
    tangent_weight = torch.sqrt(departure_energy.clamp_min(0.0))
    adaptive_coefficient = (
        tangent_weight
        * torch.sqrt(parent2 / tangent2.clamp_min(_TINY))
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
        & (tangent2 > _TINY)
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
        / (selected_norm * parent_norm).clamp_min(1.0e-8)
    ).clamp(-1.0, 1.0)
    budget = (
        (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)
    )
    selected = (
        p * parent_coefficient[..., None, None]
        + adaptive * adaptive_coefficient[..., None, None]
    )
    return (
        selected,
        active,
        safe,
        parent_cosine,
        budget,
        departure_energy,
        parent_descent,
        candidate_descent,
        parent_coefficient,
        adaptive_coefficient,
        orientation,
    )


_COMPILED_ADAPTIVE_TANGENT_CHORD_FULLGRAPH = torch.compile(
    _adaptive_tangent_chord_fullgraph, fullgraph=True, dynamic=False
)


def compiled_factorized_adaptive_tangent_chord_direction(
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
    if not (
        parent.shape == momentum.shape == gradient.shape and parent.ndim == 3
    ):
        raise RuntimeError("compiled adaptive tangent inventory changed")
    if float(beta2) != MATCHED_BETA2 or int(step) < 1 or float(eps) != LOCKED_EPS:
        raise ValueError("compiled adaptive tangent locked numerics changed")
    p, restore = _parent._group_view(
        parent, groups=groups, width=width, grouped_axis=grouped_axis
    )
    m, _ = _parent._group_view(
        momentum, groups=groups, width=width, grouped_axis=grouped_axis
    )
    g, _ = _parent._group_view(
        gradient, groups=groups, width=width, grouped_axis=grouped_axis
    )
    if (
        participation.shape != p.shape[:2]
        or congruence.shape != p.shape[:2]
        or row_second_moment.shape != p.shape[:-1]
        or column_second_moment.shape != p.shape[:2] + p.shape[-1:]
    ):
        raise RuntimeError("compiled adaptive tangent state inventory changed")
    correction = torch.scalar_tensor(
        1.0 - MATCHED_BETA2 ** int(step),
        device=p.device,
        dtype=p.dtype,
    )
    program = (
        _COMPILED_ADAPTIVE_TANGENT_CHORD_FULLGRAPH
        if parent.is_cuda
        else _adaptive_tangent_chord_fullgraph
    )
    (
        selected,
        active,
        safe,
        parent_cosine,
        budget,
        departure_energy,
        parent_descent,
        candidate_descent,
        parent_coefficient,
        adaptive_coefficient,
        orientation,
    ) = program(
        p,
        m,
        g,
        row_second_moment,
        column_second_moment,
        participation.float(),
        congruence.float(),
        correction,
    )
    torch._assert_async(torch.isfinite(selected).all())
    return restore(selected).to(parent.dtype), {
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


def factorized_adaptive_tangent_chord_compiled_scaling_formula(**kwargs):
    result = dict(_parent.factorized_adaptive_tangent_chord_scaling_formula(
        **kwargs
    ))
    result.update({
        "compiled_static_fullgraph": 1,
        "scientific_equation_changed_vs_parent": 0,
        "state_recurrence_changed_vs_parent": 0,
        "additional_persistent_state_elements_vs_parent": 0,
    })
    return result


@contextmanager
def _installed_compiled_direction():
    with _PATCH_LOCK:
        original = _parent.factorized_adaptive_tangent_chord_direction
        if original is not _ORIGINAL_PARENT_DIRECTION:
            raise RuntimeError("compiled adaptive tangent parent was patched")
        _parent.factorized_adaptive_tangent_chord_direction = (
            compiled_factorized_adaptive_tangent_chord_direction
        )
        try:
            yield
        finally:
            _parent.factorized_adaptive_tangent_chord_direction = original


def _retag(report: dict) -> dict:
    result = {}
    for key, value in report.items():
        # The compiled prefix extends the parent prefix. Telemetry persists
        # between capture steps, so retagging must be idempotent rather than
        # prepending another ``compiled_`` at every capture.
        if key.startswith(PREFIX):
            pass
        elif key.startswith(PARENT_PREFIX):
            key = PREFIX + key[len(PARENT_PREFIX):]
        if value == PARENT_FAMILY_ID:
            value = FAMILY_ID
        result[key] = value
    return result


class FactorizedAdaptiveTangentChordCompiledRouter(
    _parent.FactorizedAdaptiveTangentChordRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = "factorized_adaptive_tangent_chord_compiled_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0].pop(PARENT_PREFIX + "family_id", None)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["compiled_static_fullgraph_execution_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        with _installed_compiled_direction():
            loss = super().step(closure)
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry:
            self._last_telemetry.update({
                PREFIX + "family_id": FAMILY_ID,
                PREFIX + "compiled_static_fullgraph": 1,
                PREFIX + "scientific_equation_changed_vs_parent": 0,
                PREFIX + "state_recurrence_changed_vs_parent": 0,
            })
        return loss


class FactorizedAdaptiveTangentChordCompiledAttentionOptimizer(
    _parent.FactorizedAdaptiveTangentChordAttentionOptimizer
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0].pop(PARENT_PREFIX + "family_id", None)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["attention_compiled_static_fullgraph_execution_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        with _installed_compiled_direction():
            loss = super().step(closure)
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry:
            self._last_telemetry.update({
                PREFIX + "attention_family_id": FAMILY_ID,
                PREFIX + "attention_owner_count": 0,
                PREFIX + "attention_selected_update_elements_published": 0,
                PREFIX + "attention_compiled_static_fullgraph": 1,
            })
        return loss


_ORIGINAL_PARENT_DIRECTION = _parent.factorized_adaptive_tangent_chord_direction


__all__ = (
    "FAMILY_ID",
    "PREFIX",
    "FactorizedAdaptiveTangentChordCompiledAttentionOptimizer",
    "FactorizedAdaptiveTangentChordCompiledRouter",
    "compiled_factorized_adaptive_tangent_chord_direction",
    "factorized_adaptive_tangent_chord_compiled_scaling_formula",
)
