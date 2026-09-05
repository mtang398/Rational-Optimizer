"""Moment-exact compact realization of four-role response homotopy Muon."""

from __future__ import annotations

import torch

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_group_muon_core import _batched_zero_power
from .rlb_loss_weighted_four_role_response_homotopy_batched_muon import (
    BatchedFourRoleResponseHomotopyAttentionOptimizer,
    BatchedFourRoleResponseHomotopyRouter,
    _foreach_apply,
    _foreach_nesterov,
)
from .rlb_loss_weighted_four_role_response_homotopy_muon import (
    communicated_summary_elements,
    method_state_elements,
)


FAMILY_ID = "loss_weighted_four_role_response_homotopy_compact_muon_v3"


def compact_postpolar_group_response_homotopy(
    parent: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    participation: torch.Tensor,
    congruence: torch.Tensor,
    *,
    groups: int,
    width: int | None,
    grouped_axis: str,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the exact homotopy through two scalar coefficients per group.

    Both nested normalized chords lie in ``span(parent, sign(momentum))``.
    Five group moments therefore recover the selected direction without
    materializing the intermediate sign-equalized, family, and outer-source
    tensors.  The returned tensor is still local and parameter-shaped; no
    selected update is communicated.
    """

    if parent.shape != momentum.shape or parent.shape != gradient.shape or parent.ndim != 3:
        raise RuntimeError("compact response homotopy tensor inventory changed")
    layers, rows, columns = parent.shape
    expected = (layers, int(groups))
    if participation.shape != expected or congruence.shape != expected:
        raise RuntimeError("compact response homotopy group inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("compact response homotopy uses the locked epsilon")

    if grouped_axis == "rows":
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError("compact response homotopy row inventory changed")
        p = parent.float().view(layers, int(groups), int(width), columns)
        m = momentum.float().view_as(p)
        g = gradient.float().view_as(p)
        restore = lambda value: value.reshape_as(parent)
    elif grouped_axis == "columns":
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError("compact response homotopy column inventory changed")
        p = parent.float().transpose(-2, -1).contiguous().view(
            layers, int(groups), int(width), rows
        )
        m = momentum.float().transpose(-2, -1).contiguous().view_as(p)
        g = gradient.float().transpose(-2, -1).contiguous().view_as(p)
        restore = lambda value: value.reshape(
            layers, columns, rows
        ).transpose(-2, -1).contiguous()
    elif grouped_axis == "matrix":
        if int(groups) != 1 or width is not None:
            raise RuntimeError("compact response homotopy matrix inventory changed")
        p = parent.float()[:, None]
        m = momentum.float()[:, None]
        g = gradient.float()[:, None]
        restore = lambda value: value[:, 0]
    else:
        raise ValueError(f"unknown compact response homotopy axis: {grouped_axis}")

    dims = (-2, -1)
    tiny = torch.finfo(p.dtype).tiny
    sign = torch.sign(m)
    parent2 = p.square().sum(dim=dims)
    sign2 = sign.square().sum(dim=dims)
    parent_sign = (p * sign).sum(dim=dims)
    parent_descent = (g * p).sum(dim=dims)
    sign_descent = (g * sign).sum(dim=dims)
    valid = (
        torch.isfinite(p).all(dim=dims)
        & torch.isfinite(m).all(dim=dims)
        & torch.isfinite(parent2)
        & torch.isfinite(sign2)
        & torch.isfinite(parent_sign)
        & (parent2 > 0.0)
        & (sign2 > 0.0)
    )

    sign_scale = torch.sqrt(parent2 / sign2.clamp_min(tiny))
    c = participation.float().clamp(0.0, 1.0)
    root_c = torch.sqrt(c)
    root_one_minus_c = torch.sqrt((1.0 - c).clamp_min(0.0))
    family2 = (
        parent2
        + 2.0 * root_c * root_one_minus_c * sign_scale * parent_sign
    )
    family_scale = torch.sqrt(parent2 / family2.clamp_min(tiny))
    family_parent = family_scale * root_c
    family_sign = family_scale * root_one_minus_c * sign_scale

    a = congruence.float().clamp(0.0, 1.0)
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source_parent = a + delta * family_parent
    source_sign = delta * family_sign
    source2 = (
        source_parent.square() * parent2
        + source_sign.square() * sign2
        + 2.0 * source_parent * source_sign * parent_sign
    )
    source_scale = torch.sqrt(parent2 / source2.clamp_min(tiny))
    parent_coefficient = source_scale * source_parent
    sign_coefficient = source_scale * source_sign
    candidate_descent = (
        parent_coefficient * parent_descent
        + sign_coefficient * sign_descent
    )
    valid = (
        valid
        & torch.isfinite(family2)
        & torch.isfinite(source2)
        & (family2 > 0.0)
        & (source2 > 0.0)
    )
    safe = (
        valid
        & torch.isfinite(parent_descent)
        & (parent_descent > 0.0)
        & torch.isfinite(candidate_descent)
        & (candidate_descent > 0.0)
    )
    parent_coefficient = torch.where(
        safe, parent_coefficient, torch.ones_like(parent_coefficient)
    )
    sign_coefficient = torch.where(
        safe, sign_coefficient, torch.zeros_like(sign_coefficient)
    )
    selected2 = (
        parent_coefficient.square() * parent2
        + sign_coefficient.square() * sign2
        + 2.0 * parent_coefficient * sign_coefficient * parent_sign
    )
    selected_norm = torch.sqrt(selected2.clamp_min(0.0))
    parent_norm = torch.sqrt(parent2.clamp_min(0.0))
    cosine = (
        (parent_coefficient * parent2 + sign_coefficient * parent_sign)
        / (parent_norm * selected_norm).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    budget = (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)

    coefficient_shape = (*parent_coefficient.shape, 1, 1)
    p.mul_(parent_coefficient.view(coefficient_shape))
    p.addcmul_(sign, sign_coefficient.view(coefficient_shape))
    selected = restore(p)
    torch._assert_async(torch.isfinite(selected).all())
    return selected, {
        "active": valid,
        "safe": safe,
        "parent_cosine": cosine,
        "budget_residual": budget,
        "parent_descent": parent_descent,
        "candidate_descent": candidate_descent,
        "parent_coefficient": parent_coefficient,
        "sign_coefficient": sign_coefficient,
    }


class CompactFourRoleResponseHomotopyRouter(
    BatchedFourRoleResponseHomotopyRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = "loss_weighted_four_role_response_homotopy_compact_"
    fairness_component = "loss_weighted_four_role_response_homotopy_compact_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["loss_weighted_four_role_response_homotopy_compact_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["moment_exact_compact_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("compact response router lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("compact response router would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("compact response router refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        participation = self._global_residual_curvature(self._consume_probes())
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError("compact response router omitted congruence")
        role_records = []
        for role, index, axis in (
            ("incoming", 0, "rows"), ("outgoing", 1, "columns")
        ):
            key = "in_weight" if role == "incoming" else "out_weight"
            parameters = [pair[key] for pair in self.pairs]
            momenta = _foreach_nesterov(self, parameters)
            gradients = torch.stack([
                parameter.grad.detach() for parameter in parameters
            ]).float()
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = compact_postpolar_group_response_homotopy(
                parent, momenta, gradients,
                participation[..., index], congruence[..., index],
                groups=self.groups, width=self.width, grouped_axis=axis,
            )
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            _foreach_apply(
                parameters, selected,
                decay=1.0 - lr * weight_decay,
                alpha=-lr * adjustment,
            )
            role_records.append(metadata)
        anchor = self.state[self.pairs[0]["in_weight"]]
        updates = int(anchor.get("compact_response_homotopy_updates", 0)) + 1
        anchor["compact_response_homotopy_updates"] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            angles = congruence.reshape(-1)
            safe = torch.cat([record["safe"].reshape(-1) for record in role_records])
            cosine = torch.cat([record["parent_cosine"].reshape(-1) for record in role_records])
            budget = torch.cat([record["budget_residual"].reshape(-1) for record in role_records])
            prefix = "loss_weighted_four_role_response_homotopy_compact_"
            self._last_telemetry = {
                prefix + "family_id": FAMILY_ID,
                prefix + "owner_count": 0,
                prefix + "dense_lg_metric_elements": 0,
                prefix + "selected_update_elements_published": 0,
                prefix + "state_depends_on_total_tokens": 0,
                prefix + "state_coordinate_count": method_state_elements(
                    layers=len(self.pairs), groups=self.groups
                ),
                prefix + "summary_elements": communicated_summary_elements(
                    layers=len(self.pairs), groups=self.groups
                ),
                prefix + "updates": updates,
                prefix + "participation_min": float(values.amin().item()),
                prefix + "participation_median": float(values.median().item()),
                prefix + "participation_max": float(values.amax().item()),
                prefix + "congruence_min": float(angles.amin().item()),
                prefix + "congruence_median": float(angles.median().item()),
                prefix + "congruence_max": float(angles.amax().item()),
                prefix + "rlb_safe_fraction": float(safe.float().mean().item()),
                prefix + "rlb_parent_cosine_min": float(cosine.amin().item()),
                prefix + "rlb_parent_cosine_median": float(cosine.median().item()),
                prefix + "rlb_parent_cosine_max": float(cosine.amax().item()),
                prefix + "rlb_budget_residual_max": float(budget.amax().item()),
                prefix + "realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss


class CompactFourRoleResponseHomotopyAttentionOptimizer(
    BatchedFourRoleResponseHomotopyAttentionOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]["loss_weighted_four_role_response_homotopy_compact_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["moment_exact_compact_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("compact response attention refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        participation, congruence, router_update = self.router.consume_response_homotopy()
        anchor = self.state[self.role_parameters["qkv"][0]]
        previous = int(anchor.get("compact_response_homotopy_router_update", 0))
        if int(router_update) != previous + 1:
            raise RuntimeError("compact response attention missed a router update")
        anchor["compact_response_homotopy_router_update"] = int(router_update)
        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = _foreach_nesterov(self, parameters)
            gradients = torch.stack([
                parameter.grad.detach() for parameter in parameters
            ]).float()
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = compact_postpolar_group_response_homotopy(
                parent, momenta, gradients,
                participation[:, None], congruence[:, None],
                groups=1, width=None, grouped_axis="matrix",
            )
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            _foreach_apply(
                parameters, selected,
                decay=1.0 - lr * weight_decay,
                alpha=-lr * adjustment,
            )
            records.append(metadata)
        if self._capture_telemetry_next_step:
            safe = torch.cat([record["safe"].reshape(-1) for record in records])
            cosine = torch.cat([record["parent_cosine"].reshape(-1) for record in records])
            budget = torch.cat([record["budget_residual"].reshape(-1) for record in records])
            prefix = "loss_weighted_four_role_response_homotopy_compact_direction_"
            self._last_telemetry = {
                prefix + "family_id": FAMILY_ID,
                prefix + "owner_count": 0,
                prefix + "dense_lg_metric_elements": 0,
                prefix + "selected_update_elements_published": 0,
                prefix + "state_depends_on_total_tokens": 0,
                prefix + "router_update": int(router_update),
                prefix + "safe_fraction": float(safe.float().mean().item()),
                prefix + "parent_cosine_min": float(cosine.amin().item()),
                prefix + "parent_cosine_median": float(cosine.median().item()),
                prefix + "parent_cosine_max": float(cosine.amax().item()),
                prefix + "budget_residual_max": float(budget.amax().item()),
            }
        self._capture_telemetry_next_step = False
        return loss


__all__ = (
    "CompactFourRoleResponseHomotopyAttentionOptimizer",
    "CompactFourRoleResponseHomotopyRouter",
    "FAMILY_ID",
    "compact_postpolar_group_response_homotopy",
)
