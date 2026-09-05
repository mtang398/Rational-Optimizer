"""Four-role post-polar homotopy driven by loss-weighted Global-RLB drift."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_group_muon_core import _batched_zero_power
from .rlb_loss_weighted_four_role_sign_transport_muon import (
    LossWeightedFourRoleSignTransportOptimizer,
    LossWeightedFourRoleSignTransportRouter,
    group_intrinsic_participation,
    group_loss_weighted_intrinsic_statistics,
)
from .rlb_response_alignment_row import _evaluate_response, _jacobian_kernel_inner
from .rlb_response_fisher_muon import _response_adjoint
from .rlb_ten_probe_loss_image_muon import _version_a_factors


FAMILY_ID = "loss_weighted_four_role_response_homotopy_muon_v1"


@dataclass(frozen=True)
class RaggedPostpolarResponseHomotopy:
    local_selected: torch.Tensor
    global_active: torch.Tensor
    global_safe: torch.Tensor
    owner_count: int = 0
    dense_lg_by_lg_metric_elements: int = 0
    selected_update_elements_published: int = 0

    @property
    def communicated_summary_elements(self) -> int:
        return 5 * int(self.global_active.numel())


def arbitrary_shard_postpolar_response_homotopy(
    local_parent: torch.Tensor,
    local_momentum: torch.Tensor,
    local_gradient: torch.Tensor,
    logical_layer_ids: torch.Tensor,
    logical_group_ids: torch.Tensor,
    replica_weights: torch.Tensor,
    participation: torch.Tensor,
    congruence: torch.Tensor,
    *,
    total_layers: int,
    total_groups: int,
    process_group=None,
    eps: float = 1.0e-8,
) -> RaggedPostpolarResponseHomotopy:
    """Execute the two-stage homotopy from arbitrary parameter fragments."""

    if local_parent.ndim != 1:
        raise RuntimeError("ragged response-homotopy source inventory changed")
    count = int(local_parent.numel())
    vectors = (
        local_momentum,
        local_gradient,
        logical_layer_ids,
        logical_group_ids,
        replica_weights,
    )
    if any(value.shape != (count,) for value in vectors):
        raise RuntimeError("ragged response-homotopy metadata changed")
    if logical_layer_ids.dtype != torch.int64 or logical_group_ids.dtype != torch.int64:
        raise TypeError("ragged response-homotopy logical ids must be int64")
    expected = (int(total_layers), int(total_groups))
    if (
        int(total_layers) <= 0
        or int(total_groups) <= 0
        or participation.shape != expected
        or congruence.shape != expected
    ):
        raise RuntimeError("ragged response-homotopy logical inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("ragged response homotopy uses the locked epsilon")
    if count and (
        int(logical_layer_ids.amin()) < 0
        or int(logical_layer_ids.amax()) >= int(total_layers)
        or int(logical_group_ids.amin()) < 0
        or int(logical_group_ids.amax()) >= int(total_groups)
    ):
        raise ValueError("ragged response-homotopy logical id is out of range")
    if not torch.isfinite(replica_weights).all() or (replica_weights < 0.0).any():
        raise ValueError("ragged response-homotopy replica weights are invalid")

    parent = local_parent.float()
    momentum = local_momentum.float()
    gradient = local_gradient.float()
    weight = replica_weights.float()
    logical = logical_layer_ids * int(total_groups) + logical_group_ids
    group_count = int(total_layers) * int(total_groups)
    moments = torch.zeros(group_count, 3, device=parent.device, dtype=torch.float64)
    if count:
        sign = torch.sign(momentum)
        moments.index_add_(
            0,
            logical,
            torch.stack((parent.square(), sign.square(), parent * sign), dim=-1).double()
            * weight.double()[:, None],
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(moments, op=dist.ReduceOp.SUM, group=process_group)
    parent2, sign2, parent_sign = moments.unbind(dim=-1)
    sign_scale = torch.sqrt(parent2 / sign2.clamp_min(float(eps)))
    c = participation.float().clamp(0.0, 1.0).reshape(-1).double()
    root_c = torch.sqrt(c)
    root_one_minus_c = torch.sqrt((1.0 - c).clamp_min(0.0))
    family2 = parent2 + 2.0 * root_c * root_one_minus_c * sign_scale * parent_sign
    family_scale = torch.sqrt(parent2 / family2.clamp_min(float(eps)))
    parent_family = family_scale * (
        root_c * parent2 + root_one_minus_c * sign_scale * parent_sign
    )
    a = congruence.float().clamp(0.0, 1.0).reshape(-1).double()
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source2 = parent2 + 2.0 * a * delta * parent_family
    active = (
        torch.isfinite(moments).all(dim=-1)
        & torch.isfinite(family2)
        & torch.isfinite(source2)
        & (parent2 > 0.0)
        & (sign2 > 0.0)
        & (family2 > 0.0)
        & (source2 > 0.0)
    )

    local_c = c.float().index_select(0, logical)
    local_a = a.float().index_select(0, logical)
    local_delta = delta.float().index_select(0, logical)
    local_sign_scale = sign_scale.float().index_select(0, logical)
    local_family_scale = family_scale.float().index_select(0, logical)
    family = (
        torch.sqrt(local_c) * parent
        + torch.sqrt((1.0 - local_c).clamp_min(0.0))
        * torch.sign(momentum)
        * local_sign_scale
    ) * local_family_scale
    family = torch.where(local_c == 1.0, parent, family)
    local_source_scale = torch.sqrt(
        parent2 / source2.clamp_min(float(eps))
    ).float().index_select(0, logical)
    candidate = (local_a * parent + local_delta * family) * local_source_scale
    candidate = torch.where(local_a == 1.0, parent, candidate)
    candidate = torch.where(active.index_select(0, logical), candidate, parent)

    descent = torch.zeros(group_count, 2, device=parent.device, dtype=torch.float64)
    if count:
        descent.index_add_(
            0,
            logical,
            torch.stack((gradient * parent, gradient * candidate), dim=-1).double()
            * weight.double()[:, None],
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(descent, op=dist.ReduceOp.SUM, group=process_group)
    safe = (
        active
        & torch.isfinite(descent).all(dim=-1)
        & (descent[:, 0] > 0.0)
        & (descent[:, 1] > 0.0)
    )
    selected = torch.where(safe.index_select(0, logical), candidate, parent)
    if not torch.isfinite(selected).all():
        raise RuntimeError("ragged response homotopy is nonfinite")
    return RaggedPostpolarResponseHomotopy(
        selected,
        active.view(expected),
        safe.view(expected),
    )


def combined_group_response_statistics(
    preactivation: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    frozen_numerator: torch.Tensor,
    frozen_denominator: torch.Tensor,
    response_adjoint: torch.Tensor,
    cotangents: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    groups: int,
    width: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Share live factor evaluation across participation and drift kernels."""

    unit, live_d, radial = factors
    expected = (preactivation.shape[0], int(groups), int(width))
    if any(value.shape != expected for value in (unit, live_d, radial, response_adjoint)):
        raise RuntimeError("combined response sensor inventory changed")
    participation = group_loss_weighted_intrinsic_statistics(
        unit, live_d, radial, response_adjoint, cotangents
    )
    live_f = radial + unit * live_d
    frozen_f, frozen_d = _evaluate_response(
        unit, frozen_numerator, frozen_denominator
    )
    value = preactivation.float().view(expected)
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    incoming_weight = response_adjoint.square().mean(dim=-1)
    outgoing_weight = cotangents.float().square().mean(dim=-1)[:, None]
    incoming_cross = (
        _jacobian_kernel_inner(unit, live_f, live_d, frozen_f, frozen_d)
        * incoming_weight
    ).sum(dim=0)
    incoming_live = (
        _jacobian_kernel_inner(unit, live_f, live_d, live_f, live_d)
        * incoming_weight
    ).sum(dim=0)
    incoming_frozen = (
        _jacobian_kernel_inner(unit, frozen_f, frozen_d, frozen_f, frozen_d)
        * incoming_weight
    ).sum(dim=0)
    live_h = rms * live_f
    frozen_h = rms * frozen_f
    outgoing_cross = (
        (live_h * frozen_h).sum(dim=-1).square() * outgoing_weight
    ).sum(dim=0)
    outgoing_live = (
        live_h.square().sum(dim=-1).square() * outgoing_weight
    ).sum(dim=0)
    outgoing_frozen = (
        frozen_h.square().sum(dim=-1).square() * outgoing_weight
    ).sum(dim=0)
    response = torch.stack((
        torch.stack((incoming_cross, incoming_live, incoming_frozen), dim=-1),
        torch.stack((outgoing_cross, outgoing_live, outgoing_frozen), dim=-1),
    ), dim=1)
    torch._assert_async(torch.isfinite(response).all())
    return participation, response


def group_loss_weighted_response_congruence(
    statistics: torch.Tensor,
    exact_initializer: torch.Tensor,
) -> torch.Tensor:
    """Return L-by-G incoming/outgoing live-to-frozen response cosines."""

    if statistics.ndim != 4 or statistics.shape[-2:] != (2, 3):
        raise RuntimeError("group response congruence inventory changed")
    if exact_initializer.shape != statistics.shape[:2]:
        raise RuntimeError("group response initializer inventory changed")
    denominator = torch.sqrt(statistics[..., 1] * statistics[..., 2])
    valid = torch.isfinite(statistics).all(dim=(-2, -1)) & (denominator > 0.0).all(dim=-1)
    result = (
        statistics[..., 0] / denominator.clamp_min(torch.finfo(statistics.dtype).tiny)
    ).clamp(0.0, 1.0)
    result = torch.where(exact_initializer[..., None], torch.ones_like(result), result)
    return torch.where(valid[..., None], result, torch.ones_like(result))


def layer_loss_weighted_response_congruence(
    statistics: torch.Tensor,
    exact_initializer: torch.Tensor,
) -> torch.Tensor:
    """Aggregate group kernels before forming the canonical attention angle."""

    if statistics.ndim != 4 or statistics.shape[-2:] != (2, 3):
        raise RuntimeError("layer response congruence inventory changed")
    layer = statistics.sum(dim=1)
    denominator = torch.sqrt(layer[..., 1] * layer[..., 2])
    valid = torch.isfinite(layer).all(dim=(-2, -1)) & (denominator > 0.0).all(dim=-1)
    role = (
        layer[..., 0] / denominator.clamp_min(torch.finfo(layer.dtype).tiny)
    ).clamp(0.0, 1.0)
    result = torch.sqrt((role[:, 0] * role[:, 1]).clamp_min(0.0))
    result = torch.where(exact_initializer.all(dim=-1), torch.ones_like(result), result)
    return torch.where(valid, result, torch.ones_like(result))


def postpolar_group_response_homotopy(
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
    """Parent -> intrinsic sign family -> frozen-response homotopy."""

    if parent.shape != momentum.shape or parent.shape != gradient.shape or parent.ndim != 3:
        raise RuntimeError("response homotopy tensor inventory changed")
    layers, rows, columns = parent.shape
    expected = (layers, int(groups))
    if participation.shape != expected or congruence.shape != expected:
        raise RuntimeError("response homotopy group inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("response homotopy uses the locked epsilon")
    if grouped_axis == "rows":
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError("response homotopy row inventory changed")
        p = parent.float().view(layers, int(groups), int(width), columns)
        m = momentum.float().view_as(p)
        g = gradient.float().view_as(p)
        restore = lambda value: value.reshape_as(parent)
    elif grouped_axis == "columns":
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError("response homotopy column inventory changed")
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
            raise RuntimeError("response homotopy matrix inventory changed")
        p = parent.float()[:, None]
        m = momentum.float()[:, None]
        g = gradient.float()[:, None]
        restore = lambda value: value[:, 0]
    else:
        raise ValueError(f"unknown response homotopy grouped axis: {grouped_axis}")

    dims = (-2, -1)
    tiny = torch.finfo(p.dtype).tiny
    pnorm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
    sign = torch.sign(m)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = (
        torch.isfinite(p).all(dim=dims, keepdim=True)
        & torch.isfinite(m).all(dim=dims, keepdim=True)
        & (pnorm > 0.0)
        & (sign_norm > 0.0)
    )
    sign_equal = sign * (pnorm / sign_norm.clamp_min(tiny))
    c = participation.float().clamp(0.0, 1.0)[..., None, None]
    family_source = (
        torch.sqrt(c) * p
        + torch.sqrt((1.0 - c).clamp_min(0.0)) * sign_equal
    )
    family_norm = torch.linalg.vector_norm(family_source, dim=dims, keepdim=True)
    family = family_source * (pnorm / family_norm.clamp_min(tiny))
    family = torch.where(c == 1.0, p, family)
    a = congruence.float().clamp(0.0, 1.0)[..., None, None]
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source = a * p + delta * family
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    candidate = source * (pnorm / source_norm.clamp_min(tiny))
    candidate = torch.where(a == 1.0, p, candidate)
    candidate = torch.where(valid, candidate, p)
    parent_descent = (g * p).sum(dim=dims)
    candidate_descent = (g * candidate).sum(dim=dims)
    safe = (
        valid[..., 0, 0]
        & torch.isfinite(parent_descent)
        & (parent_descent > 0.0)
        & torch.isfinite(candidate_descent)
        & (candidate_descent > 0.0)
    )
    selected = torch.where(safe[..., None, None], candidate, p)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    parent_norm = pnorm[..., 0, 0]
    cosine = (
        (p * selected).sum(dim=dims)
        / (parent_norm * selected_norm).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    budget = (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)
    torch._assert_async(torch.isfinite(selected).all())
    return restore(selected), {
        "active": valid[..., 0, 0],
        "safe": safe,
        "parent_cosine": cosine,
        "budget_residual": budget,
        "parent_descent": parent_descent,
        "candidate_descent": candidate_descent,
    }


def method_state_elements(*, layers: int, groups: int) -> int:
    if int(layers) <= 0 or int(groups) <= 0:
        raise ValueError("response-homotopy state dimensions must be positive")
    return 10 * int(layers) * int(groups) + 2


def communicated_summary_elements(*, layers: int, groups: int) -> int:
    if int(layers) <= 0 or int(groups) <= 0:
        raise ValueError("response-homotopy summary dimensions must be positive")
    # 4LG participation + 6LG response kernels + LG initializer, then five
    # chord/descent moments for two RLB roles and two attention roles.
    return 21 * int(layers) * int(groups) + 10 * int(layers)


class LossWeightedFourRoleResponseHomotopyRouter(
    LossWeightedFourRoleSignTransportRouter
):
    """Use response drift to delay the four-role sign family continuously."""

    family_id = FAMILY_ID
    telemetry_prefix = "loss_weighted_four_role_response_homotopy_"
    fairness_component = "loss_weighted_four_role_response_homotopy_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["loss_weighted_four_role_sign_transport_family_id"] = FAMILY_ID
        group["loss_weighted_four_role_response_homotopy_family_id"] = FAMILY_ID
        anchor = self.state[self.pairs[0]["in_weight"]]
        anchor.setdefault(
            "four_role_response_frozen_numerators",
            torch.stack([
                pair["numerator"].detach().float().clone() for pair in self.pairs
            ]),
        )
        anchor.setdefault(
            "four_role_response_frozen_denominators",
            torch.stack([
                pair["denominator"].detach().float().clone() for pair in self.pairs
            ]),
        )
        self._group_response_congruence = None
        self._attention_response_congruence = None

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "rlb_incoming_lr_scale": 1.0,
            "rlb_outgoing_lr_scale": 1.0,
            "fixed32_loss_measure_lr_scale": 1.0,
            "group_intrinsic_participation_lr_scale": 1.0,
            "loss_weighted_frozen_response_lr_scale": 1.0,
            "postpolar_homotopy_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _global_residual_curvature(self, packets):
        anchor = self.state[self.pairs[0]["in_weight"]]
        frozen_numerators = anchor.get("four_role_response_frozen_numerators")
        frozen_denominators = anchor.get("four_role_response_frozen_denominators")
        if (
            frozen_numerators is None
            or frozen_numerators.shape != (len(self.pairs), self.groups, 6)
            or frozen_denominators is None
            or frozen_denominators.shape != (len(self.pairs), self.groups, 4)
        ):
            raise RuntimeError("four-role response frozen inventory changed")
        participation_values = []
        response_values = []
        exact_values = []
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            _inputs, preactivations, _features, cotangents = packet
            factors = _version_a_factors(
                preactivations,
                pair["numerator"],
                pair["denominator"],
                groups=self.groups,
                width=self.width,
                eps=self.rlb_eps,
            )
            response = _response_adjoint(
                cotangents,
                pair["out_weight"],
                factors,
                groups=self.groups,
                width=self.width,
            )
            participation, response_statistics = combined_group_response_statistics(
                preactivations,
                pair["numerator"],
                pair["denominator"],
                frozen_numerators[layer],
                frozen_denominators[layer],
                response,
                cotangents,
                factors,
                groups=self.groups,
                width=self.width,
                eps=self.rlb_eps,
            )
            participation_values.append(participation)
            response_values.append(response_statistics)
            exact_values.append(
                torch.all(
                    pair["numerator"].detach().float() == frozen_numerators[layer],
                    dim=-1,
                )
                & torch.all(
                    pair["denominator"].detach().float() == frozen_denominators[layer],
                    dim=-1,
                )
            )
        participation_statistics = torch.stack(participation_values)
        response_statistics = torch.stack(response_values)
        first_elements = participation_statistics.numel()
        packed = torch.cat((
            participation_statistics.reshape(-1), response_statistics.reshape(-1)
        ))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        participation_statistics = packed[:first_elements].view_as(
            participation_statistics
        )
        response_statistics = packed[first_elements:].view_as(response_statistics)
        exact = torch.stack(exact_values).to(torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(exact, op=dist.ReduceOp.MIN, group=self.loss_probe_group)
        participation = group_intrinsic_participation(participation_statistics)
        group_congruence = group_loss_weighted_response_congruence(
            response_statistics, exact.bool()
        )
        attention_congruence = layer_loss_weighted_response_congruence(
            response_statistics, exact.bool()
        )
        incoming = participation[..., 0].mean(dim=-1)
        outgoing = participation[..., 1].mean(dim=-1)
        self._group_participation = participation
        self._group_response_congruence = group_congruence
        self._intrinsic_participation = torch.sqrt(
            (incoming * outgoing).clamp(0.0, 1.0)
        )
        self._attention_response_congruence = attention_congruence
        return participation

    def consume_response_homotopy(self):
        if self._attention_consumed:
            raise RuntimeError("four-role response homotopy was consumed twice")
        if (
            self._intrinsic_participation is None
            or self._attention_response_congruence is None
        ):
            raise RuntimeError("four-role response attention route is incomplete")
        self._attention_consumed = True
        return (
            self._intrinsic_participation,
            self._attention_response_congruence,
            int(self._attention_update),
        )

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("four-role response router lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("four-role response router would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("four-role response router refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        packets = self._consume_probes()
        participation = self._global_residual_curvature(packets)
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError("four-role response router omitted group congruence")

        role_records = []
        for role, index, grouped_axis in (
            ("incoming", 0, "rows"),
            ("outgoing", 1, "columns"),
        ):
            key = "in_weight" if role == "incoming" else "out_weight"
            parameters = [pair[key] for pair in self.pairs]
            momenta = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = postpolar_group_response_homotopy(
                parent,
                momenta,
                gradients,
                participation[..., index],
                congruence[..., index],
                groups=self.groups,
                width=self.width,
                grouped_axis=grouped_axis,
            )
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(
                    selected[layer].to(parameter.dtype), alpha=-lr * adjustment
                )
            role_records.append(metadata)

        anchor = self.state[self.pairs[0]["in_weight"]]
        updates = int(anchor.get("four_role_response_homotopy_updates", 0)) + 1
        anchor["four_role_response_homotopy_updates"] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            angles = congruence.reshape(-1)
            safe = torch.cat([record["safe"].reshape(-1) for record in role_records])
            cosine = torch.cat([
                record["parent_cosine"].reshape(-1) for record in role_records
            ])
            budget = torch.cat([
                record["budget_residual"].reshape(-1) for record in role_records
            ])
            self._last_telemetry = {
                "loss_weighted_four_role_response_homotopy_family_id": FAMILY_ID,
                "loss_weighted_four_role_response_homotopy_owner_count": 0,
                "loss_weighted_four_role_response_homotopy_dense_lg_metric_elements": 0,
                "loss_weighted_four_role_response_homotopy_selected_update_elements_published": 0,
                "loss_weighted_four_role_response_homotopy_state_depends_on_total_tokens": 0,
                "loss_weighted_four_role_response_homotopy_state_coordinate_count": method_state_elements(
                    layers=len(self.pairs), groups=self.groups
                ),
                "loss_weighted_four_role_response_homotopy_summary_elements": communicated_summary_elements(
                    layers=len(self.pairs), groups=self.groups
                ),
                "loss_weighted_four_role_response_homotopy_updates": updates,
                "loss_weighted_four_role_response_homotopy_participation_min": float(values.amin().item()),
                "loss_weighted_four_role_response_homotopy_participation_median": float(values.median().item()),
                "loss_weighted_four_role_response_homotopy_participation_max": float(values.amax().item()),
                "loss_weighted_four_role_response_homotopy_congruence_min": float(angles.amin().item()),
                "loss_weighted_four_role_response_homotopy_congruence_median": float(angles.median().item()),
                "loss_weighted_four_role_response_homotopy_congruence_max": float(angles.amax().item()),
                "loss_weighted_four_role_response_homotopy_rlb_safe_fraction": float(safe.float().mean().item()),
                "loss_weighted_four_role_response_homotopy_rlb_parent_cosine_min": float(cosine.amin().item()),
                "loss_weighted_four_role_response_homotopy_rlb_parent_cosine_median": float(cosine.median().item()),
                "loss_weighted_four_role_response_homotopy_rlb_parent_cosine_max": float(cosine.amax().item()),
                "loss_weighted_four_role_response_homotopy_rlb_budget_residual_max": float(budget.amax().item()),
                "loss_weighted_four_role_response_homotopy_realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._group_response_congruence = None
        self._attention_response_congruence = None
        return result


class LossWeightedFourRoleResponseHomotopyOptimizer(
    LossWeightedFourRoleSignTransportOptimizer
):
    """Apply the same frozen-response homotopy to both attention roles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0][
            "loss_weighted_four_role_response_homotopy_family_id"
        ] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "loss_weighted_frozen_response_lr_scale": 1.0,
            "postpolar_homotopy_lr_scale": 1.0,
            "single_ns5_per_attention_role_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("four-role response attention refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        participation, congruence, router_update = self.router.consume_response_homotopy()
        anchor = self.state[self.role_parameters["qkv"][0]]
        previous = int(anchor.get("four_role_response_homotopy_router_update", 0))
        if int(router_update) != previous + 1:
            raise RuntimeError("four-role response attention missed a router update")
        anchor["four_role_response_homotopy_router_update"] = int(router_update)

        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = postpolar_group_response_homotopy(
                parent,
                momenta,
                gradients,
                participation[:, None],
                congruence[:, None],
                groups=1,
                width=None,
                grouped_axis="matrix",
            )
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(
                    selected[layer].to(parameter.dtype), alpha=-lr * adjustment
                )
            records.append(metadata)
        if self._capture_telemetry_next_step:
            safe = torch.cat([record["safe"].reshape(-1) for record in records])
            cosine = torch.cat([
                record["parent_cosine"].reshape(-1) for record in records
            ])
            budget = torch.cat([
                record["budget_residual"].reshape(-1) for record in records
            ])
            self._last_telemetry = {
                "loss_weighted_four_role_response_direction_family_id": FAMILY_ID,
                "loss_weighted_four_role_response_direction_owner_count": 0,
                "loss_weighted_four_role_response_direction_dense_lg_metric_elements": 0,
                "loss_weighted_four_role_response_direction_selected_update_elements_published": 0,
                "loss_weighted_four_role_response_direction_state_depends_on_total_tokens": 0,
                "loss_weighted_four_role_response_direction_router_update": int(router_update),
                "loss_weighted_four_role_response_direction_safe_fraction": float(safe.float().mean().item()),
                "loss_weighted_four_role_response_direction_parent_cosine_min": float(cosine.amin().item()),
                "loss_weighted_four_role_response_direction_parent_cosine_median": float(cosine.median().item()),
                "loss_weighted_four_role_response_direction_parent_cosine_max": float(cosine.amax().item()),
                "loss_weighted_four_role_response_direction_budget_residual_max": float(budget.amax().item()),
            }
        self._capture_telemetry_next_step = False
        return loss


__all__ = (
    "FAMILY_ID",
    "LossWeightedFourRoleResponseHomotopyOptimizer",
    "LossWeightedFourRoleResponseHomotopyRouter",
    "RaggedPostpolarResponseHomotopy",
    "arbitrary_shard_postpolar_response_homotopy",
    "combined_group_response_statistics",
    "communicated_summary_elements",
    "group_loss_weighted_response_congruence",
    "layer_loss_weighted_response_congruence",
    "method_state_elements",
    "postpolar_group_response_homotopy",
)
