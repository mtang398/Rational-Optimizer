"""Loss-weighted Global-RLB post-polar sign transport across four matrix roles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_group_muon_core import _batched_zero_power
from .rlb_loss_weighted_intrinsic_sign_attention_muon import (
    LossWeightedIntrinsicSignAttentionOptimizer,
    LossWeightedIntrinsicSignAttentionRouter,
)
from .rlb_response_alignment_row import _jacobian_kernel_inner
from .rlb_response_fisher_muon import _response_adjoint
from .rlb_ten_probe_loss_image_muon import _version_a_factors


FAMILY_ID = "loss_weighted_four_role_sign_transport_muon_v1"


@dataclass(frozen=True)
class RaggedPostpolarSignTransport:
    local_selected: torch.Tensor
    global_active: torch.Tensor
    global_safe: torch.Tensor
    owner_count: int = 0
    dense_lg_by_lg_metric_elements: int = 0
    selected_update_elements_published: int = 0

    @property
    def communicated_summary_elements(self) -> int:
        return 5 * int(self.global_active.numel())


def arbitrary_shard_postpolar_sign_transport(
    local_parent: torch.Tensor,
    local_momentum: torch.Tensor,
    local_gradient: torch.Tensor,
    logical_layer_ids: torch.Tensor,
    logical_group_ids: torch.Tensor,
    replica_weights: torch.Tensor,
    participation: torch.Tensor,
    *,
    total_layers: int,
    total_groups: int,
    process_group=None,
    eps: float = 1.0e-8,
) -> RaggedPostpolarSignTransport:
    """Apply the post-polar group transport to arbitrary parameter fragments."""

    if local_parent.ndim != 1:
        raise RuntimeError("ragged post-polar source inventory changed")
    count = int(local_parent.numel())
    vectors = (
        local_momentum,
        local_gradient,
        logical_layer_ids,
        logical_group_ids,
        replica_weights,
    )
    if any(value.shape != (count,) for value in vectors):
        raise RuntimeError("ragged post-polar metadata changed")
    if logical_layer_ids.dtype != torch.int64 or logical_group_ids.dtype != torch.int64:
        raise TypeError("ragged post-polar logical ids must be int64")
    if (
        int(total_layers) <= 0
        or int(total_groups) <= 0
        or participation.shape != (int(total_layers), int(total_groups))
    ):
        raise RuntimeError("ragged post-polar logical inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("ragged post-polar transport uses the locked epsilon")
    if count and (
        int(logical_layer_ids.amin()) < 0
        or int(logical_layer_ids.amax()) >= int(total_layers)
        or int(logical_group_ids.amin()) < 0
        or int(logical_group_ids.amax()) >= int(total_groups)
    ):
        raise ValueError("ragged post-polar logical id is out of range")
    if not torch.isfinite(replica_weights).all() or (replica_weights < 0.0).any():
        raise ValueError("ragged post-polar replica weights are invalid")

    parent = local_parent.float()
    momentum = local_momentum.float()
    gradient = local_gradient.float()
    weight = replica_weights.float()
    logical = logical_layer_ids * int(total_groups) + logical_group_ids
    groups = int(total_layers) * int(total_groups)
    moments = torch.zeros(groups, 3, device=parent.device, dtype=torch.float64)
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
    parent2, sign2, cross = moments.unbind(dim=-1)
    sign_scale = torch.sqrt(parent2 / sign2.clamp_min(float(eps)))
    c = participation.float().clamp(0.0, 1.0).reshape(-1).double()
    parent_amplitude = torch.sqrt(c)
    sign_amplitude = torch.sqrt((1.0 - c).clamp_min(0.0))
    source2 = (
        parent2
        + 2.0 * parent_amplitude * sign_amplitude * sign_scale * cross
    )
    active = (
        torch.isfinite(moments).all(dim=-1)
        & torch.isfinite(source2)
        & (parent2 > 0.0)
        & (sign2 > 0.0)
        & (source2 > 0.0)
    )
    local_c = c.float().index_select(0, logical)
    local_parent_amplitude = torch.sqrt(local_c)
    local_sign_amplitude = torch.sqrt((1.0 - local_c).clamp_min(0.0))
    local_sign_scale = sign_scale.float().index_select(0, logical)
    local_budget_scale = torch.sqrt(
        parent2 / source2.clamp_min(float(eps))
    ).float().index_select(0, logical)
    candidate = (
        local_parent_amplitude * parent
        + local_sign_amplitude * torch.sign(momentum) * local_sign_scale
    ) * local_budget_scale
    candidate = torch.where(local_c == 1.0, parent, candidate)
    candidate = torch.where(active.index_select(0, logical), candidate, parent)

    descent = torch.zeros(groups, 2, device=parent.device, dtype=torch.float64)
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
        raise RuntimeError("ragged post-polar transport is nonfinite")
    return RaggedPostpolarSignTransport(
        selected,
        active.view(int(total_layers), int(total_groups)),
        safe.view(int(total_layers), int(total_groups)),
    )


def group_loss_weighted_intrinsic_statistics(
    unit: torch.Tensor,
    derivative: torch.Tensor,
    radial: torch.Tensor,
    response_adjoint: torch.Tensor,
    cotangents: torch.Tensor,
) -> torch.Tensor:
    """Return four additive participation scalars for each rational group."""

    if not (unit.shape == derivative.shape == radial.shape == response_adjoint.shape):
        raise RuntimeError("four-role intrinsic rational inventory changed")
    if unit.ndim != 3 or cotangents.ndim != 2 or cotangents.shape[0] != unit.shape[0]:
        raise RuntimeError("four-role intrinsic probe inventory changed")
    width = float(unit.shape[-1])
    function = radial + unit * derivative
    radial_jacobian = radial / width
    trace = derivative.square().sum(dim=-1)
    trace = trace + 2.0 * (derivative * unit * radial_jacobian).sum(dim=-1)
    trace = trace + unit.square().sum(dim=-1) * radial_jacobian.square().sum(dim=-1)
    trace_square = _jacobian_kernel_inner(
        unit, function, derivative, function, derivative
    )
    tiny = torch.finfo(trace.dtype).tiny
    incoming = (
        trace.square() / (width * trace_square.clamp_min(tiny))
    ).clamp(0.0, 1.0)
    energy = function.square().sum(dim=-1)
    fourth = function.pow(4).sum(dim=-1)
    outgoing = (
        energy.square() / (width * fourth.clamp_min(tiny))
    ).clamp(0.0, 1.0)
    incoming_weight = response_adjoint.square().mean(dim=-1)
    outgoing_weight = cotangents.float().square().mean(dim=-1)[:, None]
    result = torch.stack((
        (incoming * incoming_weight).sum(dim=0),
        incoming_weight.sum(dim=0),
        (outgoing * outgoing_weight).sum(dim=0),
        outgoing_weight.expand_as(outgoing).sum(dim=0),
    ), dim=-1)
    torch._assert_async(torch.isfinite(result).all())
    return result


def group_intrinsic_participation(statistics: torch.Tensor) -> torch.Tensor:
    """Map L-by-G additive statistics to incoming/outgoing participation."""

    if statistics.ndim != 3 or statistics.shape[-1] != 4:
        raise RuntimeError("four-role participation inventory changed")
    incoming = statistics[..., 0] / statistics[..., 1].clamp_min(
        torch.finfo(statistics.dtype).tiny
    )
    outgoing = statistics[..., 2] / statistics[..., 3].clamp_min(
        torch.finfo(statistics.dtype).tiny
    )
    valid = (
        torch.isfinite(statistics).all(dim=-1)
        & (statistics[..., 1] > 0.0)
        & (statistics[..., 3] > 0.0)
    )
    value = torch.stack((incoming, outgoing), dim=-1).clamp(0.0, 1.0)
    return torch.where(valid[..., None], value, torch.ones_like(value))


def postpolar_group_sign_transport(
    parent: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    participation: torch.Tensor,
    *,
    groups: int,
    width: int | None,
    grouped_axis: str,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Transport a polar direction toward sign geometry at exact group budget."""

    if parent.shape != momentum.shape or parent.shape != gradient.shape or parent.ndim != 3:
        raise RuntimeError("post-polar transport tensor inventory changed")
    layers, rows, columns = parent.shape
    if participation.shape != (layers, int(groups)):
        raise RuntimeError("post-polar participation inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("post-polar transport uses the locked epsilon")
    if grouped_axis == "rows":
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError("post-polar row group inventory changed")
        p = parent.float().view(layers, int(groups), int(width), columns)
        m = momentum.float().view_as(p)
        g = gradient.float().view_as(p)
        restore = lambda value: value.reshape_as(parent)
    elif grouped_axis == "columns":
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError("post-polar column group inventory changed")
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
            raise RuntimeError("post-polar matrix group inventory changed")
        p = parent.float()[:, None]
        m = momentum.float()[:, None]
        g = gradient.float()[:, None]
        restore = lambda value: value[:, 0]
    else:
        raise ValueError(f"unknown post-polar grouped axis: {grouped_axis}")

    dims = (-2, -1)
    tiny = torch.finfo(p.dtype).tiny
    parent_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
    sign = torch.sign(m)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = (
        torch.isfinite(p).all(dim=dims, keepdim=True)
        & torch.isfinite(m).all(dim=dims, keepdim=True)
        & (parent_norm > 0.0)
        & (sign_norm > 0.0)
    )
    sign_equal = sign * (parent_norm / sign_norm.clamp_min(tiny))
    c = participation.float().clamp(0.0, 1.0)[..., None, None]
    source = torch.sqrt(c) * p + torch.sqrt((1.0 - c).clamp_min(0.0)) * sign_equal
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    candidate = source * (parent_norm / source_norm.clamp_min(tiny))
    candidate = torch.where(c == 1.0, p, candidate)
    candidate = torch.where(valid, candidate, p)
    parent_descent = (g * p).sum(dim=dims)
    candidate_descent = (g * candidate).sum(dim=dims)
    safe = (
        valid.flatten(2).all(dim=-1)
        & torch.isfinite(parent_descent)
        & (parent_descent > 0.0)
        & torch.isfinite(candidate_descent)
        & (candidate_descent > 0.0)
    )
    selected = torch.where(safe[..., None, None], candidate, p)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    pnorm = parent_norm[..., 0, 0]
    cosine = (
        (p * selected).sum(dim=dims)
        / (pnorm * selected_norm).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    budget = (selected_norm - pnorm).abs() / pnorm.clamp_min(1.0)
    torch._assert_async(torch.isfinite(selected).all())
    return restore(selected), {
        "active": valid[..., 0, 0],
        "safe": safe,
        "parent_cosine": cosine,
        "budget_residual": budget,
        "parent_descent": parent_descent,
        "candidate_descent": candidate_descent,
    }


def method_state_elements() -> int:
    # Router and attention transaction counters only; momentum is matched Muon state.
    return 2


def communicated_summary_elements(*, layers: int, groups: int) -> int:
    if int(layers) <= 0 or int(groups) <= 0:
        raise ValueError("four-role summary dimensions must be positive")
    # 4LG response statistics, 5LG for each RLB role, and 5L per attention role.
    return 14 * int(layers) * int(groups) + 10 * int(layers)


class LossWeightedFourRoleSignTransportRouter(LossWeightedIntrinsicSignAttentionRouter):
    """Apply Global-RLB group participation after the RLB polar map."""

    family_id = FAMILY_ID
    telemetry_prefix = "loss_weighted_four_role_sign_transport_"
    fairness_component = "loss_weighted_four_role_sign_transport_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["ten_probe_family_id"] = FAMILY_ID
        group["fixed32_transaction_family_id"] = FAMILY_ID
        group["response_fisher_family_id"] = FAMILY_ID
        group["residual_fisher_attention_family_id"] = FAMILY_ID
        group["loss_weighted_intrinsic_sign_attention_family_id"] = FAMILY_ID
        group["loss_weighted_four_role_sign_transport_family_id"] = FAMILY_ID
        self._group_participation = None

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "rlb_incoming_lr_scale": 1.0,
            "rlb_outgoing_lr_scale": 1.0,
            "fixed32_loss_measure_lr_scale": 1.0,
            "group_intrinsic_participation_lr_scale": 1.0,
            "postpolar_sign_transport_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _global_residual_curvature(self, packets):
        values = []
        for pair, packet in zip(self.pairs, packets):
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
            values.append(group_loss_weighted_intrinsic_statistics(
                *factors, response, cotangents
            ))
        statistics = torch.stack(values)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        participation = group_intrinsic_participation(statistics)
        self._group_participation = participation
        incoming = participation[..., 0].mean(dim=-1)
        outgoing = participation[..., 1].mean(dim=-1)
        self._intrinsic_participation = torch.sqrt(
            (incoming * outgoing).clamp(0.0, 1.0)
        )
        return participation

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("four-role router lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("four-role router would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("four-role router refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        packets = self._consume_probes()
        participation = self._global_residual_curvature(packets)

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
            selected, metadata = postpolar_group_sign_transport(
                parent,
                momenta,
                gradients,
                participation[..., index],
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
        updates = int(anchor.get("four_role_sign_transport_updates", 0)) + 1
        anchor["four_role_sign_transport_updates"] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._intrinsic_participation is None:
            raise RuntimeError("four-role router omitted attention participation")
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            safe = torch.cat([record["safe"].reshape(-1) for record in role_records])
            cosine = torch.cat([
                record["parent_cosine"].reshape(-1) for record in role_records
            ])
            budget = torch.cat([
                record["budget_residual"].reshape(-1) for record in role_records
            ])
            self._last_telemetry = {
                "loss_weighted_four_role_sign_transport_family_id": FAMILY_ID,
                "loss_weighted_four_role_sign_transport_owner_count": 0,
                "loss_weighted_four_role_sign_transport_dense_lg_metric_elements": 0,
                "loss_weighted_four_role_sign_transport_selected_update_elements_published": 0,
                "loss_weighted_four_role_sign_transport_state_depends_on_total_tokens": 0,
                "loss_weighted_four_role_sign_transport_state_coordinate_count": method_state_elements(),
                "loss_weighted_four_role_sign_transport_summary_elements": communicated_summary_elements(
                    layers=len(self.pairs), groups=self.groups
                ),
                "loss_weighted_four_role_sign_transport_updates": updates,
                "loss_weighted_four_role_sign_transport_participation_min": float(values.amin().item()),
                "loss_weighted_four_role_sign_transport_participation_median": float(values.median().item()),
                "loss_weighted_four_role_sign_transport_participation_max": float(values.amax().item()),
                "loss_weighted_four_role_sign_transport_rlb_safe_fraction": float(safe.float().mean().item()),
                "loss_weighted_four_role_sign_transport_rlb_parent_cosine_min": float(cosine.amin().item()),
                "loss_weighted_four_role_sign_transport_rlb_parent_cosine_median": float(cosine.median().item()),
                "loss_weighted_four_role_sign_transport_rlb_parent_cosine_max": float(cosine.amax().item()),
                "loss_weighted_four_role_sign_transport_rlb_budget_residual_max": float(budget.amax().item()),
                "loss_weighted_four_role_sign_transport_realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._group_participation = None
        return result


class LossWeightedFourRoleSignTransportOptimizer(LossWeightedIntrinsicSignAttentionOptimizer):
    """Apply the same post-polar transport to both attention matrix roles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]["loss_weighted_four_role_sign_transport_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "postpolar_sign_transport_lr_scale": 1.0,
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
            raise RuntimeError("four-role attention refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        participation, router_update = self.router.consume_intrinsic_participation()
        anchor = self.state[self.role_parameters["qkv"][0]]
        previous = int(anchor.get("four_role_sign_transport_router_update", 0))
        if int(router_update) != previous + 1:
            raise RuntimeError("four-role attention missed a router update")
        anchor["four_role_sign_transport_router_update"] = int(router_update)

        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = postpolar_group_sign_transport(
                parent,
                momenta,
                gradients,
                participation[:, None],
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
                "loss_weighted_four_role_sign_direction_family_id": FAMILY_ID,
                "loss_weighted_four_role_sign_direction_owner_count": 0,
                "loss_weighted_four_role_sign_direction_dense_lg_metric_elements": 0,
                "loss_weighted_four_role_sign_direction_selected_update_elements_published": 0,
                "loss_weighted_four_role_sign_direction_state_depends_on_total_tokens": 0,
                "loss_weighted_four_role_sign_direction_router_update": int(router_update),
                "loss_weighted_four_role_sign_direction_safe_fraction": float(safe.float().mean().item()),
                "loss_weighted_four_role_sign_direction_parent_cosine_min": float(cosine.amin().item()),
                "loss_weighted_four_role_sign_direction_parent_cosine_median": float(cosine.median().item()),
                "loss_weighted_four_role_sign_direction_parent_cosine_max": float(cosine.amax().item()),
                "loss_weighted_four_role_sign_direction_budget_residual_max": float(budget.amax().item()),
            }
        self._capture_telemetry_next_step = False
        return loss


__all__ = (
    "FAMILY_ID",
    "LossWeightedFourRoleSignTransportOptimizer",
    "LossWeightedFourRoleSignTransportRouter",
    "RaggedPostpolarSignTransport",
    "arbitrary_shard_postpolar_sign_transport",
    "communicated_summary_elements",
    "group_intrinsic_participation",
    "group_loss_weighted_intrinsic_statistics",
    "method_state_elements",
    "postpolar_group_sign_transport",
)
