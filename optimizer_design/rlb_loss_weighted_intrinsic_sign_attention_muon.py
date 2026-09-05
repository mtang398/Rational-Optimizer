"""Loss-weighted intrinsic-participation sign-chord attention Muon."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_group_muon_core import _batched_zero_power
from .rlb_residual_fisher_attention_muon import (
    ResidualFisherAttentionOptimizer,
    ResidualFisherAttentionRouter,
)
from .rlb_response_alignment_row import _jacobian_kernel_inner
from .rlb_response_fisher_muon import _response_adjoint
from .rlb_ten_probe_loss_image_muon import _version_a_factors


FAMILY_ID = "loss_weighted_intrinsic_sign_attention_muon_v1"


@dataclass(frozen=True)
class RaggedIntrinsicSignChord:
    local_selected: torch.Tensor
    global_active: torch.Tensor
    global_safe: torch.Tensor
    owner_count: int = 0
    dense_lg_by_lg_metric_elements: int = 0
    selected_update_elements_published: int = 0

    @property
    def communicated_summary_elements(self) -> int:
        return 5 * int(self.global_active.numel())


def arbitrary_shard_intrinsic_sign_chord(
    local_parent: torch.Tensor,
    local_gradient: torch.Tensor,
    logical_layer_ids: torch.Tensor,
    replica_weights: torch.Tensor,
    participation: torch.Tensor,
    *,
    total_layers: int,
    process_group=None,
    eps: float = 1.0e-8,
) -> RaggedIntrinsicSignChord:
    """Apply the same sign chord to arbitrary parameter fragments."""

    if local_parent.ndim != 1:
        raise RuntimeError("ragged intrinsic-sign source inventory changed")
    count = int(local_parent.numel())
    if (
        local_gradient.shape != (count,)
        or logical_layer_ids.shape != (count,)
        or replica_weights.shape != (count,)
    ):
        raise RuntimeError("ragged intrinsic-sign metadata changed")
    if logical_layer_ids.dtype != torch.int64:
        raise TypeError("ragged intrinsic-sign layer ids must be int64")
    if int(total_layers) <= 0 or participation.shape != (int(total_layers),):
        raise RuntimeError("ragged intrinsic-sign layer inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("ragged intrinsic-sign chord uses the locked epsilon")
    if count and (
        int(logical_layer_ids.amin()) < 0
        or int(logical_layer_ids.amax()) >= int(total_layers)
    ):
        raise ValueError("ragged intrinsic-sign layer id is out of range")
    if not torch.isfinite(replica_weights).all() or (replica_weights < 0.0).any():
        raise ValueError("ragged intrinsic-sign replica weights are invalid")
    parent = local_parent.float()
    gradient = local_gradient.float()
    weight = replica_weights.float()
    moments = torch.zeros(
        int(total_layers), 3, device=parent.device, dtype=torch.float64
    )
    if count:
        moments.index_add_(
            0,
            logical_layer_ids,
            torch.stack((
                parent.square(),
                (parent != 0.0).float(),
                parent.abs(),
            ), dim=-1).double() * weight.double()[:, None],
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(moments, op=dist.ReduceOp.SUM, group=process_group)
    p2, sign2, absolute = moments.unbind(dim=-1)
    sign_scale = torch.sqrt(p2 / sign2.clamp_min(float(eps)))
    a = participation.float().clamp(0.0, 1.0).double()
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source2 = p2 + 2.0 * a * delta * sign_scale * absolute
    active = (
        torch.isfinite(moments).all(dim=-1)
        & torch.isfinite(source2)
        & (p2 > 0.0)
        & (sign2 > 0.0)
        & (source2 > 0.0)
    )
    local_a = a.float().index_select(0, logical_layer_ids)
    local_delta = delta.float().index_select(0, logical_layer_ids)
    local_sign_scale = sign_scale.float().index_select(0, logical_layer_ids)
    local_budget_scale = torch.sqrt(
        p2 / source2.clamp_min(float(eps))
    ).float().index_select(0, logical_layer_ids)
    candidate = (
        local_a * parent
        + local_delta * torch.sign(parent) * local_sign_scale
    ) * local_budget_scale
    candidate = torch.where(local_a == 1.0, parent, candidate)
    candidate = torch.where(
        active.index_select(0, logical_layer_ids), candidate, parent
    )
    descent = torch.zeros(
        int(total_layers), 2, device=parent.device, dtype=torch.float64
    )
    if count:
        descent.index_add_(
            0,
            logical_layer_ids,
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
    selected = torch.where(
        safe.index_select(0, logical_layer_ids), candidate, parent
    )
    if not torch.isfinite(selected).all():
        raise RuntimeError("ragged intrinsic-sign chord is nonfinite")
    return RaggedIntrinsicSignChord(selected, active, safe)


def loss_weighted_intrinsic_participation_statistics(
    unit: torch.Tensor,
    derivative: torch.Tensor,
    radial: torch.Tensor,
    response_adjoint: torch.Tensor,
    cotangents: torch.Tensor,
) -> torch.Tensor:
    """Return additive incoming/outgoing weighted participation numerators."""

    if not (unit.shape == derivative.shape == radial.shape == response_adjoint.shape):
        raise RuntimeError("intrinsic-participation rational inventory changed")
    if unit.ndim != 3 or cotangents.ndim != 2 or cotangents.shape[0] != unit.shape[0]:
        raise RuntimeError("intrinsic-participation probe inventory changed")
    width = float(unit.shape[-1])
    function = radial + unit * derivative
    radial_jacobian = radial / width
    trace = derivative.square().sum(dim=-1)
    trace = trace + 2.0 * (
        derivative * unit * radial_jacobian
    ).sum(dim=-1)
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
        (incoming * incoming_weight).sum(),
        incoming_weight.sum(),
        (outgoing * outgoing_weight).sum(),
        outgoing_weight.expand_as(outgoing).sum(),
    ))
    torch._assert_async(torch.isfinite(result).all())
    return result


def layer_loss_weighted_intrinsic_participation(
    statistics: torch.Tensor,
) -> torch.Tensor:
    """Map additive four-scalar layer statistics to a canonical chord cosine."""

    if statistics.ndim != 2 or statistics.shape[-1] != 4:
        raise RuntimeError("intrinsic-participation layer inventory changed")
    incoming = statistics[:, 0] / statistics[:, 1].clamp_min(
        torch.finfo(statistics.dtype).tiny
    )
    outgoing = statistics[:, 2] / statistics[:, 3].clamp_min(
        torch.finfo(statistics.dtype).tiny
    )
    valid = (
        torch.isfinite(statistics).all(dim=-1)
        & (statistics[:, 1] > 0.0)
        & (statistics[:, 3] > 0.0)
    )
    value = torch.sqrt((incoming * outgoing).clamp(0.0, 1.0))
    return torch.where(valid, value, torch.ones_like(value))


def equal_budget_intrinsic_sign_chord(
    parent: torch.Tensor,
    participation: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Join momentum to its equal-energy coordinate-sign source."""

    if parent.ndim != 3 or participation.shape != (parent.shape[0],):
        raise RuntimeError("intrinsic sign-chord inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("intrinsic sign chord uses the locked epsilon")
    value = parent.float()
    dims = (-2, -1)
    parent_norm = torch.linalg.vector_norm(value, dim=dims, keepdim=True)
    sign = torch.sign(value)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = (
        torch.isfinite(value).all(dim=dims, keepdim=True)
        & (parent_norm > 0.0)
        & (sign_norm > 0.0)
    )
    sign_equal = sign * (
        parent_norm / sign_norm.clamp_min(torch.finfo(value.dtype).tiny)
    )
    a = participation.float().clamp(0.0, 1.0)[:, None, None]
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source = a * value + delta * sign_equal
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    selected = source * (
        parent_norm / source_norm.clamp_min(torch.finfo(value.dtype).tiny)
    )
    selected = torch.where(a == 1.0, value, selected)
    selected = torch.where(valid, selected, value)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    pnorm = parent_norm.flatten()
    cosine = (
        (value * selected).sum(dim=dims)
        / (pnorm * selected_norm).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    budget = (selected_norm - pnorm).abs() / pnorm.clamp_min(1.0)
    torch._assert_async(torch.isfinite(selected).all())
    return selected, {
        "active": valid.flatten(),
        "participation": participation.float().clamp(0.0, 1.0),
        "parent_cosine": cosine,
        "budget_residual": budget,
    }


def method_state_elements(*, layers: int, model_width: int) -> int:
    if int(layers) <= 0 or int(model_width) <= 0:
        raise ValueError("intrinsic sign dimensions must be positive")
    return int(layers) * int(model_width) + 1


def communicated_summary_elements(*, layers: int) -> int:
    if int(layers) <= 0:
        raise ValueError("intrinsic sign depth must be positive")
    # Four participation scalars and five chord/descent scalars for two roles.
    return 14 * int(layers)


class LossWeightedIntrinsicSignAttentionRouter(ResidualFisherAttentionRouter):
    """Literal RLB Muon plus one loss-weighted intrinsic attention angle."""

    family_id = FAMILY_ID
    telemetry_prefix = "loss_weighted_intrinsic_sign_attention_"
    fairness_component = "loss_weighted_intrinsic_sign_attention_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["ten_probe_family_id"] = FAMILY_ID
        group["fixed32_transaction_family_id"] = FAMILY_ID
        group["response_fisher_family_id"] = FAMILY_ID
        group["residual_fisher_attention_family_id"] = FAMILY_ID
        group["loss_weighted_intrinsic_sign_attention_family_id"] = FAMILY_ID
        self._intrinsic_participation = None

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "rlb_literal_muon_lr_scale": 1.0,
            "fixed32_loss_measure_lr_scale": 1.0,
            "loss_weighted_intrinsic_participation_lr_scale": 1.0,
            "attention_sign_chord_lr_scale": 1.0,
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
            values.append(loss_weighted_intrinsic_participation_statistics(
                *factors, response, cotangents
            ))
        statistics = torch.stack(values)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        self._intrinsic_participation = layer_loss_weighted_intrinsic_participation(
            statistics
        )
        return torch.ones(
            len(self.pairs), self.external,
            device=statistics.device, dtype=statistics.dtype,
        )

    def consume_intrinsic_participation(self):
        if self._attention_consumed or self._intrinsic_participation is None:
            raise RuntimeError("intrinsic-sign attention route is unavailable")
        self._attention_consumed = True
        return self._intrinsic_participation, int(self._attention_update)

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._intrinsic_participation = None
        loss = super().step(closure)
        if self._intrinsic_participation is None:
            raise RuntimeError("intrinsic-sign router omitted participation")
        if publish:
            value = self._intrinsic_participation
            self._last_telemetry = {
                "loss_weighted_intrinsic_sign_attention_family_id": FAMILY_ID,
                "loss_weighted_intrinsic_sign_attention_owner_count": 0,
                "loss_weighted_intrinsic_sign_attention_dense_lg_metric_elements": 0,
                "loss_weighted_intrinsic_sign_attention_selected_update_elements_published": 0,
                "loss_weighted_intrinsic_sign_attention_state_depends_on_total_tokens": 0,
                "loss_weighted_intrinsic_sign_attention_state_coordinate_count": method_state_elements(
                    layers=len(self.pairs), model_width=self.external
                ),
                "loss_weighted_intrinsic_sign_attention_summary_elements": communicated_summary_elements(
                    layers=len(self.pairs)
                ),
                "loss_weighted_intrinsic_sign_attention_updates": int(self._attention_update),
                "loss_weighted_intrinsic_sign_attention_participation_min": float(value.amin().item()),
                "loss_weighted_intrinsic_sign_attention_participation_median": float(value.median().item()),
                "loss_weighted_intrinsic_sign_attention_participation_max": float(value.amax().item()),
            }
        return loss


class LossWeightedIntrinsicSignAttentionOptimizer(ResidualFisherAttentionOptimizer):
    """One NS5 after the persistent Global-RLB intrinsic sign chord."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]["loss_weighted_intrinsic_sign_attention_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "equal_budget_sign_chord_lr_scale": 1.0,
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
            raise RuntimeError("intrinsic-sign attention refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        participation, router_update = self.router.consume_intrinsic_participation()
        anchor = self.state[self.role_parameters["qkv"][0]]
        previous = int(anchor.get("intrinsic_sign_attention_router_update", 0))
        if int(router_update) != previous + 1:
            raise RuntimeError("intrinsic-sign attention missed a router update")
        anchor["intrinsic_sign_attention_router_update"] = int(router_update)

        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            sources = torch.stack([
                self._nesterov(parameter).float() for parameter in parameters
            ])
            alternative, metadata = equal_budget_intrinsic_sign_chord(
                sources, participation
            )
            gradients = torch.stack([
                parameter.grad.detach().float() for parameter in parameters
            ])
            parent_descent = (gradients * sources).sum(dim=(-2, -1))
            alternative_descent = (gradients * alternative).sum(dim=(-2, -1))
            safe = (
                metadata["active"]
                & torch.isfinite(parent_descent)
                & (parent_descent > 0.0)
                & torch.isfinite(alternative_descent)
                & (alternative_descent > 0.0)
            )
            selected = torch.where(safe[:, None, None], alternative, sources)
            direction = _batched_zero_power(selected, self.ns_steps)
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(
                    direction[layer].to(parameter.dtype), alpha=-lr * adjustment
                )
            if self._capture_telemetry_next_step:
                records.append((safe, metadata))
        if self._capture_telemetry_next_step:
            safe = torch.cat([item[0] for item in records])
            cosine = torch.cat([item[1]["parent_cosine"] for item in records])
            budget = torch.cat([item[1]["budget_residual"] for item in records])
            self._last_telemetry = {
                "loss_weighted_intrinsic_sign_direction_family_id": FAMILY_ID,
                "loss_weighted_intrinsic_sign_direction_owner_count": 0,
                "loss_weighted_intrinsic_sign_direction_dense_lg_metric_elements": 0,
                "loss_weighted_intrinsic_sign_direction_selected_update_elements_published": 0,
                "loss_weighted_intrinsic_sign_direction_state_depends_on_total_tokens": 0,
                "loss_weighted_intrinsic_sign_direction_router_update": int(router_update),
                "loss_weighted_intrinsic_sign_direction_safe_fraction": float(safe.float().mean().item()),
                "loss_weighted_intrinsic_sign_direction_parent_cosine_min": float(cosine.amin().item()),
                "loss_weighted_intrinsic_sign_direction_parent_cosine_median": float(cosine.median().item()),
                "loss_weighted_intrinsic_sign_direction_parent_cosine_max": float(cosine.amax().item()),
                "loss_weighted_intrinsic_sign_direction_budget_residual_max": float(budget.amax().item()),
            }
        self._capture_telemetry_next_step = False
        return loss


__all__ = (
    "FAMILY_ID",
    "LossWeightedIntrinsicSignAttentionOptimizer",
    "LossWeightedIntrinsicSignAttentionRouter",
    "RaggedIntrinsicSignChord",
    "arbitrary_shard_intrinsic_sign_chord",
    "communicated_summary_elements",
    "equal_budget_intrinsic_sign_chord",
    "layer_loss_weighted_intrinsic_participation",
    "loss_weighted_intrinsic_participation_statistics",
    "method_state_elements",
)
