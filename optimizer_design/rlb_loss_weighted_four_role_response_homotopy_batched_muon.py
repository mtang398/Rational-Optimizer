"""Batched/foreach realization of four-role frozen-response homotopy Muon."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_batched_four_role_response_sensor import (
    stacked_intrinsic_and_response_statistics,
    stacked_response_adjoint,
    stacked_version_a_factors,
)
from .rlb_group_muon_core import _batched_zero_power
from .rlb_loss_weighted_four_role_response_homotopy_muon import (
    LossWeightedFourRoleResponseHomotopyOptimizer,
    LossWeightedFourRoleResponseHomotopyRouter,
    communicated_summary_elements,
    group_loss_weighted_response_congruence,
    layer_loss_weighted_response_congruence,
    method_state_elements,
    postpolar_group_response_homotopy,
)
from .rlb_loss_weighted_four_role_sign_transport_muon import (
    group_intrinsic_participation,
)


FAMILY_ID = "loss_weighted_four_role_response_homotopy_batched_muon_v2"


def _foreach_nesterov(optimizer, parameters) -> torch.Tensor:
    gradients = []
    buffers = []
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("batched response-homotopy gradient is missing")
        gradients.append(parameter.grad)
        state = optimizer.state[parameter]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(
                parameter.grad, memory_format=torch.preserve_format
            )
            state["momentum_buffer"] = buffer
        buffers.append(buffer)
    torch._foreach_lerp_(buffers, gradients, 1.0 - optimizer.momentum)
    values = torch._foreach_lerp(gradients, buffers, optimizer.momentum)
    return torch.stack(values).float()


def _foreach_apply(parameters, direction, *, decay: float, alpha: float) -> None:
    torch._foreach_mul_(parameters, float(decay))
    values = list(direction.to(parameters[0].dtype).unbind(dim=0))
    torch._foreach_add_(parameters, values, alpha=float(alpha))


class BatchedFourRoleResponseHomotopyRouter(
    LossWeightedFourRoleResponseHomotopyRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = "loss_weighted_four_role_response_homotopy_batched_"
    fairness_component = "loss_weighted_four_role_response_homotopy_batched_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["loss_weighted_four_role_response_homotopy_batched_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "rlb_incoming_lr_scale": 1.0,
            "rlb_outgoing_lr_scale": 1.0,
            "fixed32_loss_measure_lr_scale": 1.0,
            "group_intrinsic_participation_lr_scale": 1.0,
            "loss_weighted_frozen_response_lr_scale": 1.0,
            "postpolar_homotopy_lr_scale": 1.0,
            "batched_sensor_lr_scale": 1.0,
            "foreach_realization_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _global_residual_curvature(self, packets):
        anchor = self.state[self.pairs[0]["in_weight"]]
        frozen_numerators = anchor.get("four_role_response_frozen_numerators")
        frozen_denominators = anchor.get("four_role_response_frozen_denominators")
        layers = len(self.pairs)
        if (
            frozen_numerators is None
            or frozen_numerators.shape != (layers, self.groups, 6)
            or frozen_denominators is None
            or frozen_denominators.shape != (layers, self.groups, 4)
        ):
            raise RuntimeError("batched response-homotopy frozen inventory changed")
        preactivations = torch.stack([packet[1] for packet in packets])
        cotangents = torch.stack([packet[3] for packet in packets])
        numerators = torch.stack([
            pair["numerator"].detach() for pair in self.pairs
        ])
        denominators = torch.stack([
            pair["denominator"].detach() for pair in self.pairs
        ])
        outgoing = torch.stack([
            pair["out_weight"].detach() for pair in self.pairs
        ])
        factors = stacked_version_a_factors(
            preactivations, numerators, denominators,
            groups=self.groups, width=self.width, eps=self.rlb_eps,
        )
        response_adjoint = stacked_response_adjoint(
            cotangents, outgoing, factors,
            groups=self.groups, width=self.width,
        )
        participation_statistics, response_statistics = (
            stacked_intrinsic_and_response_statistics(
                preactivations, cotangents, response_adjoint, factors,
                frozen_numerators, frozen_denominators, eps=self.rlb_eps,
            )
        )
        first = participation_statistics.numel()
        packed = torch.cat((
            participation_statistics.reshape(-1), response_statistics.reshape(-1)
        ))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        participation_statistics = packed[:first].view_as(participation_statistics)
        response_statistics = packed[first:].view_as(response_statistics)
        exact = (
            torch.all(numerators.float() == frozen_numerators, dim=-1)
            & torch.all(denominators.float() == frozen_denominators, dim=-1)
        ).to(torch.int32)
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
        outgoing_participation = participation[..., 1].mean(dim=-1)
        self._group_participation = participation
        self._group_response_congruence = group_congruence
        self._intrinsic_participation = torch.sqrt(
            (incoming * outgoing_participation).clamp(0.0, 1.0)
        )
        self._attention_response_congruence = attention_congruence
        return participation

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("batched response router lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("batched response router would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("batched response router refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        participation = self._global_residual_curvature(self._consume_probes())
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError("batched response router omitted congruence")
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
            selected, metadata = postpolar_group_response_homotopy(
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
        updates = int(anchor.get("batched_response_homotopy_updates", 0)) + 1
        anchor["batched_response_homotopy_updates"] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            angles = congruence.reshape(-1)
            safe = torch.cat([record["safe"].reshape(-1) for record in role_records])
            cosine = torch.cat([record["parent_cosine"].reshape(-1) for record in role_records])
            budget = torch.cat([record["budget_residual"].reshape(-1) for record in role_records])
            prefix = "loss_weighted_four_role_response_homotopy_batched_"
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


class BatchedFourRoleResponseHomotopyAttentionOptimizer(
    LossWeightedFourRoleResponseHomotopyOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]["loss_weighted_four_role_response_homotopy_batched_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "loss_weighted_frozen_response_lr_scale": 1.0,
            "postpolar_homotopy_lr_scale": 1.0,
            "single_ns5_per_attention_role_lr_scale": 1.0,
            "foreach_realization_lr_scale": 1.0,
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
            raise RuntimeError("batched response attention refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        participation, congruence, router_update = self.router.consume_response_homotopy()
        anchor = self.state[self.role_parameters["qkv"][0]]
        previous = int(anchor.get("batched_response_homotopy_router_update", 0))
        if int(router_update) != previous + 1:
            raise RuntimeError("batched response attention missed a router update")
        anchor["batched_response_homotopy_router_update"] = int(router_update)
        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = _foreach_nesterov(self, parameters)
            gradients = torch.stack([
                parameter.grad.detach() for parameter in parameters
            ]).float()
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = postpolar_group_response_homotopy(
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
            prefix = "loss_weighted_four_role_response_homotopy_batched_direction_"
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
    "BatchedFourRoleResponseHomotopyAttentionOptimizer",
    "BatchedFourRoleResponseHomotopyRouter",
    "FAMILY_ID",
)
