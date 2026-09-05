"""Owner-free residual-coordinate attention feedback from Global-RLB loss geometry.

The method deliberately leaves the two RLB matrix updates as literal Muon.
Fixed Global-RLB loss probes provide a lagged empirical-Fisher diagonal in the
residual coordinates feeding each MLP.  That diagonal reshapes the QKV input
columns and attention-output rows before the one ordinary NS5 map for each
attention role.  Only an ``L x d`` diagonal is persistent; no activation row,
parameter-sized update, complete layer, or ``(LG) x (LG)`` matrix is owned.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import (
    _match_rms_adamw_adjustment,
)
from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_group_muon_core import _batched_zero_power
from .rlb_response_fisher_diagonal import (
    inverse_root_diagonal_scale,
    lagged_exponential_diagonal,
)
from .rlb_response_fisher_muon import ResponseFisherMuonOptimizer, _response_adjoint
from .rlb_ten_probe_loss_image_muon import _version_a_factors


FAMILY_ID = "residual_fisher_attention_muon_v1"


def residual_fisher_sums(
    inputs: torch.Tensor,
    response_adjoint: torch.Tensor,
) -> torch.Tensor:
    """Return the residual-coordinate loss curvature summed over RLB groups."""

    if inputs.ndim != 2 or response_adjoint.ndim != 3:
        raise RuntimeError("residual-Fisher tensor rank changed")
    probes, groups, _width = response_adjoint.shape
    if inputs.shape[0] != probes:
        raise RuntimeError("residual-Fisher probe inventory changed")
    if probes == 0:
        return torch.zeros(
            inputs.shape[1], device=inputs.device, dtype=inputs.dtype
        )
    loss_power = response_adjoint.square().mean(dim=(-1, -2))
    result = loss_power @ inputs.square()
    torch._assert_async(torch.isfinite(result).all())
    return result


def residual_feedback_source(
    source: torch.Tensor,
    scale: torch.Tensor,
    *,
    role: str,
) -> torch.Tensor:
    """Apply the residual metric on the mathematically matching matrix side."""

    if source.ndim != 3 or scale.ndim != 2 or source.shape[0] != scale.shape[0]:
        raise RuntimeError("residual-feedback source inventory changed")
    if role == "qkv":
        if source.shape[-1] != scale.shape[-1]:
            raise RuntimeError("QKV residual coordinate inventory changed")
        result = source.float() * scale[:, None, :]
    elif role == "attn_out":
        if source.shape[-2] != scale.shape[-1]:
            raise RuntimeError("attention-output residual coordinate inventory changed")
        result = source.float() * scale[:, :, None]
    else:
        raise ValueError(f"unknown residual-feedback role: {role}")
    torch._assert_async(torch.isfinite(result).all())
    return result


def method_state_elements(*, layers: int, model_width: int) -> int:
    if int(layers) <= 0 or int(model_width) <= 0:
        raise ValueError("residual-Fisher dimensions must be positive")
    return int(layers) * int(model_width) + 1


class ResidualFisherAttentionRouter(ResponseFisherMuonOptimizer):
    """Literal RLB Muon plus one lagged residual loss metric for attention."""

    family_id = FAMILY_ID
    telemetry_prefix = "residual_fisher_attention_"
    fairness_component = "residual_fisher_attention_feedback_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["ten_probe_family_id"] = FAMILY_ID
        self.param_groups[0]["fixed32_transaction_family_id"] = FAMILY_ID
        self.param_groups[0]["response_fisher_family_id"] = FAMILY_ID
        self.param_groups[0]["residual_fisher_attention_family_id"] = FAMILY_ID
        self._attention_scale = None
        self._attention_update = 0
        self._attention_consumed = True

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "rlb_literal_muon_lr_scale": 1.0,
            "fixed32_loss_measure_lr_scale": 1.0,
            "lagged_residual_fisher_lr_scale": 1.0,
            "residual_fisher_attention_feedback_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _global_residual_curvature(self, packets):
        local = []
        for pair, packet in zip(self.pairs, packets):
            inputs, preactivations, _features, cotangents = packet
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
            local.append(residual_fisher_sums(inputs, response))
        packed = torch.stack(local)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, group=self.loss_probe_group)
        packed.div_(float(FIXED_GLOBAL_PROBE_COUNT))
        torch._assert_async(torch.isfinite(packed).all())
        return packed

    def consume_attention_scale(self):
        if self._attention_consumed or self._attention_scale is None:
            raise RuntimeError("residual-Fisher attention route is unavailable")
        self._attention_consumed = True
        return self._attention_scale, int(self._attention_update)

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("residual-Fisher router lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("residual-Fisher router would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("residual-Fisher router refuses nonunit LR scale")

        packets = self._consume_probes()
        current = self._global_residual_curvature(packets)
        anchor = self.state[self.pairs[0]["in_weight"]]
        previous = anchor.get("residual_fisher_attention_diagonal")
        lagged, updated = lagged_exponential_diagonal(
            current, previous, decay=self.momentum
        )
        scale = inverse_root_diagonal_scale(lagged, eps=1.0e-8)
        anchor["residual_fisher_attention_diagonal"] = updated

        incoming_parameters = [pair["in_weight"] for pair in self.pairs]
        outgoing_parameters = [pair["out_weight"] for pair in self.pairs]
        incoming_sources = torch.stack([
            self._nesterov(parameter).float() for parameter in incoming_parameters
        ])
        outgoing_sources = torch.stack([
            self._nesterov(parameter).float() for parameter in outgoing_parameters
        ])
        incoming_direction = _batched_zero_power(
            incoming_sources, self.ns_steps
        )
        outgoing_direction = _batched_zero_power(
            outgoing_sources, self.ns_steps
        )
        incoming_adjustment = _match_rms_adamw_adjustment(
            incoming_parameters[0].shape
        )
        outgoing_adjustment = _match_rms_adamw_adjustment(
            outgoing_parameters[0].shape
        )
        for layer, pair in enumerate(self.pairs):
            pair["in_weight"].mul_(1.0 - lr * weight_decay).add_(
                incoming_direction[layer].to(pair["in_weight"].dtype),
                alpha=-lr * incoming_adjustment,
            )
            pair["out_weight"].mul_(1.0 - lr * weight_decay).add_(
                outgoing_direction[layer].to(pair["out_weight"].dtype),
                alpha=-lr * outgoing_adjustment,
            )

        updates = int(anchor.get("residual_fisher_attention_updates", 0)) + 1
        anchor["residual_fisher_attention_updates"] = updates
        self._attention_scale = scale
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            self._last_telemetry = {
                "residual_fisher_attention_family_id": FAMILY_ID,
                "residual_fisher_attention_global_rows": FIXED_GLOBAL_PROBE_COUNT,
                "residual_fisher_attention_local_rows": self.probe_layout.local_probe_count,
                "residual_fisher_attention_owner_count": 0,
                "residual_fisher_attention_dense_lg_metric_elements": 0,
                "residual_fisher_attention_selected_update_elements_published": 0,
                "residual_fisher_attention_state_depends_on_total_tokens": 0,
                "residual_fisher_attention_state_coordinate_count": method_state_elements(
                    layers=len(self.pairs), model_width=self.external
                ),
                "residual_fisher_attention_updates": updates,
                "residual_fisher_attention_new_state_initialized": int(previous is None),
                "residual_fisher_attention_scale_min": float(scale.amin().item()),
                "residual_fisher_attention_scale_median": float(scale.median().item()),
                "residual_fisher_attention_scale_max": float(scale.amax().item()),
                "residual_fisher_attention_realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._attention_scale = None
        self._attention_update = 0
        self._attention_consumed = True
        return result


class ResidualFisherAttentionOptimizer(torch.optim.Optimizer):
    """Apply one Global-RLB residual metric before each attention NS5."""

    _ROLES = ("qkv", "attn_out")

    def __init__(
        self,
        blocks,
        router: ResidualFisherAttentionRouter,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta2: float,
        eps: float,
        adjust_lr_fn: str,
    ):
        if (
            float(lr) != 3.0e-4
            or float(weight_decay) != 0.1
            or float(momentum) != 0.95
            or int(ns_steps) != 5
            or float(beta2) != 0.95
            or float(eps) != 1.0e-8
            or adjust_lr_fn != "match_rms_adamw"
        ):
            raise ValueError("residual-Fisher attention requires the locked cell")
        self.blocks = sorted(
            (dict(block) for block in blocks), key=lambda item: int(item["layer_index"])
        )
        if [int(block["layer_index"]) for block in self.blocks] != list(
            range(len(self.blocks))
        ):
            raise ValueError("residual-Fisher attention layer inventory changed")
        self.router = router
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        self.role_parameters = {
            "qkv": [block["qkv_weight"] for block in self.blocks],
            "attn_out": [block["attn_out_weight"] for block in self.blocks],
        }
        parameters = []
        seen = {
            id(pair[role])
            for pair in router.pairs
            for role in ("in_weight", "out_weight")
        }
        for role in self._ROLES:
            for parameter in self.role_parameters[role]:
                if parameter.ndim != 2 or id(parameter) in seen:
                    raise ValueError("residual-Fisher attention ownership changed")
                seen.add(id(parameter))
                parameters.append(parameter)
        super().__init__([{
            "params": parameters,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_scale": 1.0,
            "residual_fisher_attention_family_id": FAMILY_ID,
        }], {})
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "residual_coordinate_fisher_lr_scale": 1.0,
            "single_ns5_per_attention_role_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def set_telemetry_capture(self, enabled=True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def _nesterov(self, parameter):
        if parameter.grad is None:
            raise RuntimeError("residual-Fisher attention gradient is missing")
        state = self.state[parameter]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(parameter)
            state["momentum_buffer"] = buffer
        buffer.lerp_(parameter.grad, 1.0 - self.momentum)
        return parameter.grad.lerp(buffer, self.momentum)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("residual-Fisher attention refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        scale, router_update = self.router.consume_attention_scale()
        if scale.shape != (len(self.blocks), self.blocks[0]["qkv_weight"].shape[1]):
            raise RuntimeError("residual-Fisher attention scale inventory changed")
        anchor = self.state[self.role_parameters["qkv"][0]]
        previous = int(anchor.get("residual_fisher_attention_router_update", 0))
        if int(router_update) != previous + 1:
            raise RuntimeError("residual-Fisher attention missed a router update")
        anchor["residual_fisher_attention_router_update"] = int(router_update)

        cosines = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            sources = torch.stack([
                self._nesterov(parameter).float() for parameter in parameters
            ])
            selected = residual_feedback_source(sources, scale, role=role)
            direction = _batched_zero_power(selected, self.ns_steps)
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(
                    direction[layer].to(parameter.dtype), alpha=-lr * adjustment
                )
            if self._capture_telemetry_next_step:
                numerator = (sources * selected).sum(dim=(-2, -1))
                denominator = (
                    torch.linalg.vector_norm(sources, dim=(-2, -1))
                    * torch.linalg.vector_norm(selected, dim=(-2, -1))
                ).clamp_min(torch.finfo(sources.dtype).tiny)
                cosines.append((numerator / denominator).clamp(-1.0, 1.0))
        if self._capture_telemetry_next_step:
            values = torch.cat(cosines)
            self._last_telemetry = {
                "residual_fisher_attention_direction_family_id": FAMILY_ID,
                "residual_fisher_attention_direction_owner_count": 0,
                "residual_fisher_attention_direction_dense_lg_metric_elements": 0,
                "residual_fisher_attention_direction_selected_update_elements_published": 0,
                "residual_fisher_attention_direction_state_depends_on_total_tokens": 0,
                "residual_fisher_attention_direction_router_update": int(router_update),
                "residual_fisher_attention_direction_source_cosine_min": float(values.amin().item()),
                "residual_fisher_attention_direction_source_cosine_median": float(values.median().item()),
                "residual_fisher_attention_direction_source_cosine_max": float(values.amax().item()),
            }
        self._capture_telemetry_next_step = False
        return loss


__all__ = (
    "FAMILY_ID",
    "ResidualFisherAttentionOptimizer",
    "ResidualFisherAttentionRouter",
    "method_state_elements",
    "residual_feedback_source",
    "residual_fisher_sums",
)
