"""Lagged Global-RLB response-Fisher preconditioned matrix-sign optimizer."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import (
    _match_rms_adamw_adjustment,
    _zeropower_via_newton_schulz,
)
from .rlb_fixed32_transaction_muon_base import Fixed32TransactionMuonBase
from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_response_fisher_diagonal import (
    inverse_root_diagonal_scale,
    lagged_exponential_diagonal,
    method_state_elements,
    response_fisher_diagonal_sums,
)
from .rlb_ten_probe_loss_image_muon import _version_a_factors


FAMILY_ID = "response_fisher_muon_v1"


def _response_adjoint(
    cotangents: torch.Tensor,
    outgoing_weight: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    groups: int,
    width: int,
) -> torch.Tensor:
    unit, derivative, radial = factors
    expected = (cotangents.shape[0], int(groups), int(width))
    if any(value.shape != expected for value in (unit, derivative, radial)):
        raise RuntimeError("response-Fisher rational factor inventory changed")
    pulled = (cotangents.float() @ outgoing_weight.float()).view(expected)
    result = (
        derivative * pulled
        + unit * (radial * pulled).mean(dim=-1, keepdim=True)
    )
    torch._assert_async(torch.isfinite(result).all())
    return result


class ResponseFisherMuonOptimizer(Fixed32TransactionMuonBase):
    """Precondition Muon sources with lagged Global-RLB loss curvature."""

    family_id = FAMILY_ID
    telemetry_prefix = "response_fisher_"
    fairness_component = "response_fisher_direction_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["ten_probe_family_id"] = FAMILY_ID
        self.param_groups[0]["fixed32_transaction_family_id"] = FAMILY_ID
        self.param_groups[0]["response_fisher_family_id"] = FAMILY_ID

    def _transaction(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("response-Fisher Muon has no coefficient transaction")

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "matrix_role_lr_scale": 1.0,
            "fixed32_loss_measure_lr_scale": 1.0,
            "response_fisher_direction_lr_scale": 1.0,
            "lagged_curvature_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _global_curvature(self, packets):
        incoming_sums = []
        outgoing_sums = []
        for pair, packet in zip(self.pairs, packets):
            inputs, preactivations, features, cotangents = packet
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
            values = response_fisher_diagonal_sums(
                inputs, features, cotangents, response
            )
            incoming_sums.append(values.incoming)
            outgoing_sums.append(values.outgoing)
        packed = torch.cat((
            torch.stack(incoming_sums).reshape(-1),
            torch.stack(outgoing_sums).reshape(-1),
        ))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, group=self.loss_probe_group)
        packed.div_(float(FIXED_GLOBAL_PROBE_COUNT))
        incoming_elements = (
            len(self.pairs) * self.groups * self.external
        )
        incoming = packed[:incoming_elements].view(
            len(self.pairs), self.groups, self.external
        )
        outgoing = packed[incoming_elements:].view(
            len(self.pairs), self.groups, self.width
        )
        return incoming, outgoing

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("response-Fisher Muon lacks realized clipping")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("response-Fisher Muon refuses nonunit LR scale")

        packets = self._consume_probes()
        current_incoming, current_outgoing = self._global_curvature(packets)
        scale_samples = []
        initialized_count = 0
        for layer, pair in enumerate(self.pairs):
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            incoming_state = self.state[incoming]
            outgoing_state = self.state[outgoing]
            previous_incoming = incoming_state.get("response_fisher_diagonal")
            previous_outgoing = outgoing_state.get("response_fisher_diagonal")
            if previous_incoming is None:
                initialized_count += 1

            lagged_incoming, updated_incoming = lagged_exponential_diagonal(
                current_incoming[layer],
                previous_incoming,
                decay=self.momentum,
            )
            lagged_outgoing, updated_outgoing = lagged_exponential_diagonal(
                current_outgoing[layer],
                previous_outgoing,
                decay=self.momentum,
            )
            incoming_scale = inverse_root_diagonal_scale(
                lagged_incoming, eps=1.0e-8
            )
            outgoing_scale = inverse_root_diagonal_scale(
                lagged_outgoing, eps=1.0e-8
            )

            incoming_source = self._nesterov(incoming)
            outgoing_source = self._nesterov(outgoing)
            incoming_source.view(
                self.groups, self.width, self.external
            ).mul_(incoming_scale[:, None, :].to(incoming_source.dtype))
            outgoing_source.view(
                self.external, self.groups, self.width
            ).mul_(outgoing_scale[None, :, :].to(outgoing_source.dtype))
            incoming_direction = _zeropower_via_newton_schulz(
                incoming_source, self.ns_steps
            )
            outgoing_direction = _zeropower_via_newton_schulz(
                outgoing_source, self.ns_steps
            )
            incoming_adjustment = _match_rms_adamw_adjustment(incoming.shape)
            outgoing_adjustment = _match_rms_adamw_adjustment(outgoing.shape)

            incoming.mul_(1.0 - lr * weight_decay).add_(
                incoming_direction.to(incoming.dtype),
                alpha=-lr * incoming_adjustment,
            )
            outgoing.mul_(1.0 - lr * weight_decay).add_(
                outgoing_direction.to(outgoing.dtype),
                alpha=-lr * outgoing_adjustment,
            )
            incoming_state["response_fisher_diagonal"] = updated_incoming
            outgoing_state["response_fisher_diagonal"] = updated_outgoing
            if self._capture_telemetry_next_step:
                scale_samples.extend((incoming_scale.reshape(-1), outgoing_scale.reshape(-1)))

        anchor = self.state[self.pairs[0]["in_weight"]]
        updates = int(anchor.get("response_fisher_updates", 0)) + 1
        anchor["response_fisher_updates"] = updates
        if self._capture_telemetry_next_step:
            scales = torch.cat(scale_samples)
            self._last_telemetry = {
                "response_fisher_family_id": FAMILY_ID,
                "response_fisher_global_rows": FIXED_GLOBAL_PROBE_COUNT,
                "response_fisher_local_rows": self.probe_layout.local_probe_count,
                "response_fisher_owner_count": 0,
                "response_fisher_dense_lg_metric_elements": 0,
                "response_fisher_selected_update_elements_published": 0,
                "response_fisher_state_depends_on_total_tokens": 0,
                "response_fisher_state_coordinate_count": method_state_elements(
                    layers=len(self.pairs),
                    groups=self.groups,
                    width=self.width,
                    model_width=self.external,
                ),
                "response_fisher_decay": self.momentum,
                "response_fisher_updates": updates,
                "response_fisher_new_layers_initialized": initialized_count,
                "response_fisher_scale_min": float(scales.amin().item()),
                "response_fisher_scale_median": float(scales.median().item()),
                "response_fisher_scale_max": float(scales.amax().item()),
                "response_fisher_realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss


__all__ = (
    "FAMILY_ID",
    "ResponseFisherMuonOptimizer",
    "_response_adjoint",
)
