"""Lean execution base for fixed-row owner-free coefficient transactions."""

from __future__ import annotations

import torch

from .rlb_basis_cotangent_trust_muon import (
    _match_rms_adamw_adjustment,
    _zeropower_via_newton_schulz,
)
from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_fixed_global_probe_layout import fixed_global_probe_layout
from .rlb_ten_probe_loss_image_muon import (
    EXPECTED_MICROBATCHES,
    TenProbeLossImageMuonOptimizer,
    _one_layer_group_scores,
    _version_a_factors,
)


class Fixed32TransactionMuonBase(TenProbeLossImageMuonOptimizer):
    """Shared direct-score execution; subclasses supply only scalar math."""

    family_id = "abstract_fixed32_transaction"
    telemetry_prefix = "abstract_fixed32_"
    fairness_component = "abstract_transaction_lr_scale"

    def __init__(self, pairs, **kwargs):
        if self.__class__ is Fixed32TransactionMuonBase:
            raise TypeError("fixed32 transaction base is abstract")
        super().__init__(pairs, **kwargs)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(group=self.loss_probe_group)
            world = torch.distributed.get_world_size(group=self.loss_probe_group)
        else:
            rank, world = 0, 1
        self.probe_layout = fixed_global_probe_layout(
            FIXED_GLOBAL_PROBE_COUNT, rank, world
        )
        self.capture_rows = (
            self.probe_layout.local_probe_count + EXPECTED_MICROBATCHES - 1
        ) // EXPECTED_MICROBATCHES
        self.param_groups[0]["ten_probe_family_id"] = self.family_id
        self.param_groups[0]["fixed32_transaction_family_id"] = self.family_id

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "matrix_role_lr_scale": 1.0,
            "fixed32_loss_measure_lr_scale": 1.0,
            self.fairness_component: 1.0,
            "equality_budget_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _transaction(
        self, score_lattice, local_decay, exact_by_role,
        momentum_by_role, weights, layer_ids, *, eta,
    ):
        raise NotImplementedError

    def _extra_transaction_telemetry(self, transaction):
        return {}

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("fixed32 transaction Muon lacks realized clipping")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("fixed32 transaction Muon refuses nonunit LR scale")
        packets = self._consume_probes()

        incoming_directions = []
        outgoing_directions = []
        incoming_adjustments = []
        outgoing_adjustments = []
        exact_incoming = []
        exact_outgoing = []
        momentum_incoming = []
        momentum_outgoing = []
        budget_weights = []
        for pair in self.pairs:
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            incoming_momentum = self._nesterov(incoming)
            outgoing_momentum = self._nesterov(outgoing)
            incoming_direction = _zeropower_via_newton_schulz(
                incoming_momentum, self.ns_steps
            )
            outgoing_direction = _zeropower_via_newton_schulz(
                outgoing_momentum, self.ns_steps
            )
            incoming_adjustment = _match_rms_adamw_adjustment(incoming.shape)
            outgoing_adjustment = _match_rms_adamw_adjustment(outgoing.shape)
            incoming_view = incoming_direction.view(
                self.groups, self.width, self.external
            )
            outgoing_view = outgoing_direction.view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0)
            incoming_gradient = incoming.grad.detach().view_as(incoming_view)
            outgoing_gradient = outgoing.grad.detach().view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0)
            incoming_momentum_view = incoming_momentum.view_as(incoming_view)
            outgoing_momentum_view = outgoing_momentum.view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0)
            incoming_float = incoming_view.float() * incoming_adjustment
            outgoing_float = outgoing_view.float() * outgoing_adjustment
            exact_incoming.append(
                (incoming_gradient.float() * incoming_float).sum(dim=(-2, -1))
            )
            exact_outgoing.append(
                (outgoing_gradient.float() * outgoing_float).sum(dim=(-2, -1))
            )
            momentum_incoming.append(
                (incoming_momentum_view.float() * incoming_float).sum(dim=(-2, -1))
            )
            momentum_outgoing.append(
                (outgoing_momentum_view.float() * outgoing_float).sum(dim=(-2, -1))
            )
            budget_weights.append(
                incoming_float.square().sum(dim=(-2, -1))
                + outgoing_float.square().sum(dim=(-2, -1))
            )
            incoming_directions.append(incoming_direction)
            outgoing_directions.append(outgoing_direction)
            incoming_adjustments.append(incoming_adjustment)
            outgoing_adjustments.append(outgoing_adjustment)

        local_scores = []
        local_decay = None
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            inputs, preactivations, features, cotangents = packet
            factors = _version_a_factors(
                preactivations, pair["numerator"], pair["denominator"],
                groups=self.groups, width=self.width, eps=self.rlb_eps,
            )
            incoming_direction = (
                incoming_directions[layer].float() * incoming_adjustments[layer]
            )
            outgoing_direction_transpose = (
                outgoing_directions[layer]
                .view(self.external, self.groups, self.width)
                .permute(1, 2, 0).reshape(self.hidden, self.external).float()
                * outgoing_adjustments[layer]
            )
            score, response_adjoint = _one_layer_group_scores(
                inputs, preactivations, features, cotangents,
                incoming_direction, outgoing_direction_transpose,
                pair["out_weight"], factors, groups=self.groups, width=self.width,
            )
            decay_score, _ = _one_layer_group_scores(
                inputs, preactivations, features, cotangents,
                pair["in_weight"].float() * weight_decay,
                pair["out_weight"].T.float() * weight_decay,
                pair["out_weight"], factors, groups=self.groups, width=self.width,
                cached_response_adjoint=response_adjoint,
            )
            local_scores.append(score)
            layer_decay = decay_score.sum(dim=-1)
            local_decay = layer_decay if local_decay is None else local_decay + layer_decay
        if local_decay is None:
            raise RuntimeError("fixed32 transaction decay action was not formed")
        score_lattice = torch.stack(local_scores, dim=1).reshape(
            self.probe_layout.local_probe_count, len(self.pairs) * self.groups
        )
        exact_by_role = torch.stack((
            torch.stack(exact_incoming).reshape(-1),
            torch.stack(exact_outgoing).reshape(-1),
        ))
        momentum_by_role = torch.stack((
            torch.stack(momentum_incoming).reshape(-1),
            torch.stack(momentum_outgoing).reshape(-1),
        ))
        weights = torch.stack(budget_weights).reshape(-1)
        layer_ids = torch.arange(
            len(self.pairs), device=weights.device, dtype=torch.int64
        ).repeat_interleave(self.groups)
        selection = self._transaction(
            score_lattice, local_decay, exact_by_role, momentum_by_role,
            weights, layer_ids, eta=lr,
        )
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)

        for layer, pair in enumerate(self.pairs):
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming_direction = incoming_directions[layer]
            outgoing_direction = outgoing_directions[layer]
            incoming_direction.view(
                self.groups, self.width, self.external
            ).mul_(coefficients[layer, :, None, None].to(incoming_direction.dtype))
            outgoing_direction.view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0).mul_(
                coefficients[layer, :, None, None].to(outgoing_direction.dtype)
            )
            incoming.add_(
                incoming_direction.to(incoming.dtype),
                alpha=-lr * incoming_adjustments[layer],
            )
            outgoing.add_(
                outgoing_direction.to(outgoing.dtype),
                alpha=-lr * outgoing_adjustments[layer],
            )

        if self._capture_telemetry_next_step:
            flat = coefficients.reshape(-1)
            transaction = selection.sharded_result
            prefix = self.telemetry_prefix
            report = {
                prefix + "family_id": self.family_id,
                prefix + "owner_count": transaction.owner_count,
                prefix + "global_rows": FIXED_GLOBAL_PROBE_COUNT,
                prefix + "local_rows": self.probe_layout.local_probe_count,
                prefix + "coordinate_count": len(self.pairs) * self.groups,
                prefix + "state_coordinate_count": 0,
                prefix + "state_depends_on_total_tokens": 0,
                prefix + "dense_lg_metric_elements": transaction.dense_LG_by_LG_metric_elements,
                prefix + "selected_update_elements_published": transaction.selected_update_elements_published,
                prefix + "transaction_accepted": int(transaction.accepted.item()),
                prefix + "rank": int(transaction.rank.item()),
                prefix + "budget_residual": float(transaction.budget_residual.item()),
                prefix + "parent_score": float(transaction.parent_score.item()),
                prefix + "candidate_score": float(transaction.candidate_score.item()),
                prefix + "cross_layer_coupling_ratio": float(selection.cross_layer_coupling_ratio.item()),
                prefix + "coefficient_min": float(flat.amin().item()),
                prefix + "coefficient_median": float(flat.median().item()),
                prefix + "coefficient_max": float(flat.amax().item()),
                prefix + "realized_clip_factor": float(self._clip_factor),
            }
            report.update(self._extra_transaction_telemetry(transaction))
            self._last_telemetry = report
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss


__all__ = ("Fixed32TransactionMuonBase",)
