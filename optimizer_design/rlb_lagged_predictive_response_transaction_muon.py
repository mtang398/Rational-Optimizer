"""Matched-beta2 predictive global transaction on a response parent.

The current-batch transaction can overfit the same functional rows that it
uses to choose its structured coefficients.  This owner-free alternative
selects with the previous globally aligned EMA of the fixed-row loss image,
then incorporates the current rows only after the update is fixed.  The EMA
uses the already locked Adam beta2 exactly; it introduces no new decay,
schedule, threshold, or LR/WD scalar.

The predictive state is ``32 * (L G + 1)`` scalars and therefore independent
of the number of activation positions.  All row-space algebra remains fixed
32-dimensional and only signed ``L G`` coefficients are reconstructed on
native logical shards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_compact_four_role_response_homotopy_muon import (
    FAMILY_ID as COMPACT_PARENT_FAMILY_ID,
    compact_postpolar_group_response_homotopy,
)
from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_fixed_probe_transaction import (
    ReplicatedFixedProbeTransactionResult,
    _gather_variable_probe_rows,
    distributed_fixed_probe_transaction,
)
from .rlb_global_response_transaction_muon import (
    FAMILY_ID as CURRENT_IMPLEMENTATION_FAMILY_ID,
    GlobalResponseTransactionAttentionOptimizer,
    GlobalResponseTransactionRouter,
    global_response_transaction_scaling_formula,
)
from .rlb_group_muon_core import _batched_zero_power
from .rlb_loss_weighted_four_role_response_homotopy_batched_muon import (
    _foreach_apply,
    _foreach_nesterov,
)
from .rlb_ten_probe_loss_image_muon import _one_layer_group_scores


FAMILY_ID = "lagged_predictive_response_transaction_muon_v1"


@dataclass(frozen=True)
class MatchedBeta2PredictiveRows:
    selection_scores: torch.Tensor
    selection_decay_action: torch.Tensor
    updated_scores: torch.Tensor
    updated_decay_action: torch.Tensor
    history_used: bool
    relative_innovation: torch.Tensor


def matched_beta2_predictive_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    """Select with pre-update history and update it using locked beta2."""

    if (
        current_scores.ndim != 2
        or current_scores.shape[0] != FIXED_GLOBAL_PROBE_COUNT
        or current_decay_action.shape != (FIXED_GLOBAL_PROBE_COUNT,)
        or not current_scores.is_floating_point()
        or current_decay_action.dtype != current_scores.dtype
        or current_decay_action.device != current_scores.device
        or float(beta2) != 0.95
        or not bool(torch.isfinite(current_scores).all())
        or not bool(torch.isfinite(current_decay_action).all())
    ):
        raise RuntimeError("matched-beta2 predictive row inventory changed")
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError("predictive score and decay histories must coinitialize")
    tiny = torch.finfo(current_scores.dtype).tiny
    if previous_scores is None:
        selection_scores = current_scores
        selection_decay = current_decay_action
        updated_scores = current_scores.detach().clone()
        updated_decay = current_decay_action.detach().clone()
        relative_innovation = torch.zeros(
            (), device=current_scores.device, dtype=current_scores.dtype
        )
        history_used = False
    else:
        if (
            previous_scores.shape != current_scores.shape
            or previous_decay_action.shape != current_decay_action.shape
            or previous_scores.dtype != current_scores.dtype
            or previous_decay_action.dtype != current_scores.dtype
            or previous_scores.device != current_scores.device
            or previous_decay_action.device != current_scores.device
            or not bool(torch.isfinite(previous_scores).all())
            or not bool(torch.isfinite(previous_decay_action).all())
        ):
            raise RuntimeError("matched-beta2 predictive history changed")
        selection_scores = previous_scores
        selection_decay = previous_decay_action
        innovation2 = (
            (current_scores - previous_scores).square().sum()
            + (current_decay_action - previous_decay_action).square().sum()
        )
        current2 = (
            current_scores.square().sum()
            + current_decay_action.square().sum()
        )
        relative_innovation = torch.sqrt(
            innovation2 / current2.clamp_min(tiny)
        )
        updated_scores = (
            previous_scores * float(beta2)
            + current_scores * (1.0 - float(beta2))
        )
        updated_decay = (
            previous_decay_action * float(beta2)
            + current_decay_action * (1.0 - float(beta2))
        )
        history_used = True
    return MatchedBeta2PredictiveRows(
        selection_scores=selection_scores,
        selection_decay_action=selection_decay,
        updated_scores=updated_scores,
        updated_decay_action=updated_decay,
        history_used=history_used,
        relative_innovation=relative_innovation,
    )


def lagged_predictive_response_transaction_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
) -> dict[str, int]:
    parent = global_response_transaction_scaling_formula(
        total_positions=total_positions,
        total_layers=total_layers,
        total_groups=total_groups,
        intermediate_width=intermediate_width,
        model_width=model_width,
    )
    coordinates = int(total_layers) * int(total_groups)
    predictive = FIXED_GLOBAL_PROBE_COUNT * (coordinates + 1)
    result = dict(parent)
    result["persistent_state_elements"] = (
        parent["persistent_state_elements"] + predictive
    )
    result["predictive_state_elements"] = predictive
    return result


def _transaction_from_replicated_global_rows(
    global_scores: torch.Tensor,
    global_decay_action: torch.Tensor,
    exact_by_role: torch.Tensor,
    momentum_by_role: torch.Tensor,
    weights: torch.Tensor,
    layer_ids: torch.Tensor,
    *,
    total_layers: int,
    eta: float,
    gather_rounds: int,
    group,
) -> ReplicatedFixedProbeTransactionResult:
    """Shard coordinates after the one current-row gather."""

    rows, coordinates = global_scores.shape
    if (
        rows != FIXED_GLOBAL_PROBE_COUNT
        or global_decay_action.shape != (rows,)
        or exact_by_role.shape != (2, coordinates)
        or momentum_by_role.shape != exact_by_role.shape
        or weights.shape != (coordinates,)
        or layer_ids.shape != (coordinates,)
    ):
        raise RuntimeError("predictive replicated transaction inventory changed")
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
    else:
        rank, world = 0, 1
    coordinate_ids = torch.arange(
        coordinates, device=global_scores.device, dtype=torch.int64
    )
    local_ids = coordinate_ids[coordinate_ids.remainder(world).eq(rank)]
    decay_cross = global_scores.T @ global_decay_action / float(rows)
    sharded = distributed_fixed_probe_transaction(
        global_scores[:, local_ids],
        exact_by_role[:, local_ids],
        momentum_by_role[:, local_ids],
        decay_cross[local_ids],
        weights[local_ids],
        layer_ids[local_ids],
        local_ids,
        total_coordinates=coordinates,
        total_layers=int(total_layers),
        eta=float(eta),
        rounds=64,
        group=group,
    )
    coefficient_packet = torch.zeros(
        2 * coordinates, device=global_scores.device, dtype=global_scores.dtype
    )
    coefficient_packet[local_ids] = sharded.local_coefficients
    coefficient_packet[coordinates + local_ids] = (
        sharded.local_candidate_coefficients
    )
    if world > 1:
        dist.all_reduce(coefficient_packet, op=dist.ReduceOp.SUM, group=group)

    total_row_metric = global_scores @ global_scores.T
    total_square = total_row_metric.square().sum()
    within_square = torch.zeros_like(total_square)
    for layer in range(int(total_layers)):
        layer_scores = global_scores[:, layer_ids.eq(layer)]
        within_square.add_((layer_scores @ layer_scores.T).square().sum())
    coupling = torch.sqrt(
        (total_square - within_square).clamp_min(0.0)
        / total_square.clamp_min(torch.finfo(total_square.dtype).tiny)
    )
    return ReplicatedFixedProbeTransactionResult(
        coefficients=coefficient_packet[:coordinates],
        candidate_coefficients=coefficient_packet[coordinates:],
        sharded_result=sharded,
        local_probe_count=FIXED_GLOBAL_PROBE_COUNT // max(world, 1),
        global_probe_count=FIXED_GLOBAL_PROBE_COUNT,
        cross_layer_coupling_ratio=coupling,
        collective_rounds=int(gather_rounds) + sharded.collective_rounds
        + int(world > 1),
        score_scalars_exchanged_per_rank=FIXED_GLOBAL_PROBE_COUNT
        * (coordinates + 1),
        coefficient_scalars_exchanged_per_rank=(
            2 * coordinates if world > 1 else 0
        ),
        selected_update_elements_published=0,
        method_state_depends_on_total_tokens=False,
    )


class LaggedPredictiveResponseTransactionRouter(GlobalResponseTransactionRouter):
    """Choose current response-parent coefficients from pre-update history."""

    family_id = FAMILY_ID
    telemetry_prefix = "lagged_predictive_response_transaction_"
    fairness_component = "lagged_predictive_response_transaction_lr_scale"

    def __init__(self, pairs, **kwargs):
        beta2 = float(kwargs.get("beta2", float("nan")))
        if beta2 != 0.95:
            raise ValueError("predictive response transaction requires locked beta2=.95")
        super().__init__(pairs, **kwargs)
        self.predictive_beta2 = beta2
        self.param_groups[0][
            "lagged_predictive_response_transaction_family_id"
        ] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({
            "matched_beta2_predictive_loss_image_lr_scale": 1.0,
            "preupdate_history_selection_lr_scale": 1.0,
        })
        return report

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("predictive response transaction lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("predictive response transaction would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("predictive response transaction refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        packets = self._consume_probes()
        participation, factors, response_adjoint = self._response_sensor_with_cache(
            packets
        )
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError("predictive response transaction omitted congruence")

        role_parameters = {}
        role_selected = {}
        role_adjustment = {}
        role_records = {}
        exact = []
        momentum_descent = []
        for role, index, axis in (
            ("incoming", 0, "rows"),
            ("outgoing", 1, "columns"),
        ):
            key = "in_weight" if role == "incoming" else "out_weight"
            parameters = [pair[key] for pair in self.pairs]
            momenta = _foreach_nesterov(self, parameters)
            gradients = torch.stack([
                parameter.grad.detach() for parameter in parameters
            ]).float()
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = compact_postpolar_group_response_homotopy(
                parent,
                momenta,
                gradients,
                participation[..., index],
                congruence[..., index],
                groups=self.groups,
                width=self.width,
                grouped_axis=axis,
            )
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            if role == "incoming":
                selected_blocks = selected.view(
                    len(self.pairs), self.groups, self.width, self.external
                )
                gradient_blocks = gradients.view_as(selected_blocks)
                momentum_blocks = momenta.view_as(selected_blocks)
            else:
                selected_blocks = selected.view(
                    len(self.pairs), self.external, self.groups, self.width
                ).permute(0, 2, 3, 1)
                gradient_blocks = gradients.view_as(selected).view(
                    len(self.pairs), self.external, self.groups, self.width
                ).permute(0, 2, 3, 1)
                momentum_blocks = momenta.view(
                    len(self.pairs), self.external, self.groups, self.width
                ).permute(0, 2, 3, 1)
            scaled_blocks = selected_blocks.float() * adjustment
            exact.append(
                (gradient_blocks.float() * scaled_blocks).sum(dim=(-2, -1))
            )
            momentum_descent.append(
                (momentum_blocks.float() * scaled_blocks).sum(dim=(-2, -1))
            )
            role_parameters[role] = parameters
            role_selected[role] = selected
            role_adjustment[role] = adjustment
            role_records[role] = metadata

        incoming_blocks = role_selected["incoming"].view(
            len(self.pairs), self.groups, self.width, self.external
        ).float() * role_adjustment["incoming"]
        outgoing_blocks = role_selected["outgoing"].view(
            len(self.pairs), self.external, self.groups, self.width
        ).permute(0, 2, 3, 1).float() * role_adjustment["outgoing"]
        weights = (
            incoming_blocks.square().sum(dim=(-2, -1))
            + outgoing_blocks.square().sum(dim=(-2, -1))
        ).reshape(-1)

        local_scores = []
        local_decay = None
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            inputs, preactivations, features, cotangents = packet
            layer_factors = tuple(value[layer] for value in factors)
            score, _ = _one_layer_group_scores(
                inputs,
                preactivations,
                features,
                cotangents,
                incoming_blocks[layer].reshape(self.hidden, self.external),
                outgoing_blocks[layer].reshape(self.hidden, self.external),
                pair["out_weight"],
                layer_factors,
                groups=self.groups,
                width=self.width,
                cached_response_adjoint=response_adjoint[layer],
            )
            decay_score, _ = _one_layer_group_scores(
                inputs,
                preactivations,
                features,
                cotangents,
                pair["in_weight"].float() * weight_decay,
                pair["out_weight"].T.float() * weight_decay,
                pair["out_weight"],
                layer_factors,
                groups=self.groups,
                width=self.width,
                cached_response_adjoint=response_adjoint[layer],
            )
            local_scores.append(score)
            layer_decay = decay_score.sum(dim=-1)
            local_decay = (
                layer_decay if local_decay is None else local_decay + layer_decay
            )
        if local_decay is None:
            raise RuntimeError("predictive response transaction omitted decay action")
        score_lattice = torch.stack(local_scores, dim=1).reshape(
            self.probe_layout.local_probe_count,
            len(self.pairs) * self.groups,
        )
        current_packet = torch.cat((score_lattice, local_decay[:, None]), dim=1)
        global_packet, gather_rounds = _gather_variable_probe_rows(
            current_packet,
            expected_global_rows=FIXED_GLOBAL_PROBE_COUNT,
            group=self.loss_probe_group,
        )
        current_global_scores = global_packet[:, :-1]
        current_global_decay = global_packet[:, -1]
        exact_by_role = torch.stack([value.reshape(-1) for value in exact])
        momentum_by_role = torch.stack([
            value.reshape(-1) for value in momentum_descent
        ])
        layer_ids = torch.arange(
            len(self.pairs), device=weights.device, dtype=torch.int64
        ).repeat_interleave(self.groups)

        anchor = self.state[self.pairs[0]["in_weight"]]
        predictive = matched_beta2_predictive_rows(
            current_global_scores,
            current_global_decay,
            anchor.get("predictive_global_score_ema"),
            anchor.get("predictive_global_decay_ema"),
            beta2=self.predictive_beta2,
        )
        selection = _transaction_from_replicated_global_rows(
            predictive.selection_scores,
            predictive.selection_decay_action,
            exact_by_role,
            momentum_by_role,
            weights,
            layer_ids,
            total_layers=len(self.pairs),
            eta=lr,
            gather_rounds=gather_rounds,
            group=self.loss_probe_group,
        )
        anchor["predictive_global_score_ema"] = (
            predictive.updated_scores.detach().clone()
        )
        anchor["predictive_global_decay_ema"] = (
            predictive.updated_decay_action.detach().clone()
        )
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)

        incoming_selected = role_selected["incoming"]
        incoming_selected.view(
            len(self.pairs), self.groups, self.width, self.external
        ).mul_(coefficients[..., None, None].to(incoming_selected.dtype))
        outgoing_selected = role_selected["outgoing"]
        outgoing_selected.view(
            len(self.pairs), self.external, self.groups, self.width
        ).permute(0, 2, 3, 1).mul_(
            coefficients[..., None, None].to(outgoing_selected.dtype)
        )
        for role in ("incoming", "outgoing"):
            _foreach_apply(
                role_parameters[role],
                role_selected[role],
                decay=1.0 - lr * weight_decay,
                alpha=-lr * role_adjustment[role],
            )

        updates = int(anchor.get("predictive_response_transaction_updates", 0)) + 1
        anchor["predictive_response_transaction_updates"] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            transaction = selection.sharded_result
            flat = coefficients.reshape(-1)
            response_cosine = torch.cat([
                role_records[role]["parent_cosine"].reshape(-1)
                for role in ("incoming", "outgoing")
            ])
            response_safe = torch.cat([
                role_records[role]["safe"].reshape(-1)
                for role in ("incoming", "outgoing")
            ])
            scaling = lagged_predictive_response_transaction_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            prefix = "lagged_predictive_response_transaction_"
            self._last_telemetry = {
                prefix + "family_id": FAMILY_ID,
                prefix + "parent_family_id": COMPACT_PARENT_FAMILY_ID,
                prefix + "owner_count": transaction.owner_count,
                prefix + "global_rows": FIXED_GLOBAL_PROBE_COUNT,
                prefix + "coordinate_count": len(self.pairs) * self.groups,
                prefix + "state_coordinate_count": scaling["persistent_state_elements"],
                prefix + "predictive_state_elements": scaling["predictive_state_elements"],
                prefix + "state_depends_on_total_tokens": 0,
                prefix + "summary_elements": scaling["communicated_summary_elements"],
                prefix + "largest_dense_solve_dimension": FIXED_GLOBAL_PROBE_COUNT,
                prefix + "dense_lg_metric_elements": transaction.dense_LG_by_LG_metric_elements,
                prefix + "selected_update_elements_published": transaction.selected_update_elements_published,
                prefix + "transaction_accepted": int(transaction.accepted.item()),
                prefix + "history_used": int(predictive.history_used),
                prefix + "matched_beta2": self.predictive_beta2,
                prefix + "relative_score_innovation": float(
                    predictive.relative_innovation.item()
                ),
                prefix + "rank": int(transaction.rank.item()),
                prefix + "budget_residual": float(transaction.budget_residual.item()),
                prefix + "coefficient_min": float(flat.amin().item()),
                prefix + "coefficient_median": float(flat.median().item()),
                prefix + "coefficient_max": float(flat.amax().item()),
                prefix + "cross_layer_coupling_ratio": float(
                    selection.cross_layer_coupling_ratio.item()
                ),
                prefix + "response_parent_cosine_median": float(
                    response_cosine.median().item()
                ),
                prefix + "response_safe_fraction": float(
                    response_safe.float().mean().item()
                ),
                prefix + "realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss


class LaggedPredictiveResponseTransactionAttentionOptimizer(
    GlobalResponseTransactionAttentionOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0][
            "lagged_predictive_response_transaction_family_id"
        ] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.pop("global_response_transaction_attention_lr_scale")
        report["lagged_predictive_response_attention_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for key, value in tuple(self._last_telemetry.items()):
            if value == CURRENT_IMPLEMENTATION_FAMILY_ID:
                self._last_telemetry[key] = FAMILY_ID
        if self._last_telemetry:
            self._last_telemetry[
                "lagged_predictive_response_transaction_attention_family_id"
            ] = FAMILY_ID
        return loss


__all__ = (
    "FAMILY_ID",
    "LaggedPredictiveResponseTransactionAttentionOptimizer",
    "LaggedPredictiveResponseTransactionRouter",
    "MatchedBeta2PredictiveRows",
    "lagged_predictive_response_transaction_scaling_formula",
    "matched_beta2_predictive_rows",
)
