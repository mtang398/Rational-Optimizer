"""Owner-free global loss transaction on a four-role response parent.

The completed four-role response-homotopy experiment retains a positive
step-1000 lead but lacks the late cross-layer/group coordination visible in
the successful R01 family.  This method keeps that response-derived parent
exactly, evaluates its paired rational-group actions on the already fixed
global loss rows, and solves the equality-budget transaction in row space.

Only signed scalar coefficients are communicated and reconstructed on native
logical shards.  No complete layer, selected matrix update, activation-sized
state, or dense ``(L G) x (L G)`` metric is formed.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_batched_four_role_response_sensor import (
    stacked_intrinsic_and_response_statistics,
    stacked_response_adjoint,
    stacked_version_a_factors,
)
from .rlb_compact_four_role_response_homotopy_muon import (
    CompactFourRoleResponseHomotopyAttentionOptimizer,
    CompactFourRoleResponseHomotopyRouter,
    FAMILY_ID as COMPACT_PARENT_FAMILY_ID,
    compact_postpolar_group_response_homotopy,
)
from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_fixed_probe_transaction import replicated_fixed_probe_transaction
from .rlb_group_muon_core import _batched_zero_power
from .rlb_loss_weighted_four_role_response_homotopy_batched_muon import (
    _foreach_apply,
    _foreach_nesterov,
)
from .rlb_loss_weighted_four_role_response_homotopy_muon import (
    group_loss_weighted_response_congruence,
    layer_loss_weighted_response_congruence,
    method_state_elements as response_state_elements,
)
from .rlb_loss_weighted_four_role_sign_transport_muon import (
    group_intrinsic_participation,
)
from .rlb_ten_probe_loss_image_muon import _one_layer_group_scores


FAMILY_ID = "global_response_transaction_muon_v1"


def global_response_transaction_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
) -> dict[str, int]:
    """Closed-form persistent/communication inventory for any logical shape."""

    values = tuple(map(int, (
        total_positions,
        total_layers,
        total_groups,
        intermediate_width,
        model_width,
    )))
    if min(values) <= 0:
        raise ValueError("global response transaction dimensions must be positive")
    positions, layers, groups, hidden, model = values
    if hidden % groups:
        raise ValueError("intermediate width must be divisible by rational groups")
    coordinates = layers * groups
    # The response parent contributes 21LG+10L additive scalars.  The loss
    # transaction contributes R*LG scores, R decay scores, and fixed R^2
    # row-space algebra.  None depends on total activation positions N.
    rows = FIXED_GLOBAL_PROBE_COUNT
    return {
        "total_positions": positions,
        "persistent_state_elements": response_state_elements(
            layers=layers, groups=groups
        ),
        "communicated_summary_elements": (
            21 * coordinates + 10 * layers
            + rows * coordinates + rows + rows * rows
        ),
        "largest_dense_solve_dimension": rows,
        "dense_coordinate_metric_elements": 0,
        "owner_count": 0,
        "selected_update_elements_published": 0,
        "local_direction_arithmetic_elements": 4 * layers * hidden * model,
    }


class GlobalResponseTransactionRouter(CompactFourRoleResponseHomotopyRouter):
    """Coordinate the positive response parent with one signed global solve."""

    family_id = FAMILY_ID
    telemetry_prefix = "global_response_transaction_"
    fairness_component = "global_response_transaction_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["global_response_transaction_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({
            "fixed32_global_loss_measure_lr_scale": 1.0,
            "signed_global_group_transaction_lr_scale": 1.0,
            "row_space_equality_budget_lr_scale": 1.0,
        })
        return report

    def _response_sensor_with_cache(self, packets):
        """Evaluate the response parent once and retain local score factors."""

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
            raise RuntimeError("global response transaction frozen inventory changed")
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
            preactivations,
            numerators,
            denominators,
            groups=self.groups,
            width=self.width,
            eps=self.rlb_eps,
        )
        response_adjoint = stacked_response_adjoint(
            cotangents,
            outgoing,
            factors,
            groups=self.groups,
            width=self.width,
        )
        participation_statistics, response_statistics = (
            stacked_intrinsic_and_response_statistics(
                preactivations,
                cotangents,
                response_adjoint,
                factors,
                frozen_numerators,
                frozen_denominators,
                eps=self.rlb_eps,
            )
        )
        first = participation_statistics.numel()
        packed = torch.cat((
            participation_statistics.reshape(-1),
            response_statistics.reshape(-1),
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
        return participation, factors, response_adjoint

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("global response transaction lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("global response transaction would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("global response transaction refuses nonunit LR scale")
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
            raise RuntimeError("global response transaction omitted congruence")

        role_parameters = {}
        role_momenta = {}
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
            exact.append((gradient_blocks.float() * scaled_blocks).sum(dim=(-2, -1)))
            momentum_descent.append(
                (momentum_blocks.float() * scaled_blocks).sum(dim=(-2, -1))
            )
            role_parameters[role] = parameters
            role_momenta[role] = momenta
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
            raise RuntimeError("global response transaction omitted decay action")
        score_lattice = torch.stack(local_scores, dim=1).reshape(
            self.probe_layout.local_probe_count,
            len(self.pairs) * self.groups,
        )
        exact_by_role = torch.stack([value.reshape(-1) for value in exact])
        momentum_by_role = torch.stack([
            value.reshape(-1) for value in momentum_descent
        ])
        layer_ids = torch.arange(
            len(self.pairs), device=weights.device, dtype=torch.int64
        ).repeat_interleave(self.groups)
        selection = replicated_fixed_probe_transaction(
            score_lattice,
            local_decay,
            exact_by_role,
            momentum_by_role,
            weights,
            layer_ids,
            global_probe_count=FIXED_GLOBAL_PROBE_COUNT,
            total_layers=len(self.pairs),
            eta=lr,
            rounds=64,
            group=self.loss_probe_group,
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

        anchor = self.state[self.pairs[0]["in_weight"]]
        updates = int(anchor.get("global_response_transaction_updates", 0)) + 1
        anchor["global_response_transaction_updates"] = updates
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
            scaling = global_response_transaction_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            prefix = "global_response_transaction_"
            self._last_telemetry = {
                prefix + "family_id": FAMILY_ID,
                prefix + "parent_family_id": COMPACT_PARENT_FAMILY_ID,
                prefix + "owner_count": transaction.owner_count,
                prefix + "global_rows": FIXED_GLOBAL_PROBE_COUNT,
                prefix + "coordinate_count": len(self.pairs) * self.groups,
                prefix + "state_coordinate_count": scaling["persistent_state_elements"],
                prefix + "state_depends_on_total_tokens": 0,
                prefix + "summary_elements": scaling["communicated_summary_elements"],
                prefix + "largest_dense_solve_dimension": FIXED_GLOBAL_PROBE_COUNT,
                prefix + "dense_lg_metric_elements": transaction.dense_LG_by_LG_metric_elements,
                prefix + "selected_update_elements_published": transaction.selected_update_elements_published,
                prefix + "transaction_accepted": int(transaction.accepted.item()),
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


class GlobalResponseTransactionAttentionOptimizer(
    CompactFourRoleResponseHomotopyAttentionOptimizer
):
    """Retain the same response-derived attention route as the positive parent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]["global_response_transaction_family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report["global_response_transaction_attention_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for key, value in tuple(self._last_telemetry.items()):
            if value == COMPACT_PARENT_FAMILY_ID:
                self._last_telemetry[key] = FAMILY_ID
        if self._last_telemetry:
            self._last_telemetry[
                "global_response_transaction_attention_family_id"
            ] = FAMILY_ID
        return loss


__all__ = (
    "FAMILY_ID",
    "GlobalResponseTransactionAttentionOptimizer",
    "GlobalResponseTransactionRouter",
    "global_response_transaction_scaling_formula",
)
