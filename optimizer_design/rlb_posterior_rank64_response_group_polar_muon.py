"""Rank-64 posterior response-Fisher subspace with group-polar Muon.

Each update contributes the campaign's fixed 32 exact Global-RLB response
loss rows.  A rotation-invariant truncated SVD retains the best rank-64
factor of the matched-beta2 discounted covariance, rather than shrinking and
discarding half of a 64-row augmentation every step.  The just-updated factor
selects the current signed equality-budget transaction.

The persistent factor has at most 64*(LG+1) scalars, independent of activation
positions N.  Its largest temporal eigensystem is fixed 96-by-96 and its
largest trust solve is fixed 64-dimensional.  Only scalar score/coefficient
summaries cross shards; no layer owner or selected matrix publication exists.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_fixed_probe_transaction import (
    ReplicatedFixedProbeTransactionResult,
    distributed_fixed_probe_transaction,
)
from .rlb_lagged_predictive_response_transaction_muon import (
    MatchedBeta2PredictiveRows,
)
from .rlb_global_response_transaction_muon import (
    global_response_transaction_scaling_formula,
)
from .rlb_temporal_response_group_polar_muon import (
    TemporalResponseGroupPolarRouter,
    TemporalResponseHeadGroupPolarAttentionOptimizer,
)


FAMILY_ID = "posterior_rank64_response_group_polar_muon_v1"
PERSISTENT_ROWS = 64
MAXIMUM_AUGMENTED_ROWS = PERSISTENT_ROWS + FIXED_GLOBAL_PROBE_COUNT


def posterior_rank64_response_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    """Return the best rank-64 factor of the updated discounted covariance."""

    if current_scores.ndim != 2:
        raise RuntimeError("rank64 posterior current-row inventory changed")
    rows, coordinates = current_scores.shape
    if (
        rows != FIXED_GLOBAL_PROBE_COUNT
        or coordinates < 1
        or current_decay_action.shape != (rows,)
        or current_decay_action.dtype != current_scores.dtype
        or current_decay_action.device != current_scores.device
        or not current_scores.is_floating_point()
        or float(beta2) != 0.95
        or not bool(torch.isfinite(current_scores).all())
        or not bool(torch.isfinite(current_decay_action).all())
    ):
        raise RuntimeError("rank64 posterior current-row inventory changed")
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError("rank64 posterior histories must coinitialize")
    if previous_scores is None:
        return MatchedBeta2PredictiveRows(
            selection_scores=current_scores,
            selection_decay_action=current_decay_action,
            updated_scores=current_scores.detach().clone(),
            updated_decay_action=current_decay_action.detach().clone(),
            history_used=False,
            relative_innovation=current_scores.new_zeros(()),
        )

    assert previous_decay_action is not None
    previous_rows = int(previous_scores.shape[0])
    if (
        previous_scores.ndim != 2
        or previous_scores.shape[1] != coordinates
        or previous_rows not in (FIXED_GLOBAL_PROBE_COUNT, PERSISTENT_ROWS)
        or previous_decay_action.shape != (previous_rows,)
        or previous_scores.dtype != current_scores.dtype
        or previous_decay_action.dtype != current_scores.dtype
        or previous_scores.device != current_scores.device
        or previous_decay_action.device != current_scores.device
        or not bool(torch.isfinite(previous_scores).all())
        or not bool(torch.isfinite(previous_decay_action).all())
    ):
        raise RuntimeError("rank64 posterior checkpoint inventory changed")

    beta = float(beta2)
    augmented = torch.cat((
        previous_scores * math.sqrt(beta),
        current_scores * math.sqrt(1.0 - beta),
    ))
    augmented_decay = torch.cat((
        previous_decay_action * math.sqrt(beta),
        current_decay_action * math.sqrt(1.0 - beta),
    ))
    row_gram = augmented @ augmented.T
    row_gram = 0.5 * (row_gram + row_gram.T)
    _eigenvalues, eigenvectors = torch.linalg.eigh(row_gram)
    retained_rows = min(PERSISTENT_ROWS, int(augmented.shape[0]))
    transform = eigenvectors.flip(1)[:, :retained_rows].T
    updated_scores = transform @ augmented
    updated_decay = transform @ augmented_decay

    current_gram = current_scores @ current_scores.T
    previous_gram = previous_scores @ previous_scores.T
    cross_gram = current_scores @ previous_scores.T
    tiny = torch.finfo(current_scores.dtype).tiny
    innovation2 = (
        current_gram.square().sum()
        + previous_gram.square().sum()
        - 2.0 * cross_gram.square().sum()
    ).clamp_min(0.0)
    relative_innovation = torch.sqrt(
        innovation2 / current_gram.square().sum().clamp_min(tiny)
    )
    torch._assert_async(
        torch.isfinite(updated_scores).all()
        & torch.isfinite(updated_decay).all()
        & torch.isfinite(relative_innovation)
    )
    return MatchedBeta2PredictiveRows(
        selection_scores=updated_scores,
        selection_decay_action=updated_decay,
        updated_scores=updated_scores,
        updated_decay_action=updated_decay,
        history_used=True,
        relative_innovation=relative_innovation,
    )


def rank64_transaction_from_replicated_rows(
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
    """Column-shard a replicated fixed-rank posterior score factor."""

    rows, coordinates = global_scores.shape
    if (
        rows not in (
            FIXED_GLOBAL_PROBE_COUNT,
            PERSISTENT_ROWS,
            MAXIMUM_AUGMENTED_ROWS,
        )
        or global_decay_action.shape != (rows,)
        or exact_by_role.shape != (2, coordinates)
        or momentum_by_role.shape != exact_by_role.shape
        or weights.shape != (coordinates,)
        or layer_ids.shape != (coordinates,)
    ):
        raise RuntimeError("rank64 replicated transaction inventory changed")
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
    else:
        rank, world = 0, 1
    coordinate_ids = torch.arange(
        coordinates, device=global_scores.device, dtype=torch.int64
    )
    local_ids = coordinate_ids[coordinate_ids.remainder(world).eq(rank)]
    # B is a factor of an EMA of 32-row batch score covariances.  Growing the
    # factor rank must not silently change the empirical-measure denominator
    # from 32 to 64.  The generic transaction divides by its physical row
    # count, so scale both paired factors by sqrt(rows/32).
    measure_scale = math.sqrt(float(rows) / float(FIXED_GLOBAL_PROBE_COUNT))
    transaction_scores = global_scores * measure_scale
    transaction_decay = global_decay_action * measure_scale
    decay_cross = (
        transaction_scores.T @ transaction_decay / float(rows)
    )
    sharded = distributed_fixed_probe_transaction(
        transaction_scores[:, local_ids],
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
        local_probe_count=rows // max(world, 1),
        global_probe_count=rows,
        cross_layer_coupling_ratio=coupling,
        collective_rounds=int(gather_rounds) + sharded.collective_rounds
        + int(world > 1),
        score_scalars_exchanged_per_rank=rows * (coordinates + 1),
        coefficient_scalars_exchanged_per_rank=(
            2 * coordinates if world > 1 else 0
        ),
        selected_update_elements_published=0,
        method_state_depends_on_total_tokens=False,
    )


def posterior_rank64_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
) -> dict[str, int]:
    values = tuple(map(int, (
        total_positions, total_layers, total_groups,
        intermediate_width, model_width,
    )))
    if min(values) <= 0 or int(intermediate_width) % int(total_groups):
        raise ValueError("rank64 scaling dimensions are invalid")
    positions, layers, groups, hidden, model = values
    coordinates = layers * groups
    parent = global_response_transaction_scaling_formula(
        total_positions=positions,
        total_layers=layers,
        total_groups=groups,
        intermediate_width=hidden,
        model_width=model,
    )
    response_summary = 21 * coordinates + 10 * layers
    persistent = PERSISTENT_ROWS * (coordinates + 1)
    transaction_summary = (
        PERSISTENT_ROWS * PERSISTENT_ROWS
        + 3 * PERSISTENT_ROWS + 8 * layers + 8
    )
    return {
        "total_positions": positions,
        "persistent_state_elements": (
            parent["persistent_state_elements"] + persistent
        ),
        "posterior_factor_elements": persistent,
        "communicated_summary_elements": (
            response_summary
            + FIXED_GLOBAL_PROBE_COUNT * (coordinates + 1)
            + transaction_summary + 2 * coordinates
        ),
        "largest_temporal_dense_dimension": MAXIMUM_AUGMENTED_ROWS,
        "largest_dense_solve_dimension": PERSISTENT_ROWS,
        "dense_coordinate_metric_elements": 0,
        "owner_count": 0,
        "selected_update_elements_published": 0,
        "local_direction_arithmetic_elements": 4 * layers * hidden * model,
    }


class PosteriorRank64ResponseGroupPolarRouter(
    TemporalResponseGroupPolarRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = "posterior_rank64_response_group_polar_"
    fairness_component = "posterior_rank64_response_group_polar_lr_scale"
    predictive_rows_fn = staticmethod(posterior_rank64_response_rows)
    transaction_fn = staticmethod(rank64_transaction_from_replicated_rows)

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0][
            "posterior_rank64_response_group_polar_family_id"
        ] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({
            "posterior_rank64_subspace_lr_scale": 1.0,
            "truncated_svd_response_metric_lr_scale": 1.0,
        })
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            old = "temporal_response_group_polar_"
            new = "posterior_rank64_response_group_polar_"
            renamed = {
                key.replace(old, new, 1): (
                    FAMILY_ID
                    if value == TemporalResponseGroupPolarRouter.family_id
                    else value
                )
                for key, value in self._last_telemetry.items()
            }
            scaling = posterior_rank64_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            factor = self.state[self.pairs[0]["in_weight"]].get(
                "predictive_global_score_ema"
            )
            renamed.update({
                new + "family_id": FAMILY_ID,
                new + "selection_uses_updated_metric": 1,
                new + "persistent_rank_limit": PERSISTENT_ROWS,
                new + "realized_factor_rows": int(factor.shape[0]),
                new + "state_coordinate_count": scaling[
                    "persistent_state_elements"
                ],
                new + "predictive_state_elements": scaling[
                    "posterior_factor_elements"
                ],
                new + "summary_elements": scaling[
                    "communicated_summary_elements"
                ],
                new + "largest_dense_solve_dimension": PERSISTENT_ROWS,
                new + "largest_temporal_dense_dimension": (
                    MAXIMUM_AUGMENTED_ROWS
                ),
            })
            self._last_telemetry = renamed
        return loss


class PosteriorRank64HeadGroupPolarAttentionOptimizer(
    TemporalResponseHeadGroupPolarAttentionOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0][
            "posterior_rank64_response_group_polar_family_id"
        ] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report["posterior_rank64_attention_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value == TemporalResponseGroupPolarRouter.family_id:
                    self._last_telemetry[key] = FAMILY_ID
            self._last_telemetry.update({
                "posterior_rank64_response_group_polar_attention_family_id": (
                    FAMILY_ID
                ),
                "posterior_rank64_response_group_polar_attention_owner_count": 0,
                "posterior_rank64_response_group_polar_attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "FAMILY_ID",
    "MAXIMUM_AUGMENTED_ROWS",
    "PERSISTENT_ROWS",
    "PosteriorRank64HeadGroupPolarAttentionOptimizer",
    "PosteriorRank64ResponseGroupPolarRouter",
    "posterior_rank64_response_rows",
    "posterior_rank64_scaling_formula",
    "rank64_transaction_from_replicated_rows",
)
