"""Current/posterior consensus response metric with group-polar Muon.

The current Global-RLB loss image adapts quickly but its lead can fade; the
rank-64 posterior factor preserves discounted geometry but can lag a changing
loss landscape.  This method gives the trust transaction both views with no
tuned interpolation: it stacks each covariance factor with equal information
weight.  Exact row normalization makes the selected metric

    1/2 * (B_posterior^T B_posterior + S_current^T S_current) / 32.

Only the rank-64 posterior factor is persistent.  The current 32 rows already
exist for the update, so state remains 64*(LG+1), independent of N.  The
largest selection solve is fixed 96-dimensional and all matrix directions
stay on their native shards.
"""

from __future__ import annotations

import math

import torch

from .rlb_lagged_predictive_response_transaction_muon import (
    MatchedBeta2PredictiveRows,
)
from .rlb_posterior_rank64_response_group_polar_muon import (
    FAMILY_ID as RANK64_PARENT_FAMILY_ID,
    MAXIMUM_AUGMENTED_ROWS,
    PERSISTENT_ROWS,
    PosteriorRank64HeadGroupPolarAttentionOptimizer,
    PosteriorRank64ResponseGroupPolarRouter,
    posterior_rank64_response_rows,
    posterior_rank64_scaling_formula,
)


FAMILY_ID = "consensus_rank64_response_group_polar_muon_v1"


def consensus_rank64_response_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    """Select from equally weighted current and updated-posterior metrics."""

    posterior = posterior_rank64_response_rows(
        current_scores,
        current_decay_action,
        previous_scores,
        previous_decay_action,
        beta2=beta2,
    )
    equal_information = 1.0 / math.sqrt(2.0)
    selection_scores = torch.cat((
        posterior.updated_scores * equal_information,
        current_scores * equal_information,
    ))
    selection_decay = torch.cat((
        posterior.updated_decay_action * equal_information,
        current_decay_action * equal_information,
    ))
    if int(selection_scores.shape[0]) not in (
        2 * int(current_scores.shape[0]),
        MAXIMUM_AUGMENTED_ROWS,
    ):
        raise RuntimeError("consensus response selection inventory changed")
    return MatchedBeta2PredictiveRows(
        selection_scores=selection_scores,
        selection_decay_action=selection_decay,
        updated_scores=posterior.updated_scores,
        updated_decay_action=posterior.updated_decay_action,
        history_used=posterior.history_used,
        relative_innovation=posterior.relative_innovation,
    )


def consensus_rank64_scaling_formula(**kwargs) -> dict[str, int]:
    result = dict(posterior_rank64_scaling_formula(**kwargs))
    layers = int(kwargs["total_layers"])
    groups = int(kwargs["total_groups"])
    coordinates = layers * groups
    selection_rows = MAXIMUM_AUGMENTED_ROWS
    result["selection_rows"] = selection_rows
    result["largest_dense_solve_dimension"] = selection_rows
    result["communicated_summary_elements"] = (
        21 * coordinates + 10 * layers
        + 32 * (coordinates + 1)
        + selection_rows * selection_rows
        + 3 * selection_rows + 8 * layers + 8
        + 2 * coordinates
    )
    return result


class ConsensusRank64ResponseGroupPolarRouter(
    PosteriorRank64ResponseGroupPolarRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = "consensus_rank64_response_group_polar_"
    fairness_component = "consensus_rank64_response_group_polar_lr_scale"
    predictive_rows_fn = staticmethod(consensus_rank64_response_rows)

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0][
            "consensus_rank64_response_group_polar_family_id"
        ] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report["current_posterior_consensus_metric_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            old = "posterior_rank64_response_group_polar_"
            new = "consensus_rank64_response_group_polar_"
            renamed = {
                key.replace(old, new, 1): (
                    FAMILY_ID
                    if value == RANK64_PARENT_FAMILY_ID
                    else value
                )
                for key, value in self._last_telemetry.items()
            }
            scaling = consensus_rank64_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            realized = (
                2 * 32
                if int(renamed[new + "realized_factor_rows"]) == 32
                else MAXIMUM_AUGMENTED_ROWS
            )
            renamed.update({
                new + "family_id": FAMILY_ID,
                new + "current_posterior_equal_information": 1,
                new + "selection_rows": realized,
                new + "largest_dense_solve_dimension": realized,
                new + "summary_elements": scaling[
                    "communicated_summary_elements"
                ],
            })
            self._last_telemetry = renamed
        return loss


class ConsensusRank64HeadGroupPolarAttentionOptimizer(
    PosteriorRank64HeadGroupPolarAttentionOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0][
            "consensus_rank64_response_group_polar_family_id"
        ] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report["consensus_rank64_attention_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value == RANK64_PARENT_FAMILY_ID:
                    self._last_telemetry[key] = FAMILY_ID
            self._last_telemetry.update({
                "consensus_rank64_response_group_polar_attention_family_id": (
                    FAMILY_ID
                ),
                "consensus_rank64_response_group_polar_attention_owner_count": 0,
                "consensus_rank64_response_group_polar_attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "FAMILY_ID",
    "ConsensusRank64HeadGroupPolarAttentionOptimizer",
    "ConsensusRank64ResponseGroupPolarRouter",
    "consensus_rank64_response_rows",
    "consensus_rank64_scaling_formula",
)
