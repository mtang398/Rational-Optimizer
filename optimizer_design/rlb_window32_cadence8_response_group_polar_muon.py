"""Current-window Global-RLB response geometry with cadence-eight reuse.

The rank-64 posterior cadence experiment showed that a temporally accumulated
response covariance drives the globally selected coefficients toward the
all-ones parent faster than the successful Method-3 trajectory.  This method
keeps the same owner-free fixed-32 response measurement and the same fresh
current-gradient transaction on every step, but uses only the most recently
observed response factor during each eight-step cycle.  Thus old covariance
cannot smear current loss information.  No update tensor is cached: native
rational/head-group directions are rebuilt on every transition.

Persistent state is 32*(LG+1)+O(LG), independent of activation positions N.
The largest response/transaction algebra is fixed at 32 dimensions.
"""

from __future__ import annotations

import torch

from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_lagged_predictive_response_transaction_muon import (
    MatchedBeta2PredictiveRows,
)
from .rlb_rank64_cadence8_response_group_polar_muon import (
    MATCHED_BETA2,
    REFRESH_INTERVAL,
    POSTERIOR_FAMILY_ID as PARENT_FAMILY_ID,
    PosteriorRank64Cadence8GroupPolarRouter,
    PosteriorRank64Cadence8HeadPolarAttentionOptimizer,
)


FAMILY_ID = "window32_cadence8_response_group_polar_muon_v1"
PREFIX = "window32_cadence8_response_group_polar_"
WINDOW_ROWS = FIXED_GLOBAL_PROBE_COUNT


def current_window32_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    """Use the latest response covariance without temporal mixing."""

    if (
        current_scores.ndim != 2
        or current_scores.shape[0] != WINDOW_ROWS
        or current_scores.shape[1] < 1
        or current_decay_action.shape != (WINDOW_ROWS,)
        or current_decay_action.dtype != current_scores.dtype
        or current_decay_action.device != current_scores.device
        or not current_scores.is_floating_point()
        or float(beta2) != MATCHED_BETA2
        or not bool(torch.isfinite(current_scores).all())
        or not bool(torch.isfinite(current_decay_action).all())
    ):
        raise RuntimeError("window32 current response inventory changed")
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError("window32 histories must coinitialize")

    updated_scores = current_scores.detach().clone()
    updated_decay = current_decay_action.detach().clone()
    if previous_scores is None:
        innovation = current_scores.new_zeros(())
        history_used = False
    else:
        assert previous_decay_action is not None
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
            raise RuntimeError("window32 checkpoint inventory changed")
        current_gram = current_scores @ current_scores.T
        previous_gram = previous_scores @ previous_scores.T
        cross_gram = current_scores @ previous_scores.T
        tiny = torch.finfo(current_scores.dtype).tiny
        innovation = torch.sqrt(
            (
                current_gram.square().sum()
                + previous_gram.square().sum()
                - 2.0 * cross_gram.square().sum()
            ).clamp_min(0.0)
            / current_gram.square().sum().clamp_min(tiny)
        )
        history_used = True
    torch._assert_async(torch.isfinite(innovation))
    return MatchedBeta2PredictiveRows(
        selection_scores=updated_scores,
        selection_decay_action=updated_decay,
        updated_scores=updated_scores,
        updated_decay_action=updated_decay,
        history_used=history_used,
        relative_innovation=innovation,
    )


def window32_cadence8_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
) -> dict[str, int | float]:
    values = tuple(map(int, (
        total_positions,
        total_layers,
        total_groups,
        intermediate_width,
        model_width,
    )))
    if min(values) <= 0 or int(intermediate_width) % int(total_groups):
        raise ValueError("window32 scaling dimensions are invalid")
    positions, layers, groups, hidden, model = values
    coordinates = layers * groups
    parent_state = 10 * coordinates + 2
    factor_state = WINDOW_ROWS * (coordinates + 1)
    route_state = 4 * coordinates + 2 * layers
    response_summary = 21 * coordinates + 10 * layers
    transaction_summary = (
        WINDOW_ROWS * WINDOW_ROWS
        + 3 * WINDOW_ROWS + 8 * layers + 8
    )
    refresh_score_summary = WINDOW_ROWS * (coordinates + 1)
    return {
        "total_positions": positions,
        "persistent_state_elements": parent_state + factor_state + route_state,
        "window_factor_elements": factor_state,
        "cached_response_route_elements": route_state,
        "communicated_summary_elements": (
            response_summary + refresh_score_summary
            + transaction_summary + 2 * coordinates
        ),
        "ordinary_communicated_summary_elements": (
            transaction_summary + 2 * coordinates
        ),
        "largest_temporal_dense_dimension": WINDOW_ROWS,
        "largest_dense_solve_dimension": WINDOW_ROWS,
        "dense_coordinate_metric_elements": 0,
        "owner_count": 0,
        "selected_update_elements_published": 0,
        "response_refresh_interval": REFRESH_INTERVAL,
        "matched_beta2": MATCHED_BETA2,
        "local_direction_arithmetic_elements": 4 * layers * hidden * model,
    }


class Window32Cadence8ResponseGroupPolarRouter(
    PosteriorRank64Cadence8GroupPolarRouter
):
    metric_rows_fn = staticmethod(current_window32_rows)
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = "window32_cadence8_response_group_polar_lr_scale"

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "current_window32_response_metric_lr_scale": 1.0,
            "no_temporal_covariance_smearing_lr_scale": 1.0,
        })
        return result

    def _rename_refresh_telemetry(self):
        super()._rename_refresh_telemetry()
        scaling = window32_cadence8_scaling_formula(
            total_positions=1,
            total_layers=len(self.pairs),
            total_groups=self.groups,
            intermediate_width=self.hidden,
            model_width=self.external,
        )
        prefix = self.telemetry_prefix
        self._last_telemetry.update({
            prefix + "family_id": FAMILY_ID,
            prefix + "state_coordinate_count": scaling[
                "persistent_state_elements"
            ],
            prefix + "predictive_state_elements": scaling[
                "window_factor_elements"
            ],
            prefix + "summary_elements": scaling[
                "communicated_summary_elements"
            ],
            prefix + "largest_dense_solve_dimension": WINDOW_ROWS,
            prefix + "largest_temporal_dense_dimension": WINDOW_ROWS,
            prefix + "persistent_rank_limit": WINDOW_ROWS,
            prefix + "realized_factor_rows": WINDOW_ROWS,
            prefix + "selection_uses_current_window": 1,
            prefix + "temporal_covariance_mixing": 0,
        })


class Window32Cadence8HeadPolarAttentionOptimizer(
    PosteriorRank64Cadence8HeadPolarAttentionOptimizer
):
    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["window32_cadence8_attention_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        old = "posterior_rank64_cadence8_group_polar_"
        self._last_telemetry = {
            key.replace(old, PREFIX, 1): (
                FAMILY_ID if value == PARENT_FAMILY_ID else value
            )
            for key, value in self._last_telemetry.items()
        }
        if self._last_telemetry:
            self._last_telemetry.update({
                PREFIX + "attention_family_id": FAMILY_ID,
                PREFIX + "attention_owner_count": 0,
                PREFIX + "attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "FAMILY_ID",
    "PREFIX",
    "WINDOW_ROWS",
    "Window32Cadence8HeadPolarAttentionOptimizer",
    "Window32Cadence8ResponseGroupPolarRouter",
    "current_window32_rows",
    "window32_cadence8_scaling_formula",
)
