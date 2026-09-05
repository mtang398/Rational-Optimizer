"""Parent-orthogonal Global-RLB response geometry with cadence-eight reuse.

The equal-update-budget transaction is a sphere in whitened coefficient
coordinates whose parent is the all-ones coefficient vector.  A response row
has inner product with that parent equal to the sum of its unwhitened logical
coordinate scores.  Subtracting each row's global coordinate mean therefore
projects the Global-RLB metric exactly onto the budget tangent at the parent.

This removes the common response mode that was driving current-window
coefficients toward a highly coupled all-ones transaction.  Global-RLB still
coordinates all layers and rational groups through their relative loss
responses, while current exact gradient/momentum terms control the accepted
equal-budget step.  Centering adds only 32 row sums and one count to a refresh;
state remains independent of activation positions N and no update is owned or
published.
"""

from __future__ import annotations

import torch

from .rlb_lagged_predictive_response_transaction_muon import (
    MatchedBeta2PredictiveRows,
)
from .rlb_window32_cadence8_response_group_polar_muon import (
    FAMILY_ID as WINDOW_PARENT_FAMILY_ID,
    PREFIX as WINDOW_PARENT_PREFIX,
    WINDOW_ROWS,
    Window32Cadence8HeadPolarAttentionOptimizer,
    Window32Cadence8ResponseGroupPolarRouter,
    current_window32_rows,
    window32_cadence8_scaling_formula,
)


FAMILY_ID = "budget_tangent_window32_cadence8_group_polar_muon_v1"
PREFIX = "budget_tangent_window32_cadence8_group_polar_"


def budget_tangent_window32_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    """Project each current response row off the parent coefficient mode."""

    if current_scores.ndim != 2 or current_scores.shape[1] < 2:
        raise RuntimeError("budget-tangent response coordinate inventory changed")
    centered = current_scores - current_scores.mean(dim=1, keepdim=True)
    if previous_scores is None:
        centered_previous = None
    else:
        if previous_scores.ndim != 2 or previous_scores.shape[1] < 2:
            raise RuntimeError("budget-tangent checkpoint inventory changed")
        centered_previous = previous_scores - previous_scores.mean(
            dim=1, keepdim=True
        )
    result = current_window32_rows(
        centered,
        current_decay_action,
        centered_previous,
        previous_decay_action,
        beta2=beta2,
    )
    torch._assert_async(torch.isfinite(result.selection_scores).all())
    return result


def budget_tangent_window32_scaling_formula(**kwargs):
    result = dict(window32_cadence8_scaling_formula(**kwargs))
    result["centering_summary_elements"] = WINDOW_ROWS + 1
    result["communicated_summary_elements"] += WINDOW_ROWS + 1
    result.update({
        "parent_response_mode_rank": 1,
        "response_metric_parent_overlap": 0,
        "additional_persistent_state_elements": 0,
        "dense_budget_projector_elements": 0,
    })
    return result


class BudgetTangentWindow32Cadence8GroupPolarRouter(
    Window32Cadence8ResponseGroupPolarRouter
):
    metric_rows_fn = staticmethod(budget_tangent_window32_rows)
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = "budget_tangent_window32_cadence8_group_polar_lr_scale"

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "parent_orthogonal_response_metric_lr_scale": 1.0,
            "budget_tangent_global_coordination_lr_scale": 1.0,
        })
        return result

    def _rename_refresh_telemetry(self):
        super()._rename_refresh_telemetry()
        scaling = budget_tangent_window32_scaling_formula(
            total_positions=1,
            total_layers=len(self.pairs),
            total_groups=self.groups,
            intermediate_width=self.hidden,
            model_width=self.external,
        )
        prefix = self.telemetry_prefix
        self._last_telemetry.update({
            prefix + "family_id": FAMILY_ID,
            prefix + "summary_elements": scaling[
                "communicated_summary_elements"
            ],
            prefix + "centering_summary_elements": scaling[
                "centering_summary_elements"
            ],
            prefix + "parent_response_mode_rank": 1,
            prefix + "response_metric_parent_overlap": 0,
            prefix + "additional_persistent_state_elements": 0,
            prefix + "dense_budget_projector_elements": 0,
            prefix + "selection_uses_budget_tangent_metric": 1,
        })


class BudgetTangentWindow32Cadence8HeadPolarAttentionOptimizer(
    Window32Cadence8HeadPolarAttentionOptimizer
):
    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["budget_tangent_window32_attention_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self._last_telemetry = {
            key.replace(WINDOW_PARENT_PREFIX, PREFIX, 1): (
                FAMILY_ID if value == WINDOW_PARENT_FAMILY_ID else value
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
    "BudgetTangentWindow32Cadence8GroupPolarRouter",
    "BudgetTangentWindow32Cadence8HeadPolarAttentionOptimizer",
    "budget_tangent_window32_rows",
    "budget_tangent_window32_scaling_formula",
)
