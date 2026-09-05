"""Persistent learned-response score geometry on complete R08.

Complete R08 already exposes one exact score vector for every learned
P5/Q4 paired-radial atlas direction,

    s_t[n,j] = <e_t[n,l], Y_t[n,j]>,  j=(layer, group),

and uses the current-batch empirical Fisher ``F_t = E[s_t s_t^T]`` in its
global same-budget transaction.  The completed R01/R08 trajectories show
that this 648-coordinate cross-layer geometry is useful, while subsequent
larger current-batch atlases have not improved its durable endpoint.

R03 changes only the statistical object used by that existing transaction.
It keeps the bias-corrected exponential mean of the exact P5/Q4 score Fisher
and its weight-decay cross term,

    M_t = beta2 M_{t-1} + (1-beta2) F_t,
    C_t = beta2 C_{t-1} + (1-beta2) c_t,
    Fbar_t = M_t / (1-beta2**t),
    cbar_t = C_t / (1-beta2**t).

``beta2`` is the comparison cell's already fixed value ``0.95``.  It is not
a new cadence or tunable scalar.  At step one the bias correction makes R03
exactly equal to complete R08.  Thereafter every observation is an exact
installed-RLB functional score, so the persistent metric can acquire rank
and cross-layer support across batches instead of solving each update in a
rank-at-most-144 one-batch geometry.  The literal R08 direction remains the
first feasible point of the inherited equal-budget transaction.  LR, WD,
momentum, NS5, clipping, data order, and every update budget are unchanged.
"""

from __future__ import annotations

import torch

from .rlb_r08_next_core import R08NextCore


class R03Core(R08NextCore):
    """Complete R08 with a persistent exact-P5/Q4 score Fisher."""

    component_code = 36
    checkpoint_schema = "r03_r08_persistent_p5_q4_score_fisher_v1"
    inherited_parent = "complete_r08_radial_natural_atlas"
    new_scientific_components = (
        "persistent_exact_p5_q4_cross_layer_score_fisher",
    )

    def __init__(self, pairs, **kwargs):
        self._r03_persistent_metadata = None
        super().__init__(pairs, **kwargs)
        if float(self._r05_beta2) != 0.95:
            raise ValueError("R03 persistent Fisher must use matched beta2=0.95")

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "persistent_p5_q4_score_fisher_lr_scale": 1.0,
            "persistent_p5_q4_score_fisher_weight_decay_scale": 1.0,
            "persistent_parent_budget_lr_scale": 1.0,
        })
        return report

    @staticmethod
    def _update_persistent_metric(
        current_fisher,
        current_cross,
        moment,
        cross_moment,
        step,
        *,
        beta2,
    ):
        """Apply one exact bias-corrected EMA recurrence without mutation."""
        if (
            current_fisher.ndim != 3
            or current_fisher.shape[0] != 1
            or current_fisher.shape[-1] != current_fisher.shape[-2]
            or current_cross.shape != current_fisher.shape[:-1]
            or moment.shape != current_fisher.shape
            or cross_moment.shape != current_cross.shape
            or type(step) is not int
            or step < 0
            or float(beta2) != 0.95
        ):
            raise RuntimeError("R03 persistent metric recurrence inventory changed")
        next_moment = moment * beta2 + current_fisher * (1.0 - beta2)
        next_cross_moment = (
            cross_moment * beta2 + current_cross * (1.0 - beta2)
        )
        next_step = step + 1
        correction = 1.0 - beta2 ** next_step
        persistent_fisher = next_moment / correction
        persistent_cross = next_cross_moment / correction
        if step == 0:
            # Preserve the literal parent transaction bit-for-bit on the
            # first update; the stored moments still follow the recurrence.
            persistent_fisher = current_fisher
            persistent_cross = current_cross
        persistent_fisher = 0.5 * (
            persistent_fisher + persistent_fisher.transpose(-2, -1)
        )
        return (
            persistent_fisher,
            persistent_cross,
            next_moment,
            next_cross_moment,
            next_step,
        )

    def _reduce_global_loss_metric(self, images, cotangents, group_decay_images):
        """Replace the one-batch score metric by its matched-beta2 mean."""
        current_fisher, current_cross, count = super()._reduce_global_loss_metric(
            images, cotangents, group_decay_images
        )
        if (
            current_fisher.ndim != 3
            or current_fisher.shape[0] != 1
            or current_fisher.shape[-1] != current_fisher.shape[-2]
            or current_cross.shape != current_fisher.shape[:-1]
        ):
            raise RuntimeError("R03 score-Fisher inventory changed")
        parent_dimension = len(self.pairs) * self.groups
        atlas_dimension = 2 * parent_dimension
        if current_fisher.shape[-1] == parent_dimension:
            # The inherited R01 transaction is part of literal complete R08.
            # Persistence belongs only to R08's paired-radial atlas.
            return current_fisher, current_cross, count
        if current_fisher.shape[-1] != atlas_dimension:
            raise RuntimeError("R03 paired-radial atlas dimension changed")

        anchor = self.state[self.incoming[0]]
        moment = anchor.get("r03_score_fisher_moment")
        cross_moment = anchor.get("r03_score_cross_moment")
        step = anchor.get("r03_score_fisher_step", 0)
        if type(step) is not int or step < 0:
            raise RuntimeError("R03 persistent-Fisher step is invalid")
        if moment is None:
            moment = torch.zeros_like(current_fisher)
            cross_moment = torch.zeros_like(current_cross)
        if (
            moment.shape != current_fisher.shape
            or cross_moment is None
            or cross_moment.shape != current_cross.shape
            or moment.dtype != current_fisher.dtype
            or cross_moment.dtype != current_cross.dtype
            or moment.device != current_fisher.device
            or cross_moment.device != current_cross.device
        ):
            raise RuntimeError("R03 persistent-Fisher checkpoint state changed")

        (
            persistent_fisher,
            persistent_cross,
            moment,
            cross_moment,
            next_step,
        ) = self._update_persistent_metric(
            current_fisher,
            current_cross,
            moment,
            cross_moment,
            step,
            beta2=float(self._r05_beta2),
        )

        publish = bool(self._capture_telemetry_next_step)
        current_norm = None
        persistent_norm = None
        drift = None
        if publish:
            current_norm = torch.linalg.vector_norm(current_fisher)
            persistent_norm = torch.linalg.vector_norm(persistent_fisher)
            drift = torch.linalg.vector_norm(
                persistent_fisher - current_fisher
            )
        torch._assert_async(
            torch.isfinite(persistent_fisher).all()
            & torch.isfinite(persistent_cross).all()
        )

        anchor["r03_score_fisher_moment"] = moment
        anchor["r03_score_cross_moment"] = cross_moment
        anchor["r03_score_fisher_step"] = next_step
        self._r03_persistent_metadata = {
            "step": next_step,
            "coordinate_count": int(current_fisher.shape[-1]),
            "current_norm": current_norm,
            "persistent_norm": persistent_norm,
            "relative_drift": (
                None
                if drift is None
                else drift / current_norm.clamp_min(
                    torch.finfo(current_fisher.dtype).tiny
                )
            ),
            "count": count,
        }
        return persistent_fisher, persistent_cross, count

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r03_persistent_metadata = None
        loss = super().step(closure)
        metadata = self._r03_persistent_metadata
        if metadata is None:
            raise RuntimeError("R03 persistent score Fisher did not execute")
        if publish:
            self._last_telemetry.update({
                "rlb_r03_component_code": self.component_code,
                "rlb_r03_parent_is_complete_r08": 1,
                "rlb_r03_persistent_p5_q4_score_fisher_enabled": 1,
                "rlb_r03_bias_corrected_matched_beta2_enabled": 1,
                "rlb_r03_global_coordinate_count": metadata[
                    "coordinate_count"
                ],
                "rlb_r03_global_loss_sample_count": int(
                    metadata["count"].item()
                ),
                "rlb_r03_structural_matrix_elements": 245_366_784,
                "rlb_r03_score_fisher_step": int(metadata["step"]),
                "rlb_r03_current_score_fisher_norm": float(
                    metadata["current_norm"].item()
                ),
                "rlb_r03_persistent_score_fisher_norm": float(
                    metadata["persistent_norm"].item()
                ),
                "rlb_r03_score_fisher_relative_drift": float(
                    metadata["relative_drift"].item()
                ),
            })
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r03_persistent_metadata = None
        return result


__all__ = ("R03Core",)
