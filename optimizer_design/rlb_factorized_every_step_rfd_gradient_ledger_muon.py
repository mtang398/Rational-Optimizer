"""Compiled factorized directions with an every-step Robust-FD loss ledger."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from . import rlb_factorized_adaptive_tangent_chord_compiled_muon as _parent
from . import rlb_factorized_adaptive_tangent_chord_muon as _factorized
from . import rlb_rank64_cadence8_response_group_polar_muon as _cadence
from .rlb_diagonal_completed_biatlas_metric import (
    MAXIMUM_SELECTION_ROWS,
    diagonal_completed_biatlas_transaction,
)
from .rlb_every_step_rfd_gradient_ledger import (
    CURRENT_ROWS,
    PERSISTENT_ROWS,
    every_step_rfd_gradient_rows,
    functional_row_norm,
    trace_matched_gradient_surrogate,
)
from .rlb_fd_tail_biatlas_metric import FDTailBiatlasRows
from .rlb_fixed_probe_transaction import (
    FixedProbeTransactionResult,
    ReplicatedFixedProbeTransactionResult,
)


FAMILY_ID = "factorized_every_step_rfd_gradient_ledger_muon_v1"
PREFIX = "factorized_every_step_rfd_gradient_ledger_"
MATCHED_BETA2 = 0.95


def _retag(report: dict) -> dict:
    result = {}
    for key, value in report.items():
        if key.startswith(PREFIX):
            pass
        elif key.startswith(_parent.PREFIX):
            key = PREFIX + key[len(_parent.PREFIX):]
        elif key.startswith(_factorized.PREFIX):
            key = PREFIX + key[len(_factorized.PREFIX):]
        if value in (_parent.FAMILY_ID, _factorized.FAMILY_ID):
            value = FAMILY_ID
        result[key] = value
    return result


def factorized_every_step_rfd_gradient_ledger_scaling_formula(**kwargs):
    result = dict(
        _parent.factorized_adaptive_tangent_chord_compiled_scaling_formula(
            **kwargs
        )
    )
    layers = int(kwargs["total_layers"])
    groups = int(kwargs["total_groups"])
    coordinates = layers * groups
    if coordinates < 2 or coordinates % 2:
        raise ValueError("factorized RFD ledger requires an even LG inventory")
    # The checkpoint carries a rank-64 factor, exact represented diagonal,
    # exact decay cross, reference norm, isotropic tail, and two diagnostics.
    ledger_state = PERSISTENT_ROWS * coordinates + 2 * coordinates + 4
    result.update({
        "family_id": FAMILY_ID,
        "coordinate_count": coordinates,
        "gradient_ledger_mathematical_state_elements": ledger_state,
        "additional_persistent_state_elements": ledger_state,
        "persistent_state_elements": (
            int(result["persistent_state_elements"]) + ledger_state
        ),
        "maximum_live_factor_elements": MAXIMUM_SELECTION_ROWS * coordinates,
        "every_step_gradient_score_ledger": 1,
        "trace_matched_gradient_surrogate": 1,
        "functional_score_refresh_interval": 8,
        "matched_beta2_every_optimizer_step": 1,
        "robust_fd_midpoint_tail": 1,
        "factorized_parameter_direction_unchanged": 1,
        "adaptive_rank64_cross_coordinate_factor": 1,
        "largest_dense_solve_dimension": MAXIMUM_SELECTION_ROWS,
        "largest_transaction_dense_dimension": 32,
        "largest_temporal_dense_dimension": MAXIMUM_SELECTION_ROWS,
        "state_depends_on_total_activation_positions": 0,
        "owner_count": 0,
        "complete_layer_owners": 0,
        "complete_coordinate_owners": 0,
        "owner_local_mathematics": 0,
        "dense_lg_by_lg_metric_elements": 0,
        "selected_update_elements_published": 0,
        "new_tunable_hyperparameters": 0,
        "state_scales_as": "O(LH + LGd + 64LG)",
    })
    return result


def _previous(anchor, reference: torch.Tensor):
    scores = anchor.get("factorized_rfd_persistent_scores")
    diagonal = anchor.get("factorized_rfd_persistent_total_diagonal")
    decay = anchor.get("factorized_rfd_persistent_decay_cross")
    tail = anchor.get("factorized_rfd_persistent_isotropic_tail")
    values = (scores, diagonal, decay, tail)
    if all(value is None for value in values):
        return None, None, None, None
    if (
        any(not torch.is_tensor(value) for value in values)
        or scores.ndim != 2
        or int(scores.shape[0]) not in (CURRENT_ROWS, PERSISTENT_ROWS)
        or diagonal.shape != scores.shape[1:]
        or decay.shape != scores.shape[1:]
        or tail.numel() != 1
        or any(value.dtype != reference.dtype for value in values)
        or any(value.device != reference.device for value in values)
        or any(not bool(torch.isfinite(value).all()) for value in values)
        or not bool((diagonal >= 0.0).all())
        or not bool((tail >= 0.0).all())
    ):
        raise RuntimeError("factorized RFD checkpoint inventory changed")
    return scores, diagonal, decay, tail


def _store(anchor, rows: FDTailBiatlasRows) -> None:
    anchor["factorized_rfd_persistent_scores"] = (
        rows.persistent_scores.detach().clone()
    )
    anchor["factorized_rfd_persistent_total_diagonal"] = (
        rows.persistent_total_diagonal.detach().clone()
    )
    anchor["factorized_rfd_persistent_decay_cross"] = (
        rows.persistent_decay_cross.detach().clone()
    )
    anchor["factorized_rfd_persistent_isotropic_tail"] = (
        rows.persistent_isotropic_tail.detach().clone()
    )
    anchor["factorized_rfd_discarded_energy_fraction"] = (
        rows.discarded_energy_fraction.detach().clone()
    )
    anchor["factorized_rfd_shrinkage"] = rows.fd_shrinkage.detach().clone()


class FactorizedEveryStepRFDGradientLedgerRouter(
    _parent.FactorizedAdaptiveTangentChordCompiledRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = "factorized_every_step_rfd_gradient_ledger_lr_scale"

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0].pop(_parent.PREFIX + "family_id", None)
        self.param_groups[0].pop(_factorized.PREFIX + "family_id", None)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID
        self._rfd_rows = None
        self._rfd_functional_refresh = False
        self._rfd_gradient_scale = None
        self._rfd_decay_derivative = None
        self._rfd_selection = None

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "every_step_gradient_score_ledger_lr_scale": 1.0,
            "trace_matched_gradient_surrogate_lr_scale": 1.0,
            "robust_fd_midpoint_tail_lr_scale": 1.0,
            "matrix_free_rfd_krylov_transaction_lr_scale": 1.0,
            "unchanged_compiled_factorized_direction_lr_scale": 1.0,
            "signed_rfd_coefficients_lr_scale": 1.0,
        })
        return result

    def _advance_rows(self, scores, decay_action, *, functional: bool):
        anchor = self.state[self.pairs[0]["in_weight"]]
        previous_scores, previous_diagonal, previous_decay, previous_tail = (
            _previous(anchor, scores)
        )
        rows = every_step_rfd_gradient_rows(
            scores,
            decay_action,
            previous_scores,
            previous_diagonal,
            previous_decay,
            beta2=MATCHED_BETA2,
            previous_isotropic_tail=previous_tail,
        )
        _store(anchor, rows)
        step = int(anchor.get("factorized_rfd_step", 0)) + 1
        anchor["factorized_rfd_step"] = step
        if functional:
            anchor["factorized_rfd_reference_row_norm"] = (
                functional_row_norm(scores).detach().clone()
            )
            scale = scores.new_ones(())
        else:
            scale = self._rfd_gradient_scale
            if scale is None:
                raise RuntimeError("factorized RFD gradient scale is absent")
        self._rfd_rows = rows
        self._rfd_functional_refresh = bool(functional)
        self._rfd_gradient_scale = scale
        return rows

    def predictive_rows_fn(
        self,
        current_scores,
        current_decay_action,
        previous_scores,
        previous_decay_action,
        *,
        beta2,
    ):
        if float(beta2) != MATCHED_BETA2:
            raise ValueError("factorized RFD ledger requires matched beta2=.95")
        inherited = super().predictive_rows_fn(
            current_scores,
            current_decay_action,
            previous_scores,
            previous_decay_action,
            beta2=beta2,
        )
        self._advance_rows(
            current_scores, current_decay_action, functional=True
        )
        return inherited

    def _transaction(
        self,
        exact_by_role,
        momentum_by_role,
        weights,
        layer_ids,
        *,
        total_layers,
        eta,
        rounds=64,
    ):
        if self._rfd_rows is None:
            anchor = self.state[self.pairs[0]["in_weight"]]
            reference = anchor.get("factorized_rfd_reference_row_norm")
            decay = self._rfd_decay_derivative
            if not torch.is_tensor(reference) or not torch.is_tensor(decay):
                raise RuntimeError("factorized RFD lacks functional calibration")
            scores, decay_action, scale = trace_matched_gradient_surrogate(
                exact_by_role, decay, reference
            )
            self._rfd_gradient_scale = scale
            self._advance_rows(scores, decay_action, functional=False)
        rows = self._rfd_rows
        if rows is None:
            raise RuntimeError("factorized RFD rows are absent")
        result = diagonal_completed_biatlas_transaction(
            rows,
            exact_by_role,
            momentum_by_role,
            weights,
            layer_ids,
            total_layers=int(total_layers),
            eta=float(eta),
            rounds=int(rounds),
        )
        self._rfd_selection = result
        coordinates = int(weights.numel())
        summary_elements = int(rows.selection_scores.numel()) + 4 * coordinates
        sharded = FixedProbeTransactionResult(
            local_coefficients=result.coefficients,
            local_candidate_coefficients=result.candidate_coefficients,
            accepted=result.accepted,
            multiplier=result.multiplier,
            rank=result.factor_rank,
            eigenvalue_max=result.diagonal_maximum,
            hard_case=result.hard_case,
            parent_score=result.parent_score,
            candidate_score=result.candidate_score,
            budget_residual=result.budget_residual,
            local_coordinate_count=coordinates,
            global_coordinate_count=coordinates,
            global_probe_count=CURRENT_ROWS,
            collective_rounds=0,
            summary_elements=summary_elements,
            owner_count=0,
            dense_LG_by_LG_metric_elements=0,
            selected_update_elements_published=0,
            method_state_depends_on_total_tokens=False,
        )
        return ReplicatedFixedProbeTransactionResult(
            coefficients=result.coefficients,
            candidate_coefficients=result.candidate_coefficients,
            sharded_result=sharded,
            local_probe_count=CURRENT_ROWS,
            global_probe_count=CURRENT_ROWS,
            cross_layer_coupling_ratio=result.cross_layer_coupling_ratio,
            collective_rounds=0,
            score_scalars_exchanged_per_rank=0,
            coefficient_scalars_exchanged_per_rank=0,
            selected_update_elements_published=0,
            method_state_depends_on_total_tokens=False,
        )

    def transaction_fn(
        self,
        global_scores,
        global_decay_action,
        exact_by_role,
        momentum_by_role,
        weights,
        layer_ids,
        *,
        total_layers,
        eta,
        gather_rounds,
        group,
    ):
        del global_scores, global_decay_action, gather_rounds, group
        result = self._transaction(
            exact_by_role,
            momentum_by_role,
            weights,
            layer_ids,
            total_layers=total_layers,
            eta=eta,
        )
        self._cadence_transaction = result
        return result

    @contextmanager
    def _installed_ordinary_transaction(self):
        original = _cadence.rank64_transaction_from_replicated_rows

        def transaction(
            global_scores,
            global_decay_action,
            exact_by_role,
            momentum_by_role,
            weights,
            layer_ids,
            *,
            total_layers,
            eta,
            gather_rounds,
            group,
        ):
            del global_scores, global_decay_action, gather_rounds, group
            result = self._transaction(
                exact_by_role,
                momentum_by_role,
                weights,
                layer_ids,
                total_layers=total_layers,
                eta=eta,
            )
            self._cadence_transaction = result
            return result

        with _factorized._PATCH_LOCK:
            if _cadence.rank64_transaction_from_replicated_rows is not original:
                raise RuntimeError("factorized RFD transaction binding changed")
            _cadence.rank64_transaction_from_replicated_rows = transaction
            try:
                yield
            finally:
                _cadence.rank64_transaction_from_replicated_rows = original

    @torch.no_grad()
    def step(self, closure=None):
        self._rfd_rows = None
        self._rfd_functional_refresh = False
        self._rfd_gradient_scale = None
        self._rfd_selection = None
        decay = torch.zeros(
            (), device=self.pairs[0]["in_weight"].device, dtype=torch.float32
        )
        for pair in self.pairs:
            for key in ("in_weight", "out_weight"):
                parameter = pair[key]
                if parameter.grad is None:
                    raise RuntimeError("factorized RFD parameter lacks gradient")
                decay.add_((
                    parameter.grad.detach().float()
                    * parameter.detach().float()
                ).sum())
        self._rfd_decay_derivative = (
            decay * float(self.param_groups[0]["weight_decay"])
        )
        with self._installed_ordinary_transaction():
            loss = super().step(closure)
        rows = self._rfd_rows
        selection = self._rfd_selection
        scale = self._rfd_gradient_scale
        if rows is None or selection is None or scale is None:
            raise RuntimeError("factorized RFD did not advance exactly once")
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry:
            scaling = factorized_every_step_rfd_gradient_ledger_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            self._last_telemetry.update({
                PREFIX + "family_id": FAMILY_ID,
                PREFIX + "factorized_parameter_direction_unchanged": 1,
                PREFIX + "every_step_gradient_score_ledger": 1,
                PREFIX + "functional_score_refresh": int(
                    self._rfd_functional_refresh
                ),
                PREFIX + "trace_matched_gradient_surrogate": int(
                    not self._rfd_functional_refresh
                ),
                PREFIX + "gradient_score_scale": float(scale.item()),
                PREFIX + "matched_beta2_every_optimizer_step": 1,
                PREFIX + "ledger_step": int(
                    self.state[self.pairs[0]["in_weight"]][
                        "factorized_rfd_step"
                    ]
                ),
                PREFIX + "robust_fd_midpoint_tail": 1,
                PREFIX + "signed_coefficients_allowed": 1,
                PREFIX + "selection_factor_rows": int(
                    rows.selection_scores.shape[0]
                ),
                PREFIX + "persistent_factor_rows": int(
                    rows.persistent_scores.shape[0]
                ),
                PREFIX + "persistent_isotropic_tail": float(
                    rows.persistent_isotropic_tail.item()
                ),
                PREFIX + "fd_shrinkage": float(rows.fd_shrinkage.item()),
                PREFIX + "cross_layer_coupling_ratio": float(
                    selection.cross_layer_coupling_ratio.item()
                ),
                PREFIX + "transaction_accepted": int(selection.accepted.item()),
                PREFIX + "state_coordinate_count": scaling[
                    "persistent_state_elements"
                ],
                PREFIX + "gradient_ledger_mathematical_state_elements": scaling[
                    "gradient_ledger_mathematical_state_elements"
                ],
                PREFIX + "largest_dense_solve_dimension": (
                    MAXIMUM_SELECTION_ROWS
                ),
                PREFIX + "largest_transaction_dense_dimension": 32,
                PREFIX + "largest_temporal_dense_dimension": (
                    MAXIMUM_SELECTION_ROWS
                ),
                PREFIX + "dense_lg_metric_elements": 0,
                PREFIX + "owner_count": 0,
                PREFIX + "selected_update_elements_published": 0,
                PREFIX + "new_tunable_hyperparameters": 0,
            })
        self._rfd_decay_derivative = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        anchor = self.state[self.pairs[0]["in_weight"]]
        scores = anchor.get("factorized_rfd_persistent_scores")
        _previous(anchor, scores if torch.is_tensor(scores) else self.pairs[0]["in_weight"])
        reference = anchor.get("factorized_rfd_reference_row_norm")
        step = anchor.get("factorized_rfd_step")
        if (
            not torch.is_tensor(reference)
            or reference.numel() != 1
            or type(step) is not int
            or step < 1
        ):
            raise RuntimeError("factorized RFD reference checkpoint is absent")
        self._rfd_rows = None
        self._rfd_gradient_scale = None
        self._rfd_selection = None
        self._rfd_decay_derivative = None
        return result


class FactorizedEveryStepRFDGradientLedgerAttentionOptimizer(
    _parent.FactorizedAdaptiveTangentChordCompiledAttentionOptimizer
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0].pop(_parent.PREFIX + "family_id", None)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["unchanged_compiled_factorized_attention_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry:
            self._last_telemetry.update({
                PREFIX + "family_id": FAMILY_ID,
                PREFIX + "attention_family_id": FAMILY_ID,
                PREFIX + "attention_equation_unchanged": 1,
                PREFIX + "attention_owner_count": 0,
                PREFIX + "attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "FAMILY_ID",
    "PREFIX",
    "FactorizedEveryStepRFDGradientLedgerAttentionOptimizer",
    "FactorizedEveryStepRFDGradientLedgerRouter",
    "factorized_every_step_rfd_gradient_ledger_scaling_formula",
)
