"""Every-step trace-matched gradient scores with Robust Frequent Directions."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .rlb_diagonal_completed_biatlas_metric import (
    CURRENT_ROWS,
    MATCHED_BETA2,
    MAXIMUM_SELECTION_ROWS,
    PERSISTENT_ROWS,
)
from .rlb_fd_tail_biatlas_metric import FDTailBiatlasRows


def trace_matched_gradient_surrogate(
    exact_by_role: torch.Tensor,
    decay_derivative: torch.Tensor,
    reference_row_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift one exact batch-gradient score into a trace-matched 32-row factor."""

    if (
        exact_by_role.ndim != 2
        or exact_by_role.shape[0] != 2
        or exact_by_role.shape[1] < 2
        or exact_by_role.shape[1] % 2
        or decay_derivative.numel() != 1
        or reference_row_norm.numel() != 1
        or not exact_by_role.is_floating_point()
        or decay_derivative.dtype != exact_by_role.dtype
        or reference_row_norm.dtype != exact_by_role.dtype
        or decay_derivative.device != exact_by_role.device
        or reference_row_norm.device != exact_by_role.device
        or not bool(torch.isfinite(exact_by_role).all())
        or not bool(torch.isfinite(decay_derivative).all())
        or not bool(torch.isfinite(reference_row_norm).all())
        or not bool((reference_row_norm >= 0.0).all())
    ):
        raise RuntimeError("gradient-ledger surrogate inventory changed")
    gradient_score = exact_by_role.double().sum(dim=0)
    norm = torch.linalg.vector_norm(gradient_score)
    reference = reference_row_norm.double().reshape(())
    positive = norm > torch.finfo(torch.float64).tiny
    scale64 = torch.where(
        positive,
        reference / norm.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(norm),
    )
    score = (gradient_score * scale64).to(exact_by_role.dtype)
    decay = (decay_derivative.double().reshape(()) * scale64).to(
        exact_by_role.dtype
    )
    scores = score.unsqueeze(0).expand(CURRENT_ROWS, -1).clone()
    decay_action = decay.expand(CURRENT_ROWS).clone()
    torch._assert_async(
        torch.isfinite(scores).all()
        & torch.isfinite(decay_action).all()
        & torch.isfinite(scale64)
    )
    return scores, decay_action, scale64.to(exact_by_role.dtype)


def distributed_trace_matched_gradient_surrogate(
    partial_exact_by_role: torch.Tensor,
    partial_decay_derivative: torch.Tensor,
    reference_row_norm: torch.Tensor,
    *,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sum arbitrary additive parameter-fragment scores without an owner."""

    if (
        partial_exact_by_role.ndim != 2
        or partial_exact_by_role.shape[0] != 2
        or partial_decay_derivative.numel() != 1
    ):
        raise RuntimeError("distributed gradient-ledger inventory changed")
    packet = torch.cat((
        partial_exact_by_role.reshape(-1),
        partial_decay_derivative.reshape(1),
    )).clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packet, op=dist.ReduceOp.SUM, group=group)
    exact_count = partial_exact_by_role.numel()
    return trace_matched_gradient_surrogate(
        packet[:exact_count].view_as(partial_exact_by_role),
        packet[exact_count:].reshape(()),
        reference_row_norm,
    )


def functional_row_norm(current_scores: torch.Tensor) -> torch.Tensor:
    """RMS norm of one fixed functional-score row."""

    if (
        current_scores.ndim != 2
        or current_scores.shape[0] != CURRENT_ROWS
        or current_scores.shape[1] < 2
        or current_scores.shape[1] % 2
        or not current_scores.is_floating_point()
        or not bool(torch.isfinite(current_scores).all())
    ):
        raise RuntimeError("functional calibration rows changed")
    value = torch.sqrt(
        current_scores.double().square().sum() / float(CURRENT_ROWS)
    )
    torch._assert_async(torch.isfinite(value) & (value >= 0.0))
    return value.to(current_scores.dtype)


def _validate_current(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
) -> int:
    if (
        current_scores.ndim != 2
        or current_scores.shape[0] != CURRENT_ROWS
        or current_scores.shape[1] < 2
        or current_scores.shape[1] % 2
        or current_decay_action.shape != (CURRENT_ROWS,)
        or not current_scores.is_floating_point()
        or current_decay_action.dtype != current_scores.dtype
        or current_decay_action.device != current_scores.device
        or not bool(torch.isfinite(current_scores).all())
        or not bool(torch.isfinite(current_decay_action).all())
    ):
        raise RuntimeError("every-step RFD current-score inventory changed")
    return int(current_scores.shape[1])


def every_step_rfd_gradient_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_total_diagonal: torch.Tensor | None,
    previous_decay_cross: torch.Tensor | None,
    *,
    beta2: float,
    previous_isotropic_tail: torch.Tensor | None = None,
) -> FDTailBiatlasRows:
    """Advance a matched-beta Robust-FD factor on every optimizer step."""

    coordinates = _validate_current(current_scores, current_decay_action)
    if float(beta2) != MATCHED_BETA2:
        raise ValueError("gradient ledger requires matched beta2=.95")
    missing = (
        previous_scores is None,
        previous_total_diagonal is None,
        previous_decay_cross is None,
    )
    if any(missing) and not all(missing):
        raise RuntimeError("gradient-ledger persistent state is partial")

    score64 = current_scores.double()
    decay64 = current_decay_action.double()
    current_cross = score64.T @ decay64 / float(CURRENT_ROWS)
    zero = score64.new_zeros(())

    if previous_scores is None:
        if previous_isotropic_tail is not None:
            raise RuntimeError("gradient-ledger tail appeared before factor")
        persistent64 = score64.clone()
        represented = persistent64.square().sum(dim=0) / float(CURRENT_ROWS)
        tail_vector = torch.zeros_like(represented)
        return FDTailBiatlasRows(
            selection_scores=current_scores,
            selection_diagonal_residual=tail_vector.to(current_scores.dtype),
            persistent_scores=persistent64.to(current_scores.dtype),
            persistent_total_diagonal=represented.to(current_scores.dtype),
            persistent_diagonal_residual=tail_vector.to(current_scores.dtype),
            persistent_decay_cross=current_cross.to(current_scores.dtype),
            history_used=False,
            discarded_energy_fraction=zero.to(current_scores.dtype),
            selection_isotropic_tail=zero.to(current_scores.dtype),
            persistent_isotropic_tail=zero.to(current_scores.dtype),
            fd_shrinkage=zero.to(current_scores.dtype),
        )

    assert previous_total_diagonal is not None
    assert previous_decay_cross is not None
    rows = int(previous_scores.shape[0])
    if (
        previous_scores.ndim != 2
        or previous_scores.shape[1] != coordinates
        or rows not in (CURRENT_ROWS, PERSISTENT_ROWS)
        or previous_total_diagonal.shape != (coordinates,)
        or previous_decay_cross.shape != (coordinates,)
        or previous_scores.dtype != current_scores.dtype
        or previous_total_diagonal.dtype != current_scores.dtype
        or previous_decay_cross.dtype != current_scores.dtype
        or previous_scores.device != current_scores.device
        or previous_total_diagonal.device != current_scores.device
        or previous_decay_cross.device != current_scores.device
        or not bool(torch.isfinite(previous_scores).all())
        or not bool(torch.isfinite(previous_total_diagonal).all())
        or not bool(torch.isfinite(previous_decay_cross).all())
        or not bool((previous_total_diagonal >= 0.0).all())
        or previous_isotropic_tail is None
        or not torch.is_tensor(previous_isotropic_tail)
        or previous_isotropic_tail.numel() != 1
        or previous_isotropic_tail.device != current_scores.device
        or not previous_isotropic_tail.is_floating_point()
        or not bool(torch.isfinite(previous_isotropic_tail).all())
        or not bool((previous_isotropic_tail >= 0.0).all())
    ):
        raise RuntimeError("gradient-ledger persistent inventory changed")

    beta = float(beta2)
    previous_tail = previous_isotropic_tail.double().reshape(())
    selection = torch.cat((
        previous_scores * math.sqrt(beta),
        current_scores * math.sqrt(1.0 - beta),
    ))
    if int(selection.shape[0]) > MAXIMUM_SELECTION_ROWS:
        raise RuntimeError("gradient-ledger live row bound changed")
    selection_tail = previous_tail * beta
    decay_cross = (
        previous_decay_cross.double() * beta
        + current_cross * (1.0 - beta)
    )

    gram = selection.double() @ selection.double().T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    physical_rows = int(eigenvalues.numel())
    retained_rows = min(PERSISTENT_ROWS, physical_rows)
    discarded_rows = physical_rows - retained_rows
    shrinkage = (
        eigenvalues[discarded_rows - 1] if discarded_rows else zero
    )
    retained = eigenvalues[-retained_rows:]
    transform = eigenvectors[:, -retained_rows:].T
    unshrunk = transform @ selection.double()
    multiplier = torch.sqrt(
        (retained - shrinkage).clamp_min(0.0)
        / retained.clamp_min(torch.finfo(torch.float64).tiny)
    )
    persistent64 = multiplier[:, None] * unshrunk
    # Robust Frequent Directions is the spectral midpoint between the FD
    # lower covariance and its escaped-mass upper envelope.
    persistent_tail = selection_tail + shrinkage / (2.0 * float(CURRENT_ROWS))
    represented = persistent64.square().sum(dim=0) / float(CURRENT_ROWS)
    total_diagonal = represented + persistent_tail
    selection_tail_vector = torch.ones_like(represented) * selection_tail
    persistent_tail_vector = torch.ones_like(represented) * persistent_tail
    discarded_energy = (
        eigenvalues[:discarded_rows].sum() if discarded_rows else zero
    )
    discarded_fraction = discarded_energy / eigenvalues.sum().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    torch._assert_async(
        torch.isfinite(selection).all()
        & torch.isfinite(persistent64).all()
        & torch.isfinite(decay_cross).all()
        & torch.isfinite(persistent_tail)
        & torch.isfinite(shrinkage)
        & (persistent_tail >= 0.0)
        & (shrinkage >= 0.0)
    )
    return FDTailBiatlasRows(
        selection_scores=selection,
        selection_diagonal_residual=selection_tail_vector.to(
            current_scores.dtype
        ),
        persistent_scores=persistent64.to(current_scores.dtype),
        persistent_total_diagonal=total_diagonal.to(current_scores.dtype),
        persistent_diagonal_residual=persistent_tail_vector.to(
            current_scores.dtype
        ),
        persistent_decay_cross=decay_cross.to(current_scores.dtype),
        history_used=True,
        discarded_energy_fraction=discarded_fraction.to(current_scores.dtype),
        selection_isotropic_tail=selection_tail.to(current_scores.dtype),
        persistent_isotropic_tail=persistent_tail.to(current_scores.dtype),
        fd_shrinkage=shrinkage.to(current_scores.dtype),
    )


def distributed_every_step_rfd_gradient_rows(
    partial_current_scores: torch.Tensor,
    partial_current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_total_diagonal: torch.Tensor | None,
    previous_decay_cross: torch.Tensor | None,
    *,
    beta2: float,
    previous_isotropic_tail: torch.Tensor | None = None,
    group=None,
) -> FDTailBiatlasRows:
    """Reduce additive score fragments and advance the same recurrence."""

    coordinates = _validate_current(
        partial_current_scores, partial_current_decay_action
    )
    packet = torch.cat((
        partial_current_scores.reshape(-1),
        partial_current_decay_action,
    )).clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packet, op=dist.ReduceOp.SUM, group=group)
    score_count = CURRENT_ROWS * coordinates
    return every_step_rfd_gradient_rows(
        packet[:score_count].view(CURRENT_ROWS, coordinates),
        packet[score_count:],
        previous_scores,
        previous_total_diagonal,
        previous_decay_cross,
        beta2=float(beta2),
        previous_isotropic_tail=previous_isotropic_tail,
    )


def every_step_rfd_gradient_ledger_scaling_formula(
    *, total_positions: int, total_layers: int, total_groups: int
) -> dict[str, int | str]:
    positions, layers, groups = map(
        int, (total_positions, total_layers, total_groups)
    )
    if min(positions, layers, groups) < 1:
        raise ValueError("gradient-ledger scaling dimensions must be positive")
    coordinates = 2 * layers * groups
    return {
        "coordinate_count": coordinates,
        "persistent_factor_elements": PERSISTENT_ROWS * coordinates,
        "persistent_decay_cross_elements": coordinates,
        "persistent_isotropic_tail_elements": 1,
        "functional_trace_calibration_elements": 1,
        "mathematical_state_elements": (
            PERSISTENT_ROWS * coordinates + coordinates + 2
        ),
        "maximum_live_factor_elements": MAXIMUM_SELECTION_ROWS * coordinates,
        "largest_dense_solve_dimension": MAXIMUM_SELECTION_ROWS,
        "largest_krylov_dimension": CURRENT_ROWS,
        "dense_lg_by_lg_metric_elements": 0,
        "owner_count": 0,
        "complete_layer_owners": 0,
        "complete_coordinate_owners": 0,
        "owner_local_mathematics": 0,
        "selected_update_elements_published": 0,
        "method_specific_activation_position_state_elements": 0,
        "state_depends_on_total_activation_positions": 0,
        "state_scales_as": "O(LG) at fixed inherited row bounds",
    }


__all__ = (
    "distributed_every_step_rfd_gradient_rows",
    "distributed_trace_matched_gradient_surrogate",
    "every_step_rfd_gradient_ledger_scaling_formula",
    "every_step_rfd_gradient_rows",
    "functional_row_norm",
    "trace_matched_gradient_surrogate",
)
