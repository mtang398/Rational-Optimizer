"""Frequent-Directions score Fisher with a full-rank isotropic tail.

The fixed-rank temporal factor retains the changing leading loss-score
eigenspace.  Standard Frequent-Directions shrinkage supplies, without a
tuned damping scalar, the isotropic completion needed to prevent discarded
directions from becoming a zero-curvature nullspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist

from .rlb_diagonal_completed_biatlas_metric import (
    CURRENT_ROWS,
    EFFECTIVE_REFRESH_BETA2,
    MATCHED_BETA2,
    MAXIMUM_SELECTION_ROWS,
    PERSISTENT_ROWS,
)


@dataclass(frozen=True)
class FDTailBiatlasRows:
    """Live and persistent factors plus their isotropic completions."""

    selection_scores: torch.Tensor
    selection_diagonal_residual: torch.Tensor
    persistent_scores: torch.Tensor
    persistent_total_diagonal: torch.Tensor
    persistent_diagonal_residual: torch.Tensor
    persistent_decay_cross: torch.Tensor
    history_used: bool
    discarded_energy_fraction: torch.Tensor
    selection_isotropic_tail: torch.Tensor
    persistent_isotropic_tail: torch.Tensor
    fd_shrinkage: torch.Tensor


def _validate_current(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
) -> int:
    if (
        current_scores.ndim != 2
        or int(current_scores.shape[0]) != CURRENT_ROWS
        or int(current_scores.shape[1]) < 2
        or int(current_scores.shape[1]) % 2
        or current_decay_action.shape != (CURRENT_ROWS,)
        or not current_scores.is_floating_point()
        or current_decay_action.dtype != current_scores.dtype
        or current_decay_action.device != current_scores.device
        or not bool(torch.isfinite(current_scores).all())
        or not bool(torch.isfinite(current_decay_action).all())
    ):
        raise RuntimeError("FD-tail current score inventory changed")
    return int(current_scores.shape[1])


def _recover_tail(
    previous_scores: torch.Tensor,
    previous_total_diagonal: torch.Tensor,
) -> torch.Tensor:
    represented = (
        previous_scores.double().square().sum(dim=0) / float(CURRENT_ROWS)
    )
    residual = (
        previous_total_diagonal.double() - represented
    ).clamp_min(0.0)
    # The state is isotropic in exact arithmetic.  Median recovery only
    # removes coordinatewise roundoff introduced by FP32 checkpoint storage.
    tail = residual.median()
    scale = torch.maximum(
        represented.amax(), tail.abs()
    ).clamp_min(1.0)
    tolerance = (
        256.0 * torch.finfo(previous_total_diagonal.dtype).eps * scale
    )
    torch._assert_async(
        torch.isfinite(tail)
        & (tail >= 0.0)
        & ((residual - tail).abs().amax() <= tolerance)
    )
    return tail


def fd_tail_biatlas_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_total_diagonal: torch.Tensor | None,
    previous_decay_cross: torch.Tensor | None,
    *,
    beta2: float,
    previous_isotropic_tail: torch.Tensor | None = None,
) -> FDTailBiatlasRows:
    """Advance one elapsed-beta FD recurrence with covariance domination."""

    coordinates = _validate_current(current_scores, current_decay_action)
    if float(beta2) != MATCHED_BETA2:
        raise ValueError("FD-tail biatlas requires matched beta2=.95")
    missing = (
        previous_scores is None,
        previous_total_diagonal is None,
        previous_decay_cross is None,
    )
    if any(missing) and not all(missing):
        raise RuntimeError("FD-tail persistent state is partial")

    score64 = current_scores.double()
    decay64 = current_decay_action.double()
    current_cross = score64.T @ decay64 / float(CURRENT_ROWS)
    zero = current_scores.new_zeros((), dtype=torch.float64)

    if previous_scores is None:
        if previous_isotropic_tail is not None:
            raise RuntimeError("FD-tail scalar appeared before factor state")
        persistent = current_scores.detach().clone()
        represented = (
            persistent.double().square().sum(dim=0) / float(CURRENT_ROWS)
        )
        tail_vector = torch.zeros_like(represented)
        return FDTailBiatlasRows(
            selection_scores=current_scores,
            selection_diagonal_residual=tail_vector.to(current_scores.dtype),
            persistent_scores=persistent,
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
    previous_rows = int(previous_scores.shape[0])
    if (
        previous_scores.ndim != 2
        or previous_scores.shape[1] != coordinates
        or previous_rows not in (CURRENT_ROWS, PERSISTENT_ROWS)
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
    ):
        raise RuntimeError("FD-tail persistent inventory changed")
    if previous_isotropic_tail is None:
        previous_tail = _recover_tail(
            previous_scores, previous_total_diagonal
        )
    else:
        if (
            not torch.is_tensor(previous_isotropic_tail)
            or previous_isotropic_tail.numel() != 1
            or previous_isotropic_tail.device != current_scores.device
            or not previous_isotropic_tail.is_floating_point()
            or not bool(torch.isfinite(previous_isotropic_tail).all())
            or not bool((previous_isotropic_tail >= 0.0).all())
        ):
            raise RuntimeError("FD-tail scalar state changed")
        previous_tail = previous_isotropic_tail.double().reshape(())

    elapsed = EFFECTIVE_REFRESH_BETA2
    selection = torch.cat((
        previous_scores * math.sqrt(elapsed),
        current_scores * math.sqrt(1.0 - elapsed),
    ))
    if int(selection.shape[0]) > MAXIMUM_SELECTION_ROWS:
        raise RuntimeError("FD-tail live row bound changed")
    selection_tail = previous_tail * elapsed
    decay_cross = (
        previous_decay_cross.double() * elapsed
        + current_cross * (1.0 - elapsed)
    )

    row_gram = selection.double() @ selection.double().T
    row_gram = 0.5 * (row_gram + row_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(row_gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    physical_rows = int(eigenvalues.numel())
    retained_rows = min(PERSISTENT_ROWS, physical_rows)
    discarded_rows = physical_rows - retained_rows
    shrinkage = (
        eigenvalues[discarded_rows - 1]
        if discarded_rows else zero
    )
    retained_values = eigenvalues[-retained_rows:]
    transform = eigenvectors[:, -retained_rows:].T
    unshrunk = transform @ selection.double()
    multiplier = torch.sqrt(
        (retained_values - shrinkage).clamp_min(0.0)
        / retained_values.clamp_min(torch.finfo(torch.float64).tiny)
    )
    persistent64 = multiplier[:, None] * unshrunk
    persistent_tail = selection_tail + shrinkage / float(CURRENT_ROWS)
    represented = (
        persistent64.square().sum(dim=0) / float(CURRENT_ROWS)
    )
    total_diagonal = represented + persistent_tail
    tail_vector = torch.ones_like(total_diagonal) * persistent_tail
    selection_tail_vector = torch.ones_like(total_diagonal) * selection_tail
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
        persistent_diagonal_residual=tail_vector.to(current_scores.dtype),
        persistent_decay_cross=decay_cross.to(current_scores.dtype),
        history_used=True,
        discarded_energy_fraction=discarded_fraction.to(current_scores.dtype),
        selection_isotropic_tail=selection_tail.to(current_scores.dtype),
        persistent_isotropic_tail=persistent_tail.to(current_scores.dtype),
        fd_shrinkage=shrinkage.to(current_scores.dtype),
    )


def distributed_fd_tail_biatlas_rows(
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
    """Reduce arbitrary additive score fragments and advance the recurrence."""

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
    return fd_tail_biatlas_rows(
        packet[:score_count].view(CURRENT_ROWS, coordinates),
        packet[score_count:],
        previous_scores,
        previous_total_diagonal,
        previous_decay_cross,
        beta2=float(beta2),
        previous_isotropic_tail=previous_isotropic_tail,
    )


def fd_tail_biatlas_scaling_formula(
    *, total_positions: int, total_layers: int, total_groups: int
) -> dict[str, int | str]:
    positions, layers, groups = map(
        int, (total_positions, total_layers, total_groups)
    )
    if min(positions, layers, groups) < 1:
        raise ValueError("FD-tail scaling dimensions must be positive")
    coordinates = 2 * layers * groups
    return {
        "coordinate_count": coordinates,
        "persistent_factor_elements": PERSISTENT_ROWS * coordinates,
        "persistent_decay_cross_elements": coordinates,
        "persistent_isotropic_tail_elements": 1,
        "persistent_state_elements": (
            PERSISTENT_ROWS * coordinates + coordinates + 1
        ),
        "maximum_live_factor_elements": MAXIMUM_SELECTION_ROWS * coordinates,
        "largest_dense_solve_dimension": MAXIMUM_SELECTION_ROWS,
        "largest_krylov_dimension": CURRENT_ROWS,
        "dense_lg_by_lg_metric_elements": 0,
        "owner_count": 0,
        "complete_layer_owners": 0,
        "complete_coordinate_owners": 0,
        "selected_update_elements_published": 0,
        "method_specific_activation_position_state_elements": 0,
        "state_depends_on_total_activation_positions": 0,
        "state_scales_as": "O(LG) at fixed inherited row bounds",
    }


__all__ = (
    "FDTailBiatlasRows",
    "distributed_fd_tail_biatlas_rows",
    "fd_tail_biatlas_rows",
    "fd_tail_biatlas_scaling_formula",
)
