"""Diagonal-completed fixed-rank loss geometry for Global-RLB biatlases.

The persistent biatlas factor is deliberately fixed rank, but a pure
low-rank approximation forgets every covariance direction discarded by the
row compression.  That is harmless early, when the empirical loss geometry
is naturally low rank, and becomes increasingly inaccurate as the
bias-corrected score Fisher accumulates support across refreshes.

This module retains the exact diagonal of that persistent Fisher in addition
to the rank-64 cross-coordinate factor.  If ``B`` is the carried factor and
``v`` is the exact EMA diagonal, the represented metric is

    B.T @ B / 32 + diag(clamp(v - diag(B.T @ B / 32), 0)).

At a refresh the uncompressed live selection factor contains the carried
factor and all 32 current score rows, while the diagonal residual contains
only the decayed residual of the previous cycle.  Consequently the selected
metric has the exact new EMA diagonal before any compression.  The decay
cross term is also carried exactly as an ``O(LG)`` vector rather than being
projected into the retained row space.

The equality-budget transaction solves a diagonal-plus-low-rank quadratic in
a fixed 32-vector loss-Krylov subspace.  It never forms a coordinate-by-
coordinate matrix.  The transaction eigensolve is at most 32 dimensional and
the refresh compression is at most 96 dimensional, independent of ``L``,
``G`` and total activation positions ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist


CURRENT_ROWS = 32
PERSISTENT_ROWS = 64
MAXIMUM_SELECTION_ROWS = 96
KRYLOV_ROWS = 32
REFRESH_INTERVAL = 8
MATCHED_BETA2 = 0.95
EFFECTIVE_REFRESH_BETA2 = MATCHED_BETA2 ** REFRESH_INTERVAL


@dataclass(frozen=True)
class DiagonalCompletedBiatlasRows:
    selection_scores: torch.Tensor
    selection_diagonal_residual: torch.Tensor
    persistent_scores: torch.Tensor
    persistent_total_diagonal: torch.Tensor
    persistent_diagonal_residual: torch.Tensor
    persistent_decay_cross: torch.Tensor
    history_used: bool
    discarded_energy_fraction: torch.Tensor


@dataclass(frozen=True)
class DiagonalCompletedTransactionResult:
    coefficients: torch.Tensor
    candidate_coefficients: torch.Tensor
    accepted: torch.Tensor
    multiplier: torch.Tensor
    hard_case: torch.Tensor
    budget_residual: torch.Tensor
    parent_score: torch.Tensor
    candidate_score: torch.Tensor
    factor_rank: torch.Tensor
    diagonal_minimum: torch.Tensor
    diagonal_median: torch.Tensor
    diagonal_maximum: torch.Tensor
    cross_layer_coupling_ratio: torch.Tensor
    dense_coordinate_metric_elements: int
    largest_dense_solve_dimension: int
    selected_update_elements_published: int
    owner_count: int


@dataclass(frozen=True)
class DistributedDiagonalCompletedTransactionResult:
    local_coefficients: torch.Tensor
    local_candidate_coefficients: torch.Tensor
    global_result: DiagonalCompletedTransactionResult
    local_coordinate_count: int
    global_coordinate_count: int
    communication_elements_per_rank: int
    selected_update_elements_published: int
    owner_count: int


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
        or current_decay_action.dtype != current_scores.dtype
        or current_decay_action.device != current_scores.device
        or not current_scores.is_floating_point()
        or not bool(torch.isfinite(current_scores).all())
        or not bool(torch.isfinite(current_decay_action).all())
    ):
        raise RuntimeError("diagonal-completed current score inventory changed")
    return int(current_scores.shape[1])


def diagonal_completed_biatlas_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_total_diagonal: torch.Tensor | None,
    previous_decay_cross: torch.Tensor | None,
    *,
    beta2: float,
) -> DiagonalCompletedBiatlasRows:
    """Advance one elapsed-beta EMA while preserving its exact diagonal."""

    coordinates = _validate_current(current_scores, current_decay_action)
    if float(beta2) != MATCHED_BETA2:
        raise ValueError("diagonal-completed biatlas requires matched beta2=.95")
    current64 = current_scores.double()
    decay64 = current_decay_action.double()
    current_diagonal = current64.square().sum(dim=0) / float(CURRENT_ROWS)
    current_cross = current64.T @ decay64 / float(CURRENT_ROWS)
    zero = current_scores.new_zeros(())

    missing = (
        previous_scores is None,
        previous_total_diagonal is None,
        previous_decay_cross is None,
    )
    if any(missing) and not all(missing):
        raise RuntimeError("diagonal-completed persistent state is partial")
    if previous_scores is None:
        total_diagonal = current_diagonal
        persistent = current_scores.detach().clone()
        represented = (
            persistent.double().square().sum(dim=0) / float(CURRENT_ROWS)
        )
        residual = (total_diagonal - represented).clamp_min(0.0)
        return DiagonalCompletedBiatlasRows(
            selection_scores=current_scores,
            selection_diagonal_residual=residual.to(current_scores.dtype),
            persistent_scores=persistent,
            persistent_total_diagonal=total_diagonal.to(current_scores.dtype),
            persistent_diagonal_residual=residual.to(current_scores.dtype),
            persistent_decay_cross=current_cross.to(current_scores.dtype),
            history_used=False,
            discarded_energy_fraction=zero,
        )

    previous_rows = int(previous_scores.shape[0])
    if (
        previous_scores.ndim != 2
        or previous_scores.shape[1] != coordinates
        or previous_rows not in (CURRENT_ROWS, PERSISTENT_ROWS)
        or previous_total_diagonal is None
        or previous_total_diagonal.shape != (coordinates,)
        or previous_decay_cross is None
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
        raise RuntimeError("diagonal-completed persistent inventory changed")

    elapsed = EFFECTIVE_REFRESH_BETA2
    previous64 = previous_scores.double()
    represented_previous = (
        previous64.square().sum(dim=0) / float(CURRENT_ROWS)
    )
    previous_residual = (
        previous_total_diagonal.double() - represented_previous
    ).clamp_min(0.0)
    selection = torch.cat((
        previous_scores * math.sqrt(elapsed),
        current_scores * math.sqrt(1.0 - elapsed),
    ))
    selection_residual = previous_residual * elapsed
    total_diagonal = (
        previous_total_diagonal.double() * elapsed
        + current_diagonal * (1.0 - elapsed)
    )
    decay_cross = (
        previous_decay_cross.double() * elapsed
        + current_cross * (1.0 - elapsed)
    )
    if int(selection.shape[0]) > MAXIMUM_SELECTION_ROWS:
        raise RuntimeError("diagonal-completed live row bound changed")

    row_gram = selection.double() @ selection.double().T
    row_gram = 0.5 * (row_gram + row_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(row_gram)
    retained_rows = min(PERSISTENT_ROWS, int(selection.shape[0]))
    transform = eigenvectors[:, -retained_rows:].T
    persistent64 = transform @ selection.double()
    represented_persistent = (
        persistent64.square().sum(dim=0) / float(CURRENT_ROWS)
    )
    persistent_residual = (
        total_diagonal - represented_persistent
    ).clamp_min(0.0)
    total_energy = eigenvalues.clamp_min(0.0).sum()
    discarded_energy = (
        eigenvalues[:-retained_rows].clamp_min(0.0).sum()
        if retained_rows < int(eigenvalues.numel()) else zero.double()
    )
    discarded_fraction = discarded_energy / total_energy.clamp_min(
        torch.finfo(total_energy.dtype).tiny
    )
    persistent = persistent64.to(current_scores.dtype)
    diagonal_error = (
        represented_persistent + persistent_residual - total_diagonal
    ).abs().amax()
    diagonal_scale = total_diagonal.abs().amax().clamp_min(1.0)
    torch._assert_async(
        torch.isfinite(selection).all()
        & torch.isfinite(persistent).all()
        & torch.isfinite(selection_residual).all()
        & torch.isfinite(persistent_residual).all()
        & torch.isfinite(decay_cross).all()
        & (selection_residual >= 0.0).all()
        & (persistent_residual >= 0.0).all()
        & (
            diagonal_error
            <= 64.0 * torch.finfo(current_scores.dtype).eps * diagonal_scale
        )
    )
    return DiagonalCompletedBiatlasRows(
        selection_scores=selection,
        selection_diagonal_residual=selection_residual.to(current_scores.dtype),
        persistent_scores=persistent,
        persistent_total_diagonal=total_diagonal.to(current_scores.dtype),
        persistent_diagonal_residual=persistent_residual.to(current_scores.dtype),
        persistent_decay_cross=decay_cross.to(current_scores.dtype),
        history_used=True,
        discarded_energy_fraction=discarded_fraction.to(current_scores.dtype),
    )


def _metric_action(
    low_rank: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return diagonal * value + low_rank.T @ (low_rank @ value)


def _loss_krylov_basis(
    low_rank: torch.Tensor,
    diagonal: torch.Tensor,
    rhs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-width fully reorthogonalized Lanczos loss subspace."""

    coordinates = int(rhs.numel())
    steps = min(KRYLOV_ROWS, coordinates)
    tiny = torch.finfo(rhs.dtype).tiny
    norm = torch.linalg.vector_norm(rhs)
    torch._assert_async(torch.isfinite(norm) & (norm > 0.0))
    basis = []
    alpha_values = []
    beta_values = []
    current = rhs / norm.clamp_min(tiny)
    previous = torch.zeros_like(current)
    previous_beta = rhs.new_zeros(())
    coordinate = torch.arange(
        coordinates, device=rhs.device, dtype=rhs.dtype
    ) + 1.0
    for index in range(steps):
        basis.append(current)
        action = _metric_action(low_rank, diagonal, current)
        alpha = torch.dot(current, action)
        residual = action - alpha * current - previous_beta * previous
        # Two fixed modified-Gram-Schmidt passes keep the basis orthonormal
        # even when the diagonal has a wide dynamic range.
        for _ in range(2):
            active_basis = torch.stack(basis, dim=1)
            residual = residual - active_basis @ (active_basis.T @ residual)
        beta = torch.linalg.vector_norm(residual)
        alpha_values.append(alpha)
        if index + 1 < steps:
            beta_values.append(beta)
            threshold = (
                64.0 * torch.finfo(rhs.dtype).eps
                * torch.linalg.vector_norm(action).clamp_min(1.0)
            )
            # Exact Krylov breakdown means the current subspace is invariant.
            # Pad with a deterministic orthogonal vector and a zero tridiagonal
            # link; its RHS projection is zero, so it cannot change the solve.
            fallback = torch.cos(coordinate * float(index + 1))
            for _ in range(2):
                active_basis = torch.stack(basis, dim=1)
                fallback = fallback - active_basis @ (active_basis.T @ fallback)
            fallback_norm = torch.linalg.vector_norm(fallback)
            usable = fallback_norm > threshold
            fallback = fallback / fallback_norm.clamp_min(tiny)
            next_value = residual / beta.clamp_min(tiny)
            next_value = torch.where(usable & (beta <= threshold), fallback, next_value)
            beta_values[-1] = torch.where(beta <= threshold, beta.new_zeros(()), beta)
            previous = current
            current = next_value
            previous_beta = beta_values[-1]
    q = torch.stack(basis, dim=1)
    diagonal_t = torch.stack(alpha_values)
    tridiagonal = torch.diag(diagonal_t)
    if steps > 1:
        off = torch.stack(beta_values)
        tridiagonal = tridiagonal + torch.diag(off, diagonal=1)
        tridiagonal = tridiagonal + torch.diag(off, diagonal=-1)
    tridiagonal = 0.5 * (tridiagonal + tridiagonal.T)
    orthogonality = torch.linalg.vector_norm(
        q.T @ q - torch.eye(steps, device=q.device, dtype=q.dtype)
    )
    torch._assert_async(
        torch.isfinite(q).all()
        & torch.isfinite(tridiagonal).all()
        & (orthogonality <= 2.0e-10 * float(steps))
    )
    return q, tridiagonal


_COMPILED_LOSS_KRYLOV_BASIS = torch.compile(
    _loss_krylov_basis, fullgraph=True, dynamic=False
)


def diagonal_completed_biatlas_transaction(
    rows: DiagonalCompletedBiatlasRows,
    exact_by_role: torch.Tensor,
    momentum_by_role: torch.Tensor,
    weights: torch.Tensor,
    layer_ids: torch.Tensor,
    *,
    total_layers: int,
    eta: float,
    rounds: int = 64,
) -> DiagonalCompletedTransactionResult:
    """Select a same-budget signed update in diagonal-plus-row-space metric."""

    scores = rows.selection_scores
    diagonal_residual = rows.selection_diagonal_residual
    decay_cross = rows.persistent_decay_cross
    factor_rows, coordinates = scores.shape
    measure_rows = int(getattr(rows, "measure_rows", CURRENT_ROWS))
    maximum_selection_rows = int(getattr(
        rows, "maximum_selection_rows", MAXIMUM_SELECTION_ROWS
    ))
    if (
        measure_rows < 1
        or factor_rows < measure_rows
        or factor_rows > maximum_selection_rows
        or maximum_selection_rows < factor_rows
        or coordinates < 2
        or coordinates % 2
        or diagonal_residual.shape != (coordinates,)
        or decay_cross.shape != (coordinates,)
        or exact_by_role.shape != (2, coordinates)
        or momentum_by_role.shape != exact_by_role.shape
        or weights.shape != (coordinates,)
        or layer_ids.shape != (coordinates,)
        or layer_ids.dtype != torch.int64
        or int(total_layers) < 1
        or float(eta) <= 0.0
        or int(rounds) < 1
    ):
        raise RuntimeError("diagonal-completed transaction inventory changed")
    floating = (
        scores,
        diagonal_residual,
        decay_cross,
        exact_by_role,
        momentum_by_role,
        weights,
    )


    if (
        any(not value.is_floating_point() for value in floating)
        or any(value.dtype != scores.dtype for value in floating[1:])
        or any(value.device != scores.device for value in floating[1:])
        or layer_ids.device != scores.device
        or any(not bool(torch.isfinite(value).all()) for value in floating)
        or not bool((diagonal_residual >= 0.0).all())
        or not bool((weights > 0.0).all())
    ):
        raise RuntimeError("diagonal-completed transaction values changed")
    if coordinates and (
        int(layer_ids.amin()) < 0 or int(layer_ids.amax()) >= int(total_layers)
    ):
        raise RuntimeError("diagonal-completed layer ID is invalid")

    score64 = scores.double()
    diagonal64 = diagonal_residual.double()
    decay64 = decay_cross.double()
    exact64 = exact_by_role.double()
    momentum64 = momentum_by_role.double()
    weight64 = weights.double()
    root_weight = torch.sqrt(weight64)
    inverse_root_weight = torch.reciprocal(root_weight)
    # The physical factor always represents covariance with its logical
    # functional-row measure, even when historical rows make the live factor
    # taller than that measure.
    low_rank = (
        score64 * inverse_root_weight.unsqueeze(0) / math.sqrt(measure_rows)
    )
    if factor_rows < maximum_selection_rows:
        low_rank = torch.cat((
            low_rank,
            low_rank.new_zeros(
                maximum_selection_rows - factor_rows, coordinates
            ),
        ), dim=0)
    diagonal_whitened = diagonal64 / weight64
    rhs = (
        exact64.sum(dim=0) / float(eta) - decay64
    ) * inverse_root_weight
    parent = root_weight
    budget = weight64.sum()
    tiny = torch.finfo(torch.float64).tiny
    machine = torch.finfo(torch.float64).eps
    krylov = (
        _COMPILED_LOSS_KRYLOV_BASIS
        if scores.is_cuda else _loss_krylov_basis
    )
    basis, tridiagonal = krylov(low_rank, diagonal_whitened, rhs)
    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    eigenvalues = eigenvalues.clamp_min(0.0)
    projected_rhs = eigenvectors.T @ (basis.T @ rhs)
    lower = torch.tensor(
        machine * float(coordinates), device=scores.device, dtype=torch.float64
    )
    rhs_norm = torch.linalg.vector_norm(rhs)
    upper = (
        rhs_norm / torch.sqrt(budget)
        + eigenvalues.amax()
        + lower
    )

    lower_coordinates = projected_rhs / (eigenvalues + lower)
    hard_case = lower_coordinates.square().sum() < budget
    lo = lower
    hi = upper
    for _ in range(int(rounds)):
        middle = 0.5 * (lo + hi)
        candidate_coordinates = projected_rhs / (eigenvalues + middle)
        too_large = candidate_coordinates.square().sum() > budget
        lo = torch.where(too_large, middle, lo)
        hi = torch.where(too_large, hi, middle)
    candidate_coordinates = projected_rhs / (eigenvalues + hi)
    candidate_y = basis @ (eigenvectors @ candidate_coordinates)
    candidate_coefficients64 = candidate_y * inverse_root_weight
    # A convex unconstrained minimizer inside the equality sphere requires a
    # negative-multiplier hard-case construction.  The literal parent is the
    # certified feasible point, so this version conservatively rejects that
    # case instead of introducing a new orientation or tuning rule.
    candidate_budget = (
        weight64 * candidate_coefficients64.square()
    ).sum()
    budget_residual = (
        (candidate_budget - budget).abs() / budget.clamp_min(1.0)
    )

    parent_coefficients64 = torch.ones_like(candidate_coefficients64)
    parent_action = score64 @ parent_coefficients64
    candidate_action = score64 @ candidate_coefficients64
    parent_exact = exact64.sum()
    candidate_exact = (
        exact64.sum(dim=0) * candidate_coefficients64
    ).sum()
    parent_score = (
        -float(eta) * parent_exact
        + 0.5 * float(eta) ** 2 * (
            parent_action.square().sum() / float(measure_rows)
            + (diagonal64 * parent_coefficients64.square()).sum()
            + 2.0 * decay64.sum()
        )
    )
    candidate_score = (
        -float(eta) * candidate_exact
        + 0.5 * float(eta) ** 2 * (
            candidate_action.square().sum() / float(measure_rows)
            + (diagonal64 * candidate_coefficients64.square()).sum()
            + 2.0 * (decay64 * candidate_coefficients64).sum()
        )
    )
    parent_exact_layers = torch.zeros(
        (2, int(total_layers)), device=scores.device, dtype=torch.float64
    )
    parent_momentum_layers = torch.zeros_like(parent_exact_layers)
    candidate_exact_layers = torch.zeros_like(parent_exact_layers)
    candidate_momentum_layers = torch.zeros_like(parent_exact_layers)
    parent_exact_layers.index_add_(1, layer_ids, exact64)
    parent_momentum_layers.index_add_(1, layer_ids, momentum64)
    candidate_exact_layers.index_add_(
        1, layer_ids, exact64 * candidate_coefficients64.unsqueeze(0)
    )
    candidate_momentum_layers.index_add_(
        1, layer_ids, momentum64 * candidate_coefficients64.unsqueeze(0)
    )
    finite = (
        torch.isfinite(candidate_coefficients64).all()
        & torch.isfinite(candidate_score)
        & torch.isfinite(budget_residual)
    )
    accepted = (
        finite
        & (~hard_case)
        & (candidate_score < parent_score)
        & (budget_residual <= 1.0e-8)
        & (candidate_exact_layers > 0.0).all()
        & (candidate_momentum_layers > 0.0).all()
        & (parent_exact_layers > 0.0).all()
        & (parent_momentum_layers > 0.0).all()
    ).reshape(1)
    candidate_coefficients = candidate_coefficients64.to(scores.dtype)
    selected = torch.where(
        accepted, candidate_coefficients, torch.ones_like(candidate_coefficients)
    )
    singular_values = torch.linalg.svdvals(low_rank)
    rank_threshold = (
        machine * float(coordinates) * singular_values.amax().clamp_min(tiny)
    )
    factor_rank = (singular_values > rank_threshold).sum().reshape(1)
    measure = float(measure_rows)
    total_row_metric = score64 @ score64.T
    total_metric_square = (
        total_row_metric.square().sum() / (measure * measure)
        + 2.0 * (
            diagonal64 * score64.square().sum(dim=0) / measure
        ).sum()
        + diagonal64.square().sum()
    )
    within_layer_square = torch.zeros_like(total_metric_square)
    for layer in range(int(total_layers)):
        mask = layer_ids.eq(layer)
        layer_factor = score64[:, mask]
        layer_diagonal = diagonal64[mask]
        layer_row_metric = layer_factor @ layer_factor.T
        within_layer_square = within_layer_square + (
            layer_row_metric.square().sum() / (measure * measure)
            + 2.0 * (
                layer_diagonal * layer_factor.square().sum(dim=0) / measure
            ).sum()
            + layer_diagonal.square().sum()
        )
    cross_layer_coupling_ratio = torch.sqrt(
        (total_metric_square - within_layer_square).clamp_min(0.0)
        / total_metric_square.clamp_min(tiny)
    )
    return DiagonalCompletedTransactionResult(
        coefficients=selected,
        candidate_coefficients=candidate_coefficients,
        accepted=accepted,
        multiplier=torch.where(
            hard_case, torch.zeros_like(hi), hi
        ).reshape(1),
        hard_case=hard_case.reshape(1),
        budget_residual=budget_residual.reshape(1),
        parent_score=parent_score.reshape(1),
        candidate_score=candidate_score.reshape(1),
        factor_rank=factor_rank,
        diagonal_minimum=diagonal64.amin().reshape(1),
        diagonal_median=diagonal64.median().reshape(1),
        diagonal_maximum=diagonal64.amax().reshape(1),
        cross_layer_coupling_ratio=cross_layer_coupling_ratio.reshape(1),
        dense_coordinate_metric_elements=0,
        largest_dense_solve_dimension=int(factor_rows),
        selected_update_elements_published=0,
        owner_count=0,
    )


def distributed_diagonal_completed_biatlas_transaction(
    local_rows: DiagonalCompletedBiatlasRows,
    local_exact_by_role: torch.Tensor,
    local_momentum_by_role: torch.Tensor,
    local_weights: torch.Tensor,
    local_layer_ids: torch.Tensor,
    local_coordinate_ids: torch.Tensor,
    *,
    total_coordinates: int,
    total_layers: int,
    eta: float,
    rounds: int = 64,
    group=None,
) -> DistributedDiagonalCompletedTransactionResult:
    """Owner-free arbitrary-shard adapter using only scalar score summaries.

    Logical coordinates must already be canonical and disjoint.  Empty and
    uneven shards are legal.  The replicated packet is ``O(KLG)`` score
    scalars plus ``O(LG)`` linear summaries; it contains neither parameters
    nor a selected matrix update.
    """

    scores = local_rows.selection_scores
    local_coordinates = int(local_coordinate_ids.numel())
    factor_rows = int(scores.shape[0]) if scores.ndim == 2 else -1
    if (
        factor_rows not in (CURRENT_ROWS, 2 * CURRENT_ROWS, MAXIMUM_SELECTION_ROWS)
        or scores.shape[1] != local_coordinates
        or local_rows.selection_diagonal_residual.shape != (local_coordinates,)
        or local_rows.persistent_decay_cross.shape != (local_coordinates,)
        or local_exact_by_role.shape != (2, local_coordinates)
        or local_momentum_by_role.shape != (2, local_coordinates)
        or local_weights.shape != (local_coordinates,)
        or local_layer_ids.shape != (local_coordinates,)
        or local_coordinate_ids.shape != (local_coordinates,)
        or local_layer_ids.dtype != torch.int64
        or local_coordinate_ids.dtype != torch.int64
        or int(total_coordinates) < 2
        or int(total_coordinates) % 2
    ):
        raise RuntimeError("distributed diagonal-completed inventory changed")
    if local_coordinates and not bool(
        ((local_coordinate_ids >= 0)
         & (local_coordinate_ids < int(total_coordinates))).all()
    ):
        raise RuntimeError("distributed diagonal-completed coordinate ID is invalid")
    if not (dist.is_available() and dist.is_initialized()):
        if local_coordinates != int(total_coordinates):
            raise RuntimeError("single-rank diagonal-completed shard is incomplete")
        result = diagonal_completed_biatlas_transaction(
            local_rows,
            local_exact_by_role,
            local_momentum_by_role,
            local_weights,
            local_layer_ids,
            total_layers=int(total_layers),
            eta=float(eta),
            rounds=int(rounds),
        )
        return DistributedDiagonalCompletedTransactionResult(
            local_coefficients=result.coefficients,
            local_candidate_coefficients=result.candidate_coefficients,
            global_result=result,
            local_coordinate_count=local_coordinates,
            global_coordinate_count=int(total_coordinates),
            communication_elements_per_rank=0,
            selected_update_elements_published=0,
            owner_count=0,
        )

    device = scores.device
    dtype = scores.dtype
    ids = local_coordinate_ids
    ownership = torch.zeros(
        int(total_coordinates), device=device, dtype=torch.float64
    )
    ownership[ids] = 1.0
    global_scores = torch.zeros(
        factor_rows, int(total_coordinates), device=device, dtype=dtype
    )
    global_diagonal = torch.zeros(
        int(total_coordinates), device=device, dtype=dtype
    )
    global_cross = torch.zeros_like(global_diagonal)
    global_exact = torch.zeros(
        2, int(total_coordinates), device=device, dtype=dtype
    )
    global_momentum = torch.zeros_like(global_exact)
    global_weights = torch.zeros_like(global_diagonal)
    global_layer_ids = torch.zeros_like(global_diagonal)
    global_scores[:, ids] = scores
    global_diagonal[ids] = local_rows.selection_diagonal_residual
    global_cross[ids] = local_rows.persistent_decay_cross
    global_exact[:, ids] = local_exact_by_role
    global_momentum[:, ids] = local_momentum_by_role
    global_weights[ids] = local_weights
    global_layer_ids[ids] = local_layer_ids.to(dtype)
    packet = torch.cat((
        global_scores.reshape(-1),
        global_diagonal,
        global_cross,
        global_exact.reshape(-1),
        global_momentum.reshape(-1),
        global_weights,
        global_layer_ids,
    ))
    dist.all_reduce(packet, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(ownership, op=dist.ReduceOp.SUM, group=group)
    if not bool(ownership.eq(1.0).all()):
        raise RuntimeError(
            "distributed diagonal-completed coordinates are not disjoint"
        )
    offset = 0
    count = factor_rows * int(total_coordinates)
    global_scores = packet[offset:offset + count].view(
        factor_rows, int(total_coordinates)
    )
    offset += count
    vectors = []
    for width in (1, 1, 2, 2, 1, 1):
        count = width * int(total_coordinates)
        vectors.append(packet[offset:offset + count])
        offset += count
    global_diagonal = vectors[0]
    global_cross = vectors[1]
    global_exact = vectors[2].view(2, int(total_coordinates))
    global_momentum = vectors[3].view(2, int(total_coordinates))
    global_weights = vectors[4]
    global_layer_ids = vectors[5].round().to(torch.int64)
    global_rows = DiagonalCompletedBiatlasRows(
        selection_scores=global_scores,
        selection_diagonal_residual=global_diagonal,
        # The transaction does not consume persistent-only fields.  Reuse
        # shape-correct values without retaining another global state copy.
        persistent_scores=global_scores[:min(PERSISTENT_ROWS, factor_rows)],
        persistent_total_diagonal=(
            global_scores.double().square().sum(dim=0) / float(CURRENT_ROWS)
            + global_diagonal.double()
        ).to(dtype),
        persistent_diagonal_residual=global_diagonal,
        persistent_decay_cross=global_cross,
        history_used=local_rows.history_used,
        discarded_energy_fraction=local_rows.discarded_energy_fraction,
    )
    result = diagonal_completed_biatlas_transaction(
        global_rows,
        global_exact,
        global_momentum,
        global_weights,
        global_layer_ids,
        total_layers=int(total_layers),
        eta=float(eta),
        rounds=int(rounds),
    )
    return DistributedDiagonalCompletedTransactionResult(
        local_coefficients=result.coefficients[ids],
        local_candidate_coefficients=result.candidate_coefficients[ids],
        global_result=result,
        local_coordinate_count=local_coordinates,
        global_coordinate_count=int(total_coordinates),
        communication_elements_per_rank=int(packet.numel() + ownership.numel()),
        selected_update_elements_published=0,
        owner_count=0,
    )


def diagonal_completed_biatlas_scaling_formula(
    *,
    layers: int,
    groups: int,
    intermediate_width: int,
    model_width: int,
    activation_positions: int,
) -> dict[str, int]:
    if min(layers, groups, intermediate_width, model_width, activation_positions) < 1:
        raise ValueError("diagonal-completed scaling dimensions must be positive")
    coordinates = 2 * int(layers) * int(groups)
    return {
        "coordinate_count": coordinates,
        "persistent_factor_elements": PERSISTENT_ROWS * coordinates,
        "persistent_diagonal_elements": 2 * coordinates,
        "maximum_live_factor_elements": MAXIMUM_SELECTION_ROWS * coordinates,
        "largest_dense_solve_dimension": MAXIMUM_SELECTION_ROWS,
        "dense_coordinate_metric_elements": 0,
        "method_state_depends_on_total_activation_positions": 0,
        "method_specific_activation_position_state_elements": 0,
    }


__all__ = (
    "CURRENT_ROWS",
    "DiagonalCompletedBiatlasRows",
    "DiagonalCompletedTransactionResult",
    "DistributedDiagonalCompletedTransactionResult",
    "EFFECTIVE_REFRESH_BETA2",
    "KRYLOV_ROWS",
    "MATCHED_BETA2",
    "MAXIMUM_SELECTION_ROWS",
    "PERSISTENT_ROWS",
    "REFRESH_INTERVAL",
    "diagonal_completed_biatlas_rows",
    "diagonal_completed_biatlas_scaling_formula",
    "diagonal_completed_biatlas_transaction",
    "distributed_diagonal_completed_biatlas_transaction",
)
