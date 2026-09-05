"""Owner-free dual realization of the R01 equality-budget metric solve.

For aligned score rows ``S`` and positive budget weights ``w``, R01 uses the
whitened metric ``H = T.T @ T / K`` with ``T = S * rsqrt(w)``.  When the
coordinate count is larger than the score-row rank, this module solves the
same equality-constrained quadratic in score-row space.  It never constructs
the ``p x p`` metric on that path.

This module is intentionally independent of process topology.  A distributed
caller may form the required row-space summaries with additive reductions on
the model's existing parameter shards; no optimizer-owner concept is part of
the equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SphereSolveResult:
    coordinates: torch.Tensor
    multiplier: torch.Tensor
    hard_case: torch.Tensor
    rank: torch.Tensor
    budget_residual: torch.Tensor


@dataclass(frozen=True)
class ShardedSphereSolveResult:
    coordinate_shards: tuple[torch.Tensor, ...]
    multiplier: torch.Tensor
    hard_case: torch.Tensor
    rank: torch.Tensor
    budget_residual: torch.Tensor


def _validate_sphere_inputs(
    whitened_scores: torch.Tensor,
    rhs: torch.Tensor,
    parent_coordinates: torch.Tensor,
    budget: torch.Tensor,
) -> tuple[int, int, int]:
    if whitened_scores.ndim != 3:
        raise RuntimeError("dual R01 scores must have shape [batch, rows, coordinates]")
    batch, rows, coordinates = whitened_scores.shape
    if rows < 1 or coordinates < 1:
        raise RuntimeError("dual R01 received an empty score lattice")
    if rhs.shape != (batch, coordinates):
        raise RuntimeError("dual R01 rhs inventory changed")
    if parent_coordinates.shape != rhs.shape:
        raise RuntimeError("dual R01 parent inventory changed")
    if budget.shape != (batch,):
        raise RuntimeError("dual R01 budget inventory changed")
    if not (
        whitened_scores.dtype == rhs.dtype == parent_coordinates.dtype
        and whitened_scores.device == rhs.device == parent_coordinates.device
        and budget.dtype == rhs.dtype
        and budget.device == rhs.device
    ):
        raise RuntimeError("dual R01 dtype or device inventory changed")
    valid = (
        torch.isfinite(whitened_scores).all(dim=(-2, -1))
        & torch.isfinite(rhs).all(dim=-1)
        & torch.isfinite(parent_coordinates).all(dim=-1)
        & torch.isfinite(budget)
        & (budget > 0.0)
    )
    if not bool(valid.all().item()):
        raise RuntimeError("dual R01 received non-finite data or a nonpositive budget")
    return batch, rows, coordinates


def _fixed_bisection(
    eigenvalues: torch.Tensor,
    rhs_coordinates: torch.Tensor,
    rhs_null_norm_squared: torch.Tensor,
    budget: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    rounds: int,
) -> torch.Tensor:
    if rounds < 1:
        raise RuntimeError("dual R01 secular solve needs at least one round")
    for _ in range(rounds):
        middle = 0.5 * (lower + upper)
        row_norm_squared = (
            rhs_coordinates.square()
            / (eigenvalues + middle[:, None]).square()
        ).sum(dim=-1)
        null_norm_squared = rhs_null_norm_squared / middle.square()
        too_large = row_norm_squared + null_norm_squared > budget
        lower = torch.where(too_large, middle, lower)
        upper = torch.where(too_large, upper, middle)
    return upper


def _dense_sphere_solve(
    whitened_scores: torch.Tensor,
    rhs: torch.Tensor,
    parent_coordinates: torch.Tensor,
    budget: torch.Tensor,
    *,
    rounds: int,
) -> SphereSolveResult:
    """Reference/primal path used only when the coordinate dimension is small."""
    _batch, rows, coordinates = whitened_scores.shape
    metric = torch.bmm(
        whitened_scores.transpose(1, 2), whitened_scores
    ) / float(rows)
    metric = 0.5 * (metric + metric.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    machine = torch.finfo(metric.dtype).eps
    tiny = torch.finfo(metric.dtype).tiny
    spectral_scale = eigenvalues.abs().amax(dim=-1)
    rank_threshold = (
        machine * float(coordinates) * spectral_scale.clamp_min(tiny)
    )
    retained = eigenvalues > rank_threshold[:, None]
    projected_rhs = torch.einsum("bpi,bp->bi", eigenvectors, rhs)
    minimum = eigenvalues[:, 0]
    lower = -minimum + rank_threshold
    upper = (
        torch.linalg.vector_norm(projected_rhs, dim=-1)
        / torch.sqrt(budget)
        + spectral_scale
        + rank_threshold
    )
    lower_coordinates = projected_rhs / (
        eigenvalues + lower[:, None]
    ).clamp_min(rank_threshold[:, None])
    hard_case = lower_coordinates.square().sum(dim=-1) < budget

    zero_null = torch.zeros_like(budget)
    multiplier = _fixed_bisection(
        eigenvalues,
        projected_rhs,
        zero_null,
        budget,
        lower,
        upper,
        rounds=rounds,
    )
    root_coordinates = projected_rhs / (
        eigenvalues + multiplier[:, None]
    ).clamp_min(rank_threshold[:, None])

    separation = eigenvalues - minimum[:, None]
    minimum_mask = separation <= rank_threshold[:, None]
    hard_base = torch.where(
        minimum_mask,
        torch.zeros_like(projected_rhs),
        projected_rhs / separation.clamp_min(rank_threshold[:, None]),
    )
    remaining = (budget - hard_base.square().sum(dim=-1)).clamp_min(0.0)
    first_minimum = torch.argmax(minimum_mask.to(torch.int64), dim=-1)
    projected_parent = torch.einsum(
        "bpi,bp->bi", eigenvectors, parent_coordinates
    )
    signs = torch.sign(
        projected_parent.gather(1, first_minimum[:, None]).squeeze(1)
    )
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    hard_fill = torch.zeros_like(hard_base).scatter(
        1,
        first_minimum[:, None],
        (torch.sqrt(remaining) * signs)[:, None],
    )
    selected_eigen_coordinates = torch.where(
        hard_case[:, None], hard_base + hard_fill, root_coordinates
    )
    selected = torch.einsum(
        "bpi,bi->bp", eigenvectors, selected_eigen_coordinates
    )
    selected_budget = selected.square().sum(dim=-1)
    residual = (selected_budget - budget).abs() / budget.clamp_min(1.0)
    return SphereSolveResult(
        coordinates=selected,
        multiplier=torch.where(hard_case, -minimum, multiplier),
        hard_case=hard_case,
        rank=retained.sum(dim=-1),
        budget_residual=residual,
    )


def _dual_rank_deficient_sphere_solve(
    whitened_scores: torch.Tensor,
    rhs: torch.Tensor,
    parent_coordinates: torch.Tensor,
    budget: torch.Tensor,
    *,
    rounds: int,
) -> SphereSolveResult:
    """Solve in score-row space when ``rows < coordinates``.

    The metric then has an analytic zero-eigenvalue complement.  Both the
    ordinary positive-root solution and the equality-sphere hard case are
    represented without a coordinate-square matrix.
    """
    batch, rows, coordinates = whitened_scores.shape
    row_metric = torch.bmm(
        whitened_scores, whitened_scores.transpose(1, 2)
    ) / float(rows)
    row_metric = 0.5 * (row_metric + row_metric.transpose(1, 2))
    eigenvalues, left_vectors = torch.linalg.eigh(row_metric)
    machine = torch.finfo(row_metric.dtype).eps
    tiny = torch.finfo(row_metric.dtype).tiny
    spectral_scale = eigenvalues.abs().amax(dim=-1)
    rank_threshold = (
        machine * float(coordinates) * spectral_scale.clamp_min(tiny)
    )
    retained = eigenvalues > rank_threshold[:, None]

    # Right singular vectors of T are T^T u_i / sqrt(K mu_i).  Zeroed
    # columns are harmless and avoid ragged tensors across a batch.
    safe_eigenvalues = torch.where(retained, eigenvalues, torch.ones_like(eigenvalues))
    right_vectors = torch.bmm(
        whitened_scores.transpose(1, 2), left_vectors
    ) / torch.sqrt(float(rows) * safe_eigenvalues)[:, None, :]
    right_vectors = torch.where(
        retained[:, None, :], right_vectors, torch.zeros_like(right_vectors)
    )
    rhs_coordinates = torch.einsum("bpi,bp->bi", right_vectors, rhs)
    rhs_parallel = torch.einsum(
        "bpi,bi->bp", right_vectors, rhs_coordinates
    )
    rhs_null = rhs - rhs_parallel
    rhs_null_norm_squared = rhs_null.square().sum(dim=-1)

    lower = rank_threshold
    upper = (
        torch.linalg.vector_norm(rhs, dim=-1) / torch.sqrt(budget)
        + spectral_scale
        + rank_threshold
    )
    lower_row_norm_squared = (
        rhs_coordinates.square()
        / (eigenvalues + lower[:, None]).clamp_min(
            rank_threshold[:, None]
        ).square()
    ).sum(dim=-1)
    lower_null_norm_squared = rhs_null_norm_squared / lower.square()
    hard_case = lower_row_norm_squared + lower_null_norm_squared < budget

    multiplier = _fixed_bisection(
        eigenvalues,
        rhs_coordinates,
        rhs_null_norm_squared,
        budget,
        lower,
        upper,
        rounds=rounds,
    )
    root_row_coordinates = rhs_coordinates / (
        eigenvalues + multiplier[:, None]
    ).clamp_min(rank_threshold[:, None])
    root_solution = (
        torch.einsum("bpi,bi->bp", right_vectors, root_row_coordinates)
        + rhs_null / multiplier[:, None]
    )

    # With a zero minimum eigenvalue, a hard-case solution is the pseudoinverse
    # row-space solution plus any remaining budget in the metric null space.
    hard_row_coordinates = torch.where(
        retained,
        rhs_coordinates / safe_eigenvalues,
        torch.zeros_like(rhs_coordinates),
    )
    hard_base = torch.einsum(
        "bpi,bi->bp", right_vectors, hard_row_coordinates
    )
    parent_row_coordinates = torch.einsum(
        "bpi,bp->bi", right_vectors, parent_coordinates
    )
    parent_null = parent_coordinates - torch.einsum(
        "bpi,bi->bp", right_vectors, parent_row_coordinates
    )
    parent_null_norm = torch.linalg.vector_norm(parent_null, dim=-1)

    # If the feasible parent happens to lie wholly in the row space, project
    # the least-leverage canonical coordinate into the null space.
    leverage = right_vectors.square().sum(dim=-1)
    canonical_index = torch.argmin(leverage, dim=-1)
    canonical = torch.zeros_like(parent_coordinates).scatter(
        1, canonical_index[:, None], torch.ones((batch, 1), device=rhs.device, dtype=rhs.dtype)
    )
    canonical_row_coordinates = torch.einsum(
        "bpi,bp->bi", right_vectors, canonical
    )
    canonical_null = canonical - torch.einsum(
        "bpi,bi->bp", right_vectors, canonical_row_coordinates
    )
    use_parent = parent_null_norm > torch.sqrt(rank_threshold)
    null_axis = torch.where(use_parent[:, None], parent_null, canonical_null)
    null_axis = null_axis / torch.linalg.vector_norm(
        null_axis, dim=-1, keepdim=True
    ).clamp_min(torch.sqrt(rank_threshold)[:, None])
    remaining = (budget - hard_base.square().sum(dim=-1)).clamp_min(0.0)
    hard_solution = hard_base + torch.sqrt(remaining)[:, None] * null_axis
    selected = torch.where(hard_case[:, None], hard_solution, root_solution)
    selected_budget = selected.square().sum(dim=-1)
    residual = (selected_budget - budget).abs() / budget.clamp_min(1.0)
    return SphereSolveResult(
        coordinates=selected,
        multiplier=torch.where(hard_case, torch.zeros_like(multiplier), multiplier),
        hard_case=hard_case,
        rank=retained.sum(dim=-1),
        budget_residual=residual,
    )


def solve_equality_sphere_from_scores(
    whitened_scores: torch.Tensor,
    rhs: torch.Tensor,
    parent_coordinates: torch.Tensor,
    budget: torch.Tensor,
    *,
    rounds: int = 64,
) -> SphereSolveResult:
    """Return the global R01 whitened equality-budget solution.

    ``whitened_scores`` is ``S * rsqrt(w)``.  The caller converts the returned
    whitened coordinates to allocation coefficients with another
    multiplication by ``rsqrt(w)``.
    """
    _batch, rows, coordinates = _validate_sphere_inputs(
        whitened_scores, rhs, parent_coordinates, budget
    )
    if rows < coordinates:
        return _dual_rank_deficient_sphere_solve(
            whitened_scores,
            rhs,
            parent_coordinates,
            budget,
            rounds=rounds,
        )
    return _dense_sphere_solve(
        whitened_scores,
        rhs,
        parent_coordinates,
        budget,
        rounds=rounds,
    )


def solve_sharded_equality_sphere_from_scores(
    whitened_score_shards: tuple[torch.Tensor, ...],
    rhs_shards: tuple[torch.Tensor, ...],
    parent_coordinate_shards: tuple[torch.Tensor, ...],
    budget: torch.Tensor,
    *,
    rounds: int = 64,
) -> ShardedSphereSolveResult:
    """Dual solve from additive coordinate-shard summaries.

    This pure function emulates the tensor-parallel/FSDP realization without
    concatenating coordinate shards.  A distributed wrapper replaces the
    Python sums with reductions of the same ``K x K``, ``K``, and scalar
    summaries and then evaluates the local reconstruction on each rank.
    """
    if not whitened_score_shards:
        raise RuntimeError("sharded dual R01 received no coordinate shards")
    if not (
        len(whitened_score_shards)
        == len(rhs_shards)
        == len(parent_coordinate_shards)
    ):
        raise RuntimeError("sharded dual R01 shard inventories differ")
    first = whitened_score_shards[0]
    if first.ndim != 3:
        raise RuntimeError("sharded dual R01 scores must have rank three")
    batch, rows, _ = first.shape
    total_coordinates = 0
    for scores, rhs, parent in zip(
        whitened_score_shards, rhs_shards, parent_coordinate_shards
    ):
        if (
            scores.ndim != 3
            or scores.shape[:2] != (batch, rows)
            or rhs.shape != (batch, scores.shape[-1])
            or parent.shape != rhs.shape
            or scores.dtype != first.dtype
            or rhs.dtype != first.dtype
            or parent.dtype != first.dtype
            or scores.device != first.device
            or rhs.device != first.device
            or parent.device != first.device
        ):
            raise RuntimeError("sharded dual R01 local inventory changed")
        total_coordinates += int(scores.shape[-1])
    if rows >= total_coordinates:
        raise RuntimeError(
            "sharded dual R01 is intended for the score-rank-limited path"
        )
    if budget.shape != (batch,) or budget.dtype != first.dtype or budget.device != first.device:
        raise RuntimeError("sharded dual R01 budget inventory changed")

    row_metric = sum(
        torch.bmm(scores, scores.transpose(1, 2))
        for scores in whitened_score_shards
    ) / float(rows)
    row_metric = 0.5 * (row_metric + row_metric.transpose(1, 2))
    rhs_row = sum(
        torch.bmm(scores, rhs.unsqueeze(-1)).squeeze(-1)
        for scores, rhs in zip(whitened_score_shards, rhs_shards)
    )
    parent_row = sum(
        torch.bmm(scores, parent.unsqueeze(-1)).squeeze(-1)
        for scores, parent in zip(
            whitened_score_shards, parent_coordinate_shards
        )
    )
    rhs_norm_squared = sum(rhs.square().sum(dim=-1) for rhs in rhs_shards)
    parent_norm_squared = sum(
        parent.square().sum(dim=-1) for parent in parent_coordinate_shards
    )

    eigenvalues, left_vectors = torch.linalg.eigh(row_metric)
    machine = torch.finfo(row_metric.dtype).eps
    tiny = torch.finfo(row_metric.dtype).tiny
    spectral_scale = eigenvalues.abs().amax(dim=-1)
    rank_threshold = (
        machine
        * float(total_coordinates)
        * spectral_scale.clamp_min(tiny)
    )
    retained = eigenvalues > rank_threshold[:, None]
    safe_eigenvalues = torch.where(retained, eigenvalues, torch.ones_like(eigenvalues))
    rhs_coordinates = torch.einsum(
        "bki,bk->bi", left_vectors, rhs_row
    ) / torch.sqrt(float(rows) * safe_eigenvalues)
    rhs_coordinates = torch.where(
        retained, rhs_coordinates, torch.zeros_like(rhs_coordinates)
    )
    rhs_null_norm_squared = (
        rhs_norm_squared - rhs_coordinates.square().sum(dim=-1)
    ).clamp_min(0.0)

    lower = rank_threshold
    upper = (
        torch.sqrt(rhs_norm_squared) / torch.sqrt(budget)
        + spectral_scale
        + rank_threshold
    )
    lower_norm_squared = (
        rhs_coordinates.square()
        / (eigenvalues + lower[:, None]).clamp_min(
            rank_threshold[:, None]
        ).square()
    ).sum(dim=-1) + rhs_null_norm_squared / lower.square()
    hard_case = lower_norm_squared < budget
    multiplier = _fixed_bisection(
        eigenvalues,
        rhs_coordinates,
        rhs_null_norm_squared,
        budget,
        lower,
        upper,
        rounds=rounds,
    )

    right_vector_shards = tuple(
        torch.where(
            retained[:, None, :],
            torch.bmm(scores.transpose(1, 2), left_vectors)
            / torch.sqrt(float(rows) * safe_eigenvalues)[:, None, :],
            torch.zeros(
                (batch, scores.shape[-1], rows),
                device=scores.device,
                dtype=scores.dtype,
            ),
        )
        for scores in whitened_score_shards
    )
    root_row_coordinates = rhs_coordinates / (
        eigenvalues + multiplier[:, None]
    ).clamp_min(rank_threshold[:, None])
    root_shards = tuple(
        torch.einsum("bpi,bi->bp", right, root_row_coordinates)
        + (
            rhs
            - torch.einsum("bpi,bi->bp", right, rhs_coordinates)
        )
        / multiplier[:, None]
        for right, rhs in zip(right_vector_shards, rhs_shards)
    )

    hard_row_coordinates = torch.where(
        retained,
        rhs_coordinates / safe_eigenvalues,
        torch.zeros_like(rhs_coordinates),
    )
    hard_base_shards = tuple(
        torch.einsum("bpi,bi->bp", right, hard_row_coordinates)
        for right in right_vector_shards
    )
    parent_coordinates = torch.einsum(
        "bki,bk->bi", left_vectors, parent_row
    ) / torch.sqrt(float(rows) * safe_eigenvalues)
    parent_coordinates = torch.where(
        retained, parent_coordinates, torch.zeros_like(parent_coordinates)
    )
    parent_null_shards = tuple(
        parent - torch.einsum("bpi,bi->bp", right, parent_coordinates)
        for right, parent in zip(right_vector_shards, parent_coordinate_shards)
    )
    parent_null_norm_squared = (
        parent_norm_squared - parent_coordinates.square().sum(dim=-1)
    ).clamp_min(0.0)

    # Construct a deterministic projected canonical vector only for batches
    # whose feasible parent has no usable null-space component.
    leverage_shards = tuple(right.square().sum(dim=-1) for right in right_vector_shards)
    offsets = []
    offset = 0
    for scores in whitened_score_shards:
        offsets.append(offset)
        offset += int(scores.shape[-1])
    local_minima = torch.stack(
        [values.amin(dim=-1) for values in leverage_shards], dim=-1
    )
    chosen_shard = torch.argmin(local_minima, dim=-1)
    chosen_local = torch.stack(
        [values.argmin(dim=-1) for values in leverage_shards], dim=-1
    ).gather(1, chosen_shard[:, None]).squeeze(1)
    canonical_shards = []
    for shard_index, scores in enumerate(whitened_score_shards):
        local = torch.zeros(
            (batch, scores.shape[-1]), device=scores.device, dtype=scores.dtype
        )
        active = chosen_shard == shard_index
        if bool(active.any().item()):
            local[active, chosen_local[active]] = 1.0
        canonical_shards.append(local)
    canonical_row = sum(
        torch.bmm(scores, local.unsqueeze(-1)).squeeze(-1)
        for scores, local in zip(whitened_score_shards, canonical_shards)
    )
    canonical_coordinates = torch.einsum(
        "bki,bk->bi", left_vectors, canonical_row
    ) / torch.sqrt(float(rows) * safe_eigenvalues)
    canonical_coordinates = torch.where(
        retained, canonical_coordinates, torch.zeros_like(canonical_coordinates)
    )
    canonical_null_shards = tuple(
        local - torch.einsum("bpi,bi->bp", right, canonical_coordinates)
        for local, right in zip(canonical_shards, right_vector_shards)
    )
    canonical_null_norm_squared = sum(
        value.square().sum(dim=-1) for value in canonical_null_shards
    )
    use_parent = parent_null_norm_squared > rank_threshold
    chosen_null_norm = torch.where(
        use_parent, parent_null_norm_squared, canonical_null_norm_squared
    ).sqrt().clamp_min(torch.sqrt(rank_threshold))
    null_axis_shards = tuple(
        torch.where(use_parent[:, None], parent_null, canonical_null)
        / chosen_null_norm[:, None]
        for parent_null, canonical_null in zip(
            parent_null_shards, canonical_null_shards
        )
    )
    hard_base_norm_squared = sum(
        value.square().sum(dim=-1) for value in hard_base_shards
    )
    remaining = (budget - hard_base_norm_squared).clamp_min(0.0)
    hard_shards = tuple(
        base + torch.sqrt(remaining)[:, None] * axis
        for base, axis in zip(hard_base_shards, null_axis_shards)
    )
    selected_shards = tuple(
        torch.where(hard_case[:, None], hard, root)
        for hard, root in zip(hard_shards, root_shards)
    )
    selected_budget = sum(
        value.square().sum(dim=-1) for value in selected_shards
    )
    residual = (selected_budget - budget).abs() / budget.clamp_min(1.0)
    return ShardedSphereSolveResult(
        coordinate_shards=selected_shards,
        multiplier=torch.where(hard_case, torch.zeros_like(multiplier), multiplier),
        hard_case=hard_case,
        rank=retained.sum(dim=-1),
        budget_residual=residual,
    )


__all__ = [
    "ShardedSphereSolveResult",
    "SphereSolveResult",
    "solve_equality_sphere_from_scores",
    "solve_sharded_equality_sphere_from_scores",
]
