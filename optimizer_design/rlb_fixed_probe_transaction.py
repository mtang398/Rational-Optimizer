"""Owner-free fixed-probe realization of the Global-RLB loss transaction.

The score lattice has ``K`` globally aligned functional probes and is column
sharded over the logical RLB coordinates.  ``K`` is a method constant (144 in
the campaign), not the number of activation positions in an update.  The
transaction performs a fixed-size row-space solve and reconstructs only the
coefficients belonging to the caller's existing parameter shard.

The caller is responsible for routing additive TP/FSDP fragments into a
canonical, disjoint logical-coordinate sharding.  This routing moves only
``K + O(1)`` scalars per coordinate; it never moves a parameter-sized source
or selected update.  Pipeline and data ranks must use the same global probe
IDs before entering this routine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .rlb_dual_loss_metric import _fixed_bisection


@dataclass(frozen=True)
class FixedProbeTransactionResult:
    local_coefficients: torch.Tensor
    local_candidate_coefficients: torch.Tensor
    accepted: torch.Tensor
    multiplier: torch.Tensor
    rank: torch.Tensor
    eigenvalue_max: torch.Tensor
    hard_case: torch.Tensor
    parent_score: torch.Tensor
    candidate_score: torch.Tensor
    budget_residual: torch.Tensor
    local_coordinate_count: int
    global_coordinate_count: int
    global_probe_count: int
    collective_rounds: int
    summary_elements: int
    owner_count: int
    dense_LG_by_LG_metric_elements: int
    selected_update_elements_published: int
    method_state_depends_on_total_tokens: bool


@dataclass(frozen=True)
class ReplicatedFixedProbeTransactionResult:
    """Full coefficients for a replicated-model endpoint adapter.

    The mathematical transaction remains column sharded.  This adapter only
    exchanges the fixed probe-score lattice and the resulting scalar
    coefficients because ordinary DDP replicates every logical coordinate on
    every rank.  TP/FSDP callers should use :func:`distributed_fixed_probe_transaction`
    directly and keep the returned coefficients on their native shards.
    """

    coefficients: torch.Tensor
    candidate_coefficients: torch.Tensor
    sharded_result: FixedProbeTransactionResult
    local_probe_count: int
    global_probe_count: int
    cross_layer_coupling_ratio: torch.Tensor
    collective_rounds: int
    score_scalars_exchanged_per_rank: int
    coefficient_scalars_exchanged_per_rank: int
    selected_update_elements_published: int
    method_state_depends_on_total_tokens: bool


def _all_reduce(value: torch.Tensor, group) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)


def _distributed_shape(group) -> tuple[int, int]:
    if not (dist.is_available() and dist.is_initialized()):
        return 0, 1
    return dist.get_rank(group=group), dist.get_world_size(group=group)


def _gather_variable_probe_rows(
    local_packet: torch.Tensor,
    *,
    expected_global_rows: int,
    group,
) -> tuple[torch.Tensor, int]:
    """Gather a globally fixed number of unevenly partitioned probe rows."""

    if local_packet.ndim != 2:
        raise RuntimeError("fixed-probe row packet must be rank two")
    rank, world = _distributed_shape(group)
    del rank
    local_rows, columns = local_packet.shape
    if world == 1:
        if local_rows != int(expected_global_rows):
            raise RuntimeError("fixed-probe global row count changed")
        return local_packet, 0

    count = torch.tensor(
        [local_rows], device=local_packet.device, dtype=torch.int64
    )
    counts = [torch.empty_like(count) for _ in range(world)]
    dist.all_gather(counts, count, group=group)
    row_counts = [int(value.item()) for value in counts]
    if sum(row_counts) != int(expected_global_rows):
        raise RuntimeError(
            "fixed-probe rows must sum to the globally fixed probe count"
        )
    maximum = max(row_counts)
    padded = torch.zeros(
        (maximum, columns), device=local_packet.device, dtype=local_packet.dtype
    )
    if local_rows:
        padded[:local_rows].copy_(local_packet)
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    return torch.cat([
        value[:rows] for value, rows in zip(gathered, row_counts)
    ], dim=0), 2


def _validate(
    scores,
    exact_by_role,
    momentum_by_role,
    decay_cross,
    weights,
    layer_ids,
    coordinate_ids,
    *,
    total_coordinates,
    total_layers,
    eta,
):
    if scores.ndim != 2:
        raise RuntimeError("fixed-probe scores must have shape [K, local_p]")
    probes, local_p = scores.shape
    if probes < 2 or total_coordinates < 1 or total_layers < 1:
        raise RuntimeError("fixed-probe logical dimensions are invalid")
    if exact_by_role.shape != (2, local_p) or momentum_by_role.shape != (2, local_p):
        raise RuntimeError("fixed-probe role inventory changed")
    if decay_cross.shape != (local_p,) or weights.shape != (local_p,):
        raise RuntimeError("fixed-probe scalar inventory changed")
    if layer_ids.shape != (local_p,) or coordinate_ids.shape != (local_p,):
        raise RuntimeError("fixed-probe coordinate inventory changed")
    if layer_ids.dtype != torch.int64 or coordinate_ids.dtype != torch.int64:
        raise RuntimeError("fixed-probe IDs must be int64")
    floating = (scores, exact_by_role, momentum_by_role, decay_cross, weights)
    if (
        not scores.is_floating_point()
        or any(value.dtype != scores.dtype for value in floating[1:])
        or any(value.device != scores.device for value in floating[1:])
        or layer_ids.device != scores.device
        or coordinate_ids.device != scores.device
    ):
        raise RuntimeError("fixed-probe dtype or device inventory changed")
    if local_p:
        if (
            int(layer_ids.amin()) < 0
            or int(layer_ids.amax()) >= total_layers
            or int(coordinate_ids.amin()) < 0
            or int(coordinate_ids.amax()) >= total_coordinates
        ):
            raise RuntimeError("fixed-probe ID lies outside the global lattice")
    if float(eta) <= 0.0:
        raise RuntimeError("fixed-probe eta must be positive")
    if any(not bool(torch.isfinite(value).all()) for value in floating):
        raise RuntimeError("fixed-probe input is non-finite")
    if local_p and not bool((weights > 0.0).all()):
        raise RuntimeError("fixed-probe budget weights must be positive")
    return probes, local_p


def distributed_fixed_probe_transaction(
    scores: torch.Tensor,
    exact_by_role: torch.Tensor,
    momentum_by_role: torch.Tensor,
    decay_cross: torch.Tensor,
    weights: torch.Tensor,
    layer_ids: torch.Tensor,
    coordinate_ids: torch.Tensor,
    *,
    total_coordinates: int,
    total_layers: int,
    eta: float,
    rounds: int = 64,
    group=None,
    local_complete_replica: bool = False,
) -> FixedProbeTransactionResult:
    """Select one global equal-budget update from fixed functional probes.

    Logical coordinates must be disjoint across the reduction group.  A
    separate fragment router may sum contributions from arbitrary parameter
    shards into that canonical representation before this call.  The
    ``local_complete_replica`` systems flag is reserved for an adapter that
    has already proved it holds every canonical coordinate; it suppresses
    collectives without inferring replication from a process-group identity.
    """

    probes, local_p = _validate(
        scores,
        exact_by_role,
        momentum_by_role,
        decay_cross,
        weights,
        layer_ids,
        coordinate_ids,
        total_coordinates=int(total_coordinates),
        total_layers=int(total_layers),
        eta=float(eta),
    )
    if rounds < 1:
        raise RuntimeError("fixed-probe secular solve needs at least one round")
    if local_complete_replica:
        expected_ids = torch.arange(
            int(total_coordinates), device=coordinate_ids.device,
            dtype=torch.int64,
        )
        if local_p != int(total_coordinates) or not coordinate_ids.equal(
            expected_ids
        ):
            raise RuntimeError("complete-replica certificate does not cover lattice")

    # All numerically sensitive global geometry is fixed K-dimensional FP64.
    score64 = scores.double()
    exact64 = exact_by_role.double()
    momentum64 = momentum_by_role.double()
    decay64 = decay_cross.double()
    weight64 = weights.double()
    inverse_root = torch.rsqrt(weight64)
    whitened_scores = score64 * inverse_root.unsqueeze(0)
    parent = torch.sqrt(weight64)
    rhs = (exact64.sum(dim=0) / float(eta) - decay64) * inverse_root

    row_metric = whitened_scores @ whitened_scores.T / float(probes)
    rhs_row = whitened_scores @ rhs
    parent_row = whitened_scores @ parent
    parent_exact_layers = torch.zeros(
        (2, total_layers), device=scores.device, dtype=torch.float64
    )
    parent_momentum_layers = torch.zeros_like(parent_exact_layers)
    if local_p:
        parent_exact_layers.index_add_(1, layer_ids, exact64)
        parent_momentum_layers.index_add_(1, layer_ids, momentum64)
    first = torch.cat((
        row_metric.reshape(-1),
        rhs_row,
        parent_row,
        rhs.square().sum().reshape(1),
        parent.square().sum().reshape(1),
        exact64.sum().reshape(1),
        decay64.sum().reshape(1),
        torch.tensor([float(local_p)], device=scores.device, dtype=torch.float64),
        parent_exact_layers.reshape(-1),
        parent_momentum_layers.reshape(-1),
    ))
    if not local_complete_replica:
        _all_reduce(first, group)
    offset = 0
    global_row_metric = first[offset : offset + probes * probes].view(probes, probes)
    offset += probes * probes
    global_rhs_row = first[offset : offset + probes]
    offset += probes
    global_parent_row = first[offset : offset + probes]
    offset += probes
    rhs_norm_squared = first[offset]
    parent_budget = first[offset + 1]
    parent_exact = first[offset + 2]
    parent_decay = first[offset + 3]
    observed_coordinates = int(round(float(first[offset + 4])))
    offset += 5
    global_parent_exact_layers = first[
        offset : offset + 2 * total_layers
    ].view(2, total_layers)
    offset += 2 * total_layers
    global_parent_momentum_layers = first[
        offset : offset + 2 * total_layers
    ].view(2, total_layers)
    if observed_coordinates != int(total_coordinates):
        raise RuntimeError(
            "fixed-probe canonical shards do not cover the global coordinate count"
        )
    global_row_metric = 0.5 * (
        global_row_metric + global_row_metric.T
    )
    eigenvalues, left_vectors = torch.linalg.eigh(global_row_metric)
    machine = torch.finfo(torch.float64).eps
    tiny = torch.finfo(torch.float64).tiny
    spectral_scale = eigenvalues.abs().amax()
    rank_threshold = (
        machine
        * float(total_coordinates)
        * spectral_scale.clamp_min(tiny)
    )
    retained = eigenvalues > rank_threshold
    safe_eigenvalues = torch.where(
        retained, eigenvalues, torch.ones_like(eigenvalues)
    )
    root_spectrum = torch.sqrt(float(probes) * safe_eigenvalues)
    rhs_coordinates = (left_vectors.T @ global_rhs_row) / root_spectrum
    rhs_coordinates = torch.where(
        retained, rhs_coordinates, torch.zeros_like(rhs_coordinates)
    )
    parent_coordinates = (left_vectors.T @ global_parent_row) / root_spectrum
    parent_coordinates = torch.where(
        retained, parent_coordinates, torch.zeros_like(parent_coordinates)
    )
    rhs_null_norm_squared = (
        rhs_norm_squared - rhs_coordinates.square().sum()
    ).clamp_min(0.0)

    lower = rank_threshold.reshape(1)
    upper = (
        torch.sqrt(rhs_norm_squared) / torch.sqrt(parent_budget)
        + spectral_scale
        + rank_threshold
    ).reshape(1)
    eigen_batch = eigenvalues.unsqueeze(0)
    rhs_batch = rhs_coordinates.unsqueeze(0)
    null_batch = rhs_null_norm_squared.reshape(1)
    budget_batch = parent_budget.reshape(1)
    lower_norm = (
        rhs_coordinates.square()
        / (eigenvalues + lower[0]).clamp_min(rank_threshold).square()
    ).sum() + rhs_null_norm_squared / lower[0].square()
    hard_case = (lower_norm < parent_budget).reshape(1)
    multiplier = _fixed_bisection(
        eigen_batch,
        rhs_batch,
        null_batch,
        budget_batch,
        lower,
        upper,
        rounds=rounds,
    )[0]

    right = (whitened_scores.T @ left_vectors) / root_spectrum.unsqueeze(0)
    right = torch.where(
        retained.unsqueeze(0), right, torch.zeros_like(right)
    )
    root_row_coordinates = rhs_coordinates / (
        eigenvalues + multiplier
    ).clamp_min(rank_threshold)
    rhs_parallel_local = right @ rhs_coordinates
    root_local = right @ root_row_coordinates + (
        rhs - rhs_parallel_local
    ) / multiplier

    hard_row_coordinates = torch.where(
        retained,
        rhs_coordinates / safe_eigenvalues,
        torch.zeros_like(rhs_coordinates),
    )
    hard_base_local = right @ hard_row_coordinates
    parent_null_local = parent - right @ parent_coordinates
    parent_null_norm_squared = (
        parent_budget - parent_coordinates.square().sum()
    ).clamp_min(0.0)
    hard_base_norm_squared = hard_row_coordinates.square().sum()
    remaining = (parent_budget - hard_base_norm_squared).clamp_min(0.0)
    parent_null_usable = parent_null_norm_squared > rank_threshold
    hard_local = hard_base_local + (
        torch.sqrt(remaining)
        * parent_null_local
        / torch.sqrt(parent_null_norm_squared).clamp_min(
            torch.sqrt(rank_threshold)
        )
    )
    candidate_x = torch.where(hard_case, hard_local, root_local)
    if bool(hard_case) and not bool(parent_null_usable):
        candidate_x = parent.clone()
    candidate_coefficients64 = candidate_x * inverse_root

    candidate_score_action = score64 @ candidate_coefficients64
    candidate_exact_layers = torch.zeros_like(parent_exact_layers)
    candidate_momentum_layers = torch.zeros_like(parent_momentum_layers)
    if local_p:
        candidate_exact_layers.index_add_(
            1, layer_ids, exact64 * candidate_coefficients64.unsqueeze(0)
        )
        candidate_momentum_layers.index_add_(
            1, layer_ids, momentum64 * candidate_coefficients64.unsqueeze(0)
        )
    second = torch.cat((
        candidate_score_action,
        (exact64.sum(dim=0) * candidate_coefficients64).sum().reshape(1),
        (decay64 * candidate_coefficients64).sum().reshape(1),
        (weight64 * candidate_coefficients64.square()).sum().reshape(1),
        candidate_exact_layers.reshape(-1),
        candidate_momentum_layers.reshape(-1),
    ))
    if not local_complete_replica:
        _all_reduce(second, group)
    offset = 0
    global_candidate_score_action = second[offset : offset + probes]
    offset += probes
    candidate_exact = second[offset]
    candidate_decay = second[offset + 1]
    candidate_budget = second[offset + 2]
    offset += 3
    global_candidate_exact_layers = second[
        offset : offset + 2 * total_layers
    ].view(2, total_layers)
    offset += 2 * total_layers
    global_candidate_momentum_layers = second[
        offset : offset + 2 * total_layers
    ].view(2, total_layers)

    parent_score = (
        -float(eta) * parent_exact
        + 0.5 * float(eta) ** 2
        * (global_parent_row.square().mean() + 2.0 * parent_decay)
    )
    candidate_score = (
        -float(eta) * candidate_exact
        + 0.5 * float(eta) ** 2
        * (
            global_candidate_score_action.square().mean()
            + 2.0 * candidate_decay
        )
    )
    budget_residual = (
        (candidate_budget - parent_budget).abs()
        / parent_budget.clamp_min(1.0)
    )
    finite = (
        torch.isfinite(candidate_coefficients64).all()
        & torch.isfinite(candidate_score)
        & torch.isfinite(budget_residual)
        & torch.isfinite(global_candidate_exact_layers).all()
        & torch.isfinite(global_candidate_momentum_layers).all()
    )
    accepted = (
        finite
        & parent_null_usable.logical_or(~hard_case[0])
        & (candidate_score < parent_score)
        & (budget_residual <= 1.0e-8)
        & (global_candidate_exact_layers > 0.0).all()
        & (global_candidate_momentum_layers > 0.0).all()
        & (global_parent_exact_layers > 0.0).all()
        & (global_parent_momentum_layers > 0.0).all()
    ).reshape(1)
    selected = torch.where(
        accepted, candidate_coefficients64, torch.ones_like(candidate_coefficients64)
    ).to(scores.dtype)
    candidate = candidate_coefficients64.to(scores.dtype)
    return FixedProbeTransactionResult(
        local_coefficients=selected,
        local_candidate_coefficients=candidate,
        accepted=accepted,
        multiplier=torch.where(
            hard_case, torch.zeros_like(multiplier).reshape(1), multiplier.reshape(1)
        ),
        rank=retained.sum().reshape(1),
        eigenvalue_max=spectral_scale.reshape(1),
        hard_case=hard_case,
        parent_score=parent_score.reshape(1),
        candidate_score=candidate_score.reshape(1),
        budget_residual=budget_residual.reshape(1),
        local_coordinate_count=local_p,
        global_coordinate_count=int(total_coordinates),
        global_probe_count=probes,
        collective_rounds=0 if local_complete_replica else 2,
        summary_elements=(
            probes * probes + 3 * probes + 8 * total_layers + 8
        ),
        owner_count=0,
        dense_LG_by_LG_metric_elements=0,
        selected_update_elements_published=0,
        method_state_depends_on_total_tokens=False,
    )


def replicated_fixed_probe_transaction(
    local_scores: torch.Tensor,
    local_decay_action: torch.Tensor,
    exact_by_role: torch.Tensor,
    momentum_by_role: torch.Tensor,
    weights: torch.Tensor,
    layer_ids: torch.Tensor,
    *,
    global_probe_count: int,
    total_layers: int,
    eta: float,
    rounds: int = 64,
    group=None,
) -> ReplicatedFixedProbeTransactionResult:
    """DDP adapter for the same column-sharded fixed-probe transaction.

    Every DDP rank starts with different probe rows and a replicated logical
    coordinate lattice.  Probe rows are gathered, coordinates are partitioned
    deterministically, and only the final scalar coefficients are replicated
    again.  The selected matrix update is never communicated.

    This function exists for matched endpoint experiments on the current DDP
    trainer.  A sharded production implementation must call
    ``distributed_fixed_probe_transaction`` on native coordinate shards and
    therefore omits both replication exchanges performed here.
    """

    if local_scores.ndim != 2:
        raise RuntimeError("replicated fixed-probe scores must be [local_K, p]")
    local_rows, coordinates = local_scores.shape
    if local_decay_action.shape != (local_rows,):
        raise RuntimeError("replicated fixed-probe decay action changed")
    if exact_by_role.shape != (2, coordinates):
        raise RuntimeError("replicated fixed-probe exact inventory changed")
    if momentum_by_role.shape != exact_by_role.shape:
        raise RuntimeError("replicated fixed-probe momentum inventory changed")
    if weights.shape != (coordinates,) or layer_ids.shape != (coordinates,):
        raise RuntimeError("replicated fixed-probe coordinate inventory changed")
    if coordinates < 1:
        raise RuntimeError("fixed-probe coordinate count must be positive")
    if layer_ids.dtype != torch.int64:
        raise RuntimeError("replicated fixed-probe layer IDs must be int64")
    floating = (
        local_scores,
        local_decay_action,
        exact_by_role,
        momentum_by_role,
        weights,
    )
    if (
        any(not value.is_floating_point() for value in floating)
        or any(value.device != local_scores.device for value in floating[1:])
        or any(value.dtype != local_scores.dtype for value in floating[1:])
        or layer_ids.device != local_scores.device
    ):
        raise RuntimeError("replicated fixed-probe dtype or device changed")

    # One packet preserves the exact association between each functional row
    # and the weight-decay action measured on that same row.
    local_packet = torch.cat((
        local_scores,
        local_decay_action.unsqueeze(1),
    ), dim=1)
    global_packet, gather_rounds = _gather_variable_probe_rows(
        local_packet,
        expected_global_rows=int(global_probe_count),
        group=group,
    )
    global_scores = global_packet[:, :coordinates]
    global_decay_action = global_packet[:, coordinates]
    decay_cross = (
        global_scores.T @ global_decay_action / float(global_probe_count)
    )
    total_row_metric = global_scores @ global_scores.T
    total_metric_square = total_row_metric.square().sum()
    within_layer_square = torch.zeros_like(total_metric_square)
    for layer in range(int(total_layers)):
        layer_scores = global_scores[:, layer_ids.eq(layer)]
        layer_row_metric = layer_scores @ layer_scores.T
        within_layer_square = within_layer_square + layer_row_metric.square().sum()
    cross_layer_coupling_ratio = torch.sqrt(
        (total_metric_square - within_layer_square).clamp_min(0.0)
        / total_metric_square.clamp_min(torch.finfo(total_metric_square.dtype).tiny)
    )

    rank, world = _distributed_shape(group)
    coordinate_ids = torch.arange(
        coordinates, device=local_scores.device, dtype=torch.int64
    )
    local_coordinate_ids = coordinate_ids[
        coordinate_ids.remainder(world).eq(rank)
    ]
    sharded = distributed_fixed_probe_transaction(
        global_scores[:, local_coordinate_ids],
        exact_by_role[:, local_coordinate_ids],
        momentum_by_role[:, local_coordinate_ids],
        decay_cross[local_coordinate_ids],
        weights[local_coordinate_ids],
        layer_ids[local_coordinate_ids],
        local_coordinate_ids,
        total_coordinates=coordinates,
        total_layers=int(total_layers),
        eta=float(eta),
        rounds=int(rounds),
        group=group,
    )

    coefficient_packet = torch.zeros(
        2 * coordinates, device=local_scores.device, dtype=local_scores.dtype
    )
    coefficient_packet[local_coordinate_ids] = sharded.local_coefficients
    coefficient_packet[
        coordinates + local_coordinate_ids
    ] = sharded.local_candidate_coefficients
    if world > 1:
        dist.all_reduce(coefficient_packet, op=dist.ReduceOp.SUM, group=group)
    coefficients = coefficient_packet[:coordinates]
    candidate_coefficients = coefficient_packet[coordinates:]
    return ReplicatedFixedProbeTransactionResult(
        coefficients=coefficients,
        candidate_coefficients=candidate_coefficients,
        sharded_result=sharded,
        local_probe_count=local_rows,
        global_probe_count=int(global_probe_count),
        cross_layer_coupling_ratio=cross_layer_coupling_ratio,
        collective_rounds=gather_rounds + sharded.collective_rounds + int(world > 1),
        score_scalars_exchanged_per_rank=int(global_probe_count) * (coordinates + 1),
        coefficient_scalars_exchanged_per_rank=(2 * coordinates if world > 1 else 0),
        selected_update_elements_published=0,
        method_state_depends_on_total_tokens=False,
    )


__all__ = (
    "FixedProbeTransactionResult",
    "ReplicatedFixedProbeTransactionResult",
    "distributed_fixed_probe_transaction",
    "replicated_fixed_probe_transaction",
)
