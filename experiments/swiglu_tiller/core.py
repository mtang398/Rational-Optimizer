"""TILLER: loss-informed equal-budget tangent reweighting."""
from __future__ import annotations
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch.distributed as dist
_basis_trust___NS_COEFFICIENTS = (3.4445, -4.775, 2.0315)
_basis_trust___MUON_EPS = 1e-07

def _basis_trust___zeropower_via_newton_schulz(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Literal torch.optim.Muon NS path for the pinned runtime."""
    if matrix.ndim != 2 or steps != 5:
        raise RuntimeError('TILLER requires a 2D NS5 input')
    a, b, c = _basis_trust___NS_COEFFICIENTS
    value = matrix.bfloat16()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.T
    value.div_(value.norm().clamp(min=_basis_trust___MUON_EPS))
    for _ in range(steps):
        gram = value @ value.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        value = torch.addmm(value, gram_update, value, beta=a)
    return value.T if transposed else value

def _basis_trust___match_rms_adamw_adjustment(shape: torch.Size) -> float:
    if len(shape) != 2:
        raise RuntimeError('TILLER Muon adjustment requires a matrix')
    return 0.2 * math.sqrt(float(max(shape)))
_diagonal_completed_biatlas_metric__CURRENT_ROWS = 32
_diagonal_completed_biatlas_metric__PERSISTENT_ROWS = 64
_diagonal_completed_biatlas_metric__MAXIMUM_SELECTION_ROWS = 96
_diagonal_completed_biatlas_metric__KRYLOV_ROWS = 32
_diagonal_completed_biatlas_metric__MATCHED_BETA2 = 0.95

@dataclass(frozen=True)
class _diagonal_completed_biatlas_metric__DiagonalCompletedBiatlasRows:
    selection_scores: torch.Tensor
    selection_diagonal_residual: torch.Tensor
    persistent_scores: torch.Tensor
    persistent_total_diagonal: torch.Tensor
    persistent_diagonal_residual: torch.Tensor
    persistent_decay_cross: torch.Tensor
    history_used: bool
    discarded_energy_fraction: torch.Tensor

@dataclass(frozen=True)
class _diagonal_completed_biatlas_metric__DiagonalCompletedTransactionResult:
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

def _diagonal_completed_biatlas_metric___metric_action(low_rank: torch.Tensor, diagonal: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return diagonal * value + low_rank.T @ (low_rank @ value)

def _diagonal_completed_biatlas_metric___loss_krylov_basis(low_rank: torch.Tensor, diagonal: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-width fully reorthogonalized Lanczos loss subspace."""
    coordinates = int(rhs.numel())
    steps = min(_diagonal_completed_biatlas_metric__KRYLOV_ROWS, coordinates)
    tiny = torch.finfo(rhs.dtype).tiny
    norm = torch.linalg.vector_norm(rhs)
    torch._assert_async(torch.isfinite(norm) & (norm > 0.0))
    basis = []
    alpha_values = []
    beta_values = []
    current = rhs / norm.clamp_min(tiny)
    previous = torch.zeros_like(current)
    previous_beta = rhs.new_zeros(())
    coordinate = torch.arange(coordinates, device=rhs.device, dtype=rhs.dtype) + 1.0
    for index in range(steps):
        basis.append(current)
        action = _diagonal_completed_biatlas_metric___metric_action(low_rank, diagonal, current)
        alpha = torch.dot(current, action)
        residual = action - alpha * current - previous_beta * previous
        for _ in range(2):
            active_basis = torch.stack(basis, dim=1)
            residual = residual - active_basis @ (active_basis.T @ residual)
        beta = torch.linalg.vector_norm(residual)
        alpha_values.append(alpha)
        if index + 1 < steps:
            beta_values.append(beta)
            threshold = 64.0 * torch.finfo(rhs.dtype).eps * torch.linalg.vector_norm(action).clamp_min(1.0)
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
    orthogonality = torch.linalg.vector_norm(q.T @ q - torch.eye(steps, device=q.device, dtype=q.dtype))
    torch._assert_async(torch.isfinite(q).all() & torch.isfinite(tridiagonal).all() & (orthogonality <= 2e-10 * float(steps)))
    return (q, tridiagonal)
_diagonal_completed_biatlas_metric___COMPILED_LOSS_KRYLOV_BASIS = torch.compile(_diagonal_completed_biatlas_metric___loss_krylov_basis, fullgraph=True, dynamic=False)

def _diagonal_completed_biatlas_metric__diagonal_completed_biatlas_transaction(rows: _diagonal_completed_biatlas_metric__DiagonalCompletedBiatlasRows, exact_by_role: torch.Tensor, momentum_by_role: torch.Tensor, weights: torch.Tensor, layer_ids: torch.Tensor, *, total_layers: int, eta: float, rounds: int=64, diagnostics: bool=True) -> _diagonal_completed_biatlas_metric__DiagonalCompletedTransactionResult:
    """Select a same-budget signed update in diagonal-plus-row-space metric."""
    scores = rows.selection_scores
    diagonal_residual = rows.selection_diagonal_residual
    decay_cross = rows.persistent_decay_cross
    factor_rows, coordinates = scores.shape
    measure_rows = int(getattr(rows, 'measure_rows', _diagonal_completed_biatlas_metric__CURRENT_ROWS))
    maximum_selection_rows = int(getattr(rows, 'maximum_selection_rows', _diagonal_completed_biatlas_metric__MAXIMUM_SELECTION_ROWS))
    if measure_rows < 1 or factor_rows < measure_rows or factor_rows > maximum_selection_rows or (maximum_selection_rows < factor_rows) or (coordinates < 2) or (diagonal_residual.shape != (coordinates,)) or (decay_cross.shape != (coordinates,)) or (exact_by_role.shape != (2, coordinates)) or (momentum_by_role.shape != exact_by_role.shape) or (weights.shape != (coordinates,)) or (layer_ids.shape != (coordinates,)) or (layer_ids.dtype != torch.int64) or (int(total_layers) < 1) or (float(eta) <= 0.0) or (int(rounds) < 1):
        raise RuntimeError('diagonal-completed transaction inventory changed')
    floating = (scores, diagonal_residual, decay_cross, exact_by_role, momentum_by_role, weights)
    if any((not value.is_floating_point() for value in floating)) or any((value.dtype != scores.dtype for value in floating[1:])) or any((value.device != scores.device for value in floating[1:])) or (layer_ids.device != scores.device):
        raise RuntimeError('diagonal-completed transaction values changed')
    valid_values = torch.isfinite(scores).all()
    for value in floating[1:]:
        valid_values = valid_values & torch.isfinite(value).all()
    valid_values = valid_values & (diagonal_residual >= 0.0).all() & (weights > 0.0).all()
    if not bool(valid_values):
        raise RuntimeError('diagonal-completed transaction values changed')
    if coordinates and (not bool(((layer_ids >= 0) & (layer_ids < int(total_layers))).all())):
        raise RuntimeError('diagonal-completed layer ID is invalid')
    score64 = scores.double()
    diagonal64 = diagonal_residual.double()
    decay64 = decay_cross.double()
    exact64 = exact_by_role.double()
    momentum64 = momentum_by_role.double()
    weight64 = weights.double()
    root_weight = torch.sqrt(weight64)
    inverse_root_weight = torch.reciprocal(root_weight)
    low_rank = score64 * inverse_root_weight.unsqueeze(0) / math.sqrt(measure_rows)
    if factor_rows < maximum_selection_rows:
        low_rank = torch.cat((low_rank, low_rank.new_zeros(maximum_selection_rows - factor_rows, coordinates)), dim=0)
    diagonal_whitened = diagonal64 / weight64
    rhs = (exact64.sum(dim=0) / float(eta) - decay64) * inverse_root_weight
    parent = root_weight
    budget = weight64.sum()
    tiny = torch.finfo(torch.float64).tiny
    machine = torch.finfo(torch.float64).eps
    krylov = _diagonal_completed_biatlas_metric___COMPILED_LOSS_KRYLOV_BASIS if scores.is_cuda else _diagonal_completed_biatlas_metric___loss_krylov_basis
    basis, tridiagonal = krylov(low_rank, diagonal_whitened, rhs)
    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    eigenvalues = eigenvalues.clamp_min(0.0)
    projected_rhs = eigenvectors.T @ (basis.T @ rhs)
    lower = torch.tensor(machine * float(coordinates), device=scores.device, dtype=torch.float64)
    rhs_norm = torch.linalg.vector_norm(rhs)
    upper = rhs_norm / torch.sqrt(budget) + eigenvalues.amax() + lower
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
    candidate_budget = (weight64 * candidate_coefficients64.square()).sum()
    budget_residual = (candidate_budget - budget).abs() / budget.clamp_min(1.0)
    parent_coefficients64 = torch.ones_like(candidate_coefficients64)
    parent_action = score64 @ parent_coefficients64
    candidate_action = score64 @ candidate_coefficients64
    parent_exact = exact64.sum()
    candidate_exact = (exact64.sum(dim=0) * candidate_coefficients64).sum()
    parent_score = -float(eta) * parent_exact + 0.5 * float(eta) ** 2 * (parent_action.square().sum() / float(measure_rows) + (diagonal64 * parent_coefficients64.square()).sum() + 2.0 * decay64.sum())
    candidate_score = -float(eta) * candidate_exact + 0.5 * float(eta) ** 2 * (candidate_action.square().sum() / float(measure_rows) + (diagonal64 * candidate_coefficients64.square()).sum() + 2.0 * (decay64 * candidate_coefficients64).sum())
    parent_exact_layers = torch.zeros((2, int(total_layers)), device=scores.device, dtype=torch.float64)
    parent_momentum_layers = torch.zeros_like(parent_exact_layers)
    candidate_exact_layers = torch.zeros_like(parent_exact_layers)
    candidate_momentum_layers = torch.zeros_like(parent_exact_layers)
    parent_exact_layers.index_add_(1, layer_ids, exact64)
    parent_momentum_layers.index_add_(1, layer_ids, momentum64)
    candidate_exact_layers.index_add_(1, layer_ids, exact64 * candidate_coefficients64.unsqueeze(0))
    candidate_momentum_layers.index_add_(1, layer_ids, momentum64 * candidate_coefficients64.unsqueeze(0))
    finite = torch.isfinite(candidate_coefficients64).all() & torch.isfinite(candidate_score) & torch.isfinite(budget_residual)
    accepted = (finite & ~hard_case & (candidate_score < parent_score) & (budget_residual <= 1e-08) & (candidate_exact_layers > 0.0).all() & (candidate_momentum_layers > 0.0).all() & (parent_exact_layers > 0.0).all() & (parent_momentum_layers > 0.0).all()).reshape(1)
    candidate_coefficients = candidate_coefficients64.to(scores.dtype)
    selected = torch.where(accepted, candidate_coefficients, torch.ones_like(candidate_coefficients))
    if diagnostics:
        singular_values = torch.linalg.svdvals(low_rank)
        rank_threshold = machine * float(coordinates) * singular_values.amax().clamp_min(tiny)
        factor_rank = (singular_values > rank_threshold).sum().reshape(1)
        measure = float(measure_rows)
        total_row_metric = score64 @ score64.T
        total_metric_square = total_row_metric.square().sum() / (measure * measure) + 2.0 * (diagonal64 * score64.square().sum(dim=0) / measure).sum() + diagonal64.square().sum()
        within_layer_square = torch.zeros_like(total_metric_square)
        for layer in range(int(total_layers)):
            mask = layer_ids.eq(layer)
            layer_factor = score64[:, mask]
            layer_diagonal = diagonal64[mask]
            layer_row_metric = layer_factor @ layer_factor.T
            within_layer_square = within_layer_square + (layer_row_metric.square().sum() / (measure * measure) + 2.0 * (layer_diagonal * layer_factor.square().sum(dim=0) / measure).sum() + layer_diagonal.square().sum())
        cross_layer_coupling_ratio = torch.sqrt((total_metric_square - within_layer_square).clamp_min(0.0) / total_metric_square.clamp_min(tiny)).reshape(1)
        diagonal_minimum = diagonal64.amin().reshape(1)
        diagonal_median = diagonal64.median().reshape(1)
        diagonal_maximum = diagonal64.amax().reshape(1)
    else:
        # These fields are observability-only. The accepted coefficients and
        # every quantity used to decide them have already been computed.
        factor_rank = torch.zeros((1,), device=scores.device, dtype=torch.int64)
        diagnostic_zero = score64.new_zeros((1,))
        diagonal_minimum = diagnostic_zero
        diagonal_median = diagnostic_zero
        diagonal_maximum = diagnostic_zero
        cross_layer_coupling_ratio = diagnostic_zero
    return _diagonal_completed_biatlas_metric__DiagonalCompletedTransactionResult(coefficients=selected, candidate_coefficients=candidate_coefficients, accepted=accepted, multiplier=torch.where(hard_case, torch.zeros_like(hi), hi).reshape(1), hard_case=hard_case.reshape(1), budget_residual=budget_residual.reshape(1), parent_score=parent_score.reshape(1), candidate_score=candidate_score.reshape(1), factor_rank=factor_rank, diagonal_minimum=diagonal_minimum, diagonal_median=diagonal_median, diagonal_maximum=diagonal_maximum, cross_layer_coupling_ratio=cross_layer_coupling_ratio, dense_coordinate_metric_elements=0, largest_dense_solve_dimension=int(factor_rows), selected_update_elements_published=0, owner_count=0)
_functional_row__FIXED_GLOBAL_PROBE_COUNT = 32

@dataclass(frozen=True)
class _fixed_global_probe_layout__FixedGlobalProbeLayout:
    global_probe_count: int
    process_count: int
    process_rank: int
    global_start: int
    global_stop: int
    local_probe_count: int
    method_state_depends_on_total_tokens: bool
    method_state_depends_on_machine_count: bool

def _fixed_global_probe_layout__fixed_global_probe_layout(global_probe_count: int, process_rank: int, process_count: int) -> _fixed_global_probe_layout__FixedGlobalProbeLayout:
    """Partition fixed probe IDs evenly, allowing empty high-rank shards."""
    probes = int(global_probe_count)
    rank = int(process_rank)
    world = int(process_count)
    if probes < 2 or world < 1 or rank < 0 or (rank >= world):
        raise RuntimeError('fixed global probe layout is invalid')
    start = rank * probes // world
    stop = (rank + 1) * probes // world
    return _fixed_global_probe_layout__FixedGlobalProbeLayout(global_probe_count=probes, process_count=world, process_rank=rank, global_start=start, global_stop=stop, local_probe_count=stop - start, method_state_depends_on_total_tokens=False, method_state_depends_on_machine_count=False)

def _fixed_global_probe_layout__evenly_spaced_indices(row_count: int, selected_count: int, *, device: torch.device) -> torch.Tensor:
    """Deterministically select 0, 1, or more rows without division hazards."""
    rows = int(row_count)
    selected = int(selected_count)
    if rows < 0 or selected < 0 or selected > rows:
        raise RuntimeError('fixed probe row selection is invalid')
    if selected == 0:
        return torch.empty(0, device=device, dtype=torch.int64)
    if selected == 1:
        return torch.zeros(1, device=device, dtype=torch.int64)
    numerators = torch.arange(selected, device=device, dtype=torch.int64) * (rows - 1)
    return torch.div(numerators, selected - 1, rounding_mode='floor')

def _fixed_probe_transaction___fixed_bisection(eigenvalues: torch.Tensor, rhs_coordinates: torch.Tensor, rhs_null_norm_squared: torch.Tensor, budget: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, *, rounds: int) -> torch.Tensor:
    """Deterministic secular solve used by the fixed-probe adapter."""
    if rounds < 1:
        raise RuntimeError('TILLER secular solve needs at least one round')
    for _ in range(rounds):
        middle = 0.5 * (lower + upper)
        row_norm_squared = (rhs_coordinates.square() / (eigenvalues + middle[:, None]).square()).sum(dim=-1)
        null_norm_squared = rhs_null_norm_squared / middle.square()
        too_large = row_norm_squared + null_norm_squared > budget
        lower = torch.where(too_large, middle, lower)
        upper = torch.where(too_large, upper, middle)
    return upper

@dataclass(frozen=True)
class _fixed_probe_transaction__FixedProbeTransactionResult:
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
class _fixed_probe_transaction__ReplicatedFixedProbeTransactionResult:
    """Full coefficients for a replicated-model endpoint adapter.

    The mathematical transaction remains column sharded.  This adapter only
    exchanges the fixed probe-score lattice and the resulting scalar
    coefficients because ordinary DDP replicates every logical coordinate on
    every rank.  TP/FSDP callers should use :func:`distributed_fixed_probe_transaction`
    directly and keep the returned coefficients on their native shards.
    """
    coefficients: torch.Tensor
    candidate_coefficients: torch.Tensor
    sharded_result: _fixed_probe_transaction__FixedProbeTransactionResult
    local_probe_count: int
    global_probe_count: int
    cross_layer_coupling_ratio: torch.Tensor
    collective_rounds: int
    score_scalars_exchanged_per_rank: int
    coefficient_scalars_exchanged_per_rank: int
    selected_update_elements_published: int
    method_state_depends_on_total_tokens: bool

def _fixed_probe_transaction___all_reduce(value: torch.Tensor, group) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)

def _fixed_probe_transaction___distributed_shape(group) -> tuple[int, int]:
    if not (dist.is_available() and dist.is_initialized()):
        return (0, 1)
    return (dist.get_rank(group=group), dist.get_world_size(group=group))

def _fixed_probe_transaction___gather_variable_probe_rows(local_packet: torch.Tensor, *, expected_global_rows: int, group) -> tuple[torch.Tensor, int]:
    """Gather a globally fixed number of unevenly partitioned probe rows."""
    if local_packet.ndim != 2:
        raise RuntimeError('fixed-probe row packet must be rank two')
    rank, world = _fixed_probe_transaction___distributed_shape(group)
    del rank
    local_rows, columns = local_packet.shape
    if world == 1:
        if local_rows != int(expected_global_rows):
            raise RuntimeError('fixed-probe global row count changed')
        return (local_packet, 0)
    count = torch.tensor([local_rows], device=local_packet.device, dtype=torch.int64)
    counts = [torch.empty_like(count) for _ in range(world)]
    dist.all_gather(counts, count, group=group)
    row_counts = [int(value.item()) for value in counts]
    if sum(row_counts) != int(expected_global_rows):
        raise RuntimeError('fixed-probe rows must sum to the globally fixed probe count')
    maximum = max(row_counts)
    padded = torch.zeros((maximum, columns), device=local_packet.device, dtype=local_packet.dtype)
    if local_rows:
        padded[:local_rows].copy_(local_packet)
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    return (torch.cat([value[:rows] for value, rows in zip(gathered, row_counts)], dim=0), 2)

def _fixed_probe_transaction___validate(scores, exact_by_role, momentum_by_role, decay_cross, weights, layer_ids, coordinate_ids, *, total_coordinates, total_layers, eta):
    if scores.ndim != 2:
        raise RuntimeError('fixed-probe scores must have shape [K, local_p]')
    probes, local_p = scores.shape
    if probes < 2 or total_coordinates < 1 or total_layers < 1:
        raise RuntimeError('fixed-probe logical dimensions are invalid')
    if exact_by_role.shape != (2, local_p) or momentum_by_role.shape != (2, local_p):
        raise RuntimeError('fixed-probe role inventory changed')
    if decay_cross.shape != (local_p,) or weights.shape != (local_p,):
        raise RuntimeError('fixed-probe scalar inventory changed')
    if layer_ids.shape != (local_p,) or coordinate_ids.shape != (local_p,):
        raise RuntimeError('fixed-probe coordinate inventory changed')
    if layer_ids.dtype != torch.int64 or coordinate_ids.dtype != torch.int64:
        raise RuntimeError('fixed-probe IDs must be int64')
    floating = (scores, exact_by_role, momentum_by_role, decay_cross, weights)
    if not scores.is_floating_point() or any((value.dtype != scores.dtype for value in floating[1:])) or any((value.device != scores.device for value in floating[1:])) or (layer_ids.device != scores.device) or (coordinate_ids.device != scores.device):
        raise RuntimeError('fixed-probe dtype or device inventory changed')
    if local_p:
        if int(layer_ids.amin()) < 0 or int(layer_ids.amax()) >= total_layers or int(coordinate_ids.amin()) < 0 or (int(coordinate_ids.amax()) >= total_coordinates):
            raise RuntimeError('fixed-probe ID lies outside the global lattice')
    if float(eta) <= 0.0:
        raise RuntimeError('fixed-probe eta must be positive')
    if any((not bool(torch.isfinite(value).all()) for value in floating)):
        raise RuntimeError('fixed-probe input is non-finite')
    if local_p and (not bool((weights > 0.0).all())):
        raise RuntimeError('fixed-probe budget weights must be positive')
    return (probes, local_p)

def _fixed_probe_transaction__distributed_fixed_probe_transaction(scores: torch.Tensor, exact_by_role: torch.Tensor, momentum_by_role: torch.Tensor, decay_cross: torch.Tensor, weights: torch.Tensor, layer_ids: torch.Tensor, coordinate_ids: torch.Tensor, *, total_coordinates: int, total_layers: int, eta: float, rounds: int=64, group=None, local_complete_replica: bool=False) -> _fixed_probe_transaction__FixedProbeTransactionResult:
    """Select one global equal-budget update from fixed functional probes.

    Logical coordinates must be disjoint across the reduction group.  A
    separate fragment router may sum contributions from arbitrary parameter
    shards into that canonical representation before this call.  The
    ``local_complete_replica`` systems flag is reserved for an adapter that
    has already proved it holds every canonical coordinate; it suppresses
    collectives without inferring replication from a process-group identity.
    """
    probes, local_p = _fixed_probe_transaction___validate(scores, exact_by_role, momentum_by_role, decay_cross, weights, layer_ids, coordinate_ids, total_coordinates=int(total_coordinates), total_layers=int(total_layers), eta=float(eta))
    if rounds < 1:
        raise RuntimeError('fixed-probe secular solve needs at least one round')
    if local_complete_replica:
        expected_ids = torch.arange(int(total_coordinates), device=coordinate_ids.device, dtype=torch.int64)
        if local_p != int(total_coordinates) or not coordinate_ids.equal(expected_ids):
            raise RuntimeError('complete-replica certificate does not cover lattice')
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
    parent_exact_layers = torch.zeros((2, total_layers), device=scores.device, dtype=torch.float64)
    parent_momentum_layers = torch.zeros_like(parent_exact_layers)
    if local_p:
        parent_exact_layers.index_add_(1, layer_ids, exact64)
        parent_momentum_layers.index_add_(1, layer_ids, momentum64)
    first = torch.cat((row_metric.reshape(-1), rhs_row, parent_row, rhs.square().sum().reshape(1), parent.square().sum().reshape(1), exact64.sum().reshape(1), decay64.sum().reshape(1), torch.tensor([float(local_p)], device=scores.device, dtype=torch.float64), parent_exact_layers.reshape(-1), parent_momentum_layers.reshape(-1)))
    if not local_complete_replica:
        _fixed_probe_transaction___all_reduce(first, group)
    offset = 0
    global_row_metric = first[offset:offset + probes * probes].view(probes, probes)
    offset += probes * probes
    global_rhs_row = first[offset:offset + probes]
    offset += probes
    global_parent_row = first[offset:offset + probes]
    offset += probes
    rhs_norm_squared = first[offset]
    parent_budget = first[offset + 1]
    parent_exact = first[offset + 2]
    parent_decay = first[offset + 3]
    observed_coordinates = int(round(float(first[offset + 4])))
    offset += 5
    global_parent_exact_layers = first[offset:offset + 2 * total_layers].view(2, total_layers)
    offset += 2 * total_layers
    global_parent_momentum_layers = first[offset:offset + 2 * total_layers].view(2, total_layers)
    if observed_coordinates != int(total_coordinates):
        raise RuntimeError('fixed-probe canonical shards do not cover the global coordinate count')
    global_row_metric = 0.5 * (global_row_metric + global_row_metric.T)
    eigenvalues, left_vectors = torch.linalg.eigh(global_row_metric)
    machine = torch.finfo(torch.float64).eps
    tiny = torch.finfo(torch.float64).tiny
    spectral_scale = eigenvalues.abs().amax()
    rank_threshold = machine * float(total_coordinates) * spectral_scale.clamp_min(tiny)
    retained = eigenvalues > rank_threshold
    safe_eigenvalues = torch.where(retained, eigenvalues, torch.ones_like(eigenvalues))
    root_spectrum = torch.sqrt(float(probes) * safe_eigenvalues)
    rhs_coordinates = left_vectors.T @ global_rhs_row / root_spectrum
    rhs_coordinates = torch.where(retained, rhs_coordinates, torch.zeros_like(rhs_coordinates))
    parent_coordinates = left_vectors.T @ global_parent_row / root_spectrum
    parent_coordinates = torch.where(retained, parent_coordinates, torch.zeros_like(parent_coordinates))
    rhs_null_norm_squared = (rhs_norm_squared - rhs_coordinates.square().sum()).clamp_min(0.0)
    lower = rank_threshold.reshape(1)
    upper = (torch.sqrt(rhs_norm_squared) / torch.sqrt(parent_budget) + spectral_scale + rank_threshold).reshape(1)
    eigen_batch = eigenvalues.unsqueeze(0)
    rhs_batch = rhs_coordinates.unsqueeze(0)
    null_batch = rhs_null_norm_squared.reshape(1)
    budget_batch = parent_budget.reshape(1)
    lower_norm = (rhs_coordinates.square() / (eigenvalues + lower[0]).clamp_min(rank_threshold).square()).sum() + rhs_null_norm_squared / lower[0].square()
    hard_case = (lower_norm < parent_budget).reshape(1)
    multiplier = _fixed_probe_transaction___fixed_bisection(eigen_batch, rhs_batch, null_batch, budget_batch, lower, upper, rounds=rounds)[0]
    right = whitened_scores.T @ left_vectors / root_spectrum.unsqueeze(0)
    right = torch.where(retained.unsqueeze(0), right, torch.zeros_like(right))
    root_row_coordinates = rhs_coordinates / (eigenvalues + multiplier).clamp_min(rank_threshold)
    rhs_parallel_local = right @ rhs_coordinates
    root_local = right @ root_row_coordinates + (rhs - rhs_parallel_local) / multiplier
    hard_row_coordinates = torch.where(retained, rhs_coordinates / safe_eigenvalues, torch.zeros_like(rhs_coordinates))
    hard_base_local = right @ hard_row_coordinates
    parent_null_local = parent - right @ parent_coordinates
    parent_null_norm_squared = (parent_budget - parent_coordinates.square().sum()).clamp_min(0.0)
    hard_base_norm_squared = hard_row_coordinates.square().sum()
    remaining = (parent_budget - hard_base_norm_squared).clamp_min(0.0)
    parent_null_usable = parent_null_norm_squared > rank_threshold
    hard_local = hard_base_local + torch.sqrt(remaining) * parent_null_local / torch.sqrt(parent_null_norm_squared).clamp_min(torch.sqrt(rank_threshold))
    candidate_x = torch.where(hard_case, hard_local, root_local)
    if bool(hard_case) and (not bool(parent_null_usable)):
        candidate_x = parent.clone()
    candidate_coefficients64 = candidate_x * inverse_root
    candidate_score_action = score64 @ candidate_coefficients64
    candidate_exact_layers = torch.zeros_like(parent_exact_layers)
    candidate_momentum_layers = torch.zeros_like(parent_momentum_layers)
    if local_p:
        candidate_exact_layers.index_add_(1, layer_ids, exact64 * candidate_coefficients64.unsqueeze(0))
        candidate_momentum_layers.index_add_(1, layer_ids, momentum64 * candidate_coefficients64.unsqueeze(0))
    second = torch.cat((candidate_score_action, (exact64.sum(dim=0) * candidate_coefficients64).sum().reshape(1), (decay64 * candidate_coefficients64).sum().reshape(1), (weight64 * candidate_coefficients64.square()).sum().reshape(1), candidate_exact_layers.reshape(-1), candidate_momentum_layers.reshape(-1)))
    if not local_complete_replica:
        _fixed_probe_transaction___all_reduce(second, group)
    offset = 0
    global_candidate_score_action = second[offset:offset + probes]
    offset += probes
    candidate_exact = second[offset]
    candidate_decay = second[offset + 1]
    candidate_budget = second[offset + 2]
    offset += 3
    global_candidate_exact_layers = second[offset:offset + 2 * total_layers].view(2, total_layers)
    offset += 2 * total_layers
    global_candidate_momentum_layers = second[offset:offset + 2 * total_layers].view(2, total_layers)
    parent_score = -float(eta) * parent_exact + 0.5 * float(eta) ** 2 * (global_parent_row.square().mean() + 2.0 * parent_decay)
    candidate_score = -float(eta) * candidate_exact + 0.5 * float(eta) ** 2 * (global_candidate_score_action.square().mean() + 2.0 * candidate_decay)
    budget_residual = (candidate_budget - parent_budget).abs() / parent_budget.clamp_min(1.0)
    finite = torch.isfinite(candidate_coefficients64).all() & torch.isfinite(candidate_score) & torch.isfinite(budget_residual) & torch.isfinite(global_candidate_exact_layers).all() & torch.isfinite(global_candidate_momentum_layers).all()
    accepted = (finite & parent_null_usable.logical_or(~hard_case[0]) & (candidate_score < parent_score) & (budget_residual <= 1e-08) & (global_candidate_exact_layers > 0.0).all() & (global_candidate_momentum_layers > 0.0).all() & (global_parent_exact_layers > 0.0).all() & (global_parent_momentum_layers > 0.0).all()).reshape(1)
    selected = torch.where(accepted, candidate_coefficients64, torch.ones_like(candidate_coefficients64)).to(scores.dtype)
    candidate = candidate_coefficients64.to(scores.dtype)
    return _fixed_probe_transaction__FixedProbeTransactionResult(local_coefficients=selected, local_candidate_coefficients=candidate, accepted=accepted, multiplier=torch.where(hard_case, torch.zeros_like(multiplier).reshape(1), multiplier.reshape(1)), rank=retained.sum().reshape(1), eigenvalue_max=spectral_scale.reshape(1), hard_case=hard_case, parent_score=parent_score.reshape(1), candidate_score=candidate_score.reshape(1), budget_residual=budget_residual.reshape(1), local_coordinate_count=local_p, global_coordinate_count=int(total_coordinates), global_probe_count=probes, collective_rounds=0 if local_complete_replica else 2, summary_elements=probes * probes + 3 * probes + 8 * total_layers + 8, owner_count=0, dense_LG_by_LG_metric_elements=0, selected_update_elements_published=0, method_state_depends_on_total_tokens=False)

def _fixed_probe_transaction__replicated_fixed_probe_transaction(local_scores: torch.Tensor, local_decay_action: torch.Tensor, exact_by_role: torch.Tensor, momentum_by_role: torch.Tensor, weights: torch.Tensor, layer_ids: torch.Tensor, *, global_probe_count: int, total_layers: int, eta: float, rounds: int=64, group=None) -> _fixed_probe_transaction__ReplicatedFixedProbeTransactionResult:
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
        raise RuntimeError('replicated fixed-probe scores must be [local_K, p]')
    local_rows, coordinates = local_scores.shape
    if local_decay_action.shape != (local_rows,):
        raise RuntimeError('replicated fixed-probe decay action changed')
    if exact_by_role.shape != (2, coordinates):
        raise RuntimeError('replicated fixed-probe exact inventory changed')
    if momentum_by_role.shape != exact_by_role.shape:
        raise RuntimeError('replicated fixed-probe momentum inventory changed')
    if weights.shape != (coordinates,) or layer_ids.shape != (coordinates,):
        raise RuntimeError('replicated fixed-probe coordinate inventory changed')
    if coordinates < 1:
        raise RuntimeError('fixed-probe coordinate count must be positive')
    if layer_ids.dtype != torch.int64:
        raise RuntimeError('replicated fixed-probe layer IDs must be int64')
    floating = (local_scores, local_decay_action, exact_by_role, momentum_by_role, weights)
    if any((not value.is_floating_point() for value in floating)) or any((value.device != local_scores.device for value in floating[1:])) or any((value.dtype != local_scores.dtype for value in floating[1:])) or (layer_ids.device != local_scores.device):
        raise RuntimeError('replicated fixed-probe dtype or device changed')
    local_packet = torch.cat((local_scores, local_decay_action.unsqueeze(1)), dim=1)
    global_packet, gather_rounds = _fixed_probe_transaction___gather_variable_probe_rows(local_packet, expected_global_rows=int(global_probe_count), group=group)
    global_scores = global_packet[:, :coordinates]
    global_decay_action = global_packet[:, coordinates]
    decay_cross = global_scores.T @ global_decay_action / float(global_probe_count)
    total_row_metric = global_scores @ global_scores.T
    total_metric_square = total_row_metric.square().sum()
    within_layer_square = torch.zeros_like(total_metric_square)
    for layer in range(int(total_layers)):
        layer_scores = global_scores[:, layer_ids.eq(layer)]
        layer_row_metric = layer_scores @ layer_scores.T
        within_layer_square = within_layer_square + layer_row_metric.square().sum()
    cross_layer_coupling_ratio = torch.sqrt((total_metric_square - within_layer_square).clamp_min(0.0) / total_metric_square.clamp_min(torch.finfo(total_metric_square.dtype).tiny))
    rank, world = _fixed_probe_transaction___distributed_shape(group)
    coordinate_ids = torch.arange(coordinates, device=local_scores.device, dtype=torch.int64)
    local_coordinate_ids = coordinate_ids[coordinate_ids.remainder(world).eq(rank)]
    sharded = _fixed_probe_transaction__distributed_fixed_probe_transaction(global_scores[:, local_coordinate_ids], exact_by_role[:, local_coordinate_ids], momentum_by_role[:, local_coordinate_ids], decay_cross[local_coordinate_ids], weights[local_coordinate_ids], layer_ids[local_coordinate_ids], local_coordinate_ids, total_coordinates=coordinates, total_layers=int(total_layers), eta=float(eta), rounds=int(rounds), group=group)
    coefficient_packet = torch.zeros(2 * coordinates, device=local_scores.device, dtype=local_scores.dtype)
    coefficient_packet[local_coordinate_ids] = sharded.local_coefficients
    coefficient_packet[coordinates + local_coordinate_ids] = sharded.local_candidate_coefficients
    if world > 1:
        dist.all_reduce(coefficient_packet, op=dist.ReduceOp.SUM, group=group)
    coefficients = coefficient_packet[:coordinates]
    candidate_coefficients = coefficient_packet[coordinates:]
    return _fixed_probe_transaction__ReplicatedFixedProbeTransactionResult(coefficients=coefficients, candidate_coefficients=candidate_coefficients, sharded_result=sharded, local_probe_count=local_rows, global_probe_count=int(global_probe_count), cross_layer_coupling_ratio=cross_layer_coupling_ratio, collective_rounds=gather_rounds + sharded.collective_rounds + int(world > 1), score_scalars_exchanged_per_rank=int(global_probe_count) * (coordinates + 1), coefficient_scalars_exchanged_per_rank=2 * coordinates if world > 1 else 0, selected_update_elements_published=0, method_state_depends_on_total_tokens=False)
_group_numerics___NS_COEFFICIENTS = (3.4445, -4.775, 2.0315)
_group_numerics___NS_EPS = 1e-07

def _group_numerics___batched_zero_power(momentum: torch.Tensor, steps: int) -> torch.Tensor:
    """Muon quintic Newton--Schulz polar map for a batch of wide matrices."""
    if momentum.ndim != 3:
        raise RuntimeError(f'expected [batch, rows, cols], got {tuple(momentum.shape)}')
    transposed = momentum.shape[-2] > momentum.shape[-1]
    work = momentum.transpose(-2, -1) if transposed else momentum
    work = work.to(dtype=torch.bfloat16)
    norm = torch.linalg.vector_norm(work.float(), dim=(-2, -1), keepdim=True).clamp_min(_group_numerics___NS_EPS)
    work = work / norm.to(dtype=work.dtype)
    a, b, c = _group_numerics___NS_COEFFICIENTS
    for _ in range(int(steps)):
        gram = torch.bmm(work, work.transpose(1, 2))
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        work = torch.baddbmm(work, polynomial, work, beta=a)
    return work.transpose(-2, -1) if transposed else work

def _response_alignment_row___evaluate_response(unit: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a grouped P5/Q4 rational function and its derivative."""
    if unit.ndim != 3:
        raise RuntimeError('response-alignment unit inventory changed')
    groups = unit.shape[1]
    if numerator.shape != (groups, 6) or denominator.shape != (groups, 4):
        raise RuntimeError('response-alignment coefficient inventory changed')
    square = unit.square()
    cube = square * unit
    fourth = square.square()
    fifth = fourth * unit
    absolute = unit.abs()
    num = numerator.float()[None, :, None, :]
    den = denominator.float().abs()[None, :, None, :]
    polynomial = num[..., 0] + num[..., 1] * unit + num[..., 2] * square + num[..., 3] * cube + num[..., 4] * fourth + num[..., 5] * fifth
    polynomial_derivative = num[..., 1] + 2.0 * num[..., 2] * unit + 3.0 * num[..., 3] * square + 4.0 * num[..., 4] * cube + 5.0 * num[..., 5] * fourth
    divisor = 1.0 + den[..., 0] * absolute + den[..., 1] * square + den[..., 2] * absolute * square + den[..., 3] * fourth
    divisor_derivative = den[..., 0] * torch.sign(unit) + 2.0 * den[..., 1] * unit + 3.0 * den[..., 2] * unit * absolute + 4.0 * den[..., 3] * cube
    function = polynomial / divisor
    derivative = (polynomial_derivative * divisor - polynomial * divisor_derivative) / divisor.square()
    return (function, derivative)

def _response_alignment_row___jacobian_kernel_inner(unit: torch.Tensor, function_a: torch.Tensor, derivative_a: torch.Tensor, function_b: torch.Tensor, derivative_b: torch.Tensor) -> torch.Tensor:
    """Exact O(width) inner product of two normalized-GRAIN Jacobian kernels."""
    width = float(unit.shape[-1])
    radial_a = (function_a - unit * derivative_a) / width
    radial_b = (function_b - unit * derivative_b) / width
    product_a = derivative_a * unit
    product_b = derivative_b * unit
    square_a = derivative_a.square()
    square_b = derivative_b.square()
    unit_energy = unit.square().sum(dim=-1)
    diagonal = (square_a * square_b).sum(dim=-1)
    diagonal = diagonal + 2.0 * (square_a * product_b * radial_b).sum(dim=-1)
    diagonal = diagonal + unit_energy * (square_a * radial_b.square()).sum(dim=-1)
    diagonal = diagonal + 2.0 * (square_b * product_a * radial_a).sum(dim=-1)
    diagonal = diagonal + unit_energy * (square_b * radial_a.square()).sum(dim=-1)
    basis_a = torch.stack((product_a, radial_a), dim=-1)
    basis_b = torch.stack((product_b, radial_b), dim=-1)
    gram = basis_a.transpose(-2, -1) @ basis_b
    coupling = torch.zeros_like(gram)
    coupling[..., 0, 1] = 1.0
    coupling[..., 1, 0] = 1.0
    coupling[..., 1, 1] = unit_energy
    rank_two = (coupling @ gram @ coupling * gram).sum(dim=(-2, -1))
    return diagonal + rank_two

@dataclass(frozen=True)
class _response_fisher_diagonal__ResponseFisherDiagonalSums:
    """Unnormalised fixed-probe diagonal curvature summaries."""
    incoming: torch.Tensor
    outgoing: torch.Tensor
    probe_count: int

def _response_fisher_diagonal__response_fisher_diagonal_sums(inputs: torch.Tensor, features: torch.Tensor, cotangents: torch.Tensor, response_adjoint: torch.Tensor) -> _response_fisher_diagonal__ResponseFisherDiagonalSums:
    """Form groupwise empirical-Fisher diagonals without dense covariance.

    ``response_adjoint`` is the exact GRAIN pullback through the
    normalised rational function.  It weights input-coordinate energy for the
    incoming matrix.  Output-cotangent energy weights rational-feature energy
    for the outgoing matrix.
    """
    if inputs.ndim != 2 or cotangents.shape != inputs.shape:
        raise RuntimeError('response-Fisher input/cotangent inventory changed')
    if response_adjoint.ndim != 3:
        raise RuntimeError('response-Fisher adjoint rank changed')
    probes, groups, width = response_adjoint.shape
    if inputs.shape[0] != probes or features.shape != (probes, groups * width):
        raise RuntimeError('response-Fisher probe inventory changed')
    if probes == 0:
        return _response_fisher_diagonal__ResponseFisherDiagonalSums(incoming=torch.zeros(groups, inputs.shape[1], dtype=inputs.dtype, device=inputs.device), outgoing=torch.zeros(groups, width, dtype=features.dtype, device=features.device), probe_count=0)
    incoming_loss_power = response_adjoint.square().mean(dim=-1)
    incoming = incoming_loss_power.transpose(0, 1) @ inputs.square()
    output_loss_power = cotangents.square().mean(dim=-1)
    outgoing = (features.view(probes, groups, width).square() * output_loss_power[:, None, None]).sum(dim=0)
    torch._assert_async(torch.isfinite(incoming).all())
    torch._assert_async(torch.isfinite(outgoing).all())
    return _response_fisher_diagonal__ResponseFisherDiagonalSums(incoming=incoming, outgoing=outgoing, probe_count=int(probes))

def _response_fisher_diagonal__lagged_exponential_diagonal(current: torch.Tensor, previous: torch.Tensor | None, *, decay: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lagged factor used now and the updated persistent factor.

    On the first step the current fixed-probe statistic initializes both
    values.  Afterwards the direction uses only the preceding estimate; the
    current batch enters the state for the next step.  This prevents a
    same-batch curvature estimate from selecting its own update direction.
    """
    if current.ndim != 2:
        raise RuntimeError('response-Fisher current diagonal changed')
    torch._assert_async(torch.isfinite(current).all())
    if not 0.0 < float(decay) < 1.0:
        raise ValueError('response-Fisher decay must lie strictly inside (0,1)')
    if previous is None:
        initialized = current.detach().clone()
        return (initialized, initialized)
    if previous.shape != current.shape or previous.device != current.device:
        raise RuntimeError('response-Fisher persistent diagonal changed')
    lagged = previous
    updated = previous.detach().clone().mul_(float(decay)).add_(current, alpha=1.0 - float(decay))
    return (lagged, updated)

def _response_fisher_diagonal__inverse_root_diagonal_scale(diagonal: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Parameter-free, mean-one-reference inverse-root scaling.

    The scale is invariant to a common rescaling of the loss cotangent.  The
    only numerical stabilizer is the already locked optimizer epsilon.
    """
    if diagonal.ndim != 2:
        raise RuntimeError('response-Fisher scale diagonal changed')
    torch._assert_async(torch.isfinite(diagonal).all())
    if float(eps) != 1e-08:
        raise ValueError('response-Fisher uses the locked optimizer epsilon')
    reference = diagonal.mean()
    safe_reference = reference.clamp_min(torch.finfo(diagonal.dtype).tiny)
    normalized = diagonal / safe_reference
    scale = torch.rsqrt(normalized + float(eps))
    return torch.where(reference > 0.0, scale, torch.ones_like(scale))

def _response_fisher_diagonal__method_state_elements(*, layers: int, groups: int, width: int, model_width: int) -> int:
    """Closed-form persistent method state, excluding locked Muon momentum."""
    values = (int(layers), int(groups), int(width), int(model_width))
    if any((value <= 0 for value in values)):
        raise ValueError('response-Fisher dimensions must be positive')
    return int(layers) * int(groups) * (int(model_width) + int(width)) + 1
_batched_sensor__import__jacobian_kernel_inner = _response_alignment_row___jacobian_kernel_inner

def _batched_sensor__stacked_version_a_factors(preactivation: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor, *, groups: int, width: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the exact P5/Q4 factors for all layers in one tensor program."""
    if preactivation.ndim != 3 or preactivation.shape[-1] != int(groups) * int(width):
        raise RuntimeError('batched response preactivation inventory changed')
    layers = preactivation.shape[0]
    if numerator.shape != (layers, int(groups), 6) or denominator.shape != (layers, int(groups), 4):
        raise RuntimeError('batched response coefficient inventory changed')
    value = preactivation.float().view(layers, -1, int(groups), int(width))
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    unit = value / rms
    unit2 = unit.square()
    unit3 = unit2 * unit
    unit4 = unit2.square()
    unit5 = unit4 * unit
    absolute = unit.abs()
    num = numerator.detach().float()[:, None, :, None, :]
    den = denominator.detach().float().abs()[:, None, :, None, :]
    polynomial = num[..., 0] + num[..., 1] * unit + num[..., 2] * unit2 + num[..., 3] * unit3 + num[..., 4] * unit4 + num[..., 5] * unit5
    polynomial_derivative = num[..., 1] + 2.0 * num[..., 2] * unit + 3.0 * num[..., 3] * unit2 + 4.0 * num[..., 4] * unit3 + 5.0 * num[..., 5] * unit4
    quotient = 1.0 + den[..., 0] * absolute + den[..., 1] * unit2 + den[..., 2] * absolute * unit2 + den[..., 3] * unit4
    quotient_derivative = den[..., 0] * torch.sign(unit) + 2.0 * den[..., 1] * unit + 3.0 * den[..., 2] * unit * absolute + 4.0 * den[..., 3] * unit3
    function = polynomial / quotient
    derivative = (polynomial_derivative * quotient - polynomial * quotient_derivative) / quotient.square()
    radial = function - unit * derivative
    return (unit, derivative, radial)

def _batched_sensor__stacked_evaluate_response(unit: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate arbitrary live/frozen coefficients on stacked normalized rows."""
    if unit.ndim != 4:
        raise RuntimeError('batched response unit inventory changed')
    layers, _positions, groups, _width = unit.shape
    if numerator.shape != (layers, groups, 6) or denominator.shape != (layers, groups, 4):
        raise RuntimeError('batched response evaluation inventory changed')
    square = unit.square()
    cube = square * unit
    fourth = square.square()
    fifth = fourth * unit
    absolute = unit.abs()
    num = numerator.float()[:, None, :, None, :]
    den = denominator.float().abs()[:, None, :, None, :]
    polynomial = num[..., 0] + num[..., 1] * unit + num[..., 2] * square + num[..., 3] * cube + num[..., 4] * fourth + num[..., 5] * fifth
    polynomial_derivative = num[..., 1] + 2.0 * num[..., 2] * unit + 3.0 * num[..., 3] * square + 4.0 * num[..., 4] * cube + 5.0 * num[..., 5] * fourth
    divisor = 1.0 + den[..., 0] * absolute + den[..., 1] * square + den[..., 2] * absolute * square + den[..., 3] * fourth
    divisor_derivative = den[..., 0] * torch.sign(unit) + 2.0 * den[..., 1] * unit + 3.0 * den[..., 2] * unit * absolute + 4.0 * den[..., 3] * cube
    function = polynomial / divisor
    derivative = (polynomial_derivative * divisor - polynomial * divisor_derivative) / divisor.square()
    return (function, derivative)

def _batched_sensor__stacked_response_adjoint(cotangents: torch.Tensor, outgoing_weight: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, groups: int, width: int) -> torch.Tensor:
    """Pull every layer's residual cotangent through its outgoing GRAIN map."""
    unit, derivative, radial = factors
    expected = (cotangents.shape[0], cotangents.shape[1], int(groups), int(width))
    if cotangents.ndim != 3 or any((value.shape != expected for value in factors)):
        raise RuntimeError('batched response-adjoint factor inventory changed')
    if outgoing_weight.shape != (cotangents.shape[0], cotangents.shape[2], int(groups) * int(width)):
        raise RuntimeError('batched response-adjoint matrix inventory changed')
    pulled = torch.bmm(cotangents.float(), outgoing_weight.float()).view(expected)
    result = derivative * pulled + unit * (radial * pulled).mean(dim=-1, keepdim=True)
    torch._assert_async(torch.isfinite(result).all())
    return result

def _batched_sensor__stacked_intrinsic_and_response_statistics(preactivation: torch.Tensor, cotangents: torch.Tensor, response_adjoint: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], frozen_numerator: torch.Tensor, frozen_denominator: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact 4LG intrinsic and 6LG live/frozen response sums."""
    unit, derivative, radial = factors
    if not (unit.shape == derivative.shape == radial.shape == response_adjoint.shape and unit.ndim == 4 and (preactivation.shape[:2] == unit.shape[:2]) and (preactivation.shape[-1] == unit.shape[-2] * unit.shape[-1]) and (cotangents.shape[:2] == unit.shape[:2])):
        raise RuntimeError('batched response-statistic inventory changed')
    width = float(unit.shape[-1])
    function = radial + unit * derivative
    radial_jacobian = radial / width
    trace = derivative.square().sum(dim=-1)
    trace = trace + 2.0 * (derivative * unit * radial_jacobian).sum(dim=-1)
    trace = trace + unit.square().sum(dim=-1) * radial_jacobian.square().sum(dim=-1)
    trace_square = _batched_sensor__import__jacobian_kernel_inner(unit, function, derivative, function, derivative)
    tiny = torch.finfo(trace.dtype).tiny
    incoming = (trace.square() / (width * trace_square.clamp_min(tiny))).clamp(0.0, 1.0)
    energy = function.square().sum(dim=-1)
    fourth = function.pow(4).sum(dim=-1)
    outgoing = (energy.square() / (width * fourth.clamp_min(tiny))).clamp(0.0, 1.0)
    incoming_weight = response_adjoint.square().mean(dim=-1)
    outgoing_weight = cotangents.float().square().mean(dim=-1)[..., None]
    participation = torch.stack(((incoming * incoming_weight).sum(dim=1), incoming_weight.sum(dim=1), (outgoing * outgoing_weight).sum(dim=1), outgoing_weight.expand_as(outgoing).sum(dim=1)), dim=-1)
    frozen_function, frozen_derivative = _batched_sensor__stacked_evaluate_response(unit, frozen_numerator, frozen_denominator)
    incoming_cross = (_batched_sensor__import__jacobian_kernel_inner(unit, function, derivative, frozen_function, frozen_derivative) * incoming_weight).sum(dim=1)
    incoming_live = (_batched_sensor__import__jacobian_kernel_inner(unit, function, derivative, function, derivative) * incoming_weight).sum(dim=1)
    incoming_frozen = (_batched_sensor__import__jacobian_kernel_inner(unit, frozen_function, frozen_derivative, frozen_function, frozen_derivative) * incoming_weight).sum(dim=1)
    value = preactivation.float().view_as(unit)
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    live_h = rms * function
    frozen_h = rms * frozen_function
    outgoing_cross = ((live_h * frozen_h).sum(dim=-1).square() * outgoing_weight).sum(dim=1)
    outgoing_live = (live_h.square().sum(dim=-1).square() * outgoing_weight).sum(dim=1)
    outgoing_frozen = (frozen_h.square().sum(dim=-1).square() * outgoing_weight).sum(dim=1)
    response = torch.stack((torch.stack((incoming_cross, incoming_live, incoming_frozen), dim=-1), torch.stack((outgoing_cross, outgoing_live, outgoing_frozen), dim=-1)), dim=-2)
    torch._assert_async(torch.isfinite(participation).all())
    torch._assert_async(torch.isfinite(response).all())
    return (participation, response)

@dataclass(frozen=True)
class _fd_tail_biatlas_metric__FDTailBiatlasRows:
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
_probe_loss_image__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_probe_loss_image__import__zeropower_via_newton_schulz = _basis_trust___zeropower_via_newton_schulz
_probe_loss_image__import_evenly_spaced_indices = _fixed_global_probe_layout__evenly_spaced_indices
_probe_loss_image__import_fixed_global_probe_layout = _fixed_global_probe_layout__fixed_global_probe_layout
_probe_loss_image__import_replicated_fixed_probe_transaction = _fixed_probe_transaction__replicated_fixed_probe_transaction
_probe_loss_image__FAMILY_ID = 'ten_probe_loss_image_muon_v1'
_probe_loss_image__GLOBAL_PROBE_COUNT = 10
_probe_loss_image__EXPECTED_MICROBATCHES = 4

def _probe_loss_image___version_a_factors(preactivation: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor, *, groups: int, width: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact normalized P5/Q4 derivative factors for one layer's probes."""
    if preactivation.ndim != 2 or preactivation.shape[1] != groups * width:
        raise RuntimeError('ten-probe preactivation inventory changed')
    if numerator.shape != (groups, 6) or denominator.shape != (groups, 4):
        raise RuntimeError('ten-probe coefficient inventory changed')
    value = preactivation.float().view(-1, groups, width)
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    unit = value / rms
    unit2 = unit.square()
    unit3 = unit2 * unit
    unit4 = unit2.square()
    unit5 = unit4 * unit
    absolute = unit.abs()
    num = numerator.detach().float()[None, :, None, :]
    den = denominator.detach().float().abs()[None, :, None, :]
    polynomial = num[..., 0] + num[..., 1] * unit + num[..., 2] * unit2 + num[..., 3] * unit3 + num[..., 4] * unit4 + num[..., 5] * unit5
    polynomial_derivative = num[..., 1] + 2.0 * num[..., 2] * unit + 3.0 * num[..., 3] * unit2 + 4.0 * num[..., 4] * unit3 + 5.0 * num[..., 5] * unit4
    quotient = 1.0 + den[..., 0] * absolute + den[..., 1] * unit2 + den[..., 2] * absolute * unit2 + den[..., 3] * unit4
    quotient_derivative = den[..., 0] * torch.sign(unit) + 2.0 * den[..., 1] * unit + 3.0 * den[..., 2] * unit * absolute + 4.0 * den[..., 3] * unit3
    function = polynomial / quotient
    derivative = (polynomial_derivative * quotient - polynomial * quotient_derivative) / quotient.square()
    radial = function - unit * derivative
    return (unit, derivative, radial)

def _probe_loss_image___one_layer_group_scores(inputs: torch.Tensor, preactivations: torch.Tensor, features: torch.Tensor, cotangents: torch.Tensor, incoming_direction: torch.Tensor, outgoing_direction_transpose: torch.Tensor, outgoing_weight: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, groups: int, width: int, cached_response_adjoint: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract exact tangent images without materializing residual images."""
    probes, residual = inputs.shape
    hidden = int(groups) * int(width)
    if preactivations.shape != (probes, hidden) or features.shape != (probes, hidden) or cotangents.shape != (probes, residual) or (incoming_direction.shape != (hidden, residual)) or (outgoing_direction_transpose.shape != (hidden, residual)) or (outgoing_weight.shape != (residual, hidden)):
        raise RuntimeError('ten-probe direct-score inventory changed')
    unit, derivative, radial = factors
    expected = (probes, int(groups), int(width))
    if any((value.shape != expected for value in (unit, derivative, radial))):
        raise RuntimeError('ten-probe factor inventory changed')
    response_adjoint = cached_response_adjoint
    if response_adjoint is None:
        response_cotangent = (cotangents.float() @ outgoing_weight.float()).view(expected)
        response_adjoint = derivative * response_cotangent + unit * (radial * response_cotangent).mean(dim=-1, keepdim=True)
    elif response_adjoint.shape != expected:
        raise RuntimeError('ten-probe cached adjoint inventory changed')
    perturbation = (inputs.float() @ incoming_direction.float().T).view(expected)
    incoming_score = (perturbation * response_adjoint).sum(dim=-1)
    outgoing_projection = (cotangents.float() @ outgoing_direction_transpose.float().T).view(expected)
    outgoing_score = (features.float().view(expected) * outgoing_projection).sum(dim=-1)
    score = incoming_score + outgoing_score
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError('ten-probe loss score is nonfinite')
    return (score, response_adjoint)

class _probe_loss_image__TenProbeLossImageMuonOptimizer(torch.optim.Optimizer):
    """Full NS5 Muon with a globally fixed ten-row loss-image transaction."""

    def __init__(self, pairs, *, lr: float, weight_decay: float, momentum: float, ns_steps: int, beta2: float, eps: float, loss_probe_group=None):
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError('ten-probe Muon requires GRAIN layers')
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        if self.momentum != 0.95 or self.ns_steps != 5:
            raise ValueError('ten-probe Muon requires matched momentum .95 and NS5')
        if float(beta2) != 0.95 or float(eps) != 1e-08:
            raise ValueError('ten-probe Muon requires matched Adam beta2/eps')
        self.loss_probe_group = loss_probe_group
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank(group=loss_probe_group)
            world = dist.get_world_size(group=loss_probe_group)
        else:
            rank, world = (0, 1)
        self.probe_layout = _probe_loss_image__import_fixed_global_probe_layout(_probe_loss_image__GLOBAL_PROBE_COUNT, rank, world)
        self.capture_rows = (self.probe_layout.local_probe_count + _probe_loss_image__EXPECTED_MICROBATCHES - 1) // _probe_loss_image__EXPECTED_MICROBATCHES
        first = self.pairs[0]
        self.groups = int(first['groups'])
        self.hidden = int(first['hidden_dim'])
        self.external = int(first['in_weight'].shape[1])
        self.grain_eps = float(first['eps'])
        if self.hidden % self.groups:
            raise ValueError('ten-probe hidden width is not group divisible')
        self.width = self.hidden // self.groups
        self._pending_inputs = [None for _ in self.pairs]
        self._functional_records = [[] for _ in self.pairs]
        self._cotangent_records = [[] for _ in self.pairs]
        self._hook_handles = []
        parameters = []
        seen = set()
        for layer, pair in enumerate(self.pairs):
            incoming = pair['in_weight']
            outgoing = pair['out_weight']
            if int(pair['groups']) != self.groups or int(pair['hidden_dim']) != self.hidden or float(pair['eps']) != self.grain_eps or (incoming.shape != (self.hidden, self.external)) or (outgoing.shape != (self.external, self.hidden)) or (pair['numerator'].shape != (self.groups, 6)) or (pair['denominator'].shape != (self.groups, 4)):
                raise ValueError('ten-probe GRAIN inventory changed')
            for parameter in (incoming, outgoing):
                if id(parameter) in seen:
                    raise ValueError('ten-probe matrix ownership overlaps')
                seen.add(id(parameter))
                parameters.append(parameter)
            self._hook_handles.extend((pair['mlp'].register_forward_pre_hook(self._make_input_hook(layer)), pair['module'].register_forward_hook(self._make_feature_hook(layer)), pair['mlp'].register_forward_hook(self._make_cotangent_hook(layer))))
        defaults = {'lr': float(lr), 'weight_decay': float(weight_decay), 'lr_scale': 1.0, 'ten_probe_family_id': _probe_loss_image__FAMILY_ID, 'muon_momentum': self.momentum, 'muon_ns_steps': self.ns_steps, 'muon_adjust_lr_fn': 'match_rms_adamw'}
        super().__init__([{'params': parameters}], defaults)
        self._clip_factor = None
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def _sample_indices(self, rows: int, device: torch.device) -> torch.Tensor:
        return _probe_loss_image__import_evenly_spaced_indices(int(rows), self.capture_rows, device=device)

    def _make_input_hook(self, layer: int):

        @torch.no_grad()
        def capture(module, inputs):
            if not module.training:
                return
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError('ten-probe MLP input hook changed')
            if self._pending_inputs[layer] is not None:
                raise RuntimeError('ten-probe input remained pending')
            flat = inputs[0].detach().reshape(-1, self.external)
            indices = self._sample_indices(flat.shape[0], flat.device)
            self._pending_inputs[layer] = (int(flat.shape[0]), indices, flat.index_select(0, indices).clone())
        return capture

    def _make_feature_hook(self, layer: int):

        @torch.no_grad()
        def capture(module, inputs, output):
            if not module.training:
                return
            pending = self._pending_inputs[layer]
            if pending is None:
                raise RuntimeError('ten-probe feature lacks aligned input')
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or (not torch.is_tensor(output)):
                raise RuntimeError('ten-probe activation hook changed')
            rows, indices, sampled_input = pending
            preactivation = inputs[0].detach().reshape(-1, self.hidden)
            feature = output.detach().reshape(-1, self.hidden)
            if preactivation.shape[0] != rows or feature.shape[0] != rows:
                raise RuntimeError('ten-probe x/z/h rows differ')
            self._functional_records[layer].append((sampled_input, preactivation.index_select(0, indices).clone(), feature.index_select(0, indices).clone()))
            self._pending_inputs[layer] = None
        return capture

    def _make_cotangent_hook(self, layer: int):

        def capture(module, _inputs, output):
            if not module.training:
                return
            if not torch.is_tensor(output) or not output.requires_grad:
                raise RuntimeError('ten-probe MLP output hook changed')
            flat = output.reshape(-1, self.external)
            rows = int(flat.shape[0])
            indices = self._sample_indices(rows, flat.device)

            def capture_gradient(cotangent):
                with torch.no_grad():
                    value = cotangent.detach().reshape(-1, self.external)
                    if value.shape[0] != rows:
                        raise RuntimeError('ten-probe output/cotangent rows differ')
                    self._cotangent_records[layer].append((value.index_select(0, indices).clone(), rows))
                return cotangent
            output.register_hook(capture_gradient)
        return capture

    def _consume_probes(self):
        if self._clip_factor is None:
            raise RuntimeError('ten-probe Muon lacks realized clipping')
        packets = []
        for layer in range(len(self.pairs)):
            if self._pending_inputs[layer] is not None:
                raise RuntimeError('ten-probe input remained unmatched')
            records = self._functional_records[layer]
            cotangents = self._cotangent_records[layer]
            self._functional_records[layer] = []
            self._cotangent_records[layer] = []
            if len(records) != _probe_loss_image__EXPECTED_MICROBATCHES or len(cotangents) != _probe_loss_image__EXPECTED_MICROBATCHES:
                raise RuntimeError('ten-probe microbatch inventory changed')
            inputs = torch.cat([record[0] for record in records], dim=0)
            preactivations = torch.cat([record[1] for record in records], dim=0)
            features = torch.cat([record[2] for record in records], dim=0)
            scaled_cotangents = torch.cat([value.float() * float(rows * _probe_loss_image__EXPECTED_MICROBATCHES) for value, rows in cotangents], dim=0) * float(self._clip_factor)
            captured = self.capture_rows * _probe_loss_image__EXPECTED_MICROBATCHES
            if inputs.shape != (captured, self.external) or preactivations.shape != (captured, self.hidden) or features.shape != (captured, self.hidden) or (scaled_cotangents.shape != (captured, self.external)):
                raise RuntimeError('ten-probe captured packet changed')
            selected = _probe_loss_image__import_evenly_spaced_indices(captured, self.probe_layout.local_probe_count, device=inputs.device)
            packets.append((inputs.index_select(0, selected).float(), preactivations.index_select(0, selected).float(), features.index_select(0, selected).float(), scaled_cotangents.index_select(0, selected).float()))
        return packets

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'matrix_role_lr_scale': 1.0, 'fixed_loss_probe_lr_scale': 1.0, 'loss_image_transaction_lr_scale': 1.0, 'equality_budget_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def record_realized_clipping(self, preclip_norm, max_norm):
        if self._clip_factor is not None:
            raise RuntimeError('ten-probe Muon observed multiple clipping calls')
        value = float(preclip_norm)
        maximum = float(max_norm)
        if not math.isfinite(value) or value < 0.0 or maximum != 1.0:
            raise RuntimeError('ten-probe Muon received invalid clipping')
        self._clip_factor = min(1.0, maximum / (value + 1e-06))

    def set_telemetry_capture(self, enabled=True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def _nesterov(self, parameter: torch.Tensor) -> torch.Tensor:
        if parameter.grad is None:
            raise RuntimeError('ten-probe Muon matrix gradient is missing')
        state = self.state[parameter]
        buffer = state.get('momentum_buffer')
        if buffer is None:
            buffer = torch.zeros_like(parameter.grad, memory_format=torch.preserve_format)
            state['momentum_buffer'] = buffer
        buffer.lerp_(parameter.grad, 1.0 - self.momentum)
        return parameter.grad.lerp(buffer, self.momentum)

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('ten-probe Muon did not receive realized clipping')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('ten-probe Muon refuses nonunit LR scale')
        packets = self._consume_probes()
        incoming_directions = []
        outgoing_directions = []
        incoming_adjustments = []
        outgoing_adjustments = []
        exact_incoming = []
        exact_outgoing = []
        momentum_incoming = []
        momentum_outgoing = []
        budget_weights = []
        for pair in self.pairs:
            incoming = pair['in_weight']
            outgoing = pair['out_weight']
            incoming_momentum = self._nesterov(incoming)
            outgoing_momentum = self._nesterov(outgoing)
            incoming_direction = _probe_loss_image__import__zeropower_via_newton_schulz(incoming_momentum, self.ns_steps)
            outgoing_direction = _probe_loss_image__import__zeropower_via_newton_schulz(outgoing_momentum, self.ns_steps)
            incoming_adjustment = _probe_loss_image__import__match_rms_adamw_adjustment(incoming.shape)
            outgoing_adjustment = _probe_loss_image__import__match_rms_adamw_adjustment(outgoing.shape)
            incoming_view = incoming_direction.view(self.groups, self.width, self.external)
            outgoing_view = outgoing_direction.view(self.external, self.groups, self.width).permute(1, 2, 0)
            incoming_gradient = incoming.grad.detach().view_as(incoming_view)
            outgoing_gradient = outgoing.grad.detach().view(self.external, self.groups, self.width).permute(1, 2, 0)
            incoming_momentum_view = incoming_momentum.view_as(incoming_view)
            outgoing_momentum_view = outgoing_momentum.view(self.external, self.groups, self.width).permute(1, 2, 0)
            incoming_float = incoming_view.float() * incoming_adjustment
            outgoing_float = outgoing_view.float() * outgoing_adjustment
            exact_incoming.append((incoming_gradient.float() * incoming_float).sum(dim=(-2, -1)))
            exact_outgoing.append((outgoing_gradient.float() * outgoing_float).sum(dim=(-2, -1)))
            momentum_incoming.append((incoming_momentum_view.float() * incoming_float).sum(dim=(-2, -1)))
            momentum_outgoing.append((outgoing_momentum_view.float() * outgoing_float).sum(dim=(-2, -1)))
            budget_weights.append(incoming_float.square().sum(dim=(-2, -1)) + outgoing_float.square().sum(dim=(-2, -1)))
            incoming_directions.append(incoming_direction)
            outgoing_directions.append(outgoing_direction)
            incoming_adjustments.append(incoming_adjustment)
            outgoing_adjustments.append(outgoing_adjustment)
        local_scores = []
        local_decay = None
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            inputs, preactivations, features, cotangents = packet
            factors = _probe_loss_image___version_a_factors(preactivations, pair['numerator'], pair['denominator'], groups=self.groups, width=self.width, eps=self.grain_eps)
            incoming_direction = incoming_directions[layer].float() * incoming_adjustments[layer]
            outgoing_direction_transpose = outgoing_directions[layer].view(self.external, self.groups, self.width).permute(1, 2, 0).reshape(self.hidden, self.external).float() * outgoing_adjustments[layer]
            score, response_adjoint = _probe_loss_image___one_layer_group_scores(inputs, preactivations, features, cotangents, incoming_direction, outgoing_direction_transpose, pair['out_weight'], factors, groups=self.groups, width=self.width)
            decay_score, _ = _probe_loss_image___one_layer_group_scores(inputs, preactivations, features, cotangents, pair['in_weight'].float() * weight_decay, pair['out_weight'].T.float() * weight_decay, pair['out_weight'], factors, groups=self.groups, width=self.width, cached_response_adjoint=response_adjoint)
            local_scores.append(score)
            layer_decay = decay_score.sum(dim=-1)
            local_decay = layer_decay if local_decay is None else local_decay + layer_decay
        score_lattice = torch.stack(local_scores, dim=1).reshape(self.probe_layout.local_probe_count, len(self.pairs) * self.groups)
        if local_decay is None:
            raise RuntimeError('ten-probe decay action was not formed')
        exact_by_role = torch.stack((torch.stack(exact_incoming).reshape(-1), torch.stack(exact_outgoing).reshape(-1)))
        momentum_by_role = torch.stack((torch.stack(momentum_incoming).reshape(-1), torch.stack(momentum_outgoing).reshape(-1)))
        weights = torch.stack(budget_weights).reshape(-1)
        layer_ids = torch.arange(len(self.pairs), device=weights.device, dtype=torch.int64).repeat_interleave(self.groups)
        selection = _probe_loss_image__import_replicated_fixed_probe_transaction(score_lattice, local_decay, exact_by_role, momentum_by_role, weights, layer_ids, global_probe_count=_probe_loss_image__GLOBAL_PROBE_COUNT, total_layers=len(self.pairs), eta=lr, rounds=64, group=self.loss_probe_group)
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)
        for layer, pair in enumerate(self.pairs):
            incoming = pair['in_weight']
            outgoing = pair['out_weight']
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming_direction = incoming_directions[layer]
            outgoing_direction = outgoing_directions[layer]
            incoming_direction.view(self.groups, self.width, self.external).mul_(coefficients[layer, :, None, None].to(incoming_direction.dtype))
            outgoing_direction.view(self.external, self.groups, self.width).permute(1, 2, 0).mul_(coefficients[layer, :, None, None].to(outgoing_direction.dtype))
            incoming.add_(incoming_direction.to(incoming.dtype), alpha=-lr * incoming_adjustments[layer])
            outgoing.add_(outgoing_direction.to(outgoing.dtype), alpha=-lr * outgoing_adjustments[layer])
        if self._capture_telemetry_next_step:
            flat = coefficients.reshape(-1)
            transaction = selection.sharded_result
            self._last_telemetry = {'ten_probe_family_id': _probe_loss_image__FAMILY_ID, 'ten_probe_owner_count': transaction.owner_count, 'ten_probe_global_rows': _probe_loss_image__GLOBAL_PROBE_COUNT, 'ten_probe_local_rows': self.probe_layout.local_probe_count, 'ten_probe_coordinate_count': len(self.pairs) * self.groups, 'ten_probe_state_depends_on_total_tokens': 0, 'ten_probe_dense_lg_metric_elements': transaction.dense_LG_by_LG_metric_elements, 'ten_probe_selected_update_elements_published': transaction.selected_update_elements_published, 'ten_probe_transaction_accepted': int(transaction.accepted.item()), 'ten_probe_rank': int(transaction.rank.item()), 'ten_probe_budget_residual': float(transaction.budget_residual.item()), 'ten_probe_parent_score': float(transaction.parent_score.item()), 'ten_probe_candidate_score': float(transaction.candidate_score.item()), 'ten_probe_cross_layer_coupling_ratio': float(selection.cross_layer_coupling_ratio.item()), 'ten_probe_coefficient_min': float(flat.amin().item()), 'ten_probe_coefficient_median': float(flat.median().item()), 'ten_probe_coefficient_max': float(flat.amax().item()), 'ten_probe_realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss
_every_step_rfd_gradient_ledger__import_CURRENT_ROWS = _diagonal_completed_biatlas_metric__CURRENT_ROWS
_every_step_rfd_gradient_ledger__import_MATCHED_BETA2 = _diagonal_completed_biatlas_metric__MATCHED_BETA2
_every_step_rfd_gradient_ledger__import_MAXIMUM_SELECTION_ROWS = _diagonal_completed_biatlas_metric__MAXIMUM_SELECTION_ROWS
_every_step_rfd_gradient_ledger__import_PERSISTENT_ROWS = _diagonal_completed_biatlas_metric__PERSISTENT_ROWS
_every_step_rfd_gradient_ledger__import_FDTailBiatlasRows = _fd_tail_biatlas_metric__FDTailBiatlasRows

def _every_step_rfd_gradient_ledger__trace_matched_gradient_surrogate(exact_by_role: torch.Tensor, decay_derivative: torch.Tensor, reference_row_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift one exact batch-gradient score into a trace-matched 32-row factor."""
    if exact_by_role.ndim != 2 or exact_by_role.shape[0] != 2 or exact_by_role.shape[1] < 2 or (decay_derivative.numel() != 1) or (reference_row_norm.numel() != 1) or (not exact_by_role.is_floating_point()) or (decay_derivative.dtype != exact_by_role.dtype) or (reference_row_norm.dtype != exact_by_role.dtype) or (decay_derivative.device != exact_by_role.device) or (reference_row_norm.device != exact_by_role.device):
        raise RuntimeError('gradient-ledger surrogate inventory changed')
    valid = torch.isfinite(exact_by_role).all() & torch.isfinite(decay_derivative).all() & torch.isfinite(reference_row_norm).all() & (reference_row_norm >= 0.0).all()
    if not bool(valid):
        raise RuntimeError('gradient-ledger surrogate inventory changed')
    gradient_score = exact_by_role.double().sum(dim=0)
    norm = torch.linalg.vector_norm(gradient_score)
    reference = reference_row_norm.double().reshape(())
    positive = norm > torch.finfo(torch.float64).tiny
    scale64 = torch.where(positive, reference / norm.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(norm))
    score = (gradient_score * scale64).to(exact_by_role.dtype)
    decay = (decay_derivative.double().reshape(()) * scale64).to(exact_by_role.dtype)
    scores = score.unsqueeze(0).expand(_every_step_rfd_gradient_ledger__import_CURRENT_ROWS, -1).clone()
    decay_action = decay.expand(_every_step_rfd_gradient_ledger__import_CURRENT_ROWS).clone()
    torch._assert_async(torch.isfinite(scores).all() & torch.isfinite(decay_action).all() & torch.isfinite(scale64))
    return (scores, decay_action, scale64.to(exact_by_role.dtype))

def _every_step_rfd_gradient_ledger__functional_row_norm(current_scores: torch.Tensor, *, validate_values: bool=True) -> torch.Tensor:
    """RMS norm of one fixed functional-score row."""
    if current_scores.ndim != 2 or current_scores.shape[0] != _every_step_rfd_gradient_ledger__import_CURRENT_ROWS or current_scores.shape[1] < 2 or (not current_scores.is_floating_point()):
        raise RuntimeError('functional calibration rows changed')
    if validate_values and (not bool(torch.isfinite(current_scores).all())):
        raise RuntimeError('functional calibration rows changed')
    value = torch.sqrt(current_scores.double().square().sum() / float(_every_step_rfd_gradient_ledger__import_CURRENT_ROWS))
    torch._assert_async(torch.isfinite(value) & (value >= 0.0))
    return value.to(current_scores.dtype)

def _every_step_rfd_gradient_ledger___validate_current(current_scores: torch.Tensor, current_decay_action: torch.Tensor) -> int:
    if current_scores.ndim != 2 or current_scores.shape[0] != _every_step_rfd_gradient_ledger__import_CURRENT_ROWS or current_scores.shape[1] < 2 or (current_decay_action.shape != (_every_step_rfd_gradient_ledger__import_CURRENT_ROWS,)) or (not current_scores.is_floating_point()) or (current_decay_action.dtype != current_scores.dtype) or (current_decay_action.device != current_scores.device):
        raise RuntimeError('every-step RFD current-score inventory changed')
    if not bool(torch.isfinite(current_scores).all() & torch.isfinite(current_decay_action).all()):
        raise RuntimeError('every-step RFD current-score inventory changed')
    return int(current_scores.shape[1])

def _every_step_rfd_gradient_ledger__every_step_rfd_gradient_rows(current_scores: torch.Tensor, current_decay_action: torch.Tensor, previous_scores: torch.Tensor | None, previous_total_diagonal: torch.Tensor | None, previous_decay_cross: torch.Tensor | None, *, beta2: float, previous_isotropic_tail: torch.Tensor | None=None) -> _every_step_rfd_gradient_ledger__import_FDTailBiatlasRows:
    """Advance a matched-beta Robust-FD factor on every optimizer step."""
    coordinates = _every_step_rfd_gradient_ledger___validate_current(current_scores, current_decay_action)
    if float(beta2) != _every_step_rfd_gradient_ledger__import_MATCHED_BETA2:
        raise ValueError('gradient ledger requires matched beta2=.95')
    missing = (previous_scores is None, previous_total_diagonal is None, previous_decay_cross is None)
    if any(missing) and (not all(missing)):
        raise RuntimeError('gradient-ledger persistent state is partial')
    score64 = current_scores.double()
    decay64 = current_decay_action.double()
    current_cross = score64.T @ decay64 / float(_every_step_rfd_gradient_ledger__import_CURRENT_ROWS)
    zero = score64.new_zeros(())
    if previous_scores is None:
        if previous_isotropic_tail is not None:
            raise RuntimeError('gradient-ledger tail appeared before factor')
        persistent64 = score64.clone()
        represented = persistent64.square().sum(dim=0) / float(_every_step_rfd_gradient_ledger__import_CURRENT_ROWS)
        tail_vector = torch.zeros_like(represented)
        return _every_step_rfd_gradient_ledger__import_FDTailBiatlasRows(selection_scores=current_scores, selection_diagonal_residual=tail_vector.to(current_scores.dtype), persistent_scores=persistent64.to(current_scores.dtype), persistent_total_diagonal=represented.to(current_scores.dtype), persistent_diagonal_residual=tail_vector.to(current_scores.dtype), persistent_decay_cross=current_cross.to(current_scores.dtype), history_used=False, discarded_energy_fraction=zero.to(current_scores.dtype), selection_isotropic_tail=zero.to(current_scores.dtype), persistent_isotropic_tail=zero.to(current_scores.dtype), fd_shrinkage=zero.to(current_scores.dtype))
    assert previous_total_diagonal is not None
    assert previous_decay_cross is not None
    rows = int(previous_scores.shape[0])
    if previous_scores.ndim != 2 or previous_scores.shape[1] != coordinates or rows not in (_every_step_rfd_gradient_ledger__import_CURRENT_ROWS, _every_step_rfd_gradient_ledger__import_PERSISTENT_ROWS) or (previous_total_diagonal.shape != (coordinates,)) or (previous_decay_cross.shape != (coordinates,)) or (previous_scores.dtype != current_scores.dtype) or (previous_total_diagonal.dtype != current_scores.dtype) or (previous_decay_cross.dtype != current_scores.dtype) or (previous_scores.device != current_scores.device) or (previous_total_diagonal.device != current_scores.device) or (previous_decay_cross.device != current_scores.device) or (previous_isotropic_tail is None) or (not torch.is_tensor(previous_isotropic_tail)) or (previous_isotropic_tail.numel() != 1) or (previous_isotropic_tail.device != current_scores.device) or (not previous_isotropic_tail.is_floating_point()):
        raise RuntimeError('gradient-ledger persistent inventory changed')
    valid_previous = torch.isfinite(previous_scores).all() & torch.isfinite(previous_total_diagonal).all() & torch.isfinite(previous_decay_cross).all() & (previous_total_diagonal >= 0.0).all() & torch.isfinite(previous_isotropic_tail).all() & (previous_isotropic_tail >= 0.0).all()
    if not bool(valid_previous):
        raise RuntimeError('gradient-ledger persistent inventory changed')
    beta = float(beta2)
    previous_tail = previous_isotropic_tail.double().reshape(())
    selection = torch.cat((previous_scores * math.sqrt(beta), current_scores * math.sqrt(1.0 - beta)))
    if int(selection.shape[0]) > _every_step_rfd_gradient_ledger__import_MAXIMUM_SELECTION_ROWS:
        raise RuntimeError('gradient-ledger live row bound changed')
    selection_tail = previous_tail * beta
    decay_cross = previous_decay_cross.double() * beta + current_cross * (1.0 - beta)
    gram = selection.double() @ selection.double().T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    physical_rows = int(eigenvalues.numel())
    retained_rows = min(_every_step_rfd_gradient_ledger__import_PERSISTENT_ROWS, physical_rows)
    discarded_rows = physical_rows - retained_rows
    shrinkage = eigenvalues[discarded_rows - 1] if discarded_rows else zero
    retained = eigenvalues[-retained_rows:]
    transform = eigenvectors[:, -retained_rows:].T
    unshrunk = transform @ selection.double()
    multiplier = torch.sqrt((retained - shrinkage).clamp_min(0.0) / retained.clamp_min(torch.finfo(torch.float64).tiny))
    persistent64 = multiplier[:, None] * unshrunk
    persistent_tail = selection_tail + shrinkage / (2.0 * float(_every_step_rfd_gradient_ledger__import_CURRENT_ROWS))
    represented = persistent64.square().sum(dim=0) / float(_every_step_rfd_gradient_ledger__import_CURRENT_ROWS)
    total_diagonal = represented + persistent_tail
    selection_tail_vector = torch.ones_like(represented) * selection_tail
    persistent_tail_vector = torch.ones_like(represented) * persistent_tail
    discarded_energy = eigenvalues[:discarded_rows].sum() if discarded_rows else zero
    discarded_fraction = discarded_energy / eigenvalues.sum().clamp_min(torch.finfo(torch.float64).tiny)
    torch._assert_async(torch.isfinite(selection).all() & torch.isfinite(persistent64).all() & torch.isfinite(decay_cross).all() & torch.isfinite(persistent_tail) & torch.isfinite(shrinkage) & (persistent_tail >= 0.0) & (shrinkage >= 0.0))
    return _every_step_rfd_gradient_ledger__import_FDTailBiatlasRows(selection_scores=selection, selection_diagonal_residual=selection_tail_vector.to(current_scores.dtype), persistent_scores=persistent64.to(current_scores.dtype), persistent_total_diagonal=total_diagonal.to(current_scores.dtype), persistent_diagonal_residual=persistent_tail_vector.to(current_scores.dtype), persistent_decay_cross=decay_cross.to(current_scores.dtype), history_used=True, discarded_energy_fraction=discarded_fraction.to(current_scores.dtype), selection_isotropic_tail=selection_tail.to(current_scores.dtype), persistent_isotropic_tail=persistent_tail.to(current_scores.dtype), fd_shrinkage=shrinkage.to(current_scores.dtype))
_fixed_transaction_base__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_fixed_transaction_base__import__zeropower_via_newton_schulz = _basis_trust___zeropower_via_newton_schulz
_fixed_transaction_base__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_fixed_transaction_base__import_fixed_global_probe_layout = _fixed_global_probe_layout__fixed_global_probe_layout
_fixed_transaction_base__import_EXPECTED_MICROBATCHES = _probe_loss_image__EXPECTED_MICROBATCHES
_fixed_transaction_base__import_TenProbeLossImageMuonOptimizer = _probe_loss_image__TenProbeLossImageMuonOptimizer
_fixed_transaction_base__import__one_layer_group_scores = _probe_loss_image___one_layer_group_scores
_fixed_transaction_base__import__version_a_factors = _probe_loss_image___version_a_factors

class _fixed_transaction_base__Fixed32TransactionMuonBase(_fixed_transaction_base__import_TenProbeLossImageMuonOptimizer):
    """Shared direct-score execution; subclasses supply only scalar math."""
    family_id = 'abstract_fixed32_transaction'
    telemetry_prefix = 'abstract_fixed32_'
    fairness_component = 'abstract_transaction_lr_scale'

    def __init__(self, pairs, **kwargs):
        if self.__class__ is _fixed_transaction_base__Fixed32TransactionMuonBase:
            raise TypeError('fixed32 transaction base is abstract')
        super().__init__(pairs, **kwargs)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(group=self.loss_probe_group)
            world = torch.distributed.get_world_size(group=self.loss_probe_group)
        else:
            rank, world = (0, 1)
        self.probe_layout = _fixed_transaction_base__import_fixed_global_probe_layout(_fixed_transaction_base__import_FIXED_GLOBAL_PROBE_COUNT, rank, world)
        self.capture_rows = (self.probe_layout.local_probe_count + _fixed_transaction_base__import_EXPECTED_MICROBATCHES - 1) // _fixed_transaction_base__import_EXPECTED_MICROBATCHES
        self.param_groups[0]['ten_probe_family_id'] = self.family_id
        self.param_groups[0]['fixed32_transaction_family_id'] = self.family_id

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'matrix_role_lr_scale': 1.0, 'fixed32_loss_measure_lr_scale': 1.0, self.fairness_component: 1.0, 'equality_budget_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def _transaction(self, score_lattice, local_decay, exact_by_role, momentum_by_role, weights, layer_ids, *, eta):
        raise NotImplementedError

    def _extra_transaction_telemetry(self, transaction):
        return {}

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('fixed32 transaction Muon lacks realized clipping')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('fixed32 transaction Muon refuses nonunit LR scale')
        packets = self._consume_probes()
        incoming_directions = []
        outgoing_directions = []
        incoming_adjustments = []
        outgoing_adjustments = []
        exact_incoming = []
        exact_outgoing = []
        momentum_incoming = []
        momentum_outgoing = []
        budget_weights = []
        for pair in self.pairs:
            incoming = pair['in_weight']
            outgoing = pair['out_weight']
            incoming_momentum = self._nesterov(incoming)
            outgoing_momentum = self._nesterov(outgoing)
            incoming_direction = _fixed_transaction_base__import__zeropower_via_newton_schulz(incoming_momentum, self.ns_steps)
            outgoing_direction = _fixed_transaction_base__import__zeropower_via_newton_schulz(outgoing_momentum, self.ns_steps)
            incoming_adjustment = _fixed_transaction_base__import__match_rms_adamw_adjustment(incoming.shape)
            outgoing_adjustment = _fixed_transaction_base__import__match_rms_adamw_adjustment(outgoing.shape)
            incoming_view = incoming_direction.view(self.groups, self.width, self.external)
            outgoing_view = outgoing_direction.view(self.external, self.groups, self.width).permute(1, 2, 0)
            incoming_gradient = incoming.grad.detach().view_as(incoming_view)
            outgoing_gradient = outgoing.grad.detach().view(self.external, self.groups, self.width).permute(1, 2, 0)
            incoming_momentum_view = incoming_momentum.view_as(incoming_view)
            outgoing_momentum_view = outgoing_momentum.view(self.external, self.groups, self.width).permute(1, 2, 0)
            incoming_float = incoming_view.float() * incoming_adjustment
            outgoing_float = outgoing_view.float() * outgoing_adjustment
            exact_incoming.append((incoming_gradient.float() * incoming_float).sum(dim=(-2, -1)))
            exact_outgoing.append((outgoing_gradient.float() * outgoing_float).sum(dim=(-2, -1)))
            momentum_incoming.append((incoming_momentum_view.float() * incoming_float).sum(dim=(-2, -1)))
            momentum_outgoing.append((outgoing_momentum_view.float() * outgoing_float).sum(dim=(-2, -1)))
            budget_weights.append(incoming_float.square().sum(dim=(-2, -1)) + outgoing_float.square().sum(dim=(-2, -1)))
            incoming_directions.append(incoming_direction)
            outgoing_directions.append(outgoing_direction)
            incoming_adjustments.append(incoming_adjustment)
            outgoing_adjustments.append(outgoing_adjustment)
        local_scores = []
        local_decay = None
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            inputs, preactivations, features, cotangents = packet
            factors = _fixed_transaction_base__import__version_a_factors(preactivations, pair['numerator'], pair['denominator'], groups=self.groups, width=self.width, eps=self.grain_eps)
            incoming_direction = incoming_directions[layer].float() * incoming_adjustments[layer]
            outgoing_direction_transpose = outgoing_directions[layer].view(self.external, self.groups, self.width).permute(1, 2, 0).reshape(self.hidden, self.external).float() * outgoing_adjustments[layer]
            score, response_adjoint = _fixed_transaction_base__import__one_layer_group_scores(inputs, preactivations, features, cotangents, incoming_direction, outgoing_direction_transpose, pair['out_weight'], factors, groups=self.groups, width=self.width)
            decay_score, _ = _fixed_transaction_base__import__one_layer_group_scores(inputs, preactivations, features, cotangents, pair['in_weight'].float() * weight_decay, pair['out_weight'].T.float() * weight_decay, pair['out_weight'], factors, groups=self.groups, width=self.width, cached_response_adjoint=response_adjoint)
            local_scores.append(score)
            layer_decay = decay_score.sum(dim=-1)
            local_decay = layer_decay if local_decay is None else local_decay + layer_decay
        if local_decay is None:
            raise RuntimeError('fixed32 transaction decay action was not formed')
        score_lattice = torch.stack(local_scores, dim=1).reshape(self.probe_layout.local_probe_count, len(self.pairs) * self.groups)
        exact_by_role = torch.stack((torch.stack(exact_incoming).reshape(-1), torch.stack(exact_outgoing).reshape(-1)))
        momentum_by_role = torch.stack((torch.stack(momentum_incoming).reshape(-1), torch.stack(momentum_outgoing).reshape(-1)))
        weights = torch.stack(budget_weights).reshape(-1)
        layer_ids = torch.arange(len(self.pairs), device=weights.device, dtype=torch.int64).repeat_interleave(self.groups)
        selection = self._transaction(score_lattice, local_decay, exact_by_role, momentum_by_role, weights, layer_ids, eta=lr)
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)
        for layer, pair in enumerate(self.pairs):
            incoming = pair['in_weight']
            outgoing = pair['out_weight']
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming_direction = incoming_directions[layer]
            outgoing_direction = outgoing_directions[layer]
            incoming_direction.view(self.groups, self.width, self.external).mul_(coefficients[layer, :, None, None].to(incoming_direction.dtype))
            outgoing_direction.view(self.external, self.groups, self.width).permute(1, 2, 0).mul_(coefficients[layer, :, None, None].to(outgoing_direction.dtype))
            incoming.add_(incoming_direction.to(incoming.dtype), alpha=-lr * incoming_adjustments[layer])
            outgoing.add_(outgoing_direction.to(outgoing.dtype), alpha=-lr * outgoing_adjustments[layer])
        if self._capture_telemetry_next_step:
            flat = coefficients.reshape(-1)
            transaction = selection.sharded_result
            prefix = self.telemetry_prefix
            report = {prefix + 'family_id': self.family_id, prefix + 'owner_count': transaction.owner_count, prefix + 'global_rows': _fixed_transaction_base__import_FIXED_GLOBAL_PROBE_COUNT, prefix + 'local_rows': self.probe_layout.local_probe_count, prefix + 'coordinate_count': len(self.pairs) * self.groups, prefix + 'state_coordinate_count': 0, prefix + 'state_depends_on_total_tokens': 0, prefix + 'dense_lg_metric_elements': transaction.dense_LG_by_LG_metric_elements, prefix + 'selected_update_elements_published': transaction.selected_update_elements_published, prefix + 'transaction_accepted': int(transaction.accepted.item()), prefix + 'rank': int(transaction.rank.item()), prefix + 'budget_residual': float(transaction.budget_residual.item()), prefix + 'parent_score': float(transaction.parent_score.item()), prefix + 'candidate_score': float(transaction.candidate_score.item()), prefix + 'cross_layer_coupling_ratio': float(selection.cross_layer_coupling_ratio.item()), prefix + 'coefficient_min': float(flat.amin().item()), prefix + 'coefficient_median': float(flat.median().item()), prefix + 'coefficient_max': float(flat.amax().item()), prefix + 'realized_clip_factor': float(self._clip_factor)}
            report.update(self._extra_transaction_telemetry(transaction))
            self._last_telemetry = report
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss
_response_fisher__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_response_fisher__import__zeropower_via_newton_schulz = _basis_trust___zeropower_via_newton_schulz
_response_fisher__import_Fixed32TransactionMuonBase = _fixed_transaction_base__Fixed32TransactionMuonBase
_response_fisher__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_response_fisher__import_inverse_root_diagonal_scale = _response_fisher_diagonal__inverse_root_diagonal_scale
_response_fisher__import_lagged_exponential_diagonal = _response_fisher_diagonal__lagged_exponential_diagonal
_response_fisher__import_method_state_elements = _response_fisher_diagonal__method_state_elements
_response_fisher__import_response_fisher_diagonal_sums = _response_fisher_diagonal__response_fisher_diagonal_sums
_response_fisher__import__version_a_factors = _probe_loss_image___version_a_factors
_response_fisher__FAMILY_ID = 'response_fisher_muon_v1'

def _response_fisher___response_adjoint(cotangents: torch.Tensor, outgoing_weight: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, groups: int, width: int) -> torch.Tensor:
    unit, derivative, radial = factors
    expected = (cotangents.shape[0], int(groups), int(width))
    if any((value.shape != expected for value in (unit, derivative, radial))):
        raise RuntimeError('response-Fisher rational factor inventory changed')
    pulled = (cotangents.float() @ outgoing_weight.float()).view(expected)
    result = derivative * pulled + unit * (radial * pulled).mean(dim=-1, keepdim=True)
    torch._assert_async(torch.isfinite(result).all())
    return result

class _response_fisher__ResponseFisherMuonOptimizer(_response_fisher__import_Fixed32TransactionMuonBase):
    """Precondition Muon sources with lagged GRAIN loss curvature."""
    family_id = _response_fisher__FAMILY_ID
    telemetry_prefix = 'response_fisher_'
    fairness_component = 'response_fisher_direction_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]['ten_probe_family_id'] = _response_fisher__FAMILY_ID
        self.param_groups[0]['fixed32_transaction_family_id'] = _response_fisher__FAMILY_ID
        self.param_groups[0]['response_fisher_family_id'] = _response_fisher__FAMILY_ID

    def _transaction(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError('response-Fisher Muon has no coefficient transaction')

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'matrix_role_lr_scale': 1.0, 'fixed32_loss_measure_lr_scale': 1.0, 'response_fisher_direction_lr_scale': 1.0, 'lagged_curvature_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def _global_curvature(self, packets):
        incoming_sums = []
        outgoing_sums = []
        for pair, packet in zip(self.pairs, packets):
            inputs, preactivations, features, cotangents = packet
            factors = _response_fisher__import__version_a_factors(preactivations, pair['numerator'], pair['denominator'], groups=self.groups, width=self.width, eps=self.grain_eps)
            response = _response_fisher___response_adjoint(cotangents, pair['out_weight'], factors, groups=self.groups, width=self.width)
            values = _response_fisher__import_response_fisher_diagonal_sums(inputs, features, cotangents, response)
            incoming_sums.append(values.incoming)
            outgoing_sums.append(values.outgoing)
        packed = torch.cat((torch.stack(incoming_sums).reshape(-1), torch.stack(outgoing_sums).reshape(-1)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, group=self.loss_probe_group)
        packed.div_(float(_response_fisher__import_FIXED_GLOBAL_PROBE_COUNT))
        incoming_elements = len(self.pairs) * self.groups * self.external
        incoming = packed[:incoming_elements].view(len(self.pairs), self.groups, self.external)
        outgoing = packed[incoming_elements:].view(len(self.pairs), self.groups, self.width)
        return (incoming, outgoing)

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('response-Fisher Muon lacks realized clipping')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('response-Fisher Muon refuses nonunit LR scale')
        packets = self._consume_probes()
        current_incoming, current_outgoing = self._global_curvature(packets)
        scale_samples = []
        initialized_count = 0
        for layer, pair in enumerate(self.pairs):
            incoming = pair['in_weight']
            outgoing = pair['out_weight']
            incoming_state = self.state[incoming]
            outgoing_state = self.state[outgoing]
            previous_incoming = incoming_state.get('response_fisher_diagonal')
            previous_outgoing = outgoing_state.get('response_fisher_diagonal')
            if previous_incoming is None:
                initialized_count += 1
            lagged_incoming, updated_incoming = _response_fisher__import_lagged_exponential_diagonal(current_incoming[layer], previous_incoming, decay=self.momentum)
            lagged_outgoing, updated_outgoing = _response_fisher__import_lagged_exponential_diagonal(current_outgoing[layer], previous_outgoing, decay=self.momentum)
            incoming_scale = _response_fisher__import_inverse_root_diagonal_scale(lagged_incoming, eps=1e-08)
            outgoing_scale = _response_fisher__import_inverse_root_diagonal_scale(lagged_outgoing, eps=1e-08)
            incoming_source = self._nesterov(incoming)
            outgoing_source = self._nesterov(outgoing)
            incoming_source.view(self.groups, self.width, self.external).mul_(incoming_scale[:, None, :].to(incoming_source.dtype))
            outgoing_source.view(self.external, self.groups, self.width).mul_(outgoing_scale[None, :, :].to(outgoing_source.dtype))
            incoming_direction = _response_fisher__import__zeropower_via_newton_schulz(incoming_source, self.ns_steps)
            outgoing_direction = _response_fisher__import__zeropower_via_newton_schulz(outgoing_source, self.ns_steps)
            incoming_adjustment = _response_fisher__import__match_rms_adamw_adjustment(incoming.shape)
            outgoing_adjustment = _response_fisher__import__match_rms_adamw_adjustment(outgoing.shape)
            incoming.mul_(1.0 - lr * weight_decay).add_(incoming_direction.to(incoming.dtype), alpha=-lr * incoming_adjustment)
            outgoing.mul_(1.0 - lr * weight_decay).add_(outgoing_direction.to(outgoing.dtype), alpha=-lr * outgoing_adjustment)
            incoming_state['response_fisher_diagonal'] = updated_incoming
            outgoing_state['response_fisher_diagonal'] = updated_outgoing
            if self._capture_telemetry_next_step:
                scale_samples.extend((incoming_scale.reshape(-1), outgoing_scale.reshape(-1)))
        anchor = self.state[self.pairs[0]['in_weight']]
        updates = int(anchor.get('response_fisher_updates', 0)) + 1
        anchor['response_fisher_updates'] = updates
        if self._capture_telemetry_next_step:
            scales = torch.cat(scale_samples)
            self._last_telemetry = {'response_fisher_family_id': _response_fisher__FAMILY_ID, 'response_fisher_global_rows': _response_fisher__import_FIXED_GLOBAL_PROBE_COUNT, 'response_fisher_local_rows': self.probe_layout.local_probe_count, 'response_fisher_owner_count': 0, 'response_fisher_dense_lg_metric_elements': 0, 'response_fisher_selected_update_elements_published': 0, 'response_fisher_state_depends_on_total_tokens': 0, 'response_fisher_state_coordinate_count': _response_fisher__import_method_state_elements(layers=len(self.pairs), groups=self.groups, width=self.width, model_width=self.external), 'response_fisher_decay': self.momentum, 'response_fisher_updates': updates, 'response_fisher_new_layers_initialized': initialized_count, 'response_fisher_scale_min': float(scales.amin().item()), 'response_fisher_scale_median': float(scales.median().item()), 'response_fisher_scale_max': float(scales.amax().item()), 'response_fisher_realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss
_residual_fisher__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_residual_fisher__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_residual_fisher__import__batched_zero_power = _group_numerics___batched_zero_power
_residual_fisher__import_inverse_root_diagonal_scale = _response_fisher_diagonal__inverse_root_diagonal_scale
_residual_fisher__import_lagged_exponential_diagonal = _response_fisher_diagonal__lagged_exponential_diagonal
_residual_fisher__import_ResponseFisherMuonOptimizer = _response_fisher__ResponseFisherMuonOptimizer
_residual_fisher__import__response_adjoint = _response_fisher___response_adjoint
_residual_fisher__import__version_a_factors = _probe_loss_image___version_a_factors
_residual_fisher__FAMILY_ID = 'residual_fisher_attention_muon_v1'

def _residual_fisher__residual_fisher_sums(inputs: torch.Tensor, response_adjoint: torch.Tensor) -> torch.Tensor:
    """Return the residual-coordinate loss curvature summed over GRAIN groups."""
    if inputs.ndim != 2 or response_adjoint.ndim != 3:
        raise RuntimeError('residual-Fisher tensor rank changed')
    probes, groups, _width = response_adjoint.shape
    if inputs.shape[0] != probes:
        raise RuntimeError('residual-Fisher probe inventory changed')
    if probes == 0:
        return torch.zeros(inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
    loss_power = response_adjoint.square().mean(dim=(-1, -2))
    result = loss_power @ inputs.square()
    torch._assert_async(torch.isfinite(result).all())
    return result

def _residual_fisher__residual_feedback_source(source: torch.Tensor, scale: torch.Tensor, *, role: str) -> torch.Tensor:
    """Apply the residual metric on the mathematically matching matrix side."""
    if source.ndim != 3 or scale.ndim != 2 or source.shape[0] != scale.shape[0]:
        raise RuntimeError('residual-feedback source inventory changed')
    if role == 'qkv':
        if source.shape[-1] != scale.shape[-1]:
            raise RuntimeError('QKV residual coordinate inventory changed')
        result = source.float() * scale[:, None, :]
    elif role == 'attn_out':
        if source.shape[-2] != scale.shape[-1]:
            raise RuntimeError('attention-output residual coordinate inventory changed')
        result = source.float() * scale[:, :, None]
    else:
        raise ValueError(f'unknown residual-feedback role: {role}')
    torch._assert_async(torch.isfinite(result).all())
    return result

def _residual_fisher__method_state_elements(*, layers: int, model_width: int) -> int:
    if int(layers) <= 0 or int(model_width) <= 0:
        raise ValueError('residual-Fisher dimensions must be positive')
    return int(layers) * int(model_width) + 1

class _residual_fisher__ResidualFisherAttentionRouter(_residual_fisher__import_ResponseFisherMuonOptimizer):
    """Literal GRAIN Muon plus one lagged residual loss metric for attention."""
    family_id = _residual_fisher__FAMILY_ID
    telemetry_prefix = 'residual_fisher_attention_'
    fairness_component = 'residual_fisher_attention_feedback_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]['ten_probe_family_id'] = _residual_fisher__FAMILY_ID
        self.param_groups[0]['fixed32_transaction_family_id'] = _residual_fisher__FAMILY_ID
        self.param_groups[0]['response_fisher_family_id'] = _residual_fisher__FAMILY_ID
        self.param_groups[0]['residual_fisher_attention_family_id'] = _residual_fisher__FAMILY_ID
        self._attention_scale = None
        self._attention_update = 0
        self._attention_consumed = True

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'rlb_literal_muon_lr_scale': 1.0, 'fixed32_loss_measure_lr_scale': 1.0, 'lagged_residual_fisher_lr_scale': 1.0, 'residual_fisher_attention_feedback_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def _global_residual_curvature(self, packets):
        local = []
        for pair, packet in zip(self.pairs, packets):
            inputs, preactivations, _features, cotangents = packet
            factors = _residual_fisher__import__version_a_factors(preactivations, pair['numerator'], pair['denominator'], groups=self.groups, width=self.width, eps=self.grain_eps)
            response = _residual_fisher__import__response_adjoint(cotangents, pair['out_weight'], factors, groups=self.groups, width=self.width)
            local.append(_residual_fisher__residual_fisher_sums(inputs, response))
        packed = torch.stack(local)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, group=self.loss_probe_group)
        packed.div_(float(_residual_fisher__import_FIXED_GLOBAL_PROBE_COUNT))
        torch._assert_async(torch.isfinite(packed).all())
        return packed

    def consume_attention_scale(self):
        if self._attention_consumed or self._attention_scale is None:
            raise RuntimeError('residual-Fisher attention route is unavailable')
        self._attention_consumed = True
        return (self._attention_scale, int(self._attention_update))

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('residual-Fisher router lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('residual-Fisher router would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('residual-Fisher router refuses nonunit LR scale')
        packets = self._consume_probes()
        current = self._global_residual_curvature(packets)
        anchor = self.state[self.pairs[0]['in_weight']]
        previous = anchor.get('residual_fisher_attention_diagonal')
        lagged, updated = _residual_fisher__import_lagged_exponential_diagonal(current, previous, decay=self.momentum)
        scale = _residual_fisher__import_inverse_root_diagonal_scale(lagged, eps=1e-08)
        anchor['residual_fisher_attention_diagonal'] = updated
        incoming_parameters = [pair['in_weight'] for pair in self.pairs]
        outgoing_parameters = [pair['out_weight'] for pair in self.pairs]
        incoming_sources = torch.stack([self._nesterov(parameter).float() for parameter in incoming_parameters])
        outgoing_sources = torch.stack([self._nesterov(parameter).float() for parameter in outgoing_parameters])
        incoming_direction = _residual_fisher__import__batched_zero_power(incoming_sources, self.ns_steps)
        outgoing_direction = _residual_fisher__import__batched_zero_power(outgoing_sources, self.ns_steps)
        incoming_adjustment = _residual_fisher__import__match_rms_adamw_adjustment(incoming_parameters[0].shape)
        outgoing_adjustment = _residual_fisher__import__match_rms_adamw_adjustment(outgoing_parameters[0].shape)
        for layer, pair in enumerate(self.pairs):
            pair['in_weight'].mul_(1.0 - lr * weight_decay).add_(incoming_direction[layer].to(pair['in_weight'].dtype), alpha=-lr * incoming_adjustment)
            pair['out_weight'].mul_(1.0 - lr * weight_decay).add_(outgoing_direction[layer].to(pair['out_weight'].dtype), alpha=-lr * outgoing_adjustment)
        updates = int(anchor.get('residual_fisher_attention_updates', 0)) + 1
        anchor['residual_fisher_attention_updates'] = updates
        self._attention_scale = scale
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            self._last_telemetry = {'residual_fisher_attention_family_id': _residual_fisher__FAMILY_ID, 'residual_fisher_attention_global_rows': _residual_fisher__import_FIXED_GLOBAL_PROBE_COUNT, 'residual_fisher_attention_local_rows': self.probe_layout.local_probe_count, 'residual_fisher_attention_owner_count': 0, 'residual_fisher_attention_dense_lg_metric_elements': 0, 'residual_fisher_attention_selected_update_elements_published': 0, 'residual_fisher_attention_state_depends_on_total_tokens': 0, 'residual_fisher_attention_state_coordinate_count': _residual_fisher__method_state_elements(layers=len(self.pairs), model_width=self.external), 'residual_fisher_attention_updates': updates, 'residual_fisher_attention_new_state_initialized': int(previous is None), 'residual_fisher_attention_scale_min': float(scale.amin().item()), 'residual_fisher_attention_scale_median': float(scale.median().item()), 'residual_fisher_attention_scale_max': float(scale.amax().item()), 'residual_fisher_attention_realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._attention_scale = None
        self._attention_update = 0
        self._attention_consumed = True
        return result

class _residual_fisher__ResidualFisherAttentionOptimizer(torch.optim.Optimizer):
    """Apply one GRAIN residual metric before each attention NS5."""
    _ROLES = ('qkv', 'attn_out')

    def __init__(self, blocks, router: _residual_fisher__ResidualFisherAttentionRouter, *, lr: float, weight_decay: float, momentum: float, ns_steps: int, beta2: float, eps: float, adjust_lr_fn: str):
        if float(lr) != 0.0003 or float(weight_decay) != 0.1 or float(momentum) != 0.95 or (int(ns_steps) != 5) or (float(beta2) != 0.95) or (float(eps) != 1e-08) or (adjust_lr_fn != 'match_rms_adamw'):
            raise ValueError('residual-Fisher attention requires the locked cell')
        self.blocks = sorted((dict(block) for block in blocks), key=lambda item: int(item['layer_index']))
        if [int(block['layer_index']) for block in self.blocks] != list(range(len(self.blocks))):
            raise ValueError('residual-Fisher attention layer inventory changed')
        self.router = router
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        self.role_parameters = {'qkv': [block['qkv_weight'] for block in self.blocks], 'attn_out': [block['attn_out_weight'] for block in self.blocks]}
        parameters = []
        seen = {id(pair[role]) for pair in router.pairs for role in ('in_weight', 'out_weight')}
        for role in self._ROLES:
            for parameter in self.role_parameters[role]:
                if parameter.ndim != 2 or id(parameter) in seen:
                    raise ValueError('residual-Fisher attention ownership changed')
                seen.add(id(parameter))
                parameters.append(parameter)
        super().__init__([{'params': parameters, 'lr': float(lr), 'weight_decay': float(weight_decay), 'lr_scale': 1.0, 'residual_fisher_attention_family_id': _residual_fisher__FAMILY_ID}], {})
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'qkv_lr_scale': 1.0, 'attention_output_lr_scale': 1.0, 'residual_coordinate_fisher_lr_scale': 1.0, 'single_ns5_per_attention_role_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def set_telemetry_capture(self, enabled=True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def _nesterov(self, parameter):
        if parameter.grad is None:
            raise RuntimeError('residual-Fisher attention gradient is missing')
        state = self.state[parameter]
        buffer = state.get('momentum_buffer')
        if buffer is None:
            buffer = torch.zeros_like(parameter)
            state['momentum_buffer'] = buffer
        buffer.lerp_(parameter.grad, 1.0 - self.momentum)
        return parameter.grad.lerp(buffer, self.momentum)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('residual-Fisher attention refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        scale, router_update = self.router.consume_attention_scale()
        if scale.shape != (len(self.blocks), self.blocks[0]['qkv_weight'].shape[1]):
            raise RuntimeError('residual-Fisher attention scale inventory changed')
        anchor = self.state[self.role_parameters['qkv'][0]]
        previous = int(anchor.get('residual_fisher_attention_router_update', 0))
        if int(router_update) != previous + 1:
            raise RuntimeError('residual-Fisher attention missed a router update')
        anchor['residual_fisher_attention_router_update'] = int(router_update)
        cosines = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            sources = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            selected = _residual_fisher__residual_feedback_source(sources, scale, role=role)
            direction = _residual_fisher__import__batched_zero_power(selected, self.ns_steps)
            adjustment = _residual_fisher__import__match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(direction[layer].to(parameter.dtype), alpha=-lr * adjustment)
            if self._capture_telemetry_next_step:
                numerator = (sources * selected).sum(dim=(-2, -1))
                denominator = (torch.linalg.vector_norm(sources, dim=(-2, -1)) * torch.linalg.vector_norm(selected, dim=(-2, -1))).clamp_min(torch.finfo(sources.dtype).tiny)
                cosines.append((numerator / denominator).clamp(-1.0, 1.0))
        if self._capture_telemetry_next_step:
            values = torch.cat(cosines)
            self._last_telemetry = {'residual_fisher_attention_direction_family_id': _residual_fisher__FAMILY_ID, 'residual_fisher_attention_direction_owner_count': 0, 'residual_fisher_attention_direction_dense_lg_metric_elements': 0, 'residual_fisher_attention_direction_selected_update_elements_published': 0, 'residual_fisher_attention_direction_state_depends_on_total_tokens': 0, 'residual_fisher_attention_direction_router_update': int(router_update), 'residual_fisher_attention_direction_source_cosine_min': float(values.amin().item()), 'residual_fisher_attention_direction_source_cosine_median': float(values.median().item()), 'residual_fisher_attention_direction_source_cosine_max': float(values.amax().item())}
        self._capture_telemetry_next_step = False
        return loss
_intrinsic_sign__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_intrinsic_sign__import__batched_zero_power = _group_numerics___batched_zero_power
_intrinsic_sign__import_ResidualFisherAttentionOptimizer = _residual_fisher__ResidualFisherAttentionOptimizer
_intrinsic_sign__import_ResidualFisherAttentionRouter = _residual_fisher__ResidualFisherAttentionRouter
_intrinsic_sign__import__jacobian_kernel_inner = _response_alignment_row___jacobian_kernel_inner
_intrinsic_sign__import__response_adjoint = _response_fisher___response_adjoint
_intrinsic_sign__import__version_a_factors = _probe_loss_image___version_a_factors
_intrinsic_sign__FAMILY_ID = 'loss_weighted_intrinsic_sign_attention_muon_v1'

def _intrinsic_sign__loss_weighted_intrinsic_participation_statistics(unit: torch.Tensor, derivative: torch.Tensor, radial: torch.Tensor, response_adjoint: torch.Tensor, cotangents: torch.Tensor) -> torch.Tensor:
    """Return additive incoming/outgoing weighted participation numerators."""
    if not unit.shape == derivative.shape == radial.shape == response_adjoint.shape:
        raise RuntimeError('intrinsic-participation rational inventory changed')
    if unit.ndim != 3 or cotangents.ndim != 2 or cotangents.shape[0] != unit.shape[0]:
        raise RuntimeError('intrinsic-participation probe inventory changed')
    width = float(unit.shape[-1])
    function = radial + unit * derivative
    radial_jacobian = radial / width
    trace = derivative.square().sum(dim=-1)
    trace = trace + 2.0 * (derivative * unit * radial_jacobian).sum(dim=-1)
    trace = trace + unit.square().sum(dim=-1) * radial_jacobian.square().sum(dim=-1)
    trace_square = _intrinsic_sign__import__jacobian_kernel_inner(unit, function, derivative, function, derivative)
    tiny = torch.finfo(trace.dtype).tiny
    incoming = (trace.square() / (width * trace_square.clamp_min(tiny))).clamp(0.0, 1.0)
    energy = function.square().sum(dim=-1)
    fourth = function.pow(4).sum(dim=-1)
    outgoing = (energy.square() / (width * fourth.clamp_min(tiny))).clamp(0.0, 1.0)
    incoming_weight = response_adjoint.square().mean(dim=-1)
    outgoing_weight = cotangents.float().square().mean(dim=-1)[:, None]
    result = torch.stack(((incoming * incoming_weight).sum(), incoming_weight.sum(), (outgoing * outgoing_weight).sum(), outgoing_weight.expand_as(outgoing).sum()))
    torch._assert_async(torch.isfinite(result).all())
    return result

def _intrinsic_sign__layer_loss_weighted_intrinsic_participation(statistics: torch.Tensor) -> torch.Tensor:
    """Map additive four-scalar layer statistics to a canonical chord cosine."""
    if statistics.ndim != 2 or statistics.shape[-1] != 4:
        raise RuntimeError('intrinsic-participation layer inventory changed')
    incoming = statistics[:, 0] / statistics[:, 1].clamp_min(torch.finfo(statistics.dtype).tiny)
    outgoing = statistics[:, 2] / statistics[:, 3].clamp_min(torch.finfo(statistics.dtype).tiny)
    valid = torch.isfinite(statistics).all(dim=-1) & (statistics[:, 1] > 0.0) & (statistics[:, 3] > 0.0)
    value = torch.sqrt((incoming * outgoing).clamp(0.0, 1.0))
    return torch.where(valid, value, torch.ones_like(value))

def _intrinsic_sign__equal_budget_intrinsic_sign_chord(parent: torch.Tensor, participation: torch.Tensor, *, eps: float=1e-08) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Join momentum to its equal-energy coordinate-sign source."""
    if parent.ndim != 3 or participation.shape != (parent.shape[0],):
        raise RuntimeError('intrinsic sign-chord inventory changed')
    if float(eps) != 1e-08:
        raise ValueError('intrinsic sign chord uses the locked epsilon')
    value = parent.float()
    dims = (-2, -1)
    parent_norm = torch.linalg.vector_norm(value, dim=dims, keepdim=True)
    sign = torch.sign(value)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = torch.isfinite(value).all(dim=dims, keepdim=True) & (parent_norm > 0.0) & (sign_norm > 0.0)
    sign_equal = sign * (parent_norm / sign_norm.clamp_min(torch.finfo(value.dtype).tiny))
    a = participation.float().clamp(0.0, 1.0)[:, None, None]
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source = a * value + delta * sign_equal
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    selected = source * (parent_norm / source_norm.clamp_min(torch.finfo(value.dtype).tiny))
    selected = torch.where(a == 1.0, value, selected)
    selected = torch.where(valid, selected, value)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    pnorm = parent_norm.flatten()
    cosine = ((value * selected).sum(dim=dims) / (pnorm * selected_norm).clamp_min(float(eps))).clamp(-1.0, 1.0)
    budget = (selected_norm - pnorm).abs() / pnorm.clamp_min(1.0)
    torch._assert_async(torch.isfinite(selected).all())
    return (selected, {'active': valid.flatten(), 'participation': participation.float().clamp(0.0, 1.0), 'parent_cosine': cosine, 'budget_residual': budget})

def _intrinsic_sign__method_state_elements(*, layers: int, model_width: int) -> int:
    if int(layers) <= 0 or int(model_width) <= 0:
        raise ValueError('intrinsic sign dimensions must be positive')
    return int(layers) * int(model_width) + 1

def _intrinsic_sign__communicated_summary_elements(*, layers: int) -> int:
    if int(layers) <= 0:
        raise ValueError('intrinsic sign depth must be positive')
    return 14 * int(layers)

class _intrinsic_sign__LossWeightedIntrinsicSignAttentionRouter(_intrinsic_sign__import_ResidualFisherAttentionRouter):
    """Literal GRAIN Muon plus one loss-weighted intrinsic attention angle."""
    family_id = _intrinsic_sign__FAMILY_ID
    telemetry_prefix = 'loss_weighted_intrinsic_sign_attention_'
    fairness_component = 'loss_weighted_intrinsic_sign_attention_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group['ten_probe_family_id'] = _intrinsic_sign__FAMILY_ID
        group['fixed32_transaction_family_id'] = _intrinsic_sign__FAMILY_ID
        group['response_fisher_family_id'] = _intrinsic_sign__FAMILY_ID
        group['residual_fisher_attention_family_id'] = _intrinsic_sign__FAMILY_ID
        group['loss_weighted_intrinsic_sign_attention_family_id'] = _intrinsic_sign__FAMILY_ID
        self._intrinsic_participation = None

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'rlb_literal_muon_lr_scale': 1.0, 'fixed32_loss_measure_lr_scale': 1.0, 'loss_weighted_intrinsic_participation_lr_scale': 1.0, 'attention_sign_chord_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def _global_residual_curvature(self, packets):
        values = []
        for pair, packet in zip(self.pairs, packets):
            _inputs, preactivations, _features, cotangents = packet
            factors = _intrinsic_sign__import__version_a_factors(preactivations, pair['numerator'], pair['denominator'], groups=self.groups, width=self.width, eps=self.grain_eps)
            response = _intrinsic_sign__import__response_adjoint(cotangents, pair['out_weight'], factors, groups=self.groups, width=self.width)
            values.append(_intrinsic_sign__loss_weighted_intrinsic_participation_statistics(*factors, response, cotangents))
        statistics = torch.stack(values)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        self._intrinsic_participation = _intrinsic_sign__layer_loss_weighted_intrinsic_participation(statistics)
        return torch.ones(len(self.pairs), self.external, device=statistics.device, dtype=statistics.dtype)

    def consume_intrinsic_participation(self):
        if self._attention_consumed or self._intrinsic_participation is None:
            raise RuntimeError('intrinsic-sign attention route is unavailable')
        self._attention_consumed = True
        return (self._intrinsic_participation, int(self._attention_update))

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._intrinsic_participation = None
        loss = super().step(closure)
        if self._intrinsic_participation is None:
            raise RuntimeError('intrinsic-sign router omitted participation')
        if publish:
            value = self._intrinsic_participation
            self._last_telemetry = {'loss_weighted_intrinsic_sign_attention_family_id': _intrinsic_sign__FAMILY_ID, 'loss_weighted_intrinsic_sign_attention_owner_count': 0, 'loss_weighted_intrinsic_sign_attention_dense_lg_metric_elements': 0, 'loss_weighted_intrinsic_sign_attention_selected_update_elements_published': 0, 'loss_weighted_intrinsic_sign_attention_state_depends_on_total_tokens': 0, 'loss_weighted_intrinsic_sign_attention_state_coordinate_count': _intrinsic_sign__method_state_elements(layers=len(self.pairs), model_width=self.external), 'loss_weighted_intrinsic_sign_attention_summary_elements': _intrinsic_sign__communicated_summary_elements(layers=len(self.pairs)), 'loss_weighted_intrinsic_sign_attention_updates': int(self._attention_update), 'loss_weighted_intrinsic_sign_attention_participation_min': float(value.amin().item()), 'loss_weighted_intrinsic_sign_attention_participation_median': float(value.median().item()), 'loss_weighted_intrinsic_sign_attention_participation_max': float(value.amax().item())}
        return loss

class _intrinsic_sign__LossWeightedIntrinsicSignAttentionOptimizer(_intrinsic_sign__import_ResidualFisherAttentionOptimizer):
    """One NS5 after the persistent GRAIN intrinsic sign chord."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['loss_weighted_intrinsic_sign_attention_family_id'] = _intrinsic_sign__FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'qkv_lr_scale': 1.0, 'attention_output_lr_scale': 1.0, 'equal_budget_sign_chord_lr_scale': 1.0, 'single_ns5_per_attention_role_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('intrinsic-sign attention refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        participation, router_update = self.router.consume_intrinsic_participation()
        anchor = self.state[self.role_parameters['qkv'][0]]
        previous = int(anchor.get('intrinsic_sign_attention_router_update', 0))
        if int(router_update) != previous + 1:
            raise RuntimeError('intrinsic-sign attention missed a router update')
        anchor['intrinsic_sign_attention_router_update'] = int(router_update)
        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            sources = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            alternative, metadata = _intrinsic_sign__equal_budget_intrinsic_sign_chord(sources, participation)
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent_descent = (gradients * sources).sum(dim=(-2, -1))
            alternative_descent = (gradients * alternative).sum(dim=(-2, -1))
            safe = metadata['active'] & torch.isfinite(parent_descent) & (parent_descent > 0.0) & torch.isfinite(alternative_descent) & (alternative_descent > 0.0)
            selected = torch.where(safe[:, None, None], alternative, sources)
            direction = _intrinsic_sign__import__batched_zero_power(selected, self.ns_steps)
            adjustment = _intrinsic_sign__import__match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(direction[layer].to(parameter.dtype), alpha=-lr * adjustment)
            if self._capture_telemetry_next_step:
                records.append((safe, metadata))
        if self._capture_telemetry_next_step:
            safe = torch.cat([item[0] for item in records])
            cosine = torch.cat([item[1]['parent_cosine'] for item in records])
            budget = torch.cat([item[1]['budget_residual'] for item in records])
            self._last_telemetry = {'loss_weighted_intrinsic_sign_direction_family_id': _intrinsic_sign__FAMILY_ID, 'loss_weighted_intrinsic_sign_direction_owner_count': 0, 'loss_weighted_intrinsic_sign_direction_dense_lg_metric_elements': 0, 'loss_weighted_intrinsic_sign_direction_selected_update_elements_published': 0, 'loss_weighted_intrinsic_sign_direction_state_depends_on_total_tokens': 0, 'loss_weighted_intrinsic_sign_direction_router_update': int(router_update), 'loss_weighted_intrinsic_sign_direction_safe_fraction': float(safe.float().mean().item()), 'loss_weighted_intrinsic_sign_direction_parent_cosine_min': float(cosine.amin().item()), 'loss_weighted_intrinsic_sign_direction_parent_cosine_median': float(cosine.median().item()), 'loss_weighted_intrinsic_sign_direction_parent_cosine_max': float(cosine.amax().item()), 'loss_weighted_intrinsic_sign_direction_budget_residual_max': float(budget.amax().item())}
        self._capture_telemetry_next_step = False
        return loss
_sign_transport__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_sign_transport__import__batched_zero_power = _group_numerics___batched_zero_power
_sign_transport__import_LossWeightedIntrinsicSignAttentionOptimizer = _intrinsic_sign__LossWeightedIntrinsicSignAttentionOptimizer
_sign_transport__import_LossWeightedIntrinsicSignAttentionRouter = _intrinsic_sign__LossWeightedIntrinsicSignAttentionRouter
_sign_transport__import__jacobian_kernel_inner = _response_alignment_row___jacobian_kernel_inner
_sign_transport__import__response_adjoint = _response_fisher___response_adjoint
_sign_transport__import__version_a_factors = _probe_loss_image___version_a_factors
_sign_transport__FAMILY_ID = 'loss_weighted_four_role_sign_transport_muon_v1'

def _sign_transport__group_loss_weighted_intrinsic_statistics(unit: torch.Tensor, derivative: torch.Tensor, radial: torch.Tensor, response_adjoint: torch.Tensor, cotangents: torch.Tensor) -> torch.Tensor:
    """Return four additive participation scalars for each rational group."""
    if not unit.shape == derivative.shape == radial.shape == response_adjoint.shape:
        raise RuntimeError('four-role intrinsic rational inventory changed')
    if unit.ndim != 3 or cotangents.ndim != 2 or cotangents.shape[0] != unit.shape[0]:
        raise RuntimeError('four-role intrinsic probe inventory changed')
    width = float(unit.shape[-1])
    function = radial + unit * derivative
    radial_jacobian = radial / width
    trace = derivative.square().sum(dim=-1)
    trace = trace + 2.0 * (derivative * unit * radial_jacobian).sum(dim=-1)
    trace = trace + unit.square().sum(dim=-1) * radial_jacobian.square().sum(dim=-1)
    trace_square = _sign_transport__import__jacobian_kernel_inner(unit, function, derivative, function, derivative)
    tiny = torch.finfo(trace.dtype).tiny
    incoming = (trace.square() / (width * trace_square.clamp_min(tiny))).clamp(0.0, 1.0)
    energy = function.square().sum(dim=-1)
    fourth = function.pow(4).sum(dim=-1)
    outgoing = (energy.square() / (width * fourth.clamp_min(tiny))).clamp(0.0, 1.0)
    incoming_weight = response_adjoint.square().mean(dim=-1)
    outgoing_weight = cotangents.float().square().mean(dim=-1)[:, None]
    result = torch.stack(((incoming * incoming_weight).sum(dim=0), incoming_weight.sum(dim=0), (outgoing * outgoing_weight).sum(dim=0), outgoing_weight.expand_as(outgoing).sum(dim=0)), dim=-1)
    torch._assert_async(torch.isfinite(result).all())
    return result

def _sign_transport__group_intrinsic_participation(statistics: torch.Tensor) -> torch.Tensor:
    """Map L-by-G additive statistics to incoming/outgoing participation."""
    if statistics.ndim != 3 or statistics.shape[-1] != 4:
        raise RuntimeError('four-role participation inventory changed')
    incoming = statistics[..., 0] / statistics[..., 1].clamp_min(torch.finfo(statistics.dtype).tiny)
    outgoing = statistics[..., 2] / statistics[..., 3].clamp_min(torch.finfo(statistics.dtype).tiny)
    valid = torch.isfinite(statistics).all(dim=-1) & (statistics[..., 1] > 0.0) & (statistics[..., 3] > 0.0)
    value = torch.stack((incoming, outgoing), dim=-1).clamp(0.0, 1.0)
    return torch.where(valid[..., None], value, torch.ones_like(value))

def _sign_transport__postpolar_group_sign_transport(parent: torch.Tensor, momentum: torch.Tensor, gradient: torch.Tensor, participation: torch.Tensor, *, groups: int, width: int | None, grouped_axis: str, eps: float=1e-08) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Transport a polar direction toward sign geometry at exact group budget."""
    if parent.shape != momentum.shape or parent.shape != gradient.shape or parent.ndim != 3:
        raise RuntimeError('post-polar transport tensor inventory changed')
    layers, rows, columns = parent.shape
    if participation.shape != (layers, int(groups)):
        raise RuntimeError('post-polar participation inventory changed')
    if float(eps) != 1e-08:
        raise ValueError('post-polar transport uses the locked epsilon')
    if grouped_axis == 'rows':
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError('post-polar row group inventory changed')
        p = parent.float().view(layers, int(groups), int(width), columns)
        m = momentum.float().view_as(p)
        g = gradient.float().view_as(p)
        restore = lambda value: value.reshape_as(parent)
    elif grouped_axis == 'columns':
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError('post-polar column group inventory changed')
        p = parent.float().transpose(-2, -1).contiguous().view(layers, int(groups), int(width), rows)
        m = momentum.float().transpose(-2, -1).contiguous().view_as(p)
        g = gradient.float().transpose(-2, -1).contiguous().view_as(p)
        restore = lambda value: value.reshape(layers, columns, rows).transpose(-2, -1).contiguous()
    elif grouped_axis == 'matrix':
        if int(groups) != 1 or width is not None:
            raise RuntimeError('post-polar matrix group inventory changed')
        p = parent.float()[:, None]
        m = momentum.float()[:, None]
        g = gradient.float()[:, None]
        restore = lambda value: value[:, 0]
    else:
        raise ValueError(f'unknown post-polar grouped axis: {grouped_axis}')
    dims = (-2, -1)
    tiny = torch.finfo(p.dtype).tiny
    parent_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
    sign = torch.sign(m)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = torch.isfinite(p).all(dim=dims, keepdim=True) & torch.isfinite(m).all(dim=dims, keepdim=True) & (parent_norm > 0.0) & (sign_norm > 0.0)
    sign_equal = sign * (parent_norm / sign_norm.clamp_min(tiny))
    c = participation.float().clamp(0.0, 1.0)[..., None, None]
    source = torch.sqrt(c) * p + torch.sqrt((1.0 - c).clamp_min(0.0)) * sign_equal
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    candidate = source * (parent_norm / source_norm.clamp_min(tiny))
    candidate = torch.where(c == 1.0, p, candidate)
    candidate = torch.where(valid, candidate, p)
    parent_descent = (g * p).sum(dim=dims)
    candidate_descent = (g * candidate).sum(dim=dims)
    safe = valid.flatten(2).all(dim=-1) & torch.isfinite(parent_descent) & (parent_descent > 0.0) & torch.isfinite(candidate_descent) & (candidate_descent > 0.0)
    selected = torch.where(safe[..., None, None], candidate, p)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    pnorm = parent_norm[..., 0, 0]
    cosine = ((p * selected).sum(dim=dims) / (pnorm * selected_norm).clamp_min(float(eps))).clamp(-1.0, 1.0)
    budget = (selected_norm - pnorm).abs() / pnorm.clamp_min(1.0)
    torch._assert_async(torch.isfinite(selected).all())
    return (restore(selected), {'active': valid[..., 0, 0], 'safe': safe, 'parent_cosine': cosine, 'budget_residual': budget, 'parent_descent': parent_descent, 'candidate_descent': candidate_descent})

def _sign_transport__method_state_elements() -> int:
    return 2

def _sign_transport__communicated_summary_elements(*, layers: int, groups: int) -> int:
    if int(layers) <= 0 or int(groups) <= 0:
        raise ValueError('four-role summary dimensions must be positive')
    return 14 * int(layers) * int(groups) + 10 * int(layers)

class _sign_transport__LossWeightedFourRoleSignTransportRouter(_sign_transport__import_LossWeightedIntrinsicSignAttentionRouter):
    """Apply GRAIN group participation after the GRAIN polar map."""
    family_id = _sign_transport__FAMILY_ID
    telemetry_prefix = 'loss_weighted_four_role_sign_transport_'
    fairness_component = 'loss_weighted_four_role_sign_transport_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group['ten_probe_family_id'] = _sign_transport__FAMILY_ID
        group['fixed32_transaction_family_id'] = _sign_transport__FAMILY_ID
        group['response_fisher_family_id'] = _sign_transport__FAMILY_ID
        group['residual_fisher_attention_family_id'] = _sign_transport__FAMILY_ID
        group['loss_weighted_intrinsic_sign_attention_family_id'] = _sign_transport__FAMILY_ID
        group['loss_weighted_four_role_sign_transport_family_id'] = _sign_transport__FAMILY_ID
        self._group_participation = None

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'rlb_incoming_lr_scale': 1.0, 'rlb_outgoing_lr_scale': 1.0, 'fixed32_loss_measure_lr_scale': 1.0, 'group_intrinsic_participation_lr_scale': 1.0, 'postpolar_sign_transport_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def _global_residual_curvature(self, packets):
        values = []
        for pair, packet in zip(self.pairs, packets):
            _inputs, preactivations, _features, cotangents = packet
            factors = _sign_transport__import__version_a_factors(preactivations, pair['numerator'], pair['denominator'], groups=self.groups, width=self.width, eps=self.grain_eps)
            response = _sign_transport__import__response_adjoint(cotangents, pair['out_weight'], factors, groups=self.groups, width=self.width)
            values.append(_sign_transport__group_loss_weighted_intrinsic_statistics(*factors, response, cotangents))
        statistics = torch.stack(values)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        participation = _sign_transport__group_intrinsic_participation(statistics)
        self._group_participation = participation
        incoming = participation[..., 0].mean(dim=-1)
        outgoing = participation[..., 1].mean(dim=-1)
        self._intrinsic_participation = torch.sqrt((incoming * outgoing).clamp(0.0, 1.0))
        return participation

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('four-role router lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('four-role router would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('four-role router refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        packets = self._consume_probes()
        participation = self._global_residual_curvature(packets)
        role_records = []
        for role, index, grouped_axis in (('incoming', 0, 'rows'), ('outgoing', 1, 'columns')):
            key = 'in_weight' if role == 'incoming' else 'out_weight'
            parameters = [pair[key] for pair in self.pairs]
            momenta = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent = _sign_transport__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _sign_transport__postpolar_group_sign_transport(parent, momenta, gradients, participation[..., index], groups=self.groups, width=self.width, grouped_axis=grouped_axis)
            adjustment = _sign_transport__import__match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(selected[layer].to(parameter.dtype), alpha=-lr * adjustment)
            role_records.append(metadata)
        anchor = self.state[self.pairs[0]['in_weight']]
        updates = int(anchor.get('four_role_sign_transport_updates', 0)) + 1
        anchor['four_role_sign_transport_updates'] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._intrinsic_participation is None:
            raise RuntimeError('four-role router omitted attention participation')
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            safe = torch.cat([record['safe'].reshape(-1) for record in role_records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in role_records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in role_records])
            self._last_telemetry = {'loss_weighted_four_role_sign_transport_family_id': _sign_transport__FAMILY_ID, 'loss_weighted_four_role_sign_transport_owner_count': 0, 'loss_weighted_four_role_sign_transport_dense_lg_metric_elements': 0, 'loss_weighted_four_role_sign_transport_selected_update_elements_published': 0, 'loss_weighted_four_role_sign_transport_state_depends_on_total_tokens': 0, 'loss_weighted_four_role_sign_transport_state_coordinate_count': _sign_transport__method_state_elements(), 'loss_weighted_four_role_sign_transport_summary_elements': _sign_transport__communicated_summary_elements(layers=len(self.pairs), groups=self.groups), 'loss_weighted_four_role_sign_transport_updates': updates, 'loss_weighted_four_role_sign_transport_participation_min': float(values.amin().item()), 'loss_weighted_four_role_sign_transport_participation_median': float(values.median().item()), 'loss_weighted_four_role_sign_transport_participation_max': float(values.amax().item()), 'loss_weighted_four_role_sign_transport_rlb_safe_fraction': float(safe.float().mean().item()), 'loss_weighted_four_role_sign_transport_rlb_parent_cosine_min': float(cosine.amin().item()), 'loss_weighted_four_role_sign_transport_rlb_parent_cosine_median': float(cosine.median().item()), 'loss_weighted_four_role_sign_transport_rlb_parent_cosine_max': float(cosine.amax().item()), 'loss_weighted_four_role_sign_transport_rlb_budget_residual_max': float(budget.amax().item()), 'loss_weighted_four_role_sign_transport_realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._group_participation = None
        return result

class _sign_transport__LossWeightedFourRoleSignTransportOptimizer(_sign_transport__import_LossWeightedIntrinsicSignAttentionOptimizer):
    """Apply the same post-polar transport to both attention matrix roles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['loss_weighted_four_role_sign_transport_family_id'] = _sign_transport__FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'qkv_lr_scale': 1.0, 'attention_output_lr_scale': 1.0, 'postpolar_sign_transport_lr_scale': 1.0, 'single_ns5_per_attention_role_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('four-role attention refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        participation, router_update = self.router.consume_intrinsic_participation()
        anchor = self.state[self.role_parameters['qkv'][0]]
        previous = int(anchor.get('four_role_sign_transport_router_update', 0))
        if int(router_update) != previous + 1:
            raise RuntimeError('four-role attention missed a router update')
        anchor['four_role_sign_transport_router_update'] = int(router_update)
        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent = _sign_transport__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _sign_transport__postpolar_group_sign_transport(parent, momenta, gradients, participation[:, None], groups=1, width=None, grouped_axis='matrix')
            adjustment = _sign_transport__import__match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(selected[layer].to(parameter.dtype), alpha=-lr * adjustment)
            records.append(metadata)
        if self._capture_telemetry_next_step:
            safe = torch.cat([record['safe'].reshape(-1) for record in records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in records])
            self._last_telemetry = {'loss_weighted_four_role_sign_direction_family_id': _sign_transport__FAMILY_ID, 'loss_weighted_four_role_sign_direction_owner_count': 0, 'loss_weighted_four_role_sign_direction_dense_lg_metric_elements': 0, 'loss_weighted_four_role_sign_direction_selected_update_elements_published': 0, 'loss_weighted_four_role_sign_direction_state_depends_on_total_tokens': 0, 'loss_weighted_four_role_sign_direction_router_update': int(router_update), 'loss_weighted_four_role_sign_direction_safe_fraction': float(safe.float().mean().item()), 'loss_weighted_four_role_sign_direction_parent_cosine_min': float(cosine.amin().item()), 'loss_weighted_four_role_sign_direction_parent_cosine_median': float(cosine.median().item()), 'loss_weighted_four_role_sign_direction_parent_cosine_max': float(cosine.amax().item()), 'loss_weighted_four_role_sign_direction_budget_residual_max': float(budget.amax().item())}
        self._capture_telemetry_next_step = False
        return loss
_response_homotopy__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_response_homotopy__import__batched_zero_power = _group_numerics___batched_zero_power
_response_homotopy__import_LossWeightedFourRoleSignTransportOptimizer = _sign_transport__LossWeightedFourRoleSignTransportOptimizer
_response_homotopy__import_LossWeightedFourRoleSignTransportRouter = _sign_transport__LossWeightedFourRoleSignTransportRouter
_response_homotopy__import_group_intrinsic_participation = _sign_transport__group_intrinsic_participation
_response_homotopy__import_group_loss_weighted_intrinsic_statistics = _sign_transport__group_loss_weighted_intrinsic_statistics
_response_homotopy__import__evaluate_response = _response_alignment_row___evaluate_response
_response_homotopy__import__jacobian_kernel_inner = _response_alignment_row___jacobian_kernel_inner
_response_homotopy__import__response_adjoint = _response_fisher___response_adjoint
_response_homotopy__import__version_a_factors = _probe_loss_image___version_a_factors
_response_homotopy__FAMILY_ID = 'loss_weighted_four_role_response_homotopy_muon_v1'

def _response_homotopy__combined_group_response_statistics(preactivation: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor, frozen_numerator: torch.Tensor, frozen_denominator: torch.Tensor, response_adjoint: torch.Tensor, cotangents: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, groups: int, width: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Share live factor evaluation across participation and drift kernels."""
    unit, live_d, radial = factors
    expected = (preactivation.shape[0], int(groups), int(width))
    if any((value.shape != expected for value in (unit, live_d, radial, response_adjoint))):
        raise RuntimeError('combined response sensor inventory changed')
    participation = _response_homotopy__import_group_loss_weighted_intrinsic_statistics(unit, live_d, radial, response_adjoint, cotangents)
    live_f = radial + unit * live_d
    frozen_f, frozen_d = _response_homotopy__import__evaluate_response(unit, frozen_numerator, frozen_denominator)
    value = preactivation.float().view(expected)
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    incoming_weight = response_adjoint.square().mean(dim=-1)
    outgoing_weight = cotangents.float().square().mean(dim=-1)[:, None]
    incoming_cross = (_response_homotopy__import__jacobian_kernel_inner(unit, live_f, live_d, frozen_f, frozen_d) * incoming_weight).sum(dim=0)
    incoming_live = (_response_homotopy__import__jacobian_kernel_inner(unit, live_f, live_d, live_f, live_d) * incoming_weight).sum(dim=0)
    incoming_frozen = (_response_homotopy__import__jacobian_kernel_inner(unit, frozen_f, frozen_d, frozen_f, frozen_d) * incoming_weight).sum(dim=0)
    live_h = rms * live_f
    frozen_h = rms * frozen_f
    outgoing_cross = ((live_h * frozen_h).sum(dim=-1).square() * outgoing_weight).sum(dim=0)
    outgoing_live = (live_h.square().sum(dim=-1).square() * outgoing_weight).sum(dim=0)
    outgoing_frozen = (frozen_h.square().sum(dim=-1).square() * outgoing_weight).sum(dim=0)
    response = torch.stack((torch.stack((incoming_cross, incoming_live, incoming_frozen), dim=-1), torch.stack((outgoing_cross, outgoing_live, outgoing_frozen), dim=-1)), dim=1)
    torch._assert_async(torch.isfinite(response).all())
    return (participation, response)

def _response_homotopy__group_loss_weighted_response_congruence(statistics: torch.Tensor, exact_initializer: torch.Tensor) -> torch.Tensor:
    """Return L-by-G incoming/outgoing live-to-frozen response cosines."""
    if statistics.ndim != 4 or statistics.shape[-2:] != (2, 3):
        raise RuntimeError('group response congruence inventory changed')
    if exact_initializer.shape != statistics.shape[:2]:
        raise RuntimeError('group response initializer inventory changed')
    denominator = torch.sqrt(statistics[..., 1] * statistics[..., 2])
    valid = torch.isfinite(statistics).all(dim=(-2, -1)) & (denominator > 0.0).all(dim=-1)
    result = (statistics[..., 0] / denominator.clamp_min(torch.finfo(statistics.dtype).tiny)).clamp(0.0, 1.0)
    result = torch.where(exact_initializer[..., None], torch.ones_like(result), result)
    return torch.where(valid[..., None], result, torch.ones_like(result))

def _response_homotopy__layer_loss_weighted_response_congruence(statistics: torch.Tensor, exact_initializer: torch.Tensor) -> torch.Tensor:
    """Aggregate group kernels before forming the canonical attention angle."""
    if statistics.ndim != 4 or statistics.shape[-2:] != (2, 3):
        raise RuntimeError('layer response congruence inventory changed')
    layer = statistics.sum(dim=1)
    denominator = torch.sqrt(layer[..., 1] * layer[..., 2])
    valid = torch.isfinite(layer).all(dim=(-2, -1)) & (denominator > 0.0).all(dim=-1)
    role = (layer[..., 0] / denominator.clamp_min(torch.finfo(layer.dtype).tiny)).clamp(0.0, 1.0)
    result = torch.sqrt((role[:, 0] * role[:, 1]).clamp_min(0.0))
    result = torch.where(exact_initializer.all(dim=-1), torch.ones_like(result), result)
    return torch.where(valid, result, torch.ones_like(result))

def _response_homotopy__postpolar_group_response_homotopy(parent: torch.Tensor, momentum: torch.Tensor, gradient: torch.Tensor, participation: torch.Tensor, congruence: torch.Tensor, *, groups: int, width: int | None, grouped_axis: str, eps: float=1e-08) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Parent -> intrinsic sign family -> frozen-response homotopy."""
    if parent.shape != momentum.shape or parent.shape != gradient.shape or parent.ndim != 3:
        raise RuntimeError('response homotopy tensor inventory changed')
    layers, rows, columns = parent.shape
    expected = (layers, int(groups))
    if participation.shape != expected or congruence.shape != expected:
        raise RuntimeError('response homotopy group inventory changed')
    if float(eps) != 1e-08:
        raise ValueError('response homotopy uses the locked epsilon')
    if grouped_axis == 'rows':
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError('response homotopy row inventory changed')
        p = parent.float().view(layers, int(groups), int(width), columns)
        m = momentum.float().view_as(p)
        g = gradient.float().view_as(p)
        restore = lambda value: value.reshape_as(parent)
    elif grouped_axis == 'columns':
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError('response homotopy column inventory changed')
        p = parent.float().transpose(-2, -1).contiguous().view(layers, int(groups), int(width), rows)
        m = momentum.float().transpose(-2, -1).contiguous().view_as(p)
        g = gradient.float().transpose(-2, -1).contiguous().view_as(p)
        restore = lambda value: value.reshape(layers, columns, rows).transpose(-2, -1).contiguous()
    elif grouped_axis == 'matrix':
        if int(groups) != 1 or width is not None:
            raise RuntimeError('response homotopy matrix inventory changed')
        p = parent.float()[:, None]
        m = momentum.float()[:, None]
        g = gradient.float()[:, None]
        restore = lambda value: value[:, 0]
    else:
        raise ValueError(f'unknown response homotopy grouped axis: {grouped_axis}')
    dims = (-2, -1)
    tiny = torch.finfo(p.dtype).tiny
    pnorm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
    sign = torch.sign(m)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = torch.isfinite(p).all(dim=dims, keepdim=True) & torch.isfinite(m).all(dim=dims, keepdim=True) & (pnorm > 0.0) & (sign_norm > 0.0)
    sign_equal = sign * (pnorm / sign_norm.clamp_min(tiny))
    c = participation.float().clamp(0.0, 1.0)[..., None, None]
    family_source = torch.sqrt(c) * p + torch.sqrt((1.0 - c).clamp_min(0.0)) * sign_equal
    family_norm = torch.linalg.vector_norm(family_source, dim=dims, keepdim=True)
    family = family_source * (pnorm / family_norm.clamp_min(tiny))
    family = torch.where(c == 1.0, p, family)
    a = congruence.float().clamp(0.0, 1.0)[..., None, None]
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source = a * p + delta * family
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    candidate = source * (pnorm / source_norm.clamp_min(tiny))
    candidate = torch.where(a == 1.0, p, candidate)
    candidate = torch.where(valid, candidate, p)
    parent_descent = (g * p).sum(dim=dims)
    candidate_descent = (g * candidate).sum(dim=dims)
    safe = valid[..., 0, 0] & torch.isfinite(parent_descent) & (parent_descent > 0.0) & torch.isfinite(candidate_descent) & (candidate_descent > 0.0)
    selected = torch.where(safe[..., None, None], candidate, p)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    parent_norm = pnorm[..., 0, 0]
    cosine = ((p * selected).sum(dim=dims) / (parent_norm * selected_norm).clamp_min(float(eps))).clamp(-1.0, 1.0)
    budget = (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)
    torch._assert_async(torch.isfinite(selected).all())
    return (restore(selected), {'active': valid[..., 0, 0], 'safe': safe, 'parent_cosine': cosine, 'budget_residual': budget, 'parent_descent': parent_descent, 'candidate_descent': candidate_descent})

def _response_homotopy__method_state_elements(*, layers: int, groups: int) -> int:
    if int(layers) <= 0 or int(groups) <= 0:
        raise ValueError('response-homotopy state dimensions must be positive')
    return 10 * int(layers) * int(groups) + 2

def _response_homotopy__communicated_summary_elements(*, layers: int, groups: int) -> int:
    if int(layers) <= 0 or int(groups) <= 0:
        raise ValueError('response-homotopy summary dimensions must be positive')
    return 21 * int(layers) * int(groups) + 10 * int(layers)

class _response_homotopy__LossWeightedFourRoleResponseHomotopyRouter(_response_homotopy__import_LossWeightedFourRoleSignTransportRouter):
    """Use response drift to delay the four-role sign family continuously."""
    family_id = _response_homotopy__FAMILY_ID
    telemetry_prefix = 'loss_weighted_four_role_response_homotopy_'
    fairness_component = 'loss_weighted_four_role_response_homotopy_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group['loss_weighted_four_role_sign_transport_family_id'] = _response_homotopy__FAMILY_ID
        group['loss_weighted_four_role_response_homotopy_family_id'] = _response_homotopy__FAMILY_ID
        anchor = self.state[self.pairs[0]['in_weight']]
        anchor.setdefault('four_role_response_frozen_numerators', torch.stack([pair['numerator'].detach().float().clone() for pair in self.pairs]))
        anchor.setdefault('four_role_response_frozen_denominators', torch.stack([pair['denominator'].detach().float().clone() for pair in self.pairs]))
        self._group_response_congruence = None
        self._attention_response_congruence = None

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'rlb_incoming_lr_scale': 1.0, 'rlb_outgoing_lr_scale': 1.0, 'fixed32_loss_measure_lr_scale': 1.0, 'group_intrinsic_participation_lr_scale': 1.0, 'loss_weighted_frozen_response_lr_scale': 1.0, 'postpolar_homotopy_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def _global_residual_curvature(self, packets):
        anchor = self.state[self.pairs[0]['in_weight']]
        frozen_numerators = anchor.get('four_role_response_frozen_numerators')
        frozen_denominators = anchor.get('four_role_response_frozen_denominators')
        if frozen_numerators is None or frozen_numerators.shape != (len(self.pairs), self.groups, 6) or frozen_denominators is None or (frozen_denominators.shape != (len(self.pairs), self.groups, 4)):
            raise RuntimeError('four-role response frozen inventory changed')
        participation_values = []
        response_values = []
        exact_values = []
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            _inputs, preactivations, _features, cotangents = packet
            factors = _response_homotopy__import__version_a_factors(preactivations, pair['numerator'], pair['denominator'], groups=self.groups, width=self.width, eps=self.grain_eps)
            response = _response_homotopy__import__response_adjoint(cotangents, pair['out_weight'], factors, groups=self.groups, width=self.width)
            participation, response_statistics = _response_homotopy__combined_group_response_statistics(preactivations, pair['numerator'], pair['denominator'], frozen_numerators[layer], frozen_denominators[layer], response, cotangents, factors, groups=self.groups, width=self.width, eps=self.grain_eps)
            participation_values.append(participation)
            response_values.append(response_statistics)
            exact_values.append(torch.all(pair['numerator'].detach().float() == frozen_numerators[layer], dim=-1) & torch.all(pair['denominator'].detach().float() == frozen_denominators[layer], dim=-1))
        participation_statistics = torch.stack(participation_values)
        response_statistics = torch.stack(response_values)
        first_elements = participation_statistics.numel()
        packed = torch.cat((participation_statistics.reshape(-1), response_statistics.reshape(-1)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        participation_statistics = packed[:first_elements].view_as(participation_statistics)
        response_statistics = packed[first_elements:].view_as(response_statistics)
        exact = torch.stack(exact_values).to(torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(exact, op=dist.ReduceOp.MIN, group=self.loss_probe_group)
        participation = _response_homotopy__import_group_intrinsic_participation(participation_statistics)
        group_congruence = _response_homotopy__group_loss_weighted_response_congruence(response_statistics, exact.bool())
        attention_congruence = _response_homotopy__layer_loss_weighted_response_congruence(response_statistics, exact.bool())
        incoming = participation[..., 0].mean(dim=-1)
        outgoing = participation[..., 1].mean(dim=-1)
        self._group_participation = participation
        self._group_response_congruence = group_congruence
        self._intrinsic_participation = torch.sqrt((incoming * outgoing).clamp(0.0, 1.0))
        self._attention_response_congruence = attention_congruence
        return participation

    def consume_response_homotopy(self):
        if self._attention_consumed:
            raise RuntimeError('four-role response homotopy was consumed twice')
        if self._intrinsic_participation is None or self._attention_response_congruence is None:
            raise RuntimeError('four-role response attention route is incomplete')
        self._attention_consumed = True
        return (self._intrinsic_participation, self._attention_response_congruence, int(self._attention_update))

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('four-role response router lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('four-role response router would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('four-role response router refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        packets = self._consume_probes()
        participation = self._global_residual_curvature(packets)
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError('four-role response router omitted group congruence')
        role_records = []
        for role, index, grouped_axis in (('incoming', 0, 'rows'), ('outgoing', 1, 'columns')):
            key = 'in_weight' if role == 'incoming' else 'out_weight'
            parameters = [pair[key] for pair in self.pairs]
            momenta = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent = _response_homotopy__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _response_homotopy__postpolar_group_response_homotopy(parent, momenta, gradients, participation[..., index], congruence[..., index], groups=self.groups, width=self.width, grouped_axis=grouped_axis)
            adjustment = _response_homotopy__import__match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(selected[layer].to(parameter.dtype), alpha=-lr * adjustment)
            role_records.append(metadata)
        anchor = self.state[self.pairs[0]['in_weight']]
        updates = int(anchor.get('four_role_response_homotopy_updates', 0)) + 1
        anchor['four_role_response_homotopy_updates'] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            angles = congruence.reshape(-1)
            safe = torch.cat([record['safe'].reshape(-1) for record in role_records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in role_records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in role_records])
            self._last_telemetry = {'loss_weighted_four_role_response_homotopy_family_id': _response_homotopy__FAMILY_ID, 'loss_weighted_four_role_response_homotopy_owner_count': 0, 'loss_weighted_four_role_response_homotopy_dense_lg_metric_elements': 0, 'loss_weighted_four_role_response_homotopy_selected_update_elements_published': 0, 'loss_weighted_four_role_response_homotopy_state_depends_on_total_tokens': 0, 'loss_weighted_four_role_response_homotopy_state_coordinate_count': _response_homotopy__method_state_elements(layers=len(self.pairs), groups=self.groups), 'loss_weighted_four_role_response_homotopy_summary_elements': _response_homotopy__communicated_summary_elements(layers=len(self.pairs), groups=self.groups), 'loss_weighted_four_role_response_homotopy_updates': updates, 'loss_weighted_four_role_response_homotopy_participation_min': float(values.amin().item()), 'loss_weighted_four_role_response_homotopy_participation_median': float(values.median().item()), 'loss_weighted_four_role_response_homotopy_participation_max': float(values.amax().item()), 'loss_weighted_four_role_response_homotopy_congruence_min': float(angles.amin().item()), 'loss_weighted_four_role_response_homotopy_congruence_median': float(angles.median().item()), 'loss_weighted_four_role_response_homotopy_congruence_max': float(angles.amax().item()), 'loss_weighted_four_role_response_homotopy_rlb_safe_fraction': float(safe.float().mean().item()), 'loss_weighted_four_role_response_homotopy_rlb_parent_cosine_min': float(cosine.amin().item()), 'loss_weighted_four_role_response_homotopy_rlb_parent_cosine_median': float(cosine.median().item()), 'loss_weighted_four_role_response_homotopy_rlb_parent_cosine_max': float(cosine.amax().item()), 'loss_weighted_four_role_response_homotopy_rlb_budget_residual_max': float(budget.amax().item()), 'loss_weighted_four_role_response_homotopy_realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._group_response_congruence = None
        self._attention_response_congruence = None
        return result

class _response_homotopy__LossWeightedFourRoleResponseHomotopyOptimizer(_response_homotopy__import_LossWeightedFourRoleSignTransportOptimizer):
    """Apply the same frozen-response homotopy to both attention roles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['loss_weighted_four_role_response_homotopy_family_id'] = _response_homotopy__FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'qkv_lr_scale': 1.0, 'attention_output_lr_scale': 1.0, 'loss_weighted_frozen_response_lr_scale': 1.0, 'postpolar_homotopy_lr_scale': 1.0, 'single_ns5_per_attention_role_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('four-role response attention refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        participation, congruence, router_update = self.router.consume_response_homotopy()
        anchor = self.state[self.role_parameters['qkv'][0]]
        previous = int(anchor.get('four_role_response_homotopy_router_update', 0))
        if int(router_update) != previous + 1:
            raise RuntimeError('four-role response attention missed a router update')
        anchor['four_role_response_homotopy_router_update'] = int(router_update)
        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = torch.stack([self._nesterov(parameter).float() for parameter in parameters])
            gradients = torch.stack([parameter.grad.detach().float() for parameter in parameters])
            parent = _response_homotopy__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _response_homotopy__postpolar_group_response_homotopy(parent, momenta, gradients, participation[:, None], congruence[:, None], groups=1, width=None, grouped_axis='matrix')
            adjustment = _response_homotopy__import__match_rms_adamw_adjustment(parameters[0].shape)
            for layer, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay).add_(selected[layer].to(parameter.dtype), alpha=-lr * adjustment)
            records.append(metadata)
        if self._capture_telemetry_next_step:
            safe = torch.cat([record['safe'].reshape(-1) for record in records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in records])
            self._last_telemetry = {'loss_weighted_four_role_response_direction_family_id': _response_homotopy__FAMILY_ID, 'loss_weighted_four_role_response_direction_owner_count': 0, 'loss_weighted_four_role_response_direction_dense_lg_metric_elements': 0, 'loss_weighted_four_role_response_direction_selected_update_elements_published': 0, 'loss_weighted_four_role_response_direction_state_depends_on_total_tokens': 0, 'loss_weighted_four_role_response_direction_router_update': int(router_update), 'loss_weighted_four_role_response_direction_safe_fraction': float(safe.float().mean().item()), 'loss_weighted_four_role_response_direction_parent_cosine_min': float(cosine.amin().item()), 'loss_weighted_four_role_response_direction_parent_cosine_median': float(cosine.median().item()), 'loss_weighted_four_role_response_direction_parent_cosine_max': float(cosine.amax().item()), 'loss_weighted_four_role_response_direction_budget_residual_max': float(budget.amax().item())}
        self._capture_telemetry_next_step = False
        return loss
_batched_homotopy__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_batched_homotopy__import_stacked_intrinsic_and_response_statistics = _batched_sensor__stacked_intrinsic_and_response_statistics
_batched_homotopy__import_stacked_response_adjoint = _batched_sensor__stacked_response_adjoint
_batched_homotopy__import_stacked_version_a_factors = _batched_sensor__stacked_version_a_factors
_batched_homotopy__import__batched_zero_power = _group_numerics___batched_zero_power
_batched_homotopy__import_LossWeightedFourRoleResponseHomotopyOptimizer = _response_homotopy__LossWeightedFourRoleResponseHomotopyOptimizer
_batched_homotopy__import_LossWeightedFourRoleResponseHomotopyRouter = _response_homotopy__LossWeightedFourRoleResponseHomotopyRouter
_batched_homotopy__import_communicated_summary_elements = _response_homotopy__communicated_summary_elements
_batched_homotopy__import_group_loss_weighted_response_congruence = _response_homotopy__group_loss_weighted_response_congruence
_batched_homotopy__import_layer_loss_weighted_response_congruence = _response_homotopy__layer_loss_weighted_response_congruence
_batched_homotopy__import_method_state_elements = _response_homotopy__method_state_elements
_batched_homotopy__import_postpolar_group_response_homotopy = _response_homotopy__postpolar_group_response_homotopy
_batched_homotopy__import_group_intrinsic_participation = _sign_transport__group_intrinsic_participation
_batched_homotopy__FAMILY_ID = 'loss_weighted_four_role_response_homotopy_batched_muon_v2'

def _batched_homotopy___foreach_nesterov(optimizer, parameters) -> torch.Tensor:
    gradients = []
    buffers = []
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError('batched response-homotopy gradient is missing')
        gradients.append(parameter.grad)
        state = optimizer.state[parameter]
        buffer = state.get('momentum_buffer')
        if buffer is None:
            buffer = torch.zeros_like(parameter.grad, memory_format=torch.preserve_format)
            state['momentum_buffer'] = buffer
        buffers.append(buffer)
    torch._foreach_lerp_(buffers, gradients, 1.0 - optimizer.momentum)
    values = torch._foreach_lerp(gradients, buffers, optimizer.momentum)
    return torch.stack(values).float()

def _batched_homotopy___foreach_apply(parameters, direction, *, decay: float, alpha: float) -> None:
    torch._foreach_mul_(parameters, float(decay))
    values = list(direction.to(parameters[0].dtype).unbind(dim=0))
    torch._foreach_add_(parameters, values, alpha=float(alpha))

class _batched_homotopy__BatchedFourRoleResponseHomotopyRouter(_batched_homotopy__import_LossWeightedFourRoleResponseHomotopyRouter):
    family_id = _batched_homotopy__FAMILY_ID
    telemetry_prefix = 'loss_weighted_four_role_response_homotopy_batched_'
    fairness_component = 'loss_weighted_four_role_response_homotopy_batched_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]['loss_weighted_four_role_response_homotopy_batched_family_id'] = _batched_homotopy__FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'rlb_incoming_lr_scale': 1.0, 'rlb_outgoing_lr_scale': 1.0, 'fixed32_loss_measure_lr_scale': 1.0, 'group_intrinsic_participation_lr_scale': 1.0, 'loss_weighted_frozen_response_lr_scale': 1.0, 'postpolar_homotopy_lr_scale': 1.0, 'batched_sensor_lr_scale': 1.0, 'foreach_realization_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    def _global_residual_curvature(self, packets):
        anchor = self.state[self.pairs[0]['in_weight']]
        frozen_numerators = anchor.get('four_role_response_frozen_numerators')
        frozen_denominators = anchor.get('four_role_response_frozen_denominators')
        layers = len(self.pairs)
        if frozen_numerators is None or frozen_numerators.shape != (layers, self.groups, 6) or frozen_denominators is None or (frozen_denominators.shape != (layers, self.groups, 4)):
            raise RuntimeError('batched response-homotopy frozen inventory changed')
        preactivations = torch.stack([packet[1] for packet in packets])
        cotangents = torch.stack([packet[3] for packet in packets])
        numerators = torch.stack([pair['numerator'].detach() for pair in self.pairs])
        denominators = torch.stack([pair['denominator'].detach() for pair in self.pairs])
        outgoing = torch.stack([pair['out_weight'].detach() for pair in self.pairs])
        factors = _batched_homotopy__import_stacked_version_a_factors(preactivations, numerators, denominators, groups=self.groups, width=self.width, eps=self.grain_eps)
        response_adjoint = _batched_homotopy__import_stacked_response_adjoint(cotangents, outgoing, factors, groups=self.groups, width=self.width)
        participation_statistics, response_statistics = _batched_homotopy__import_stacked_intrinsic_and_response_statistics(preactivations, cotangents, response_adjoint, factors, frozen_numerators, frozen_denominators, eps=self.grain_eps)
        first = participation_statistics.numel()
        packed = torch.cat((participation_statistics.reshape(-1), response_statistics.reshape(-1)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        participation_statistics = packed[:first].view_as(participation_statistics)
        response_statistics = packed[first:].view_as(response_statistics)
        exact = (torch.all(numerators.float() == frozen_numerators, dim=-1) & torch.all(denominators.float() == frozen_denominators, dim=-1)).to(torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(exact, op=dist.ReduceOp.MIN, group=self.loss_probe_group)
        participation = _batched_homotopy__import_group_intrinsic_participation(participation_statistics)
        group_congruence = _batched_homotopy__import_group_loss_weighted_response_congruence(response_statistics, exact.bool())
        attention_congruence = _batched_homotopy__import_layer_loss_weighted_response_congruence(response_statistics, exact.bool())
        incoming = participation[..., 0].mean(dim=-1)
        outgoing_participation = participation[..., 1].mean(dim=-1)
        self._group_participation = participation
        self._group_response_congruence = group_congruence
        self._intrinsic_participation = torch.sqrt((incoming * outgoing_participation).clamp(0.0, 1.0))
        self._attention_response_congruence = attention_congruence
        return participation

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('batched response router lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('batched response router would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('batched response router refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        participation = self._global_residual_curvature(self._consume_probes())
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError('batched response router omitted congruence')
        role_records = []
        for role, index, axis in (('incoming', 0, 'rows'), ('outgoing', 1, 'columns')):
            key = 'in_weight' if role == 'incoming' else 'out_weight'
            parameters = [pair[key] for pair in self.pairs]
            momenta = _batched_homotopy___foreach_nesterov(self, parameters)
            gradients = torch.stack([parameter.grad.detach() for parameter in parameters]).float()
            parent = _batched_homotopy__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _batched_homotopy__import_postpolar_group_response_homotopy(parent, momenta, gradients, participation[..., index], congruence[..., index], groups=self.groups, width=self.width, grouped_axis=axis)
            adjustment = _batched_homotopy__import__match_rms_adamw_adjustment(parameters[0].shape)
            _batched_homotopy___foreach_apply(parameters, selected, decay=1.0 - lr * weight_decay, alpha=-lr * adjustment)
            role_records.append(metadata)
        anchor = self.state[self.pairs[0]['in_weight']]
        updates = int(anchor.get('batched_response_homotopy_updates', 0)) + 1
        anchor['batched_response_homotopy_updates'] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            angles = congruence.reshape(-1)
            safe = torch.cat([record['safe'].reshape(-1) for record in role_records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in role_records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in role_records])
            prefix = 'loss_weighted_four_role_response_homotopy_batched_'
            self._last_telemetry = {prefix + 'family_id': _batched_homotopy__FAMILY_ID, prefix + 'owner_count': 0, prefix + 'dense_lg_metric_elements': 0, prefix + 'selected_update_elements_published': 0, prefix + 'state_depends_on_total_tokens': 0, prefix + 'state_coordinate_count': _batched_homotopy__import_method_state_elements(layers=len(self.pairs), groups=self.groups), prefix + 'summary_elements': _batched_homotopy__import_communicated_summary_elements(layers=len(self.pairs), groups=self.groups), prefix + 'updates': updates, prefix + 'participation_min': float(values.amin().item()), prefix + 'participation_median': float(values.median().item()), prefix + 'participation_max': float(values.amax().item()), prefix + 'congruence_min': float(angles.amin().item()), prefix + 'congruence_median': float(angles.median().item()), prefix + 'congruence_max': float(angles.amax().item()), prefix + 'rlb_safe_fraction': float(safe.float().mean().item()), prefix + 'rlb_parent_cosine_min': float(cosine.amin().item()), prefix + 'rlb_parent_cosine_median': float(cosine.median().item()), prefix + 'rlb_parent_cosine_max': float(cosine.amax().item()), prefix + 'rlb_budget_residual_max': float(budget.amax().item()), prefix + 'realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

class _batched_homotopy__BatchedFourRoleResponseHomotopyAttentionOptimizer(_batched_homotopy__import_LossWeightedFourRoleResponseHomotopyOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['loss_weighted_four_role_response_homotopy_batched_family_id'] = _batched_homotopy__FAMILY_ID

    def lr_wd_fairness_audit(self):
        return {'global_lr_scale': 1.0, 'qkv_lr_scale': 1.0, 'attention_output_lr_scale': 1.0, 'loss_weighted_frozen_response_lr_scale': 1.0, 'postpolar_homotopy_lr_scale': 1.0, 'single_ns5_per_attention_role_lr_scale': 1.0, 'foreach_realization_lr_scale': 1.0, 'weight_decay_scale': 1.0}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('batched response attention refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        participation, congruence, router_update = self.router.consume_response_homotopy()
        anchor = self.state[self.role_parameters['qkv'][0]]
        previous = int(anchor.get('batched_response_homotopy_router_update', 0))
        if int(router_update) != previous + 1:
            raise RuntimeError('batched response attention missed a router update')
        anchor['batched_response_homotopy_router_update'] = int(router_update)
        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = _batched_homotopy___foreach_nesterov(self, parameters)
            gradients = torch.stack([parameter.grad.detach() for parameter in parameters]).float()
            parent = _batched_homotopy__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _batched_homotopy__import_postpolar_group_response_homotopy(parent, momenta, gradients, participation[:, None], congruence[:, None], groups=1, width=None, grouped_axis='matrix')
            adjustment = _batched_homotopy__import__match_rms_adamw_adjustment(parameters[0].shape)
            _batched_homotopy___foreach_apply(parameters, selected, decay=1.0 - lr * weight_decay, alpha=-lr * adjustment)
            records.append(metadata)
        if self._capture_telemetry_next_step:
            safe = torch.cat([record['safe'].reshape(-1) for record in records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in records])
            prefix = 'loss_weighted_four_role_response_homotopy_batched_direction_'
            self._last_telemetry = {prefix + 'family_id': _batched_homotopy__FAMILY_ID, prefix + 'owner_count': 0, prefix + 'dense_lg_metric_elements': 0, prefix + 'selected_update_elements_published': 0, prefix + 'state_depends_on_total_tokens': 0, prefix + 'router_update': int(router_update), prefix + 'safe_fraction': float(safe.float().mean().item()), prefix + 'parent_cosine_min': float(cosine.amin().item()), prefix + 'parent_cosine_median': float(cosine.median().item()), prefix + 'parent_cosine_max': float(cosine.amax().item()), prefix + 'budget_residual_max': float(budget.amax().item())}
        self._capture_telemetry_next_step = False
        return loss
_compact_homotopy__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_compact_homotopy__import__batched_zero_power = _group_numerics___batched_zero_power
_compact_homotopy__import_BatchedFourRoleResponseHomotopyAttentionOptimizer = _batched_homotopy__BatchedFourRoleResponseHomotopyAttentionOptimizer
_compact_homotopy__import_BatchedFourRoleResponseHomotopyRouter = _batched_homotopy__BatchedFourRoleResponseHomotopyRouter
_compact_homotopy__import__foreach_apply = _batched_homotopy___foreach_apply
_compact_homotopy__import__foreach_nesterov = _batched_homotopy___foreach_nesterov
_compact_homotopy__import_communicated_summary_elements = _response_homotopy__communicated_summary_elements
_compact_homotopy__import_method_state_elements = _response_homotopy__method_state_elements
_compact_homotopy__FAMILY_ID = 'loss_weighted_four_role_response_homotopy_compact_muon_v3'

def _compact_homotopy__compact_postpolar_group_response_homotopy(parent: torch.Tensor, momentum: torch.Tensor, gradient: torch.Tensor, participation: torch.Tensor, congruence: torch.Tensor, *, groups: int, width: int | None, grouped_axis: str, eps: float=1e-08) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the exact homotopy through two scalar coefficients per group.

    Both nested normalized chords lie in ``span(parent, sign(momentum))``.
    Five group moments therefore recover the selected direction without
    materializing the intermediate sign-equalized, family, and outer-source
    tensors.  The returned tensor is still local and parameter-shaped; no
    selected update is communicated.
    """
    if parent.shape != momentum.shape or parent.shape != gradient.shape or parent.ndim != 3:
        raise RuntimeError('compact response homotopy tensor inventory changed')
    layers, rows, columns = parent.shape
    expected = (layers, int(groups))
    if participation.shape != expected or congruence.shape != expected:
        raise RuntimeError('compact response homotopy group inventory changed')
    if float(eps) != 1e-08:
        raise ValueError('compact response homotopy uses the locked epsilon')
    if grouped_axis == 'rows':
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError('compact response homotopy row inventory changed')
        p = parent.float().view(layers, int(groups), int(width), columns)
        m = momentum.float().view_as(p)
        g = gradient.float().view_as(p)
        restore = lambda value: value.reshape_as(parent)
    elif grouped_axis == 'columns':
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError('compact response homotopy column inventory changed')
        p = parent.float().transpose(-2, -1).contiguous().view(layers, int(groups), int(width), rows)
        m = momentum.float().transpose(-2, -1).contiguous().view_as(p)
        g = gradient.float().transpose(-2, -1).contiguous().view_as(p)
        restore = lambda value: value.reshape(layers, columns, rows).transpose(-2, -1).contiguous()
    elif grouped_axis == 'matrix':
        if int(groups) != 1 or width is not None:
            raise RuntimeError('compact response homotopy matrix inventory changed')
        p = parent.float()[:, None]
        m = momentum.float()[:, None]
        g = gradient.float()[:, None]
        restore = lambda value: value[:, 0]
    else:
        raise ValueError(f'unknown compact response homotopy axis: {grouped_axis}')
    dims = (-2, -1)
    tiny = torch.finfo(p.dtype).tiny
    sign = torch.sign(m)
    parent2 = p.square().sum(dim=dims)
    sign2 = sign.square().sum(dim=dims)
    parent_sign = (p * sign).sum(dim=dims)
    parent_descent = (g * p).sum(dim=dims)
    sign_descent = (g * sign).sum(dim=dims)
    valid = torch.isfinite(p).all(dim=dims) & torch.isfinite(m).all(dim=dims) & torch.isfinite(parent2) & torch.isfinite(sign2) & torch.isfinite(parent_sign) & (parent2 > 0.0) & (sign2 > 0.0)
    sign_scale = torch.sqrt(parent2 / sign2.clamp_min(tiny))
    c = participation.float().clamp(0.0, 1.0)
    root_c = torch.sqrt(c)
    root_one_minus_c = torch.sqrt((1.0 - c).clamp_min(0.0))
    family2 = parent2 + 2.0 * root_c * root_one_minus_c * sign_scale * parent_sign
    family_scale = torch.sqrt(parent2 / family2.clamp_min(tiny))
    family_parent = family_scale * root_c
    family_sign = family_scale * root_one_minus_c * sign_scale
    a = congruence.float().clamp(0.0, 1.0)
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    source_parent = a + delta * family_parent
    source_sign = delta * family_sign
    source2 = source_parent.square() * parent2 + source_sign.square() * sign2 + 2.0 * source_parent * source_sign * parent_sign
    source_scale = torch.sqrt(parent2 / source2.clamp_min(tiny))
    parent_coefficient = source_scale * source_parent
    sign_coefficient = source_scale * source_sign
    candidate_descent = parent_coefficient * parent_descent + sign_coefficient * sign_descent
    valid = valid & torch.isfinite(family2) & torch.isfinite(source2) & (family2 > 0.0) & (source2 > 0.0)
    safe = valid & torch.isfinite(parent_descent) & (parent_descent > 0.0) & torch.isfinite(candidate_descent) & (candidate_descent > 0.0)
    parent_coefficient = torch.where(safe, parent_coefficient, torch.ones_like(parent_coefficient))
    sign_coefficient = torch.where(safe, sign_coefficient, torch.zeros_like(sign_coefficient))
    selected2 = parent_coefficient.square() * parent2 + sign_coefficient.square() * sign2 + 2.0 * parent_coefficient * sign_coefficient * parent_sign
    selected_norm = torch.sqrt(selected2.clamp_min(0.0))
    parent_norm = torch.sqrt(parent2.clamp_min(0.0))
    cosine = ((parent_coefficient * parent2 + sign_coefficient * parent_sign) / (parent_norm * selected_norm).clamp_min(float(eps))).clamp(-1.0, 1.0)
    budget = (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)
    coefficient_shape = (*parent_coefficient.shape, 1, 1)
    p.mul_(parent_coefficient.view(coefficient_shape))
    p.addcmul_(sign, sign_coefficient.view(coefficient_shape))
    selected = restore(p)
    torch._assert_async(torch.isfinite(selected).all())
    return (selected, {'active': valid, 'safe': safe, 'parent_cosine': cosine, 'budget_residual': budget, 'parent_descent': parent_descent, 'candidate_descent': candidate_descent, 'parent_coefficient': parent_coefficient, 'sign_coefficient': sign_coefficient})

class _compact_homotopy__CompactFourRoleResponseHomotopyRouter(_compact_homotopy__import_BatchedFourRoleResponseHomotopyRouter):
    family_id = _compact_homotopy__FAMILY_ID
    telemetry_prefix = 'loss_weighted_four_role_response_homotopy_compact_'
    fairness_component = 'loss_weighted_four_role_response_homotopy_compact_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]['loss_weighted_four_role_response_homotopy_compact_family_id'] = _compact_homotopy__FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result['moment_exact_compact_lr_scale'] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('compact response router lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('compact response router would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('compact response router refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        participation = self._global_residual_curvature(self._consume_probes())
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError('compact response router omitted congruence')
        role_records = []
        for role, index, axis in (('incoming', 0, 'rows'), ('outgoing', 1, 'columns')):
            key = 'in_weight' if role == 'incoming' else 'out_weight'
            parameters = [pair[key] for pair in self.pairs]
            momenta = _compact_homotopy__import__foreach_nesterov(self, parameters)
            gradients = torch.stack([parameter.grad.detach() for parameter in parameters]).float()
            parent = _compact_homotopy__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _compact_homotopy__compact_postpolar_group_response_homotopy(parent, momenta, gradients, participation[..., index], congruence[..., index], groups=self.groups, width=self.width, grouped_axis=axis)
            adjustment = _compact_homotopy__import__match_rms_adamw_adjustment(parameters[0].shape)
            _compact_homotopy__import__foreach_apply(parameters, selected, decay=1.0 - lr * weight_decay, alpha=-lr * adjustment)
            role_records.append(metadata)
        anchor = self.state[self.pairs[0]['in_weight']]
        updates = int(anchor.get('compact_response_homotopy_updates', 0)) + 1
        anchor['compact_response_homotopy_updates'] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            values = participation.reshape(-1)
            angles = congruence.reshape(-1)
            safe = torch.cat([record['safe'].reshape(-1) for record in role_records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in role_records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in role_records])
            prefix = 'loss_weighted_four_role_response_homotopy_compact_'
            self._last_telemetry = {prefix + 'family_id': _compact_homotopy__FAMILY_ID, prefix + 'owner_count': 0, prefix + 'dense_lg_metric_elements': 0, prefix + 'selected_update_elements_published': 0, prefix + 'state_depends_on_total_tokens': 0, prefix + 'state_coordinate_count': _compact_homotopy__import_method_state_elements(layers=len(self.pairs), groups=self.groups), prefix + 'summary_elements': _compact_homotopy__import_communicated_summary_elements(layers=len(self.pairs), groups=self.groups), prefix + 'updates': updates, prefix + 'participation_min': float(values.amin().item()), prefix + 'participation_median': float(values.median().item()), prefix + 'participation_max': float(values.amax().item()), prefix + 'congruence_min': float(angles.amin().item()), prefix + 'congruence_median': float(angles.median().item()), prefix + 'congruence_max': float(angles.amax().item()), prefix + 'rlb_safe_fraction': float(safe.float().mean().item()), prefix + 'rlb_parent_cosine_min': float(cosine.amin().item()), prefix + 'rlb_parent_cosine_median': float(cosine.median().item()), prefix + 'rlb_parent_cosine_max': float(cosine.amax().item()), prefix + 'rlb_budget_residual_max': float(budget.amax().item()), prefix + 'realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

class _compact_homotopy__CompactFourRoleResponseHomotopyAttentionOptimizer(_compact_homotopy__import_BatchedFourRoleResponseHomotopyAttentionOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['loss_weighted_four_role_response_homotopy_compact_family_id'] = _compact_homotopy__FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result['moment_exact_compact_lr_scale'] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('compact response attention refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        participation, congruence, router_update = self.router.consume_response_homotopy()
        anchor = self.state[self.role_parameters['qkv'][0]]
        previous = int(anchor.get('compact_response_homotopy_router_update', 0))
        if int(router_update) != previous + 1:
            raise RuntimeError('compact response attention missed a router update')
        anchor['compact_response_homotopy_router_update'] = int(router_update)
        records = []
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = _compact_homotopy__import__foreach_nesterov(self, parameters)
            gradients = torch.stack([parameter.grad.detach() for parameter in parameters]).float()
            parent = _compact_homotopy__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _compact_homotopy__compact_postpolar_group_response_homotopy(parent, momenta, gradients, participation[:, None], congruence[:, None], groups=1, width=None, grouped_axis='matrix')
            adjustment = _compact_homotopy__import__match_rms_adamw_adjustment(parameters[0].shape)
            _compact_homotopy__import__foreach_apply(parameters, selected, decay=1.0 - lr * weight_decay, alpha=-lr * adjustment)
            records.append(metadata)
        if self._capture_telemetry_next_step:
            safe = torch.cat([record['safe'].reshape(-1) for record in records])
            cosine = torch.cat([record['parent_cosine'].reshape(-1) for record in records])
            budget = torch.cat([record['budget_residual'].reshape(-1) for record in records])
            prefix = 'loss_weighted_four_role_response_homotopy_compact_direction_'
            self._last_telemetry = {prefix + 'family_id': _compact_homotopy__FAMILY_ID, prefix + 'owner_count': 0, prefix + 'dense_lg_metric_elements': 0, prefix + 'selected_update_elements_published': 0, prefix + 'state_depends_on_total_tokens': 0, prefix + 'router_update': int(router_update), prefix + 'safe_fraction': float(safe.float().mean().item()), prefix + 'parent_cosine_min': float(cosine.amin().item()), prefix + 'parent_cosine_median': float(cosine.median().item()), prefix + 'parent_cosine_max': float(cosine.amax().item()), prefix + 'budget_residual_max': float(budget.amax().item())}
        self._capture_telemetry_next_step = False
        return loss
_response_transaction__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_response_transaction__import_stacked_intrinsic_and_response_statistics = _batched_sensor__stacked_intrinsic_and_response_statistics
_response_transaction__import_stacked_response_adjoint = _batched_sensor__stacked_response_adjoint
_response_transaction__import_stacked_version_a_factors = _batched_sensor__stacked_version_a_factors
_response_transaction__import_CompactFourRoleResponseHomotopyAttentionOptimizer = _compact_homotopy__CompactFourRoleResponseHomotopyAttentionOptimizer
_response_transaction__import_CompactFourRoleResponseHomotopyRouter = _compact_homotopy__CompactFourRoleResponseHomotopyRouter
_response_transaction__import_COMPACT_PARENT_FAMILY_ID = _compact_homotopy__FAMILY_ID
_response_transaction__import_compact_postpolar_group_response_homotopy = _compact_homotopy__compact_postpolar_group_response_homotopy
_response_transaction__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_response_transaction__import_replicated_fixed_probe_transaction = _fixed_probe_transaction__replicated_fixed_probe_transaction
_response_transaction__import__batched_zero_power = _group_numerics___batched_zero_power
_response_transaction__import__foreach_apply = _batched_homotopy___foreach_apply
_response_transaction__import__foreach_nesterov = _batched_homotopy___foreach_nesterov
_response_transaction__import_group_loss_weighted_response_congruence = _response_homotopy__group_loss_weighted_response_congruence
_response_transaction__import_layer_loss_weighted_response_congruence = _response_homotopy__layer_loss_weighted_response_congruence
_response_transaction__import_response_state_elements = _response_homotopy__method_state_elements
_response_transaction__import_group_intrinsic_participation = _sign_transport__group_intrinsic_participation
_response_transaction__import__one_layer_group_scores = _probe_loss_image___one_layer_group_scores
_response_transaction__FAMILY_ID = 'global_response_transaction_muon_v1'

def _response_transaction__global_response_transaction_scaling_formula(*, total_positions: int, total_layers: int, total_groups: int, intermediate_width: int, model_width: int) -> dict[str, int]:
    """Closed-form persistent/communication inventory for any logical shape."""
    values = tuple(map(int, (total_positions, total_layers, total_groups, intermediate_width, model_width)))
    if min(values) <= 0:
        raise ValueError('global response transaction dimensions must be positive')
    positions, layers, groups, hidden, model = values
    if hidden % groups:
        raise ValueError('intermediate width must be divisible by rational groups')
    coordinates = layers * groups
    rows = _response_transaction__import_FIXED_GLOBAL_PROBE_COUNT
    return {'total_positions': positions, 'persistent_state_elements': _response_transaction__import_response_state_elements(layers=layers, groups=groups), 'communicated_summary_elements': 21 * coordinates + 10 * layers + rows * coordinates + rows + rows * rows, 'largest_dense_solve_dimension': rows, 'dense_coordinate_metric_elements': 0, 'owner_count': 0, 'selected_update_elements_published': 0, 'local_direction_arithmetic_elements': 4 * layers * hidden * model}

class _response_transaction__GlobalResponseTransactionRouter(_response_transaction__import_CompactFourRoleResponseHomotopyRouter):
    """Coordinate the positive response parent with one signed global solve."""
    family_id = _response_transaction__FAMILY_ID
    telemetry_prefix = 'global_response_transaction_'
    fairness_component = 'global_response_transaction_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]['global_response_transaction_family_id'] = _response_transaction__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({'fixed32_global_loss_measure_lr_scale': 1.0, 'signed_global_group_transaction_lr_scale': 1.0, 'row_space_equality_budget_lr_scale': 1.0})
        return report

    def _response_sensor_with_cache(self, packets):
        """Evaluate the response parent once and retain local score factors."""
        anchor = self.state[self.pairs[0]['in_weight']]
        frozen_numerators = anchor.get('four_role_response_frozen_numerators')
        frozen_denominators = anchor.get('four_role_response_frozen_denominators')
        layers = len(self.pairs)
        if frozen_numerators is None or frozen_numerators.shape != (layers, self.groups, 6) or frozen_denominators is None or (frozen_denominators.shape != (layers, self.groups, 4)):
            raise RuntimeError('global response transaction frozen inventory changed')
        preactivations = torch.stack([packet[1] for packet in packets])
        cotangents = torch.stack([packet[3] for packet in packets])
        numerators = torch.stack([pair['numerator'].detach() for pair in self.pairs])
        denominators = torch.stack([pair['denominator'].detach() for pair in self.pairs])
        outgoing = torch.stack([pair['out_weight'].detach() for pair in self.pairs])
        factors = _response_transaction__import_stacked_version_a_factors(preactivations, numerators, denominators, groups=self.groups, width=self.width, eps=self.grain_eps)
        response_adjoint = _response_transaction__import_stacked_response_adjoint(cotangents, outgoing, factors, groups=self.groups, width=self.width)
        participation_statistics, response_statistics = _response_transaction__import_stacked_intrinsic_and_response_statistics(preactivations, cotangents, response_adjoint, factors, frozen_numerators, frozen_denominators, eps=self.grain_eps)
        first = participation_statistics.numel()
        packed = torch.cat((participation_statistics.reshape(-1), response_statistics.reshape(-1)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.loss_probe_group)
        participation_statistics = packed[:first].view_as(participation_statistics)
        response_statistics = packed[first:].view_as(response_statistics)
        exact = (torch.all(numerators.float() == frozen_numerators, dim=-1) & torch.all(denominators.float() == frozen_denominators, dim=-1)).to(torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(exact, op=dist.ReduceOp.MIN, group=self.loss_probe_group)
        participation = _response_transaction__import_group_intrinsic_participation(participation_statistics)
        group_congruence = _response_transaction__import_group_loss_weighted_response_congruence(response_statistics, exact.bool())
        attention_congruence = _response_transaction__import_layer_loss_weighted_response_congruence(response_statistics, exact.bool())
        incoming = participation[..., 0].mean(dim=-1)
        outgoing_participation = participation[..., 1].mean(dim=-1)
        self._group_participation = participation
        self._group_response_congruence = group_congruence
        self._intrinsic_participation = torch.sqrt((incoming * outgoing_participation).clamp(0.0, 1.0))
        self._attention_response_congruence = attention_congruence
        return (participation, factors, response_adjoint)

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('global response transaction lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('global response transaction would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('global response transaction refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        packets = self._consume_probes()
        participation, factors, response_adjoint = self._response_sensor_with_cache(packets)
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError('global response transaction omitted congruence')
        role_parameters = {}
        role_momenta = {}
        role_selected = {}
        role_adjustment = {}
        role_records = {}
        exact = []
        momentum_descent = []
        for role, index, axis in (('incoming', 0, 'rows'), ('outgoing', 1, 'columns')):
            key = 'in_weight' if role == 'incoming' else 'out_weight'
            parameters = [pair[key] for pair in self.pairs]
            momenta = _response_transaction__import__foreach_nesterov(self, parameters)
            gradients = torch.stack([parameter.grad.detach() for parameter in parameters]).float()
            parent = _response_transaction__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _response_transaction__import_compact_postpolar_group_response_homotopy(parent, momenta, gradients, participation[..., index], congruence[..., index], groups=self.groups, width=self.width, grouped_axis=axis)
            adjustment = _response_transaction__import__match_rms_adamw_adjustment(parameters[0].shape)
            if role == 'incoming':
                selected_blocks = selected.view(len(self.pairs), self.groups, self.width, self.external)
                gradient_blocks = gradients.view_as(selected_blocks)
                momentum_blocks = momenta.view_as(selected_blocks)
            else:
                selected_blocks = selected.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
                gradient_blocks = gradients.view_as(selected).view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
                momentum_blocks = momenta.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
            scaled_blocks = selected_blocks.float() * adjustment
            exact.append((gradient_blocks.float() * scaled_blocks).sum(dim=(-2, -1)))
            momentum_descent.append((momentum_blocks.float() * scaled_blocks).sum(dim=(-2, -1)))
            role_parameters[role] = parameters
            role_momenta[role] = momenta
            role_selected[role] = selected
            role_adjustment[role] = adjustment
            role_records[role] = metadata
        incoming_blocks = role_selected['incoming'].view(len(self.pairs), self.groups, self.width, self.external).float() * role_adjustment['incoming']
        outgoing_blocks = role_selected['outgoing'].view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1).float() * role_adjustment['outgoing']
        weights = (incoming_blocks.square().sum(dim=(-2, -1)) + outgoing_blocks.square().sum(dim=(-2, -1))).reshape(-1)
        local_scores = []
        local_decay = None
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            inputs, preactivations, features, cotangents = packet
            layer_factors = tuple((value[layer] for value in factors))
            score, _ = _response_transaction__import__one_layer_group_scores(inputs, preactivations, features, cotangents, incoming_blocks[layer].reshape(self.hidden, self.external), outgoing_blocks[layer].reshape(self.hidden, self.external), pair['out_weight'], layer_factors, groups=self.groups, width=self.width, cached_response_adjoint=response_adjoint[layer])
            decay_score, _ = _response_transaction__import__one_layer_group_scores(inputs, preactivations, features, cotangents, pair['in_weight'].float() * weight_decay, pair['out_weight'].T.float() * weight_decay, pair['out_weight'], layer_factors, groups=self.groups, width=self.width, cached_response_adjoint=response_adjoint[layer])
            local_scores.append(score)
            layer_decay = decay_score.sum(dim=-1)
            local_decay = layer_decay if local_decay is None else local_decay + layer_decay
        if local_decay is None:
            raise RuntimeError('global response transaction omitted decay action')
        score_lattice = torch.stack(local_scores, dim=1).reshape(self.probe_layout.local_probe_count, len(self.pairs) * self.groups)
        exact_by_role = torch.stack([value.reshape(-1) for value in exact])
        momentum_by_role = torch.stack([value.reshape(-1) for value in momentum_descent])
        layer_ids = torch.arange(len(self.pairs), device=weights.device, dtype=torch.int64).repeat_interleave(self.groups)
        selection = _response_transaction__import_replicated_fixed_probe_transaction(score_lattice, local_decay, exact_by_role, momentum_by_role, weights, layer_ids, global_probe_count=_response_transaction__import_FIXED_GLOBAL_PROBE_COUNT, total_layers=len(self.pairs), eta=lr, rounds=64, group=self.loss_probe_group)
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)
        incoming_selected = role_selected['incoming']
        incoming_selected.view(len(self.pairs), self.groups, self.width, self.external).mul_(coefficients[..., None, None].to(incoming_selected.dtype))
        outgoing_selected = role_selected['outgoing']
        outgoing_selected.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1).mul_(coefficients[..., None, None].to(outgoing_selected.dtype))
        for role in ('incoming', 'outgoing'):
            _response_transaction__import__foreach_apply(role_parameters[role], role_selected[role], decay=1.0 - lr * weight_decay, alpha=-lr * role_adjustment[role])
        anchor = self.state[self.pairs[0]['in_weight']]
        updates = int(anchor.get('global_response_transaction_updates', 0)) + 1
        anchor['global_response_transaction_updates'] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            transaction = selection.sharded_result
            flat = coefficients.reshape(-1)
            response_cosine = torch.cat([role_records[role]['parent_cosine'].reshape(-1) for role in ('incoming', 'outgoing')])
            response_safe = torch.cat([role_records[role]['safe'].reshape(-1) for role in ('incoming', 'outgoing')])
            scaling = _response_transaction__global_response_transaction_scaling_formula(total_positions=1, total_layers=len(self.pairs), total_groups=self.groups, intermediate_width=self.hidden, model_width=self.external)
            prefix = 'global_response_transaction_'
            self._last_telemetry = {prefix + 'family_id': _response_transaction__FAMILY_ID, prefix + 'parent_family_id': _response_transaction__import_COMPACT_PARENT_FAMILY_ID, prefix + 'owner_count': transaction.owner_count, prefix + 'global_rows': _response_transaction__import_FIXED_GLOBAL_PROBE_COUNT, prefix + 'coordinate_count': len(self.pairs) * self.groups, prefix + 'state_coordinate_count': scaling['persistent_state_elements'], prefix + 'state_depends_on_total_tokens': 0, prefix + 'summary_elements': scaling['communicated_summary_elements'], prefix + 'largest_dense_solve_dimension': _response_transaction__import_FIXED_GLOBAL_PROBE_COUNT, prefix + 'dense_lg_metric_elements': transaction.dense_LG_by_LG_metric_elements, prefix + 'selected_update_elements_published': transaction.selected_update_elements_published, prefix + 'transaction_accepted': int(transaction.accepted.item()), prefix + 'rank': int(transaction.rank.item()), prefix + 'budget_residual': float(transaction.budget_residual.item()), prefix + 'coefficient_min': float(flat.amin().item()), prefix + 'coefficient_median': float(flat.median().item()), prefix + 'coefficient_max': float(flat.amax().item()), prefix + 'cross_layer_coupling_ratio': float(selection.cross_layer_coupling_ratio.item()), prefix + 'response_parent_cosine_median': float(response_cosine.median().item()), prefix + 'response_safe_fraction': float(response_safe.float().mean().item()), prefix + 'realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

class _response_transaction__GlobalResponseTransactionAttentionOptimizer(_response_transaction__import_CompactFourRoleResponseHomotopyAttentionOptimizer):
    """Retain the same response-derived attention route as the positive parent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['global_response_transaction_family_id'] = _response_transaction__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report['global_response_transaction_attention_lr_scale'] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for key, value in tuple(self._last_telemetry.items()):
            if value == _response_transaction__import_COMPACT_PARENT_FAMILY_ID:
                self._last_telemetry[key] = _response_transaction__FAMILY_ID
        if self._last_telemetry:
            self._last_telemetry['global_response_transaction_attention_family_id'] = _response_transaction__FAMILY_ID
        return loss
_predictive_transaction__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_predictive_transaction__import_COMPACT_PARENT_FAMILY_ID = _compact_homotopy__FAMILY_ID
_predictive_transaction__import_compact_postpolar_group_response_homotopy = _compact_homotopy__compact_postpolar_group_response_homotopy
_predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_predictive_transaction__import_ReplicatedFixedProbeTransactionResult = _fixed_probe_transaction__ReplicatedFixedProbeTransactionResult
_predictive_transaction__import__gather_variable_probe_rows = _fixed_probe_transaction___gather_variable_probe_rows
_predictive_transaction__import_distributed_fixed_probe_transaction = _fixed_probe_transaction__distributed_fixed_probe_transaction
_predictive_transaction__import_CURRENT_IMPLEMENTATION_FAMILY_ID = _response_transaction__FAMILY_ID
_predictive_transaction__import_GlobalResponseTransactionAttentionOptimizer = _response_transaction__GlobalResponseTransactionAttentionOptimizer
_predictive_transaction__import_GlobalResponseTransactionRouter = _response_transaction__GlobalResponseTransactionRouter
_predictive_transaction__import_global_response_transaction_scaling_formula = _response_transaction__global_response_transaction_scaling_formula
_predictive_transaction__import__batched_zero_power = _group_numerics___batched_zero_power
_predictive_transaction__import__foreach_apply = _batched_homotopy___foreach_apply
_predictive_transaction__import__foreach_nesterov = _batched_homotopy___foreach_nesterov
_predictive_transaction__import__one_layer_group_scores = _probe_loss_image___one_layer_group_scores
_predictive_transaction__FAMILY_ID = 'lagged_predictive_response_transaction_muon_v1'

@dataclass(frozen=True)
class _predictive_transaction__MatchedBeta2PredictiveRows:
    selection_scores: torch.Tensor
    selection_decay_action: torch.Tensor
    updated_scores: torch.Tensor
    updated_decay_action: torch.Tensor
    history_used: bool
    relative_innovation: torch.Tensor

def _predictive_transaction__matched_beta2_predictive_rows(current_scores: torch.Tensor, current_decay_action: torch.Tensor, previous_scores: torch.Tensor | None, previous_decay_action: torch.Tensor | None, *, beta2: float) -> _predictive_transaction__MatchedBeta2PredictiveRows:
    """Select with pre-update history and update it using locked beta2."""
    if current_scores.ndim != 2 or current_scores.shape[0] != _predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT or current_decay_action.shape != (_predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT,) or (not current_scores.is_floating_point()) or (current_decay_action.dtype != current_scores.dtype) or (current_decay_action.device != current_scores.device) or (float(beta2) != 0.95) or (not bool(torch.isfinite(current_scores).all())) or (not bool(torch.isfinite(current_decay_action).all())):
        raise RuntimeError('matched-beta2 predictive row inventory changed')
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError('predictive score and decay histories must coinitialize')
    tiny = torch.finfo(current_scores.dtype).tiny
    if previous_scores is None:
        selection_scores = current_scores
        selection_decay = current_decay_action
        updated_scores = current_scores.detach().clone()
        updated_decay = current_decay_action.detach().clone()
        relative_innovation = torch.zeros((), device=current_scores.device, dtype=current_scores.dtype)
        history_used = False
    else:
        if previous_scores.shape != current_scores.shape or previous_decay_action.shape != current_decay_action.shape or previous_scores.dtype != current_scores.dtype or (previous_decay_action.dtype != current_scores.dtype) or (previous_scores.device != current_scores.device) or (previous_decay_action.device != current_scores.device) or (not bool(torch.isfinite(previous_scores).all())) or (not bool(torch.isfinite(previous_decay_action).all())):
            raise RuntimeError('matched-beta2 predictive history changed')
        selection_scores = previous_scores
        selection_decay = previous_decay_action
        innovation2 = (current_scores - previous_scores).square().sum() + (current_decay_action - previous_decay_action).square().sum()
        current2 = current_scores.square().sum() + current_decay_action.square().sum()
        relative_innovation = torch.sqrt(innovation2 / current2.clamp_min(tiny))
        updated_scores = previous_scores * float(beta2) + current_scores * (1.0 - float(beta2))
        updated_decay = previous_decay_action * float(beta2) + current_decay_action * (1.0 - float(beta2))
        history_used = True
    return _predictive_transaction__MatchedBeta2PredictiveRows(selection_scores=selection_scores, selection_decay_action=selection_decay, updated_scores=updated_scores, updated_decay_action=updated_decay, history_used=history_used, relative_innovation=relative_innovation)

def _predictive_transaction__lagged_predictive_response_transaction_scaling_formula(*, total_positions: int, total_layers: int, total_groups: int, intermediate_width: int, model_width: int) -> dict[str, int]:
    parent = _predictive_transaction__import_global_response_transaction_scaling_formula(total_positions=total_positions, total_layers=total_layers, total_groups=total_groups, intermediate_width=intermediate_width, model_width=model_width)
    coordinates = int(total_layers) * int(total_groups)
    predictive = _predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT * (coordinates + 1)
    result = dict(parent)
    result['persistent_state_elements'] = parent['persistent_state_elements'] + predictive
    result['predictive_state_elements'] = predictive
    return result

def _predictive_transaction___transaction_from_replicated_global_rows(global_scores: torch.Tensor, global_decay_action: torch.Tensor, exact_by_role: torch.Tensor, momentum_by_role: torch.Tensor, weights: torch.Tensor, layer_ids: torch.Tensor, *, total_layers: int, eta: float, gather_rounds: int, group) -> _predictive_transaction__import_ReplicatedFixedProbeTransactionResult:
    """Shard coordinates after the one current-row gather."""
    rows, coordinates = global_scores.shape
    if rows != _predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT or global_decay_action.shape != (rows,) or exact_by_role.shape != (2, coordinates) or (momentum_by_role.shape != exact_by_role.shape) or (weights.shape != (coordinates,)) or (layer_ids.shape != (coordinates,)):
        raise RuntimeError('predictive replicated transaction inventory changed')
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
    else:
        rank, world = (0, 1)
    coordinate_ids = torch.arange(coordinates, device=global_scores.device, dtype=torch.int64)
    local_ids = coordinate_ids[coordinate_ids.remainder(world).eq(rank)]
    decay_cross = global_scores.T @ global_decay_action / float(rows)
    sharded = _predictive_transaction__import_distributed_fixed_probe_transaction(global_scores[:, local_ids], exact_by_role[:, local_ids], momentum_by_role[:, local_ids], decay_cross[local_ids], weights[local_ids], layer_ids[local_ids], local_ids, total_coordinates=coordinates, total_layers=int(total_layers), eta=float(eta), rounds=64, group=group)
    coefficient_packet = torch.zeros(2 * coordinates, device=global_scores.device, dtype=global_scores.dtype)
    coefficient_packet[local_ids] = sharded.local_coefficients
    coefficient_packet[coordinates + local_ids] = sharded.local_candidate_coefficients
    if world > 1:
        dist.all_reduce(coefficient_packet, op=dist.ReduceOp.SUM, group=group)
    total_row_metric = global_scores @ global_scores.T
    total_square = total_row_metric.square().sum()
    within_square = torch.zeros_like(total_square)
    for layer in range(int(total_layers)):
        layer_scores = global_scores[:, layer_ids.eq(layer)]
        within_square.add_((layer_scores @ layer_scores.T).square().sum())
    coupling = torch.sqrt((total_square - within_square).clamp_min(0.0) / total_square.clamp_min(torch.finfo(total_square.dtype).tiny))
    return _predictive_transaction__import_ReplicatedFixedProbeTransactionResult(coefficients=coefficient_packet[:coordinates], candidate_coefficients=coefficient_packet[coordinates:], sharded_result=sharded, local_probe_count=_predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT // max(world, 1), global_probe_count=_predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT, cross_layer_coupling_ratio=coupling, collective_rounds=int(gather_rounds) + sharded.collective_rounds + int(world > 1), score_scalars_exchanged_per_rank=_predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT * (coordinates + 1), coefficient_scalars_exchanged_per_rank=2 * coordinates if world > 1 else 0, selected_update_elements_published=0, method_state_depends_on_total_tokens=False)

class _predictive_transaction__LaggedPredictiveResponseTransactionRouter(_predictive_transaction__import_GlobalResponseTransactionRouter):
    """Choose current response-parent coefficients from pre-update history."""
    family_id = _predictive_transaction__FAMILY_ID
    telemetry_prefix = 'lagged_predictive_response_transaction_'
    fairness_component = 'lagged_predictive_response_transaction_lr_scale'

    def __init__(self, pairs, **kwargs):
        beta2 = float(kwargs.get('beta2', float('nan')))
        if beta2 != 0.95:
            raise ValueError('predictive response transaction requires locked beta2=.95')
        super().__init__(pairs, **kwargs)
        self.predictive_beta2 = beta2
        self.param_groups[0]['lagged_predictive_response_transaction_family_id'] = _predictive_transaction__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({'matched_beta2_predictive_loss_image_lr_scale': 1.0, 'preupdate_history_selection_lr_scale': 1.0})
        return report

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('predictive response transaction lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('predictive response transaction would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('predictive response transaction refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        self._group_response_congruence = None
        self._attention_response_congruence = None
        packets = self._consume_probes()
        participation, factors, response_adjoint = self._response_sensor_with_cache(packets)
        congruence = self._group_response_congruence
        if congruence is None:
            raise RuntimeError('predictive response transaction omitted congruence')
        role_parameters = {}
        role_selected = {}
        role_adjustment = {}
        role_records = {}
        exact = []
        momentum_descent = []
        for role, index, axis in (('incoming', 0, 'rows'), ('outgoing', 1, 'columns')):
            key = 'in_weight' if role == 'incoming' else 'out_weight'
            parameters = [pair[key] for pair in self.pairs]
            momenta = _predictive_transaction__import__foreach_nesterov(self, parameters)
            gradients = torch.stack([parameter.grad.detach() for parameter in parameters]).float()
            parent = _predictive_transaction__import__batched_zero_power(momenta, self.ns_steps).float()
            selected, metadata = _predictive_transaction__import_compact_postpolar_group_response_homotopy(parent, momenta, gradients, participation[..., index], congruence[..., index], groups=self.groups, width=self.width, grouped_axis=axis)
            adjustment = _predictive_transaction__import__match_rms_adamw_adjustment(parameters[0].shape)
            if role == 'incoming':
                selected_blocks = selected.view(len(self.pairs), self.groups, self.width, self.external)
                gradient_blocks = gradients.view_as(selected_blocks)
                momentum_blocks = momenta.view_as(selected_blocks)
            else:
                selected_blocks = selected.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
                gradient_blocks = gradients.view_as(selected).view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
                momentum_blocks = momenta.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
            scaled_blocks = selected_blocks.float() * adjustment
            exact.append((gradient_blocks.float() * scaled_blocks).sum(dim=(-2, -1)))
            momentum_descent.append((momentum_blocks.float() * scaled_blocks).sum(dim=(-2, -1)))
            role_parameters[role] = parameters
            role_selected[role] = selected
            role_adjustment[role] = adjustment
            role_records[role] = metadata
        incoming_blocks = role_selected['incoming'].view(len(self.pairs), self.groups, self.width, self.external).float() * role_adjustment['incoming']
        outgoing_blocks = role_selected['outgoing'].view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1).float() * role_adjustment['outgoing']
        weights = (incoming_blocks.square().sum(dim=(-2, -1)) + outgoing_blocks.square().sum(dim=(-2, -1))).reshape(-1)
        local_scores = []
        local_decay = None
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            inputs, preactivations, features, cotangents = packet
            layer_factors = tuple((value[layer] for value in factors))
            score, _ = _predictive_transaction__import__one_layer_group_scores(inputs, preactivations, features, cotangents, incoming_blocks[layer].reshape(self.hidden, self.external), outgoing_blocks[layer].reshape(self.hidden, self.external), pair['out_weight'], layer_factors, groups=self.groups, width=self.width, cached_response_adjoint=response_adjoint[layer])
            decay_score, _ = _predictive_transaction__import__one_layer_group_scores(inputs, preactivations, features, cotangents, pair['in_weight'].float() * weight_decay, pair['out_weight'].T.float() * weight_decay, pair['out_weight'], layer_factors, groups=self.groups, width=self.width, cached_response_adjoint=response_adjoint[layer])
            local_scores.append(score)
            layer_decay = decay_score.sum(dim=-1)
            local_decay = layer_decay if local_decay is None else local_decay + layer_decay
        if local_decay is None:
            raise RuntimeError('predictive response transaction omitted decay action')
        score_lattice = torch.stack(local_scores, dim=1).reshape(self.probe_layout.local_probe_count, len(self.pairs) * self.groups)
        current_packet = torch.cat((score_lattice, local_decay[:, None]), dim=1)
        global_packet, gather_rounds = _predictive_transaction__import__gather_variable_probe_rows(current_packet, expected_global_rows=_predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT, group=self.loss_probe_group)
        current_global_scores = global_packet[:, :-1]
        current_global_decay = global_packet[:, -1]
        exact_by_role = torch.stack([value.reshape(-1) for value in exact])
        momentum_by_role = torch.stack([value.reshape(-1) for value in momentum_descent])
        layer_ids = torch.arange(len(self.pairs), device=weights.device, dtype=torch.int64).repeat_interleave(self.groups)
        anchor = self.state[self.pairs[0]['in_weight']]
        predictive = _predictive_transaction__matched_beta2_predictive_rows(current_global_scores, current_global_decay, anchor.get('predictive_global_score_ema'), anchor.get('predictive_global_decay_ema'), beta2=self.predictive_beta2)
        selection = _predictive_transaction___transaction_from_replicated_global_rows(predictive.selection_scores, predictive.selection_decay_action, exact_by_role, momentum_by_role, weights, layer_ids, total_layers=len(self.pairs), eta=lr, gather_rounds=gather_rounds, group=self.loss_probe_group)
        anchor['predictive_global_score_ema'] = predictive.updated_scores.detach().clone()
        anchor['predictive_global_decay_ema'] = predictive.updated_decay_action.detach().clone()
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)
        incoming_selected = role_selected['incoming']
        incoming_selected.view(len(self.pairs), self.groups, self.width, self.external).mul_(coefficients[..., None, None].to(incoming_selected.dtype))
        outgoing_selected = role_selected['outgoing']
        outgoing_selected.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1).mul_(coefficients[..., None, None].to(outgoing_selected.dtype))
        for role in ('incoming', 'outgoing'):
            _predictive_transaction__import__foreach_apply(role_parameters[role], role_selected[role], decay=1.0 - lr * weight_decay, alpha=-lr * role_adjustment[role])
        updates = int(anchor.get('predictive_response_transaction_updates', 0)) + 1
        anchor['predictive_response_transaction_updates'] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            transaction = selection.sharded_result
            flat = coefficients.reshape(-1)
            response_cosine = torch.cat([role_records[role]['parent_cosine'].reshape(-1) for role in ('incoming', 'outgoing')])
            response_safe = torch.cat([role_records[role]['safe'].reshape(-1) for role in ('incoming', 'outgoing')])
            scaling = _predictive_transaction__lagged_predictive_response_transaction_scaling_formula(total_positions=1, total_layers=len(self.pairs), total_groups=self.groups, intermediate_width=self.hidden, model_width=self.external)
            prefix = 'lagged_predictive_response_transaction_'
            self._last_telemetry = {prefix + 'family_id': _predictive_transaction__FAMILY_ID, prefix + 'parent_family_id': _predictive_transaction__import_COMPACT_PARENT_FAMILY_ID, prefix + 'owner_count': transaction.owner_count, prefix + 'global_rows': _predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT, prefix + 'coordinate_count': len(self.pairs) * self.groups, prefix + 'state_coordinate_count': scaling['persistent_state_elements'], prefix + 'predictive_state_elements': scaling['predictive_state_elements'], prefix + 'state_depends_on_total_tokens': 0, prefix + 'summary_elements': scaling['communicated_summary_elements'], prefix + 'largest_dense_solve_dimension': _predictive_transaction__import_FIXED_GLOBAL_PROBE_COUNT, prefix + 'dense_lg_metric_elements': transaction.dense_LG_by_LG_metric_elements, prefix + 'selected_update_elements_published': transaction.selected_update_elements_published, prefix + 'transaction_accepted': int(transaction.accepted.item()), prefix + 'history_used': int(predictive.history_used), prefix + 'matched_beta2': self.predictive_beta2, prefix + 'relative_score_innovation': float(predictive.relative_innovation.item()), prefix + 'rank': int(transaction.rank.item()), prefix + 'budget_residual': float(transaction.budget_residual.item()), prefix + 'coefficient_min': float(flat.amin().item()), prefix + 'coefficient_median': float(flat.median().item()), prefix + 'coefficient_max': float(flat.amax().item()), prefix + 'cross_layer_coupling_ratio': float(selection.cross_layer_coupling_ratio.item()), prefix + 'response_parent_cosine_median': float(response_cosine.median().item()), prefix + 'response_safe_fraction': float(response_safe.float().mean().item()), prefix + 'realized_clip_factor': float(self._clip_factor)}
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

class _predictive_transaction__LaggedPredictiveResponseTransactionAttentionOptimizer(_predictive_transaction__import_GlobalResponseTransactionAttentionOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['lagged_predictive_response_transaction_family_id'] = _predictive_transaction__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.pop('global_response_transaction_attention_lr_scale')
        report['lagged_predictive_response_attention_lr_scale'] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for key, value in tuple(self._last_telemetry.items()):
            if value == _predictive_transaction__import_CURRENT_IMPLEMENTATION_FAMILY_ID:
                self._last_telemetry[key] = _predictive_transaction__FAMILY_ID
        if self._last_telemetry:
            self._last_telemetry['lagged_predictive_response_transaction_attention_family_id'] = _predictive_transaction__FAMILY_ID
        return loss
_temporal_geometry__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_temporal_geometry__import__literal_zero_power = _group_numerics___batched_zero_power
_temporal_geometry__import_LaggedPredictiveResponseTransactionAttentionOptimizer = _predictive_transaction__LaggedPredictiveResponseTransactionAttentionOptimizer
_temporal_geometry__import_LaggedPredictiveResponseTransactionRouter = _predictive_transaction__LaggedPredictiveResponseTransactionRouter
_temporal_geometry__import_MatchedBeta2PredictiveRows = _predictive_transaction__MatchedBeta2PredictiveRows
_temporal_geometry__FAMILY_ID = 'temporal_response_group_polar_muon_v1'
_temporal_geometry__ATTENTION_HEADS = 16
_temporal_geometry__ATTENTION_HEAD_GROUP_SIZE = 4
_temporal_geometry___PATCH_LOCK = threading.RLock()
_temporal_geometry___EXPECTED_PREDICTIVE_ROWS = _predictive_transaction__matched_beta2_predictive_rows
_temporal_geometry___EXPECTED_TRANSACTION = _predictive_transaction___transaction_from_replicated_global_rows
_temporal_geometry___EXPECTED_ROUTER_POLAR = _predictive_transaction__import__batched_zero_power
_temporal_geometry___EXPECTED_ATTENTION_POLAR = _compact_homotopy__import__batched_zero_power

def _temporal_geometry___validate_predictive_rows(current_scores: torch.Tensor, current_decay_action: torch.Tensor, previous_scores: torch.Tensor | None, previous_decay_action: torch.Tensor | None, *, beta2: float) -> None:
    rows = _temporal_geometry__import_FIXED_GLOBAL_PROBE_COUNT
    if current_scores.ndim != 2 or current_scores.shape[0] != rows or current_decay_action.shape != (rows,) or (current_decay_action.dtype != current_scores.dtype) or (current_decay_action.device != current_scores.device) or (not current_scores.is_floating_point()) or (float(beta2) != 0.95) or (not bool(torch.isfinite(current_scores).all())) or (not bool(torch.isfinite(current_decay_action).all())):
        raise RuntimeError('temporal response sketch inventory changed')
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError('temporal response sketch histories must coinitialize')
    if previous_scores is not None and (previous_scores.shape != current_scores.shape or previous_decay_action is None or previous_decay_action.shape != current_decay_action.shape or (previous_scores.dtype != current_scores.dtype) or (previous_decay_action.dtype != current_scores.dtype) or (previous_scores.device != current_scores.device) or (previous_decay_action.device != current_scores.device) or (not bool(torch.isfinite(previous_scores).all())) or (not bool(torch.isfinite(previous_decay_action).all()))):
        raise RuntimeError('temporal response sketch checkpoint changed')

def _temporal_geometry__temporal_response_predictive_rows(current_scores: torch.Tensor, current_decay_action: torch.Tensor, previous_scores: torch.Tensor | None, previous_decay_action: torch.Tensor | None, *, beta2: float) -> _temporal_geometry__import_MatchedBeta2PredictiveRows:
    """Predict with the old covariance factor, then update it by FD.

    If ``B`` is the previous factor and ``S`` the current fixed loss image,
    the augmented factor is ``[sqrt(beta2) B; sqrt(1-beta2) S]``.  The top 32
    Frequent-Directions rows and the identically transformed decay action are
    retained.  Consequently row rotations or permutations do not define the
    temporal correspondence, unlike an elementwise row EMA.
    """
    _temporal_geometry___validate_predictive_rows(current_scores, current_decay_action, previous_scores, previous_decay_action, beta2=beta2)
    if previous_scores is None:
        zero = current_scores.new_zeros(())
        return _temporal_geometry__import_MatchedBeta2PredictiveRows(selection_scores=current_scores, selection_decay_action=current_decay_action, updated_scores=current_scores.detach().clone(), updated_decay_action=current_decay_action.detach().clone(), history_used=False, relative_innovation=zero)
    assert previous_decay_action is not None
    rows = _temporal_geometry__import_FIXED_GLOBAL_PROBE_COUNT
    beta = float(beta2)
    augmented = torch.cat((previous_scores * math.sqrt(beta), current_scores * math.sqrt(1.0 - beta)), dim=0)
    augmented_decay = torch.cat((previous_decay_action * math.sqrt(beta), current_decay_action * math.sqrt(1.0 - beta)), dim=0)
    row_gram = augmented @ augmented.T
    row_gram = 0.5 * (row_gram + row_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(row_gram)
    eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(1)
    kept = eigenvalues[:rows]
    shrinkage = eigenvalues[rows]
    tiny = torch.finfo(current_scores.dtype).tiny
    ratios = torch.sqrt((kept - shrinkage).clamp_min(0.0) / kept.clamp_min(tiny))
    transform = ratios[:, None] * eigenvectors[:, :rows].T
    updated_scores = transform @ augmented
    updated_decay = transform @ augmented_decay
    current_gram = current_scores @ current_scores.T
    previous_gram = previous_scores @ previous_scores.T
    cross_gram = current_scores @ previous_scores.T
    innovation2 = (current_gram.square().sum() + previous_gram.square().sum() - 2.0 * cross_gram.square().sum()).clamp_min(0.0)
    relative_innovation = torch.sqrt(innovation2 / current_gram.square().sum().clamp_min(tiny))
    torch._assert_async(torch.isfinite(updated_scores).all() & torch.isfinite(updated_decay).all() & torch.isfinite(relative_innovation))
    return _temporal_geometry__import_MatchedBeta2PredictiveRows(selection_scores=previous_scores, selection_decay_action=previous_decay_action, updated_scores=updated_scores, updated_decay_action=updated_decay, history_used=True, relative_innovation=relative_innovation)

def _temporal_geometry__rational_group_zero_power(source: torch.Tensor, steps: int, *, groups: int, width: int) -> torch.Tensor:
    """Apply literal NS5 to rational-group row or column blocks."""
    if source.ndim != 3 or int(steps) != 5:
        raise RuntimeError('rational-group polar requires [layers,rows,cols] NS5')
    layers, rows, columns = map(int, source.shape)
    if int(groups) <= 0 or int(width) <= 0 or int(groups) * int(width) <= 0:
        raise ValueError('rational-group polar dimensions must be positive')
    hidden = int(groups) * int(width)
    if rows == hidden:
        blocks = source.reshape(layers, int(groups), int(width), columns).reshape(layers * int(groups), int(width), columns)
        polar = _temporal_geometry__import__literal_zero_power(blocks, steps).reshape(layers, int(groups), int(width), columns).reshape_as(source)
        block_rank = min(int(width), columns)
    elif columns == hidden:
        blocks = source.reshape(layers, rows, int(groups), int(width)).permute(0, 2, 1, 3).reshape(layers * int(groups), rows, int(width))
        polar = _temporal_geometry__import__literal_zero_power(blocks, steps).reshape(layers, int(groups), rows, int(width)).permute(0, 2, 1, 3).reshape_as(source)
        block_rank = min(rows, int(width))
    else:
        raise RuntimeError('rational-group polar cannot identify grouped axis')
    global_rank = min(rows, columns)
    nominal_block_rank = int(groups) * int(block_rank)
    if global_rank <= 0 or nominal_block_rank <= 0:
        raise RuntimeError('rational-group polar rank is empty')
    return polar.mul(math.sqrt(float(global_rank) / float(nominal_block_rank)))

def _temporal_geometry__attention_head_group_zero_power(source: torch.Tensor, steps: int, *, heads: int=_temporal_geometry__ATTENTION_HEADS, head_group_size: int=_temporal_geometry__ATTENTION_HEAD_GROUP_SIZE) -> torch.Tensor:
    """Apply literal NS5 to native Q/K/V or output head groups."""
    if source.ndim != 3 or int(steps) != 5:
        raise RuntimeError('attention head-group polar requires batched NS5')
    layers, rows, columns = map(int, source.shape)
    if columns % int(heads) or int(heads) % int(head_group_size):
        raise RuntimeError('attention head-group inventory is not divisible')
    head_width = columns // int(heads)
    group_count = int(heads) // int(head_group_size)
    block_width = int(head_group_size) * head_width
    if rows == 3 * columns:
        blocks = source.reshape(layers, 3, group_count, block_width, columns).reshape(layers * 3 * group_count, block_width, columns)
        polar = _temporal_geometry__import__literal_zero_power(blocks, steps).reshape(layers, 3, group_count, block_width, columns).reshape_as(source)
        return polar.mul(1.0 / math.sqrt(3.0))
    if rows == columns:
        blocks = source.reshape(layers, rows, group_count, block_width).permute(0, 2, 1, 3).reshape(layers * group_count, rows, block_width)
        return _temporal_geometry__import__literal_zero_power(blocks, steps).reshape(layers, group_count, rows, block_width).permute(0, 2, 1, 3).reshape_as(source)
    raise RuntimeError('attention head-group polar supports only QKV/output roles')

class _temporal_geometry__TemporalResponseGroupPolarRouter(_temporal_geometry__import_LaggedPredictiveResponseTransactionRouter):
    family_id = _temporal_geometry__FAMILY_ID
    telemetry_prefix = 'temporal_response_group_polar_'
    fairness_component = 'temporal_response_group_polar_lr_scale'
    predictive_rows_fn = staticmethod(_temporal_geometry__temporal_response_predictive_rows)
    transaction_fn = staticmethod(_temporal_geometry___EXPECTED_TRANSACTION)

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]['temporal_response_group_polar_family_id'] = _temporal_geometry__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({'rotation_invariant_temporal_response_sketch_lr_scale': 1.0, 'rational_group_polar_lr_scale': 1.0, 'nominal_polar_rank_calibration_lr_scale': 1.0})
        return report

    @torch.no_grad()
    def step(self, closure=None):
        global _predictive_transaction___transaction_from_replicated_global_rows, _predictive_transaction__import__batched_zero_power, _predictive_transaction__matched_beta2_predictive_rows
        with _temporal_geometry___PATCH_LOCK:
            if _predictive_transaction__matched_beta2_predictive_rows is not _temporal_geometry___EXPECTED_PREDICTIVE_ROWS or _predictive_transaction___transaction_from_replicated_global_rows is not _temporal_geometry___EXPECTED_TRANSACTION or _predictive_transaction__import__batched_zero_power is not _temporal_geometry___EXPECTED_ROUTER_POLAR:
                raise RuntimeError('temporal response router binding changed')

            def grouped(source, steps):
                return _temporal_geometry__rational_group_zero_power(source, steps, groups=self.groups, width=self.width)
            row_update = self.predictive_rows_fn
            transaction = self.transaction_fn
            if not callable(row_update) or not callable(transaction):
                raise RuntimeError('temporal response transaction hook changed')
            _predictive_transaction__matched_beta2_predictive_rows = row_update
            _predictive_transaction___transaction_from_replicated_global_rows = transaction
            _predictive_transaction__import__batched_zero_power = grouped
            try:
                loss = super().step(closure)
            finally:
                _predictive_transaction__matched_beta2_predictive_rows = _temporal_geometry___EXPECTED_PREDICTIVE_ROWS
                _predictive_transaction___transaction_from_replicated_global_rows = _temporal_geometry___EXPECTED_TRANSACTION
                _predictive_transaction__import__batched_zero_power = _temporal_geometry___EXPECTED_ROUTER_POLAR
        if self._last_telemetry:
            renamed = {}
            old = 'lagged_predictive_response_transaction_'
            new = 'temporal_response_group_polar_'
            for key, value in self._last_telemetry.items():
                renamed[key.replace(old, new, 1)] = _temporal_geometry__FAMILY_ID if value == _predictive_transaction__FAMILY_ID else value
            self._last_telemetry = renamed
            self._last_telemetry.update({new + 'family_id': _temporal_geometry__FAMILY_ID, new + 'temporal_sketch_rows': _temporal_geometry__import_FIXED_GLOBAL_PROBE_COUNT, new + 'largest_temporal_dense_dimension': 2 * _temporal_geometry__import_FIXED_GLOBAL_PROBE_COUNT, new + 'rational_group_polar': 1, new + 'owner_count': 0, new + 'dense_lg_metric_elements': 0, new + 'selected_update_elements_published': 0})
        return loss

class _temporal_geometry__TemporalResponseHeadGroupPolarAttentionOptimizer(_temporal_geometry__import_LaggedPredictiveResponseTransactionAttentionOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['temporal_response_group_polar_family_id'] = _temporal_geometry__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report['attention_head_group_polar_lr_scale'] = 1.0
        report['attention_nominal_polar_rank_calibration_lr_scale'] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        global _compact_homotopy__import__batched_zero_power
        with _temporal_geometry___PATCH_LOCK:
            if _compact_homotopy__import__batched_zero_power is not _temporal_geometry___EXPECTED_ATTENTION_POLAR:
                raise RuntimeError('temporal response attention binding changed')
            _compact_homotopy__import__batched_zero_power = _temporal_geometry__attention_head_group_zero_power
            try:
                loss = super().step(closure)
            finally:
                _compact_homotopy__import__batched_zero_power = _temporal_geometry___EXPECTED_ATTENTION_POLAR
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value in {_predictive_transaction__FAMILY_ID, _predictive_transaction__import_CURRENT_IMPLEMENTATION_FAMILY_ID}:
                    self._last_telemetry[key] = _temporal_geometry__FAMILY_ID
            self._last_telemetry.update({'temporal_response_group_polar_attention_family_id': _temporal_geometry__FAMILY_ID, 'temporal_response_group_polar_attention_head_count': _temporal_geometry__ATTENTION_HEADS, 'temporal_response_group_polar_attention_head_group_size': _temporal_geometry__ATTENTION_HEAD_GROUP_SIZE, 'temporal_response_group_polar_attention_ns_steps': 5, 'temporal_response_group_polar_attention_owner_count': 0, 'temporal_response_group_polar_attention_selected_update_elements_published': 0})
        return loss
_posterior_geometry__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_posterior_geometry__import_ReplicatedFixedProbeTransactionResult = _fixed_probe_transaction__ReplicatedFixedProbeTransactionResult
_posterior_geometry__import_distributed_fixed_probe_transaction = _fixed_probe_transaction__distributed_fixed_probe_transaction
_posterior_geometry__import_MatchedBeta2PredictiveRows = _predictive_transaction__MatchedBeta2PredictiveRows
_posterior_geometry__import_global_response_transaction_scaling_formula = _response_transaction__global_response_transaction_scaling_formula
_posterior_geometry__import_TemporalResponseGroupPolarRouter = _temporal_geometry__TemporalResponseGroupPolarRouter
_posterior_geometry__import_TemporalResponseHeadGroupPolarAttentionOptimizer = _temporal_geometry__TemporalResponseHeadGroupPolarAttentionOptimizer
_posterior_geometry__FAMILY_ID = 'posterior_rank64_response_group_polar_muon_v1'
_posterior_geometry__PERSISTENT_ROWS = 64
_posterior_geometry__MAXIMUM_AUGMENTED_ROWS = _posterior_geometry__PERSISTENT_ROWS + _posterior_geometry__import_FIXED_GLOBAL_PROBE_COUNT

def _posterior_geometry__posterior_rank64_response_rows(current_scores: torch.Tensor, current_decay_action: torch.Tensor, previous_scores: torch.Tensor | None, previous_decay_action: torch.Tensor | None, *, beta2: float) -> _posterior_geometry__import_MatchedBeta2PredictiveRows:
    """Return the best rank-64 factor of the updated discounted covariance."""
    if current_scores.ndim != 2:
        raise RuntimeError('rank64 posterior current-row inventory changed')
    rows, coordinates = current_scores.shape
    if rows != _posterior_geometry__import_FIXED_GLOBAL_PROBE_COUNT or coordinates < 1 or current_decay_action.shape != (rows,) or (current_decay_action.dtype != current_scores.dtype) or (current_decay_action.device != current_scores.device) or (not current_scores.is_floating_point()) or (float(beta2) != 0.95) or (not bool(torch.isfinite(current_scores).all())) or (not bool(torch.isfinite(current_decay_action).all())):
        raise RuntimeError('rank64 posterior current-row inventory changed')
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError('rank64 posterior histories must coinitialize')
    if previous_scores is None:
        return _posterior_geometry__import_MatchedBeta2PredictiveRows(selection_scores=current_scores, selection_decay_action=current_decay_action, updated_scores=current_scores.detach().clone(), updated_decay_action=current_decay_action.detach().clone(), history_used=False, relative_innovation=current_scores.new_zeros(()))
    assert previous_decay_action is not None
    previous_rows = int(previous_scores.shape[0])
    if previous_scores.ndim != 2 or previous_scores.shape[1] != coordinates or previous_rows not in (_posterior_geometry__import_FIXED_GLOBAL_PROBE_COUNT, _posterior_geometry__PERSISTENT_ROWS) or (previous_decay_action.shape != (previous_rows,)) or (previous_scores.dtype != current_scores.dtype) or (previous_decay_action.dtype != current_scores.dtype) or (previous_scores.device != current_scores.device) or (previous_decay_action.device != current_scores.device) or (not bool(torch.isfinite(previous_scores).all())) or (not bool(torch.isfinite(previous_decay_action).all())):
        raise RuntimeError('rank64 posterior checkpoint inventory changed')
    beta = float(beta2)
    augmented = torch.cat((previous_scores * math.sqrt(beta), current_scores * math.sqrt(1.0 - beta)))
    augmented_decay = torch.cat((previous_decay_action * math.sqrt(beta), current_decay_action * math.sqrt(1.0 - beta)))
    row_gram = augmented @ augmented.T
    row_gram = 0.5 * (row_gram + row_gram.T)
    _eigenvalues, eigenvectors = torch.linalg.eigh(row_gram)
    retained_rows = min(_posterior_geometry__PERSISTENT_ROWS, int(augmented.shape[0]))
    transform = eigenvectors.flip(1)[:, :retained_rows].T
    updated_scores = transform @ augmented
    updated_decay = transform @ augmented_decay
    current_gram = current_scores @ current_scores.T
    previous_gram = previous_scores @ previous_scores.T
    cross_gram = current_scores @ previous_scores.T
    tiny = torch.finfo(current_scores.dtype).tiny
    innovation2 = (current_gram.square().sum() + previous_gram.square().sum() - 2.0 * cross_gram.square().sum()).clamp_min(0.0)
    relative_innovation = torch.sqrt(innovation2 / current_gram.square().sum().clamp_min(tiny))
    torch._assert_async(torch.isfinite(updated_scores).all() & torch.isfinite(updated_decay).all() & torch.isfinite(relative_innovation))
    return _posterior_geometry__import_MatchedBeta2PredictiveRows(selection_scores=updated_scores, selection_decay_action=updated_decay, updated_scores=updated_scores, updated_decay_action=updated_decay, history_used=True, relative_innovation=relative_innovation)

def _posterior_geometry__rank64_transaction_from_replicated_rows(global_scores: torch.Tensor, global_decay_action: torch.Tensor, exact_by_role: torch.Tensor, momentum_by_role: torch.Tensor, weights: torch.Tensor, layer_ids: torch.Tensor, *, total_layers: int, eta: float, gather_rounds: int, group) -> _posterior_geometry__import_ReplicatedFixedProbeTransactionResult:
    """Column-shard a replicated fixed-rank posterior score factor."""
    rows, coordinates = global_scores.shape
    if rows not in (_posterior_geometry__import_FIXED_GLOBAL_PROBE_COUNT, _posterior_geometry__PERSISTENT_ROWS, _posterior_geometry__MAXIMUM_AUGMENTED_ROWS) or global_decay_action.shape != (rows,) or exact_by_role.shape != (2, coordinates) or (momentum_by_role.shape != exact_by_role.shape) or (weights.shape != (coordinates,)) or (layer_ids.shape != (coordinates,)):
        raise RuntimeError('rank64 replicated transaction inventory changed')
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
    else:
        rank, world = (0, 1)
    coordinate_ids = torch.arange(coordinates, device=global_scores.device, dtype=torch.int64)
    local_ids = coordinate_ids[coordinate_ids.remainder(world).eq(rank)]
    measure_scale = math.sqrt(float(rows) / float(_posterior_geometry__import_FIXED_GLOBAL_PROBE_COUNT))
    transaction_scores = global_scores * measure_scale
    transaction_decay = global_decay_action * measure_scale
    decay_cross = transaction_scores.T @ transaction_decay / float(rows)
    sharded = _posterior_geometry__import_distributed_fixed_probe_transaction(transaction_scores[:, local_ids], exact_by_role[:, local_ids], momentum_by_role[:, local_ids], decay_cross[local_ids], weights[local_ids], layer_ids[local_ids], local_ids, total_coordinates=coordinates, total_layers=int(total_layers), eta=float(eta), rounds=64, group=group)
    coefficient_packet = torch.zeros(2 * coordinates, device=global_scores.device, dtype=global_scores.dtype)
    coefficient_packet[local_ids] = sharded.local_coefficients
    coefficient_packet[coordinates + local_ids] = sharded.local_candidate_coefficients
    if world > 1:
        dist.all_reduce(coefficient_packet, op=dist.ReduceOp.SUM, group=group)
    total_row_metric = global_scores @ global_scores.T
    total_square = total_row_metric.square().sum()
    within_square = torch.zeros_like(total_square)
    for layer in range(int(total_layers)):
        layer_scores = global_scores[:, layer_ids.eq(layer)]
        within_square.add_((layer_scores @ layer_scores.T).square().sum())
    coupling = torch.sqrt((total_square - within_square).clamp_min(0.0) / total_square.clamp_min(torch.finfo(total_square.dtype).tiny))
    return _posterior_geometry__import_ReplicatedFixedProbeTransactionResult(coefficients=coefficient_packet[:coordinates], candidate_coefficients=coefficient_packet[coordinates:], sharded_result=sharded, local_probe_count=rows // max(world, 1), global_probe_count=rows, cross_layer_coupling_ratio=coupling, collective_rounds=int(gather_rounds) + sharded.collective_rounds + int(world > 1), score_scalars_exchanged_per_rank=rows * (coordinates + 1), coefficient_scalars_exchanged_per_rank=2 * coordinates if world > 1 else 0, selected_update_elements_published=0, method_state_depends_on_total_tokens=False)

def _posterior_geometry__posterior_rank64_scaling_formula(*, total_positions: int, total_layers: int, total_groups: int, intermediate_width: int, model_width: int) -> dict[str, int]:
    values = tuple(map(int, (total_positions, total_layers, total_groups, intermediate_width, model_width)))
    if min(values) <= 0 or int(intermediate_width) % int(total_groups):
        raise ValueError('rank64 scaling dimensions are invalid')
    positions, layers, groups, hidden, model = values
    coordinates = layers * groups
    parent = _posterior_geometry__import_global_response_transaction_scaling_formula(total_positions=positions, total_layers=layers, total_groups=groups, intermediate_width=hidden, model_width=model)
    response_summary = 21 * coordinates + 10 * layers
    persistent = _posterior_geometry__PERSISTENT_ROWS * (coordinates + 1)
    transaction_summary = _posterior_geometry__PERSISTENT_ROWS * _posterior_geometry__PERSISTENT_ROWS + 3 * _posterior_geometry__PERSISTENT_ROWS + 8 * layers + 8
    return {'total_positions': positions, 'persistent_state_elements': parent['persistent_state_elements'] + persistent, 'posterior_factor_elements': persistent, 'communicated_summary_elements': response_summary + _posterior_geometry__import_FIXED_GLOBAL_PROBE_COUNT * (coordinates + 1) + transaction_summary + 2 * coordinates, 'largest_temporal_dense_dimension': _posterior_geometry__MAXIMUM_AUGMENTED_ROWS, 'largest_dense_solve_dimension': _posterior_geometry__PERSISTENT_ROWS, 'dense_coordinate_metric_elements': 0, 'owner_count': 0, 'selected_update_elements_published': 0, 'local_direction_arithmetic_elements': 4 * layers * hidden * model}

class _posterior_geometry__PosteriorRank64ResponseGroupPolarRouter(_posterior_geometry__import_TemporalResponseGroupPolarRouter):
    family_id = _posterior_geometry__FAMILY_ID
    telemetry_prefix = 'posterior_rank64_response_group_polar_'
    fairness_component = 'posterior_rank64_response_group_polar_lr_scale'
    predictive_rows_fn = staticmethod(_posterior_geometry__posterior_rank64_response_rows)
    transaction_fn = staticmethod(_posterior_geometry__rank64_transaction_from_replicated_rows)

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]['posterior_rank64_response_group_polar_family_id'] = _posterior_geometry__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({'posterior_rank64_subspace_lr_scale': 1.0, 'truncated_svd_response_metric_lr_scale': 1.0})
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            old = 'temporal_response_group_polar_'
            new = 'posterior_rank64_response_group_polar_'
            renamed = {key.replace(old, new, 1): _posterior_geometry__FAMILY_ID if value == _posterior_geometry__import_TemporalResponseGroupPolarRouter.family_id else value for key, value in self._last_telemetry.items()}
            scaling = _posterior_geometry__posterior_rank64_scaling_formula(total_positions=1, total_layers=len(self.pairs), total_groups=self.groups, intermediate_width=self.hidden, model_width=self.external)
            factor = self.state[self.pairs[0]['in_weight']].get('predictive_global_score_ema')
            renamed.update({new + 'family_id': _posterior_geometry__FAMILY_ID, new + 'selection_uses_updated_metric': 1, new + 'persistent_rank_limit': _posterior_geometry__PERSISTENT_ROWS, new + 'realized_factor_rows': int(factor.shape[0]), new + 'state_coordinate_count': scaling['persistent_state_elements'], new + 'predictive_state_elements': scaling['posterior_factor_elements'], new + 'summary_elements': scaling['communicated_summary_elements'], new + 'largest_dense_solve_dimension': _posterior_geometry__PERSISTENT_ROWS, new + 'largest_temporal_dense_dimension': _posterior_geometry__MAXIMUM_AUGMENTED_ROWS})
            self._last_telemetry = renamed
        return loss

class _posterior_geometry__PosteriorRank64HeadGroupPolarAttentionOptimizer(_posterior_geometry__import_TemporalResponseHeadGroupPolarAttentionOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]['posterior_rank64_response_group_polar_family_id'] = _posterior_geometry__FAMILY_ID

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report['posterior_rank64_attention_lr_scale'] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value == _posterior_geometry__import_TemporalResponseGroupPolarRouter.family_id:
                    self._last_telemetry[key] = _posterior_geometry__FAMILY_ID
            self._last_telemetry.update({'posterior_rank64_response_group_polar_attention_family_id': _posterior_geometry__FAMILY_ID, 'posterior_rank64_response_group_polar_attention_owner_count': 0, 'posterior_rank64_response_group_polar_attention_selected_update_elements_published': 0})
        return loss
_cadence_geometry__import__match_rms_adamw_adjustment = _basis_trust___match_rms_adamw_adjustment
_cadence_geometry__import_compact_postpolar_group_response_homotopy = _compact_homotopy__compact_postpolar_group_response_homotopy
_cadence_geometry__import_MatchedBeta2PredictiveRows = _predictive_transaction__MatchedBeta2PredictiveRows
_cadence_geometry__import__foreach_apply = _predictive_transaction__import__foreach_apply
_cadence_geometry__import__foreach_nesterov = _predictive_transaction__import__foreach_nesterov
_cadence_geometry__import_POSTERIOR_PARENT_FAMILY_ID = _posterior_geometry__FAMILY_ID
_cadence_geometry__import_PERSISTENT_ROWS = _posterior_geometry__PERSISTENT_ROWS
_cadence_geometry__import_PosteriorRank64HeadGroupPolarAttentionOptimizer = _posterior_geometry__PosteriorRank64HeadGroupPolarAttentionOptimizer
_cadence_geometry__import_PosteriorRank64ResponseGroupPolarRouter = _posterior_geometry__PosteriorRank64ResponseGroupPolarRouter
_cadence_geometry__import_posterior_rank64_response_rows = _posterior_geometry__posterior_rank64_response_rows
_cadence_geometry__import_rank64_transaction_from_replicated_rows = _posterior_geometry__rank64_transaction_from_replicated_rows
_cadence_geometry__import_rational_group_zero_power = _temporal_geometry__rational_group_zero_power
_cadence_geometry__REFRESH_INTERVAL = 8
_cadence_geometry__MATCHED_BETA2 = 0.95
_cadence_geometry__EFFECTIVE_REFRESH_BETA2 = _cadence_geometry__MATCHED_BETA2 ** _cadence_geometry__REFRESH_INTERVAL
_cadence_geometry__POSTERIOR_FAMILY_ID = 'posterior_rank64_cadence8_group_polar_muon_v1'

def _cadence_geometry___elapsed_posterior_rows(current_scores: torch.Tensor, current_decay_action: torch.Tensor, previous_scores: torch.Tensor | None, previous_decay_action: torch.Tensor | None, *, beta2: float) -> _cadence_geometry__import_MatchedBeta2PredictiveRows:
    """Apply exactly beta2**8 between observed response geometries."""
    if float(beta2) != _cadence_geometry__MATCHED_BETA2:
        raise RuntimeError('cadence8 response metric requires locked beta2=.95')
    if previous_scores is None:
        return _cadence_geometry__import_posterior_rank64_response_rows(current_scores, current_decay_action, None, None, beta2=beta2)
    if previous_decay_action is None:
        raise RuntimeError('cadence8 response histories must coinitialize')
    previous_scale = math.sqrt(_cadence_geometry__EFFECTIVE_REFRESH_BETA2 / _cadence_geometry__MATCHED_BETA2)
    current_scale = math.sqrt((1.0 - _cadence_geometry__EFFECTIVE_REFRESH_BETA2) / (1.0 - _cadence_geometry__MATCHED_BETA2))
    return _cadence_geometry__import_posterior_rank64_response_rows(current_scores * current_scale, current_decay_action * current_scale, previous_scores * previous_scale, previous_decay_action * previous_scale, beta2=beta2)

def _cadence_geometry__periodic_posterior_rank64_rows(*args, **kwargs) -> _cadence_geometry__import_MatchedBeta2PredictiveRows:
    return _cadence_geometry___elapsed_posterior_rows(*args, **kwargs)

def _cadence_geometry__cadence8_scaling_formula(*, total_positions: int, total_layers: int, total_groups: int, intermediate_width: int, model_width: int, consensus: bool) -> dict[str, int | float]:
    values = tuple(map(int, (total_positions, total_layers, total_groups, intermediate_width, model_width)))
    if min(values) <= 0 or int(intermediate_width) % int(total_groups):
        raise ValueError('cadence8 scaling dimensions are invalid')
    positions, layers, groups, hidden, model = values
    coordinates = layers * groups
    posterior_factor = _cadence_geometry__import_PERSISTENT_ROWS * (coordinates + 1)
    parent_state = 10 * coordinates + 2
    route_state = 4 * coordinates + 2 * layers
    selection_state = 96 * (coordinates + 1) if consensus else 0
    largest = 96 if consensus else 64
    response_summary = 21 * coordinates + 10 * layers
    transaction_summary = largest * largest + 3 * largest + 8 * layers + 8
    refresh_score_summary = 32 * (coordinates + 1)
    return {'total_positions': positions, 'persistent_state_elements': parent_state + posterior_factor + route_state + selection_state, 'posterior_factor_elements': posterior_factor, 'cached_response_route_elements': route_state, 'cached_selection_factor_elements': selection_state, 'communicated_summary_elements': response_summary + refresh_score_summary + transaction_summary + 2 * coordinates, 'ordinary_communicated_summary_elements': transaction_summary + 2 * coordinates, 'largest_temporal_dense_dimension': 96, 'largest_dense_solve_dimension': largest, 'dense_coordinate_metric_elements': 0, 'owner_count': 0, 'selected_update_elements_published': 0, 'response_refresh_interval': _cadence_geometry__REFRESH_INTERVAL, 'matched_beta2': _cadence_geometry__MATCHED_BETA2, 'effective_refresh_beta2': _cadence_geometry__EFFECTIVE_REFRESH_BETA2, 'local_direction_arithmetic_elements': 4 * layers * hidden * model}

class _cadence_geometry___Cadence8Rank64RouterMixin:
    metric_rows_fn = None
    base_family_id = ''
    family_id = ''
    base_prefix = ''
    telemetry_prefix = ''
    cache_selection_separately = False

    def __init__(self, pairs, **kwargs):
        self._cadence_transition = 0
        self._capture_response_this_transition = True
        self._cadence_predictive = None
        self._cadence_transaction = None
        self._cached_route = None
        self._cached_selection_scores = None
        self._cached_selection_decay = None
        super().__init__(pairs, **kwargs)
        self.param_groups[0][self.telemetry_prefix + 'family_id'] = self.family_id
        self.param_groups[0][self.telemetry_prefix + 'refresh_interval'] = _cadence_geometry__REFRESH_INTERVAL

    def _make_input_hook(self, layer):
        parent = super()._make_input_hook(layer)

        @torch.no_grad()
        def capture(module, inputs):
            if self._capture_response_this_transition:
                return parent(module, inputs)
            return None
        return capture

    def _make_feature_hook(self, layer):
        parent = super()._make_feature_hook(layer)

        @torch.no_grad()
        def capture(module, inputs, output):
            if self._capture_response_this_transition:
                return parent(module, inputs, output)
            return None
        return capture

    def _make_cotangent_hook(self, layer):
        parent = super()._make_cotangent_hook(layer)

        def capture(module, inputs, output):
            if self._capture_response_this_transition:
                return parent(module, inputs, output)
            return None
        return capture

    def predictive_rows_fn(self, *args, **kwargs):
        if not callable(self.metric_rows_fn):
            raise RuntimeError('cadence8 metric row function is missing')
        result = self.metric_rows_fn(*args, **kwargs)
        self._cadence_predictive = result
        return result

    def transaction_fn(self, *args, **kwargs):
        result = _cadence_geometry__import_rank64_transaction_from_replicated_rows(*args, **kwargs)
        self._cadence_transaction = result
        return result

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({'elapsed_beta2_cadence8_response_metric_lr_scale': 1.0, 'current_linear_transaction_every_step_lr_scale': 1.0, 'cached_scalar_response_route_lr_scale': 1.0})
        return result

    def _cache_refresh_state(self):
        predictive = self._cadence_predictive
        transaction = self._cadence_transaction
        if predictive is None or transaction is None:
            raise RuntimeError('cadence8 refresh omitted metric or transaction')
        route = {'group_participation': self._group_participation.detach().clone(), 'group_congruence': self._group_response_congruence.detach().clone(), 'intrinsic_participation': self._intrinsic_participation.detach().clone(), 'attention_congruence': self._attention_response_congruence.detach().clone()}
        self._cached_route = route
        anchor = self.state[self.pairs[0]['in_weight']]
        for name, value in route.items():
            anchor['cadence8_' + name] = value
        anchor['cadence8_relative_score_innovation'] = predictive.relative_innovation.detach().clone()
        if self.cache_selection_separately:
            scores = predictive.selection_scores.detach().clone()
            decay = predictive.selection_decay_action.detach().clone()
            anchor['cadence8_selection_scores'] = scores
            anchor['cadence8_selection_decay'] = decay
            self._cached_selection_scores = scores
            self._cached_selection_decay = decay
        else:
            self._cached_selection_scores = anchor['predictive_global_score_ema']
            self._cached_selection_decay = anchor['predictive_global_decay_ema']
        anchor['cadence8_transition'] = int(self._cadence_transition)

    def _restore_route(self):
        route = self._cached_route
        if route is None or set(route) != {'group_participation', 'group_congruence', 'intrinsic_participation', 'attention_congruence'}:
            raise RuntimeError('cadence8 cached response route is incomplete')
        self._group_participation = route['group_participation']
        self._group_response_congruence = route['group_congruence']
        self._intrinsic_participation = route['intrinsic_participation']
        self._attention_response_congruence = route['attention_congruence']

    def _advance_cadence(self, *, refreshed: bool, publish: bool):
        self._cadence_transition += 1
        self.state[self.pairs[0]['in_weight']]['cadence8_transition'] = self._cadence_transition
        self._capture_response_this_transition = self._cadence_transition % _cadence_geometry__REFRESH_INTERVAL == 0
        if publish:
            prefix = self.telemetry_prefix
            self._last_telemetry.update({prefix + 'response_refresh_interval': _cadence_geometry__REFRESH_INTERVAL, prefix + 'response_refreshed': int(refreshed), prefix + 'response_age': 0 if refreshed else (self._cadence_transition - 1) % _cadence_geometry__REFRESH_INTERVAL, prefix + 'effective_refresh_beta2': _cadence_geometry__EFFECTIVE_REFRESH_BETA2, prefix + 'current_gradient_transaction': 1, prefix + 'cached_parameter_update_elements': 0})

    def _rename_refresh_telemetry(self):
        old = self.base_prefix
        new = self.telemetry_prefix
        self._last_telemetry = {key.replace(old, new, 1): self.family_id if value == self.base_family_id else value for key, value in self._last_telemetry.items()}
        scaling = _cadence_geometry__cadence8_scaling_formula(total_positions=1, total_layers=len(self.pairs), total_groups=self.groups, intermediate_width=self.hidden, model_width=self.external, consensus=self.cache_selection_separately)
        self._last_telemetry.update({new + 'family_id': self.family_id, new + 'state_coordinate_count': scaling['persistent_state_elements'], new + 'cached_response_route_elements': scaling['cached_response_route_elements'], new + 'cached_selection_factor_elements': scaling['cached_selection_factor_elements']})

    @torch.no_grad()
    def _ordinary_step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError('cadence8 router lacks realized clipping')
        if not self._attention_consumed:
            raise RuntimeError('cadence8 router would overwrite attention state')
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get('lr_scale', 1.0)) != 1.0:
            raise RuntimeError('cadence8 router refuses nonunit LR scale')
        lr = float(group['lr'])
        weight_decay = float(group['weight_decay'])
        self._restore_route()
        participation = self._group_participation
        congruence = self._group_response_congruence
        role_parameters = {}
        role_selected = {}
        role_scaled = {}
        role_adjustment = {}
        role_records = {}
        exact = []
        momentum_descent = []
        for role, index, axis in (('incoming', 0, 'rows'), ('outgoing', 1, 'columns')):
            key = 'in_weight' if role == 'incoming' else 'out_weight'
            parameters = [pair[key] for pair in self.pairs]
            momenta = _cadence_geometry__import__foreach_nesterov(self, parameters)
            gradients = torch.stack([parameter.grad.detach() for parameter in parameters]).float()
            parent = _cadence_geometry__import_rational_group_zero_power(momenta, self.ns_steps, groups=self.groups, width=self.width).float()
            selected, metadata = _cadence_geometry__import_compact_postpolar_group_response_homotopy(parent, momenta, gradients, participation[..., index], congruence[..., index], groups=self.groups, width=self.width, grouped_axis=axis)
            adjustment = _cadence_geometry__import__match_rms_adamw_adjustment(parameters[0].shape)
            if role == 'incoming':
                blocks = selected.view(len(self.pairs), self.groups, self.width, self.external)
                gradient_blocks = gradients.view_as(blocks)
                momentum_blocks = momenta.view_as(blocks)
            else:
                blocks = selected.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
                gradient_blocks = gradients.view_as(selected).view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
                momentum_blocks = momenta.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1)
            scaled = blocks.float() * adjustment
            exact.append((gradient_blocks * scaled).sum(dim=(-2, -1)))
            momentum_descent.append((momentum_blocks.float() * scaled).sum(dim=(-2, -1)))
            role_parameters[role] = parameters
            role_selected[role] = selected
            role_scaled[role] = scaled
            role_adjustment[role] = adjustment
            role_records[role] = metadata
        incoming = role_scaled['incoming']
        outgoing = role_scaled['outgoing']
        weights = (incoming.square().sum(dim=(-2, -1)) + outgoing.square().sum(dim=(-2, -1))).reshape(-1)
        exact_by_role = torch.stack([value.reshape(-1) for value in exact])
        momentum_by_role = torch.stack([value.reshape(-1) for value in momentum_descent])
        layer_ids = torch.arange(len(self.pairs), device=weights.device, dtype=torch.int64).repeat_interleave(self.groups)
        if self._cached_selection_scores is None or self._cached_selection_decay is None:
            raise RuntimeError('cadence8 selection factor is missing')
        selection = _cadence_geometry__import_rank64_transaction_from_replicated_rows(self._cached_selection_scores, self._cached_selection_decay, exact_by_role, momentum_by_role, weights, layer_ids, total_layers=len(self.pairs), eta=lr, gather_rounds=0, group=self.loss_probe_group)
        self._cadence_transaction = selection
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)
        incoming_selected = role_selected['incoming']
        incoming_selected.view(len(self.pairs), self.groups, self.width, self.external).mul_(coefficients[..., None, None].to(incoming_selected.dtype))
        outgoing_selected = role_selected['outgoing']
        outgoing_selected.view(len(self.pairs), self.external, self.groups, self.width).permute(0, 2, 3, 1).mul_(coefficients[..., None, None].to(outgoing_selected.dtype))
        for role in ('incoming', 'outgoing'):
            _cadence_geometry__import__foreach_apply(role_parameters[role], role_selected[role], decay=1.0 - lr * weight_decay, alpha=-lr * role_adjustment[role])
        anchor = self.state[self.pairs[0]['in_weight']]
        updates = int(anchor.get('predictive_response_transaction_updates', 0)) + 1
        anchor['predictive_response_transaction_updates'] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            flat = coefficients.reshape(-1)
            response_cosine = torch.cat([role_records[role]['parent_cosine'].reshape(-1) for role in ('incoming', 'outgoing')])
            response_safe = torch.cat([role_records[role]['safe'].reshape(-1) for role in ('incoming', 'outgoing')])
            transaction = selection.sharded_result
            prefix = self.telemetry_prefix
            self._last_telemetry.update({prefix + 'family_id': self.family_id, prefix + 'transaction_accepted': int(transaction.accepted.item()), prefix + 'rank': int(transaction.rank.item()), prefix + 'budget_residual': float(transaction.budget_residual.item()), prefix + 'coefficient_min': float(flat.amin().item()), prefix + 'coefficient_median': float(flat.median().item()), prefix + 'coefficient_max': float(flat.amax().item()), prefix + 'cross_layer_coupling_ratio': float(selection.cross_layer_coupling_ratio.item()), prefix + 'response_parent_cosine_median': float(response_cosine.median().item()), prefix + 'response_safe_fraction': float(response_safe.float().mean().item()), prefix + 'realized_clip_factor': float(self._clip_factor), prefix + 'history_used': int(self._cadence_transition >= _cadence_geometry__REFRESH_INTERVAL), prefix + 'realized_factor_rows': int(anchor['predictive_global_score_ema'].shape[0])})
            # Older checkpoints predate this diagnostic scalar. Its original
            # value cannot be recovered from the current factor alone.
            innovation = anchor.get('cadence8_relative_score_innovation')
            if innovation is not None:
                self._last_telemetry[prefix + 'relative_score_innovation'] = float(innovation.item())
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        refreshed = bool(self._capture_response_this_transition)
        publish = bool(self._capture_telemetry_next_step)
        if refreshed:
            self._cadence_predictive = None
            self._cadence_transaction = None
            loss = super().step(closure)
            self._cache_refresh_state()
            if publish:
                self._rename_refresh_telemetry()
        else:
            loss = self._ordinary_step(closure)
        self._advance_cadence(refreshed=refreshed, publish=publish)
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        anchor = self.state[self.pairs[0]['in_weight']]
        step = anchor.get('cadence8_transition')
        if not isinstance(step, int) or step < 0:
            raise RuntimeError('cadence8 checkpoint transition changed')
        self._cadence_transition = step
        self._capture_response_this_transition = step % _cadence_geometry__REFRESH_INTERVAL == 0
        names = ('group_participation', 'group_congruence', 'intrinsic_participation', 'attention_congruence')
        route = {name: anchor.get('cadence8_' + name) for name in names}
        if any((not torch.is_tensor(value) for value in route.values())):
            raise RuntimeError('cadence8 checkpoint route changed')
        self._cached_route = route
        if self.cache_selection_separately:
            self._cached_selection_scores = anchor.get('cadence8_selection_scores')
            self._cached_selection_decay = anchor.get('cadence8_selection_decay')
        else:
            self._cached_selection_scores = anchor.get('predictive_global_score_ema')
            self._cached_selection_decay = anchor.get('predictive_global_decay_ema')
        if not torch.is_tensor(self._cached_selection_scores) or not torch.is_tensor(self._cached_selection_decay):
            raise RuntimeError('cadence8 checkpoint selection factor changed')
        innovation = anchor.get('cadence8_relative_score_innovation')
        if innovation is not None and (not torch.is_tensor(innovation) or innovation.numel() != 1 or not innovation.is_floating_point() or not bool(torch.isfinite(innovation).all())):
            raise RuntimeError('cadence8 checkpoint innovation diagnostic changed')
        self._cadence_predictive = None
        return result

class _cadence_geometry__PosteriorRank64Cadence8GroupPolarRouter(_cadence_geometry___Cadence8Rank64RouterMixin, _cadence_geometry__import_PosteriorRank64ResponseGroupPolarRouter):
    metric_rows_fn = staticmethod(_cadence_geometry__periodic_posterior_rank64_rows)
    base_family_id = _cadence_geometry__import_POSTERIOR_PARENT_FAMILY_ID
    family_id = _cadence_geometry__POSTERIOR_FAMILY_ID
    base_prefix = 'posterior_rank64_response_group_polar_'
    telemetry_prefix = 'posterior_rank64_cadence8_group_polar_'
    fairness_component = 'posterior_rank64_cadence8_group_polar_lr_scale'
    cache_selection_separately = False

class _cadence_geometry__PosteriorRank64Cadence8HeadPolarAttentionOptimizer(_cadence_geometry__import_PosteriorRank64HeadGroupPolarAttentionOptimizer):

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result['posterior_rank64_cadence8_attention_lr_scale'] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value == _cadence_geometry__import_POSTERIOR_PARENT_FAMILY_ID:
                    self._last_telemetry[key] = _cadence_geometry__POSTERIOR_FAMILY_ID
            self._last_telemetry.update({'posterior_rank64_cadence8_group_polar_attention_family_id': _cadence_geometry__POSTERIOR_FAMILY_ID, 'posterior_rank64_cadence8_group_polar_attention_owner_count': 0, 'posterior_rank64_cadence8_group_polar_attention_selected_update_elements_published': 0})
        return loss
_window_geometry__import_FIXED_GLOBAL_PROBE_COUNT = _functional_row__FIXED_GLOBAL_PROBE_COUNT
_window_geometry__import_MatchedBeta2PredictiveRows = _predictive_transaction__MatchedBeta2PredictiveRows
_window_geometry__import_MATCHED_BETA2 = _cadence_geometry__MATCHED_BETA2
_window_geometry__import_REFRESH_INTERVAL = _cadence_geometry__REFRESH_INTERVAL
_window_geometry__import_PARENT_FAMILY_ID = _cadence_geometry__POSTERIOR_FAMILY_ID
_window_geometry__import_PosteriorRank64Cadence8GroupPolarRouter = _cadence_geometry__PosteriorRank64Cadence8GroupPolarRouter
_window_geometry__import_PosteriorRank64Cadence8HeadPolarAttentionOptimizer = _cadence_geometry__PosteriorRank64Cadence8HeadPolarAttentionOptimizer
_window_geometry__FAMILY_ID = 'window32_cadence8_response_group_polar_muon_v1'
_window_geometry__PREFIX = 'window32_cadence8_response_group_polar_'
_window_geometry__WINDOW_ROWS = _window_geometry__import_FIXED_GLOBAL_PROBE_COUNT

def _window_geometry__current_window32_rows(current_scores: torch.Tensor, current_decay_action: torch.Tensor, previous_scores: torch.Tensor | None, previous_decay_action: torch.Tensor | None, *, beta2: float) -> _window_geometry__import_MatchedBeta2PredictiveRows:
    """Use the latest response covariance without temporal mixing."""
    if current_scores.ndim != 2 or current_scores.shape[0] != _window_geometry__WINDOW_ROWS or current_scores.shape[1] < 1 or (current_decay_action.shape != (_window_geometry__WINDOW_ROWS,)) or (current_decay_action.dtype != current_scores.dtype) or (current_decay_action.device != current_scores.device) or (not current_scores.is_floating_point()) or (float(beta2) != _window_geometry__import_MATCHED_BETA2) or (not bool(torch.isfinite(current_scores).all())) or (not bool(torch.isfinite(current_decay_action).all())):
        raise RuntimeError('window32 current response inventory changed')
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError('window32 histories must coinitialize')
    updated_scores = current_scores.detach().clone()
    updated_decay = current_decay_action.detach().clone()
    if previous_scores is None:
        innovation = current_scores.new_zeros(())
        history_used = False
    else:
        assert previous_decay_action is not None
        if previous_scores.shape != current_scores.shape or previous_decay_action.shape != current_decay_action.shape or previous_scores.dtype != current_scores.dtype or (previous_decay_action.dtype != current_scores.dtype) or (previous_scores.device != current_scores.device) or (previous_decay_action.device != current_scores.device) or (not bool(torch.isfinite(previous_scores).all())) or (not bool(torch.isfinite(previous_decay_action).all())):
            raise RuntimeError('window32 checkpoint inventory changed')
        current_gram = current_scores @ current_scores.T
        previous_gram = previous_scores @ previous_scores.T
        cross_gram = current_scores @ previous_scores.T
        tiny = torch.finfo(current_scores.dtype).tiny
        innovation = torch.sqrt((current_gram.square().sum() + previous_gram.square().sum() - 2.0 * cross_gram.square().sum()).clamp_min(0.0) / current_gram.square().sum().clamp_min(tiny))
        history_used = True
    torch._assert_async(torch.isfinite(innovation))
    return _window_geometry__import_MatchedBeta2PredictiveRows(selection_scores=updated_scores, selection_decay_action=updated_decay, updated_scores=updated_scores, updated_decay_action=updated_decay, history_used=history_used, relative_innovation=innovation)

def _window_geometry__window32_cadence8_scaling_formula(*, total_positions: int, total_layers: int, total_groups: int, intermediate_width: int, model_width: int) -> dict[str, int | float]:
    values = tuple(map(int, (total_positions, total_layers, total_groups, intermediate_width, model_width)))
    if min(values) <= 0 or int(intermediate_width) % int(total_groups):
        raise ValueError('window32 scaling dimensions are invalid')
    positions, layers, groups, hidden, model = values
    coordinates = layers * groups
    parent_state = 10 * coordinates + 2
    factor_state = _window_geometry__WINDOW_ROWS * (coordinates + 1)
    route_state = 4 * coordinates + 2 * layers
    response_summary = 21 * coordinates + 10 * layers
    transaction_summary = _window_geometry__WINDOW_ROWS * _window_geometry__WINDOW_ROWS + 3 * _window_geometry__WINDOW_ROWS + 8 * layers + 8
    refresh_score_summary = _window_geometry__WINDOW_ROWS * (coordinates + 1)
    return {'total_positions': positions, 'persistent_state_elements': parent_state + factor_state + route_state, 'window_factor_elements': factor_state, 'cached_response_route_elements': route_state, 'communicated_summary_elements': response_summary + refresh_score_summary + transaction_summary + 2 * coordinates, 'ordinary_communicated_summary_elements': transaction_summary + 2 * coordinates, 'largest_temporal_dense_dimension': _window_geometry__WINDOW_ROWS, 'largest_dense_solve_dimension': _window_geometry__WINDOW_ROWS, 'dense_coordinate_metric_elements': 0, 'owner_count': 0, 'selected_update_elements_published': 0, 'response_refresh_interval': _window_geometry__import_REFRESH_INTERVAL, 'matched_beta2': _window_geometry__import_MATCHED_BETA2, 'local_direction_arithmetic_elements': 4 * layers * hidden * model}

class _window_geometry__Window32Cadence8ResponseGroupPolarRouter(_window_geometry__import_PosteriorRank64Cadence8GroupPolarRouter):
    metric_rows_fn = staticmethod(_window_geometry__current_window32_rows)
    family_id = _window_geometry__FAMILY_ID
    telemetry_prefix = _window_geometry__PREFIX
    fairness_component = 'window32_cadence8_response_group_polar_lr_scale'

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({'current_window32_response_metric_lr_scale': 1.0, 'no_temporal_covariance_smearing_lr_scale': 1.0})
        return result

    def _rename_refresh_telemetry(self):
        super()._rename_refresh_telemetry()
        scaling = _window_geometry__window32_cadence8_scaling_formula(total_positions=1, total_layers=len(self.pairs), total_groups=self.groups, intermediate_width=self.hidden, model_width=self.external)
        prefix = self.telemetry_prefix
        self._last_telemetry.update({prefix + 'family_id': _window_geometry__FAMILY_ID, prefix + 'state_coordinate_count': scaling['persistent_state_elements'], prefix + 'predictive_state_elements': scaling['window_factor_elements'], prefix + 'summary_elements': scaling['communicated_summary_elements'], prefix + 'largest_dense_solve_dimension': _window_geometry__WINDOW_ROWS, prefix + 'largest_temporal_dense_dimension': _window_geometry__WINDOW_ROWS, prefix + 'persistent_rank_limit': _window_geometry__WINDOW_ROWS, prefix + 'realized_factor_rows': _window_geometry__WINDOW_ROWS, prefix + 'selection_uses_current_window': 1, prefix + 'temporal_covariance_mixing': 0})

class _window_geometry__Window32Cadence8HeadPolarAttentionOptimizer(_window_geometry__import_PosteriorRank64Cadence8HeadPolarAttentionOptimizer):

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result['window32_cadence8_attention_lr_scale'] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        old = 'posterior_rank64_cadence8_group_polar_'
        self._last_telemetry = {key.replace(old, _window_geometry__PREFIX, 1): _window_geometry__FAMILY_ID if value == _window_geometry__import_PARENT_FAMILY_ID else value for key, value in self._last_telemetry.items()}
        if self._last_telemetry:
            self._last_telemetry.update({_window_geometry__PREFIX + 'attention_family_id': _window_geometry__FAMILY_ID, _window_geometry__PREFIX + 'attention_owner_count': 0, _window_geometry__PREFIX + 'attention_selected_update_elements_published': 0})
        return loss
_factorized_chord__import_WINDOW32_FAMILY_ID = _window_geometry__FAMILY_ID
_factorized_chord__import_WINDOW32_PREFIX = _window_geometry__PREFIX
_factorized_chord__import_Window32Cadence8HeadPolarAttentionOptimizer = _window_geometry__Window32Cadence8HeadPolarAttentionOptimizer
_factorized_chord__import_Window32Cadence8ResponseGroupPolarRouter = _window_geometry__Window32Cadence8ResponseGroupPolarRouter
_factorized_chord__import_window32_cadence8_scaling_formula = _window_geometry__window32_cadence8_scaling_formula
_factorized_chord__FAMILY_ID = 'factorized_adaptive_tangent_chord_muon_v1'
_factorized_chord__PREFIX = 'factorized_adaptive_tangent_chord_'
_factorized_chord__LOCKED_EPS = 1e-08
_factorized_chord__MATCHED_BETA2 = 0.95
_factorized_chord___PATCH_LOCK = threading.RLock()
_factorized_chord___EXPECTED_REFRESH_DIRECTION = _predictive_transaction__import_compact_postpolar_group_response_homotopy
_factorized_chord___EXPECTED_ORDINARY_DIRECTION = _cadence_geometry__import_compact_postpolar_group_response_homotopy
_factorized_chord___EXPECTED_ATTENTION_DIRECTION = _compact_homotopy__compact_postpolar_group_response_homotopy

def _factorized_chord___group_view(value: torch.Tensor, *, groups: int, width: int | None, grouped_axis: str):
    """Return grouped matrices and the exact inverse view operation."""
    if value.ndim != 3:
        raise RuntimeError('TILLER requires batched matrices')
    layers, rows, columns = map(int, value.shape)
    if grouped_axis == 'rows':
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError('TILLER row-group inventory changed')
        return (value.float().view(layers, int(groups), int(width), columns), lambda selected: selected.reshape_as(value))
    if grouped_axis == 'columns':
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError('TILLER column-group inventory changed')
        grouped = value.float().transpose(-2, -1).contiguous().view(layers, int(groups), int(width), rows)
        return (grouped, lambda selected: selected.reshape(layers, columns, rows).transpose(-2, -1).contiguous())
    if grouped_axis == 'matrix':
        if int(groups) != 1 or width is not None:
            raise RuntimeError('TILLER matrix inventory changed')
        return (value.float()[:, None], lambda selected: selected[:, 0])
    raise ValueError(f'unknown TILLER grouped axis: {grouped_axis}')

def _factorized_chord___group_shape(value: torch.Tensor, *, groups: int, width: int | None, grouped_axis: str) -> tuple[int, int, int, int]:
    """Validate a grouped matrix and return its logical shape without copying."""
    if value.ndim != 3:
        raise RuntimeError('TILLER requires batched matrices')
    layers, rows, columns = map(int, value.shape)
    if grouped_axis == 'rows':
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError('TILLER row-group inventory changed')
        return (layers, int(groups), int(width), columns)
    if grouped_axis == 'columns':
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError('TILLER column-group inventory changed')
        return (layers, int(groups), int(width), rows)
    if grouped_axis == 'matrix':
        if int(groups) != 1 or width is not None:
            raise RuntimeError('TILLER matrix inventory changed')
        return (layers, 1, rows, columns)
    raise ValueError(f'unknown TILLER grouped axis: {grouped_axis}')

def _factorized_chord__factorized_adaptive_tangent_chord_direction(parent: torch.Tensor, momentum: torch.Tensor, gradient: torch.Tensor, row_second_moment: torch.Tensor, column_second_moment: torch.Tensor, participation: torch.Tensor, congruence: torch.Tensor, *, groups: int, width: int | None, grouped_axis: str, beta2: float, step: int, eps: float=_factorized_chord__LOCKED_EPS):
    """Spend the stable response energy on a factorized adaptive tangent."""
    if not (parent.shape == momentum.shape == gradient.shape and parent.ndim == 3):
        raise RuntimeError('factorized adaptive tangent inventory changed')
    if float(beta2) != _factorized_chord__MATCHED_BETA2 or int(step) < 1 or float(eps) != _factorized_chord__LOCKED_EPS:
        raise ValueError('factorized adaptive tangent locked numerics changed')
    p, restore = _factorized_chord___group_view(parent.float(), groups=groups, width=width, grouped_axis=grouped_axis)
    m, _ = _factorized_chord___group_view(momentum.float(), groups=groups, width=width, grouped_axis=grouped_axis)
    g, _ = _factorized_chord___group_view(gradient.float(), groups=groups, width=width, grouped_axis=grouped_axis)
    if participation.shape != p.shape[:2] or congruence.shape != p.shape[:2] or row_second_moment.shape != p.shape[:-1] or (column_second_moment.shape != p.shape[:2] + p.shape[-1:]):
        raise RuntimeError('factorized adaptive tangent state inventory changed')
    squared = g.square()
    row_second_moment.mul_(beta2).add_(squared.sum(dim=-1), alpha=1.0 - beta2)
    column_second_moment.mul_(beta2).add_(squared.sum(dim=-2), alpha=1.0 - beta2)
    correction = 1.0 - float(beta2) ** int(step)
    tiny = torch.finfo(torch.float32).tiny
    row_total = row_second_moment.sum(dim=-1, keepdim=True).clamp_min(tiny)
    variance = row_second_moment[..., :, None] * column_second_moment[..., None, :] / row_total[..., :, None] / correction
    adaptive = m / (variance.sqrt() + float(eps))
    dims = (-2, -1)
    parent2 = p.square().sum(dim=dims)
    adaptive2 = adaptive.square().sum(dim=dims)
    parent_adaptive = (p * adaptive).sum(dim=dims)
    parent_descent = (g * p).sum(dim=dims)
    adaptive_descent = (g * adaptive).sum(dim=dims)
    projection = parent_adaptive / parent2.clamp_min(tiny)
    tangent2 = (adaptive2 - parent_adaptive.square() / parent2.clamp_min(tiny)).clamp_min(0.0)
    tangent_descent_signed = adaptive_descent - projection * parent_descent
    orientation = torch.where(tangent_descent_signed >= 0.0, torch.ones_like(tangent_descent_signed), -torch.ones_like(tangent_descent_signed))
    information = participation.float().clamp(0.0, 1.0)
    alignment = congruence.float().clamp(0.0, 1.0)
    departure_energy = (information * (1.0 - alignment.square()).clamp_min(0.0)).clamp(0.0, 1.0)
    parent_weight = torch.sqrt((1.0 - departure_energy).clamp_min(0.0))
    tangent_weight = torch.sqrt(departure_energy.clamp_min(0.0))
    adaptive_coefficient = tangent_weight * torch.sqrt(parent2 / tangent2.clamp_min(tiny)) * orientation
    parent_coefficient = parent_weight - adaptive_coefficient * projection
    candidate_descent = parent_coefficient * parent_descent + adaptive_coefficient * adaptive_descent
    finite = torch.isfinite(p).all(dim=dims) & torch.isfinite(m).all(dim=dims) & torch.isfinite(g).all(dim=dims) & torch.isfinite(adaptive).all(dim=dims) & torch.isfinite(parent_coefficient) & torch.isfinite(adaptive_coefficient) & torch.isfinite(candidate_descent)
    active = finite & (parent2 > 0.0) & (adaptive2 > 0.0) & (tangent2 > tiny)
    safe = active & (parent_descent > 0.0) & (candidate_descent > 0.0)
    parent_coefficient = torch.where(safe, parent_coefficient, torch.ones_like(parent_coefficient))
    adaptive_coefficient = torch.where(safe, adaptive_coefficient, torch.zeros_like(adaptive_coefficient))
    selected2 = (parent_coefficient.square() * parent2 + adaptive_coefficient.square() * adaptive2 + 2.0 * parent_coefficient * adaptive_coefficient * parent_adaptive).clamp_min(0.0)
    selected_norm = torch.sqrt(selected2)
    parent_norm = torch.sqrt(parent2.clamp_min(0.0))
    parent_cosine = ((parent_coefficient * parent2 + adaptive_coefficient * parent_adaptive) / (selected_norm * parent_norm).clamp_min(float(eps))).clamp(-1.0, 1.0)
    budget = (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)
    shape = (*parent_coefficient.shape, 1, 1)
    p.mul_(parent_coefficient.view(shape))
    p.addcmul_(adaptive, adaptive_coefficient.view(shape))
    selected = restore(p).to(parent.dtype)
    torch._assert_async(torch.isfinite(selected).all())
    return (selected, {'active': active, 'safe': safe, 'parent_cosine': parent_cosine, 'budget_residual': budget, 'departure_energy': departure_energy, 'parent_descent': parent_descent, 'candidate_descent': candidate_descent, 'parent_coefficient': parent_coefficient, 'adaptive_coefficient': adaptive_coefficient, 'tangent_orientation': orientation})

def _factorized_chord___factor_state_elements(*, layers: int, groups: int, hidden: int, external: int):
    return 2 * layers * (hidden + groups * external) + 6 * layers * external

def _factorized_chord__factorized_adaptive_tangent_chord_scaling_formula(**kwargs):
    result = dict(_factorized_chord__import_window32_cadence8_scaling_formula(**kwargs))
    layers = int(kwargs['total_layers'])
    groups = int(kwargs['total_groups'])
    hidden = int(kwargs['intermediate_width'])
    external = int(kwargs['model_width'])
    factor_state = _factorized_chord___factor_state_elements(layers=layers, groups=groups, hidden=hidden, external=external)
    direction_summary = 3 * factor_state + 5 * layers * (groups + 1)
    result.update({'factorized_row_column_state_elements': factor_state, 'persistent_state_elements': int(result['persistent_state_elements']) + factor_state, 'arbitrary_shard_direction_summary_elements': direction_summary, 'communicated_summary_elements': int(result['communicated_summary_elements']) + direction_summary, 'ordinary_communicated_summary_elements': int(result['ordinary_communicated_summary_elements']) + direction_summary, 'additional_persistent_state_elements': factor_state, 'state_depends_on_total_activation_positions': 0, 'state_scales_linearly_with_width': 1, 'additional_native_polar_maps_per_role': 0, 'dense_tangent_projector_elements': 0, 'additional_dense_solve_dimension': 0})
    return result

def _factorized_chord___state_for_direction(optimizer, *, parent: torch.Tensor, groups: int, width: int | None, grouped_axis: str, key_prefix: str):
    grouped_shape = _factorized_chord___group_shape(parent, groups=groups, width=width, grouped_axis=grouped_axis)
    if hasattr(optimizer, 'pairs'):
        anchor = optimizer.state[optimizer.pairs[0]['in_weight']]
    else:
        anchor = optimizer.state[optimizer.role_parameters['qkv'][0]]
    row_key = key_prefix + '_row_second_moment'
    column_key = key_prefix + '_column_second_moment'
    row = anchor.get(row_key)
    column = anchor.get(column_key)
    if row is None:
        row = torch.zeros(grouped_shape[:-1], device=parent.device, dtype=torch.float32)
        anchor[row_key] = row
    if column is None:
        column = torch.zeros(grouped_shape[:2] + grouped_shape[-1:], device=parent.device, dtype=torch.float32)
        anchor[column_key] = column
    if row.shape != grouped_shape[:-1] or column.shape != grouped_shape[:2] + grouped_shape[-1:]:
        raise RuntimeError('factorized adaptive tangent checkpoint inventory changed')
    return (anchor, row, column)

def _factorized_chord___retag(report: dict) -> dict:
    return {key.replace(_factorized_chord__import_WINDOW32_PREFIX, _factorized_chord__PREFIX, 1): _factorized_chord__FAMILY_ID if value == _factorized_chord__import_WINDOW32_FAMILY_ID else value for key, value in report.items()}

class _factorized_chord__FactorizedAdaptiveTangentChordRouter(_factorized_chord__import_Window32Cadence8ResponseGroupPolarRouter):
    family_id = _factorized_chord__FAMILY_ID
    telemetry_prefix = _factorized_chord__PREFIX
    fairness_component = 'factorized_adaptive_tangent_chord_lr_scale'

    def __init__(self, pairs, **kwargs):
        if float(kwargs.get('beta2', -1.0)) != _factorized_chord__MATCHED_BETA2:
            raise ValueError('factorized adaptive tangent requires beta2=.95')
        if float(kwargs.get('eps', -1.0)) != _factorized_chord__LOCKED_EPS:
            raise ValueError('factorized adaptive tangent requires eps=1e-8')
        super().__init__(pairs, **kwargs)
        self.param_groups[0][_factorized_chord__PREFIX + 'family_id'] = _factorized_chord__FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({'matched_beta2_factorized_tangent_lr_scale': 1.0, 'response_energy_chord_lr_scale': 1.0, 'signed_current_gradient_orientation_lr_scale': 1.0, 'exact_native_budget_lr_scale': 1.0})
        return result

    @torch.no_grad()
    def step(self, closure=None):
        global _cadence_geometry__import_compact_postpolar_group_response_homotopy, _predictive_transaction__import_compact_postpolar_group_response_homotopy
        records = []

        def direction(parent, momentum, gradient, participation, congruence, *, groups, width, grouped_axis, eps=_factorized_chord__LOCKED_EPS):
            anchor, rows, columns = _factorized_chord___state_for_direction(self, parent=parent, groups=groups, width=width, grouped_axis=grouped_axis, key_prefix='factorized_adaptive_tangent_' + grouped_axis)
            update = int(anchor.get('predictive_response_transaction_updates', 0)) + 1
            selected, metadata = _factorized_chord__factorized_adaptive_tangent_chord_direction(parent, momentum, gradient, rows, columns, participation, congruence, groups=groups, width=width, grouped_axis=grouped_axis, beta2=_factorized_chord__MATCHED_BETA2, step=update, eps=eps)
            records.append(metadata)
            return (selected, metadata)
        with _factorized_chord___PATCH_LOCK:
            if _predictive_transaction__import_compact_postpolar_group_response_homotopy is not _factorized_chord___EXPECTED_REFRESH_DIRECTION or _cadence_geometry__import_compact_postpolar_group_response_homotopy is not _factorized_chord___EXPECTED_ORDINARY_DIRECTION:
                raise RuntimeError('factorized adaptive router binding changed')
            _predictive_transaction__import_compact_postpolar_group_response_homotopy = direction
            _cadence_geometry__import_compact_postpolar_group_response_homotopy = direction
            try:
                loss = super().step(closure)
            finally:
                _predictive_transaction__import_compact_postpolar_group_response_homotopy = _factorized_chord___EXPECTED_REFRESH_DIRECTION
                _cadence_geometry__import_compact_postpolar_group_response_homotopy = _factorized_chord___EXPECTED_ORDINARY_DIRECTION
        self._last_telemetry = _factorized_chord___retag(self._last_telemetry)
        if self._last_telemetry and records:
            safe = torch.cat([x['safe'].reshape(-1) for x in records])
            cosine = torch.cat([x['parent_cosine'].reshape(-1) for x in records])
            budget = torch.cat([x['budget_residual'].reshape(-1) for x in records])
            energy = torch.cat([x['departure_energy'].reshape(-1) for x in records])
            signed = torch.cat([x['adaptive_coefficient'].reshape(-1) for x in records])
            scaling = _factorized_chord__factorized_adaptive_tangent_chord_scaling_formula(total_positions=1, total_layers=len(self.pairs), total_groups=self.groups, intermediate_width=self.hidden, model_width=self.external)
            self._last_telemetry.update({_factorized_chord__PREFIX + 'family_id': _factorized_chord__FAMILY_ID, _factorized_chord__PREFIX + 'adaptive_tangent_safe_fraction': float(safe.float().mean()), _factorized_chord__PREFIX + 'adaptive_tangent_parent_cosine_median': float(cosine.median()), _factorized_chord__PREFIX + 'adaptive_tangent_budget_residual_max': float(budget.amax()), _factorized_chord__PREFIX + 'departure_energy_median': float(energy.median()), _factorized_chord__PREFIX + 'adaptive_coefficient_min': float(signed.amin()), _factorized_chord__PREFIX + 'adaptive_coefficient_max': float(signed.amax()), _factorized_chord__PREFIX + 'factorized_row_column_state_elements': scaling['factorized_row_column_state_elements'], _factorized_chord__PREFIX + 'arbitrary_shard_direction_summary_elements': scaling['arbitrary_shard_direction_summary_elements'], _factorized_chord__PREFIX + 'state_depends_on_total_activation_positions': 0, _factorized_chord__PREFIX + 'state_scales_linearly_with_width': 1, _factorized_chord__PREFIX + 'additional_native_polar_maps_per_role': 0, _factorized_chord__PREFIX + 'dense_tangent_projector_elements': 0})
        return loss

class _factorized_chord__FactorizedAdaptiveTangentChordAttentionOptimizer(_factorized_chord__import_Window32Cadence8HeadPolarAttentionOptimizer):
    family_id = _factorized_chord__FAMILY_ID
    telemetry_prefix = _factorized_chord__PREFIX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0][_factorized_chord__PREFIX + 'family_id'] = _factorized_chord__FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({'attention_matched_beta2_factorized_tangent_lr_scale': 1.0, 'attention_response_energy_chord_lr_scale': 1.0, 'attention_signed_current_gradient_orientation_lr_scale': 1.0})
        return result

    @torch.no_grad()
    def step(self, closure=None):
        global _compact_homotopy__compact_postpolar_group_response_homotopy
        records = []
        calls = 0
        anchor = self.state[self.role_parameters['qkv'][0]]
        update = int(anchor.get('factorized_adaptive_tangent_updates', 0)) + 1

        def direction(parent, momentum, gradient, participation, congruence, *, groups, width, grouped_axis, eps=_factorized_chord__LOCKED_EPS):
            nonlocal calls
            if calls >= 2:
                raise RuntimeError('factorized adaptive attention role count changed')
            role = 'qkv' if calls == 0 else 'attention_output'
            calls += 1
            _anchor, rows, columns = _factorized_chord___state_for_direction(self, parent=parent, groups=groups, width=width, grouped_axis=grouped_axis, key_prefix='factorized_adaptive_tangent_' + role)
            selected, metadata = _factorized_chord__factorized_adaptive_tangent_chord_direction(parent, momentum, gradient, rows, columns, participation, congruence, groups=groups, width=width, grouped_axis=grouped_axis, beta2=_factorized_chord__MATCHED_BETA2, step=update, eps=eps)
            records.append(metadata)
            return (selected, metadata)
        with _factorized_chord___PATCH_LOCK:
            if _compact_homotopy__compact_postpolar_group_response_homotopy is not _factorized_chord___EXPECTED_ATTENTION_DIRECTION:
                raise RuntimeError('factorized adaptive attention binding changed')
            _compact_homotopy__compact_postpolar_group_response_homotopy = direction
            try:
                loss = super().step(closure)
            finally:
                _compact_homotopy__compact_postpolar_group_response_homotopy = _factorized_chord___EXPECTED_ATTENTION_DIRECTION
        if calls != 2:
            raise RuntimeError('factorized adaptive attention omitted a role')
        anchor['factorized_adaptive_tangent_updates'] = update
        self._last_telemetry = _factorized_chord___retag(self._last_telemetry)
        if self._last_telemetry:
            safe = torch.cat([x['safe'].reshape(-1) for x in records])
            cosine = torch.cat([x['parent_cosine'].reshape(-1) for x in records])
            self._last_telemetry.update({_factorized_chord__PREFIX + 'attention_family_id': _factorized_chord__FAMILY_ID, _factorized_chord__PREFIX + 'attention_owner_count': 0, _factorized_chord__PREFIX + 'attention_selected_update_elements_published': 0, _factorized_chord__PREFIX + 'attention_adaptive_tangent_safe_fraction': float(safe.float().mean()), _factorized_chord__PREFIX + 'attention_adaptive_tangent_parent_cosine_median': float(cosine.median())})
        return loss
_compiled_chord__FAMILY_ID = 'factorized_adaptive_tangent_chord_compiled_muon_v2'
_compiled_chord__PREFIX = 'factorized_adaptive_tangent_chord_compiled_'
_compiled_chord__PARENT_FAMILY_ID = _factorized_chord__FAMILY_ID
_compiled_chord__PARENT_PREFIX = _factorized_chord__PREFIX
_compiled_chord__LOCKED_EPS = _factorized_chord__LOCKED_EPS
_compiled_chord__MATCHED_BETA2 = _factorized_chord__MATCHED_BETA2
_compiled_chord___PATCH_LOCK = threading.RLock()
_compiled_chord___TINY = torch.finfo(torch.float32).tiny

def _compiled_chord___adaptive_tangent_chord_fullgraph(p: torch.Tensor, m: torch.Tensor, g: torch.Tensor, row_second_moment: torch.Tensor, column_second_moment: torch.Tensor, participation: torch.Tensor, congruence: torch.Tensor, correction: torch.Tensor):
    squared = g.square()
    row_second_moment.mul_(0.95).add_(squared.sum(dim=-1), alpha=0.05)
    column_second_moment.mul_(0.95).add_(squared.sum(dim=-2), alpha=0.05)
    row_total = row_second_moment.sum(dim=-1, keepdim=True).clamp_min(_compiled_chord___TINY)
    variance = row_second_moment[..., :, None] * column_second_moment[..., None, :] / row_total[..., :, None]
    variance.div_(correction)
    adaptive = m / (variance.sqrt() + 1e-08)
    dims = (-2, -1)
    parent2 = p.square().sum(dim=dims)
    adaptive2 = adaptive.square().sum(dim=dims)
    parent_adaptive = (p * adaptive).sum(dim=dims)
    parent_descent = (g * p).sum(dim=dims)
    adaptive_descent = (g * adaptive).sum(dim=dims)
    projection = parent_adaptive / parent2.clamp_min(_compiled_chord___TINY)
    tangent2 = (adaptive2 - parent_adaptive.square() / parent2.clamp_min(_compiled_chord___TINY)).clamp_min(0.0)
    tangent_descent = adaptive_descent - projection * parent_descent
    orientation = torch.where(tangent_descent >= 0.0, torch.ones_like(tangent_descent), -torch.ones_like(tangent_descent))
    information = participation.clamp(0.0, 1.0)
    alignment = congruence.clamp(0.0, 1.0)
    departure_energy = (information * (1.0 - alignment.square()).clamp_min(0.0)).clamp(0.0, 1.0)
    parent_weight = torch.sqrt((1.0 - departure_energy).clamp_min(0.0))
    tangent_weight = torch.sqrt(departure_energy.clamp_min(0.0))
    adaptive_coefficient = tangent_weight * torch.sqrt(parent2 / tangent2.clamp_min(_compiled_chord___TINY)) * orientation
    parent_coefficient = parent_weight - adaptive_coefficient * projection
    candidate_descent = parent_coefficient * parent_descent + adaptive_coefficient * adaptive_descent
    finite = torch.isfinite(p).all(dim=dims) & torch.isfinite(m).all(dim=dims) & torch.isfinite(g).all(dim=dims) & torch.isfinite(adaptive).all(dim=dims) & torch.isfinite(parent_coefficient) & torch.isfinite(adaptive_coefficient) & torch.isfinite(candidate_descent)
    active = finite & (parent2 > 0.0) & (adaptive2 > 0.0) & (tangent2 > _compiled_chord___TINY)
    safe = active & (parent_descent > 0.0) & (candidate_descent > 0.0)
    parent_coefficient = torch.where(safe, parent_coefficient, torch.ones_like(parent_coefficient))
    adaptive_coefficient = torch.where(safe, adaptive_coefficient, torch.zeros_like(adaptive_coefficient))
    selected2 = (parent_coefficient.square() * parent2 + adaptive_coefficient.square() * adaptive2 + 2.0 * parent_coefficient * adaptive_coefficient * parent_adaptive).clamp_min(0.0)
    selected_norm = torch.sqrt(selected2)
    parent_norm = torch.sqrt(parent2.clamp_min(0.0))
    parent_cosine = ((parent_coefficient * parent2 + adaptive_coefficient * parent_adaptive) / (selected_norm * parent_norm).clamp_min(1e-08)).clamp(-1.0, 1.0)
    budget = (selected_norm - parent_norm).abs() / parent_norm.clamp_min(1.0)
    selected = p * parent_coefficient[..., None, None] + adaptive * adaptive_coefficient[..., None, None]
    return (selected, active, safe, parent_cosine, budget, departure_energy, parent_descent, candidate_descent, parent_coefficient, adaptive_coefficient, orientation)
_compiled_chord___COMPILED_ADAPTIVE_TANGENT_CHORD_FULLGRAPH = torch.compile(_compiled_chord___adaptive_tangent_chord_fullgraph, fullgraph=True, dynamic=False)

def _compiled_chord__compiled_factorized_adaptive_tangent_chord_direction(parent: torch.Tensor, momentum: torch.Tensor, gradient: torch.Tensor, row_second_moment: torch.Tensor, column_second_moment: torch.Tensor, participation: torch.Tensor, congruence: torch.Tensor, *, groups: int, width: int | None, grouped_axis: str, beta2: float, step: int, eps: float=_compiled_chord__LOCKED_EPS):
    if not (parent.shape == momentum.shape == gradient.shape and parent.ndim == 3):
        raise RuntimeError('compiled adaptive tangent inventory changed')
    if float(beta2) != _compiled_chord__MATCHED_BETA2 or int(step) < 1 or float(eps) != _compiled_chord__LOCKED_EPS:
        raise ValueError('compiled adaptive tangent locked numerics changed')
    p, restore = _factorized_chord___group_view(parent, groups=groups, width=width, grouped_axis=grouped_axis)
    m, _ = _factorized_chord___group_view(momentum, groups=groups, width=width, grouped_axis=grouped_axis)
    g, _ = _factorized_chord___group_view(gradient, groups=groups, width=width, grouped_axis=grouped_axis)
    if participation.shape != p.shape[:2] or congruence.shape != p.shape[:2] or row_second_moment.shape != p.shape[:-1] or (column_second_moment.shape != p.shape[:2] + p.shape[-1:]):
        raise RuntimeError('compiled adaptive tangent state inventory changed')
    correction = torch.scalar_tensor(1.0 - _compiled_chord__MATCHED_BETA2 ** int(step), device=p.device, dtype=p.dtype)
    program = _compiled_chord___COMPILED_ADAPTIVE_TANGENT_CHORD_FULLGRAPH if parent.is_cuda else _compiled_chord___adaptive_tangent_chord_fullgraph
    selected, active, safe, parent_cosine, budget, departure_energy, parent_descent, candidate_descent, parent_coefficient, adaptive_coefficient, orientation = program(p, m, g, row_second_moment, column_second_moment, participation.float(), congruence.float(), correction)
    torch._assert_async(torch.isfinite(selected).all())
    return (restore(selected).to(parent.dtype), {'active': active, 'safe': safe, 'parent_cosine': parent_cosine, 'budget_residual': budget, 'departure_energy': departure_energy, 'parent_descent': parent_descent, 'candidate_descent': candidate_descent, 'parent_coefficient': parent_coefficient, 'adaptive_coefficient': adaptive_coefficient, 'tangent_orientation': orientation})

def _compiled_chord__factorized_adaptive_tangent_chord_compiled_scaling_formula(**kwargs):
    result = dict(_factorized_chord__factorized_adaptive_tangent_chord_scaling_formula(**kwargs))
    result.update({'compiled_static_fullgraph': 1, 'scientific_equation_changed_vs_parent': 0, 'state_recurrence_changed_vs_parent': 0, 'additional_persistent_state_elements_vs_parent': 0})
    return result

@contextmanager
def _compiled_chord___installed_compiled_direction():
    global _factorized_chord__factorized_adaptive_tangent_chord_direction
    with _compiled_chord___PATCH_LOCK:
        original = _factorized_chord__factorized_adaptive_tangent_chord_direction
        if original is not _compiled_chord___ORIGINAL_PARENT_DIRECTION:
            raise RuntimeError('compiled adaptive tangent parent was patched')
        _factorized_chord__factorized_adaptive_tangent_chord_direction = _compiled_chord__compiled_factorized_adaptive_tangent_chord_direction
        try:
            yield
        finally:
            _factorized_chord__factorized_adaptive_tangent_chord_direction = original

def _compiled_chord___retag(report: dict) -> dict:
    result = {}
    for key, value in report.items():
        if key.startswith(_compiled_chord__PREFIX):
            pass
        elif key.startswith(_compiled_chord__PARENT_PREFIX):
            key = _compiled_chord__PREFIX + key[len(_compiled_chord__PARENT_PREFIX):]
        if value == _compiled_chord__PARENT_FAMILY_ID:
            value = _compiled_chord__FAMILY_ID
        result[key] = value
    return result

class _compiled_chord__FactorizedAdaptiveTangentChordCompiledRouter(_factorized_chord__FactorizedAdaptiveTangentChordRouter):
    family_id = _compiled_chord__FAMILY_ID
    telemetry_prefix = _compiled_chord__PREFIX
    fairness_component = 'factorized_adaptive_tangent_chord_compiled_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0].pop(_compiled_chord__PARENT_PREFIX + 'family_id', None)
        self.param_groups[0][_compiled_chord__PREFIX + 'family_id'] = _compiled_chord__FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result['compiled_static_fullgraph_execution_lr_scale'] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        with _compiled_chord___installed_compiled_direction():
            loss = super().step(closure)
        self._last_telemetry = _compiled_chord___retag(self._last_telemetry)
        if self._last_telemetry:
            self._last_telemetry.update({_compiled_chord__PREFIX + 'family_id': _compiled_chord__FAMILY_ID, _compiled_chord__PREFIX + 'compiled_static_fullgraph': 1, _compiled_chord__PREFIX + 'scientific_equation_changed_vs_parent': 0, _compiled_chord__PREFIX + 'state_recurrence_changed_vs_parent': 0})
        return loss

class _compiled_chord__FactorizedAdaptiveTangentChordCompiledAttentionOptimizer(_factorized_chord__FactorizedAdaptiveTangentChordAttentionOptimizer):
    family_id = _compiled_chord__FAMILY_ID
    telemetry_prefix = _compiled_chord__PREFIX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0].pop(_compiled_chord__PARENT_PREFIX + 'family_id', None)
        self.param_groups[0][_compiled_chord__PREFIX + 'family_id'] = _compiled_chord__FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result['attention_compiled_static_fullgraph_execution_lr_scale'] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        with _compiled_chord___installed_compiled_direction():
            loss = super().step(closure)
        self._last_telemetry = _compiled_chord___retag(self._last_telemetry)
        if self._last_telemetry:
            self._last_telemetry.update({_compiled_chord__PREFIX + 'attention_family_id': _compiled_chord__FAMILY_ID, _compiled_chord__PREFIX + 'attention_owner_count': 0, _compiled_chord__PREFIX + 'attention_selected_update_elements_published': 0, _compiled_chord__PREFIX + 'attention_compiled_static_fullgraph': 1})
        return loss
_compiled_chord___ORIGINAL_PARENT_DIRECTION = _factorized_chord__factorized_adaptive_tangent_chord_direction
_tiller_core__import_MAXIMUM_SELECTION_ROWS = _diagonal_completed_biatlas_metric__MAXIMUM_SELECTION_ROWS
_tiller_core__import_diagonal_completed_biatlas_transaction = _diagonal_completed_biatlas_metric__diagonal_completed_biatlas_transaction
_tiller_core__import_CURRENT_ROWS = _every_step_rfd_gradient_ledger__import_CURRENT_ROWS
_tiller_core__import_PERSISTENT_ROWS = _every_step_rfd_gradient_ledger__import_PERSISTENT_ROWS
_tiller_core__import_every_step_rfd_gradient_rows = _every_step_rfd_gradient_ledger__every_step_rfd_gradient_rows
_tiller_core__import_functional_row_norm = _every_step_rfd_gradient_ledger__functional_row_norm
_tiller_core__import_trace_matched_gradient_surrogate = _every_step_rfd_gradient_ledger__trace_matched_gradient_surrogate
_tiller_core__import_FDTailBiatlasRows = _fd_tail_biatlas_metric__FDTailBiatlasRows
_tiller_core__import_FixedProbeTransactionResult = _fixed_probe_transaction__FixedProbeTransactionResult
_tiller_core__import_ReplicatedFixedProbeTransactionResult = _fixed_probe_transaction__ReplicatedFixedProbeTransactionResult
FAMILY_ID = 'tiller_v1'
PREFIX = 'tiller_'
MATCHED_BETA2 = 0.95

def _retag(report: dict) -> dict:
    result = {}
    for key, value in report.items():
        if key.startswith(PREFIX):
            pass
        elif key.startswith(_compiled_chord__PREFIX):
            key = PREFIX + key[len(_compiled_chord__PREFIX):]
        elif key.startswith(_factorized_chord__PREFIX):
            key = PREFIX + key[len(_factorized_chord__PREFIX):]
        if value in (_compiled_chord__FAMILY_ID, _factorized_chord__FAMILY_ID):
            value = FAMILY_ID
        result[key] = value
    return result

def tiller_scaling_formula(**kwargs):
    result = dict(_compiled_chord__factorized_adaptive_tangent_chord_compiled_scaling_formula(**kwargs))
    layers = int(kwargs['total_layers'])
    groups = int(kwargs['total_groups'])
    coordinates = layers * groups
    if coordinates < 1:
        raise ValueError('TILLER requires at least one layer/group coordinate')
    ledger_state = _tiller_core__import_PERSISTENT_ROWS * coordinates + 2 * coordinates + 4
    result.update({'family_id': FAMILY_ID, 'coordinate_count': coordinates, 'ledger_checkpoint_tensor_elements': ledger_state, 'additional_persistent_state_elements': ledger_state, 'persistent_state_elements': int(result['persistent_state_elements']) + ledger_state, 'maximum_live_factor_elements': _tiller_core__import_MAXIMUM_SELECTION_ROWS * coordinates, 'every_step_gradient_score_ledger': 1, 'trace_matched_gradient_surrogate': 1, 'functional_score_refresh_interval': 8, 'matched_beta2_every_optimizer_step': 1, 'robust_fd_midpoint_tail': 1, 'factorized_parameter_direction_unchanged': 1, 'adaptive_rank64_cross_coordinate_factor': 1, 'largest_dense_solve_dimension': _tiller_core__import_MAXIMUM_SELECTION_ROWS, 'largest_transaction_dense_dimension': 32, 'largest_temporal_dense_dimension': _tiller_core__import_MAXIMUM_SELECTION_ROWS, 'state_depends_on_total_activation_positions': 0, 'owner_count': 0, 'complete_layer_owners': 0, 'complete_coordinate_owners': 0, 'owner_local_mathematics': 0, 'dense_lg_by_lg_metric_elements': 0, 'selected_update_elements_published': 0, 'new_tunable_hyperparameters': 0, 'state_scales_as': 'O(LH + LGd + 64LG)'})
    return result

def _previous(anchor, reference: torch.Tensor, *, validate_values: bool=True):
    scores = anchor.get('factorized_rfd_persistent_scores')
    diagonal = anchor.get('factorized_rfd_persistent_total_diagonal')
    decay = anchor.get('factorized_rfd_persistent_decay_cross')
    tail = anchor.get('factorized_rfd_persistent_isotropic_tail')
    values = (scores, diagonal, decay, tail)
    if all((value is None for value in values)):
        return (None, None, None, None)
    if any((not torch.is_tensor(value) for value in values)) or scores.ndim != 2 or int(scores.shape[0]) not in (_tiller_core__import_CURRENT_ROWS, _tiller_core__import_PERSISTENT_ROWS) or (diagonal.shape != scores.shape[1:]) or (decay.shape != scores.shape[1:]) or (tail.numel() != 1) or any((value.dtype != reference.dtype for value in values)) or any((value.device != reference.device for value in values)):
        raise RuntimeError('TILLER checkpoint inventory changed')
    if validate_values:
        valid = torch.isfinite(scores).all() & torch.isfinite(diagonal).all() & torch.isfinite(decay).all() & torch.isfinite(tail).all() & (diagonal >= 0.0).all() & (tail >= 0.0).all()
        if not bool(valid):
            raise RuntimeError('TILLER checkpoint inventory changed')
    return (scores, diagonal, decay, tail)

def _store(anchor, rows: _tiller_core__import_FDTailBiatlasRows) -> None:
    anchor['factorized_rfd_persistent_scores'] = rows.persistent_scores.detach().clone()
    anchor['factorized_rfd_persistent_total_diagonal'] = rows.persistent_total_diagonal.detach().clone()
    anchor['factorized_rfd_persistent_decay_cross'] = rows.persistent_decay_cross.detach().clone()
    anchor['factorized_rfd_persistent_isotropic_tail'] = rows.persistent_isotropic_tail.detach().clone()
    anchor['factorized_rfd_discarded_energy_fraction'] = rows.discarded_energy_fraction.detach().clone()
    anchor['factorized_rfd_shrinkage'] = rows.fd_shrinkage.detach().clone()

class TILLERRouter(_compiled_chord__FactorizedAdaptiveTangentChordCompiledRouter):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = 'rlb_tiller_lr_scale'

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0].pop(_compiled_chord__PREFIX + 'family_id', None)
        self.param_groups[0].pop(_factorized_chord__PREFIX + 'family_id', None)
        self.param_groups[0][PREFIX + 'family_id'] = FAMILY_ID
        self._rfd_rows = None
        self._rfd_functional_refresh = False
        self._rfd_gradient_scale = None
        self._rfd_decay_derivative = None
        self._rfd_selection = None

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({'every_step_gradient_score_ledger_lr_scale': 1.0, 'trace_matched_gradient_surrogate_lr_scale': 1.0, 'robust_fd_midpoint_tail_lr_scale': 1.0, 'matrix_free_rfd_krylov_transaction_lr_scale': 1.0, 'unchanged_compiled_factorized_direction_lr_scale': 1.0, 'signed_rfd_coefficients_lr_scale': 1.0})
        return result

    def _advance_rows(self, scores, decay_action, *, functional: bool):
        anchor = self.state[self.pairs[0]['in_weight']]
        # The ledger validates these exact values immediately below. Keep the
        # structural checkpoint check here without synchronizing twice.
        previous_scores, previous_diagonal, previous_decay, previous_tail = _previous(anchor, scores, validate_values=False)
        rows = _tiller_core__import_every_step_rfd_gradient_rows(scores, decay_action, previous_scores, previous_diagonal, previous_decay, beta2=MATCHED_BETA2, previous_isotropic_tail=previous_tail)
        _store(anchor, rows)
        step = int(anchor.get('factorized_rfd_step', 0)) + 1
        anchor['factorized_rfd_step'] = step
        if functional:
            # ``every_step_rfd_gradient_rows`` has just validated ``scores``.
            anchor['factorized_rfd_reference_row_norm'] = _tiller_core__import_functional_row_norm(scores, validate_values=False).detach().clone()
            scale = scores.new_ones(())
        else:
            scale = self._rfd_gradient_scale
            if scale is None:
                raise RuntimeError('TILLER gradient scale is absent')
        self._rfd_rows = rows
        self._rfd_functional_refresh = bool(functional)
        self._rfd_gradient_scale = scale
        return rows

    def predictive_rows_fn(self, current_scores, current_decay_action, previous_scores, previous_decay_action, *, beta2):
        if float(beta2) != MATCHED_BETA2:
            raise ValueError('TILLER requires matched beta2=.95')
        inherited = super().predictive_rows_fn(current_scores, current_decay_action, previous_scores, previous_decay_action, beta2=beta2)
        self._advance_rows(current_scores, current_decay_action, functional=True)
        return inherited

    def _transaction(self, exact_by_role, momentum_by_role, weights, layer_ids, *, total_layers, eta, rounds=64):
        if self._rfd_rows is None:
            anchor = self.state[self.pairs[0]['in_weight']]
            reference = anchor.get('factorized_rfd_reference_row_norm')
            decay = self._rfd_decay_derivative
            if not torch.is_tensor(reference) or not torch.is_tensor(decay):
                raise RuntimeError('TILLER lacks functional calibration')
            scores, decay_action, scale = _tiller_core__import_trace_matched_gradient_surrogate(exact_by_role, decay, reference)
            self._rfd_gradient_scale = scale
            self._advance_rows(scores, decay_action, functional=False)
        rows = self._rfd_rows
        if rows is None:
            raise RuntimeError('TILLER ledger rows are absent')
        result = _tiller_core__import_diagonal_completed_biatlas_transaction(rows, exact_by_role, momentum_by_role, weights, layer_ids, total_layers=int(total_layers), eta=float(eta), rounds=int(rounds), diagnostics=bool(self._capture_telemetry_next_step))
        self._rfd_selection = result
        coordinates = int(weights.numel())
        summary_elements = int(rows.selection_scores.numel()) + 4 * coordinates
        sharded = _tiller_core__import_FixedProbeTransactionResult(local_coefficients=result.coefficients, local_candidate_coefficients=result.candidate_coefficients, accepted=result.accepted, multiplier=result.multiplier, rank=result.factor_rank, eigenvalue_max=result.diagonal_maximum, hard_case=result.hard_case, parent_score=result.parent_score, candidate_score=result.candidate_score, budget_residual=result.budget_residual, local_coordinate_count=coordinates, global_coordinate_count=coordinates, global_probe_count=_tiller_core__import_CURRENT_ROWS, collective_rounds=0, summary_elements=summary_elements, owner_count=0, dense_LG_by_LG_metric_elements=0, selected_update_elements_published=0, method_state_depends_on_total_tokens=False)
        return _tiller_core__import_ReplicatedFixedProbeTransactionResult(coefficients=result.coefficients, candidate_coefficients=result.candidate_coefficients, sharded_result=sharded, local_probe_count=_tiller_core__import_CURRENT_ROWS, global_probe_count=_tiller_core__import_CURRENT_ROWS, cross_layer_coupling_ratio=result.cross_layer_coupling_ratio, collective_rounds=0, score_scalars_exchanged_per_rank=0, coefficient_scalars_exchanged_per_rank=0, selected_update_elements_published=0, method_state_depends_on_total_tokens=False)

    def transaction_fn(self, global_scores, global_decay_action, exact_by_role, momentum_by_role, weights, layer_ids, *, total_layers, eta, gather_rounds, group):
        del global_scores, global_decay_action, gather_rounds, group
        result = self._transaction(exact_by_role, momentum_by_role, weights, layer_ids, total_layers=total_layers, eta=eta)
        self._cadence_transaction = result
        return result

    @contextmanager
    def _installed_ordinary_transaction(self):
        global _cadence_geometry__import_rank64_transaction_from_replicated_rows
        original = _cadence_geometry__import_rank64_transaction_from_replicated_rows

        def transaction(global_scores, global_decay_action, exact_by_role, momentum_by_role, weights, layer_ids, *, total_layers, eta, gather_rounds, group):
            del global_scores, global_decay_action, gather_rounds, group
            result = self._transaction(exact_by_role, momentum_by_role, weights, layer_ids, total_layers=total_layers, eta=eta)
            self._cadence_transaction = result
            return result
        with _factorized_chord___PATCH_LOCK:
            if _cadence_geometry__import_rank64_transaction_from_replicated_rows is not original:
                raise RuntimeError('TILLER transaction binding changed')
            _cadence_geometry__import_rank64_transaction_from_replicated_rows = transaction
            try:
                yield
            finally:
                _cadence_geometry__import_rank64_transaction_from_replicated_rows = original

    @torch.no_grad()
    def step(self, closure=None):
        # Telemetry describes one optimizer step.  Parent implementations only
        # replace the report when capture is enabled, so retaining the previous
        # report would make their audit-only reductions and scalar reads run on
        # every subsequent non-capture step.
        self._last_telemetry = {}
        self._rfd_rows = None
        self._rfd_functional_refresh = False
        self._rfd_gradient_scale = None
        self._rfd_selection = None
        if not bool(self._capture_response_this_transition):
            decay = torch.zeros((), device=self.pairs[0]['in_weight'].device, dtype=torch.float32)
            for pair in self.pairs:
                for key in ('in_weight', 'out_weight'):
                    parameter = pair[key]
                    if parameter.grad is None:
                        raise RuntimeError('TILLER parameter lacks gradient')
                    decay.add_((parameter.grad.detach().float() * parameter.detach().float()).sum())
            self._rfd_decay_derivative = decay * float(self.param_groups[0]['weight_decay'])
        else:
            for pair in self.pairs:
                for key in ('in_weight', 'out_weight'):
                    if pair[key].grad is None:
                        raise RuntimeError('TILLER parameter lacks gradient')
        with self._installed_ordinary_transaction():
            loss = super().step(closure)
        rows = self._rfd_rows
        selection = self._rfd_selection
        scale = self._rfd_gradient_scale
        if rows is None or selection is None or scale is None:
            raise RuntimeError('TILLER did not advance exactly once')
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry:
            scaling = tiller_scaling_formula(total_positions=1, total_layers=len(self.pairs), total_groups=self.groups, intermediate_width=self.hidden, model_width=self.external)
            self._last_telemetry.update({PREFIX + 'family_id': FAMILY_ID, PREFIX + 'factorized_parameter_direction_unchanged': 1, PREFIX + 'every_step_gradient_score_ledger': 1, PREFIX + 'functional_score_refresh': int(self._rfd_functional_refresh), PREFIX + 'trace_matched_gradient_surrogate': int(not self._rfd_functional_refresh), PREFIX + 'gradient_score_scale': float(scale.item()), PREFIX + 'matched_beta2_every_optimizer_step': 1, PREFIX + 'ledger_step': int(self.state[self.pairs[0]['in_weight']]['factorized_rfd_step']), PREFIX + 'robust_fd_midpoint_tail': 1, PREFIX + 'signed_coefficients_allowed': 1, PREFIX + 'selection_factor_rows': int(rows.selection_scores.shape[0]), PREFIX + 'persistent_factor_rows': int(rows.persistent_scores.shape[0]), PREFIX + 'persistent_isotropic_tail': float(rows.persistent_isotropic_tail.item()), PREFIX + 'fd_shrinkage': float(rows.fd_shrinkage.item()), PREFIX + 'cross_layer_coupling_ratio': float(selection.cross_layer_coupling_ratio.item()), PREFIX + 'transaction_accepted': int(selection.accepted.item()), PREFIX + 'state_coordinate_count': scaling['persistent_state_elements'], PREFIX + 'ledger_checkpoint_tensor_elements': scaling['ledger_checkpoint_tensor_elements'], PREFIX + 'largest_dense_solve_dimension': _tiller_core__import_MAXIMUM_SELECTION_ROWS, PREFIX + 'largest_transaction_dense_dimension': 32, PREFIX + 'largest_temporal_dense_dimension': _tiller_core__import_MAXIMUM_SELECTION_ROWS, PREFIX + 'dense_lg_metric_elements': 0, PREFIX + 'owner_count': 0, PREFIX + 'selected_update_elements_published': 0, PREFIX + 'new_tunable_hyperparameters': 0})
        self._rfd_decay_derivative = None
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        anchor = self.state[self.pairs[0]['in_weight']]
        scores = anchor.get('factorized_rfd_persistent_scores')
        _previous(anchor, scores if torch.is_tensor(scores) else self.pairs[0]['in_weight'])
        reference = anchor.get('factorized_rfd_reference_row_norm')
        step = anchor.get('factorized_rfd_step')
        if not torch.is_tensor(reference) or reference.numel() != 1 or type(step) is not int or (step < 1):
            raise RuntimeError('TILLER reference checkpoint is absent')
        self._rfd_rows = None
        self._rfd_gradient_scale = None
        self._rfd_selection = None
        self._rfd_decay_derivative = None
        return result

class TILLERAttentionOptimizer(_compiled_chord__FactorizedAdaptiveTangentChordCompiledAttentionOptimizer):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0].pop(_compiled_chord__PREFIX + 'family_id', None)
        self.param_groups[0][PREFIX + 'family_id'] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result['unchanged_compiled_factorized_attention_lr_scale'] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        # Do not let a report from an earlier capture make inherited
        # telemetry-only work execute on this step.
        self._last_telemetry = {}
        loss = super().step(closure)
        self._last_telemetry = _retag(self._last_telemetry)
        if self._last_telemetry:
            self._last_telemetry.update({PREFIX + 'family_id': FAMILY_ID, PREFIX + 'attention_family_id': FAMILY_ID, PREFIX + 'attention_equation_unchanged': 1, PREFIX + 'attention_owner_count': 0, PREFIX + 'attention_selected_update_elements_published': 0})
        return loss
