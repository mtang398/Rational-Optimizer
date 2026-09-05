"""Owner-free rank-ten trust allocation from Global-RLB coefficient gradients.

The installed Global-RLB P5/Q4 function has six numerator and four
denominator coefficients per rational group.  Their ordinary backward
gradients are exact loss-cotangent contractions against those ten basis
channels.  This module uses those already-computed contractions as a
deterministic ``10 x (L G)`` score sketch.

Only row-space systems are solved.  No ``(L G) x (L G)`` tensor is formed and
the method has no activation-position-dependent state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rlb_dual_loss_metric import solve_equality_sphere_from_scores


FAMILY_ID = "basis_cotangent_trust_muon_v1"
NUMERATOR_CHANNELS = 6
DENOMINATOR_CHANNELS = 4
RATIONAL_CHANNELS = NUMERATOR_CHANNELS + DENOMINATOR_CHANNELS


@dataclass(frozen=True)
class BasisCotangentTrustSelection:
    coefficients: torch.Tensor
    candidate_coefficients: torch.Tensor
    accepted: torch.Tensor
    parent_score: torch.Tensor
    candidate_score: torch.Tensor
    budget_residual: torch.Tensor
    exact_descent: torch.Tensor
    momentum_descent: torch.Tensor
    rank: torch.Tensor
    multiplier: torch.Tensor
    calibration: torch.Tensor
    sketch_row_norm_min: torch.Tensor
    sketch_row_norm_max: torch.Tensor


def _validate_inputs(
    exact_by_role: torch.Tensor,
    momentum_by_role: torch.Tensor,
    weights: torch.Tensor,
    coefficient_cotangents: torch.Tensor,
    eta: float,
) -> tuple[int, int, int]:
    if exact_by_role.ndim != 3:
        raise RuntimeError("exact role contractions must be [roles,layers,groups]")
    roles, layers, groups = map(int, exact_by_role.shape)
    if roles < 1 or layers < 1 or groups < 1:
        raise RuntimeError("basis-cotangent allocation received an empty lattice")
    if momentum_by_role.shape != exact_by_role.shape:
        raise RuntimeError("exact and momentum role inventories differ")
    if weights.shape != (layers, groups):
        raise RuntimeError("basis-cotangent budget inventory changed")
    if coefficient_cotangents.shape != (layers, groups, RATIONAL_CHANNELS):
        raise RuntimeError("basis-cotangent P5/Q4 inventory changed")
    tensors = (
        momentum_by_role,
        weights,
        coefficient_cotangents,
    )
    if any(
        value.dtype != exact_by_role.dtype or value.device != exact_by_role.device
        for value in tensors
    ):
        raise RuntimeError("basis-cotangent dtype or device inventories differ")
    if not exact_by_role.is_floating_point():
        raise RuntimeError("basis-cotangent inputs must be floating point")
    if not (float(eta) > 0.0):
        raise RuntimeError("basis-cotangent allocation requires positive LR")
    return roles, layers, groups


def _fallback(
    exact_by_role: torch.Tensor,
    momentum_by_role: torch.Tensor,
    weights: torch.Tensor,
    *,
    row_norm_min: torch.Tensor,
    row_norm_max: torch.Tensor,
) -> BasisCotangentTrustSelection:
    layers, groups = map(int, weights.shape)
    one = torch.ones((layers, groups), device=weights.device, dtype=weights.dtype)
    exact = exact_by_role.sum().reshape(1)
    momentum = momentum_by_role.sum().reshape(1)
    zero = torch.zeros((1,), device=weights.device, dtype=weights.dtype)
    return BasisCotangentTrustSelection(
        coefficients=one,
        candidate_coefficients=one,
        accepted=torch.zeros((1,), device=weights.device, dtype=torch.bool),
        parent_score=zero,
        candidate_score=zero,
        budget_residual=zero,
        exact_descent=exact,
        momentum_descent=momentum,
        rank=torch.zeros((1,), device=weights.device, dtype=torch.int64),
        multiplier=zero,
        calibration=zero,
        sketch_row_norm_min=row_norm_min.reshape(1),
        sketch_row_norm_max=row_norm_max.reshape(1),
    )


def basis_cotangent_trust_allocate(
    exact_by_role: torch.Tensor,
    momentum_by_role: torch.Tensor,
    weights: torch.Tensor,
    coefficient_cotangents: torch.Tensor,
    *,
    eta: float,
    rounds: int = 64,
) -> BasisCotangentTrustSelection:
    """Select one same-energy coefficient per ``(layer, group)`` direction.

    ``rounds`` controls only the deterministic FP64 secular solve.  Training
    semantics use the fixed value 64, which drives the bisection error below
    the floating-point certificate tolerance; it is not an optimizer tuning
    parameter.
    """

    roles, layers, groups = _validate_inputs(
        exact_by_role,
        momentum_by_role,
        weights,
        coefficient_cotangents,
        eta,
    )
    del roles
    if rounds != 64:
        raise RuntimeError("basis-cotangent v1 requires the fixed FP64 solve")

    # The decision system is deliberately tiny and evaluated in FP64.  Matrix
    # directions stay in their native dtype and are never published.
    exact64 = exact_by_role.double()
    momentum64 = momentum_by_role.double()
    weight64 = weights.double()
    cotangent64 = coefficient_cotangents.double()
    finite = (
        torch.isfinite(exact64).all()
        & torch.isfinite(momentum64).all()
        & torch.isfinite(weight64).all()
        & torch.isfinite(cotangent64).all()
        & (weight64 > 0.0).all()
    )

    coordinates = layers * groups
    raw_rows = cotangent64.permute(2, 0, 1).reshape(
        RATIONAL_CHANNELS, coordinates
    )
    row_norms = torch.linalg.vector_norm(raw_rows, dim=-1)
    row_norm_min = row_norms.amin()
    row_norm_max = row_norms.amax()
    machine_tiny = torch.finfo(torch.float64).tiny
    nonzero_rows = row_norms > machine_tiny
    normalized_rows = torch.where(
        nonzero_rows[:, None],
        raw_rows / row_norms.clamp_min(machine_tiny)[:, None],
        torch.zeros_like(raw_rows),
    )

    exact = exact64.sum(dim=0).reshape(-1)
    momentum = momentum64.sum(dim=0).reshape(-1)
    weight = weight64.reshape(-1)
    parent = torch.ones_like(exact)
    parent_linear = exact.sum()
    parent_response = normalized_rows @ parent
    raw_parent_curvature = parent_response.square().mean()
    valid = bool(
        finite
        & nonzero_rows.any()
        & torch.isfinite(parent_linear)
        & (parent_linear > 0.0)
        & torch.isfinite(raw_parent_curvature)
        & (raw_parent_curvature > machine_tiny)
    )
    if not valid:
        return _fallback(
            exact_by_role,
            momentum_by_role,
            weights,
            row_norm_min=row_norm_min.to(weights.dtype),
            row_norm_max=row_norm_max.to(weights.dtype),
        )

    # Radial calibration is algebraic: at alpha=1 the first derivative of
    # -eta b^T alpha + eta^2 ||S alpha||^2/(2K) is zero in the parent radial
    # direction.  No damping or blend coefficient is introduced.
    calibration = torch.sqrt(
        parent_linear / (float(eta) * raw_parent_curvature)
    )
    score_rows = normalized_rows * calibration
    inverse_root_weight = torch.rsqrt(weight)
    whitened_scores = score_rows * inverse_root_weight[None]
    rhs = (exact / float(eta)) * inverse_root_weight
    parent_coordinates = torch.sqrt(weight)
    budget = weight.sum()
    solve = solve_equality_sphere_from_scores(
        whitened_scores.unsqueeze(0),
        rhs.unsqueeze(0),
        parent_coordinates.unsqueeze(0),
        budget.reshape(1),
        rounds=rounds,
    )
    candidate = solve.coordinates[0] * inverse_root_weight

    def objective(coefficients: torch.Tensor) -> torch.Tensor:
        response = score_rows @ coefficients
        return (
            -float(eta) * torch.dot(exact, coefficients)
            + 0.5 * float(eta) ** 2 * response.square().mean()
        )

    parent_score = objective(parent)
    candidate_score = objective(candidate)
    candidate_exact_by_role = (
        exact64 * candidate.view(1, layers, groups)
    ).sum(dim=-1)
    candidate_momentum_by_role = (
        momentum64 * candidate.view(1, layers, groups)
    ).sum(dim=-1)
    parent_exact_by_role = exact64.sum(dim=-1)
    parent_momentum_by_role = momentum64.sum(dim=-1)
    exact_descent = candidate_exact_by_role.sum()
    momentum_descent = candidate_momentum_by_role.sum()
    candidate_budget = torch.dot(weight, candidate.square())
    budget_residual = (candidate_budget - budget).abs() / budget.clamp_min(1.0)
    tolerance = 1.0e-8
    accepted = bool(
        torch.isfinite(candidate).all()
        & torch.isfinite(candidate_score)
        & torch.isfinite(exact_descent)
        & torch.isfinite(momentum_descent)
        & (candidate_score < parent_score)
        & (budget_residual <= tolerance)
        & (candidate_exact_by_role > 0.0).all()
        & (candidate_momentum_by_role > 0.0).all()
        & (parent_exact_by_role > 0.0).all()
        & (parent_momentum_by_role > 0.0).all()
    )
    accepted_tensor = torch.tensor(
        [accepted], device=weights.device, dtype=torch.bool
    )
    selected = torch.where(accepted_tensor, candidate, parent).view(
        layers, groups
    )
    return BasisCotangentTrustSelection(
        coefficients=selected.to(weights.dtype),
        candidate_coefficients=candidate.view(layers, groups).to(weights.dtype),
        accepted=accepted_tensor,
        parent_score=parent_score.reshape(1).to(weights.dtype),
        candidate_score=torch.where(
            accepted_tensor,
            candidate_score.reshape(1),
            parent_score.reshape(1),
        ).to(weights.dtype),
        budget_residual=torch.where(
            accepted_tensor,
            budget_residual.reshape(1),
            torch.zeros_like(budget_residual).reshape(1),
        ).to(weights.dtype),
        exact_descent=torch.where(
            accepted_tensor,
            exact_descent.reshape(1),
            exact.sum().reshape(1),
        ).to(weights.dtype),
        momentum_descent=torch.where(
            accepted_tensor,
            momentum_descent.reshape(1),
            momentum.sum().reshape(1),
        ).to(weights.dtype),
        rank=solve.rank,
        multiplier=solve.multiplier.to(weights.dtype),
        calibration=calibration.reshape(1).to(weights.dtype),
        sketch_row_norm_min=row_norm_min.reshape(1).to(weights.dtype),
        sketch_row_norm_max=row_norm_max.reshape(1).to(weights.dtype),
    )


__all__ = (
    "BasisCotangentTrustSelection",
    "DENOMINATOR_CHANNELS",
    "FAMILY_ID",
    "NUMERATOR_CHANNELS",
    "RATIONAL_CHANNELS",
    "basis_cotangent_trust_allocate",
)

