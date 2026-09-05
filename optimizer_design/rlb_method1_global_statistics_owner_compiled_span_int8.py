"""Compiled exact secular bisection for streamed global-owner Method1.

The qualified group-span trust-region solve performs a fixed 64-round
bisection on small `[local_layer, 18]` tensors.  Eager execution launches the
same elementwise kernels 64 times for each of three nested solves.  This
module preserves all 64 rounds and every surrounding equation, but captures
the fixed bisection as one compiled CUDA program.  Newton--Schulz, cadence,
owner partition, LR, WD, and the block-256 INT8 wire are unchanged.
"""

from __future__ import annotations

import torch

from . import rlb_method1_local_layer_owner as _owner_module
from .rlb_method1_global_statistics_owner_int8 import (
    _CONSTRUCTION_LOCK,
    Method1GlobalStatisticsOwnerInt8Composite,
)
from .rlb_method1_global_statistics_owner_linear_images_int8 import (
    Method1GlobalStatisticsOwnerLinearImagesInt8Composite,
    _LinearImageReuseGlobalStatisticsOwnerRouter,
)
from .rlb_method1_local_layer_owner_int8_direct import (
    Method1LocalLayerOwnerInt8DirectComposite,
)
from .rlb_recursive_inverse_numerics import Method1RecursiveInverseRouter


FAMILY_ID = "method1_global_owner_linear_images_compiled_span64_int8_v1"
BISECTION_ROUNDS = 64


def _secular_bisection_program(
    eigenvalues,
    coordinates,
    parent_budget,
    rank_threshold,
    lower,
    upper,
):
    """Literal fixed-round parent bisection, isolated for CUDA compilation."""
    for _ in range(BISECTION_ROUNDS):
        middle = 0.5 * (lower + upper)
        trial = coordinates / (
            eigenvalues + middle[:, None]
        ).clamp_min(rank_threshold[:, None])
        too_large = trial.square().sum(dim=-1) > parent_budget
        lower = torch.where(too_large, middle, lower)
        upper = torch.where(too_large, upper, middle)
    return upper


_compiled_secular_bisection = torch.compile(
    _secular_bisection_program, fullgraph=True, dynamic=False
)


def _select_group_span_impl(
    cls,
    fisher,
    decay_cross,
    exact_linear,
    momentum_linear,
    budget_weights,
    eta,
    *,
    root_solver,
):
    """Parent group-span solve with only its bisection executor injected."""
    if (
        fisher.ndim != 3
        or decay_cross.shape != exact_linear.shape
        or exact_linear.shape != momentum_linear.shape
        or exact_linear.shape != budget_weights.shape
        or fisher.shape[:2] != exact_linear.shape
        or fisher.shape[-1] != exact_linear.shape[-1]
    ):
        raise RuntimeError("compiled Method1 group-span inventory changed")
    if float(eta) <= 0.0:
        raise RuntimeError("compiled Method1 group-span received nonpositive LR")
    _layers, groups = exact_linear.shape
    machine = torch.finfo(fisher.dtype).eps
    tiny = torch.finfo(fisher.dtype).tiny
    parent_budget = budget_weights.sum(dim=-1)
    valid_weights = (
        torch.isfinite(budget_weights).all(dim=-1)
        & (budget_weights > 0.0).all(dim=-1)
        & (parent_budget > 0.0)
    )
    torch._assert_async(valid_weights.all())

    inverse_root_weight = torch.rsqrt(budget_weights)
    whitened = (
        fisher
        * inverse_root_weight[:, :, None]
        * inverse_root_weight[:, None, :]
    )
    whitened = 0.5 * (whitened + whitened.transpose(-2, -1))
    rhs = (exact_linear / float(eta) - decay_cross) * inverse_root_weight
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    spectral_scale = eigenvalues.abs().amax(dim=-1)
    rank_threshold = machine * float(groups) * spectral_scale.clamp_min(tiny)
    retained = eigenvalues > rank_threshold[:, None]
    coordinates = torch.einsum("lgi,lg->li", eigenvectors, rhs)

    minimum = eigenvalues[:, 0]
    lower = -minimum + rank_threshold
    upper = (
        torch.linalg.vector_norm(coordinates, dim=-1)
        / torch.sqrt(parent_budget).clamp_min(tiny)
        + spectral_scale
        + rank_threshold
    )
    lower_values = coordinates / (
        eigenvalues + lower[:, None]
    ).clamp_min(rank_threshold[:, None])
    hard_case = lower_values.square().sum(dim=-1) < parent_budget
    upper = root_solver(
        eigenvalues,
        coordinates,
        parent_budget,
        rank_threshold,
        lower,
        upper,
    )
    root_coordinates = coordinates / (
        eigenvalues + upper[:, None]
    ).clamp_min(rank_threshold[:, None])

    separation = eigenvalues - minimum[:, None]
    minimum_mask = separation <= rank_threshold[:, None]
    hard_base = torch.where(
        minimum_mask,
        torch.zeros_like(coordinates),
        coordinates / separation.clamp_min(rank_threshold[:, None]),
    )
    remaining = (
        parent_budget - hard_base.square().sum(dim=-1)
    ).clamp_min(0.0)
    first_minimum = torch.argmax(minimum_mask.to(torch.int64), dim=-1)
    parent_q = torch.sqrt(budget_weights)
    parent_coordinates = torch.einsum("lgi,lg->li", eigenvectors, parent_q)
    signs = torch.sign(
        parent_coordinates.gather(1, first_minimum[:, None]).squeeze(1)
    )
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    hard_fill = torch.zeros_like(hard_base).scatter(
        1,
        first_minimum[:, None],
        (torch.sqrt(remaining) * signs)[:, None],
    )
    selected_coordinates = torch.where(
        hard_case[:, None], hard_base + hard_fill, root_coordinates
    )
    candidate = torch.einsum(
        "lgi,li->lg", eigenvectors, selected_coordinates
    ) * inverse_root_weight

    parent = torch.ones_like(candidate)
    common_positive = (
        (candidate == candidate[:, :1]).all(dim=-1)
        & (candidate[:, 0] > 0.0)
    )
    candidate = torch.where(common_positive[:, None], parent, candidate)
    candidate_budget = (budget_weights * candidate.square()).sum(dim=-1)
    budget_residual = (
        (candidate_budget - parent_budget).abs()
        / parent_budget.clamp_min(1.0)
    )
    parent_score = cls._quadratic_scores(
        parent, fisher, decay_cross, exact_linear, eta
    )
    candidate_score = cls._quadratic_scores(
        candidate, fisher, decay_cross, exact_linear, eta
    )
    candidate_exact_descent = (exact_linear * candidate).sum(dim=-1)
    candidate_momentum_descent = (momentum_linear * candidate).sum(dim=-1)
    finite = (
        torch.isfinite(fisher).all(dim=(-2, -1))
        & torch.isfinite(decay_cross).all(dim=-1)
        & torch.isfinite(exact_linear).all(dim=-1)
        & torch.isfinite(momentum_linear).all(dim=-1)
        & torch.isfinite(candidate).all(dim=-1)
        & torch.isfinite(parent_score)
        & torch.isfinite(candidate_score)
    )
    valid = (
        finite
        & valid_weights
        & (candidate_exact_descent > 0.0)
        & (candidate_momentum_descent > 0.0)
        & (budget_residual <= 2048.0 * machine)
    )
    accepted = valid & (candidate_score < parent_score)
    selected = torch.where(accepted[:, None], candidate, parent)
    selected_score = torch.where(accepted, candidate_score, parent_score)
    parent_exact_descent = exact_linear.sum(dim=-1)
    parent_momentum_descent = momentum_linear.sum(dim=-1)
    return selected, {
        "accepted": accepted,
        "rank": retained.sum(dim=-1),
        "rank_threshold": rank_threshold,
        "eigenvalue_max": spectral_scale,
        "coefficient_min": candidate.amin(dim=-1),
        "coefficient_median": candidate.median(dim=-1).values,
        "coefficient_max": candidate.amax(dim=-1),
        "candidate_exact_descent": candidate_exact_descent,
        "candidate_momentum_descent": candidate_momentum_descent,
        "selected_exact_descent": torch.where(
            accepted, candidate_exact_descent, parent_exact_descent
        ),
        "selected_momentum_descent": torch.where(
            accepted, candidate_momentum_descent, parent_momentum_descent
        ),
        "budget_residual": budget_residual,
        "parent_score": parent_score,
        "candidate_score": candidate_score,
        "selected_score": selected_score,
        "improvement": parent_score - selected_score,
    }


class _CompiledSpan64Mixin:
    @classmethod
    def _select_group_span_coefficients(
        cls,
        fisher,
        decay_cross,
        exact_linear,
        momentum_linear,
        budget_weights,
        eta,
    ):
        solver = (
            _compiled_secular_bisection
            if fisher.is_cuda
            else _secular_bisection_program
        )
        return _select_group_span_impl(
            cls,
            fisher,
            decay_cross,
            exact_linear,
            momentum_linear,
            budget_weights,
            eta,
            root_solver=solver,
        )


class _CompiledSpanGlobalStatisticsOwnerRouter(
    _CompiledSpan64Mixin,
    _LinearImageReuseGlobalStatisticsOwnerRouter,
):
    checkpoint_schema = FAMILY_ID + "_router"


class Method1GlobalStatisticsOwnerCompiledSpanInt8Composite(
    Method1GlobalStatisticsOwnerLinearImagesInt8Composite
):
    """Linear-image Method1 with the exact 64-round solve launch-fused."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        router_kwargs = {
            key: kwargs[key]
            for key in ("lr", "weight_decay", "momentum", "ns_steps", "beta2", "eps")
        }
        with _CONSTRUCTION_LOCK:
            original = _owner_module.Method1RecursiveInverseRouter
            if original is not Method1RecursiveInverseRouter:
                raise RuntimeError("Method1 owner router constructor was already patched")
            _owner_module.Method1RecursiveInverseRouter = (
                _CompiledSpanGlobalStatisticsOwnerRouter
            )
            try:
                Method1LocalLayerOwnerInt8DirectComposite.__init__(
                    self, blocks, adamw, **kwargs
                )
            finally:
                _owner_module.Method1RecursiveInverseRouter = original
        if not isinstance(self.router, _CompiledSpanGlobalStatisticsOwnerRouter):
            raise RuntimeError("compiled-span global owner was not installed")

        self.capture_broker = Method1RecursiveInverseRouter(
            self.all_blocks, **router_kwargs
        )
        self._owner_original_probe_count = int(self.router.probe_count)
        self._owner_original_input_capture_count = int(
            self.router.input_capture_count
        )
        self._last_global_functional_rows = 0
        self._last_global_response_rows = 0
        self._last_global_input_rows = 0
        self._last_global_feature_samples = 0
        self._sync_capture_plan()

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "group_span_bisection_rounds": BISECTION_ROUNDS,
            "group_span_bisection_executor": "one_compiled_fixed_shape_program",
            "group_span_equations_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "floating_point_association_may_change": True,
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "BISECTION_ROUNDS",
    "FAMILY_ID",
    "Method1GlobalStatisticsOwnerCompiledSpanInt8Composite",
    "_CompiledSpanGlobalStatisticsOwnerRouter",
    "_secular_bisection_program",
    "_select_group_span_impl",
)
