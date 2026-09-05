"""Periodic R01 global-allocation refresh for the fast row-polar path.

This opt-in approximation executes the full 324-coordinate downstream-loss
Fisher transaction once every eight optimizer steps.  Between refreshes it
reuses the last selected coefficient pattern, rescales that pattern to the
*current* paired Frobenius budget, and admits it only when every layer/role
retains positive exact-gradient and Nesterov action.  Any failed certificate
falls back to the literal all-ones parent coefficients.

The response router, endpoint construction, momentum, LR, WD, and current
budget/descent certificates still execute every step.  This is a numerical
method variant and needs its own quality trajectory.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_stale_metric import (
    METRIC_REFRESH_INTERVAL,
    R01StaleMetric8RowOptimizer,
    R02StaleMetric8RowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
ALLOCATION_REFRESH_INTERVAL = 8
STALE_ALLOCATION_FAMILY_ID = "r01_9150_stale8_metric_and_allocation_row_v1"


class _PeriodicGlobalAllocationMixin:
    checkpoint_schema = "r01_stale8_metric_and_allocation_row_v1"

    def __init__(self, pairs, **kwargs):
        self._allocation_refresh_step = 0
        self._capture_full_allocation_this_step = True
        self._cached_allocation_coefficients = None
        self._cached_allocation_metadata = None
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_allocation_refresh_interval"] = ALLOCATION_REFRESH_INTERVAL
        group["r01_stale_allocation_family_id"] = STALE_ALLOCATION_FAMILY_ID

    def _recover_selected_coefficients(
        self,
        incoming_endpoint,
        outgoing_endpoint_transpose,
        incoming_selected,
        outgoing_selected,
    ):
        layers = len(self.pairs)
        shape = (layers, self.groups, self.width, self.external_width)
        endpoint_a = incoming_endpoint.view(shape)
        endpoint_c = outgoing_endpoint_transpose.view(shape)
        selected_a = incoming_selected.view(shape)
        selected_c = outgoing_selected.view(shape)
        numerator = (
            (endpoint_a * selected_a).sum(dim=(-2, -1))
            + (endpoint_c * selected_c).sum(dim=(-2, -1))
        )
        denominator = (
            endpoint_a.square().sum(dim=(-2, -1))
            + endpoint_c.square().sum(dim=(-2, -1))
        )
        tiny = torch.finfo(denominator.dtype).tiny
        torch._assert_async(torch.isfinite(denominator).all())
        torch._assert_async((denominator > tiny).all())
        coefficients = numerator / denominator.clamp_min(tiny)
        torch._assert_async(torch.isfinite(coefficients).all())
        return coefficients

    def _stale_allocation(
        self,
        incoming_endpoint,
        outgoing_endpoint_transpose,
        *,
        force_parent,
    ):
        cached = self._cached_allocation_coefficients
        template = self._cached_allocation_metadata
        if cached is None or template is None:
            raise RuntimeError("stale R01 allocation was not initialized")
        layers = len(self.pairs)
        shape = (layers, self.groups, self.width, self.external_width)
        incoming_blocks = incoming_endpoint.view(shape)
        outgoing_blocks = outgoing_endpoint_transpose.view(shape)
        incoming_gradients = torch.stack([
            parameter.grad for parameter in self.incoming
        ]).float().view(shape)
        outgoing_gradients = torch.stack([
            parameter.grad for parameter in self.outgoing
        ]).float().transpose(-2, -1).view(shape)
        incoming_momentum = self._current_nesterov_stack(
            self.incoming
        ).view(shape)
        outgoing_momentum = self._current_nesterov_stack(
            self.outgoing, transpose=True
        ).view(shape)

        incoming_exact = (incoming_gradients * incoming_blocks).sum(
            dim=(-2, -1)
        )
        outgoing_exact = (outgoing_gradients * outgoing_blocks).sum(
            dim=(-2, -1)
        )
        incoming_momentum_linear = (incoming_momentum * incoming_blocks).sum(
            dim=(-2, -1)
        )
        outgoing_momentum_linear = (outgoing_momentum * outgoing_blocks).sum(
            dim=(-2, -1)
        )
        budget_weights = (
            incoming_blocks.square().sum(dim=(-2, -1))
            + outgoing_blocks.square().sum(dim=(-2, -1))
        )
        parent_budget = budget_weights.sum()
        raw_budget = (budget_weights * cached.square()).sum()
        tiny = torch.finfo(budget_weights.dtype).tiny
        scale = torch.sqrt(parent_budget / raw_budget.clamp_min(tiny))
        candidate = cached * scale
        candidate_budget = (budget_weights * candidate.square()).sum()
        budget_residual = (
            (candidate_budget - parent_budget).abs()
            / parent_budget.clamp_min(tiny)
        )

        candidate_incoming_exact = (incoming_exact * candidate).sum(dim=-1)
        candidate_outgoing_exact = (outgoing_exact * candidate).sum(dim=-1)
        candidate_incoming_momentum = (
            incoming_momentum_linear * candidate
        ).sum(dim=-1)
        candidate_outgoing_momentum = (
            outgoing_momentum_linear * candidate
        ).sum(dim=-1)
        finite = (
            torch.isfinite(candidate).all()
            & torch.isfinite(budget_residual)
            & torch.isfinite(candidate_incoming_exact).all()
            & torch.isfinite(candidate_outgoing_exact).all()
            & torch.isfinite(candidate_incoming_momentum).all()
            & torch.isfinite(candidate_outgoing_momentum).all()
        )
        role_descent_valid = (
            (candidate_incoming_exact > 0.0).all()
            & (candidate_outgoing_exact > 0.0).all()
            & (candidate_incoming_momentum > 0.0).all()
            & (candidate_outgoing_momentum > 0.0).all()
        )
        budget_tolerance = (
            2048.0 * torch.finfo(budget_weights.dtype).eps
        )
        accepted = (
            finite
            & role_descent_valid
            & (budget_residual <= budget_tolerance)
            & (~force_parent.any())
        )
        coefficients = torch.where(
            accepted, candidate, torch.ones_like(candidate)
        )
        incoming_selected = (
            incoming_blocks * coefficients[:, :, None, None]
        ).reshape_as(incoming_endpoint)
        outgoing_selected = (
            outgoing_blocks * coefficients[:, :, None, None]
        ).reshape_as(outgoing_endpoint_transpose)

        selected_incoming_exact = torch.where(
            accepted,
            candidate_incoming_exact,
            incoming_exact.sum(dim=-1),
        )
        selected_outgoing_exact = torch.where(
            accepted,
            candidate_outgoing_exact,
            outgoing_exact.sum(dim=-1),
        )
        selected_incoming_momentum = torch.where(
            accepted,
            candidate_incoming_momentum,
            incoming_momentum_linear.sum(dim=-1),
        )
        selected_outgoing_momentum = torch.where(
            accepted,
            candidate_outgoing_momentum,
            outgoing_momentum_linear.sum(dim=-1),
        )
        selected_exact = (
            selected_incoming_exact.sum() + selected_outgoing_exact.sum()
        ).reshape(1)
        selected_momentum = (
            selected_incoming_momentum.sum()
            + selected_outgoing_momentum.sum()
        ).reshape(1)
        current_budget_residual = torch.where(
            accepted,
            budget_residual,
            torch.zeros_like(budget_residual),
        ).reshape(1)
        global_count = template["global_count"].clone()
        clip_factor = torch.tensor(
            float(self._r09_clip_factor),
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        zero = torch.zeros((1,), device=coefficients.device, dtype=coefficients.dtype)
        metadata = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in template.items()
        }
        metadata.update({
            "accepted": accepted.reshape(1),
            "selected_exact_descent": selected_exact,
            "selected_momentum_descent": selected_momentum,
            "budget_residual": current_budget_residual,
            "improvement": zero,
            "global_count": global_count,
            "clip_factor": clip_factor,
            "selected_coefficient_min": coefficients.amin(),
            "selected_coefficient_median": coefficients.median(),
            "selected_coefficient_max": coefficients.amax(),
            "selected_incoming_exact_descent_min": selected_incoming_exact.amin(),
            "selected_outgoing_exact_descent_min": selected_outgoing_exact.amin(),
            "selected_incoming_momentum_descent_min": (
                selected_incoming_momentum.amin()
            ),
            "selected_outgoing_momentum_descent_min": (
                selected_outgoing_momentum.amin()
            ),
        })
        self._r01_global_metadata = metadata
        repeated = lambda value: value.reshape(1).expand(layers)
        self._r09_span_metadata = {
            "accepted": repeated(accepted),
            "rank": repeated(metadata["rank"][0]),
            "eigenvalue_max": repeated(metadata["eigenvalue_max"][0]),
            "coefficient_min": repeated(coefficients.amin()),
            "coefficient_median": repeated(coefficients.median()),
            "coefficient_max": repeated(coefficients.amax()),
            "selected_exact_descent": repeated(selected_exact[0]),
            "selected_momentum_descent": repeated(selected_momentum[0]),
            "budget_residual": repeated(current_budget_residual[0]),
            "improvement": repeated(zero[0]),
            "global_count": global_count,
            "clip_factor": clip_factor,
        }
        choices = torch.where(
            accepted,
            torch.full(
                (layers,), 3, device=coefficients.device, dtype=torch.int64
            ),
            torch.zeros((layers,), device=coefficients.device, dtype=torch.int64),
        )
        parent_score = metadata["parent_score"][0]
        candidate_score = metadata["candidate_score"][0]
        scores = torch.stack((
            parent_score, candidate_score, parent_score, candidate_score
        )).reshape(1, 4).expand(layers, 4)
        return incoming_selected, outgoing_selected, {
            "choices": choices,
            "scores": scores,
            "score_margin": torch.zeros(
                (layers,), device=coefficients.device, dtype=coefficients.dtype
            ),
            "energies": torch.zeros_like(scores),
            "global_count": global_count,
        }

    def _select_functional_corner(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        incoming_parent,
        incoming_endpoint,
        outgoing_parent_transpose,
        outgoing_endpoint_transpose,
        incoming_parent_descent,
        incoming_endpoint_descent,
        outgoing_parent_descent,
        outgoing_endpoint_descent,
        lr,
        *,
        force_parent,
    ):
        if self._capture_full_allocation_this_step:
            selected_a, selected_c, packet = super()._select_functional_corner(
                functional_inputs,
                functional_preactivations,
                functional_features,
                incoming_parent,
                incoming_endpoint,
                outgoing_parent_transpose,
                outgoing_endpoint_transpose,
                incoming_parent_descent,
                incoming_endpoint_descent,
                outgoing_parent_descent,
                outgoing_endpoint_descent,
                lr,
                force_parent=force_parent,
            )
            self._cached_allocation_coefficients = (
                self._recover_selected_coefficients(
                    incoming_endpoint,
                    outgoing_endpoint_transpose,
                    selected_a,
                    selected_c,
                ).detach().clone()
            )
            self._cached_allocation_metadata = {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in self._r01_global_metadata.items()
            }
            return selected_a, selected_c, packet
        return self._stale_allocation(
            incoming_endpoint,
            outgoing_endpoint_transpose,
            force_parent=force_parent,
        )

    @torch.no_grad()
    def step(self, closure=None):
        refresh = self._capture_full_allocation_this_step
        result = super().step(closure)
        self._allocation_refresh_step += 1
        self._capture_full_allocation_this_step = (
            self._allocation_refresh_step % ALLOCATION_REFRESH_INTERVAL == 0
        )
        if refresh and (
            self._cached_allocation_coefficients is None
            or self._cached_allocation_metadata is None
        ):
            raise RuntimeError("R01 allocation refresh did not populate its cache")
        return result

    def stale_allocation_runtime_report(self):
        return {
            "family_id": STALE_ALLOCATION_FAMILY_ID,
            "metric_refresh_interval": METRIC_REFRESH_INTERVAL,
            "allocation_refresh_interval": ALLOCATION_REFRESH_INTERVAL,
            "current_budget_closed_every_step": True,
            "current_exact_and_nesterov_descent_gated_every_step": True,
            "response_router_every_step": True,
            "lr_or_wd_changed": False,
        }


class R01StaleMetricAllocation8RowOptimizer(
    _PeriodicGlobalAllocationMixin,
    R01StaleMetric8RowOptimizer,
):
    pass


R02StaleMetricAllocation8RowAttentionOptimizer = (
    R02StaleMetric8RowAttentionOptimizer
)


__all__ = (
    "ALLOCATION_REFRESH_INTERVAL",
    "ARCHIVE_CERTIFICATE",
    "R01StaleMetricAllocation8RowOptimizer",
    "R02StaleMetricAllocation8RowAttentionOptimizer",
    "STALE_ALLOCATION_FAMILY_ID",
)
