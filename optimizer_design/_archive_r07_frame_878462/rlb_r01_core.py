"""One cross-layer downstream-loss transaction in the Global-RLB tangent span.

Current R02 produces a complete paired incoming/outgoing direction for every
learned rational group.  R01 keeps those directions and the complete R02
attention transaction.  It uses the aligned per-token downstream cotangent to
represent the loss action of all 18 x 18 group directions in one common basis.

For coordinate ``j=(layer, group)``, let ``Y_j`` be the exact tangent image of
the current RMS-rescaled P5/Q4 response and let ``e_l`` be the cotangent at the
corresponding MLP output.  The per-token score is

    s[n,j] = <e[n,l], Y_j[n]>.

Unlike a layerwise solve, the empirical Fisher ``E[s s^T]`` retains the cross-
layer blocks.  R01 solves one equality-constrained quadratic transaction over
all 324 coordinates.  Its constraint is the literal total squared Frobenius
budget of the complete current-R02 RLB direction.  The scheduled LR and WD are
then applied once by the inherited transaction; every internal LR/WD scale is
one.  The complete R02 direction is feasible, evaluated first, and used on any
failed finiteness, descent, budget, or parent-limit certificate.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_r09_core import R09LossMetricCore


class R01Core(R09LossMetricCore):
    """Current R02 with one global 324-coordinate RLB loss-space solve."""

    component_code = 12
    checkpoint_schema = "r01_global_cross_layer_rlb_metric_v1"
    inherited_parent = "current_r02_response_homotopy_chord"
    new_scientific_components = (
        "global_cross_layer_downstream_loss_metric_transaction",
    )

    def __init__(self, pairs, **kwargs):
        self._r01_global_metadata = None
        super().__init__(pairs, **kwargs)

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "global_cross_layer_rlb_metric_lr_scale": 1.0,
            "global_rational_group_budget_lr_scale": 1.0,
            "cross_layer_fisher_coordinate_lr_scale": 1.0,
            "global_parent_first_transaction_lr_scale": 1.0,
            "global_weight_decay_cross_metric_scale": 1.0,
        })
        return report

    @staticmethod
    def _reduce_global_loss_metric(images, cotangents, group_decay_images):
        """Return the all-rank 324-coordinate score Fisher and WD cross term."""
        if images.ndim != 4 or cotangents.ndim != 3:
            raise RuntimeError("R01 global loss-metric tensor rank changed")
        layers, samples, groups, residual = images.shape
        if (
            cotangents.shape != (layers, samples, residual)
            or group_decay_images.shape != images.shape
        ):
            raise RuntimeError("R01 global loss-metric inventory changed")

        # Rows are the same deterministic token positions in every layer, so
        # flattening layer/group creates the score of one joint model tangent.
        scores = torch.einsum(
            "lngd,lnd->nlg", images, cotangents
        ).reshape(samples, layers * groups)
        decay_scores = torch.einsum(
            "lngd,lnd->nlg", group_decay_images, cotangents
        ).sum(dim=(1, 2))
        fisher_sum = scores.transpose(0, 1) @ scores
        cross_sum = scores.transpose(0, 1) @ decay_scores
        count = torch.tensor(
            float(samples), device=images.device, dtype=images.dtype
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(fisher_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cross_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        torch._assert_async(torch.isfinite(count) & (count > 0.0))
        fisher = fisher_sum / count
        fisher = 0.5 * (fisher + fisher.transpose(-2, -1))
        return fisher.unsqueeze(0), (cross_sum / count).unsqueeze(0), count

    @staticmethod
    def _cross_layer_coupling_ratio(fisher, layers, groups):
        """Measure how much of the fitted metric lies between distinct layers."""
        dimension = int(layers) * int(groups)
        if fisher.shape != (1, dimension, dimension):
            raise RuntimeError("R01 cross-layer Fisher inventory changed")
        coordinates = torch.arange(dimension, device=fisher.device)
        layer_ids = torch.div(coordinates, groups, rounding_mode="floor")
        cross_layer = layer_ids[:, None] != layer_ids[None, :]
        cross_norm = torch.linalg.vector_norm(fisher[0][cross_layer])
        total_norm = torch.linalg.vector_norm(fisher[0])
        return cross_norm / total_norm.clamp_min(torch.finfo(fisher.dtype).tiny)

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
        del (
            incoming_parent,
            outgoing_parent_transpose,
            incoming_parent_descent,
            incoming_endpoint_descent,
            outgoing_parent_descent,
            outgoing_endpoint_descent,
        )
        cotangents = self._r09_loss_cotangents
        if any(value is None for value in (
            functional_inputs,
            functional_preactivations,
            functional_features,
            cotangents,
        )):
            raise RuntimeError("R01 did not receive aligned global loss rows")
        layers = len(self.pairs)
        if force_parent.shape != (layers,):
            raise RuntimeError("R01 parent-limit inventory changed")

        factors = self._functional_jvp_factors(functional_preactivations)
        images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_endpoint,
            outgoing_endpoint_transpose,
            factors=factors,
        )
        weight_decay = float(self.param_groups[0]["weight_decay"])
        group_decay = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            torch.stack(self.incoming).float() * weight_decay,
            torch.stack(self.outgoing).float().transpose(-2, -1) * weight_decay,
            factors=factors,
        )
        fisher, decay_cross, global_count = self._reduce_global_loss_metric(
            images, cotangents, group_decay
        )

        incoming_blocks = incoming_endpoint.view(
            layers, self.groups, self.width, self.external_width
        )
        outgoing_blocks = outgoing_endpoint_transpose.view_as(incoming_blocks)
        incoming_gradients = torch.stack([
            parameter.grad for parameter in self.incoming
        ]).float().view_as(incoming_blocks)
        outgoing_gradients = torch.stack([
            parameter.grad for parameter in self.outgoing
        ]).float().transpose(-2, -1).view_as(outgoing_blocks)
        incoming_momentum = self._current_nesterov_stack(
            self.incoming
        ).view_as(incoming_blocks)
        outgoing_momentum = self._current_nesterov_stack(
            self.outgoing, transpose=True
        ).view_as(outgoing_blocks)

        incoming_exact = (
            incoming_gradients * incoming_blocks
        ).sum(dim=(-2, -1))
        outgoing_exact = (
            outgoing_gradients * outgoing_blocks
        ).sum(dim=(-2, -1))
        incoming_momentum_linear = (
            incoming_momentum * incoming_blocks
        ).sum(dim=(-2, -1))
        outgoing_momentum_linear = (
            outgoing_momentum * outgoing_blocks
        ).sum(dim=(-2, -1))
        budget_weights = (
            incoming_blocks.square().sum(dim=(-2, -1))
            + outgoing_blocks.square().sum(dim=(-2, -1))
        )
        exact_linear = (incoming_exact + outgoing_exact).reshape(1, -1)
        momentum_linear = (
            incoming_momentum_linear + outgoing_momentum_linear
        ).reshape(1, -1)
        flat_budget = budget_weights.reshape(1, -1)

        flat_coefficients, metadata = self._select_group_span_coefficients(
            fisher,
            decay_cross,
            exact_linear,
            momentum_linear,
            flat_budget,
            lr,
        )
        candidate_coefficients = flat_coefficients.view(layers, self.groups)

        # The inherited optimizer certifies descent separately for each matrix
        # role and layer.  Keep that exact feasible cone while solving the
        # cross-layer budget globally.
        candidate_incoming_exact = (
            incoming_exact * candidate_coefficients
        ).sum(dim=-1)
        candidate_outgoing_exact = (
            outgoing_exact * candidate_coefficients
        ).sum(dim=-1)
        candidate_incoming_momentum = (
            incoming_momentum_linear * candidate_coefficients
        ).sum(dim=-1)
        candidate_outgoing_momentum = (
            outgoing_momentum_linear * candidate_coefficients
        ).sum(dim=-1)
        role_descent_valid = (
            (candidate_incoming_exact > 0.0).all()
            & (candidate_outgoing_exact > 0.0).all()
            & (candidate_incoming_momentum > 0.0).all()
            & (candidate_outgoing_momentum > 0.0).all()
        )
        accepted = (
            metadata["accepted"][0]
            & role_descent_valid
            & (~force_parent.any())
        )
        coefficients = torch.where(
            accepted, candidate_coefficients, torch.ones_like(candidate_coefficients)
        )
        incoming_selected = (
            incoming_blocks * coefficients[:, :, None, None]
        ).reshape_as(incoming_endpoint)
        outgoing_selected = (
            outgoing_blocks * coefficients[:, :, None, None]
        ).reshape_as(outgoing_endpoint_transpose)

        parent_exact = exact_linear.sum(dim=-1)
        parent_momentum = momentum_linear.sum(dim=-1)
        metadata["accepted"] = accepted.reshape(1)
        metadata["selected_exact_descent"] = torch.where(
            accepted, metadata["candidate_exact_descent"], parent_exact
        )
        metadata["selected_momentum_descent"] = torch.where(
            accepted, metadata["candidate_momentum_descent"], parent_momentum
        )
        metadata["selected_score"] = torch.where(
            accepted, metadata["candidate_score"], metadata["parent_score"]
        )
        metadata["improvement"] = (
            metadata["parent_score"] - metadata["selected_score"]
        )
        metadata["global_count"] = global_count
        metadata["clip_factor"] = torch.tensor(
            float(self._r09_clip_factor),
            device=global_count.device,
            dtype=global_count.dtype,
        )
        metadata["cross_layer_coupling_ratio"] = (
            self._cross_layer_coupling_ratio(fisher, layers, self.groups)
        )
        metadata["selected_coefficient_min"] = coefficients.amin()
        metadata["selected_coefficient_median"] = coefficients.median()
        metadata["selected_coefficient_max"] = coefficients.amax()
        metadata["selected_incoming_exact_descent_min"] = torch.where(
            accepted,
            candidate_incoming_exact.amin(),
            incoming_exact.sum(dim=-1).amin(),
        )
        metadata["selected_outgoing_exact_descent_min"] = torch.where(
            accepted,
            candidate_outgoing_exact.amin(),
            outgoing_exact.sum(dim=-1).amin(),
        )
        metadata["selected_incoming_momentum_descent_min"] = torch.where(
            accepted,
            candidate_incoming_momentum.amin(),
            incoming_momentum_linear.sum(dim=-1).amin(),
        )
        metadata["selected_outgoing_momentum_descent_min"] = torch.where(
            accepted,
            candidate_outgoing_momentum.amin(),
            outgoing_momentum_linear.sum(dim=-1).amin(),
        )
        self._r01_global_metadata = metadata

        # R09's inherited telemetry publisher expects layer-shaped summaries.
        # These views describe the one global decision without changing it.
        repeated = lambda value: value.reshape(1).expand(layers)
        self._r09_span_metadata = {
            "accepted": repeated(accepted),
            "rank": repeated(metadata["rank"][0]),
            "eigenvalue_max": repeated(metadata["eigenvalue_max"][0]),
            "coefficient_min": repeated(metadata["coefficient_min"][0]),
            "coefficient_median": repeated(metadata["coefficient_median"][0]),
            "coefficient_max": repeated(metadata["coefficient_max"][0]),
            "selected_exact_descent": repeated(
                metadata["selected_exact_descent"][0]
            ),
            "selected_momentum_descent": repeated(
                metadata["selected_momentum_descent"][0]
            ),
            "budget_residual": repeated(metadata["budget_residual"][0]),
            "improvement": repeated(metadata["improvement"][0]),
            "global_count": global_count,
            "clip_factor": metadata["clip_factor"],
        }

        choices = torch.where(
            accepted,
            torch.full((layers,), 3, device=coefficients.device, dtype=torch.int64),
            torch.zeros((layers,), device=coefficients.device, dtype=torch.int64),
        )
        parent_score = metadata["parent_score"][0]
        candidate_score = metadata["candidate_score"][0]
        scores = torch.stack((
            parent_score, candidate_score, parent_score, candidate_score
        )).reshape(1, 4).expand(layers, 4)
        score_margin = (parent_score - candidate_score).abs().expand(layers)
        return incoming_selected, outgoing_selected, {
            "choices": choices,
            "scores": scores,
            "score_margin": score_margin,
            "energies": torch.zeros_like(scores),
            "global_count": global_count,
        }

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r01_global_metadata = None
        loss = super().step(closure)
        metadata = self._r01_global_metadata
        if metadata is None:
            raise RuntimeError("R01 did not execute its global RLB transaction")
        if publish:
            scale_min, scale_max = self._r09_cotangent_scale_range
            self._last_telemetry.update({
                "rlb_r01_component_code": self.component_code,
                "rlb_r01_parent_is_current_r02": 1,
                "rlb_r01_global_cross_layer_metric_enabled": 1,
                "rlb_r01_layer_count": int(len(self.pairs)),
                "rlb_r01_group_count": int(self.groups),
                "rlb_r01_group_width": int(self.width),
                "rlb_r01_global_tangent_dimension": int(
                    len(self.pairs) * self.groups
                ),
                "rlb_r01_global_loss_sample_count": int(
                    metadata["global_count"].item()
                ),
                "rlb_r01_parent_transaction_count": int(
                    (~metadata["accepted"]).sum().item()
                ),
                "rlb_r01_global_metric_transaction_count": int(
                    metadata["accepted"].sum().item()
                ),
                "rlb_r01_fisher_rank": int(metadata["rank"][0].item()),
                "rlb_r01_fisher_eigenvalue_max": float(
                    metadata["eigenvalue_max"][0].item()
                ),
                "rlb_r01_cross_layer_coupling_ratio": float(
                    metadata["cross_layer_coupling_ratio"].item()
                ),
                "rlb_r01_candidate_coefficient_min": float(
                    metadata["coefficient_min"][0].item()
                ),
                "rlb_r01_candidate_coefficient_median": float(
                    metadata["coefficient_median"][0].item()
                ),
                "rlb_r01_candidate_coefficient_max": float(
                    metadata["coefficient_max"][0].item()
                ),
                "rlb_r01_selected_coefficient_min": float(
                    metadata["selected_coefficient_min"].item()
                ),
                "rlb_r01_selected_coefficient_median": float(
                    metadata["selected_coefficient_median"].item()
                ),
                "rlb_r01_selected_coefficient_max": float(
                    metadata["selected_coefficient_max"].item()
                ),
                "rlb_r01_exact_descent": float(
                    metadata["selected_exact_descent"][0].item()
                ),
                "rlb_r01_momentum_descent": float(
                    metadata["selected_momentum_descent"][0].item()
                ),
                "rlb_r01_incoming_exact_descent_min": float(
                    metadata["selected_incoming_exact_descent_min"].item()
                ),
                "rlb_r01_outgoing_exact_descent_min": float(
                    metadata["selected_outgoing_exact_descent_min"].item()
                ),
                "rlb_r01_incoming_momentum_descent_min": float(
                    metadata["selected_incoming_momentum_descent_min"].item()
                ),
                "rlb_r01_outgoing_momentum_descent_min": float(
                    metadata["selected_outgoing_momentum_descent_min"].item()
                ),
                "rlb_r01_budget_residual": float(
                    metadata["budget_residual"][0].item()
                ),
                "rlb_r01_surrogate_improvement": float(
                    metadata["improvement"][0].item()
                ),
                "rlb_r01_realized_clip_factor": float(
                    metadata["clip_factor"].item()
                ),
                "rlb_r01_per_token_cotangent_scale_min": scale_min,
                "rlb_r01_per_token_cotangent_scale_max": scale_max,
                "rlb_r01_structural_matrix_elements": 245_366_784,
            })
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r01_global_metadata = None
        return result
