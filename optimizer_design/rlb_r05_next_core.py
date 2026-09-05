"""Paired hidden-coordinate magnitude geometry on complete R01.

Global-RLB gives every hidden coordinate one incoming row and one outgoing
column around the same learned grouped P5/Q4 response.  This core promotes the
*shared* radial motion of that exact pair to an optimizer coordinate.  Its
bias-corrected first/second moments use the immutable AdamW betas and epsilon,
but no separate LR or WD.  The resulting paired-radial direction is made
orthogonal to the complete R01 group direction and matched to its Frobenius
budget.

For each layer/group, ``P`` is the complete R01 group direction and ``M`` is
the equal-budget orthogonal paired-magnitude direction.  The two axes

    C+ = (P + M) / 2,   C- = (P - M) / 2

are orthogonal and each has half of ``P``'s squared norm.  Therefore the
all-ones coefficient vector is literal R01 and the inherited global
downstream-loss trust-region solver can choose all 648 coefficients while
preserving exactly R01's total matrix-update energy.  Fixed decoupled WD is
modeled once and applied once by the inherited transaction.
"""

from __future__ import annotations

import torch

from .rlb_r01_core import R01Core


class R05NextCore(R01Core):
    """Complete R01 plus a paired RLB hidden-magnitude functional atlas."""

    component_code = 14
    checkpoint_schema = "r05_r01_paired_hidden_magnitude_atlas_v1"
    inherited_parent = "current_r01_global_cross_layer_rlb_metric"
    new_scientific_components = (
        "paired_hidden_coordinate_magnitude_functional_atlas",
    )

    def __init__(self, pairs, **kwargs):
        self._r05_next_metadata = None
        super().__init__(pairs, **kwargs)
        self._r05_beta1 = 0.90
        self._r05_beta2 = 0.95
        self._r05_eps = 1.0e-8

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "paired_hidden_magnitude_lr_scale": 1.0,
            "paired_hidden_magnitude_weight_decay_scale": 1.0,
            "magnitude_direction_budget_lr_scale": 1.0,
            "global_parent_magnitude_atlas_lr_scale": 1.0,
            "inherited_r01_budget_lr_scale": 1.0,
        })
        return report

    def _publish_downstream_functional_atlas(self, **_packet):
        """Neutral hook for an atomic cross-module structural transaction."""
        return None

    def _group_functional_decay_images(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        weight_decay,
        *,
        factors,
    ):
        return self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            self._stack_incoming_parameters() * weight_decay,
            self._stack_outgoing_parameters(transpose=True) * weight_decay,
            factors=factors,
        )

    def _group_functional_decay_scores(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        cotangents,
        weight_decay,
        *,
        factors,
    ):
        return self._group_tangent_scores(
            functional_inputs,
            functional_preactivations,
            functional_features,
            cotangents,
            self._stack_incoming_parameters() * weight_decay,
            self._stack_outgoing_parameters(transpose=True) * weight_decay,
            factors=factors,
        )

    def _paired_magnitude_state(self, gradients):
        """Return the exact matched-Adam direction for shared pair magnitudes."""
        expected = (len(self.pairs), self.groups, self.width)
        if gradients.shape != expected:
            raise RuntimeError("R05 paired-magnitude gradient inventory changed")
        anchor = self.state[self.incoming[0]]
        first = anchor.get("r05_next_magnitude_first")
        second = anchor.get("r05_next_magnitude_second")
        step = anchor.get("r05_next_magnitude_step", 0)
        if first is None:
            first = torch.zeros_like(gradients)
            second = torch.zeros_like(gradients)
        if (
            first.shape != expected
            or second is None
            or second.shape != expected
            or type(step) is not int
            or step < 0
        ):
            raise RuntimeError("R05 paired-magnitude state changed")
        next_step = step + 1
        first.mul_(self._r05_beta1).add_(
            gradients, alpha=1.0 - self._r05_beta1
        )
        second.mul_(self._r05_beta2).addcmul_(
            gradients, gradients, value=1.0 - self._r05_beta2
        )
        bias1 = 1.0 - self._r05_beta1 ** next_step
        bias2 = 1.0 - self._r05_beta2 ** next_step
        direction = (first / bias1) / (
            torch.sqrt(second / bias2) + self._r05_eps
        )
        valid = (
            torch.isfinite(direction).all()
            & torch.isfinite(first).all()
            & torch.isfinite(second).all()
        )
        torch._assert_async(valid)
        anchor["r05_next_magnitude_first"] = first
        anchor["r05_next_magnitude_second"] = second
        anchor["r05_next_magnitude_step"] = next_step
        return direction, next_step

    def _complete_magnitude_atlas(self, **_kwargs):
        """Optional implementation-only fusion hook for successor runtimes."""
        return None

    @staticmethod
    def _orthogonal_equal_budget_magnitude(
        parent_incoming,
        parent_outgoing,
        unit_incoming,
        unit_outgoing,
        magnitude_direction,
    ):
        """Lift one shared channel vector and close each group budget exactly."""
        if (
            parent_incoming.shape != parent_outgoing.shape
            or unit_incoming.shape != parent_incoming.shape
            or unit_outgoing.shape != parent_outgoing.shape
            or magnitude_direction.shape != parent_incoming.shape[:-1]
        ):
            raise RuntimeError("R05 paired-magnitude lift inventory changed")
        raw_incoming = unit_incoming * magnitude_direction[..., None]
        raw_outgoing = unit_outgoing * magnitude_direction[..., None]
        parent_budget = (
            parent_incoming.square().sum(dim=(-2, -1))
            + parent_outgoing.square().sum(dim=(-2, -1))
        )
        cross = (
            (raw_incoming * parent_incoming).sum(dim=(-2, -1))
            + (raw_outgoing * parent_outgoing).sum(dim=(-2, -1))
        )
        tiny = torch.finfo(parent_incoming.dtype).tiny
        projection = cross / parent_budget.clamp_min(tiny)
        magnitude_incoming = (
            raw_incoming - projection[..., None, None] * parent_incoming
        )
        magnitude_outgoing = (
            raw_outgoing - projection[..., None, None] * parent_outgoing
        )
        magnitude_budget = (
            magnitude_incoming.square().sum(dim=(-2, -1))
            + magnitude_outgoing.square().sum(dim=(-2, -1))
        )
        scale = torch.sqrt(
            parent_budget / magnitude_budget.clamp_min(tiny)
        )
        magnitude_incoming = magnitude_incoming * scale[..., None, None]
        magnitude_outgoing = magnitude_outgoing * scale[..., None, None]
        closed_budget = (
            magnitude_incoming.square().sum(dim=(-2, -1))
            + magnitude_outgoing.square().sum(dim=(-2, -1))
        )
        closed_cross = (
            (magnitude_incoming * parent_incoming).sum(dim=(-2, -1))
            + (magnitude_outgoing * parent_outgoing).sum(dim=(-2, -1))
        )
        budget_residual = (
            (closed_budget - parent_budget).abs()
            / parent_budget.clamp_min(1.0)
        )
        orthogonality_residual = (
            closed_cross.abs()
            / torch.sqrt(
                closed_budget * parent_budget
            ).clamp_min(1.0)
        )
        machine = torch.finfo(parent_incoming.dtype).eps
        valid = (
            torch.isfinite(magnitude_incoming).all(dim=(-2, -1))
            & torch.isfinite(magnitude_outgoing).all(dim=(-2, -1))
            & torch.isfinite(parent_budget)
            & torch.isfinite(magnitude_budget)
            & (parent_budget > 0.0)
            & (magnitude_budget > machine * parent_budget)
            & (budget_residual <= 4096.0 * machine)
            & (orthogonality_residual <= 4096.0 * machine)
        )
        return magnitude_incoming, magnitude_outgoing, {
            "valid": valid,
            "parent_budget": parent_budget,
            "budget_residual": budget_residual,
            "orthogonality_residual": orthogonality_residual,
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
        # Form literal complete R01 first.  It remains the all-ones feasible
        # point of the augmented atlas and is returned on every failed check.
        r01_incoming, r01_outgoing, r01_packet = super()._select_functional_corner(
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
        cotangents = self._r09_loss_cotangents
        if any(value is None for value in (
            functional_inputs,
            functional_preactivations,
            functional_features,
            cotangents,
        )):
            raise RuntimeError("R05 did not receive aligned global loss rows")
        layers = len(self.pairs)
        shape = (layers, self.groups, self.width, self.external_width)
        parent_incoming_blocks = r01_incoming.view(shape)
        parent_outgoing_blocks = r01_outgoing.view(shape)
        current_incoming = self._stack_incoming_parameters().view(shape)
        current_outgoing = self._stack_outgoing_parameters(
            transpose=True
        ).view(shape)
        incoming_norm = torch.linalg.vector_norm(
            current_incoming, dim=-1, keepdim=True
        )
        outgoing_norm = torch.linalg.vector_norm(
            current_outgoing, dim=-1, keepdim=True
        )
        tiny = torch.finfo(current_incoming.dtype).tiny
        unit_incoming = current_incoming / incoming_norm.clamp_min(tiny)
        unit_outgoing = current_outgoing / outgoing_norm.clamp_min(tiny)

        incoming_gradients = self._stack_incoming_gradients().view(shape)
        outgoing_gradients = self._stack_outgoing_gradients(
            transpose=True
        ).view(shape)
        radial_gradient = (
            (incoming_gradients * unit_incoming).sum(dim=-1)
            + (outgoing_gradients * unit_outgoing).sum(dim=-1)
        )
        magnitude_state, magnitude_step = self._paired_magnitude_state(
            radial_gradient
        )
        incoming_momentum = self._current_nesterov_stack(
            self.incoming
        ).view(shape)
        outgoing_momentum = self._current_nesterov_stack(
            self.outgoing, transpose=True
        ).view(shape)
        complete_atlas = self._complete_magnitude_atlas(
            parent_incoming=parent_incoming_blocks,
            parent_outgoing=parent_outgoing_blocks,
            unit_incoming=unit_incoming,
            unit_outgoing=unit_outgoing,
            magnitude_state=magnitude_state,
            incoming_gradients=incoming_gradients,
            outgoing_gradients=outgoing_gradients,
            incoming_momentum=incoming_momentum,
            outgoing_momentum=outgoing_momentum,
        )
        if complete_atlas is None:
            (
                magnitude_incoming_blocks,
                magnitude_outgoing_blocks,
                magnitude_metadata,
            ) = self._orthogonal_equal_budget_magnitude(
                parent_incoming_blocks,
                parent_outgoing_blocks,
                unit_incoming,
                unit_outgoing,
                magnitude_state,
            )
            plus_incoming = 0.5 * (
                parent_incoming_blocks + magnitude_incoming_blocks
            )
            minus_incoming = 0.5 * (
                parent_incoming_blocks - magnitude_incoming_blocks
            )
            plus_outgoing = 0.5 * (
                parent_outgoing_blocks + magnitude_outgoing_blocks
            )
            minus_outgoing = 0.5 * (
                parent_outgoing_blocks - magnitude_outgoing_blocks
            )
            precomputed_coordinates = None
        else:
            (
                magnitude_incoming_blocks,
                magnitude_outgoing_blocks,
                magnitude_metadata,
                plus_incoming,
                minus_incoming,
                plus_outgoing,
                minus_outgoing,
                precomputed_coordinates,
            ) = complete_atlas
        stream_atlas = bool(getattr(
            self, "_r07_stream_r05_atlas_enabled", False
        ))
        if not stream_atlas:
            atlas_incoming = torch.cat((plus_incoming, minus_incoming), dim=1)
            atlas_outgoing = torch.cat((plus_outgoing, minus_outgoing), dim=1)

        factors = self._functional_jvp_factors(functional_preactivations)
        weight_decay = float(self.param_groups[0]["weight_decay"])
        if getattr(self, "_r07_direct_score_contraction_enabled", False):
            plus_scores = self._group_tangent_scores(
                functional_inputs,
                functional_preactivations,
                functional_features,
                cotangents,
                plus_incoming.reshape_as(r01_incoming),
                plus_outgoing.reshape_as(r01_outgoing),
                factors=factors,
            )
            minus_scores = self._group_tangent_scores(
                functional_inputs,
                functional_preactivations,
                functional_features,
                cotangents,
                minus_incoming.reshape_as(r01_incoming),
                minus_outgoing.reshape_as(r01_outgoing),
                factors=factors,
            )
            atlas_scores = torch.cat((plus_scores, minus_scores), dim=2)
            decay_group_scores = self._group_functional_decay_scores(
                functional_inputs,
                functional_preactivations,
                functional_features,
                cotangents,
                weight_decay,
                factors=factors,
            )
            atlas_decay_scores = torch.cat((
                decay_group_scores,
                torch.zeros_like(decay_group_scores),
            ), dim=2)
            metric_cotangents = torch.ones_like(cotangents[..., :1])
            atlas_images = atlas_scores[..., None]
            atlas_decay = atlas_decay_scores[..., None]
            fisher, decay_cross, global_count = self._reduce_global_loss_metric(
                atlas_images, metric_cotangents, atlas_decay
            )
        else:
            parent_images = (
                self._selected_group_images()
                if getattr(self, "_r07_linear_image_reuse_enabled", False)
                else None
            )
            if parent_images is None:
                plus_images = self._group_tangent_images(
                    functional_inputs,
                    functional_preactivations,
                    functional_features,
                    plus_incoming.reshape_as(r01_incoming),
                    plus_outgoing.reshape_as(r01_outgoing),
                    factors=factors,
                )
                minus_images = self._group_tangent_images(
                    functional_inputs,
                    functional_preactivations,
                    functional_features,
                    minus_incoming.reshape_as(r01_incoming),
                    minus_outgoing.reshape_as(r01_outgoing),
                    factors=factors,
                )
            else:
                magnitude_images = self._group_tangent_images(
                    functional_inputs,
                    functional_preactivations,
                    functional_features,
                    magnitude_incoming_blocks.reshape_as(r01_incoming),
                    magnitude_outgoing_blocks.reshape_as(r01_outgoing),
                    factors=factors,
                )
                plus_images = 0.5 * (parent_images + magnitude_images)
                minus_images = 0.5 * (parent_images - magnitude_images)
            atlas_images = torch.cat((plus_images, minus_images), dim=2)
            decay_group_images = self._group_functional_decay_images(
                functional_inputs,
                functional_preactivations,
                functional_features,
                weight_decay,
                factors=factors,
            )
            zero_decay = torch.zeros_like(decay_group_images)
            atlas_decay = torch.cat((decay_group_images, zero_decay), dim=2)
            fisher, decay_cross, global_count = self._reduce_global_loss_metric(
                atlas_images, cotangents, atlas_decay
            )

        if precomputed_coordinates is not None:
            (
                incoming_exact_coordinates,
                outgoing_exact_coordinates,
                incoming_momentum_coordinates,
                outgoing_momentum_coordinates,
                atlas_budget,
            ) = precomputed_coordinates
        elif stream_atlas:
            # Preserve the registered reduction layout exactly; only the
            # dead persistent atlas copies are streamed away.
            incoming_exact_coordinates = (
                incoming_gradients[:, :, None] * torch.stack((
                    plus_incoming, minus_incoming
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
            outgoing_exact_coordinates = (
                outgoing_gradients[:, :, None] * torch.stack((
                    plus_outgoing, minus_outgoing
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
            incoming_momentum_coordinates = (
                incoming_momentum[:, :, None] * torch.stack((
                    plus_incoming, minus_incoming
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
            outgoing_momentum_coordinates = (
                outgoing_momentum[:, :, None] * torch.stack((
                    plus_outgoing, minus_outgoing
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
        else:
            incoming_exact_coordinates = (
                incoming_gradients[:, :, None] * torch.stack((
                    plus_incoming, minus_incoming
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
            outgoing_exact_coordinates = (
                outgoing_gradients[:, :, None] * torch.stack((
                    plus_outgoing, minus_outgoing
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
            incoming_momentum_coordinates = (
                incoming_momentum[:, :, None] * torch.stack((
                    plus_incoming, minus_incoming
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
            outgoing_momentum_coordinates = (
                outgoing_momentum[:, :, None] * torch.stack((
                    plus_outgoing, minus_outgoing
                ), dim=2)
            ).sum(dim=(-2, -1)).permute(0, 2, 1).reshape(1, -1)
        exact_linear = incoming_exact_coordinates + outgoing_exact_coordinates
        momentum_linear = (
            incoming_momentum_coordinates + outgoing_momentum_coordinates
        )
        if precomputed_coordinates is not None:
            pass
        elif stream_atlas:
            atlas_budget = torch.cat((
                plus_incoming.square().sum(dim=(-2, -1))
                + plus_outgoing.square().sum(dim=(-2, -1)),
                minus_incoming.square().sum(dim=(-2, -1))
                + minus_outgoing.square().sum(dim=(-2, -1)),
            ), dim=1).reshape(1, -1)
        else:
            atlas_budget = (
                atlas_incoming.square().sum(dim=(-2, -1))
                + atlas_outgoing.square().sum(dim=(-2, -1))
            ).reshape(1, -1)

        coefficients, solve = self._select_group_span_coefficients(
            fisher,
            decay_cross,
            exact_linear,
            momentum_linear,
            atlas_budget,
            lr,
        )
        coefficients = coefficients.view(layers, 2 * self.groups)
        if stream_atlas:
            plus_coefficients = coefficients[:, :self.groups]
            minus_coefficients = coefficients[:, self.groups:]
            selected_incoming_blocks = (
                plus_incoming * plus_coefficients[..., None, None]
                + minus_incoming * minus_coefficients[..., None, None]
            )
            selected_outgoing_blocks = (
                plus_outgoing * plus_coefficients[..., None, None]
                + minus_outgoing * minus_coefficients[..., None, None]
            )
        else:
            selected_incoming_blocks = (
                atlas_incoming * coefficients[..., None, None]
            )
            selected_outgoing_blocks = (
                atlas_outgoing * coefficients[..., None, None]
            )
            selected_incoming_blocks = (
                selected_incoming_blocks[:, :self.groups]
                + selected_incoming_blocks[:, self.groups:]
            )
            selected_outgoing_blocks = (
                selected_outgoing_blocks[:, :self.groups]
                + selected_outgoing_blocks[:, self.groups:]
            )

        candidate_incoming_exact = (
            incoming_gradients * selected_incoming_blocks
        ).sum(dim=(-2, -1)).sum(dim=-1)
        candidate_outgoing_exact = (
            outgoing_gradients * selected_outgoing_blocks
        ).sum(dim=(-2, -1)).sum(dim=-1)
        candidate_incoming_momentum = (
            incoming_momentum * selected_incoming_blocks
        ).sum(dim=(-2, -1)).sum(dim=-1)
        candidate_outgoing_momentum = (
            outgoing_momentum * selected_outgoing_blocks
        ).sum(dim=(-2, -1)).sum(dim=-1)
        role_descent_valid = (
            (candidate_incoming_exact > 0.0).all()
            & (candidate_outgoing_exact > 0.0).all()
            & (candidate_incoming_momentum > 0.0).all()
            & (candidate_outgoing_momentum > 0.0).all()
        )
        accepted = (
            solve["accepted"][0]
            & magnitude_metadata["valid"].all()
            & role_descent_valid
            & (~force_parent.any())
        )
        selected_incoming = torch.where(
            accepted,
            selected_incoming_blocks.reshape_as(r01_incoming),
            r01_incoming,
        )
        selected_outgoing = torch.where(
            accepted,
            selected_outgoing_blocks.reshape_as(r01_outgoing),
            r01_outgoing,
        )
        if (
            getattr(self, "_r07_linear_image_reuse_enabled", False)
            and not getattr(self, "_r07_direct_score_contraction_enabled", False)
        ):
            candidate_group_images = (
                plus_images * coefficients[:, None, :self.groups, None]
                + minus_images * coefficients[:, None, self.groups:, None]
            )
            parent_group_images = (
                parent_images
                if parent_images is not None
                else plus_images + minus_images
            )
            self._remember_selected_group_images(torch.where(
                accepted,
                candidate_group_images,
                parent_group_images,
            ))
        selected_score = torch.where(
            accepted, solve["selected_score"][0], solve["parent_score"][0]
        )
        improvement = solve["parent_score"][0] - selected_score
        plus_coefficients = coefficients[:, :self.groups]
        minus_coefficients = coefficients[:, self.groups:]
        parent_coefficients = 0.5 * (
            plus_coefficients + minus_coefficients
        )
        magnitude_coefficients = 0.5 * (
            plus_coefficients - minus_coefficients
        )
        self._r05_next_metadata = {
            "accepted": accepted,
            "magnitude_step": magnitude_step,
            "global_count": global_count,
            "rank": solve["rank"][0],
            "budget_residual": solve["budget_residual"][0],
            "magnitude_budget_residual": magnitude_metadata[
                "budget_residual"
            ].amax(),
            "magnitude_orthogonality_residual": magnitude_metadata[
                "orthogonality_residual"
            ].amax(),
            "parent_coefficient_min": parent_coefficients.amin(),
            "parent_coefficient_median": parent_coefficients.median(),
            "parent_coefficient_max": parent_coefficients.amax(),
            "magnitude_coefficient_abs_max": magnitude_coefficients.abs().amax(),
            "magnitude_coefficient_rms": torch.sqrt(
                magnitude_coefficients.square().mean()
            ),
            "improvement": improvement,
            "incoming_exact_descent_min": torch.where(
                accepted,
                candidate_incoming_exact.amin(),
                (incoming_gradients * parent_incoming_blocks)
                .sum(dim=(-2, -1)).sum(dim=-1).amin(),
            ),
            "outgoing_exact_descent_min": torch.where(
                accepted,
                candidate_outgoing_exact.amin(),
                (outgoing_gradients * parent_outgoing_blocks)
                .sum(dim=(-2, -1)).sum(dim=-1).amin(),
            ),
        }

        choices = torch.where(
            accepted,
            torch.full(
                (layers,), 3, device=r01_incoming.device, dtype=torch.int64
            ),
            torch.zeros((layers,), device=r01_incoming.device, dtype=torch.int64),
        )
        scores = torch.stack((
            solve["parent_score"][0],
            solve["candidate_score"][0],
            solve["parent_score"][0],
            solve["candidate_score"][0],
        )).reshape(1, 4).expand(layers, 4)
        packet = dict(r01_packet)
        packet.update({
            "choices": choices,
            "scores": scores,
            "score_margin": improvement.abs().expand(layers),
            "global_count": global_count,
        })
        actual_coefficients = torch.where(
            accepted, coefficients, torch.ones_like(coefficients)
        )
        self._publish_downstream_functional_atlas(
            atlas_images=atlas_images,
            atlas_decay=atlas_decay,
            cotangents=cotangents,
            atlas_incoming=(None if stream_atlas else atlas_incoming),
            atlas_outgoing=(None if stream_atlas else atlas_outgoing),
            parent_coefficients=actual_coefficients,
            parent_incoming=selected_incoming,
            parent_outgoing=selected_outgoing,
            exact_linear=exact_linear[0],
            momentum_linear=momentum_linear[0],
            incoming_exact_linear=incoming_exact_coordinates[0],
            outgoing_exact_linear=outgoing_exact_coordinates[0],
            incoming_momentum_linear=incoming_momentum_coordinates[0],
            outgoing_momentum_linear=outgoing_momentum_coordinates[0],
            atlas_budget=atlas_budget[0],
            parent_accepted=accepted,
        )
        return selected_incoming, selected_outgoing, packet

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r05_next_metadata = None
        loss = super().step(closure)
        metadata = self._r05_next_metadata
        if metadata is None:
            raise RuntimeError("R05 did not execute its paired-magnitude atlas")
        if publish:
            # The nested parent telemetry names the inherited R01 component;
            # the outer R05 telemetry below names the new complete method.
            self._last_telemetry["rlb_r01_component_code"] = 12
            self._last_telemetry.update({
                "rlb_r05_component_code": self.component_code,
                "rlb_r05_parent_is_complete_r01": 1,
                "rlb_r05_paired_hidden_magnitude_enabled": 1,
                "rlb_r05_global_parent_magnitude_atlas_enabled": 1,
                "rlb_r05_layer_count": len(self.pairs),
                "rlb_r05_group_count": self.groups,
                "rlb_r05_group_width": self.width,
                "rlb_r05_atlas_dimension": 2 * len(self.pairs) * self.groups,
                "rlb_r05_global_loss_sample_count": int(
                    metadata["global_count"].item()
                ),
                "rlb_r05_magnitude_step": metadata["magnitude_step"],
                "rlb_r05_metric_transaction_count": int(
                    metadata["accepted"].item()
                ),
                "rlb_r05_parent_transaction_count": int(
                    (~metadata["accepted"]).item()
                ),
                "rlb_r05_fisher_rank": int(metadata["rank"].item()),
                "rlb_r05_budget_residual": float(
                    metadata["budget_residual"].item()
                ),
                "rlb_r05_magnitude_budget_residual_max": float(
                    metadata["magnitude_budget_residual"].item()
                ),
                "rlb_r05_magnitude_orthogonality_residual_max": float(
                    metadata["magnitude_orthogonality_residual"].item()
                ),
                "rlb_r05_parent_coefficient_min": float(
                    metadata["parent_coefficient_min"].item()
                ),
                "rlb_r05_parent_coefficient_median": float(
                    metadata["parent_coefficient_median"].item()
                ),
                "rlb_r05_parent_coefficient_max": float(
                    metadata["parent_coefficient_max"].item()
                ),
                "rlb_r05_magnitude_coefficient_abs_max": float(
                    metadata["magnitude_coefficient_abs_max"].item()
                ),
                "rlb_r05_magnitude_coefficient_rms": float(
                    metadata["magnitude_coefficient_rms"].item()
                ),
                "rlb_r05_surrogate_improvement": float(
                    metadata["improvement"].item()
                ),
                "rlb_r05_incoming_exact_descent_min": float(
                    metadata["incoming_exact_descent_min"].item()
                ),
                "rlb_r05_outgoing_exact_descent_min": float(
                    metadata["outgoing_exact_descent_min"].item()
                ),
                "rlb_r05_structural_matrix_elements": 245_366_784,
            })
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r05_next_metadata = None
        return result


__all__ = ("R05NextCore",)
