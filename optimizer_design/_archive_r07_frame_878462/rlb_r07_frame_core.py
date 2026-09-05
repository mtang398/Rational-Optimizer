"""Cross-role frame geometry on literal complete R03.

For one Global-RLB group, the incoming bank ``A_g`` and the transposed
outgoing bank ``C_g=B_g^T`` both have shape ``width x external``.  They are
the analysis and synthesis legs surrounding the same learned P5/Q4 response.
R07 stacks their clipped Nesterov sources along the row axis and applies one
fixed Muon NS5 polar map to the resulting ``(2*width) x external`` frame.

This differs from polarizing the two roles independently and from R06's
horizontal paired-channel polar.  The R07 spectral constraint lets all rows
from both roles compete for one residual-space row frame.  The resulting
direction is residualized against literal R03, restored to the exact R03
paired layer budget, and offered to the same current P5/Q4-plus-group-RMS
functional allocator through two half-energy axes.  LR, WD, momentum,
clipping, data order, and update budget are unchanged.
"""

from __future__ import annotations

import torch

from .rlb_group_muon_core import _batched_zero_power
from .rlb_r01_core import R01Core
from .rlb_r03_core import R03Core


class R07FrameCore(R03Core):
    """Complete R03 plus one cross-role RLB frame-polar transaction."""

    component_code = 46
    checkpoint_schema = "r07_r03_cross_role_frame_polar_v1"
    inherited_parent = "complete_r03_persistent_p5_q4_score_geometry"
    new_scientific_components = (
        "cross_role_rlb_frame_coupling",
        "fixed_cross_role_frame_ns5_polarization",
        "exact_p5_q4_layer_axis_functional_allocation",
    )

    def __init__(
        self,
        pairs,
        *,
        use_cross_role_frame: bool = True,
        use_frame_ns5: bool = True,
        use_functional_allocator: bool = True,
        **kwargs,
    ):
        if type(use_cross_role_frame) is not bool:
            raise ValueError("R07 cross-role-frame flag must be boolean")
        if type(use_frame_ns5) is not bool:
            raise ValueError("R07 frame-NS5 flag must be boolean")
        if type(use_functional_allocator) is not bool:
            raise ValueError("R07 functional-allocator flag must be boolean")
        self.use_cross_role_frame = use_cross_role_frame
        self.use_frame_ns5 = use_frame_ns5
        self.use_functional_allocator = use_functional_allocator
        self._r07_frame_metadata = None
        super().__init__(pairs, **kwargs)

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "cross_role_frame_lr_scale": 1.0,
            "cross_role_frame_weight_decay_scale": 1.0,
            "cross_role_frame_ns5_lr_scale": 1.0,
            "cross_role_frame_parent_budget_lr_scale": 1.0,
            "cross_role_frame_functional_allocator_lr_scale": 1.0,
        })
        return report

    @staticmethod
    def _frame_source(
        incoming_momentum,
        outgoing_momentum,
        *,
        ns_steps,
        cross_role,
        polarize,
        measure=True,
    ):
        """Return the registered stacked-frame source or a direct deletion."""
        if (
            incoming_momentum.shape != outgoing_momentum.shape
            or incoming_momentum.ndim != 4
            or int(ns_steps) != 5
            or type(cross_role) is not bool
            or type(polarize) is not bool
            or type(measure) is not bool
        ):
            raise RuntimeError("R07 frame-source inventory changed")
        layers, groups, width, external = incoming_momentum.shape
        if polarize and cross_role:
            stacked = torch.cat((incoming_momentum, outgoing_momentum), dim=-2)
            source = _batched_zero_power(
                stacked.reshape(layers * groups, 2 * width, external),
                int(ns_steps),
            ).float().view(layers, groups, 2 * width, external)
            incoming, outgoing = source.split(width, dim=-2)
        elif polarize:
            incoming = _batched_zero_power(
                incoming_momentum.reshape(layers * groups, width, external),
                int(ns_steps),
            ).float().view_as(incoming_momentum)
            outgoing = _batched_zero_power(
                outgoing_momentum.reshape(layers * groups, width, external),
                int(ns_steps),
            ).float().view_as(outgoing_momentum)
            source = torch.cat((incoming, outgoing), dim=-2)
        else:
            incoming = incoming_momentum.clone()
            outgoing = outgoing_momentum.clone()
            source = torch.cat((incoming, outgoing), dim=-2)

        incoming_descent = (
            incoming_momentum * incoming
        ).sum(dim=(-2, -1))
        outgoing_descent = (
            outgoing_momentum * outgoing
        ).sum(dim=(-2, -1))
        total_descent = incoming_descent + outgoing_descent
        if measure:
            gram = source @ source.transpose(-2, -1)
            diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
            diagonal_scale = diagonal.mean(dim=-1, keepdim=True)
            identity = torch.eye(
                2 * width, device=gram.device, dtype=gram.dtype
            ).view(1, 1, 2 * width, 2 * width)
            normalized = gram / diagonal_scale.clamp_min(
                torch.finfo(gram.dtype).tiny
            )[..., None]
            gram_residual = torch.linalg.matrix_norm(
                normalized - identity, ord="fro", dim=(-2, -1)
            ) / torch.sqrt(gram.new_tensor(float(2 * width)))
            cross_gram = incoming @ outgoing.transpose(-2, -1)
            cross_role_residual = torch.linalg.matrix_norm(
                cross_gram, ord="fro", dim=(-2, -1)
            ) / torch.sqrt(gram.new_tensor(float(width)))
        else:
            gram_residual = source.new_zeros((layers, groups))
            cross_role_residual = source.new_zeros((layers, groups))

        valid = (
            torch.isfinite(source).all(dim=(-2, -1))
            & torch.isfinite(incoming_descent)
            & torch.isfinite(outgoing_descent)
            & torch.isfinite(total_descent)
            & (total_descent > 0.0)
        )
        return incoming, outgoing, {
            "valid": valid,
            "incoming_source_descent_min": incoming_descent.amin(),
            "outgoing_source_descent_min": outgoing_descent.amin(),
            "total_source_descent_min": total_descent.amin(),
            "frame_gram_residual_max": gram_residual.amax(),
            "cross_role_gram_residual_max": cross_role_residual.amax(),
        }

    @staticmethod
    def _orthogonal_equal_layer_budget(
        parent_incoming,
        parent_outgoing,
        source_incoming,
        source_outgoing,
    ):
        """Remove literal R03 and restore its exact paired layer budget."""
        if not (
            parent_incoming.shape == parent_outgoing.shape
            == source_incoming.shape == source_outgoing.shape
            and parent_incoming.ndim == 4
        ):
            raise RuntimeError("R07 layer-budget inventory changed")
        dims = (-3, -2, -1)
        parent_budget = (
            parent_incoming.square().sum(dim=dims)
            + parent_outgoing.square().sum(dim=dims)
        )
        cross = (
            (source_incoming * parent_incoming).sum(dim=dims)
            + (source_outgoing * parent_outgoing).sum(dim=dims)
        )
        tiny = torch.finfo(parent_incoming.dtype).tiny
        projection = cross / parent_budget.clamp_min(tiny)
        incoming = source_incoming - projection[:, None, None, None] * parent_incoming
        outgoing = source_outgoing - projection[:, None, None, None] * parent_outgoing
        residual_budget = (
            incoming.square().sum(dim=dims)
            + outgoing.square().sum(dim=dims)
        )
        scale = torch.sqrt(parent_budget / residual_budget.clamp_min(tiny))
        incoming = incoming * scale[:, None, None, None]
        outgoing = outgoing * scale[:, None, None, None]
        closed_budget = (
            incoming.square().sum(dim=dims)
            + outgoing.square().sum(dim=dims)
        )
        closed_cross = (
            (incoming * parent_incoming).sum(dim=dims)
            + (outgoing * parent_outgoing).sum(dim=dims)
        )
        budget_residual = (
            (closed_budget - parent_budget).abs()
            / parent_budget.clamp_min(1.0)
        )
        orthogonality_residual = closed_cross.abs() / torch.sqrt(
            closed_budget * parent_budget
        ).clamp_min(1.0)
        machine = torch.finfo(parent_incoming.dtype).eps
        valid = (
            torch.isfinite(incoming).all(dim=dims)
            & torch.isfinite(outgoing).all(dim=dims)
            & torch.isfinite(parent_budget)
            & torch.isfinite(residual_budget)
            & (parent_budget > 0.0)
            & (residual_budget > machine * parent_budget)
            & (budget_residual <= 4096.0 * machine)
            & (orthogonality_residual <= 4096.0 * machine)
        )
        return incoming, outgoing, {
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
        # Literal R03 is formed and its persistent score state advances once.
        r03_incoming, r03_outgoing, r03_packet = super()._select_functional_corner(
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
            raise RuntimeError("R07 did not receive aligned functional rows")

        layers = len(self.pairs)
        shape = (layers, self.groups, self.width, self.external_width)
        parent_incoming_blocks = r03_incoming.view(shape)
        parent_outgoing_blocks = r03_outgoing.view(shape)
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

        source_incoming, source_outgoing, source_metadata = self._frame_source(
            incoming_momentum,
            outgoing_momentum,
            ns_steps=self.ns_steps,
            cross_role=self.use_cross_role_frame,
            polarize=self.use_frame_ns5,
            measure=bool(self._capture_telemetry_next_step),
        )
        axis_incoming, axis_outgoing, axis_metadata = (
            self._orthogonal_equal_layer_budget(
                parent_incoming_blocks,
                parent_outgoing_blocks,
                source_incoming,
                source_outgoing,
            )
        )
        geometry_valid = (
            source_metadata["valid"].all()
            & axis_metadata["valid"].all()
        )
        axis_incoming = torch.where(
            geometry_valid, axis_incoming, torch.zeros_like(axis_incoming)
        )
        axis_outgoing = torch.where(
            geometry_valid, axis_outgoing, torch.zeros_like(axis_outgoing)
        )

        if not self.use_functional_allocator:
            candidate_incoming_exact = (
                incoming_gradients * axis_incoming
            ).sum(dim=(-3, -2, -1))
            candidate_outgoing_exact = (
                outgoing_gradients * axis_outgoing
            ).sum(dim=(-3, -2, -1))
            candidate_incoming_nesterov = (
                incoming_momentum * axis_incoming
            ).sum(dim=(-3, -2, -1))
            candidate_outgoing_nesterov = (
                outgoing_momentum * axis_outgoing
            ).sum(dim=(-3, -2, -1))
            accepted = (
                geometry_valid
                & (candidate_incoming_exact > 0.0).all()
                & (candidate_outgoing_exact > 0.0).all()
                & (candidate_incoming_nesterov > 0.0).all()
                & (candidate_outgoing_nesterov > 0.0).all()
                & (~force_parent.any())
            )
            selected_incoming = torch.where(
                accepted, axis_incoming, parent_incoming_blocks
            )
            selected_outgoing = torch.where(
                accepted, axis_outgoing, parent_outgoing_blocks
            )
            count = torch.tensor(
                float(functional_inputs.shape[1]),
                device=functional_inputs.device,
                dtype=functional_inputs.dtype,
            )
            self._r07_frame_metadata = {
                "accepted": accepted,
                "global_count": count,
                "rank": torch.zeros((), device=count.device, dtype=torch.int64),
                "budget_residual": torch.zeros_like(count),
                "improvement": torch.zeros_like(count),
                "parent_coefficient_min": torch.zeros_like(count),
                "parent_coefficient_median": torch.zeros_like(count),
                "parent_coefficient_max": torch.zeros_like(count),
                "frame_coefficient_abs_max": torch.ones_like(count),
                "frame_coefficient_rms": torch.ones_like(count),
                "incoming_exact_descent_min": torch.where(
                    accepted,
                    candidate_incoming_exact.amin(),
                    (incoming_gradients * parent_incoming_blocks)
                    .sum(dim=(-3, -2, -1)).amin(),
                ),
                "outgoing_exact_descent_min": torch.where(
                    accepted,
                    candidate_outgoing_exact.amin(),
                    (outgoing_gradients * parent_outgoing_blocks)
                    .sum(dim=(-3, -2, -1)).amin(),
                ),
                **source_metadata,
                "axis_budget_residual_max": axis_metadata[
                    "budget_residual"
                ].amax(),
                "axis_orthogonality_residual_max": axis_metadata[
                    "orthogonality_residual"
                ].amax(),
            }
            packet = dict(r03_packet)
            zeros = torch.zeros(
                (layers, 4), device=count.device, dtype=count.dtype
            )
            packet.update({
                "choices": torch.where(
                    accepted,
                    torch.full(
                        (layers,), 3, device=count.device, dtype=torch.int64
                    ),
                    torch.zeros(
                        (layers,), device=count.device, dtype=torch.int64
                    ),
                ),
                "scores": zeros,
                "score_margin": torch.zeros(
                    layers, device=count.device, dtype=count.dtype
                ),
                "energies": zeros,
                "global_count": count,
            })
            return (
                selected_incoming.reshape_as(r03_incoming),
                selected_outgoing.reshape_as(r03_outgoing),
                packet,
            )

        plus_incoming = 0.5 * (parent_incoming_blocks + axis_incoming)
        minus_incoming = 0.5 * (parent_incoming_blocks - axis_incoming)
        plus_outgoing = 0.5 * (parent_outgoing_blocks + axis_outgoing)
        minus_outgoing = 0.5 * (parent_outgoing_blocks - axis_outgoing)

        factors = self._functional_jvp_factors(functional_preactivations)
        plus_images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            plus_incoming.reshape_as(r03_incoming),
            plus_outgoing.reshape_as(r03_outgoing),
            factors=factors,
        ).sum(dim=2)
        minus_images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            minus_incoming.reshape_as(r03_incoming),
            minus_outgoing.reshape_as(r03_outgoing),
            factors=factors,
        ).sum(dim=2)
        atlas_images = torch.stack((plus_images, minus_images), dim=2)

        weight_decay = float(self.param_groups[0]["weight_decay"])
        decay_image = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            torch.stack(self.incoming).float() * weight_decay,
            torch.stack(self.outgoing).float().transpose(-2, -1) * weight_decay,
            factors=factors,
        ).sum(dim=2)
        atlas_decay = torch.stack(
            (decay_image, torch.zeros_like(decay_image)), dim=2
        )
        fisher, decay_cross, global_count = R01Core._reduce_global_loss_metric(
            atlas_images, cotangents, atlas_decay
        )

        incoming_axes = torch.stack((plus_incoming, minus_incoming), dim=1)
        outgoing_axes = torch.stack((plus_outgoing, minus_outgoing), dim=1)
        incoming_exact = (
            incoming_gradients[:, None] * incoming_axes
        ).sum(dim=(-3, -2, -1))
        outgoing_exact = (
            outgoing_gradients[:, None] * outgoing_axes
        ).sum(dim=(-3, -2, -1))
        incoming_nesterov = (
            incoming_momentum[:, None] * incoming_axes
        ).sum(dim=(-3, -2, -1))
        outgoing_nesterov = (
            outgoing_momentum[:, None] * outgoing_axes
        ).sum(dim=(-3, -2, -1))
        budget = (
            incoming_axes.square().sum(dim=(-3, -2, -1))
            + outgoing_axes.square().sum(dim=(-3, -2, -1))
        )
        exact_linear = (incoming_exact + outgoing_exact).reshape(1, -1)
        nesterov_linear = (
            incoming_nesterov + outgoing_nesterov
        ).reshape(1, -1)
        flat_budget = budget.reshape(1, -1)
        flat_coefficients, solve = self._select_group_span_coefficients(
            fisher,
            decay_cross,
            exact_linear,
            nesterov_linear,
            flat_budget,
            lr,
        )
        coefficients = flat_coefficients.view(layers, 2)
        candidate_incoming = (
            incoming_axes * coefficients[:, :, None, None, None]
        ).sum(dim=1)
        candidate_outgoing = (
            outgoing_axes * coefficients[:, :, None, None, None]
        ).sum(dim=1)

        candidate_incoming_exact = (
            incoming_gradients * candidate_incoming
        ).sum(dim=(-3, -2, -1))
        candidate_outgoing_exact = (
            outgoing_gradients * candidate_outgoing
        ).sum(dim=(-3, -2, -1))
        candidate_incoming_nesterov = (
            incoming_momentum * candidate_incoming
        ).sum(dim=(-3, -2, -1))
        candidate_outgoing_nesterov = (
            outgoing_momentum * candidate_outgoing
        ).sum(dim=(-3, -2, -1))
        role_safe = (
            (candidate_incoming_exact > 0.0).all()
            & (candidate_outgoing_exact > 0.0).all()
            & (candidate_incoming_nesterov > 0.0).all()
            & (candidate_outgoing_nesterov > 0.0).all()
        )
        accepted = (
            solve["accepted"][0]
            & role_safe
            & geometry_valid
            & (~force_parent.any())
        )
        selected_incoming = torch.where(
            accepted, candidate_incoming, parent_incoming_blocks
        )
        selected_outgoing = torch.where(
            accepted, candidate_outgoing, parent_outgoing_blocks
        )
        selected_coefficients = torch.where(
            accepted, coefficients, torch.ones_like(coefficients)
        )
        plus_coefficient = selected_coefficients[:, 0]
        minus_coefficient = selected_coefficients[:, 1]
        parent_coefficient = 0.5 * (plus_coefficient + minus_coefficient)
        frame_coefficient = 0.5 * (plus_coefficient - minus_coefficient)
        selected_score = torch.where(
            accepted, solve["candidate_score"][0], solve["parent_score"][0]
        )
        self._r07_frame_metadata = {
            "accepted": accepted,
            "global_count": global_count,
            "rank": solve["rank"][0],
            "budget_residual": solve["budget_residual"][0],
            "improvement": solve["parent_score"][0] - selected_score,
            "parent_coefficient_min": parent_coefficient.amin(),
            "parent_coefficient_median": parent_coefficient.median(),
            "parent_coefficient_max": parent_coefficient.amax(),
            "frame_coefficient_abs_max": frame_coefficient.abs().amax(),
            "frame_coefficient_rms": torch.sqrt(frame_coefficient.square().mean()),
            "incoming_exact_descent_min": torch.where(
                accepted,
                candidate_incoming_exact.amin(),
                (incoming_gradients * parent_incoming_blocks)
                .sum(dim=(-3, -2, -1)).amin(),
            ),
            "outgoing_exact_descent_min": torch.where(
                accepted,
                candidate_outgoing_exact.amin(),
                (outgoing_gradients * parent_outgoing_blocks)
                .sum(dim=(-3, -2, -1)).amin(),
            ),
            **source_metadata,
            "axis_budget_residual_max": axis_metadata["budget_residual"].amax(),
            "axis_orthogonality_residual_max": axis_metadata[
                "orthogonality_residual"
            ].amax(),
        }
        choices = torch.where(
            accepted,
            torch.full(
                (layers,), 3, device=coefficients.device, dtype=torch.int64
            ),
            torch.zeros(
                (layers,), device=coefficients.device, dtype=torch.int64
            ),
        )
        packet = dict(r03_packet)
        parent_score = solve["parent_score"][0]
        candidate_score = solve["candidate_score"][0]
        scores = torch.stack((
            parent_score,
            candidate_score,
            parent_score,
            candidate_score,
        )).reshape(1, 4).expand(layers, 4)
        packet.update({
            "choices": choices,
            "scores": scores,
            "score_margin": (parent_score - candidate_score).abs().expand(layers),
            "energies": torch.zeros_like(scores),
            "global_count": global_count,
        })
        return (
            selected_incoming.reshape_as(r03_incoming),
            selected_outgoing.reshape_as(r03_outgoing),
            packet,
        )

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r07_frame_metadata = None
        loss = super().step(closure)
        metadata = self._r07_frame_metadata
        if metadata is None:
            raise RuntimeError("R07 cross-role frame transaction did not execute")
        if publish:
            self._last_telemetry.update({
                "rlb_r07_component_code": self.component_code,
                "rlb_r07_parent_is_complete_r03": 1,
                "rlb_r07_cross_role_frame_enabled": int(
                    self.use_cross_role_frame
                ),
                "rlb_r07_fixed_frame_ns5_enabled": int(self.use_frame_ns5),
                "rlb_r07_ns_steps": int(self.ns_steps),
                "rlb_r07_exact_p5_q4_layer_allocator_enabled": int(
                    self.use_functional_allocator
                ),
                "rlb_r07_transaction_count": int(metadata["accepted"].item()),
                "rlb_r07_parent_count": int((~metadata["accepted"]).item()),
                "rlb_r07_coordinate_count": (
                    2 * len(self.pairs) if self.use_functional_allocator else 0
                ),
                "rlb_r07_global_loss_sample_count": int(
                    metadata["global_count"].item()
                ),
                "rlb_r07_solver_rank": int(metadata["rank"].item()),
                "rlb_r07_budget_residual": float(
                    metadata["budget_residual"].item()
                ),
                "rlb_r07_axis_budget_residual_max": float(
                    metadata["axis_budget_residual_max"].item()
                ),
                "rlb_r07_axis_orthogonality_residual_max": float(
                    metadata["axis_orthogonality_residual_max"].item()
                ),
                "rlb_r07_frame_gram_residual_max": float(
                    metadata["frame_gram_residual_max"].item()
                ),
                "rlb_r07_cross_role_gram_residual_max": float(
                    metadata["cross_role_gram_residual_max"].item()
                ),
                "rlb_r07_incoming_source_descent_min": float(
                    metadata["incoming_source_descent_min"].item()
                ),
                "rlb_r07_outgoing_source_descent_min": float(
                    metadata["outgoing_source_descent_min"].item()
                ),
                "rlb_r07_total_source_descent_min": float(
                    metadata["total_source_descent_min"].item()
                ),
                "rlb_r07_parent_coefficient_min": float(
                    metadata["parent_coefficient_min"].item()
                ),
                "rlb_r07_parent_coefficient_median": float(
                    metadata["parent_coefficient_median"].item()
                ),
                "rlb_r07_parent_coefficient_max": float(
                    metadata["parent_coefficient_max"].item()
                ),
                "rlb_r07_frame_coefficient_abs_max": float(
                    metadata["frame_coefficient_abs_max"].item()
                ),
                "rlb_r07_frame_coefficient_rms": float(
                    metadata["frame_coefficient_rms"].item()
                ),
                "rlb_r07_surrogate_improvement": float(
                    metadata["improvement"].item()
                ),
                "rlb_r07_incoming_exact_descent_min": float(
                    metadata["incoming_exact_descent_min"].item()
                ),
                "rlb_r07_outgoing_exact_descent_min": float(
                    metadata["outgoing_exact_descent_min"].item()
                ),
                "rlb_r07_structural_matrix_elements": 245_366_784,
            })
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r07_frame_metadata = None
        return result


__all__ = ("R07FrameCore",)
