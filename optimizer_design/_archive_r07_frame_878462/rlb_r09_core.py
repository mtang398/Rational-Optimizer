"""Downstream-loss geometry in the exact Global-RLB group tangent span.

Current R02 supplies a strong direction inside every learned rational group.
R09 keeps that complete matrix and attention parent, then uses the structural
decomposition that only Global-RLB exposes.  For group ``g`` it forms

    Y_g = B_g J_g(z) D^A_g x + D^B_g h_g,

where ``J_g`` is the exact Jacobian of the installed RMS-rescaled P5/Q4
response and ``(D^A_g,D^B_g)`` are current R02's paired directions.  The
backpropagated MLP-output cotangent turns these images into per-token scores.
Their all-rank second moment is an empirical Fisher/Gauss--Newton metric in
the 18-dimensional RLB group span.  The linear term is the exact clipped
full-batch directional derivative, not the sampled approximation.

Each layer solves the resulting equality-constrained trust-region problem
under exactly the paired Frobenius budget of current R02.  The scheduled LR
and decoupled WD are each applied once by the unchanged parent transaction;
all internal LR/WD scales remain one.  The literal parent is evaluated first
and wins ties or any failed finiteness, descent, or budget certificate.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .rlb_r02_core import R02Core
from .rlb_r05_core import R05Core


class R09LossMetricCore(R02Core):
    """Current R02 plus an exact RLB group-span empirical-Fisher decision."""

    component_code = 8
    checkpoint_schema = "r09_r02_downstream_loss_metric_v1"
    inherited_parent = "current_r02_response_homotopy_chord"
    new_scientific_components = (
        "downstream_loss_metric_in_exact_rlb_group_tangent_span",
    )

    def __init__(self, pairs, **kwargs):
        pairs = list(pairs)
        self._r09_cotangent_records = [[] for _ in pairs]
        self._r09_loss_cotangents = None
        self._r09_clip_factor = None
        self._r09_span_metadata = None
        self._r09_output_hook_handles = []
        super().__init__(pairs, **kwargs)
        for index, pair in enumerate(self.pairs):
            self._r09_output_hook_handles.append(
                pair["mlp"].register_forward_hook(
                    self._make_loss_cotangent_hook(index)
                )
            )

    # Current R02 removes R05's aligned x/z/h packet because its own endpoint
    # does not consume it.  This generation restores the same deterministic
    # rows for a different loss-metric transaction; no R02 equation changes.
    def _make_input_hook(self, index):
        return R05Core._make_input_hook(self, index)

    def _make_feature_hook(self, index):
        return R05Core._make_feature_hook(self, index)

    def _make_loss_cotangent_hook(self, layer_index):
        def capture(_module, _inputs, output):
            if not _module.training:
                return
            if not torch.is_tensor(output) or not output.requires_grad:
                raise RuntimeError("R09 MLP output is not a differentiable tensor")
            flat = output.reshape(-1, self.external_width)
            rows = int(flat.shape[0])
            indices = self._functional_row_indices(rows, flat.device)

            def capture_cotangent(cotangent):
                with torch.no_grad():
                    value = cotangent.detach().reshape(
                        -1, self.external_width
                    )
                    if value.shape[0] != rows:
                        raise RuntimeError("R09 output/cotangent rows differ")
                    self._r09_cotangent_records[layer_index].append((
                        value.index_select(0, indices).clone(), rows,
                    ))
                return cotangent

            output.register_hook(capture_cotangent)

        return capture

    def record_realized_clipping(self, preclip_norm, max_norm):
        """Record PyTorch's exact global clipping coefficient once per step."""
        if self._r09_clip_factor is not None:
            raise RuntimeError("R09 observed multiple clipping calls")
        value = float(preclip_norm)
        maximum = float(max_norm)
        if not math.isfinite(value) or value < 0.0 or maximum != 1.0:
            raise RuntimeError("R09 received an invalid clipping certificate")
        self._r09_clip_factor = min(1.0, maximum / (value + 1.0e-6))

    def _consume_functional_samples(self):
        samples = R05Core._consume_functional_samples(self)
        if self._r09_clip_factor is None:
            raise RuntimeError("R09 did not receive realized global clipping")
        packets = []
        scale_min = float("inf")
        scale_max = 0.0
        for layer_index, records in enumerate(self._r09_cotangent_records):
            self._r09_cotangent_records[layer_index] = []
            if len(records) != self.expected_microbatches:
                raise RuntimeError(
                    f"R09 layer {layer_index} did not observe four cotangents"
                )
            scaled = []
            for value, rows in records:
                # CE is a mean over this microbatch and the trainer divides
                # by grad_accum.  This exact factor recovers the per-token
                # score before applying the realized global clipping scalar.
                loss_scale = float(rows * self.expected_microbatches)
                scale_min = min(scale_min, loss_scale)
                scale_max = max(scale_max, loss_scale)
                scaled.append(value.float() * loss_scale)
            combined = torch.cat(scaled, dim=0)
            local_rows = self.probe_capture_count * self.expected_microbatches
            if combined.shape != (local_rows, self.external_width):
                raise RuntimeError("R09 aligned cotangent inventory changed")
            numerators = torch.arange(
                self.probe_count,
                device=combined.device,
                dtype=torch.int64,
            ) * (local_rows - 1)
            selected = torch.div(
                numerators, self.probe_count - 1, rounding_mode="floor"
            )
            packets.append(combined.index_select(0, selected))
        self._r09_loss_cotangents = (
            torch.stack(packets) * float(self._r09_clip_factor)
        )
        self._r09_cotangent_scale_range = (scale_min, scale_max)
        return samples

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "downstream_loss_metric_lr_scale": 1.0,
            "exact_rlb_group_tangent_lr_scale": 1.0,
            "paired_group_span_budget_lr_scale": 1.0,
            "empirical_fisher_coordinate_lr_scale": 1.0,
            "parent_first_loss_model_lr_scale": 1.0,
            "weight_decay_cross_metric_scale": 1.0,
        })
        return report

    def _current_nesterov_stack(self, parameters, *, transpose=False):
        values = []
        for parameter in parameters:
            gradient = parameter.grad
            buffer = self.state[parameter].get("momentum_buffer")
            if gradient is None or buffer is None:
                raise RuntimeError("R09 requested an unformed R02 momentum")
            value = gradient.detach().float().lerp(
                buffer.detach().float(), self.momentum
            )
            values.append(value.transpose(-2, -1) if transpose else value)
        return torch.stack(values)

    def _group_tangent_images(
        self,
        inputs,
        preactivations,
        features,
        incoming_direction,
        outgoing_direction_transpose,
        *,
        factors,
    ):
        """Return exact joint RLB tangent images [layer,row,group,residual]."""
        layers = len(self.pairs)
        samples = inputs.shape[1]
        expected_matrix = (layers, self.hidden, self.external_width)
        if (
            inputs.shape != (layers, samples, self.external_width)
            or preactivations.shape != (layers, samples, self.hidden)
            or features.shape != (layers, samples, self.hidden)
            or incoming_direction.shape != expected_matrix
            or outgoing_direction_transpose.shape != expected_matrix
        ):
            raise RuntimeError("R09 functional group-span inventory changed")
        perturbation = torch.bmm(
            inputs, incoming_direction.transpose(-2, -1)
        )
        response = self._functional_jvp(
            preactivations, perturbation, factors=factors
        ).view(layers, samples, self.groups, self.width)
        outgoing_weights = torch.stack(self.outgoing).float().view(
            layers, self.external_width, self.groups, self.width
        ).permute(0, 2, 3, 1)
        incoming_image = torch.einsum(
            "lngw,lgwd->lngd", response, outgoing_weights
        )
        feature_blocks = features.view(
            layers, samples, self.groups, self.width
        )
        direction_blocks = outgoing_direction_transpose.view(
            layers, self.groups, self.width, self.external_width
        )
        outgoing_image = torch.einsum(
            "lngw,lgwd->lngd", feature_blocks, direction_blocks
        )
        return incoming_image + outgoing_image

    @staticmethod
    def _reduce_loss_metric(images, cotangents, common_decay_image):
        """Form the all-rank score Fisher and decay cross term."""
        if images.ndim != 4 or cotangents.ndim != 3:
            raise RuntimeError("R09 loss-metric tensor rank changed")
        layers, samples, groups, residual = images.shape
        if (
            cotangents.shape != (layers, samples, residual)
            or common_decay_image.shape != (layers, samples, residual)
        ):
            raise RuntimeError("R09 loss-metric aligned inventory changed")
        scores = torch.einsum("lngd,lnd->lng", images, cotangents)
        decay_scores = torch.einsum(
            "lnd,lnd->ln", common_decay_image, cotangents
        )
        fisher_sum = torch.einsum("lng,lnv->lgv", scores, scores)
        cross_sum = torch.einsum("lng,ln->lg", scores, decay_scores)
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
        return fisher, cross_sum / count, count

    @staticmethod
    def _quadratic_scores(coefficients, fisher, decay_cross, linear, eta):
        first = (linear * coefficients).sum(dim=-1)
        quadratic = torch.einsum(
            "lg,lgv,lv->l", coefficients, fisher, coefficients
        )
        cross = 2.0 * (decay_cross * coefficients).sum(dim=-1)
        return -float(eta) * first + 0.5 * float(eta) ** 2 * (
            quadratic + cross
        )

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
        """Solve the exact same-budget empirical-Fisher trust region."""
        if (
            fisher.ndim != 3
            or decay_cross.shape != exact_linear.shape
            or exact_linear.shape != momentum_linear.shape
            or exact_linear.shape != budget_weights.shape
            or fisher.shape[:2] != exact_linear.shape
            or fisher.shape[-1] != exact_linear.shape[-1]
        ):
            raise RuntimeError("R09 group-span solve inventory changed")
        if float(eta) <= 0.0:
            raise RuntimeError("R09 received a nonpositive scheduled LR")
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
        rhs = (
            exact_linear / float(eta) - decay_cross
        ) * inverse_root_weight
        eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
        spectral_scale = eigenvalues.abs().amax(dim=-1)
        rank_threshold = (
            machine * float(groups) * spectral_scale.clamp_min(tiny)
        )
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
        for _ in range(64):
            middle = 0.5 * (lower + upper)
            trial = coordinates / (
                eigenvalues + middle[:, None]
            ).clamp_min(rank_threshold[:, None])
            too_large = trial.square().sum(dim=-1) > parent_budget
            lower = torch.where(too_large, middle, lower)
            upper = torch.where(too_large, upper, middle)
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
        parent_coordinates = torch.einsum(
            "lgi,lg->li", eigenvectors, parent_q
        )
        signs = torch.sign(
            parent_coordinates.gather(
                1, first_minimum[:, None]
            ).squeeze(1)
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
        candidate_budget = (
            budget_weights * candidate.square()
        ).sum(dim=-1)
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
        candidate_momentum_descent = (
            momentum_linear * candidate
        ).sum(dim=-1)
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
        selected_score = torch.where(
            accepted, candidate_score, parent_score
        )
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
            force_parent,
        )
        cotangents = self._r09_loss_cotangents
        if any(value is None for value in (
            functional_inputs,
            functional_preactivations,
            functional_features,
            cotangents,
        )):
            raise RuntimeError("R09 did not receive aligned loss-metric rows")
        factors = self._functional_jvp_factors(functional_preactivations)
        images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_endpoint,
            outgoing_endpoint_transpose,
            factors=factors,
        )
        group_decay = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            torch.stack(self.incoming).float() * float(
                self.param_groups[0]["weight_decay"]
            ),
            torch.stack(self.outgoing).float().transpose(-2, -1) * float(
                self.param_groups[0]["weight_decay"]
            ),
            factors=factors,
        )
        fisher, decay_cross, global_count = self._reduce_loss_metric(
            images, cotangents, group_decay.sum(dim=2)
        )

        layers = len(self.pairs)
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
        exact_linear = (
            (incoming_gradients * incoming_blocks).sum(dim=(-2, -1))
            + (outgoing_gradients * outgoing_blocks).sum(dim=(-2, -1))
        )
        incoming_momentum = self._current_nesterov_stack(
            self.incoming
        ).view_as(incoming_blocks)
        outgoing_momentum = self._current_nesterov_stack(
            self.outgoing, transpose=True
        ).view_as(outgoing_blocks)
        momentum_linear = (
            (incoming_momentum * incoming_blocks).sum(dim=(-2, -1))
            + (outgoing_momentum * outgoing_blocks).sum(dim=(-2, -1))
        )
        budget_weights = (
            incoming_blocks.square().sum(dim=(-2, -1))
            + outgoing_blocks.square().sum(dim=(-2, -1))
        )
        coefficients, metadata = self._select_group_span_coefficients(
            fisher,
            decay_cross,
            exact_linear,
            momentum_linear,
            budget_weights,
            lr,
        )
        incoming_selected = (
            incoming_blocks * coefficients[:, :, None, None]
        ).reshape_as(incoming_endpoint)
        outgoing_selected = (
            outgoing_blocks * coefficients[:, :, None, None]
        ).reshape_as(outgoing_endpoint_transpose)
        metadata["global_count"] = global_count
        metadata["clip_factor"] = torch.tensor(
            float(self._r09_clip_factor),
            device=global_count.device,
            dtype=global_count.dtype,
        )
        self._r09_span_metadata = metadata

        choices = metadata["accepted"].to(torch.int64)
        scores = torch.stack((
            metadata["parent_score"],
            metadata["candidate_score"],
            metadata["parent_score"],
            metadata["candidate_score"],
        ), dim=-1)
        return incoming_selected, outgoing_selected, {
            "choices": choices,
            "scores": scores,
            "score_margin": (
                metadata["parent_score"] - metadata["candidate_score"]
            ).abs(),
            "energies": torch.zeros_like(scores),
            "global_count": global_count,
        }

    def _assert_r09_quiescent(self):
        if any(self._r09_cotangent_records):
            raise RuntimeError("R09 checkpoint encountered pending cotangents")
        if self._r09_loss_cotangents is not None:
            raise RuntimeError("R09 checkpoint encountered a materialized cotangent")
        if self._r09_clip_factor is not None:
            raise RuntimeError("R09 checkpoint encountered pending clipping")

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r09_span_metadata = None
        try:
            loss = super().step(closure)
            metadata = self._r09_span_metadata
            if metadata is None:
                raise RuntimeError("R09 did not execute its loss-metric transaction")
            if publish:
                accepted = metadata["accepted"]
                scale_min, scale_max = self._r09_cotangent_scale_range
                self._last_telemetry.update({
                    "rlb_r09_component_code": self.component_code,
                    "rlb_r09_parent_is_current_r02": 1,
                    "rlb_r09_group_count": int(self.groups),
                    "rlb_r09_group_width": int(self.width),
                    "rlb_r09_group_span_block_count": int(
                        len(self.pairs) * self.groups
                    ),
                    "rlb_r09_global_loss_sample_count": int(
                        metadata["global_count"].item()
                    ),
                    "rlb_r09_parent_layer_count": int(
                        (~accepted).sum().item()
                    ),
                    "rlb_r09_loss_metric_layer_count": int(
                        accepted.sum().item()
                    ),
                    "rlb_r09_fisher_rank_min": int(
                        metadata["rank"].amin().item()
                    ),
                    "rlb_r09_fisher_rank_median": int(
                        metadata["rank"].median().item()
                    ),
                    "rlb_r09_fisher_rank_max": int(
                        metadata["rank"].amax().item()
                    ),
                    "rlb_r09_fisher_eigenvalue_max": float(
                        metadata["eigenvalue_max"].amax().item()
                    ),
                    "rlb_r09_coefficient_min": float(
                        metadata["coefficient_min"].amin().item()
                    ),
                    "rlb_r09_coefficient_median": float(
                        metadata["coefficient_median"].median().item()
                    ),
                    "rlb_r09_coefficient_max": float(
                        metadata["coefficient_max"].amax().item()
                    ),
                    "rlb_r09_exact_descent_min": float(
                        metadata["selected_exact_descent"].amin().item()
                    ),
                    "rlb_r09_momentum_descent_min": float(
                        metadata["selected_momentum_descent"].amin().item()
                    ),
                    "rlb_r09_budget_residual_max": float(
                        metadata["budget_residual"].amax().item()
                    ),
                    "rlb_r09_surrogate_improvement_min": float(
                        metadata["improvement"].amin().item()
                    ),
                    "rlb_r09_surrogate_improvement_median": float(
                        metadata["improvement"].median().item()
                    ),
                    "rlb_r09_surrogate_improvement_max": float(
                        metadata["improvement"].amax().item()
                    ),
                    "rlb_r09_realized_clip_factor": float(
                        metadata["clip_factor"].item()
                    ),
                    "rlb_r09_per_token_cotangent_scale_min": scale_min,
                    "rlb_r09_per_token_cotangent_scale_max": scale_max,
                    "rlb_r09_structural_matrix_elements": 245_366_784,
                })
            return loss
        finally:
            self._r09_loss_cotangents = None
            self._r09_clip_factor = None

    def state_dict(self):
        self._assert_r09_quiescent()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r09_cotangent_records = [[] for _ in self.pairs]
        self._r09_loss_cotangents = None
        self._r09_clip_factor = None
        self._r09_span_metadata = None
        return result
