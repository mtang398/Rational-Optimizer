"""Exact-RLB group transaction with a fixed-probe, owner-free row-space solve.

This class preserves the archived R01 parent direction, exact P5/Q4 tangent
scores, equality Frobenius budget, role/layer descent certificates, LR, WD,
momentum, and NS5 cell.  It changes only the realization of the global loss
transaction:

* 144 probes are a global method constant, not a per-rank or per-token count;
* no ``(L G) x (L G)`` Fisher is materialized;
* the solve is fixed-probe row-space algebra;
* endpoint DDP uses scalar score/coordinate exchanges only;
* native TP/FSDP implementations use the same column-sharded kernel directly.

The current endpoint trainer is replicated DDP, so this module uses the DDP
adapter.  It never communicates a matrix source or selected matrix update.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_fixed_probe_transaction import replicated_fixed_probe_transaction
from .rlb_fixed_global_probe_layout import (
    evenly_spaced_indices,
    fixed_global_probe_layout,
    required_capture_rows_per_microbatch,
)
from .rlb_owner_free_strict_local import OwnerFreeStrictLocalR01Optimizer


FIXED_GLOBAL_PROBE_COUNT = 144
METHOD_ID = "owner_free_fixed_probe_exact_r01_v1"


class OwnerFreeFixedProbeR01Optimizer(OwnerFreeStrictLocalR01Optimizer):
    """Archived exact R01 direction with a scalable global transaction."""

    component_code = 101
    checkpoint_schema = "owner_free_fixed_probe_exact_r01_v1"
    inherited_parent = "current_r02_response_homotopy_chord"
    new_scientific_components = (
        "globally_fixed_loss_probe_measure",
        "owner_free_column_sharded_row_space_transaction",
    )
    execution_variant = METHOD_ID

    def __init__(
        self,
        *args,
        global_probe_count=FIXED_GLOBAL_PROBE_COUNT,
        loss_probe_group=None,
        **kwargs,
    ):
        if int(global_probe_count) != FIXED_GLOBAL_PROBE_COUNT:
            raise ValueError("the fixed-probe R01 measure must contain 144 rows")
        self.fixed_global_probe_count = int(global_probe_count)
        self.loss_probe_group = loss_probe_group
        self._loss_probe_layout = None
        self._loss_probe_capture_count = None
        self._fixed_probe_metadata = None
        super().__init__(*args, **kwargs)

    def _ensure_loss_probe_layout(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank(group=self.loss_probe_group)
            world = dist.get_world_size(group=self.loss_probe_group)
        else:
            rank, world = 0, 1
        layout = fixed_global_probe_layout(
            self.fixed_global_probe_count, rank, world
        )
        if self._loss_probe_layout is None:
            self._loss_probe_layout = layout
            self._loss_probe_capture_count = (
                required_capture_rows_per_microbatch(
                    layout.local_probe_count,
                    self.expected_microbatches,
                    minimum_capture_rows=self.probe_capture_count,
                )
            )
        elif self._loss_probe_layout != layout:
            raise RuntimeError("fixed global probe topology changed mid-transaction")
        return self._loss_probe_layout

    def _functional_row_indices(self, row_count, device):
        """Capture enough rows for this rank's fixed global probe shard."""

        self._ensure_loss_probe_layout()
        capture = int(self._loss_probe_capture_count)
        if int(row_count) < capture:
            raise RuntimeError("fixed global functional capture exceeds microbatch")
        return evenly_spaced_indices(
            int(row_count), capture, device=device
        )

    def _consume_functional_samples(self):
        """Align a globally fixed, uneven probe partition with cotangents."""

        layout = self._ensure_loss_probe_layout()
        if self._r09_clip_factor is None:
            raise RuntimeError("fixed global probes lack realized global clipping")
        packets = []
        cotangent_packets = []
        scale_min = float("inf")
        scale_max = 0.0
        capture = int(self._loss_probe_capture_count)
        captured_rows = capture * self.expected_microbatches
        selected = evenly_spaced_indices(
            captured_rows,
            layout.local_probe_count,
            device=self.incoming[0].device,
        )
        for layer_index, records in enumerate(self._functional_records):
            pending = self._functional_pending_inputs[layer_index]
            self._functional_pending_inputs[layer_index] = None
            self._functional_records[layer_index] = []
            cotangent_records = self._r09_cotangent_records[layer_index]
            self._r09_cotangent_records[layer_index] = []
            if pending is not None:
                raise RuntimeError("fixed global functional input remained unmatched")
            if (
                len(records) != self.expected_microbatches
                or len(cotangent_records) != self.expected_microbatches
            ):
                raise RuntimeError("fixed global probes lost an accumulation row")
            inputs = torch.cat([record[0] for record in records], dim=0)
            preactivations = torch.cat([record[1] for record in records], dim=0)
            features = torch.cat([record[2] for record in records], dim=0)
            if (
                inputs.shape != (captured_rows, self.external_width)
                or preactivations.shape != (captured_rows, self.hidden)
                or features.shape != (captured_rows, self.hidden)
            ):
                raise RuntimeError("fixed global x/z/h capture inventory changed")
            scaled_cotangents = []
            for value, original_rows in cotangent_records:
                loss_scale = float(original_rows * self.expected_microbatches)
                scale_min = min(scale_min, loss_scale)
                scale_max = max(scale_max, loss_scale)
                scaled_cotangents.append(value.float() * loss_scale)
            cotangents = torch.cat(scaled_cotangents, dim=0)
            if cotangents.shape != (captured_rows, self.external_width):
                raise RuntimeError("fixed global cotangent capture inventory changed")
            packet = (
                inputs.index_select(0, selected).float(),
                preactivations.index_select(0, selected).float(),
                features.index_select(0, selected).float(),
            )
            self._functional_packets[layer_index] = packet
            packets.append(packet)
            cotangent_packets.append(
                cotangents.index_select(0, selected)
            )
        self._r09_loss_cotangents = (
            torch.stack(cotangent_packets) * float(self._r09_clip_factor)
        )
        self._r09_cotangent_scale_range = (scale_min, scale_max)
        stacked = tuple(torch.stack(items) for items in zip(*packets))
        self._functional_packets = [None for _ in self.pairs]
        return stacked

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "fixed_probe_measure_lr_scale": 1.0,
            "owner_free_row_space_lr_scale": 1.0,
            "column_sharded_coefficient_lr_scale": 1.0,
            "selected_update_publication_lr_scale": 1.0,
        })
        return report

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
            raise RuntimeError("fixed-probe R01 lacks aligned functional rows")
        layers = len(self.pairs)
        if force_parent.shape != (layers,):
            raise RuntimeError("fixed-probe R01 parent-limit inventory changed")

        factors = self._functional_jvp_factors(functional_preactivations)
        endpoint_score_images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_endpoint,
            outgoing_endpoint_transpose,
            factors=factors,
        )
        weight_decay = float(self.param_groups[0]["weight_decay"])
        decay_score_images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            torch.stack(self.incoming).float() * weight_decay,
            torch.stack(self.outgoing).float().transpose(-2, -1) * weight_decay,
            factors=factors,
        )
        expected_scores = (
            layers, functional_inputs.shape[1], self.groups, 1
        )
        if (
            endpoint_score_images.shape != expected_scores
            or decay_score_images.shape != expected_scores
        ):
            raise RuntimeError("fixed-probe direct-score inventory changed")
        local_scores = endpoint_score_images[..., 0].permute(
            1, 0, 2
        ).reshape(functional_inputs.shape[1], layers * self.groups)
        local_decay_action = decay_score_images[..., 0].sum(dim=(0, 2))

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
        exact_by_role = torch.stack((
            incoming_exact.reshape(-1),
            outgoing_exact.reshape(-1),
        ))
        momentum_by_role = torch.stack((
            incoming_momentum_linear.reshape(-1),
            outgoing_momentum_linear.reshape(-1),
        ))
        flat_budget = budget_weights.reshape(-1)
        layer_ids = torch.arange(
            layers,
            device=local_scores.device,
            dtype=torch.int64,
        ).repeat_interleave(self.groups)
        transaction = replicated_fixed_probe_transaction(
            local_scores,
            local_decay_action,
            exact_by_role,
            momentum_by_role,
            flat_budget,
            layer_ids,
            global_probe_count=self.fixed_global_probe_count,
            total_layers=layers,
            eta=float(lr),
            group=self.loss_probe_group,
        )
        sharded = transaction.sharded_result
        candidate_coefficients = transaction.candidate_coefficients.view(
            layers, self.groups
        )

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
            sharded.accepted[0]
            & role_descent_valid
            & (~force_parent.any())
        )
        coefficients = torch.where(
            accepted,
            candidate_coefficients,
            torch.ones_like(candidate_coefficients),
        )
        incoming_selected = (
            incoming_blocks * coefficients[:, :, None, None]
        ).reshape_as(incoming_endpoint)
        outgoing_selected = (
            outgoing_blocks * coefficients[:, :, None, None]
        ).reshape_as(outgoing_endpoint_transpose)

        candidate_exact = (
            exact_by_role.sum(dim=0) * candidate_coefficients.reshape(-1)
        ).sum().reshape(1)
        candidate_momentum = (
            momentum_by_role.sum(dim=0) * candidate_coefficients.reshape(-1)
        ).sum().reshape(1)
        parent_exact = exact_by_role.sum().reshape(1)
        parent_momentum = momentum_by_role.sum().reshape(1)
        selected_exact = torch.where(accepted, candidate_exact, parent_exact)
        selected_momentum = torch.where(
            accepted, candidate_momentum, parent_momentum
        )
        selected_score = torch.where(
            accepted, sharded.candidate_score, sharded.parent_score
        )
        global_count = torch.tensor(
            float(self.fixed_global_probe_count),
            device=local_scores.device,
            dtype=local_scores.dtype,
        )
        metadata = {
            "accepted": accepted.reshape(1),
            "rank": sharded.rank,
            "eigenvalue_max": sharded.eigenvalue_max.to(local_scores.dtype),
            "coefficient_min": transaction.candidate_coefficients.amin().reshape(1),
            "coefficient_median": transaction.candidate_coefficients.median().reshape(1),
            "coefficient_max": transaction.candidate_coefficients.amax().reshape(1),
            "candidate_exact_descent": candidate_exact,
            "candidate_momentum_descent": candidate_momentum,
            "selected_exact_descent": selected_exact,
            "selected_momentum_descent": selected_momentum,
            "parent_score": sharded.parent_score.to(local_scores.dtype),
            "candidate_score": sharded.candidate_score.to(local_scores.dtype),
            "selected_score": selected_score.to(local_scores.dtype),
            "budget_residual": sharded.budget_residual.to(local_scores.dtype),
            "improvement": (
                sharded.parent_score.to(local_scores.dtype)
                - selected_score.to(local_scores.dtype)
            ),
            "global_count": global_count,
            "clip_factor": torch.tensor(
                float(self._r09_clip_factor),
                device=local_scores.device,
                dtype=local_scores.dtype,
            ),
            "cross_layer_coupling_ratio": (
                transaction.cross_layer_coupling_ratio.to(local_scores.dtype)
            ),
            "selected_coefficient_min": coefficients.amin(),
            "selected_coefficient_median": coefficients.median(),
            "selected_coefficient_max": coefficients.amax(),
            "selected_incoming_exact_descent_min": torch.where(
                accepted,
                candidate_incoming_exact.amin(),
                incoming_exact.sum(dim=-1).amin(),
            ),
            "selected_outgoing_exact_descent_min": torch.where(
                accepted,
                candidate_outgoing_exact.amin(),
                outgoing_exact.sum(dim=-1).amin(),
            ),
            "selected_incoming_momentum_descent_min": torch.where(
                accepted,
                candidate_incoming_momentum.amin(),
                incoming_momentum_linear.sum(dim=-1).amin(),
            ),
            "selected_outgoing_momentum_descent_min": torch.where(
                accepted,
                candidate_outgoing_momentum.amin(),
                outgoing_momentum_linear.sum(dim=-1).amin(),
            ),
        }
        self._r01_global_metadata = metadata
        self._fixed_probe_metadata = {
            "global_probe_count": self.fixed_global_probe_count,
            "local_probe_count": transaction.local_probe_count,
            "collective_rounds": transaction.collective_rounds,
            "score_scalars_exchanged_per_rank": (
                transaction.score_scalars_exchanged_per_rank
            ),
            "coefficient_scalars_exchanged_per_rank": (
                transaction.coefficient_scalars_exchanged_per_rank
            ),
            "selected_update_elements_published": 0,
            "dense_coordinate_metric_elements": 0,
            "complete_layer_owners": 0,
            "method_state_depends_on_total_tokens": False,
            "method_state_depends_on_machine_count": False,
            "probe_process_count": self._loss_probe_layout.process_count,
            "probe_process_rank": self._loss_probe_layout.process_rank,
            "local_fixed_probe_count": self._loss_probe_layout.local_probe_count,
        }

        repeated = lambda value: value.reshape(1).expand(layers)
        self._r09_span_metadata = {
            "accepted": repeated(accepted),
            "rank": repeated(metadata["rank"][0]),
            "eigenvalue_max": repeated(metadata["eigenvalue_max"][0]),
            "coefficient_min": repeated(metadata["coefficient_min"][0]),
            "coefficient_median": repeated(metadata["coefficient_median"][0]),
            "coefficient_max": repeated(metadata["coefficient_max"][0]),
            "selected_exact_descent": repeated(selected_exact[0]),
            "selected_momentum_descent": repeated(selected_momentum[0]),
            "budget_residual": repeated(metadata["budget_residual"][0]),
            "improvement": repeated(metadata["improvement"][0]),
            "global_count": global_count,
            "clip_factor": metadata["clip_factor"],
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
        parent_score = metadata["parent_score"][0]
        candidate_score = metadata["candidate_score"][0]
        choice_scores = torch.stack((
            parent_score, candidate_score, parent_score, candidate_score
        )).reshape(1, 4).expand(layers, 4)
        score_margin = (parent_score - candidate_score).abs().expand(layers)
        return incoming_selected, outgoing_selected, {
            "choices": choices,
            "scores": choice_scores,
            "score_margin": score_margin,
            "energies": torch.zeros_like(choice_scores),
            "global_count": global_count,
        }


__all__ = (
    "FIXED_GLOBAL_PROBE_COUNT",
    "METHOD_ID",
    "OwnerFreeFixedProbeR01Optimizer",
)
