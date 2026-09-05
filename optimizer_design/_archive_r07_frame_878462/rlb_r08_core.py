"""Staged complete-Kronecker response LMO for the Global-RLB factor pair.

This file intentionally lives outside the repository while the frozen R09
job imports the current R08 capture implementation.  It is promoted into the
R08 slot only after that job releases its source seal.

For one Global-RLB group, the incoming momentum has the local quadratic
factors

    H_g = E[J_g^T B_g^T B_g J_g],    C_x = E[X^T X],

while the outgoing momentum has the rational-feature factor

    K_g = E[F_g F_g^T].

Cholesky coordinate factors and their exact adjoints surround separate NS5
polar decisions.  Every metric is normalized to unit determinant.  The
external LR, WD, Nesterov recurrence, NS5 map, and Muon shape calibration are
unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .rlb_group_muon_core import _batched_zero_power
from .rlb_response_capture_core import RLBResponseCaptureCore


class R08RevisionCore(RLBResponseCaptureCore):
    """Role-specific RLB response LMOs with complete incoming curvature."""

    def __init__(
        self,
        pairs,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
    ):
        super().__init__(
            pairs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
        )
        # Across four microbatches and four data-parallel ranks this gives
        # exactly one current input sample per residual coordinate:
        # 64 * 4 * 4 = 1024.  The count is fixed by the installed 256-channel
        # rational group width and the contracted accumulation/world sizes.
        if self.width % self.expected_microbatches:
            raise ValueError("R08 revision requires group width divisible by accumulation")
        self.input_capture_count = self.width // self.expected_microbatches
        exact_m1_signature = (self.groups, self.width, self.external_width) == (
            18,
            256,
            1024,
        )
        if exact_m1_signature and (
            self.input_capture_count * self.expected_microbatches * 4
            != self.external_width
        ):
            raise ValueError("R08 revision requires the exact four-rank M1 sample inventory")
        self._input_records = [[] for _ in self.pairs]
        self._input_hook_handles = []
        for index, pair in enumerate(self.pairs):
            self._input_hook_handles.append(
                pair["mlp"].register_forward_pre_hook(
                    self._make_input_hook(index)
                )
            )
        self._last_input_record_count = 0
        self._last_input_local_sample_count = 0

    def _sample_input_rows(self, value):
        flat = value.detach().reshape(-1, self.external_width)
        if flat.shape[0] < self.input_capture_count:
            raise RuntimeError("training tensor is smaller than the R08 input sample")
        if self.input_capture_count == 1:
            indices = torch.zeros(1, device=flat.device, dtype=torch.int64)
        else:
            numerators = torch.arange(
                self.input_capture_count,
                device=flat.device,
                dtype=torch.int64,
            ) * (flat.shape[0] - 1)
            indices = torch.div(
                numerators,
                self.input_capture_count - 1,
                rounding_mode="floor",
            )
        return flat.index_select(0, indices).clone()

    def _make_input_hook(self, index):
        @torch.no_grad()
        def capture(module, inputs):
            if not module.training:
                return
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("R08 input hook received an invalid MLP input")
            self._input_records[index].append(
                self._sample_input_rows(inputs[0])
            )

        return capture

    @torch.no_grad()
    def record_input_batch(self, layer_index, inputs):
        """Provide the same deterministic records in tests and preflight."""
        self._input_records[layer_index].append(self._sample_input_rows(inputs))

    def _consume_input_moments(self):
        samples = []
        for index, records in enumerate(self._input_records):
            if len(records) != self.expected_microbatches:
                raise RuntimeError(
                    f"R08 layer {index} did not observe four input microbatches"
                )
            combined = torch.cat(records, dim=0).float()
            self._input_records[index] = []
            expected = self.input_capture_count * self.expected_microbatches
            if combined.shape != (expected, self.external_width):
                raise RuntimeError("R08 input sample inventory changed")
            samples.append(combined)
        sample_batch = torch.stack(samples)
        moments = torch.bmm(sample_batch.transpose(1, 2), sample_batch)
        counts = torch.full(
            (len(self.pairs),),
            float(sample_batch.shape[1]),
            device=sample_batch.device,
            dtype=sample_batch.dtype,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        moments.div_(counts[:, None, None])
        self._last_input_record_count = self.expected_microbatches
        self._last_input_local_sample_count = int(sample_batch.shape[1])
        return moments, counts

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "incoming_lr_scale": 1.0,
            "outgoing_lr_scale": 1.0,
            "incoming_response_metric_lr_scale": 1.0,
            "residual_input_metric_lr_scale": 1.0,
            "outgoing_feature_metric_lr_scale": 1.0,
            "separate_role_norm_lr_scale": 1.0,
            "unit_volume_lr_scale": 1.0,
            "polar_lr_scale": 1.0,
            "phase_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    @staticmethod
    def _left_coordinate(lower, volume, value):
        result = torch.linalg.solve_triangular(lower, value, upper=False)
        return result * volume[..., None, None]

    @staticmethod
    def _left_adjoint(lower, volume, value):
        result = torch.linalg.solve_triangular(
            lower.transpose(-2, -1), value, upper=True
        )
        return result * volume[..., None, None]

    @staticmethod
    def _right_coordinate(lower, volume, value):
        result = torch.linalg.solve_triangular(
            lower, value.transpose(-2, -1), upper=False
        ).transpose(-2, -1)
        return result * volume[..., None, None]

    @staticmethod
    def _right_adjoint(lower, volume, value):
        result = torch.linalg.solve_triangular(
            lower.transpose(-2, -1),
            value.transpose(-2, -1),
            upper=True,
        ).transpose(-2, -1)
        return result * volume[..., None, None]

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("R08 revision refuses a nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])

        incoming_metrics = torch.stack([
            self._layer_response_metric(index) for index in range(len(self.pairs))
        ])
        feature_items = [
            self._consume_feature_moment(index) for index in range(len(self.pairs))
        ]
        outgoing_metrics = torch.stack([item[0] for item in feature_items])
        outgoing_counts = torch.tensor(
            [item[1] for item in feature_items],
            device=outgoing_metrics.device,
            dtype=outgoing_metrics.dtype,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(incoming_metrics, op=dist.ReduceOp.SUM)
            incoming_metrics.div_(dist.get_world_size())
            dist.all_reduce(outgoing_metrics, op=dist.ReduceOp.SUM)
            dist.all_reduce(outgoing_counts, op=dist.ReduceOp.SUM)
        outgoing_metrics.div_(outgoing_counts[:, None, None, None])
        input_metrics, input_counts = self._consume_input_moments()

        (
            incoming_lower,
            incoming_volume,
            hidden_shift,
            _,
            incoming_volume_residual,
        ) = self._unit_volume_cholesky(incoming_metrics, capture_spectrum=False)
        (
            outgoing_lower,
            outgoing_volume,
            _,
            _,
            outgoing_volume_residual,
        ) = self._unit_volume_cholesky(outgoing_metrics, capture_spectrum=False)
        (
            input_lower,
            input_volume,
            input_shift,
            _,
            input_volume_residual,
        ) = self._unit_volume_cholesky(input_metrics, capture_spectrum=False)

        incoming_momenta = torch.stack(
            [self._nesterov(parameter) for parameter in self.incoming]
        ).float()
        outgoing_momenta = torch.stack(
            [self._nesterov(parameter) for parameter in self.outgoing]
        ).float()
        layer_count = len(self.pairs)

        incoming_blocks = incoming_momenta.view(
            layer_count, self.groups, self.width, self.external_width
        )
        incoming_work_blocks = self._left_coordinate(
            incoming_lower, incoming_volume, incoming_blocks
        )
        incoming_work = self._right_coordinate(
            input_lower,
            input_volume,
            incoming_work_blocks.reshape_as(incoming_momenta),
        )
        incoming_polar = _batched_zero_power(
            incoming_work, self.ns_steps
        ).float()
        incoming_pullback = self._right_adjoint(
            input_lower, input_volume, incoming_polar
        )
        incoming_direction = self._left_adjoint(
            incoming_lower,
            incoming_volume,
            incoming_pullback.view_as(incoming_blocks),
        ).reshape_as(incoming_momenta)

        outgoing_transpose = outgoing_momenta.transpose(-2, -1)
        outgoing_blocks = outgoing_transpose.view(
            layer_count, self.groups, self.width, self.external_width
        )
        outgoing_work_blocks = self._left_coordinate(
            outgoing_lower, outgoing_volume, outgoing_blocks
        )
        outgoing_work = outgoing_work_blocks.reshape_as(outgoing_transpose)
        outgoing_polar = _batched_zero_power(
            outgoing_work, self.ns_steps
        ).float()
        outgoing_direction_transpose = self._left_adjoint(
            outgoing_lower,
            outgoing_volume,
            outgoing_polar.view_as(outgoing_blocks),
        ).reshape_as(outgoing_transpose)
        outgoing_direction = outgoing_direction_transpose.transpose(-2, -1)

        matched_scale = 0.2 * math.sqrt(
            max(self.hidden, self.external_width)
        )
        incoming_direction.mul_(matched_scale)
        outgoing_direction.mul_(matched_scale)

        incoming_descent = (
            incoming_momenta * incoming_direction
        ).sum(dim=(-2, -1))
        outgoing_descent = (
            outgoing_momenta * outgoing_direction
        ).sum(dim=(-2, -1))
        incoming_coordinate_descent = (
            incoming_work * incoming_polar
        ).sum(dim=(-2, -1)) * matched_scale
        outgoing_coordinate_descent = (
            outgoing_work * outgoing_polar
        ).sum(dim=(-2, -1)) * matched_scale
        incoming_adjoint_residual = (
            (incoming_descent - incoming_coordinate_descent).abs()
            / incoming_coordinate_descent.abs().clamp_min(1.0e-20)
        )
        outgoing_adjoint_residual = (
            (outgoing_descent - outgoing_coordinate_descent).abs()
            / outgoing_coordinate_descent.abs().clamp_min(1.0e-20)
        )
        if (
            torch.any(incoming_descent <= 0.0)
            or torch.any(outgoing_descent <= 0.0)
            or torch.any(~torch.isfinite(incoming_direction))
            or torch.any(~torch.isfinite(outgoing_direction))
        ):
            raise RuntimeError("R08 complete-Kronecker LMO lost finite descent")
        if (
            torch.any(incoming_adjoint_residual > 3.0e-4)
            or torch.any(outgoing_adjoint_residual > 3.0e-4)
        ):
            raise RuntimeError("R08 complete-Kronecker adjoint identity was lost")

        for index, (incoming, outgoing) in enumerate(
            zip(self.incoming, self.outgoing)
        ):
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming.add_(incoming_direction[index].to(incoming.dtype), alpha=-lr)
            outgoing.add_(outgoing_direction[index].to(outgoing.dtype), alpha=-lr)

        if self._capture_telemetry_next_step:
            self._last_telemetry = {
                "rlb_r08_group_count": int(self.groups),
                "rlb_r08_group_width": int(self.width),
                "rlb_r08_probe_microbatches": int(
                    self._last_probe_record_count
                ),
                "rlb_r08_feature_microbatches": int(
                    self._last_feature_record_count
                ),
                "rlb_r08_feature_local_sample_count": int(
                    self._last_feature_sample_count
                ),
                "rlb_r08_input_capture_per_microbatch": int(
                    self.input_capture_count
                ),
                "rlb_r08_input_microbatches": int(
                    self._last_input_record_count
                ),
                "rlb_r08_input_local_sample_count": int(
                    self._last_input_local_sample_count
                ),
                "rlb_r08_input_global_sample_count": int(
                    input_counts[0].item()
                ),
                "rlb_r08_hidden_precision_shift_relative": float(hidden_shift),
                "rlb_r08_input_precision_shift_relative": float(input_shift),
                "rlb_r08_incoming_volume_residual_max": float(
                    incoming_volume_residual.amax().item()
                ),
                "rlb_r08_outgoing_volume_residual_max": float(
                    outgoing_volume_residual.amax().item()
                ),
                "rlb_r08_input_volume_residual_max": float(
                    input_volume_residual.amax().item()
                ),
                "rlb_r08_incoming_descent_min": float(
                    incoming_descent.amin().item()
                ),
                "rlb_r08_outgoing_descent_min": float(
                    outgoing_descent.amin().item()
                ),
                "rlb_r08_adjoint_residual_max": float(
                    torch.maximum(
                        incoming_adjoint_residual,
                        outgoing_adjoint_residual,
                    ).amax().item()
                ),
            }
        self._capture_telemetry_next_step = False
        return loss
