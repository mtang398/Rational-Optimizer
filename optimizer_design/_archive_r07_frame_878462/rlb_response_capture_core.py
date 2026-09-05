"""Shared capture and analytic-Jacobian machinery for Global-RLB optimizers.

This module contains no optimizer proposal.  It deterministically records the
current P5/Q4 preactivations and features and exposes their exact response
metrics to the opaque candidate that subclasses it.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class RLBResponseCaptureCore(torch.optim.Optimizer):
    def __init__(
        self,
        pairs,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
    ):
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("R08 requires complete Global-RLB layers")
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        if self.momentum != 0.95 or self.ns_steps != 5:
            raise ValueError("R08 requires the matched Muon recurrence")

        incoming = []
        outgoing = []
        signature = None
        seen = set()
        for pair in self.pairs:
            in_weight = pair["in_weight"]
            out_weight = pair["out_weight"]
            numerator = pair["numerator"]
            denominator = pair["denominator"]
            groups = int(pair["groups"])
            hidden = int(pair["hidden_dim"])
            if hidden % groups != 0:
                raise ValueError("Global-RLB hidden width is not group divisible")
            width = hidden // groups
            current = (
                groups,
                width,
                in_weight.shape[1],
                out_weight.shape[0],
                float(pair["eps"]),
            )
            if in_weight.shape != (hidden, current[2]):
                raise ValueError("invalid Global-RLB incoming matrix")
            if out_weight.shape != (current[3], hidden) or current[2] != current[3]:
                raise ValueError("invalid Global-RLB outgoing matrix")
            if numerator.shape != (groups, 6) or denominator.shape != (groups, 4):
                raise ValueError("R08 requires the installed grouped P5/Q4")
            if signature is None:
                signature = current
            elif signature != current:
                raise ValueError("Global-RLB structural shapes differ across layers")
            for parameter in (in_weight, out_weight):
                if id(parameter) in seen:
                    raise ValueError("RLB matrix occurs in multiple layers")
                seen.add(id(parameter))
            incoming.append(in_weight)
            outgoing.append(out_weight)

        self.groups, self.width, self.external_width, _, self.rlb_eps = signature
        self.hidden = self.groups * self.width
        self.incoming = incoming
        self.outgoing = outgoing
        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_scale": 1.0,
        }
        super().__init__([{"params": incoming + outgoing}], defaults)

        self.probe_capture_count = self.groups
        self.probe_count = 2 * self.groups
        self.feature_capture_count = 2 * self.width
        self.expected_microbatches = 4
        self._probe_records = [[] for _ in self.pairs]
        self._feature_moment_sums = [None for _ in self.pairs]
        self._feature_sample_counts = [0 for _ in self.pairs]
        self._feature_record_counts = [0 for _ in self.pairs]
        self._hook_handles = []
        for index, pair in enumerate(self.pairs):
            self._hook_handles.append(
                pair["module"].register_forward_pre_hook(
                    self._make_probe_hook(index)
                )
            )
            self._hook_handles.append(
                pair["module"].register_forward_hook(
                    self._make_feature_hook(index)
                )
            )
        self._last_probe_record_count = 0
        self._last_feature_record_count = 0
        self._last_feature_sample_count = 0
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def _make_probe_hook(self, index):
        @torch.no_grad()
        def capture(module, inputs):
            if not module.training:
                return
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("R08 hook received invalid preactivation input")
            flat = inputs[0].detach().reshape(-1, self.hidden)
            if flat.shape[0] < self.probe_capture_count:
                raise RuntimeError("training activation is smaller than the R08 probe")
            numerators = torch.arange(
                self.probe_capture_count, device=flat.device, dtype=torch.int64
            ) * (flat.shape[0] - 1)
            indices = torch.div(
                numerators, self.probe_capture_count - 1, rounding_mode="floor"
            )
            self._probe_records[index].append(flat.index_select(0, indices).clone())

        return capture

    def _make_feature_hook(self, index):
        @torch.no_grad()
        def capture(module, inputs, output):
            if not module.training:
                return
            if not torch.is_tensor(output):
                raise RuntimeError("R08 hook received invalid rational features")
            flat = output.detach().reshape(-1, self.hidden)
            if flat.shape[0] < self.feature_capture_count:
                raise RuntimeError("training activation is smaller than the R08 feature sample")
            numerators = torch.arange(
                self.feature_capture_count, device=flat.device, dtype=torch.int64
            ) * (flat.shape[0] - 1)
            indices = torch.div(
                numerators, self.feature_capture_count - 1, rounding_mode="floor"
            )
            features = flat.index_select(0, indices).float().view(
                self.feature_capture_count, self.groups, self.width
            )
            moment = torch.einsum("ngw,ngv->gwv", features, features)
            if self._feature_moment_sums[index] is None:
                self._feature_moment_sums[index] = moment
            else:
                self._feature_moment_sums[index].add_(moment)
            self._feature_sample_counts[index] += self.feature_capture_count
            self._feature_record_counts[index] += 1

        return capture

    def _consume_probe(self, layer_index):
        records = self._probe_records[layer_index]
        if not records:
            raise RuntimeError("R08 has no training preactivation probe")
        combined = torch.cat(records, dim=0)
        self._probe_records[layer_index] = []
        if len(records) != self.expected_microbatches:
            raise RuntimeError("R08 requires the exact four accumulation microbatches")
        if combined.shape[0] < self.probe_count:
            raise RuntimeError("R08 did not observe the complete preactivation probe")
        numerators = torch.arange(
            self.probe_count, device=combined.device, dtype=torch.int64
        ) * (combined.shape[0] - 1)
        indices = torch.div(
            numerators, self.probe_count - 1, rounding_mode="floor"
        )
        self._last_probe_record_count = len(records)
        return combined.index_select(0, indices)

    def _consume_feature_moment(self, layer_index):
        moment = self._feature_moment_sums[layer_index]
        sample_count = self._feature_sample_counts[layer_index]
        record_count = self._feature_record_counts[layer_index]
        self._feature_moment_sums[layer_index] = None
        self._feature_sample_counts[layer_index] = 0
        self._feature_record_counts[layer_index] = 0
        if moment is None or sample_count <= 0:
            raise RuntimeError("R08 has no training rational-feature moment")
        if record_count != self.expected_microbatches:
            raise RuntimeError("R08 requires the exact four accumulation microbatches")
        self._last_feature_record_count = record_count
        self._last_feature_sample_count = sample_count
        return moment, sample_count

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "incoming_lr_scale": 1.0,
            "outgoing_lr_scale": 1.0,
            "paired_response_metric_lr_scale": 1.0,
            "joint_role_norm_lr_scale": 1.0,
            "unit_volume_lr_scale": 1.0,
            "polar_lr_scale": 1.0,
            "phase_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def set_telemetry_capture(self, enabled: bool = True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def _nesterov(self, parameter):
        if parameter.grad is None:
            raise RuntimeError("R08 matrix gradient is missing")
        state = self.state[parameter]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(parameter)
            state["momentum_buffer"] = buffer
        buffer.lerp_(parameter.grad, 1.0 - self.momentum)
        return parameter.grad.lerp(buffer, self.momentum)

    def _layer_response_metric(self, layer_index):
        z = self._consume_probe(layer_index).float().view(
            self.probe_count, self.groups, self.width
        )
        pair = self.pairs[layer_index]
        rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + self.rlb_eps)
        t = z / rms
        t2 = t.square()
        t3 = t2 * t
        t4 = t2.square()
        t5 = t4 * t
        abs_t = t.abs()
        numerator = pair["numerator"].float().view(1, self.groups, 1, 6)
        denominator = pair["denominator"].float().abs().view(
            1, self.groups, 1, 4
        )
        powers = torch.stack(
            (torch.ones_like(t), t, t2, t3, t4, t5), dim=-1
        )
        denominator_powers = torch.stack((abs_t, t2, abs_t * t2, t4), dim=-1)
        derivative_powers = torch.stack(
            (
                torch.zeros_like(t),
                torch.ones_like(t),
                2.0 * t,
                3.0 * t2,
                4.0 * t3,
                5.0 * t4,
            ),
            dim=-1,
        )
        denominator_derivative_powers = torch.stack(
            (torch.sign(t), 2.0 * t, 3.0 * t * abs_t, 4.0 * t3), dim=-1
        )
        polynomial = (powers * numerator).sum(dim=-1)
        polynomial_derivative = (derivative_powers * numerator).sum(dim=-1)
        divisor = 1.0 + (denominator_powers * denominator).sum(dim=-1)
        divisor_derivative = (
            denominator_derivative_powers * denominator
        ).sum(dim=-1)
        function = polynomial / divisor
        derivative = (
            polynomial_derivative * divisor - polynomial * divisor_derivative
        ) / divisor.square()
        radial = function - t * derivative

        outgoing = (
            pair["out_weight"]
            .detach()
            .float()
            .view(self.external_width, self.groups, self.width)
            .permute(1, 0, 2)
        )
        outgoing_gram = torch.einsum("gdw,gdv->gwv", outgoing, outgoing)
        sample_count = float(z.shape[0])
        derivative_gram = torch.einsum(
            "ngw,ngv->gwv", derivative, derivative
        ) / sample_count
        metric = outgoing_gram * derivative_gram
        outgoing_radial = torch.einsum("gwv,ngv->ngw", outgoing_gram, radial)
        cross = torch.einsum("ngw,ngv->gwv", derivative * outgoing_radial, t)
        cross.div_(sample_count * float(self.width))
        metric.add_(cross).add_(cross.transpose(-2, -1))
        radial_energy = (radial * outgoing_radial).sum(dim=-1)
        radial_metric = torch.einsum(
            "ng,ngw,ngv->gwv", radial_energy, t, t
        )
        radial_metric.div_(sample_count * float(self.width * self.width))
        metric.add_(radial_metric)
        return 0.5 * (metric + metric.transpose(-2, -1))

    @staticmethod
    def _unit_volume_inverse_sqrt(metric):
        """Reference symmetric realization used only by equivalence tests."""
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        mean_diagonal = metric.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        if torch.any(mean_diagonal <= 0.0) or torch.any(~torch.isfinite(metric)):
            raise RuntimeError("R08 structural metric is not finite positive scale")
        relative_shift = torch.finfo(metric.dtype).eps * metric.shape[-1]
        shift = relative_shift * mean_diagonal
        stabilized = metric + shift.unsqueeze(-1).unsqueeze(-1) * torch.eye(
            metric.shape[-1], device=metric.device, dtype=metric.dtype
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(stabilized)
        if torch.any(eigenvalues <= 0.0) or torch.any(~torch.isfinite(eigenvalues)):
            raise RuntimeError("R08 precision-shifted structural metric is not SPD")
        normalized = eigenvalues * torch.exp(
            -eigenvalues.log().mean(dim=-1, keepdim=True)
        )
        inverse_sqrt = (
            eigenvectors * torch.rsqrt(normalized).unsqueeze(-2)
        ) @ eigenvectors.transpose(-2, -1)
        volume_residual = normalized.log().mean(dim=-1).abs()
        return inverse_sqrt, relative_shift, eigenvalues, volume_residual

    @staticmethod
    def _unit_volume_cholesky(metric, *, capture_spectrum=False):
        """Return the coordinate factor for the exact unit-volume metric LMO.

        ``lower`` satisfies ``metric + shift I = lower @ lower.T`` and
        ``volume_scale`` is the square root of its geometric-mean eigenvalue.
        A spectrum is formed only on sparse telemetry steps; it is never part
        of the update path.
        """
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        mean_diagonal = metric.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        if torch.any(mean_diagonal <= 0.0) or torch.any(~torch.isfinite(metric)):
            raise RuntimeError("R08 structural metric is not finite positive scale")
        relative_shift = torch.finfo(metric.dtype).eps * metric.shape[-1]
        shift = relative_shift * mean_diagonal
        stabilized = metric + shift.unsqueeze(-1).unsqueeze(-1) * torch.eye(
            metric.shape[-1], device=metric.device, dtype=metric.dtype
        )
        lower, info = torch.linalg.cholesky_ex(stabilized, check_errors=False)
        if torch.any(info != 0) or torch.any(~torch.isfinite(lower)):
            raise RuntimeError("R08 precision-shifted structural metric is not SPD")
        log_volume_scale = lower.diagonal(dim1=-2, dim2=-1).log().mean(dim=-1)
        volume_scale = log_volume_scale.exp()
        normalized_log_determinant = 2.0 * (
            lower.diagonal(dim1=-2, dim2=-1).log()
            - log_volume_scale.unsqueeze(-1)
        ).mean(dim=-1)
        if capture_spectrum:
            eigenvalues = torch.linalg.eigvalsh(stabilized)
            if torch.any(eigenvalues <= 0.0) or torch.any(~torch.isfinite(eigenvalues)):
                raise RuntimeError("R08 telemetry spectrum is not finite positive")
        else:
            eigenvalues = None
        return (
            lower,
            volume_scale,
            relative_shift,
            eigenvalues,
            normalized_log_determinant.abs(),
        )

    def step(self, closure=None):
        raise NotImplementedError("response capture infrastructure has no update rule")
