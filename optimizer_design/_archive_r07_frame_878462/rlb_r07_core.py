"""Opaque R07 optimizer driven by intrinsic Global-RLB Jacobian shape.

R07 retains the positive R05 generation-one optimizer on the two matrices
surrounding each rational activation.  For attention, it replaces R04's
current-versus-initializer selector by the current P5/Q4 Jacobian's spectral
participation ratio.  The selector is active for an intrinsically anisotropic
rational response, invariant to a conformal response rescaling, and equals
one exactly for an isotropic response.  It changes optimizer geometry, never
LR, WD, schedule, clipping, or update budget.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_r04_core import R04AttentionCore, R04RLBRouterCore


class R07RLBRouterCore(R04RLBRouterCore):
    """R05 pair parent plus an intrinsic current-response attention sensor."""

    def __init__(
        self,
        pairs,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta2: float,
        eps: float,
    ):
        pairs = list(pairs)
        self._r07_local_participation = [None for _ in pairs]
        self._r07_attention_participation = None
        self._r07_last_participation = None
        super().__init__(
            pairs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            eps=eps,
        )

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        for key in tuple(report):
            if key.startswith("block_response_sensor"):
                report.pop(key)
        report.update({
            "intrinsic_jacobian_participation_lr_scale": 1.0,
            "intrinsic_attention_router_lr_scale": 1.0,
        })
        return report

    @classmethod
    def _jacobian_participation(cls, u, function, derivative):
        """Return the normalized stable rank of each exact RLB Jacobian.

        For singular values ``s_i`` of one width-m Jacobian,

            c = (sum_i s_i^2)^2 / (m sum_i s_i^4).

        Hence ``1/m <= c <= 1`` and ``sqrt(1-c)`` is a canonical bounded
        anisotropy amplitude.  Both numerator and denominator scale by the
        fourth power under a conformal response scaling, so the statistic is
        exactly scale invariant in real arithmetic.
        """
        width = float(u.shape[-1])
        radial = (function - u * derivative) / width
        trace = derivative.square().sum(dim=-1)
        trace = trace + 2.0 * (derivative * u * radial).sum(dim=-1)
        trace = trace + u.square().sum(dim=-1) * radial.square().sum(dim=-1)
        trace_square = cls._jacobian_kernel_inner(
            u, function, derivative, function, derivative
        )
        tiny = torch.finfo(trace.dtype).tiny
        participation = trace.square() / (width * trace_square.clamp_min(tiny))
        finite = torch.isfinite(participation) & (trace > 0.0) & (trace_square > 0.0)
        torch._assert_async(finite.all())
        return participation.clamp_(0.0, 1.0)

    def _consume_probe(self, layer_index):
        probe = super()._consume_probe(layer_index)
        z = probe.float().view(self.probe_count, self.groups, self.width)
        rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + self.rlb_eps)
        u = z / rms
        pair = self.pairs[layer_index]
        function, derivative = self._evaluate_response(
            u, pair["numerator"], pair["denominator"]
        )
        participation = self._jacobian_participation(u, function, derivative)
        self._r07_local_participation[layer_index] = torch.stack((
            participation.sum(),
            torch.tensor(
                float(participation.numel()),
                device=participation.device,
                dtype=participation.dtype,
            ),
        ))
        return probe

    def _consume_router_alignments(self):
        # The RLB pair keeps the literal relative-morphology R05 decision.
        relative_alignments = super()._consume_router_alignments()
        if any(value is None for value in self._r07_local_participation):
            raise RuntimeError("R07 did not form every intrinsic response statistic")
        statistics = torch.stack(self._r07_local_participation)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        participation = statistics[:, 0] / statistics[:, 1].clamp_min(1.0)
        valid = torch.isfinite(participation) & (participation >= 0.0) & (
            participation <= 1.0
        )
        torch._assert_async(valid.all())
        self._r07_attention_participation = participation[:, None].expand(-1, 2).clone()
        self._r07_last_participation = participation
        self._r07_local_participation = [None for _ in self.pairs]
        return relative_alignments

    def current_attention_alignments(self):
        if self._r07_attention_participation is None:
            raise RuntimeError("R07 attention requested a stale intrinsic sensor")
        expected = (len(self.pairs), 2)
        if self._r07_attention_participation.shape != expected:
            raise RuntimeError("R07 intrinsic attention inventory changed")
        return self._r07_attention_participation

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r07_attention_participation = None
        self._r07_last_participation = None
        loss = super().step(closure)
        if self._r07_attention_participation is None:
            raise RuntimeError("R07 did not publish its current intrinsic sensor")
        if publish:
            renamed = {
                key.replace("rlb_r04_", "rlb_r07_", 1): value
                for key, value in self._last_telemetry.items()
                if key.startswith("rlb_r04_")
            }
            participation = self._r07_last_participation
            anisotropy = torch.sqrt((1.0 - participation).clamp_min(0.0))
            renamed.update({
                "rlb_r07_attention_participation_min": float(participation.amin().item()),
                "rlb_r07_attention_participation_median": float(participation.median().item()),
                "rlb_r07_attention_participation_max": float(participation.amax().item()),
                "rlb_r07_attention_anisotropy_max": float(anisotropy.amax().item()),
                "rlb_r07_structural_matrix_elements": 245_366_784,
            })
            self._last_telemetry = renamed
        return loss


class R07AttentionCore(R04AttentionCore):
    """Factorized adaptive attention routed by intrinsic RLB anisotropy."""

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.pop("rlb_morphology_router_lr_scale", None)
        report["intrinsic_rlb_morphology_router_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        loss = super().step(closure)
        if publish:
            self._last_telemetry = {
                key.replace("rlb_r04_", "rlb_r07_", 1): value
                for key, value in self._last_telemetry.items()
                if key.startswith("rlb_r04_")
            }
        return loss
