"""Opaque R06 optimizer with role-specific intrinsic RLB morphology.

R06 retains the complete positive R05 generation-one geometry.  It augments
the learned-current-versus-initial response alignment with two scale-free
properties of the current rational function: normalized Jacobian stable rank
for the incoming role and normalized response participation for the outgoing
role.  Their products route R05's already equal-budget adaptive coordinate.
Attention uses the geometric mean of the two intrinsic participation ratios.
No statistic changes LR, WD, momentum, NS count, schedule, or update budget.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_r07_core import R07AttentionCore, R07RLBRouterCore


class R06RLBRouterCore(R07RLBRouterCore):
    """R05 parent with intrinsic current-function routing on all four roles."""

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
        self._r06_local_output_participation = [None for _ in pairs]
        self._r06_role_participation = None
        self._r06_pair_alignments = None
        self._r06_attention_alignments = None
        self._r06_last_output_participation = None
        self._r06_last_pair_alignments = None
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
        report.update({
            "intrinsic_role_participation_lr_scale": 1.0,
            "relative_intrinsic_product_router_lr_scale": 1.0,
            "intrinsic_attention_geometric_mean_lr_scale": 1.0,
        })
        return report

    @staticmethod
    def _response_participation(function):
        """Normalized participation of the current response across a group."""
        width = float(function.shape[-1])
        energy = function.square().sum(dim=-1)
        fourth = function.pow(4).sum(dim=-1)
        tiny = torch.finfo(function.dtype).tiny
        participation = energy.square() / (width * fourth.clamp_min(tiny))
        finite = torch.isfinite(participation) & (energy > 0.0) & (fourth > 0.0)
        torch._assert_async(finite.all())
        return participation.clamp_(0.0, 1.0)

    @staticmethod
    def _product_router(relative, intrinsic):
        if relative.shape != intrinsic.shape:
            raise RuntimeError("R06 role sensor inventories differ")
        valid = (
            torch.isfinite(relative)
            & torch.isfinite(intrinsic)
            & (relative >= 0.0)
            & (relative <= 1.0)
            & (intrinsic >= 0.0)
            & (intrinsic <= 1.0)
        )
        torch._assert_async(valid.all())
        product = relative * intrinsic
        return torch.where((relative == 1.0) & (intrinsic == 1.0), relative, product)

    def _consume_probe(self, layer_index):
        probe = super()._consume_probe(layer_index)
        z = probe.float().view(self.probe_count, self.groups, self.width)
        rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + self.rlb_eps)
        u = z / rms
        pair = self.pairs[layer_index]
        function, _ = self._evaluate_response(
            u, pair["numerator"], pair["denominator"]
        )
        participation = self._response_participation(function)
        self._r06_local_output_participation[layer_index] = torch.stack((
            participation.sum(),
            torch.tensor(
                float(participation.numel()),
                device=participation.device,
                dtype=participation.dtype,
            ),
        ))
        return probe

    def _consume_router_alignments(self):
        relative = super()._consume_router_alignments()
        if any(value is None for value in self._r06_local_output_participation):
            raise RuntimeError("R06 did not form every output participation statistic")
        statistics = torch.stack(self._r06_local_output_participation)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        output = statistics[:, 0] / statistics[:, 1].clamp_min(1.0)
        valid = torch.isfinite(output) & (output >= 0.0) & (output <= 1.0)
        torch._assert_async(valid.all())

        incoming = self._r07_attention_participation[:, 0]
        role_participation = torch.stack((incoming, output), dim=1)
        pair_alignments = self._product_router(relative, role_participation)
        attention = torch.sqrt((incoming * output).clamp_min(0.0))

        self._r06_role_participation = role_participation
        self._r06_pair_alignments = pair_alignments
        self._r06_attention_alignments = attention[:, None].expand(-1, 2).clone()
        self._r06_last_output_participation = output
        self._r06_last_pair_alignments = pair_alignments
        self._r06_local_output_participation = [None for _ in self.pairs]
        return pair_alignments

    def current_attention_alignments(self):
        if self._r06_attention_alignments is None:
            raise RuntimeError("R06 attention requested a stale intrinsic sensor")
        expected = (len(self.pairs), 2)
        if self._r06_attention_alignments.shape != expected:
            raise RuntimeError("R06 intrinsic attention inventory changed")
        return self._r06_attention_alignments

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r06_role_participation = None
        self._r06_pair_alignments = None
        self._r06_attention_alignments = None
        self._r06_last_output_participation = None
        self._r06_last_pair_alignments = None
        loss = super().step(closure)
        if self._r06_pair_alignments is None or self._r06_attention_alignments is None:
            raise RuntimeError("R06 did not publish its current intrinsic route")
        if publish:
            renamed = {
                key.replace("rlb_r07_", "rlb_r06_", 1): value
                for key, value in self._last_telemetry.items()
                if key.startswith("rlb_r07_")
            }
            output = self._r06_last_output_participation
            pair = self._r06_last_pair_alignments
            attention = self._r06_attention_alignments[:, 0]
            renamed.update({
                "rlb_r06_output_participation_min": float(output.amin().item()),
                "rlb_r06_output_participation_median": float(output.median().item()),
                "rlb_r06_output_participation_max": float(output.amax().item()),
                "rlb_r06_pair_alignment_min": float(pair.amin().item()),
                "rlb_r06_pair_alignment_median": float(pair.median().item()),
                "rlb_r06_pair_alignment_max": float(pair.amax().item()),
                "rlb_r06_attention_intrinsic_min": float(attention.amin().item()),
                "rlb_r06_attention_intrinsic_median": float(attention.median().item()),
                "rlb_r06_attention_intrinsic_max": float(attention.amax().item()),
                "rlb_r06_structural_matrix_elements": 245_366_784,
            })
            self._last_telemetry = renamed
        return loss


class R06AttentionCore(R07AttentionCore):
    """Factorized attention routed by the two-role intrinsic RLB sensor."""

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.pop("intrinsic_rlb_morphology_router_lr_scale", None)
        report["two_role_intrinsic_rlb_router_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        loss = super().step(closure)
        if publish:
            self._last_telemetry = {
                key.replace("rlb_r07_", "rlb_r06_", 1): value
                for key, value in self._last_telemetry.items()
                if key.startswith("rlb_r07_")
            }
        return loss
