"""On-policy functional-trust optimizer for RLB rational coefficients."""

from __future__ import annotations

import torch

from .adaptive_metric_onpolicy_optimizer import RationalAdaptiveMetricOnPolicyOptimizer


class RationalFunctionalTrustOnPolicyOptimizer(RationalAdaptiveMetricOnPolicyOptimizer):
    """RLB optimizer that gates coefficient motion by live function-space risk.

    The matrix path stays close to the best measured Jacobian/on-policy gauge
    optimizer. Rational coefficients are treated more strictly: before the
    child coefficient optimizer steps, this wrapper measures the functional
    size of each coefficient gradient on the actual RLB input distribution and
    attenuates groups whose coefficient activity, functional risk, denominator
    pressure, or gradient-direction instability is high.
    """

    def __init__(
        self,
        *args,
        trust_coeff_strength: float = 1.0,
        trust_radius: float = 0.018,
        trust_min_scale: float = 0.05,
        trust_max_scale: float = 1.15,
        trust_activity_target: float = 0.85,
        trust_activity_width: float = 0.55,
        trust_pressure_weight: float = 0.25,
        trust_agreement_decay: float = 0.90,
        trust_agreement_floor: float = 0.15,
        trust_metric_blend: float = 0.45,
        trust_denominator_risk: float = 1.75,
        trust_atom_risk: float = 1.00,
        trust_numerator_risk: float = 1.00,
        trust_depth_gain: float = 0.10,
        **kwargs,
    ):
        super().__init__(*args, coeff_strength=0.0, **kwargs)
        self.trust_coeff_strength = float(trust_coeff_strength)
        self.trust_radius = float(trust_radius)
        self.trust_min_scale = float(trust_min_scale)
        self.trust_max_scale = float(trust_max_scale)
        self.trust_activity_target = float(trust_activity_target)
        self.trust_activity_width = float(trust_activity_width)
        self.trust_pressure_weight = float(trust_pressure_weight)
        self.trust_agreement_decay = float(trust_agreement_decay)
        self.trust_agreement_floor = float(trust_agreement_floor)
        self.trust_metric_blend = float(trust_metric_blend)
        self.trust_denominator_risk = float(trust_denominator_risk)
        self.trust_atom_risk = float(trust_atom_risk)
        self.trust_numerator_risk = float(trust_numerator_risk)
        self.trust_depth_gain = float(trust_depth_gain)
        if self.trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive")
        if not 0.0 <= self.trust_agreement_decay < 1.0:
            raise ValueError("trust_agreement_decay must be in [0, 1)")
        if self.trust_max_scale < self.trust_min_scale or self.trust_min_scale < 0.0:
            raise ValueError("trust scales must satisfy 0 <= min <= max")

    def _role_risk_weight(self, role: str) -> float:
        if role == "denominator":
            return self.trust_denominator_risk
        if role == "atom":
            return self.trust_atom_risk
        return self.trust_numerator_risk

    def _role_gram(self, stats: dict, role: str):
        if role == "numerator":
            return stats.get("num_gram")
        if role == "denominator":
            return stats.get("den_gram")
        if role == "atom":
            return stats.get("atom_gram")
        return None

    def _gradient_agreement(self, group: dict, role: str, flat_grad: torch.Tensor) -> torch.Tensor:
        state = group.setdefault("_functional_trust", {})
        key = f"{role}_grad_ema"
        grad = flat_grad.detach().float()
        previous = state.get(key)
        if previous is None or previous.shape != grad.shape or previous.device != grad.device:
            state[key] = grad.clone()
            return torch.ones(grad.shape[0], device=grad.device, dtype=grad.dtype)
        prev = previous.to(device=grad.device, dtype=grad.dtype)
        numerator = (grad * prev).sum(dim=1)
        denominator = torch.sqrt(grad.square().sum(dim=1) * prev.square().sum(dim=1) + self.eps)
        cosine = (numerator / denominator.clamp_min(self.eps)).clamp(-1.0, 1.0)
        previous.mul_(self.trust_agreement_decay).add_(grad, alpha=1.0 - self.trust_agreement_decay)
        return ((cosine + 1.0) * 0.5).clamp(self.trust_agreement_floor, 1.0)

    def _activity_gate(self, group: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        groups = int(group["groups"])
        state = group.get("_onpolicy")
        if state is None:
            return torch.ones(groups, device=device, dtype=dtype)
        in_rel = state["in_rel_ema"].to(device=device, dtype=dtype).clamp_min(self.eps)
        out_rel = state["out_rel_ema"].to(device=device, dtype=dtype).clamp_min(self.eps)
        rat_rel = state["rat_rel_ema"].to(device=device, dtype=dtype).clamp_min(self.eps)
        matrix_rel = torch.sqrt(in_rel * out_rel).clamp_min(self.eps)
        log_ratio = torch.log(rat_rel / matrix_rel)
        target = torch.log(torch.full_like(log_ratio, max(self.trust_activity_target, self.eps)))
        width = max(self.trust_activity_width, self.eps)
        over_target = ((log_ratio - target) / width).clamp_min(0.0)
        activity = torch.exp(-over_target)

        pressure = (torch.log(in_rel) - torch.log(out_rel)).abs()
        pressure_gate = torch.exp(-self.trust_pressure_weight * pressure)
        return (activity * pressure_gate).clamp(0.0, 1.0)

    def _depth_gate(self, group: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        depth = self._depth(group)
        factor = min(1.25, max(0.75, 1.0 + self.trust_depth_gain * (0.5 - depth)))
        return torch.full((int(group["groups"]),), factor, device=device, dtype=dtype)

    def _functional_risk_gate(
        self,
        group: dict,
        role: str,
        flat_grad: torch.Tensor,
        gram: torch.Tensor,
        output_rms: torch.Tensor,
    ) -> torch.Tensor:
        grad_f = flat_grad.detach().float()
        gram_f = gram.detach().float()
        if gram_f.dim() != 3 or gram_f.size(0) != grad_f.size(0):
            return torch.ones(grad_f.shape[0], device=grad_f.device, dtype=grad_f.dtype)
        if gram_f.size(1) != grad_f.size(1) or gram_f.size(2) != grad_f.size(1):
            return torch.ones(grad_f.shape[0], device=grad_f.device, dtype=grad_f.dtype)
        quad = torch.einsum("gi,gij,gj->g", grad_f, gram_f, grad_f).clamp_min(0.0)
        output = output_rms.detach().float().to(device=grad_f.device).clamp_min(self.eps)
        risk = torch.sqrt(quad + self.eps) / output
        risk = risk * self._role_risk_weight(role)
        radius = torch.full_like(risk, self.trust_radius)
        return (radius / (radius + risk)).clamp(0.0, 1.0)

    def _headroom_gate(self, group: dict, role: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        groups = int(group["groups"])
        if role != "atom":
            return torch.ones(groups, device=device, dtype=dtype)
        logits = group["coeff_logits"].detach().float().reshape(groups, -1)
        headroom = (1.0 - torch.tanh(logits).square()).mean(dim=1)
        return headroom.to(device=device, dtype=dtype).clamp(0.10, 1.0)

    def _trust_scale(
        self,
        group: dict,
        role: str,
        flat_grad: torch.Tensor,
        gram: torch.Tensor,
        output_rms: torch.Tensor,
    ) -> torch.Tensor:
        device = flat_grad.device
        dtype = flat_grad.dtype
        risk_gate = self._functional_risk_gate(group, role, flat_grad, gram, output_rms).to(device=device, dtype=dtype)
        activity_gate = self._activity_gate(group, device, dtype)
        agreement_gate = self._gradient_agreement(group, role, flat_grad).to(device=device, dtype=dtype)
        depth_gate = self._depth_gate(group, device, dtype)
        headroom_gate = self._headroom_gate(group, role, device, dtype)
        gate = (risk_gate * activity_gate * agreement_gate * depth_gate * headroom_gate).clamp(0.0, 1.0)
        scale = self.trust_min_scale + gate * (self.trust_max_scale - self.trust_min_scale)
        strength = min(1.0, max(0.0, self.trust_coeff_strength))
        return (1.0 - strength) + strength * scale

    def _precondition_trusted_param(self, group: dict, role: str, param: torch.Tensor, stats: dict):
        if param.grad is None:
            return
        gram = self._role_gram(stats, role)
        output_rms = stats.get("output_rms")
        if gram is None or output_rms is None:
            return
        grad = param.grad
        original_shape = grad.shape
        flat_grad = grad.reshape(grad.shape[0], -1)
        if flat_grad.shape[0] != int(group["groups"]):
            return

        gram = gram.to(device=grad.device)
        trust_scale = self._trust_scale(group, role, flat_grad, gram, output_rms).to(device=grad.device, dtype=grad.dtype)
        metric = self._metric_solve(gram, flat_grad, self.coeff_metric_damping).reshape(original_shape)
        blend = (self.trust_metric_blend * trust_scale).view(-1, *([1] * (grad.dim() - 1)))
        scale = trust_scale.view(-1, *([1] * (grad.dim() - 1)))
        grad.mul_(1.0 - blend).add_(metric * blend).mul_(scale)

    @torch.no_grad()
    def _precondition_coefficient_gradients(self):
        if self.trust_coeff_strength <= 0.0:
            return
        for group in self.balance_groups:
            module = group.get("module")
            stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
            if not stats:
                continue
            self._precondition_trusted_param(group, "numerator", group["numerator"], stats)
            self._precondition_trusted_param(group, "denominator", group["denominator"], stats)
            self._precondition_trusted_param(group, "atom", group["coeff_logits"], stats)
