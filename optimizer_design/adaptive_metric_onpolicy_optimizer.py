"""Adaptive on-policy metric optimizer for fused RLB rational FFNs.

This optimizer targets the no-GLU Rational Local Basis FFN. It keeps the
function-preserving group gauge used by the earlier on-policy optimizers, but
uses empirical rational output/derivative gains computed from the actual
normalized RLB inputs seen during training. The matrix metric is layer- and
time-dependent. An empirical coefficient Gram metric is available for ablation,
but is off by default because the child FunctionSpaceRationalOptimizer already
preconditions these tiny rational tensors.
"""

from __future__ import annotations

import torch

from .jacobian_onpolicy_optimizer import RationalJacobianOnPolicyOptimizer
from .onpolicy_balance_optimizer import _smoothstep


class RationalAdaptiveMetricOnPolicyOptimizer(RationalJacobianOnPolicyOptimizer):
    """RLB-only optimizer using live empirical rational-function metrics.

    The wrapper performs, in order:

    1. update live gradient-pressure EMAs;
    2. optionally precondition rational coefficient gradients with per-group
       empirical Gram matrices gathered from the current activation distribution;
    3. precondition matrix gradients with on-policy rational output/derivative
       gains;
    4. step child optimizers;
    5. apply the existing function-preserving RLB gauge balance.

    It is intentionally not defined for SiLU/SwiGLU or GLU rational variants.
    """

    def __init__(
        self,
        *args,
        stat_every: int = 4,
        stat_samples: int = 512,
        coeff_strength: float = 0.0,
        coeff_start: float = 0.02,
        coeff_end: float = 0.65,
        coeff_late_decay: float = 0.35,
        coeff_metric_damping: float = 0.03,
        coeff_norm_clip: float = 3.0,
        coeff_max_blend: float = 0.85,
        coeff_depth_gain: float = 0.20,
        matrix_time_gain: float = 0.15,
        matrix_depth_gain: float = 0.10,
        quotient_strength: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stat_every = max(1, int(stat_every))
        self.stat_samples = max(1, int(stat_samples))
        self.coeff_strength = float(coeff_strength)
        self.coeff_start = float(coeff_start)
        self.coeff_end = float(coeff_end)
        self.coeff_late_decay = float(coeff_late_decay)
        self.coeff_metric_damping = float(coeff_metric_damping)
        self.coeff_norm_clip = max(1.0, float(coeff_norm_clip))
        self.coeff_max_blend = min(1.0, max(0.0, float(coeff_max_blend)))
        self.coeff_depth_gain = float(coeff_depth_gain)
        self.matrix_time_gain = float(matrix_time_gain)
        self.matrix_depth_gain = float(matrix_depth_gain)
        self.quotient_strength = float(quotient_strength)
        if self.coeff_metric_damping < 0.0:
            raise ValueError("coeff_metric_damping must be non-negative")

        for group in self.balance_groups:
            module = group.get("module")
            if module is None:
                continue
            setattr(module, "_rlb_optimizer_track_stats", True)
            setattr(module, "_rlb_optimizer_stat_every", self.stat_every)
            setattr(module, "_rlb_optimizer_stat_samples", self.stat_samples)

    def _coefficient_phase(self) -> float:
        progress = self._progress()
        grow = _smoothstep(self.coeff_start, self.coeff_end, progress)
        late = _smoothstep(0.70, 0.95, progress)
        return grow * (1.0 - self.coeff_late_decay * late)

    def _matrix_phase(self) -> float:
        progress = self._progress()
        late = _smoothstep(0.25, 0.90, progress)
        return 1.0 + self.matrix_time_gain * late

    def _coefficient_blend(self, group: dict, role_scale: float) -> float:
        depth = self._depth(group)
        layer = min(1.35, max(0.65, 1.0 + self.coeff_depth_gain * (0.5 - depth)))
        blend = self.coeff_strength * self._coefficient_phase() * layer * float(role_scale)
        return min(self.coeff_max_blend, max(0.0, blend))

    def _matrix_strength_for_group(self, group: dict) -> float:
        depth = self._depth(group)
        layer = min(1.35, max(0.65, 1.0 + self.matrix_depth_gain * (depth - 0.5)))
        return max(0.0, self.matrix_strength * self._matrix_phase() * layer)

    @torch.no_grad()
    def _project_gauge_gradients(self):
        quotient = self.quotient_strength * _smoothstep(0.02, 0.30, self._progress())
        if quotient <= 0.0:
            return
        for group in self.balance_groups:
            views = self._group_weight_views(group)
            if views is None:
                continue
            in_view, out_view = views
            in_weight = group["in_weight"]
            out_weight = group["out_weight"]
            if in_weight.grad is None or out_weight.grad is None:
                continue
            groups = int(group["groups"])
            hidden_dim = int(group["hidden_dim"])
            width = hidden_dim // groups
            in_grad = in_weight.grad.view(groups, width, -1)
            out_grad = out_weight.grad.view(out_weight.shape[0], groups, width).permute(1, 2, 0)
            numerator = (in_grad * in_view).sum(dim=(1, 2)) - (out_grad * out_view).sum(dim=(1, 2))
            denominator = in_view.square().sum(dim=(1, 2)) + out_view.square().sum(dim=(1, 2))
            coeff = quotient * numerator / denominator.clamp_min(self.eps)
            in_grad.sub_(coeff.view(groups, 1, 1) * in_view)
            out_grad.add_(coeff.view(groups, 1, 1) * out_view)

    def _metric_solve(self, gram: torch.Tensor, grad: torch.Tensor, damping: float) -> torch.Tensor:
        if grad.numel() == 0:
            return grad
        gram_f = gram.detach().float()
        grad_f = grad.detach().float()
        if gram_f.dim() != 3 or grad_f.dim() != 2 or gram_f.size(0) != grad_f.size(0):
            return grad
        width = grad_f.size(-1)
        if gram_f.size(1) != width or gram_f.size(2) != width:
            return grad

        gram_f = 0.5 * (gram_f + gram_f.transpose(-1, -2))
        diag_mean = gram_f.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(self.eps)
        eye = torch.eye(width, device=gram_f.device, dtype=gram_f.dtype).expand_as(gram_f)
        matrix = gram_f + (float(damping) * diag_mean + self.eps).view(-1, 1, 1) * eye
        try:
            solved = torch.linalg.solve(matrix, grad_f.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            return grad

        finite = torch.isfinite(solved).all(dim=-1, keepdim=True)
        solved = torch.where(finite, solved, grad_f)
        grad_norm = torch.sqrt(grad_f.square().mean(dim=-1, keepdim=True) + self.eps)
        solved_norm = torch.sqrt(solved.square().mean(dim=-1, keepdim=True) + self.eps)
        rel = (grad_norm / solved_norm).clamp(1.0 / self.coeff_norm_clip, self.coeff_norm_clip)
        return (solved * rel).to(device=grad.device, dtype=grad.dtype)

    def _apply_metric_gradient(
        self,
        param: torch.Tensor,
        gram: torch.Tensor | None,
        blend: float,
        role_damping: float = 1.0,
    ):
        if blend <= 0.0 or gram is None or param.grad is None:
            return
        grad = param.grad
        original_shape = grad.shape
        if grad.dim() < 2:
            return
        flat_grad = grad.reshape(grad.shape[0], -1)
        preconditioned = self._metric_solve(
            gram.to(device=grad.device),
            flat_grad,
            self.coeff_metric_damping * float(role_damping),
        ).reshape(original_shape)
        grad.mul_(1.0 - blend).add_(preconditioned, alpha=blend)

    @torch.no_grad()
    def _precondition_coefficient_gradients(self):
        if self.coeff_strength <= 0.0:
            return
        for group in self.balance_groups:
            module = group.get("module")
            stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
            if not stats:
                continue
            num_blend = self._coefficient_blend(group, 1.00)
            den_blend = self._coefficient_blend(group, 0.90)
            atom_blend = self._coefficient_blend(group, 1.10)
            self._apply_metric_gradient(group["numerator"], stats.get("num_gram"), num_blend)
            self._apply_metric_gradient(group["denominator"], stats.get("den_gram"), den_blend, role_damping=1.5)
            self._apply_metric_gradient(group["coeff_logits"], stats.get("atom_gram"), atom_blend, role_damping=0.75)

    @torch.no_grad()
    def _precondition_matrix_gradients(self):
        if self.matrix_strength <= 0.0:
            return
        for group in self.balance_groups:
            views = self._group_weight_views(group)
            if views is None:
                continue
            in_weight = group["in_weight"]
            out_weight = group["out_weight"]
            if in_weight.grad is None or out_weight.grad is None:
                continue

            strength = self._matrix_strength_for_group(group)
            if strength <= 0.0:
                continue
            groups = int(group["groups"])
            hidden_dim = int(group["hidden_dim"])
            width = hidden_dim // groups
            state = group.setdefault("_adaptive_metric_precond", {})
            refresh = self.step_index % self.matrix_every == 1
            if refresh or "in_scale" not in state or "out_scale" not in state:
                module = group.get("module")
                stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
                if stats and "output_rms" in stats and "derivative_rms" in stats:
                    out_gain = stats["output_rms"]
                    deriv_gain = stats["derivative_rms"]
                else:
                    out_gain, deriv_gain = self._curve_gain(group)
                in_scale = self._centered_inverse_scale(
                    deriv_gain, strength, self.eps
                ).clamp(self.matrix_min_scale, self.matrix_max_scale)
                out_scale = self._centered_inverse_scale(
                    out_gain, strength, self.eps
                ).clamp(self.matrix_min_scale, self.matrix_max_scale)
                state["in_scale"] = in_scale.to(device=in_weight.device, dtype=in_weight.dtype)
                state["out_scale"] = out_scale.to(device=out_weight.device, dtype=out_weight.dtype)

            in_grad = in_weight.grad.view(groups, width, -1)
            out_grad = out_weight.grad.view(out_weight.shape[0], groups, width).permute(1, 2, 0)
            in_grad.mul_(state["in_scale"].view(groups, 1, 1))
            out_grad.mul_(state["out_scale"].view(groups, 1, 1))

    def step(self):
        self.step_index += 1
        self._update_onpolicy_stats()
        self._project_gauge_gradients()
        self._precondition_coefficient_gradients()
        self._precondition_matrix_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        self._balance()
