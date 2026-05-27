"""Quotient-Jacobian on-policy optimizer wrapper for fused RLB FFNs.

This combines the two RLB-specific pieces that were individually useful:
matrix-gradient projection away from the exact group-gauge direction and
rational output/derivative Jacobian preconditioning. It deliberately avoids
extra coefficient-gradient conditioning; rational coefficients remain handled
by the child FunctionSpaceRationalOptimizer.
"""

from __future__ import annotations

import torch

from .jacobian_onpolicy_optimizer import RationalJacobianOnPolicyOptimizer
from .onpolicy_balance_optimizer import _smoothstep


class RationalQuotientJacobianOnPolicyOptimizer(RationalJacobianOnPolicyOptimizer):
    """RLB optimizer with quotient projection plus rational Jacobian scaling."""

    def __init__(
        self,
        *args,
        quotient_strength: float = 1.0,
        quotient_start: float = 0.02,
        quotient_end: float = 0.30,
        quotient_depth_gain: float = 0.10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.quotient_strength = float(quotient_strength)
        self.quotient_start = float(quotient_start)
        self.quotient_end = float(quotient_end)
        self.quotient_depth_gain = float(quotient_depth_gain)

    def _quotient_strength_for_group(self, group: dict) -> float:
        phase = _smoothstep(self.quotient_start, self.quotient_end, self._progress())
        depth = self._depth(group)
        layer = min(1.25, max(0.75, 1.0 + self.quotient_depth_gain * (0.5 - depth)))
        return max(0.0, self.quotient_strength * phase * layer)

    @torch.no_grad()
    def _project_gauge_gradients(self):
        if self.quotient_strength <= 0.0:
            return
        for group in self.balance_groups:
            quotient = self._quotient_strength_for_group(group)
            if quotient <= 0.0:
                continue
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

    def step(self):
        self.step_index += 1
        self._update_onpolicy_stats()
        self._project_gauge_gradients()
        self._precondition_matrix_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        self._balance()
