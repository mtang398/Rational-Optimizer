"""Quotient-gauge optimizer wrapper for fused RLB rational FFNs.

RLB has the exact group gauge

    W_in,g  <- c W_in,g
    W_out,g <- W_out,g / c

for positive c. This wrapper projects matrix gradients away from that
function-preserving gauge direction before the child optimizers step, then uses
the existing on-policy balance update to choose a stable gauge representative.
"""

from __future__ import annotations

import torch

from .onpolicy_balance_optimizer import RationalOnPolicyBalanceOptimizer


class RationalQuotientOnPolicyOptimizer(RationalOnPolicyBalanceOptimizer):
    """RLB optimizer that removes pure gauge motion from matrix gradients."""

    def __init__(self, *args, quotient_strength: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.quotient_strength = float(quotient_strength)

    @torch.no_grad()
    def _project_gauge_gradients(self):
        if self.quotient_strength <= 0.0:
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
            coeff = numerator / denominator.clamp_min(self.eps)
            coeff = coeff * self.quotient_strength
            in_grad.sub_(coeff.view(groups, 1, 1) * in_view)
            out_grad.add_(coeff.view(groups, 1, 1) * out_view)

    def step(self):
        self.step_index += 1
        self._update_onpolicy_stats()
        self._project_gauge_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        self._balance()
