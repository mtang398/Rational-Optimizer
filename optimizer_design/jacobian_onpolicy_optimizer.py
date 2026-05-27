"""Rational-Jacobian on-policy optimizer wrapper for fused RLB FFNs.

For an RLB group,

    v_g = x W_in,g
    u_g = v_g / rms(v_g)
    h_g = rms(v_g) R_g(u_g)
    y   = h W_out

matrix gradients do not have the same functional scale in every group. The
input-side matrix is filtered through the local rational derivative R'_g, while
the output-side matrix sees the rational feature amplitude R_g. This wrapper
preconditions those two matrix-gradient blocks by live rational curve gains
before the child optimizers step, then applies the usual RLB on-policy gauge
balance.
"""

from __future__ import annotations

import torch

from .onpolicy_balance_optimizer import RationalOnPolicyBalanceOptimizer


class RationalJacobianOnPolicyOptimizer(RationalOnPolicyBalanceOptimizer):
    """RLB optimizer using rational derivative/output gains as matrix preconditioners."""

    def __init__(
        self,
        *args,
        matrix_strength: float = 0.5,
        matrix_min_scale: float = 0.5,
        matrix_max_scale: float = 2.0,
        matrix_every: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.matrix_strength = float(matrix_strength)
        self.matrix_min_scale = float(matrix_min_scale)
        self.matrix_max_scale = float(matrix_max_scale)
        self.matrix_every = max(1, int(matrix_every))
        if self.matrix_min_scale <= 0.0:
            raise ValueError("matrix_min_scale must be positive")
        if self.matrix_max_scale < self.matrix_min_scale:
            raise ValueError("matrix_max_scale must be >= matrix_min_scale")

    @staticmethod
    def _centered_inverse_scale(gain: torch.Tensor, strength: float, eps: float) -> torch.Tensor:
        gain_f = gain.detach().float().clamp_min(eps)
        center = torch.exp(torch.log(gain_f).mean()).clamp_min(eps)
        return (center / gain_f).pow(strength)

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

            groups = int(group["groups"])
            hidden_dim = int(group["hidden_dim"])
            width = hidden_dim // groups
            state = group.setdefault("_jacobian_precond", {})
            refresh = self.step_index % self.matrix_every == 1
            if refresh or "in_scale" not in state or "out_scale" not in state:
                out_gain, deriv_gain = self._curve_gain(group)
                in_scale = self._centered_inverse_scale(
                    deriv_gain, self.matrix_strength, self.eps
                ).clamp(self.matrix_min_scale, self.matrix_max_scale)
                out_scale = self._centered_inverse_scale(
                    out_gain, self.matrix_strength, self.eps
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
        self._precondition_matrix_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        self._balance()
