"""Factored AdamW optimizer for matrix-heavy Transformer experiments."""

from __future__ import annotations

import math

import torch


class FactoredAdamW(torch.optim.Optimizer):
    """AdamW with Adafactor-style factored second moments for 2D tensors.

    This optimizer keeps Adam's first moment, decoupled weight decay, and bias
    correction, but replaces full elementwise second moments on large matrices
    with row/column factored statistics. Non-matrix and small tensors fall back
    to standard AdamW moments. It is intended as a higher-leverage alternative
    to scalar per-parameter AdamW for Transformer matrices.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        factored: bool = True,
        factored_min_dim: int = 128,
        clip_threshold: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("betas must be in [0, 1)")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "factored": bool(factored),
            "factored_min_dim": int(factored_min_dim),
            "clip_threshold": float(clip_threshold),
        }
        super().__init__(params, defaults)

    @staticmethod
    def _should_factor(param: torch.Tensor, group: dict) -> bool:
        if not bool(group.get("factored", True)):
            return False
        if param.dim() != 2:
            return False
        min_dim = int(group.get("factored_min_dim", 128))
        return param.shape[0] >= min_dim and param.shape[1] >= min_dim

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            clip_threshold = float(group.get("clip_threshold", 1.0))
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("FactoredAdamW does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if self._should_factor(param, group):
                        state["exp_avg_sq_row"] = torch.zeros(param.shape[0], device=param.device, dtype=torch.float32)
                        state["exp_avg_sq_col"] = torch.zeros(param.shape[1], device=param.device, dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                state["step"] += 1
                step = int(state["step"])
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                bias_correction1 = 1.0 - beta1 ** step

                if "exp_avg_sq_row" in state:
                    grad_sq = grad.detach().float().square().add_(eps)
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1.0 - beta2)
                    col.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1.0 - beta2)
                    row_bc = row / (1.0 - beta2 ** step)
                    col_bc = col / (1.0 - beta2 ** step)
                    row_mean = row_bc.mean().clamp_min(eps)
                    denom = torch.sqrt((row_bc / row_mean).view(-1, 1) * col_bc.view(1, -1)).to(
                        device=param.device,
                        dtype=param.dtype,
                    ).add_(eps)
                    update = exp_avg / bias_correction1 / denom
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bias_correction2 = 1.0 - beta2 ** step
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                    update = exp_avg / bias_correction1 / denom

                if clip_threshold > 0.0:
                    rms = torch.sqrt(update.float().square().mean()).clamp_min(eps)
                    clip = min(1.0, clip_threshold / float(rms.item()))
                    update = update * clip
                param.add_(update, alpha=-lr)
        return loss
