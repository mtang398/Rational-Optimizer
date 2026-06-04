"""Generic optimizer baselines for RationalOPT language-model experiments.

These implementations are intentionally self-contained so manifest-run experiments can
run without relying on external optimizer packages. SOAP/Shampoo and CAME are
labeled as style baselines unless they are later matched line-by-line to a
specific reference implementation.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


def _to_group_list(params):
    return list(params) if isinstance(params, Iterable) else params


def _decoupled_weight_decay(param: torch.Tensor, lr: float, weight_decay: float):
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)


def _factored_second_moment(row: torch.Tensor, col: torch.Tensor, eps: float) -> torch.Tensor:
    row_mean = row.mean(dim=-1, keepdim=True).clamp_min(eps)
    col_mean = col.mean(dim=-2, keepdim=True).clamp_min(eps)
    global_mean = row_mean.mean().clamp_min(eps)
    return row_mean * col_mean / global_mean


def _should_factor(param: torch.Tensor, factored_min_dim: int) -> bool:
    return param.dim() == 2 and min(param.shape) >= int(factored_min_dim)


def _ademamix_linear_warmup(step: int, value_end: float, value_start: float, warmup_steps: int | None) -> float:
    if warmup_steps is None or int(warmup_steps) <= 0 or step >= int(warmup_steps):
        return float(value_end)
    mix = float(step) / float(warmup_steps)
    return (1.0 - mix) * float(value_start) + mix * float(value_end)


def _ademamix_half_life_warmup(step: int, beta_end: float, beta_start: float, warmup_steps: int | None) -> float:
    if warmup_steps is None or int(warmup_steps) <= 0 or step >= int(warmup_steps):
        return float(beta_end)

    def half_life(beta: float) -> float:
        beta = min(max(float(beta), 1e-8), 1.0 - 1e-12)
        return math.log(0.5) / math.log(beta) - 1.0

    def inverse_half_life(value: float) -> float:
        return math.pow(0.5, 1.0 / (float(value) + 1.0))

    mix = float(step) / float(warmup_steps)
    start = half_life(beta_start)
    end = half_life(beta_end)
    return inverse_half_life((1.0 - mix) * start + mix * end)


class Lion(Optimizer):
    """Lion optimizer with decoupled weight decay."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(_to_group_list(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                _decoupled_weight_decay(param, lr, weight_decay)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                param.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class AdEMAMix(Optimizer):
    """AdEMAMix with paper-style slow-EMA treatment and warmups.

    The slow EMA is intentionally not bias-corrected. This matches the
    reference implementation and keeps the old-gradient term from dominating
    early training. ``alpha_warmup`` is linear; ``beta3_warmup`` interpolates
    the EMA half-life from beta1 to beta3.
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.999, 0.9999),
        eps=1e-8,
        weight_decay=0.0,
        alpha=5.0,
        beta3_warmup=None,
        alpha_warmup=None,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
        )
        super().__init__(_to_group_list(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2, beta3_final = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            alpha_final = float(group["alpha"])
            beta3_warmup = group.get("beta3_warmup")
            alpha_warmup = group.get("alpha_warmup")
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_fast"] = torch.zeros_like(param) if beta1 != 0.0 else None
                    state["exp_avg_slow"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                step = int(state["step"])
                exp_avg_fast = state["exp_avg_fast"]
                exp_avg_slow = state["exp_avg_slow"]
                exp_avg_sq = state["exp_avg_sq"]

                alpha = _ademamix_linear_warmup(step, alpha_final, 0.0, alpha_warmup)
                beta3 = _ademamix_half_life_warmup(step, beta3_final, beta1, beta3_warmup)

                _decoupled_weight_decay(param, lr, weight_decay)
                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    bias1 = 1.0 - beta1 ** step
                    fast_update = exp_avg_fast / bias1
                else:
                    fast_update = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1.0 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(eps)
                update = fast_update + alpha * exp_avg_slow
                param.addcdiv_(update, denom, value=-lr)
        return loss


class ScheduleFreeAdamW(Optimizer):
    """Schedule-free AdamW-style baseline with Polyak interpolation.

    This keeps a trainable sequence ``z`` and a running average. Gradients are
    evaluated at the interpolation stored in ``param``; after every AdamW update
    to ``z``, ``param`` is set to ``(1-beta1) * z + beta1 * average``. This is a
    self-contained schedule-free style baseline for horizon-sensitivity tests.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, warmup_steps=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, warmup_steps=warmup_steps)
        super().__init__(_to_group_list(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            warmup_steps = int(group["warmup_steps"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("ScheduleFreeAdamW does not support sparse gradients")
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = param.detach().clone()
                    state["avg"] = param.detach().clone()
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                step = int(state["step"])
                z = state["z"]
                avg = state["avg"]
                exp_avg_sq = state["exp_avg_sq"]

                _decoupled_weight_decay(z, lr, weight_decay)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(eps)
                z.addcdiv_(grad, denom, value=-lr)

                avg_step = max(1, step - warmup_steps + 1)
                if step <= warmup_steps and warmup_steps > 0:
                    weight = 1.0 / float(step)
                else:
                    weight = 1.0 / float(avg_step)
                avg.mul_(1.0 - weight).add_(z, alpha=weight)
                param.copy_(z.lerp(avg, beta1))
        return loss


class CAMEStyleAdamW(Optimizer):
    """CAME/Adafactor-style AdamW with factored second moments and confidence."""

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.999, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        factored_min_dim=128,
        clip_threshold=1.0,
        confidence_scale=1.0,
        confidence_min=0.25,
        confidence_max=4.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            factored_min_dim=factored_min_dim,
            clip_threshold=clip_threshold,
            confidence_scale=confidence_scale,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
        )
        super().__init__(_to_group_list(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2, beta3 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            factored_min_dim = int(group["factored_min_dim"])
            clip_threshold = float(group["clip_threshold"])
            confidence_scale = float(group["confidence_scale"])
            confidence_min = float(group["confidence_min"])
            confidence_max = float(group["confidence_max"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("CAMEStyleAdamW does not support sparse gradients")
                state = self.state[param]
                factored = _should_factor(param, factored_min_dim)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(param.shape[0], 1, device=param.device, dtype=torch.float32)
                        state["exp_avg_sq_col"] = torch.zeros(1, param.shape[1], device=param.device, dtype=torch.float32)
                        state["instability_row"] = torch.zeros(param.shape[0], 1, device=param.device, dtype=torch.float32)
                        state["instability_col"] = torch.zeros(1, param.shape[1], device=param.device, dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
                        state["instability"] = torch.zeros_like(param, dtype=torch.float32)
                state["step"] += 1
                step = int(state["step"])
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                bias3 = 1.0 - beta3 ** step
                grad_f = grad.detach().float()
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad_f, alpha=1.0 - beta1)

                grad_sq = grad_f.square().add_(eps)
                if factored:
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(grad_sq.mean(dim=1, keepdim=True), alpha=1.0 - beta2)
                    col.mul_(beta2).add_(grad_sq.mean(dim=0, keepdim=True), alpha=1.0 - beta2)
                    v = _factored_second_moment(row / bias2, col / bias2, eps)
                    residual = (grad_sq - v).abs() / v.clamp_min(eps)
                    inst_row = state["instability_row"]
                    inst_col = state["instability_col"]
                    inst_row.mul_(beta3).add_(residual.mean(dim=1, keepdim=True), alpha=1.0 - beta3)
                    inst_col.mul_(beta3).add_(residual.mean(dim=0, keepdim=True), alpha=1.0 - beta3)
                    instability = _factored_second_moment(inst_row / bias3, inst_col / bias3, eps)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1.0 - beta2)
                    v = exp_avg_sq / bias2
                    residual = (grad_sq - v).abs() / v.clamp_min(eps)
                    instability_state = state["instability"]
                    instability_state.mul_(beta3).add_(residual, alpha=1.0 - beta3)
                    instability = instability_state / bias3

                update = (exp_avg / bias1) / v.sqrt().add(eps)
                confidence = (1.0 + confidence_scale * instability).rsqrt().clamp(confidence_min, confidence_max)
                update = update * confidence
                if clip_threshold > 0:
                    rms = torch.sqrt(update.square().mean()).clamp_min(eps)
                    update = update / torch.clamp(rms / clip_threshold, min=1.0)
                _decoupled_weight_decay(param, lr, weight_decay)
                param.add_(update.to(dtype=param.dtype), alpha=-lr)
        return loss


class SOAPStyleAdamW(Optimizer):
    """SOAP/Shampoo-style eigenbasis AdamW baseline for 2D tensors.

    2D parameters below the configured side threshold use factored Shampoo
    preconditioners to define an eigenbasis for AdamW moments. Larger matrices
    and non-matrix tensors fall back to ordinary AdamW updates.
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=50,
        large_side_identity_threshold=2048,
        one_sided=False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            large_side_identity_threshold=large_side_identity_threshold,
            one_sided=one_sided,
        )
        super().__init__(_to_group_list(params), defaults)

    @staticmethod
    def _eigh(matrix: torch.Tensor, eps: float) -> torch.Tensor:
        matrix_f = matrix.float()
        matrix_f = 0.5 * (matrix_f + matrix_f.transpose(-1, -2))
        eye = torch.eye(matrix_f.shape[0], device=matrix_f.device, dtype=matrix_f.dtype)
        try:
            _, eigvec = torch.linalg.eigh(matrix_f + eps * eye)
        except RuntimeError:
            eigvec = eye
        return eigvec

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
            freq = max(1, int(group["precondition_frequency"]))
            threshold = int(group["large_side_identity_threshold"])
            one_sided = bool(group["one_sided"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAPStyleAdamW does not support sparse gradients")
                use_soap = param.dim() == 2 and max(param.shape) <= threshold and min(param.shape) > 1
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
                    if use_soap:
                        rows, cols = param.shape
                        state["left_precond"] = torch.zeros(rows, rows, device=param.device, dtype=torch.float32)
                        state["right_precond"] = torch.zeros(cols, cols, device=param.device, dtype=torch.float32)
                        state["left_basis"] = torch.eye(rows, device=param.device, dtype=torch.float32)
                        state["right_basis"] = torch.eye(cols, device=param.device, dtype=torch.float32)
                state["step"] += 1
                step = int(state["step"])
                _decoupled_weight_decay(param, lr, weight_decay)

                if not use_soap:
                    grad_f = grad.detach().float()
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad_f, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1.0 - beta2)
                    bias1 = 1.0 - beta1 ** step
                    bias2 = 1.0 - beta2 ** step
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(eps)
                    update = (exp_avg / bias1) / denom
                    param.add_(update.to(dtype=param.dtype), alpha=-lr)
                    continue

                grad_f = grad.float()
                left = state["left_precond"]
                right = state["right_precond"]
                left.mul_(beta2).add_(grad_f @ grad_f.t(), alpha=1.0 - beta2)
                if not one_sided:
                    right.mul_(beta2).add_(grad_f.t() @ grad_f, alpha=1.0 - beta2)

                if step == 1 or step % freq == 0:
                    old_left = state["left_basis"]
                    old_right = state["right_basis"]
                    new_left = self._eigh(left, eps).to(device=param.device)
                    new_right = old_right if one_sided else self._eigh(right, eps).to(device=param.device)
                    exp_avg = state["exp_avg"]
                    raw_avg = old_left @ exp_avg @ old_right.t()
                    state["exp_avg"].copy_(new_left.t() @ raw_avg @ new_right)
                    state["exp_avg_sq"].zero_()
                    state["left_basis"] = new_left
                    state["right_basis"] = new_right

                left_basis = state["left_basis"].to(device=param.device, dtype=torch.float32)
                right_basis = state["right_basis"].to(device=param.device, dtype=torch.float32)
                grad_rot = left_basis.t() @ grad_f @ right_basis
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad_rot, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_rot, grad_rot, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                update_rot = exp_avg.div(bias1) / exp_avg_sq.sqrt().div(math.sqrt(bias2)).add(eps)
                update = left_basis @ update_rot.float() @ right_basis.t()
                param.add_(update.to(dtype=param.dtype), alpha=-lr)
        return loss
