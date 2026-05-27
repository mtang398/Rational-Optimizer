"""Function-space optimizer for trainable rational activation parameters.

This optimizer is intentionally not AdamW/Muon with tuned hyperparameters. It
uses a fixed probe grid to scale coefficient gradients by the RMS size of their
induced basis functions. The update is then normalized at the activation-group
level, so each rational curve receives a controlled function-space step rather
than a raw coefficient-gradient step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class RationalOptimizerStats:
    rational_parameter_tensors: int
    rational_parameters: int


class FunctionSpaceRationalOptimizer(torch.optim.Optimizer):
    """Direct function-space scaled gradient descent for rational coefficients.

    The optimizer expects param groups to include a ``role`` field. Supported
    roles are ``numerator``, ``denominator``, ``atom``, ``center``, and
    ``other``. The learning rate is supplied by the training loop's scheduler in
    the usual PyTorch way through a rational-curve metric. Coefficient gradients
    are first mapped through an approximate
    function-space metric, then normalized per rational group by a running curve
    energy scalar.
    """

    def __init__(
        self,
        params: Iterable[dict],
        lr: float,
        numerator_lr_scale: float = 1.0,
        denominator_lr_scale: float = 1.125,
        atom_lr_scale: float = 2.25,
        center_lr_scale: float = 0.10,
        other_lr_scale: float = 0.25,
        trust: float = 0.01,
        probe_range: float = 5.0,
        probe_points: int = 257,
        curve_decay: float = 0.95,
        update_gain: float = 4.5,
        metric: str = "diag",
        metric_damping: float = 1e-3,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if probe_range <= 0.0:
            raise ValueError("probe_range must be positive")
        if probe_points < 3:
            raise ValueError("probe_points must be at least 3")
        if not 0.0 <= curve_decay < 1.0:
            raise ValueError("curve_decay must be in [0, 1)")
        if metric not in {"diag", "gram"}:
            raise ValueError("metric must be 'diag' or 'gram'")
        if metric_damping < 0.0:
            raise ValueError("metric_damping must be non-negative")
        defaults = {
            "lr": lr,
            "numerator_lr_scale": numerator_lr_scale,
            "denominator_lr_scale": denominator_lr_scale,
            "atom_lr_scale": atom_lr_scale,
            "center_lr_scale": center_lr_scale,
            "other_lr_scale": other_lr_scale,
            "trust": trust,
            "probe_range": probe_range,
            "probe_points": probe_points,
            "curve_decay": curve_decay,
            "update_gain": update_gain,
            "metric": metric,
            "metric_damping": metric_damping,
            "eps": eps,
        }
        super().__init__(params, defaults)
        self._scale_cache: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _role_lr_scale(group: dict, role: str) -> float:
        if role == "numerator":
            return float(group["numerator_lr_scale"])
        if role == "denominator":
            return float(group["denominator_lr_scale"])
        if role == "atom":
            return float(group["atom_lr_scale"])
        if role == "center":
            return float(group["center_lr_scale"])
        return float(group["other_lr_scale"])

    @staticmethod
    def _curve_reduce_dims(param: torch.Tensor, role: str):
        if param.dim() == 0:
            return None
        if role in {"numerator", "denominator"}:
            return (-1,)
        if role == "atom" and param.dim() > 1:
            return tuple(range(1, param.dim()))
        if role == "center" and param.dim() > 1:
            return tuple(range(1, param.dim()))
        return tuple(range(param.dim()))

    def _normalize_curve_update(self, update: torch.Tensor, param: torch.Tensor, role: str, group: dict):
        dims = self._curve_reduce_dims(param, role)
        if dims is None:
            rms_sq = update.pow(2)
        else:
            rms_sq = update.pow(2).mean(dim=dims, keepdim=True)
        state = self.state[param]
        energy = state.get("curve_energy")
        if energy is None:
            energy = rms_sq.detach().clone()
        else:
            decay = float(group["curve_decay"])
            energy.mul_(decay).add_(rms_sq.detach(), alpha=1.0 - decay)
        state["curve_energy"] = energy
        return update / torch.sqrt(energy + float(group["eps"]))

    def _basis_degrees(self, width: int, role: str):
        if role == "numerator":
            return tuple(range(width)), 1 if width > 1 else 0, 0.02, 2.0
        if role == "denominator":
            return tuple(range(1, width + 1)), 1, 0.02, 1.0
        return None

    def _gram_metric(self, param: torch.Tensor, role: str, group: dict):
        width = int(param.shape[-1]) if param.dim() > 0 else 1
        degree_info = self._basis_degrees(width, role)
        if degree_info is None:
            return None
        degrees, _, _, _ = degree_info
        key = (
            "gram",
            role,
            width,
            float(group["probe_range"]),
            int(group["probe_points"]),
            float(group["metric_damping"]),
            param.device,
            param.dtype,
        )
        cached = self._scale_cache.get(key)
        if cached is None:
            t = torch.linspace(
                -float(group["probe_range"]),
                float(group["probe_range"]),
                int(group["probe_points"]),
                device=param.device,
                dtype=param.dtype,
            )
            basis = torch.stack([t.pow(degree) for degree in degrees], dim=-1)
            rms = torch.sqrt(basis.pow(2).mean(dim=0) + float(group["eps"]))
            normalized = basis / rms
            gram = normalized.transpose(0, 1).matmul(normalized) / normalized.size(0)
            eye = torch.eye(width, device=param.device, dtype=param.dtype)
            inv_gram = torch.linalg.inv(gram + float(group["metric_damping"]) * eye)
            cached = (rms.reciprocal(), inv_gram)
            self._scale_cache[key] = cached
        return cached

    def _metric_precondition(self, grad: torch.Tensor, param: torch.Tensor, role: str, group: dict) -> torch.Tensor:
        if str(group.get("metric", "gram")) != "gram" or role not in {"numerator", "denominator"}:
            return grad * self._basis_scale(param, role, group)
        metric = self._gram_metric(param, role, group)
        if metric is None:
            return grad
        inv_rms, inv_gram = metric
        width = int(param.shape[-1])
        flat = grad.reshape(-1, width)
        scaled = flat * inv_rms.view(1, width)
        preconditioned = scaled.matmul(inv_gram.transpose(0, 1))
        preconditioned = preconditioned * inv_rms.view(1, width)
        return preconditioned.reshape_as(grad)

    def _basis_scale(self, param: torch.Tensor, role: str, group: dict) -> torch.Tensor:
        width = int(param.shape[-1]) if param.dim() > 0 else 1
        degree_info = self._basis_degrees(width, role)
        if degree_info is None:
            return torch.ones_like(param)
        degrees, anchor_degree, min_scale, max_scale = degree_info

        key = (
            role,
            width,
            float(group["probe_range"]),
            int(group["probe_points"]),
            param.device,
            param.dtype,
        )
        cached = self._scale_cache.get(key)
        if cached is None:
            t = torch.linspace(
                -float(group["probe_range"]),
                float(group["probe_range"]),
                int(group["probe_points"]),
                device=param.device,
                dtype=param.dtype,
            )
            powers = torch.stack([t.abs().pow(degree) for degree in degrees], dim=-1)
            rms = torch.sqrt(powers.pow(2).mean(dim=0) + float(group["eps"]))
            inverse = rms.reciprocal()
            anchor_index = degrees.index(anchor_degree) if anchor_degree in degrees else 0
            scale = inverse / inverse[anchor_index].clamp_min(float(group["eps"]))
            scale = torch.clamp(scale, min=min_scale, max=max_scale)
            cached = scale
            self._scale_cache[key] = cached

        view_shape = [1] * param.dim()
        view_shape[-1] = width
        return cached.view(view_shape).expand_as(param)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            role = str(group.get("role", "other"))
            lr = float(group["lr"]) * self._role_lr_scale(group, role)
            trust = float(group["trust"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("FunctionSpaceRationalOptimizer does not support sparse gradients")
                state = self.state[param]
                state["step"] = int(state.get("step", 0)) + 1
                update = self._metric_precondition(grad, param, role, group)
                update = self._normalize_curve_update(update, param, role, group)
                update = update.mul(-lr * float(group["update_gain"]))
                if trust > 0.0:
                    update = update.clamp(min=-trust, max=trust)
                param.add_(update)
        return loss
