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
        denominator_lr_scale_final: float | None = None,
        atom_lr_scale: float = 2.25,
        atom_lr_scale_final: float | None = None,
        center_lr_scale: float = 0.10,
        other_lr_scale: float = 0.25,
        trust: float = 0.01,
        trust_final: float | None = None,
        probe_range: float = 5.0,
        probe_points: int = 257,
        curve_decay: float = 0.95,
        update_gain: float = 4.5,
        update_gain_final: float | None = None,
        update_gain_decay_start: float = 1.1,
        update_gain_decay_end: float = 1.1,
        update_depth_gain: float = 0.0,
        update_switch_depth_shift: float = 0.0,
        reset_on_switch: bool = False,
        selector_groups: Iterable[dict] | None = None,
        select_strength: float = 0.0,
        select_start: float = 0.25,
        select_end: float = 0.55,
        select_activity_threshold: float = 0.10,
        select_activity_width: float = 0.40,
        select_pressure_weight: float = 0.25,
        denominator_decay: float = 0.0,
        atom_decay: float = 0.0,
        total_steps: int = 0,
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
            "denominator_lr_scale_final": denominator_lr_scale_final,
            "atom_lr_scale": atom_lr_scale,
            "atom_lr_scale_final": atom_lr_scale_final,
            "center_lr_scale": center_lr_scale,
            "other_lr_scale": other_lr_scale,
            "trust": trust,
            "trust_final": trust_final,
            "probe_range": probe_range,
            "probe_points": probe_points,
            "curve_decay": curve_decay,
            "update_gain": update_gain,
            "update_gain_final": update_gain_final,
            "update_gain_decay_start": update_gain_decay_start,
            "update_gain_decay_end": update_gain_decay_end,
            "update_depth_gain": update_depth_gain,
            "update_switch_depth_shift": update_switch_depth_shift,
            "reset_on_switch": bool(reset_on_switch),
            "select_strength": select_strength,
            "select_start": select_start,
            "select_end": select_end,
            "select_activity_threshold": select_activity_threshold,
            "select_activity_width": select_activity_width,
            "select_pressure_weight": select_pressure_weight,
            "denominator_decay": denominator_decay,
            "atom_decay": atom_decay,
            "total_steps": int(total_steps),
            "metric": metric,
            "metric_damping": metric_damping,
            "eps": eps,
        }
        super().__init__(params, defaults)
        self._scale_cache: dict[tuple, torch.Tensor] = {}
        self.selector_groups = list(selector_groups) if selector_groups is not None else []
        self._selector_phase_cache: dict[tuple[int, int], float] = {}

    @staticmethod
    def _schedule_phase(group: dict, step: int) -> float:
        total_steps = int(group.get("total_steps", 0))
        if total_steps <= 1:
            return 0.0
        progress = min(1.0, max(0.0, float(step) / float(total_steps)))
        start = float(group.get("update_gain_decay_start", 1.1))
        end = float(group.get("update_gain_decay_end", 1.1))
        shift = float(group.get("update_switch_depth_shift", 0.0))
        if shift != 0.0:
            layer_index = int(group.get("layer_index", -1))
            num_layers = int(group.get("num_layers", 0))
            if layer_index >= 0 and num_layers > 1:
                depth = float(layer_index) / float(num_layers - 1)
                offset = shift * (depth - 0.5)
                start = min(1.0, max(0.0, start + offset))
                end = min(1.0, max(0.0, end + offset))
        return FunctionSpaceRationalOptimizer._smoothstep(start, end, progress)

    @staticmethod
    def _scheduled_value(group: dict, key: str, final_key: str, step: int, phase: float | None = None) -> float:
        value = float(group[key])
        final = group.get(final_key)
        if final is None:
            return value
        if phase is None:
            phase = FunctionSpaceRationalOptimizer._schedule_phase(group, step)
        return value * (1.0 - phase) + float(final) * phase

    @staticmethod
    def _role_lr_scale(group: dict, role: str, step: int, phase: float | None = None) -> float:
        if role == "numerator":
            return float(group["numerator_lr_scale"])
        if role == "denominator":
            return FunctionSpaceRationalOptimizer._scheduled_value(
                group, "denominator_lr_scale", "denominator_lr_scale_final", step, phase
            )
        if role == "atom":
            return FunctionSpaceRationalOptimizer._scheduled_value(
                group, "atom_lr_scale", "atom_lr_scale_final", step, phase
            )
        if role == "center":
            return float(group["center_lr_scale"])
        return float(group["other_lr_scale"])

    @staticmethod
    def _role_decay(group: dict, role: str, phase: float) -> float:
        if phase <= 0.0:
            return 0.0
        if role == "denominator":
            return float(group.get("denominator_decay", 0.0)) * phase
        if role == "atom":
            return float(group.get("atom_decay", 0.0)) * phase
        return 0.0

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

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: float) -> float:
        if edge1 <= edge0:
            return 1.0 if x >= edge1 else 0.0
        t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _scheduled_update_gain(group: dict, step: int, phase: float | None = None) -> float:
        gain = float(group["update_gain"])
        final = group.get("update_gain_final")
        total_steps = int(group.get("total_steps", 0))
        if phase is None:
            phase = 0.0
        if final is None or total_steps <= 1:
            scheduled = gain
        else:
            if phase == 0.0:
                phase = FunctionSpaceRationalOptimizer._schedule_phase(group, step)
            scheduled = gain * (1.0 - phase) + float(final) * phase

        depth_gain = float(group.get("update_depth_gain", 0.0)) * (1.0 - phase)
        if depth_gain == 0.0:
            return scheduled
        layer_index = int(group.get("layer_index", -1))
        num_layers = int(group.get("num_layers", 0))
        if layer_index < 0 or num_layers <= 1:
            return scheduled
        depth = float(layer_index) / float(num_layers - 1)
        factor = min(1.50, max(0.50, 1.0 + depth_gain * (0.5 - depth)))
        return scheduled * factor

    def _onpolicy_switch_phase(self, group: dict, step: int) -> float:
        strength = float(group.get("select_strength", 0.0))
        if strength <= 0.0:
            return 0.0
        selector_index = int(group.get("selector_index", -1))
        if selector_index < 0 or selector_index >= len(self.selector_groups):
            return 0.0
        cache_key = (int(step), selector_index)
        cached = self._selector_phase_cache.get(cache_key)
        if cached is not None:
            return cached

        total_steps = int(group.get("total_steps", 0))
        if total_steps <= 1:
            return 0.0
        progress = min(1.0, max(0.0, float(step) / float(total_steps)))
        gate = self._smoothstep(float(group.get("select_start", 0.25)), float(group.get("select_end", 0.55)), progress)
        if gate <= 0.0:
            self._selector_phase_cache[cache_key] = 0.0
            return 0.0

        state = self.selector_groups[selector_index].get("_onpolicy")
        if state is None:
            self._selector_phase_cache[cache_key] = 0.0
            return 0.0

        eps = float(group["eps"])
        in_rel = state["in_rel_ema"].detach().float().clamp_min(eps)
        out_rel = state["out_rel_ema"].detach().float().clamp_min(eps)
        rat_rel = state["rat_rel_ema"].detach().float().clamp_min(eps)
        log_in = torch.log(in_rel)
        log_out = torch.log(out_rel)
        matrix_log = 0.5 * (log_in + log_out)
        rational_activity = (torch.log(rat_rel) - matrix_log).mean()
        pressure = (log_in - log_out).abs().mean()
        signal = rational_activity + float(group.get("select_pressure_weight", 0.25)) * pressure
        threshold = float(group.get("select_activity_threshold", 0.10))
        width = max(float(group.get("select_activity_width", 0.40)), eps)
        trigger = self._smoothstep(threshold, threshold + width, float(signal.item()))
        phase = min(1.0, max(0.0, strength * gate * trigger))
        self._selector_phase_cache[cache_key] = phase
        return phase

    def _effective_phase(self, group: dict, step: int) -> float:
        scheduled = self._schedule_phase(group, step)
        selected = self._onpolicy_switch_phase(group, step)
        return min(1.0, max(scheduled, selected))

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
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("FunctionSpaceRationalOptimizer does not support sparse gradients")
                state = self.state[param]
                state["step"] = int(state.get("step", 0)) + 1
                step = state["step"]
                phase = self._effective_phase(group, step)
                if bool(group.get("reset_on_switch", False)) and phase >= 1.0 and not state.get("switch_reset_done", False):
                    state.pop("curve_energy", None)
                    state["switch_reset_done"] = True
                lr = float(group["lr"]) * self._role_lr_scale(group, role, step, phase)
                trust = self._scheduled_value(group, "trust", "trust_final", step, phase)
                update_gain = self._scheduled_update_gain(group, step, phase)
                update = self._metric_precondition(grad, param, role, group)
                update = self._normalize_curve_update(update, param, role, group)
                update = update.mul(-lr * update_gain)
                decay = self._role_decay(group, role, phase)
                if decay > 0.0:
                    update.add_(param, alpha=-lr * decay)
                if trust > 0.0:
                    update = update.clamp(min=-trust, max=trust)
                param.add_(update)
        return loss
