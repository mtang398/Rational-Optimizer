"""Stagewise optimizer for rational activation coefficients."""

from __future__ import annotations

import torch

from .function_space_rational_optimizer import FunctionSpaceRationalOptimizer


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


class SwitchingRationalOptimizer:
    """Blend AdamW and function-space updates for rational coefficients.

    Early training keeps the rational coefficients on AdamW, which is the most
    stable baseline observed so far. Later training shifts each layer into the
    function-space rational optimizer, with deeper layers allowed to switch at a
    different time. The switch can also be pulled forward by on-policy pressure
    statistics collected by the outer RLB optimizer wrapper.
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        total_steps: int = 0,
        switch_start: float = 0.36,
        switch_end: float = 0.58,
        switch_depth_shift: float = -0.16,
        adam_lr_scale: float = 1.0,
        function_lr_scale: float = 1.0,
        select_strength: float = 0.35,
        select_start: float = 0.20,
        select_end: float = 0.55,
        select_activity_threshold: float = 0.08,
        select_activity_width: float = 0.32,
        select_pressure_weight: float = 0.25,
        selector_groups=None,
        **function_kwargs,
    ):
        self.total_steps = int(total_steps)
        self.switch_start = float(switch_start)
        self.switch_end = float(switch_end)
        self.switch_depth_shift = float(switch_depth_shift)
        self.adam_lr_scale = float(adam_lr_scale)
        self.function_lr_scale = float(function_lr_scale)
        self.select_strength = float(select_strength)
        self.select_start = float(select_start)
        self.select_end = float(select_end)
        self.select_activity_threshold = float(select_activity_threshold)
        self.select_activity_width = float(select_activity_width)
        self.select_pressure_weight = float(select_pressure_weight)
        self.selector_groups = list(selector_groups) if selector_groups is not None else []
        self.eps = float(eps)
        self.step_index = 0
        function_eps = float(function_kwargs.pop("function_eps", eps))

        function_groups = [dict(group) for group in params]
        adam_groups = []
        for group in params:
            adam_group = {
                "params": list(group["params"]),
                "weight_decay": float(group.get("weight_decay", weight_decay)),
                "layer_index": int(group.get("layer_index", -1)),
                "num_layers": int(group.get("num_layers", 0)),
                "selector_index": int(group.get("selector_index", -1)),
            }
            adam_groups.append(adam_group)

        self.adam = torch.optim.AdamW(
            adam_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.function = FunctionSpaceRationalOptimizer(
            function_groups,
            lr=lr,
            eps=function_eps,
            total_steps=total_steps,
            **function_kwargs,
        )
        self.param_groups = self.adam.param_groups + self.function.param_groups

    def zero_grad(self, set_to_none=True):
        seen = set()
        for group in self.adam.param_groups:
            for param in group["params"]:
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                if param.grad is None:
                    continue
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()

    def state_dict(self):
        return {
            "adam": self.adam.state_dict(),
            "function": self.function.state_dict(),
            "step_index": self.step_index,
        }

    def load_state_dict(self, state_dict):
        self.adam.load_state_dict(state_dict["adam"])
        self.function.load_state_dict(state_dict["function"])
        self.step_index = int(state_dict.get("step_index", 0))

    @staticmethod
    def _depth(group: dict) -> float:
        layer = int(group.get("layer_index", -1))
        layers = max(1, int(group.get("num_layers", 1)))
        if layer < 0 or layers <= 1:
            return 0.5
        return min(1.0, max(0.0, float(layer) / float(layers - 1)))

    def _scheduled_phase(self, group: dict) -> float:
        if self.total_steps <= 1:
            return 0.0
        progress = min(1.0, max(0.0, float(self.step_index) / float(self.total_steps)))
        offset = self.switch_depth_shift * (self._depth(group) - 0.5)
        start = min(1.0, max(0.0, self.switch_start + offset))
        end = min(1.0, max(0.0, self.switch_end + offset))
        return _smoothstep(start, end, progress)

    def _selected_phase(self, group: dict) -> float:
        if self.select_strength <= 0.0 or self.total_steps <= 1:
            return 0.0
        selector_index = int(group.get("selector_index", -1))
        if selector_index < 0 or selector_index >= len(self.selector_groups):
            return 0.0
        progress = min(1.0, max(0.0, float(self.step_index) / float(self.total_steps)))
        gate = _smoothstep(self.select_start, self.select_end, progress)
        if gate <= 0.0:
            return 0.0
        state = self.selector_groups[selector_index].get("_onpolicy")
        if state is None:
            return 0.0
        eps = self.eps
        in_rel = state["in_rel_ema"].detach().float().clamp_min(eps)
        out_rel = state["out_rel_ema"].detach().float().clamp_min(eps)
        rat_rel = state["rat_rel_ema"].detach().float().clamp_min(eps)
        matrix_log = 0.5 * (torch.log(in_rel) + torch.log(out_rel))
        rational_activity = torch.log(rat_rel) - matrix_log
        pressure = (torch.log(in_rel) - torch.log(out_rel)).abs()
        signal = rational_activity.mean() + self.select_pressure_weight * pressure.mean()
        width = max(self.select_activity_width, eps)
        trigger = _smoothstep(
            self.select_activity_threshold,
            self.select_activity_threshold + width,
            float(signal.item()),
        )
        return min(1.0, max(0.0, self.select_strength * gate * trigger))

    def _phase(self, group: dict) -> float:
        return min(1.0, max(self._scheduled_phase(group), self._selected_phase(group)))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("SwitchingRationalOptimizer does not support closures")
        self.step_index += 1

        saved_adam_lrs = []
        for group in self.adam.param_groups:
            lr = float(group["lr"])
            saved_adam_lrs.append(lr)
            group["lr"] = lr * self.adam_lr_scale * (1.0 - self._phase(group))

        saved_function_lrs = []
        for group in self.function.param_groups:
            lr = float(group["lr"])
            saved_function_lrs.append(lr)
            group["lr"] = lr * self.function_lr_scale * self._phase(group)

        self.adam.step()
        self.function.step()

        for group, lr in zip(self.adam.param_groups, saved_adam_lrs):
            group["lr"] = lr
        for group, lr in zip(self.function.param_groups, saved_function_lrs):
            group["lr"] = lr
        return None
