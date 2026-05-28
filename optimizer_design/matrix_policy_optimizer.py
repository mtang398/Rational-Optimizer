"""Layerwise matrix-policy optimizer for RLB FFN matrices."""

from __future__ import annotations

import torch


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


class RationalMatrixPolicyOptimizer:
    """Layer/side-specific AdamW policy for RLB FFN matrices.

    The verified default uses a short early Muon phase on RLB matrices,
    then switches back to the RLB-specific AdamW matrix policy.
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        total_steps: int = 0,
        selector_groups=None,
        muon_strength: float = 0.75,
        muon_lr_scale: float = 1.00,
        adam_lr_scale: float = 3.0,
        adam_lr_scale_final: float | None = None,
        adam_decay_start: float = 1.1,
        adam_decay_end: float = 1.1,
        adam_decay_depth_shift: float = 0.0,
        adam_beta2_final: float | None = None,
        adam_beta2_decay_start: float = 1.1,
        adam_beta2_decay_end: float = 1.1,
        adam_beta2_decay_depth_shift: float = 0.0,
        adam_role_strength: float = 1.20,
        adam_stat_strength: float = 0.0,
        adam_pressure_balance: float = 0.0,
        adam_stat_start: float = 0.0,
        adam_stat_end: float = 0.0,
        adam_min_lr_scale: float = 0.40,
        adam_max_lr_scale: float = 4.0,
        adam_reset_on_switch: bool = False,
        start: float = 0.02,
        end: float = 0.12,
        decay_start: float = 0.20,
        decay_end: float = 0.36,
        muon_decay_depth_shift: float = 0.0,
        muon_input_decay_shift: float = 0.0,
        muon_output_decay_shift: float = 0.0,
        muon_reset_adam_state: bool = False,
        final_muon: float = 0.0,
        min_muon: float = 0.0,
        max_muon: float = 0.75,
        input_depth_gain: float = -0.50,
        output_depth_gain: float = 1.00,
        pressure_weight: float = 0.30,
        activity_weight: float = 0.65,
        activity_target: float = 0.05,
        activity_width: float = 0.45,
        pressure_clip: float = 1.50,
        group_gain_strength: float = 0.0,
        group_pressure_strength: float = 0.0,
        group_activity_damping: float = 0.0,
        group_activity_target: float = 0.05,
        group_activity_width: float = 0.45,
        group_start: float = 0.02,
        group_end: float = 0.35,
        group_min_scale: float = 0.65,
        group_max_scale: float = 1.55,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str | None = "match_rms_adamw",
    ):
        self.total_steps = int(total_steps)
        self.selector_groups = list(selector_groups) if selector_groups is not None else []
        self.muon_strength = float(muon_strength)
        self.muon_lr_scale = float(muon_lr_scale)
        self.adam_lr_scale = float(adam_lr_scale)
        self.adam_lr_scale_final = None if adam_lr_scale_final is None else float(adam_lr_scale_final)
        self.adam_decay_start = float(adam_decay_start)
        self.adam_decay_end = float(adam_decay_end)
        self.adam_decay_depth_shift = float(adam_decay_depth_shift)
        self.adam_beta1 = float(betas[0])
        self.adam_beta2 = float(betas[1])
        self.adam_beta2_final = None if adam_beta2_final is None else float(adam_beta2_final)
        self.adam_beta2_decay_start = float(adam_beta2_decay_start)
        self.adam_beta2_decay_end = float(adam_beta2_decay_end)
        self.adam_beta2_decay_depth_shift = float(adam_beta2_decay_depth_shift)
        self.adam_role_strength = float(adam_role_strength)
        self.adam_stat_strength = float(adam_stat_strength)
        self.adam_pressure_balance = float(adam_pressure_balance)
        self.adam_stat_start = float(adam_stat_start)
        self.adam_stat_end = float(adam_stat_end)
        self.adam_min_lr_scale = float(adam_min_lr_scale)
        self.adam_max_lr_scale = float(adam_max_lr_scale)
        self.adam_reset_on_switch = bool(adam_reset_on_switch)
        self.start = float(start)
        self.end = float(end)
        self.decay_start = float(decay_start)
        self.decay_end = float(decay_end)
        self.muon_decay_depth_shift = float(muon_decay_depth_shift)
        self.muon_input_decay_shift = float(muon_input_decay_shift)
        self.muon_output_decay_shift = float(muon_output_decay_shift)
        self.muon_reset_adam_state = bool(muon_reset_adam_state)
        self.final_muon = float(final_muon)
        self.min_muon = float(min_muon)
        self.max_muon = float(max_muon)
        self.input_depth_gain = float(input_depth_gain)
        self.output_depth_gain = float(output_depth_gain)
        self.pressure_weight = float(pressure_weight)
        self.activity_weight = float(activity_weight)
        self.activity_target = float(activity_target)
        self.activity_width = max(float(activity_width), eps)
        self.pressure_clip = float(pressure_clip)
        self.group_gain_strength = float(group_gain_strength)
        self.group_pressure_strength = float(group_pressure_strength)
        self.group_activity_damping = float(group_activity_damping)
        self.group_activity_target = float(group_activity_target)
        self.group_activity_width = max(float(group_activity_width), eps)
        self.group_start = float(group_start)
        self.group_end = float(group_end)
        self.group_min_scale = float(group_min_scale)
        self.group_max_scale = float(group_max_scale)
        self.eps = float(eps)
        self.step_index = 0
        self.use_muon = (
            self.muon_lr_scale != 0.0
            and self.max_muon > 0.0
            and (self.muon_strength != 0.0 or self.final_muon != 0.0 or self.min_muon > 0.0)
        )
        if self.adam_beta2_final is not None and not (0.0 <= self.adam_beta2_final < 1.0):
            raise ValueError("adam_beta2_final must be in [0, 1)")
        if self.adam_min_lr_scale < 0.0:
            raise ValueError("adam_min_lr_scale must be non-negative")
        if self.adam_max_lr_scale < self.adam_min_lr_scale:
            raise ValueError("adam_max_lr_scale must be >= adam_min_lr_scale")
        if self.max_muon < self.min_muon:
            raise ValueError("max_muon must be >= min_muon")
        if self.group_min_scale <= 0.0:
            raise ValueError("group_min_scale must be positive")
        if self.group_max_scale < self.group_min_scale:
            raise ValueError("group_max_scale must be >= group_min_scale")

        adam_groups = []
        muon_groups = []
        for group in params:
            meta = {
                "layer_index": int(group.get("layer_index", -1)),
                "num_layers": int(group.get("num_layers", 0)),
                "selector_index": int(group.get("selector_index", -1)),
                "matrix_role": str(group.get("matrix_role", "matrix")),
                "weight_decay": float(group.get("weight_decay", weight_decay)),
            }
            adam_groups.append({"params": list(group["params"]), **meta})
            muon_groups.append({"params": list(group["params"]), **meta})

        self.adam = torch.optim.AdamW(
            adam_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.muon = None
        if self.use_muon:
            self.muon = torch.optim.Muon(
                muon_groups,
                lr=lr,
                weight_decay=weight_decay,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr_fn,
            )
        self.param_groups = self.adam.param_groups if self.muon is None else self.adam.param_groups + self.muon.param_groups

    def zero_grad(self, set_to_none=True):
        seen = set()
        for optimizer in (self.adam, self.muon):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
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
        state = {
            "adam": self.adam.state_dict(),
            "step_index": self.step_index,
            "use_muon": self.use_muon,
        }
        if self.muon is not None:
            state["muon"] = self.muon.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.adam.load_state_dict(state_dict["adam"])
        if self.muon is not None and "muon" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
        self.step_index = int(state_dict.get("step_index", 0))

    def _progress(self) -> float:
        if self.total_steps <= 1:
            return 0.0
        return min(1.0, max(0.0, float(self.step_index) / float(self.total_steps)))

    @staticmethod
    def _centered_inverse_scale(gain: torch.Tensor, strength: float, eps: float) -> torch.Tensor:
        gain_f = gain.detach().float().clamp_min(eps)
        center = torch.exp(torch.log(gain_f).mean()).clamp_min(eps)
        return (center / gain_f).pow(float(strength))

    def _group_policy_phase(self) -> float:
        return _smoothstep(self.group_start, self.group_end, self._progress())

    def _group_policy_enabled(self) -> bool:
        return (
            self.group_gain_strength != 0.0
            or self.group_pressure_strength != 0.0
            or self.group_activity_damping != 0.0
        )

    @staticmethod
    def _depth(group: dict) -> float:
        layer = int(group.get("layer_index", -1))
        layers = max(1, int(group.get("num_layers", 1)))
        if layer < 0 or layers <= 1:
            return 0.5
        return min(1.0, max(0.0, float(layer) / float(layers - 1)))

    def _role_depth_factor(self, group: dict) -> float:
        depth = self._depth(group)
        role = str(group.get("matrix_role", "matrix"))
        gain = self.output_depth_gain if role == "out" else self.input_depth_gain
        return min(1.40, max(0.55, 1.0 + gain * (depth - 0.5)))

    def _adam_decay_phase(self, group: dict) -> float:
        if self.adam_lr_scale_final is None:
            return 0.0
        progress = self._progress()
        offset = self.adam_decay_depth_shift * (self._depth(group) - 0.5)
        start = min(1.0, max(0.0, self.adam_decay_start + offset))
        end = min(1.0, max(0.0, self.adam_decay_end + offset))
        return _smoothstep(start, end, progress)

    def _adam_lr_scale(self, group: dict) -> float:
        if self.adam_lr_scale_final is None:
            scheduled = self.adam_lr_scale
        else:
            phase = self._adam_decay_phase(group)
            scheduled = self.adam_lr_scale * (1.0 - phase) + self.adam_lr_scale_final * phase
        factor = self._adam_role_factor(group) * self._adam_stat_factor(group)
        scale = scheduled * factor
        return min(self.adam_max_lr_scale, max(self.adam_min_lr_scale, scale))

    def _adam_beta2_phase(self, group: dict) -> float:
        if self.adam_beta2_final is None:
            return 0.0
        progress = self._progress()
        offset = self.adam_beta2_decay_depth_shift * (self._depth(group) - 0.5)
        start = min(1.0, max(0.0, self.adam_beta2_decay_start + offset))
        end = min(1.0, max(0.0, self.adam_beta2_decay_end + offset))
        return _smoothstep(start, end, progress)

    def _adam_betas(self, group: dict) -> tuple[float, float]:
        if self.adam_beta2_final is None:
            return group["betas"]
        phase = self._adam_beta2_phase(group)
        beta2 = self.adam_beta2 * (1.0 - phase) + self.adam_beta2_final * phase
        return (self.adam_beta1, min(0.9999, max(0.0, beta2)))

    def _adam_role_factor(self, group: dict) -> float:
        if self.adam_role_strength == 0.0:
            return 1.0
        role_factor = self._role_depth_factor(group)
        return max(0.10, 1.0 + self.adam_role_strength * (role_factor - 1.0))

    def _adam_stat_phase(self) -> float:
        return _smoothstep(self.adam_stat_start, self.adam_stat_end, self._progress())

    def _adam_stat_factor(self, group: dict) -> float:
        if self.adam_stat_strength == 0.0 and self.adam_pressure_balance == 0.0:
            return 1.0
        phase = self._adam_stat_phase()
        if phase <= 0.0:
            return 1.0
        selector_index = int(group.get("selector_index", -1))
        if selector_index < 0 or selector_index >= len(self.selector_groups):
            return 1.0
        state = self.selector_groups[selector_index].get("_onpolicy")
        if state is None:
            return 1.0

        in_rel = state["in_rel_ema"].detach().float().clamp_min(self.eps)
        out_rel = state["out_rel_ema"].detach().float().clamp_min(self.eps)
        rat_rel = state["rat_rel_ema"].detach().float().clamp_min(self.eps)
        log_in = torch.log(in_rel)
        log_out = torch.log(out_rel)
        pressure = (log_in - log_out).mean().clamp(min=-self.pressure_clip, max=self.pressure_clip)
        role = str(group.get("matrix_role", "matrix"))
        pressure_direction = -pressure if role == "in" else pressure
        pressure_factor = torch.exp((self.adam_pressure_balance * phase) * pressure_direction)

        matrix_log = 0.5 * (log_in + log_out)
        rational_activity = (torch.log(rat_rel) - matrix_log).mean()
        excess_activity = torch.relu((rational_activity - self.activity_target) / self.activity_width)
        activity_factor = torch.exp(-self.activity_weight * self.adam_stat_strength * phase * excess_activity)

        factor = pressure_factor * activity_factor
        return float(factor.clamp(0.35, 1.45).item())

    def _group_policy_scale(self, group: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self._group_policy_enabled():
            return None
        phase = self._group_policy_phase()
        if phase <= 0.0:
            return None
        selector_index = int(group.get("selector_index", -1))
        if selector_index < 0 or selector_index >= len(self.selector_groups):
            return None
        curve_group = self.selector_groups[selector_index]
        groups = int(curve_group.get("groups", 0))
        if groups <= 0:
            return None

        scale = torch.ones(groups, device=device, dtype=torch.float32)
        role = str(group.get("matrix_role", "matrix"))
        module = curve_group.get("module")
        stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
        if self.group_gain_strength != 0.0 and stats:
            key = "derivative_rms" if role == "in" else "output_rms"
            gain = stats.get(key)
            if torch.is_tensor(gain) and gain.numel() == groups:
                scale.mul_(
                    self._centered_inverse_scale(
                        gain.to(device=device),
                        self.group_gain_strength * phase,
                        self.eps,
                    ).to(device=device)
                )

        state = curve_group.get("_onpolicy")
        if state is not None and (self.group_pressure_strength != 0.0 or self.group_activity_damping != 0.0):
            in_rel = state["in_rel_ema"].to(device=device, dtype=torch.float32).clamp_min(self.eps)
            out_rel = state["out_rel_ema"].to(device=device, dtype=torch.float32).clamp_min(self.eps)
            rat_rel = state["rat_rel_ema"].to(device=device, dtype=torch.float32).clamp_min(self.eps)
            if in_rel.numel() == groups and out_rel.numel() == groups and rat_rel.numel() == groups:
                if self.group_pressure_strength != 0.0:
                    pressure = (torch.log(in_rel) - torch.log(out_rel)).clamp(
                        min=-self.pressure_clip,
                        max=self.pressure_clip,
                    )
                    direction = -pressure if role == "in" else pressure
                    scale.mul_(torch.exp(self.group_pressure_strength * phase * direction))
                if self.group_activity_damping != 0.0:
                    matrix_log = 0.5 * (torch.log(in_rel) + torch.log(out_rel))
                    rational_activity = torch.log(rat_rel) - matrix_log
                    excess = torch.relu((rational_activity - self.group_activity_target) / self.group_activity_width)
                    scale.mul_(torch.exp(-self.group_activity_damping * phase * excess))

        scale = scale.clamp_min(self.eps)
        scale = scale / torch.exp(torch.log(scale).mean()).clamp_min(self.eps)
        scale = scale.clamp(self.group_min_scale, self.group_max_scale)
        return scale.to(device=device, dtype=dtype)

    def _apply_group_policy_to_gradients(self):
        if not self._group_policy_enabled():
            return
        for group in self.adam.param_groups:
            role = str(group.get("matrix_role", "matrix"))
            if role not in {"in", "out"}:
                continue
            selector_index = int(group.get("selector_index", -1))
            if selector_index < 0 or selector_index >= len(self.selector_groups):
                continue
            curve_group = self.selector_groups[selector_index]
            groups = int(curve_group.get("groups", 0))
            hidden_dim = int(curve_group.get("hidden_dim", 0))
            if groups <= 0 or hidden_dim <= 0 or hidden_dim % groups != 0:
                continue
            width = hidden_dim // groups
            for param in group["params"]:
                if param.grad is None:
                    continue
                scale = self._group_policy_scale(group, param.grad.device, param.grad.dtype)
                if scale is None:
                    continue
                if role == "in":
                    if param.grad.shape[0] != hidden_dim:
                        continue
                    param.grad.view(groups, width, -1).mul_(scale.view(groups, 1, 1))
                else:
                    if param.grad.shape[1] != hidden_dim:
                        continue
                    param.grad.view(param.grad.shape[0], groups, width).permute(1, 2, 0).mul_(
                        scale.view(groups, 1, 1)
                    )

    def _maybe_reset_adam_state(self, group: dict):
        if not self.adam_reset_on_switch or self.adam_lr_scale_final is None:
            return
        if group.get("_adam_reset_done"):
            return
        if self._adam_decay_phase(group) < 0.999:
            return
        for param in group["params"]:
            self.adam.state.pop(param, None)
        group["_adam_reset_done"] = True

    def _stat_factor(self, group: dict) -> float:
        selector_index = int(group.get("selector_index", -1))
        if selector_index < 0 or selector_index >= len(self.selector_groups):
            return 1.0
        state = self.selector_groups[selector_index].get("_onpolicy")
        if state is None:
            return 1.0

        in_rel = state["in_rel_ema"].detach().float().clamp_min(self.eps)
        out_rel = state["out_rel_ema"].detach().float().clamp_min(self.eps)
        rat_rel = state["rat_rel_ema"].detach().float().clamp_min(self.eps)
        log_in = torch.log(in_rel)
        log_out = torch.log(out_rel)
        pressure = (log_in - log_out).abs().mean().clamp(max=self.pressure_clip)
        matrix_log = 0.5 * (log_in + log_out)
        rational_activity = (torch.log(rat_rel) - matrix_log).mean()
        excess_activity = torch.relu((rational_activity - self.activity_target) / self.activity_width)
        penalty = self.pressure_weight * pressure + self.activity_weight * excess_activity
        return float(torch.exp(-penalty).clamp(0.10, 1.15).item())

    def _muon_fraction(self, group: dict) -> float:
        progress = self._progress()
        depth_offset = self.muon_decay_depth_shift * (self._depth(group) - 0.5)
        role = str(group.get("matrix_role", "matrix"))
        role_offset = self.muon_output_decay_shift if role == "out" else self.muon_input_decay_shift
        offset = depth_offset + role_offset
        start = min(1.0, max(0.0, self.start + offset))
        end = min(1.0, max(0.0, self.end + offset))
        decay_start = min(1.0, max(0.0, self.decay_start + offset))
        decay_end = min(1.0, max(0.0, self.decay_end + offset))
        on_phase = _smoothstep(start, end, progress)
        off_phase = _smoothstep(decay_start, decay_end, progress)
        base_strength = self.muon_strength * (1.0 - off_phase) + self.final_muon * off_phase
        base = base_strength * on_phase
        value = base * self._role_depth_factor(group) * self._stat_factor(group)
        return min(self.max_muon, max(self.min_muon, value))

    def _maybe_reset_adam_after_muon(self, group: dict, fraction: float):
        if not self.muon_reset_adam_state:
            return
        if fraction > 1.0e-4:
            group["_muon_was_active"] = True
            return
        if not group.get("_muon_was_active") or group.get("_muon_adam_reset_done"):
            return
        for param in group["params"]:
            self.adam.state.pop(param, None)
        group["_muon_adam_reset_done"] = True

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("RationalMatrixPolicyOptimizer does not support closures")
        self.step_index += 1

        self._apply_group_policy_to_gradients()

        saved_adam_lrs = []
        saved_adam_betas = []
        for group in self.adam.param_groups:
            lr = float(group["lr"])
            saved_adam_lrs.append(lr)
            saved_adam_betas.append(group["betas"])
            self._maybe_reset_adam_state(group)
            fraction = self._muon_fraction(group) if self.muon is not None else 0.0
            self._maybe_reset_adam_after_muon(group, fraction)
            group["lr"] = lr * self._adam_lr_scale(group) * (1.0 - fraction)
            group["betas"] = self._adam_betas(group)

        saved_muon_lrs = []
        if self.muon is not None:
            for group in self.muon.param_groups:
                lr = float(group["lr"])
                saved_muon_lrs.append(lr)
                fraction = self._muon_fraction(group)
                group["lr"] = lr * self.muon_lr_scale * fraction

        self.adam.step()
        if self.muon is not None:
            self.muon.step()

        for group, lr, betas in zip(self.adam.param_groups, saved_adam_lrs, saved_adam_betas):
            group["lr"] = lr
            group["betas"] = betas
        if self.muon is not None:
            for group, lr in zip(self.muon.param_groups, saved_muon_lrs):
                group["lr"] = lr
        return None
