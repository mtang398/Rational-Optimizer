"""Layerwise matrix-policy optimizer for RLB Transformer MLP matrices."""

from __future__ import annotations

import torch


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


class RationalMatrixPolicyOptimizer:
    """Layer/side-specific AdamW policy for RLB Transformer MLP matrices.

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
        adam_beta2_input_final: float | None = None,
        adam_beta2_output_final: float | None = None,
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
        muon_tail_strength: float = 0.0,
        muon_tail_start: float = 0.24,
        muon_tail_end: float = 0.48,
        muon_tail_decay_start: float = 0.90,
        muon_tail_decay_end: float = 1.00,
        muon_tail_pressure_weight: float = 0.35,
        muon_tail_activity_weight: float = 0.35,
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
        functional_balance_strength: float = 0.0,
        functional_balance_start: float = 0.02,
        functional_balance_end: float = 0.12,
        functional_balance_decay_start: float = 0.30,
        functional_balance_decay_end: float = 0.50,
        functional_balance_target_depth_gain: float = 0.80,
        functional_balance_clip: float = 0.47,
        functional_balance_min_scale: float = 0.80,
        functional_balance_max_scale: float = 1.25,
        functional_metric_strength: float = 0.0,
        functional_metric_start: float = 0.02,
        functional_metric_end: float = 0.20,
        functional_metric_decay_start: float = 1.1,
        functional_metric_decay_end: float = 1.1,
        functional_metric_min_scale: float = 0.70,
        functional_metric_max_scale: float = 1.45,
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
        self.adam_beta2_input_final = None if adam_beta2_input_final is None else float(adam_beta2_input_final)
        self.adam_beta2_output_final = None if adam_beta2_output_final is None else float(adam_beta2_output_final)
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
        self.muon_tail_strength = float(muon_tail_strength)
        self.muon_tail_start = float(muon_tail_start)
        self.muon_tail_end = float(muon_tail_end)
        self.muon_tail_decay_start = float(muon_tail_decay_start)
        self.muon_tail_decay_end = float(muon_tail_decay_end)
        self.muon_tail_pressure_weight = float(muon_tail_pressure_weight)
        self.muon_tail_activity_weight = float(muon_tail_activity_weight)
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
        self.functional_balance_strength = float(functional_balance_strength)
        self.functional_balance_start = float(functional_balance_start)
        self.functional_balance_end = float(functional_balance_end)
        self.functional_balance_decay_start = float(functional_balance_decay_start)
        self.functional_balance_decay_end = float(functional_balance_decay_end)
        self.functional_balance_target_depth_gain = float(functional_balance_target_depth_gain)
        self.functional_balance_clip = float(functional_balance_clip)
        self.functional_balance_min_scale = float(functional_balance_min_scale)
        self.functional_balance_max_scale = float(functional_balance_max_scale)
        self.functional_metric_strength = float(functional_metric_strength)
        self.functional_metric_start = float(functional_metric_start)
        self.functional_metric_end = float(functional_metric_end)
        self.functional_metric_decay_start = float(functional_metric_decay_start)
        self.functional_metric_decay_end = float(functional_metric_decay_end)
        self.functional_metric_min_scale = float(functional_metric_min_scale)
        self.functional_metric_max_scale = float(functional_metric_max_scale)
        self.eps = float(eps)
        self.step_index = 0
        self.use_muon = (
            self.muon_lr_scale != 0.0
            and self.max_muon > 0.0
            and (
                self.muon_strength != 0.0
                or self.final_muon != 0.0
                or self.muon_tail_strength != 0.0
                or self.min_muon > 0.0
            )
        )
        self._muon_fraction_cache_step = -1
        self._muon_fraction_cache = {}
        for name, value in (
            ("adam_beta2_final", self.adam_beta2_final),
            ("adam_beta2_input_final", self.adam_beta2_input_final),
            ("adam_beta2_output_final", self.adam_beta2_output_final),
        ):
            if value is not None and not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must be in [0, 1)")
        if self.adam_min_lr_scale < 0.0:
            raise ValueError("adam_min_lr_scale must be non-negative")
        if self.adam_max_lr_scale < self.adam_min_lr_scale:
            raise ValueError("adam_max_lr_scale must be >= adam_min_lr_scale")
        if self.max_muon < self.min_muon:
            raise ValueError("max_muon must be >= min_muon")
        if self.muon_tail_strength < 0.0:
            raise ValueError("muon_tail_strength must be non-negative")
        if self.group_min_scale <= 0.0:
            raise ValueError("group_min_scale must be positive")
        if self.group_max_scale < self.group_min_scale:
            raise ValueError("group_max_scale must be >= group_min_scale")
        if self.functional_balance_strength < 0.0:
            raise ValueError("functional_balance_strength must be non-negative")
        if self.functional_balance_clip < 0.0:
            raise ValueError("functional_balance_clip must be non-negative")
        if self.functional_balance_min_scale <= 0.0:
            raise ValueError("functional_balance_min_scale must be positive")
        if self.functional_balance_max_scale < self.functional_balance_min_scale:
            raise ValueError("functional_balance_max_scale must be >= functional_balance_min_scale")
        if self.functional_metric_strength < 0.0:
            raise ValueError("functional_metric_strength must be non-negative")
        if self.functional_metric_min_scale <= 0.0:
            raise ValueError("functional_metric_min_scale must be positive")
        if self.functional_metric_max_scale < self.functional_metric_min_scale:
            raise ValueError("functional_metric_max_scale must be >= functional_metric_min_scale")

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
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def set_telemetry_capture(self, enabled: bool = True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

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
        self._muon_fraction_cache_step = -1
        self._muon_fraction_cache = {}

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

    def _functional_balance_phase(self) -> float:
        if self.functional_balance_strength <= 0.0:
            return 0.0
        progress = self._progress()
        on_phase = _smoothstep(self.functional_balance_start, self.functional_balance_end, progress)
        off_phase = _smoothstep(self.functional_balance_decay_start, self.functional_balance_decay_end, progress)
        return on_phase * (1.0 - off_phase)

    def _functional_balance_enabled(self) -> bool:
        return self.functional_balance_strength > 0.0

    def _functional_metric_phase(self) -> float:
        if self.functional_metric_strength <= 0.0:
            return 0.0
        progress = self._progress()
        on_phase = _smoothstep(self.functional_metric_start, self.functional_metric_end, progress)
        off_phase = _smoothstep(self.functional_metric_decay_start, self.functional_metric_decay_end, progress)
        return self.functional_metric_strength * on_phase * (1.0 - off_phase)

    def _functional_metric_enabled(self) -> bool:
        return self.functional_metric_strength > 0.0

    def _group_policy_enabled(self) -> bool:
        return (
            self.group_gain_strength != 0.0
            or self.group_pressure_strength != 0.0
            or self.group_activity_damping != 0.0
            or self._functional_balance_enabled()
            or self._functional_metric_enabled()
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
        factor = self._adam_role_factor(group) * self._adam_stat_factor(group) * self._functional_metric_role_factor(group)
        scale = scheduled * factor
        return min(self.adam_max_lr_scale, max(self.adam_min_lr_scale, scale))

    def _adam_beta2_final_for(self, group: dict) -> float | None:
        role = str(group.get("matrix_role", "matrix"))
        if role == "in" and self.adam_beta2_input_final is not None:
            return self.adam_beta2_input_final
        if role == "out" and self.adam_beta2_output_final is not None:
            return self.adam_beta2_output_final
        return self.adam_beta2_final

    def _adam_beta2_phase(self, group: dict) -> float:
        if self._adam_beta2_final_for(group) is None:
            return 0.0
        progress = self._progress()
        offset = self.adam_beta2_decay_depth_shift * (self._depth(group) - 0.5)
        start = min(1.0, max(0.0, self.adam_beta2_decay_start + offset))
        end = min(1.0, max(0.0, self.adam_beta2_decay_end + offset))
        return _smoothstep(start, end, progress)

    def _adam_betas(self, group: dict) -> tuple[float, float]:
        beta2_final = self._adam_beta2_final_for(group)
        if beta2_final is None:
            return group["betas"]
        phase = self._adam_beta2_phase(group)
        beta2 = self.adam_beta2 * (1.0 - phase) + beta2_final * phase
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
        group_phase = self._group_policy_phase()
        balance_phase = self._functional_balance_phase()
        if group_phase <= 0.0 and balance_phase <= 0.0:
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
        if group_phase > 0.0 and self.group_gain_strength != 0.0 and stats:
            key = "derivative_rms" if role == "in" else "output_rms"
            gain = stats.get(key)
            if torch.is_tensor(gain) and gain.numel() == groups:
                scale.mul_(
                    self._centered_inverse_scale(
                        gain.to(device=device),
                        self.group_gain_strength * group_phase,
                        self.eps,
                    ).to(device=device)
                )

        state = curve_group.get("_onpolicy")
        if group_phase > 0.0 and state is not None and (self.group_pressure_strength != 0.0 or self.group_activity_damping != 0.0):
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
                    scale.mul_(torch.exp(self.group_pressure_strength * group_phase * direction))
                if self.group_activity_damping != 0.0:
                    matrix_log = 0.5 * (torch.log(in_rel) + torch.log(out_rel))
                    rational_activity = torch.log(rat_rel) - matrix_log
                    excess = torch.relu((rational_activity - self.group_activity_target) / self.group_activity_width)
                    scale.mul_(torch.exp(-self.group_activity_damping * group_phase * excess))

        balance_scale = self._functional_balance_scale(group, curve_group, balance_phase, device)
        if balance_scale is not None:
            scale.mul_(balance_scale)

        metric_scale = self._functional_metric_group_scale(group, curve_group, device)
        if metric_scale is not None:
            scale.mul_(metric_scale)

        scale = scale.clamp_min(self.eps)
        scale = scale / torch.exp(torch.log(scale).mean()).clamp_min(self.eps)
        scale = scale.clamp(self.group_min_scale, self.group_max_scale)
        return scale.to(device=device, dtype=dtype)

    def _functional_metric_log_scales(
        self,
        curve_group: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        phase = self._functional_metric_phase()
        if phase <= 0.0:
            return None
        groups = int(curve_group.get("groups", 0))
        hidden_dim = int(curve_group.get("hidden_dim", 0))
        if groups <= 0 or hidden_dim <= 0 or hidden_dim % groups != 0:
            return None
        in_weight = curve_group.get("in_weight")
        out_weight = curve_group.get("out_weight")
        if in_weight is None or out_weight is None:
            return None
        if in_weight.shape[0] != hidden_dim or out_weight.shape[1] != hidden_dim:
            return None
        module = curve_group.get("module")
        stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
        if not stats:
            return None
        output_rms = stats.get("output_rms")
        derivative_rms = stats.get("derivative_rms")
        if not (torch.is_tensor(output_rms) and torch.is_tensor(derivative_rms)):
            return None
        if output_rms.numel() != groups or derivative_rms.numel() != groups:
            return None

        width = hidden_dim // groups
        in_view = in_weight.detach().view(groups, width, -1).float()
        out_view = out_weight.detach().view(out_weight.shape[0], groups, width).permute(1, 2, 0).float()
        in_rms = torch.sqrt(in_view.square().mean(dim=(1, 2)) + self.eps).to(device=device).clamp_min(self.eps)
        out_rms = torch.sqrt(out_view.square().mean(dim=(1, 2)) + self.eps).to(device=device).clamp_min(self.eps)
        output = output_rms.detach().float().to(device=device).clamp_min(self.eps)
        derivative = derivative_rms.detach().float().to(device=device).clamp_min(self.eps)

        curve_direction = torch.sqrt(output.square() + derivative.square()).clamp_min(self.eps)
        log_sens_in = torch.log(out_rms) + torch.log(curve_direction)
        log_sens_out = torch.log(in_rms) + torch.log(output)
        joint_center = 0.5 * (log_sens_in.mean() + log_sens_out.mean())
        log_in = -0.5 * phase * (log_sens_in - joint_center)
        log_out = -0.5 * phase * (log_sens_out - joint_center)

        min_log = torch.log(torch.tensor(self.functional_metric_min_scale, device=device, dtype=torch.float32))
        max_log = torch.log(torch.tensor(self.functional_metric_max_scale, device=device, dtype=torch.float32))
        return (
            log_in.clamp(min=min_log, max=max_log),
            log_out.clamp(min=min_log, max=max_log),
            log_sens_in,
            log_sens_out,
        )

    def _functional_metric_role_factor(self, group: dict) -> float:
        if not self._functional_metric_enabled():
            return 1.0
        selector_index = int(group.get("selector_index", -1))
        if selector_index < 0 or selector_index >= len(self.selector_groups):
            return 1.0
        curve_group = self.selector_groups[selector_index]
        param = next((p for p in group.get("params", []) if p is not None), None)
        if param is None:
            return 1.0
        scales = self._functional_metric_log_scales(curve_group, param.device)
        if scales is None:
            return 1.0
        role = str(group.get("matrix_role", "matrix"))
        log_scale = scales[0] if role == "in" else scales[1] if role == "out" else None
        if log_scale is None:
            return 1.0
        return float(torch.exp(log_scale.mean()).clamp(self.functional_metric_min_scale, self.functional_metric_max_scale).item())

    def _functional_metric_group_scale(
        self,
        group: dict,
        curve_group: dict,
        device: torch.device,
    ) -> torch.Tensor | None:
        scales = self._functional_metric_log_scales(curve_group, device)
        if scales is None:
            return None
        role = str(group.get("matrix_role", "matrix"))
        log_scale = scales[0] if role == "in" else scales[1] if role == "out" else None
        if log_scale is None:
            return None
        residual = log_scale - log_scale.mean()
        return torch.exp(residual).clamp(self.functional_metric_min_scale, self.functional_metric_max_scale)

    def _functional_balance_log_imbalance(self, group: dict, curve_group: dict, device: torch.device) -> torch.Tensor | None:
        if not self._functional_balance_enabled():
            return None
        groups = int(curve_group.get("groups", 0))
        hidden_dim = int(curve_group.get("hidden_dim", 0))
        if groups <= 0 or hidden_dim <= 0 or hidden_dim % groups != 0:
            return None
        in_weight = curve_group.get("in_weight")
        out_weight = curve_group.get("out_weight")
        if in_weight is None or out_weight is None or in_weight.grad is None or out_weight.grad is None:
            return None
        if in_weight.shape[0] != hidden_dim or out_weight.shape[1] != hidden_dim:
            return None
        module = curve_group.get("module")
        stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
        if not stats:
            return None
        derivative_rms = stats.get("derivative_rms")
        output_rms = stats.get("output_rms")
        if not (torch.is_tensor(derivative_rms) and torch.is_tensor(output_rms)):
            return None
        if derivative_rms.numel() != groups or output_rms.numel() != groups:
            return None

        width = hidden_dim // groups
        in_grad = in_weight.grad.detach().view(groups, width, -1).float()
        out_grad = out_weight.grad.detach().view(out_weight.shape[0], groups, width).permute(1, 2, 0).float()
        out_view = out_weight.detach().view(out_weight.shape[0], groups, width).permute(1, 2, 0).float()
        in_update = torch.sqrt(in_grad.square().mean(dim=(1, 2)) + self.eps).to(device=device)
        out_update = torch.sqrt(out_grad.square().mean(dim=(1, 2)) + self.eps).to(device=device)
        out_weight_rms = torch.sqrt(out_view.square().mean(dim=(1, 2)) + self.eps).to(device=device)
        derivative = derivative_rms.detach().float().to(device=device).clamp_min(self.eps)
        output = output_rms.detach().float().to(device=device).clamp_min(self.eps)

        m_in = out_weight_rms * derivative * in_update
        m_out = output * out_update
        target_log = self.functional_balance_target_depth_gain * (self._depth(group) - 0.5)
        imbalance = torch.log(m_out.clamp_min(self.eps)) - torch.log(m_in.clamp_min(self.eps)) - target_log
        clip = float(self.functional_balance_clip)
        if clip > 0.0:
            imbalance = imbalance.clamp(min=-clip, max=clip)
        return imbalance

    def _functional_balance_scale(
        self,
        group: dict,
        curve_group: dict,
        phase: float,
        device: torch.device,
    ) -> torch.Tensor | None:
        if phase <= 0.0:
            return None
        imbalance = self._functional_balance_log_imbalance(group, curve_group, device)
        if imbalance is None:
            return None
        role = str(group.get("matrix_role", "matrix"))
        direction = 1.0 if role == "in" else -1.0
        scale = torch.exp(direction * self.functional_balance_strength * phase * imbalance)
        return scale.clamp(self.functional_balance_min_scale, self.functional_balance_max_scale)

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

    def _pressure_activity_terms(self, group: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
        selector_index = int(group.get("selector_index", -1))
        if selector_index < 0 or selector_index >= len(self.selector_groups):
            return None
        state = self.selector_groups[selector_index].get("_onpolicy")
        if state is None:
            return None

        in_rel = state["in_rel_ema"].detach().float().clamp_min(self.eps)
        out_rel = state["out_rel_ema"].detach().float().clamp_min(self.eps)
        rat_rel = state["rat_rel_ema"].detach().float().clamp_min(self.eps)
        log_in = torch.log(in_rel)
        log_out = torch.log(out_rel)
        pressure = (log_in - log_out).abs().mean().clamp(max=self.pressure_clip)
        matrix_log = 0.5 * (log_in + log_out)
        rational_activity = (torch.log(rat_rel) - matrix_log).mean()
        excess_activity = torch.relu((rational_activity - self.activity_target) / self.activity_width)
        return pressure, excess_activity

    def _stat_factor(self, group: dict) -> float:
        terms = self._pressure_activity_terms(group)
        if terms is None:
            return 1.0
        pressure, excess_activity = terms
        penalty = self.pressure_weight * pressure + self.activity_weight * excess_activity
        return float(torch.exp(-penalty).clamp(0.10, 1.15).item())

    def _muon_tail_fraction(self, group: dict) -> float:
        if self.muon_tail_strength <= 0.0:
            return 0.0
        progress = self._progress()
        on_phase = _smoothstep(self.muon_tail_start, self.muon_tail_end, progress)
        off_phase = _smoothstep(self.muon_tail_decay_start, self.muon_tail_decay_end, progress)
        window = on_phase * (1.0 - off_phase)
        if window <= 0.0:
            return 0.0
        terms = self._pressure_activity_terms(group)
        if terms is None:
            confidence = 1.0
        else:
            pressure, excess_activity = terms
            penalty = self.muon_tail_pressure_weight * pressure + self.muon_tail_activity_weight * excess_activity
            confidence = float(torch.exp(-penalty).clamp(0.10, 1.15).item())
        return self.muon_tail_strength * window * self._role_depth_factor(group) * confidence

    @staticmethod
    def _muon_cache_key(group: dict) -> tuple[int, int, str]:
        return (
            int(group.get("layer_index", -1)),
            int(group.get("selector_index", -1)),
            str(group.get("matrix_role", "matrix")),
        )

    def _compute_muon_fraction(self, group: dict) -> float:
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
        value = max(value, self._muon_tail_fraction(group))
        return min(self.max_muon, max(self.min_muon, value))

    def _muon_fraction(self, group: dict) -> float:
        if self._muon_fraction_cache_step != self.step_index:
            self._muon_fraction_cache_step = self.step_index
            self._muon_fraction_cache = {}
        key = self._muon_cache_key(group)
        cached = self._muon_fraction_cache.get(key)
        if cached is not None:
            return cached
        value = self._compute_muon_fraction(group)
        self._muon_fraction_cache[key] = value
        return value

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

    @staticmethod
    def _mean(values):
        if not values:
            return None
        return float(sum(values) / len(values))

    @staticmethod
    def _mean_std_min_max(values):
        if not values:
            return None, None, None, None
        tensor = torch.tensor(values, dtype=torch.float32)
        return (
            float(tensor.mean().item()),
            float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
            float(tensor.min().item()),
            float(tensor.max().item()),
        )

    @staticmethod
    def _append_role(mapping: dict, role: str, value):
        if value is None:
            return
        mapping.setdefault(str(role), []).append(float(value))

    @staticmethod
    def _role_means(mapping: dict) -> dict:
        return {role: RationalMatrixPolicyOptimizer._mean(values) for role, values in sorted(mapping.items())}

    def _policy_telemetry_before_step(self):
        muon_by_role = {}
        adam_lr_by_role = {}
        group_scales = []
        functional_balance = []
        functional_metric_role = {}
        functional_metric_group_scales = []
        functional_metric_sensitivity = {}
        pressures = []
        activities = []
        for group in self.adam.param_groups:
            role = str(group.get("matrix_role", "matrix"))
            fraction = self._muon_fraction(group) if self.muon is not None else 0.0
            self._append_role(muon_by_role, role, fraction)
            self._append_role(adam_lr_by_role, role, self._adam_lr_scale(group))

            param = next((p for p in group.get("params", []) if p is not None), None)
            if param is not None:
                scale = self._group_policy_scale(group, param.device, torch.float32)
                if scale is not None:
                    group_scales.extend(float(x) for x in scale.detach().float().reshape(-1).cpu())
                selector_index_for_balance = int(group.get("selector_index", -1))
                if 0 <= selector_index_for_balance < len(self.selector_groups):
                    imbalance = self._functional_balance_log_imbalance(
                        group,
                        self.selector_groups[selector_index_for_balance],
                        param.device,
                    )
                    if imbalance is not None:
                        functional_balance.extend(float(x) for x in imbalance.detach().float().reshape(-1).cpu())
                    metric = self._functional_metric_log_scales(
                        self.selector_groups[selector_index_for_balance],
                        param.device,
                    )
                    if metric is not None:
                        log_in, log_out, sens_in, sens_out = metric
                        role_log = log_in if role == "in" else log_out if role == "out" else None
                        role_sens = sens_in if role == "in" else sens_out if role == "out" else None
                        if role_log is not None:
                            role_factor = torch.exp(role_log.mean()).clamp(
                                self.functional_metric_min_scale,
                                self.functional_metric_max_scale,
                            )
                            self._append_role(functional_metric_role, role, float(role_factor.item()))
                            residual = torch.exp(role_log - role_log.mean())
                            functional_metric_group_scales.extend(float(x) for x in residual.detach().float().reshape(-1).cpu())
                        if role_sens is not None:
                            functional_metric_sensitivity.setdefault(role, []).extend(
                                float(x) for x in role_sens.detach().float().reshape(-1).cpu()
                            )

            selector_index = int(group.get("selector_index", -1))
            if 0 <= selector_index < len(self.selector_groups):
                state = self.selector_groups[selector_index].get("_onpolicy")
                if state is not None:
                    in_rel = state["in_rel_ema"].detach().float().clamp_min(self.eps)
                    out_rel = state["out_rel_ema"].detach().float().clamp_min(self.eps)
                    rat_rel = state["rat_rel_ema"].detach().float().clamp_min(self.eps)
                    pressure = torch.log(in_rel) - torch.log(out_rel)
                    matrix_log = 0.5 * (torch.log(in_rel) + torch.log(out_rel))
                    activity = torch.log(rat_rel) - matrix_log
                    pressures.extend(float(x) for x in pressure.reshape(-1).cpu())
                    activities.extend(float(x) for x in activity.reshape(-1).cpu())

        group_mean, group_std, group_min, group_max = self._mean_std_min_max(group_scales)
        balance_mean, balance_std, balance_min, balance_max = self._mean_std_min_max(functional_balance)
        metric_scale_mean, metric_scale_std, metric_scale_min, metric_scale_max = self._mean_std_min_max(functional_metric_group_scales)
        pressure_mean, pressure_std, _, _ = self._mean_std_min_max(pressures)
        activity_mean, activity_std, _, _ = self._mean_std_min_max(activities)
        return {
            "matrix_policy_muon_mix_mean_by_role": self._role_means(muon_by_role),
            "matrix_policy_adam_lr_scale_mean_by_role": self._role_means(adam_lr_by_role),
            "matrix_policy_group_scale_mean": group_mean,
            "matrix_policy_group_scale_std": group_std,
            "matrix_policy_group_scale_min": group_min,
            "matrix_policy_group_scale_max": group_max,
            "matrix_policy_functional_balance_log_ratio_mean": balance_mean,
            "matrix_policy_functional_balance_log_ratio_std": balance_std,
            "matrix_policy_functional_balance_log_ratio_min": balance_min,
            "matrix_policy_functional_balance_log_ratio_max": balance_max,
            "matrix_policy_functional_metric_role_scale_mean_by_role": self._role_means(functional_metric_role),
            "matrix_policy_functional_metric_group_scale_mean": metric_scale_mean,
            "matrix_policy_functional_metric_group_scale_std": metric_scale_std,
            "matrix_policy_functional_metric_group_scale_min": metric_scale_min,
            "matrix_policy_functional_metric_group_scale_max": metric_scale_max,
            "matrix_policy_functional_metric_log_sensitivity_mean_by_role": self._role_means(functional_metric_sensitivity),
            "matrix_policy_pressure_mean": pressure_mean,
            "matrix_policy_pressure_std": pressure_std,
            "matrix_policy_activity_mean": activity_mean,
            "matrix_policy_activity_std": activity_std,
        }

    def _capture_pre_step_weights(self):
        snapshots = {}
        for group in self.adam.param_groups:
            for param in group.get("params", []):
                if param is None:
                    continue
                ident = id(param)
                if ident not in snapshots:
                    snapshots[ident] = param.detach().clone()
        return snapshots

    def _update_telemetry_after_step(self, telemetry: dict, snapshots: dict):
        update_by_role = {}
        weight_by_role = {}
        ratio_by_role = {}
        for group in self.adam.param_groups:
            role = str(group.get("matrix_role", "matrix"))
            for param in group.get("params", []):
                before = snapshots.get(id(param))
                if before is None:
                    continue
                delta = param.detach() - before.to(device=param.device, dtype=param.dtype)
                update_rms = torch.sqrt(delta.float().square().mean() + self.eps)
                weight_rms = torch.sqrt(param.detach().float().square().mean() + self.eps)
                ratio = update_rms / weight_rms.clamp_min(self.eps)
                self._append_role(update_by_role, role, float(update_rms.item()))
                self._append_role(weight_by_role, role, float(weight_rms.item()))
                self._append_role(ratio_by_role, role, float(ratio.item()))
        telemetry.update(
            {
                "matrix_policy_update_rms_by_role": self._role_means(update_by_role),
                "matrix_policy_weight_rms_by_role": self._role_means(weight_by_role),
                "matrix_policy_update_to_weight_rms_by_role": self._role_means(ratio_by_role),
            }
        )
        self._last_telemetry = telemetry

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("RationalMatrixPolicyOptimizer does not support closures")
        self.step_index += 1
        capture_telemetry = self._capture_telemetry_next_step
        self._capture_telemetry_next_step = False
        telemetry = self._policy_telemetry_before_step() if capture_telemetry else {}
        snapshots = self._capture_pre_step_weights() if capture_telemetry else {}

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
                group["lr"] = lr * self.muon_lr_scale * fraction * self._functional_metric_role_factor(group)

        self.adam.step()
        if self.muon is not None:
            self.muon.step()

        for group, lr, betas in zip(self.adam.param_groups, saved_adam_lrs, saved_adam_betas):
            group["lr"] = lr
            group["betas"] = betas
        if self.muon is not None:
            for group, lr in zip(self.muon.param_groups, saved_muon_lrs):
                group["lr"] = lr
        if capture_telemetry:
            self._update_telemetry_after_step(telemetry, snapshots)
        return None
