"""Internal MatrixPolicy support wrapper for fused RLB rational FFNs.

This module intentionally exposes only ``RationalTransportOnPolicyOptimizer``.
The smaller on-policy balance, matrix-metric, and adaptive-stat pieces are kept
as private base classes because MatrixPolicy still relies on their mechanics:
live RLB pressure statistics, optional matrix/coeff preconditioning, exact
W_in/W_out gauge rebalance, and rational-curve amplitude transport.
"""

from __future__ import annotations

import math

import torch

def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


class _RLBGaugeBalanceBase:
    """Internal base with live RLB gauge balancing.

    Ordinary Transformer parameters are updated by the child optimizers. This
    wrapper then applies a function-preserving group gauge transform. The target
    gauge combines:

    * current rational curve output/derivative gains on a probe grid;
    * EMA of actual input-factor versus output-factor gradient pressure;
    * EMA of rational coefficient gradient activity, used only to modulate how
      strongly the gauge correction is applied.

    The optimizer is therefore rational-specific, layer-specific, and
    time-varying, while preserving the model function under the gauge step.
    """

    def __init__(
        self,
        optimizers,
        balance_groups,
        total_steps: int,
        target_weight: float = 1.0,
        metric_every: int = 10,
        probe_range: float = 5.0,
        probe_points: int = 129,
        strength: float = 0.50,
        max_log_step: float = 0.030,
        start: float = 0.03,
        end: float = 0.35,
        depth_gain: float = 0.15,
        every: int = 5,
        stat_decay: float = 0.95,
        pressure_weight: float = 0.25,
        pressure_clip: float = 1.25,
        rational_activity_weight: float = 0.10,
        activity_gain_min: float = 0.75,
        activity_gain_max: float = 1.25,
        covariant_state: bool = False,
        eps: float = 1e-8,
    ):
        if not 0.0 <= stat_decay < 1.0:
            raise ValueError("stat_decay must be in [0, 1)")
        if pressure_clip <= 0.0:
            raise ValueError("pressure_clip must be positive")
        self.optimizers = list(optimizers)
        self.balance_groups = list(balance_groups)
        self.total_steps = int(total_steps)
        self.target_weight = float(target_weight)
        self.metric_every = max(1, int(metric_every))
        self.probe_range = float(probe_range)
        self.probe_points = max(9, int(probe_points))
        self.strength = float(strength)
        self.max_log_step = float(max_log_step)
        self.start = float(start)
        self.end = float(end)
        self.depth_gain = float(depth_gain)
        self.every = max(1, int(every))
        self.stat_decay = float(stat_decay)
        self.pressure_weight = float(pressure_weight)
        self.pressure_clip = float(pressure_clip)
        self.rational_activity_weight = float(rational_activity_weight)
        self.activity_gain_min = float(activity_gain_min)
        self.activity_gain_max = float(activity_gain_max)
        self.covariant_state = bool(covariant_state)
        self.eps = float(eps)
        self.step_index = 0
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}
        for group in self.balance_groups:
            self._pair_probe_input(group)

    def set_telemetry_capture(self, enabled: bool = True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        stats = []
        for group in self.balance_groups:
            state = group.get("_onpolicy")
            if state is None:
                stats.append(None)
            else:
                stats.append({key: value.detach().clone() for key, value in state.items()})
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "step_index": self.step_index,
            "onpolicy_stats": stats,
        }

    def load_state_dict(self, state_dict):
        for optimizer, optimizer_state in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        self.step_index = int(state_dict.get("step_index", 0))
        stats = state_dict.get("onpolicy_stats")
        if stats is not None:
            for group, state in zip(self.balance_groups, stats):
                if state is not None:
                    group["_onpolicy"] = {key: value.detach().clone() for key, value in state.items()}

    def _progress(self) -> float:
        if self.total_steps <= 1:
            return 0.0
        return min(1.0, max(0.0, float(self.step_index) / float(self.total_steps)))

    @staticmethod
    def _depth(group: dict) -> float:
        layer = int(group.get("layer_index", -1))
        layers = max(1, int(group.get("num_layers", 1)))
        if layer < 0 or layers <= 1:
            return 0.5
        return min(1.0, max(0.0, float(layer) / float(layers - 1)))

    def _layer_scale(self, group: dict) -> float:
        return min(1.25, max(0.75, 1.0 + self.depth_gain * (self._depth(group) - 0.5)))

    @torch.no_grad()
    def _curve_gain(self, group: dict):
        numerator = group["numerator"]
        denominator = group["denominator"]
        coeff_logits = group.get("coeff_logits")
        centers = group.get("centers")
        beta = group.get("beta")
        coeff_limit = float(group.get("coeff_limit", 0.0))
        dtype = numerator.dtype
        device = numerator.device
        t = torch.linspace(-self.probe_range, self.probe_range, self.probe_points, device=device, dtype=dtype)
        ax = t.abs()
        t2 = t * t
        t3 = t2 * t
        t4 = t2 * t2
        t5 = t4 * t
        ax3 = ax * t2

        a = numerator.to(dtype=dtype)
        b = denominator.abs().to(dtype=dtype)
        p = (
            a[:, 0:1]
            + a[:, 1:2] * t
            + a[:, 2:3] * t2
            + a[:, 3:4] * t3
            + a[:, 4:5] * t4
            + a[:, 5:6] * t5
        )
        dp = (
            a[:, 1:2]
            + 2.0 * a[:, 2:3] * t
            + 3.0 * a[:, 3:4] * t2
            + 4.0 * a[:, 4:5] * t3
            + 5.0 * a[:, 5:6] * t4
        )
        q = 1.0 + b[:, 0:1] * ax + b[:, 1:2] * t2 + b[:, 2:3] * ax3 + b[:, 3:4] * t4
        dq = b[:, 0:1] * torch.sign(t) + 2.0 * b[:, 1:2] * t + 3.0 * b[:, 2:3] * t * ax + 4.0 * b[:, 3:4] * t3
        f = p / q
        df = (dp * q - p * dq) / (q * q)

        if coeff_logits is not None and coeff_logits.numel() > 0 and centers is not None and beta is not None:
            coeff = coeff_limit * torch.tanh(coeff_logits).to(dtype=dtype)
            center = centers.to(device=device, dtype=dtype).unsqueeze(-1)
            beta_v = beta.to(device=device, dtype=dtype).unsqueeze(-1)
            u = t.view(1, 1, -1) - center
            u2 = u * u
            den = 1.0 + beta_v * u2
            inv_den = den.reciprocal()
            inv_den2 = inv_den * inv_den
            zero = (1.0 + beta_v * center * center).reciprocal()
            odd = u * inv_den
            bump = inv_den - zero
            odd_dt = (1.0 - beta_v * u2) * inv_den2
            bump_dt = -2.0 * beta_v * u * inv_den2
            c_odd = coeff[..., 0].unsqueeze(-1)
            c_bump = coeff[..., 1].unsqueeze(-1)
            f = f + (c_odd * odd + c_bump * bump).sum(dim=1)
            df = df + (c_odd * odd_dt + c_bump * bump_dt).sum(dim=1)

        out_gain = torch.sqrt(f.square().mean(dim=-1) + self.eps)
        deriv_gain = torch.sqrt(df.square().mean(dim=-1) + self.eps)
        return out_gain.detach(), deriv_gain.detach()

    @torch.no_grad()
    def _group_weight_views(self, group: dict):
        in_weight = group["in_weight"]
        out_weight = group["out_weight"]
        groups = int(group["groups"])
        hidden_dim = int(group["hidden_dim"])
        if hidden_dim % groups != 0:
            return None
        cache_key = (
            id(in_weight),
            id(out_weight),
            tuple(in_weight.shape),
            tuple(out_weight.shape),
            groups,
            hidden_dim,
        )
        cache = group.get("_weight_view_cache")
        if cache is not None and cache.get("key") == cache_key:
            return cache["views"]
        width = hidden_dim // groups
        in_view = in_weight.view(groups, width, -1)
        out_view = out_weight.view(out_weight.shape[0], groups, width).permute(1, 2, 0)
        views = (in_view, out_view)
        group["_weight_view_cache"] = {"key": cache_key, "views": views}
        return views

    @torch.no_grad()
    def _update_onpolicy_stats(self):
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
            in_norm = torch.sqrt(in_view.square().mean(dim=(1, 2)) + self.eps)
            out_norm = torch.sqrt(out_view.square().mean(dim=(1, 2)) + self.eps)
            in_grad_norm = torch.sqrt(in_grad.square().mean(dim=(1, 2)) + self.eps)
            out_grad_norm = torch.sqrt(out_grad.square().mean(dim=(1, 2)) + self.eps)
            in_rel = in_grad_norm / in_norm.clamp_min(self.eps)
            out_rel = out_grad_norm / out_norm.clamp_min(self.eps)

            rational_terms = []
            for key in ("numerator", "denominator", "coeff_logits"):
                param = group.get(key)
                if param is None or param.numel() == 0 or param.grad is None:
                    continue
                grad = param.grad
                rational_terms.append(grad.reshape(groups, -1).square().mean(dim=1))
            if rational_terms:
                rat_rel = torch.sqrt(torch.stack(rational_terms, dim=0).mean(dim=0) + self.eps)
            else:
                rat_rel = torch.sqrt(0.5 * (in_rel.square() + out_rel.square()) + self.eps)

            state = group.get("_onpolicy")
            if state is None:
                group["_onpolicy"] = {
                    "in_rel_ema": in_rel.detach().clone(),
                    "out_rel_ema": out_rel.detach().clone(),
                    "rat_rel_ema": rat_rel.detach().clone(),
                }
                continue
            decay = self.stat_decay
            state["in_rel_ema"].mul_(decay).add_(in_rel.detach(), alpha=1.0 - decay)
            state["out_rel_ema"].mul_(decay).add_(out_rel.detach(), alpha=1.0 - decay)
            state["rat_rel_ema"].mul_(decay).add_(rat_rel.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def _scale_optimizer_state(self, param: torch.Tensor, scale: torch.Tensor, in_like: bool):
        if not self.covariant_state:
            return
        for optimizer in self.optimizers:
            state = getattr(optimizer, "state", {}).get(param)
            if not state:
                continue
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                key_text = str(key)
                factor = scale.square() if "sq" in key_text or "square" in key_text else scale
                factor = factor.to(device=value.device, dtype=value.dtype)
                if value.shape == param.shape:
                    if in_like:
                        view = value.view(scale.numel(), -1, value.shape[-1])
                        view.mul_(factor.view(scale.numel(), 1, 1))
                    else:
                        groups = scale.numel()
                        width = value.shape[1] // groups
                        view = value.view(value.shape[0], groups, width).permute(1, 2, 0)
                        view.mul_(factor.view(groups, 1, 1))
                    continue

                if key_text == "exp_avg_sq_row" and value.dim() == 1:
                    if in_like and value.numel() == param.shape[0]:
                        value.view(scale.numel(), -1).mul_(factor.view(scale.numel(), 1))
                    elif (not in_like) and value.numel() == param.shape[0]:
                        value.mul_(factor.mean())
                    continue
                if key_text == "exp_avg_sq_col" and value.dim() == 1:
                    if in_like and value.numel() == param.shape[1]:
                        value.mul_(factor.mean())
                    elif (not in_like) and value.numel() == param.shape[1]:
                        value.view(scale.numel(), -1).mul_(factor.view(scale.numel(), 1))
                    continue

    @torch.no_grad()
    def _pair_probe_input(self, group: dict):
        in_weight = group["in_weight"]
        input_dim = int(in_weight.shape[1])
        cache_key = (input_dim, in_weight.device, in_weight.dtype)
        cache = group.get("_pair_probe_cache")
        if cache is None or cache.get("key") != cache_key:
            index = torch.arange(2 * input_dim, device=in_weight.device, dtype=torch.float32).view(2, input_dim)
            probe = torch.sin(index * 0.017) + 0.5 * torch.cos(index * 0.031)
            probe = probe / torch.sqrt(probe.square().mean(dim=-1, keepdim=True) + self.eps)
            cache = {"key": cache_key, "input": probe.to(dtype=in_weight.dtype)}
            group["_pair_probe_cache"] = cache
        return cache["input"]

    @torch.no_grad()
    def _pair_probe_output(self, group: dict):
        module = group.get("module")
        if module is None:
            return None
        in_weight = group["in_weight"]
        out_weight = group["out_weight"]
        probe_input = self._pair_probe_input(group)

        had_track = hasattr(module, "_rlb_optimizer_track_stats")
        old_track = getattr(module, "_rlb_optimizer_track_stats", None)
        had_stats = hasattr(module, "_rlb_optimizer_stats")
        old_stats = getattr(module, "_rlb_optimizer_stats", None)
        try:
            if had_track:
                setattr(module, "_rlb_optimizer_track_stats", False)
            hidden = torch.nn.functional.linear(probe_input, in_weight)
            activated = module(hidden)
            return torch.nn.functional.linear(activated, out_weight).detach().float()
        finally:
            if had_track:
                setattr(module, "_rlb_optimizer_track_stats", old_track)
            if had_stats:
                setattr(module, "_rlb_optimizer_stats", old_stats)

    @torch.no_grad()
    def _balance(self, capture_telemetry: bool = False):
        telemetry = {
            "matrix_policy_pair_rescale_enabled": self.strength > 0.0,
            "matrix_policy_pair_rescale_applied": False,
        }
        if not self.balance_groups:
            return telemetry
        if self.step_index % self.every != 0:
            return telemetry
        schedule = self.strength * _smoothstep(self.start, self.end, self._progress())
        if schedule <= 0.0 and not capture_telemetry:
            return telemetry
        log_activity_min = math.log(max(self.activity_gain_min, self.eps))
        log_activity_max = math.log(max(self.activity_gain_max, self.eps))
        log_moves = []
        target_mismatches = []
        clipped = 0
        move_count = 0
        probe_relative_deltas = []
        pair_rescale_scheduled = schedule > 0.0
        for group in self.balance_groups:
            views = self._group_weight_views(group)
            if views is None:
                continue
            in_view, out_view = views
            in_norm = torch.sqrt(in_view.square().mean(dim=(1, 2)) + self.eps)
            out_norm = torch.sqrt(out_view.square().mean(dim=(1, 2)) + self.eps)

            metric_step = int(group.get("_functional_metric_step", -1))
            if metric_step < 0 or (self.step_index - metric_step) >= self.metric_every:
                out_gain, deriv_gain = self._curve_gain(group)
                if pair_rescale_scheduled:
                    group["_functional_out_gain"] = out_gain
                    group["_functional_deriv_gain"] = deriv_gain
                    group["_functional_metric_step"] = self.step_index
            else:
                out_gain = group.get("_functional_out_gain")
                deriv_gain = group.get("_functional_deriv_gain")
            target_log_ratio = torch.zeros_like(in_norm)
            if out_gain is not None and deriv_gain is not None and self.target_weight != 0.0:
                out_gain = out_gain.to(device=in_norm.device, dtype=in_norm.dtype).clamp_min(self.eps)
                deriv_gain = deriv_gain.to(device=in_norm.device, dtype=in_norm.dtype).clamp_min(self.eps)
                target_log_ratio = 0.5 * (torch.log(deriv_gain) - torch.log(out_gain))
                target_log_ratio = target_log_ratio * self.target_weight

            state = group.get("_onpolicy")
            activity_gain = torch.ones_like(in_norm)
            if state is not None:
                in_rel = state["in_rel_ema"].to(device=in_norm.device, dtype=in_norm.dtype).clamp_min(self.eps)
                out_rel = state["out_rel_ema"].to(device=in_norm.device, dtype=in_norm.dtype).clamp_min(self.eps)
                rat_rel = state["rat_rel_ema"].to(device=in_norm.device, dtype=in_norm.dtype).clamp_min(self.eps)
                pressure = torch.log(in_rel) - torch.log(out_rel)
                pressure = pressure.clamp(min=-self.pressure_clip, max=self.pressure_clip)
                target_log_ratio = target_log_ratio + self.pressure_weight * pressure

                matrix_rel = torch.sqrt(in_rel * out_rel).clamp_min(self.eps)
                rational_activity = torch.log(rat_rel) - torch.log(matrix_rel)
                rational_activity = rational_activity.clamp(min=-self.pressure_clip, max=self.pressure_clip)
                log_activity_gain = self.rational_activity_weight * rational_activity
                log_activity_gain = log_activity_gain.clamp(min=log_activity_min, max=log_activity_max)
                activity_gain = torch.exp(log_activity_gain)

            current_log_ratio = torch.log(in_norm) - torch.log(out_norm)
            target_mismatch = target_log_ratio - current_log_ratio
            raw_log_scale = 0.5 * target_mismatch
            raw_log_scale = raw_log_scale * (schedule * self._layer_scale(group)) * activity_gain
            log_scale = raw_log_scale.clamp(min=-self.max_log_step, max=self.max_log_step)
            probe_before = self._pair_probe_output(group) if capture_telemetry else None
            if pair_rescale_scheduled:
                scale = torch.exp(log_scale)
                in_view.mul_(scale.view(scale.numel(), 1, 1))
                out_inv_scale = scale.reciprocal()
                out_view.mul_(out_inv_scale.view(scale.numel(), 1, 1))
            if capture_telemetry:
                probe_after = self._pair_probe_output(group)
                if probe_before is not None and probe_after is not None:
                    delta = torch.sqrt((probe_after - probe_before).square().mean())
                    baseline = torch.sqrt(probe_before.square().mean() + self.eps)
                    probe_relative_deltas.append(float((delta / baseline.clamp_min(self.eps)).item()))
                log_moves.extend(float(x) for x in log_scale.detach().float().reshape(-1).cpu())
                target_mismatches.extend(
                    float(x) for x in target_mismatch.detach().float().abs().reshape(-1).cpu()
                )
                clipped += int((raw_log_scale.detach().abs() > self.max_log_step).sum().item())
                move_count += int(raw_log_scale.numel())
            if pair_rescale_scheduled:
                self._scale_optimizer_state(group["in_weight"], scale, in_like=True)
                self._scale_optimizer_state(group["out_weight"], out_inv_scale, in_like=False)
        if capture_telemetry:
            def mean_or_none(values):
                return None if not values else float(sum(values) / len(values))

            telemetry.update(
                {
                    "matrix_policy_pair_rescale_scheduled": pair_rescale_scheduled,
                    "matrix_policy_pair_rescale_diagnosed_group_count": move_count,
                    "matrix_policy_pair_rescale_attempted_count": move_count if pair_rescale_scheduled else 0,
                    "matrix_policy_pair_rescale_applied": any(abs(value) > 0.0 for value in log_moves),
                    "matrix_policy_pair_log_move_mean": mean_or_none(log_moves),
                    "matrix_policy_pair_log_move_abs_mean": mean_or_none([abs(x) for x in log_moves]),
                    "matrix_policy_pair_log_move_abs_max": None if not log_moves else max(abs(x) for x in log_moves),
                    "matrix_policy_pair_target_mismatch_abs_mean": mean_or_none(target_mismatches),
                    "matrix_policy_pair_clip_fraction": None if move_count == 0 else float(clipped) / float(move_count),
                    "matrix_policy_pair_local_probe_relative_delta_mean": mean_or_none(probe_relative_deltas),
                    "matrix_policy_pair_local_probe_relative_delta_max": (
                        None if not probe_relative_deltas else max(probe_relative_deltas)
                    ),
                }
            )
        return telemetry

    def step(self):
        self.step_index += 1
        capture_telemetry = self._capture_telemetry_next_step
        self._capture_telemetry_next_step = False
        self._update_onpolicy_stats()
        for optimizer in self.optimizers:
            optimizer.step()
        pair_telemetry = self._balance(capture_telemetry=capture_telemetry)
        if capture_telemetry:
            self._last_telemetry = pair_telemetry


class _RLBMatrixMetricBase(_RLBGaugeBalanceBase):
    """Internal base for rational derivative/output matrix preconditioning."""

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


class _RLBAdaptiveMetricBase(_RLBMatrixMetricBase):
    """Internal base using live empirical rational-function metrics.

    The wrapper performs, in order:

    1. update live gradient-pressure EMAs;
    2. optionally precondition rational coefficient gradients with per-group
       empirical Gram matrices gathered from the current activation distribution;
    3. precondition matrix gradients with on-policy rational output/derivative
       gains;
    4. step child optimizers;
    5. apply the existing function-preserving RLB gauge balance.

    It is intentionally not defined for SiLU/SwiGLU or GLU rational variants.
    """

    def __init__(
        self,
        *args,
        stat_every: int = 4,
        stat_samples: int = 512,
        coeff_strength: float = 0.0,
        coeff_start: float = 0.02,
        coeff_end: float = 0.65,
        coeff_late_decay: float = 0.35,
        coeff_metric_damping: float = 0.03,
        coeff_norm_clip: float = 3.0,
        coeff_max_blend: float = 0.85,
        coeff_depth_gain: float = 0.20,
        matrix_time_gain: float = 0.15,
        matrix_depth_gain: float = 0.10,
        quotient_strength: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stat_every = max(1, int(stat_every))
        self.stat_samples = max(1, int(stat_samples))
        self.coeff_strength = float(coeff_strength)
        self.coeff_start = float(coeff_start)
        self.coeff_end = float(coeff_end)
        self.coeff_late_decay = float(coeff_late_decay)
        self.coeff_metric_damping = float(coeff_metric_damping)
        self.coeff_norm_clip = max(1.0, float(coeff_norm_clip))
        self.coeff_max_blend = min(1.0, max(0.0, float(coeff_max_blend)))
        self.coeff_depth_gain = float(coeff_depth_gain)
        self.matrix_time_gain = float(matrix_time_gain)
        self.matrix_depth_gain = float(matrix_depth_gain)
        self.quotient_strength = float(quotient_strength)
        if self.coeff_metric_damping < 0.0:
            raise ValueError("coeff_metric_damping must be non-negative")

        child_uses_live_gains = any(
            float(getattr(optimizer, "group_gain_strength", 0.0)) != 0.0
            for optimizer in self.optimizers
        )
        self._live_optimizer_stats_consumed = bool(
            child_uses_live_gains
            or self.coeff_strength > 0.0
            or self.matrix_strength > 0.0
        )
        for group in self.balance_groups:
            module = group.get("module")
            if module is None:
                continue
            setattr(module, "_rlb_optimizer_track_stats", True)
            setattr(module, "_rlb_optimizer_stat_every", self.stat_every)
            setattr(module, "_rlb_optimizer_stat_samples", self.stat_samples)
            setattr(
                module,
                "_rlb_optimizer_stats_training_only",
                self._live_optimizer_stats_consumed,
            )
            setattr(
                module,
                "_rlb_optimizer_sync_stats",
                self._live_optimizer_stats_consumed,
            )
            setattr(module, "_rlb_optimizer_stats_synced_version", -1)

    def _enable_live_optimizer_stats(self):
        if self._live_optimizer_stats_consumed:
            return
        self._live_optimizer_stats_consumed = True
        for group in self.balance_groups:
            module = group.get("module")
            if module is None:
                continue
            setattr(module, "_rlb_optimizer_stats_training_only", True)
            setattr(module, "_rlb_optimizer_sync_stats", True)
            setattr(module, "_rlb_optimizer_stats_synced_version", -1)

    @torch.no_grad()
    def _synchronize_live_optimizer_stats(self):
        if (
            self.coeff_strength > 0.0
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise RuntimeError(
                "distributed empirical coefficient Gram preconditioning requires "
                "globally synchronized Gram statistics"
            )
        if not self._live_optimizer_stats_consumed or not self.balance_groups:
            return

        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        world_size = torch.distributed.get_world_size() if distributed else 1
        entries = []
        for group in self.balance_groups:
            module = group.get("module")
            groups = int(group.get("groups", 0))
            if module is None or groups <= 0:
                raise RuntimeError("RLB live-stat synchronization found an invalid balance group")
            sync_enabled = bool(getattr(module, "_rlb_optimizer_sync_stats", False))
            stats = getattr(module, "_rlb_optimizer_stats", None)
            version = int(getattr(module, "_rlb_optimizer_stat_version", 0))
            synced_version = int(getattr(module, "_rlb_optimizer_stats_synced_version", -1))
            pending = bool(sync_enabled and version != synced_version)
            required = ("output_sq_sum", "derivative_sq_sum", "sample_count")
            valid = bool(
                pending
                and isinstance(stats, dict)
                and all(torch.is_tensor(stats.get(key)) for key in required)
                and int(stats.get("stat_version", -1)) == version
            )
            tensors = []
            if valid:
                tensors = [stats[key].detach().reshape(-1) for key in required]
                valid = all(tensor.numel() == groups for tensor in tensors)
            entries.append(
                {
                    "module": module,
                    "stats": stats,
                    "version": version,
                    "synced_version": synced_version,
                    "sync_enabled": sync_enabled,
                    "pending": pending,
                    "valid": valid,
                    "groups": groups,
                    "layer_token": int(group.get("layer_index", -1)) + 1,
                    "tensors": tensors,
                }
            )

        tensor_devices = {
            tensor.device
            for entry in entries
            for tensor in entry["tensors"]
        }
        if len(tensor_devices) > 1:
            raise RuntimeError("RLB live-stat caches must share one device")
        if tensor_devices:
            device = next(iter(tensor_devices))
        else:
            device = self.balance_groups[0]["in_weight"].device

        chunks = []
        for entry in entries:
            groups = int(entry["groups"])
            pending = bool(entry["pending"])
            valid = bool(entry["valid"])
            version = int(entry["version"])
            layer_token = int(entry["layer_token"])
            header = torch.tensor(
                [
                    float(entry["sync_enabled"]),
                    float(groups),
                    float(groups * groups),
                    float(layer_token),
                    float(layer_token * layer_token),
                    float(pending),
                    float(version if pending else 0),
                    float(version * version if pending else 0),
                    float(valid),
                ],
                device=device,
                dtype=torch.float64,
            )
            chunks.append(header)
            if valid:
                chunks.extend(
                    tensor.to(device=device, dtype=torch.float64)
                    for tensor in entry["tensors"]
                )
            else:
                chunks.extend(
                    torch.zeros(groups, device=device, dtype=torch.float64)
                    for _ in range(3)
                )
        payload = torch.cat(chunks)
        if distributed:
            torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)

        offset = 0
        for entry in entries:
            groups = int(entry["groups"])
            header = payload[offset : offset + 9]
            offset += 9
            output_sum = payload[offset : offset + groups]
            offset += groups
            derivative_sum = payload[offset : offset + groups]
            offset += groups
            sample_count = payload[offset : offset + groups]
            offset += groups

            (
                sync_sum,
                group_sum,
                group_sq_sum,
                layer_sum,
                layer_sq_sum,
                pending_sum,
                version_sum,
                version_sq_sum,
                valid_sum,
            ) = (
                float(value.item()) for value in header
            )
            if sync_sum not in (0.0, float(world_size)):
                raise RuntimeError("RLB live-stat synchronization flags differ across ranks")
            if world_size * group_sq_sum != group_sum * group_sum:
                raise RuntimeError("RLB live-stat group descriptors differ across ranks")
            if int(round(group_sum / world_size)) != groups:
                raise RuntimeError("RLB live-stat group order differs across ranks")
            if world_size * layer_sq_sum != layer_sum * layer_sum:
                raise RuntimeError("RLB live-stat layer order differs across ranks")
            if int(round(layer_sum / world_size)) != int(entry["layer_token"]):
                raise RuntimeError("RLB live-stat layer descriptor differs across ranks")
            if sync_sum == 0.0:
                continue
            if pending_sum not in (0.0, float(world_size)):
                raise RuntimeError("RLB live-stat cache versions differ across ranks")
            if pending_sum == 0.0:
                continue
            if world_size * version_sq_sum != version_sum * version_sum:
                raise RuntimeError("RLB live-stat refresh versions differ across ranks")
            if valid_sum != float(world_size):
                raise RuntimeError("RLB live-stat cache is incomplete on at least one rank")
            if torch.any(sample_count <= 0.0):
                raise RuntimeError("RLB live-stat synchronization found a non-positive sample count")

            stats = entry["stats"]
            module = entry["module"]
            if not isinstance(stats, dict):
                raise RuntimeError("RLB live-stat cache disappeared during synchronization")
            module_eps = float(getattr(module, "eps", self.eps))
            stats["output_sq_sum"] = output_sum.float().detach()
            stats["derivative_sq_sum"] = derivative_sum.float().detach()
            stats["sample_count"] = sample_count.float().detach()
            stats["output_rms"] = torch.sqrt(output_sum / sample_count + module_eps).float().detach()
            stats["derivative_rms"] = torch.sqrt(
                derivative_sum / sample_count + module_eps
            ).float().detach()
            stats["sync_world_size"] = world_size
            stats["optimizer_global"] = True
            stats["optimizer_global_version"] = int(entry["version"])
            setattr(
                module,
                "_rlb_optimizer_stats_synced_version",
                int(entry["version"]),
            )

    def _coefficient_phase(self) -> float:
        progress = self._progress()
        grow = _smoothstep(self.coeff_start, self.coeff_end, progress)
        late = _smoothstep(0.70, 0.95, progress)
        return grow * (1.0 - self.coeff_late_decay * late)

    def _matrix_phase(self) -> float:
        progress = self._progress()
        late = _smoothstep(0.25, 0.90, progress)
        return 1.0 + self.matrix_time_gain * late

    def _coefficient_blend(self, group: dict, role_scale: float) -> float:
        depth = self._depth(group)
        layer = min(1.35, max(0.65, 1.0 + self.coeff_depth_gain * (0.5 - depth)))
        blend = self.coeff_strength * self._coefficient_phase() * layer * float(role_scale)
        return min(self.coeff_max_blend, max(0.0, blend))

    def _matrix_strength_for_group(self, group: dict) -> float:
        depth = self._depth(group)
        layer = min(1.35, max(0.65, 1.0 + self.matrix_depth_gain * (depth - 0.5)))
        return max(0.0, self.matrix_strength * self._matrix_phase() * layer)

    @torch.no_grad()
    def _project_gauge_gradients(self):
        quotient = self.quotient_strength * _smoothstep(0.02, 0.30, self._progress())
        if quotient <= 0.0:
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
            coeff = quotient * numerator / denominator.clamp_min(self.eps)
            in_grad.sub_(coeff.view(groups, 1, 1) * in_view)
            out_grad.add_(coeff.view(groups, 1, 1) * out_view)

    def _metric_solve(self, gram: torch.Tensor, grad: torch.Tensor, damping: float) -> torch.Tensor:
        if grad.numel() == 0:
            return grad
        gram_f = gram.detach().float()
        grad_f = grad.detach().float()
        if gram_f.dim() != 3 or grad_f.dim() != 2 or gram_f.size(0) != grad_f.size(0):
            return grad
        width = grad_f.size(-1)
        if gram_f.size(1) != width or gram_f.size(2) != width:
            return grad

        gram_f = 0.5 * (gram_f + gram_f.transpose(-1, -2))
        diag_mean = gram_f.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(self.eps)
        eye = torch.eye(width, device=gram_f.device, dtype=gram_f.dtype).expand_as(gram_f)
        matrix = gram_f + (float(damping) * diag_mean + self.eps).view(-1, 1, 1) * eye
        try:
            solved = torch.linalg.solve(matrix, grad_f.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            return grad

        finite = torch.isfinite(solved).all(dim=-1, keepdim=True)
        solved = torch.where(finite, solved, grad_f)
        grad_norm = torch.sqrt(grad_f.square().mean(dim=-1, keepdim=True) + self.eps)
        solved_norm = torch.sqrt(solved.square().mean(dim=-1, keepdim=True) + self.eps)
        rel = (grad_norm / solved_norm).clamp(1.0 / self.coeff_norm_clip, self.coeff_norm_clip)
        return (solved * rel).to(device=grad.device, dtype=grad.dtype)

    def _apply_metric_gradient(
        self,
        param: torch.Tensor,
        gram: torch.Tensor | None,
        blend: float,
        role_damping: float = 1.0,
    ):
        if blend <= 0.0 or gram is None or param.grad is None:
            return
        grad = param.grad
        original_shape = grad.shape
        if grad.dim() < 2:
            return
        flat_grad = grad.reshape(grad.shape[0], -1)
        preconditioned = self._metric_solve(
            gram.to(device=grad.device),
            flat_grad,
            self.coeff_metric_damping * float(role_damping),
        ).reshape(original_shape)
        grad.mul_(1.0 - blend).add_(preconditioned, alpha=blend)

    @torch.no_grad()
    def _precondition_coefficient_gradients(self):
        if self.coeff_strength <= 0.0:
            return
        for group in self.balance_groups:
            module = group.get("module")
            stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
            if not stats:
                continue
            num_blend = self._coefficient_blend(group, 1.00)
            den_blend = self._coefficient_blend(group, 0.90)
            atom_blend = self._coefficient_blend(group, 1.10)
            self._apply_metric_gradient(group["numerator"], stats.get("num_gram"), num_blend)
            self._apply_metric_gradient(group["denominator"], stats.get("den_gram"), den_blend, role_damping=1.5)
            coeff_logits = group.get("coeff_logits")
            if coeff_logits is not None and coeff_logits.numel() > 0:
                self._apply_metric_gradient(coeff_logits, stats.get("atom_gram"), atom_blend, role_damping=0.75)

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

            strength = self._matrix_strength_for_group(group)
            if strength <= 0.0:
                continue
            groups = int(group["groups"])
            hidden_dim = int(group["hidden_dim"])
            width = hidden_dim // groups
            state = group.setdefault("_adaptive_metric_precond", {})
            refresh = self.step_index % self.matrix_every == 1
            if refresh or "in_scale" not in state or "out_scale" not in state:
                module = group.get("module")
                stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
                if stats and "output_rms" in stats and "derivative_rms" in stats:
                    out_gain = stats["output_rms"]
                    deriv_gain = stats["derivative_rms"]
                else:
                    out_gain, deriv_gain = self._curve_gain(group)
                in_scale = self._centered_inverse_scale(
                    deriv_gain, strength, self.eps
                ).clamp(self.matrix_min_scale, self.matrix_max_scale)
                out_scale = self._centered_inverse_scale(
                    out_gain, strength, self.eps
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
        self._synchronize_live_optimizer_stats()
        self._project_gauge_gradients()
        self._precondition_coefficient_gradients()
        self._precondition_matrix_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        self._balance()


class RationalTransportOnPolicyOptimizer(_RLBAdaptiveMetricBase):
    """RLB optimizer with exact rational-curve amplitude transport.

    The step order is intentionally aggressive but still function-aware:

    1. collect live gradient pressure;
    2. optionally project matrix gradients away from the group gauge;
    3. optionally apply empirical coefficient metrics;
    4. precondition matrix gradients by rational output/derivative gains;
    5. additionally equalize live group update pressure;
    6. step child optimizers;
    7. apply matrix gauge balance;
    8. apply rational-curve amplitude transport and compensate ``W_out``.
    """

    def __init__(
        self,
        *args,
        transport_strength: float = 0.0,
        transport_final_strength: float | None = None,
        transport_start: float = 0.04,
        transport_end: float = 0.70,
        transport_decay_start: float = 1.1,
        transport_decay_end: float = 1.1,
        transport_every: int = 5,
        transport_max_log_step: float = 0.025,
        transport_derivative_weight: float = 0.50,
        transport_headroom: float = 0.92,
        transport_depth_gain: float = 0.30,
        transport_derivative_depth_gain: float = 0.35,
        matrix_input_depth_gain: float = 0.0,
        matrix_output_depth_gain: float = 0.0,
        matrix_live_stats: bool = False,
        pressure_precond_strength: float = 0.0,
        pressure_precond_depth_gain: float = 0.25,
        pressure_precond_min_scale: float = 0.70,
        pressure_precond_max_scale: float = 1.40,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transport_strength = float(transport_strength)
        self.transport_final_strength = (
            None if transport_final_strength is None else float(transport_final_strength)
        )
        self.transport_start = float(transport_start)
        self.transport_end = float(transport_end)
        self.transport_decay_start = float(transport_decay_start)
        self.transport_decay_end = float(transport_decay_end)
        self.transport_every = max(1, int(transport_every))
        self.transport_max_log_step = float(transport_max_log_step)
        self.transport_derivative_weight = min(1.0, max(0.0, float(transport_derivative_weight)))
        self.transport_headroom = min(0.999, max(0.10, float(transport_headroom)))
        self.transport_depth_gain = float(transport_depth_gain)
        self.transport_derivative_depth_gain = float(transport_derivative_depth_gain)
        self.matrix_input_depth_gain = float(matrix_input_depth_gain)
        self.matrix_output_depth_gain = float(matrix_output_depth_gain)
        self.matrix_live_stats = bool(matrix_live_stats)
        self.pressure_precond_strength = float(pressure_precond_strength)
        self.pressure_precond_depth_gain = float(pressure_precond_depth_gain)
        self.pressure_precond_min_scale = float(pressure_precond_min_scale)
        self.pressure_precond_max_scale = float(pressure_precond_max_scale)
        if self.transport_max_log_step < 0.0:
            raise ValueError("transport_max_log_step must be non-negative")
        if self.pressure_precond_min_scale <= 0.0:
            raise ValueError("pressure_precond_min_scale must be positive")
        if self.pressure_precond_max_scale < self.pressure_precond_min_scale:
            raise ValueError("pressure_precond_max_scale must be >= pressure_precond_min_scale")
        if self.transport_strength > 0.0 or (
            self.matrix_strength > 0.0 and self.matrix_live_stats
        ):
            self._enable_live_optimizer_stats()

    def _transport_phase(self) -> float:
        progress = self._progress()
        phase = _smoothstep(self.transport_start, self.transport_end, progress)
        strength = self.transport_strength
        if self.transport_final_strength is not None:
            late = _smoothstep(self.transport_decay_start, self.transport_decay_end, progress)
            strength = self.transport_strength * (1.0 - late) + self.transport_final_strength * late
        return max(0.0, strength * phase)

    def _depth_factor(self, group: dict, gain: float, late_positive: bool) -> float:
        depth = self._depth(group)
        direction = depth - 0.5 if late_positive else 0.5 - depth
        return min(1.50, max(0.50, 1.0 + float(gain) * direction))

    def _derivative_weight_for_group(self, group: dict) -> float:
        depth = self._depth(group)
        value = self.transport_derivative_weight + self.transport_derivative_depth_gain * (0.5 - depth)
        return min(0.95, max(0.05, value))

    @torch.no_grad()
    def _precondition_matrix_gradients_by_pressure(self):
        strength = self.pressure_precond_strength * _smoothstep(0.05, 0.60, self._progress())
        if strength <= 0.0:
            return
        for group in self.balance_groups:
            group_strength = strength * self._depth_factor(group, self.pressure_precond_depth_gain, late_positive=False)
            if group_strength <= 0.0:
                continue
            state = group.get("_onpolicy")
            if state is None:
                continue
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
            in_rel = state["in_rel_ema"].to(device=in_weight.device, dtype=in_weight.dtype)
            out_rel = state["out_rel_ema"].to(device=out_weight.device, dtype=out_weight.dtype)
            in_scale = self._centered_inverse_scale(in_rel, group_strength, self.eps).clamp(
                self.pressure_precond_min_scale,
                self.pressure_precond_max_scale,
            )
            out_scale = self._centered_inverse_scale(out_rel, group_strength, self.eps).clamp(
                self.pressure_precond_min_scale,
                self.pressure_precond_max_scale,
            )
            in_grad = in_weight.grad.view(groups, width, -1)
            out_grad = out_weight.grad.view(out_weight.shape[0], groups, width).permute(1, 2, 0)
            in_grad.mul_(in_scale.view(groups, 1, 1))
            out_grad.mul_(out_scale.view(groups, 1, 1))

    @torch.no_grad()
    def _precondition_matrix_gradients(self):
        if self.matrix_strength > 0.0:
            for group in self.balance_groups:
                views = self._group_weight_views(group)
                if views is None:
                    continue
                in_weight = group["in_weight"]
                out_weight = group["out_weight"]
                if in_weight.grad is None or out_weight.grad is None:
                    continue

                base_strength = self._matrix_strength_for_group(group)
                if base_strength <= 0.0:
                    continue
                depth = self._depth(group)
                in_factor = min(1.70, max(0.40, 1.0 + self.matrix_input_depth_gain * (0.5 - depth)))
                out_factor = min(1.70, max(0.40, 1.0 + self.matrix_output_depth_gain * (depth - 0.5)))
                in_strength = max(0.0, base_strength * in_factor)
                out_strength = max(0.0, base_strength * out_factor)
                if in_strength <= 0.0 and out_strength <= 0.0:
                    continue

                groups = int(group["groups"])
                hidden_dim = int(group["hidden_dim"])
                width = hidden_dim // groups
                state = group.setdefault("_adaptive_metric_precond", {})
                refresh = self.step_index % self.matrix_every == 1
                if refresh or "in_scale" not in state or "out_scale" not in state:
                    out_gain, deriv_gain = self._matrix_gains(group, in_weight.device, in_weight.dtype)
                    in_scale = self._centered_inverse_scale(
                        deriv_gain, in_strength, self.eps
                    ).clamp(self.matrix_min_scale, self.matrix_max_scale)
                    out_scale = self._centered_inverse_scale(
                        out_gain, out_strength, self.eps
                    ).clamp(self.matrix_min_scale, self.matrix_max_scale)
                    state["in_scale"] = in_scale.to(device=in_weight.device, dtype=in_weight.dtype)
                    state["out_scale"] = out_scale.to(device=out_weight.device, dtype=out_weight.dtype)

                in_grad = in_weight.grad.view(groups, width, -1)
                out_grad = out_weight.grad.view(out_weight.shape[0], groups, width).permute(1, 2, 0)
                in_grad.mul_(state["in_scale"].view(groups, 1, 1))
                out_grad.mul_(state["out_scale"].view(groups, 1, 1))
        self._precondition_matrix_gradients_by_pressure()


    @torch.no_grad()
    def _matrix_gains(self, group: dict, device: torch.device, dtype: torch.dtype):
        if self.matrix_live_stats:
            return self._transport_gains(group, device, dtype)
        out_gain, deriv_gain = self._curve_gain(group)
        return (
            out_gain.to(device=device, dtype=dtype).clamp_min(self.eps),
            deriv_gain.to(device=device, dtype=dtype).clamp_min(self.eps),
        )

    @torch.no_grad()
    def _transport_gains(self, group: dict, device: torch.device, dtype: torch.dtype):
        module = group.get("module")
        stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
        if stats and "output_rms" in stats and "derivative_rms" in stats:
            out_gain = stats["output_rms"]
            deriv_gain = stats["derivative_rms"]
        else:
            out_gain, deriv_gain = self._curve_gain(group)
        return (
            out_gain.to(device=device, dtype=dtype).clamp_min(self.eps),
            deriv_gain.to(device=device, dtype=dtype).clamp_min(self.eps),
        )

    @torch.no_grad()
    def _clip_transport_for_atom_headroom(self, group: dict, log_scale: torch.Tensor) -> torch.Tensor:
        coeff_logits = group.get("coeff_logits")
        if coeff_logits is None or coeff_logits.numel() == 0:
            return log_scale
        groups = int(group["groups"])
        limit = float(group["coeff_limit"])
        coeff = limit * torch.tanh(coeff_logits.detach().float())
        max_abs = coeff.reshape(groups, -1).abs().amax(dim=1).to(device=log_scale.device, dtype=log_scale.dtype)
        max_coeff = max(self.eps, self.transport_headroom * limit)
        max_scale = torch.where(
            max_abs > self.eps,
            torch.full_like(max_abs, max_coeff) / max_abs.clamp_min(self.eps),
            torch.full_like(max_abs, 1.0 / self.eps),
        )
        max_log_scale = torch.log(max_scale.clamp_min(self.eps))
        return torch.minimum(log_scale, max_log_scale)

    @torch.no_grad()
    def _scale_rational_curve(self, group: dict, scale: torch.Tensor):
        groups = int(group["groups"])
        numerator = group["numerator"]
        numerator.view(groups, -1).mul_(scale.to(device=numerator.device, dtype=numerator.dtype).view(groups, 1))

        coeff_logits = group.get("coeff_logits")
        if coeff_logits is None or coeff_logits.numel() == 0:
            return
        limit = float(group["coeff_limit"])
        coeff = limit * torch.tanh(coeff_logits.detach().float())
        scaled = coeff * scale.to(device=coeff_logits.device, dtype=torch.float32).view(groups, 1, 1)
        clipped = (scaled / max(limit, self.eps)).clamp(-0.999999, 0.999999)
        coeff_logits.copy_(torch.atanh(clipped).to(dtype=coeff_logits.dtype))

    @torch.no_grad()
    def _transport_curve_amplitude(self):
        if self.transport_strength <= 0.0 or not self.balance_groups:
            return
        if self.step_index % self.transport_every != 0:
            return
        phase = self._transport_phase()
        if phase <= 0.0:
            return

        for group in self.balance_groups:
            views = self._group_weight_views(group)
            if views is None:
                continue
            _, out_view = views
            out_weight = group["out_weight"]
            groups = int(group["groups"])

            out_gain, deriv_gain = self._transport_gains(group, out_weight.device, out_weight.dtype)
            derivative_weight = self._derivative_weight_for_group(group)
            log_curve = (
                (1.0 - derivative_weight) * torch.log(out_gain)
                + derivative_weight * torch.log(deriv_gain)
            )
            target = log_curve.mean()
            layer_phase = phase * self._layer_scale(group) * self._depth_factor(
                group, self.transport_depth_gain, late_positive=True
            )
            log_scale = (target - log_curve) * layer_phase
            log_scale = log_scale.clamp(min=-self.transport_max_log_step, max=self.transport_max_log_step)
            log_scale = self._clip_transport_for_atom_headroom(group, log_scale)
            scale = torch.exp(log_scale)
            if torch.allclose(scale, torch.ones_like(scale), rtol=0.0, atol=1e-6):
                continue

            self._scale_rational_curve(group, scale)
            out_inv_scale = scale.reciprocal()
            out_view.mul_(out_inv_scale.view(groups, 1, 1))
            self._scale_optimizer_state(out_weight, out_inv_scale, in_like=False)
            group.pop("_adaptive_metric_precond", None)
            group.pop("_jacobian_precond", None)
            for key in ("_functional_out_gain", "_functional_deriv_gain"):
                cached = group.get(key)
                if torch.is_tensor(cached) and cached.numel() == scale.numel():
                    group[key] = cached.to(device=scale.device, dtype=scale.dtype) * scale.detach()

    def step(self):
        self.step_index += 1
        capture_telemetry = self._capture_telemetry_next_step
        self._capture_telemetry_next_step = False
        self._update_onpolicy_stats()
        self._synchronize_live_optimizer_stats()
        self._project_gauge_gradients()
        self._precondition_coefficient_gradients()
        self._precondition_matrix_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        pair_telemetry = self._balance(capture_telemetry=capture_telemetry)
        self._transport_curve_amplitude()
        if capture_telemetry:
            self._last_telemetry = pair_telemetry
