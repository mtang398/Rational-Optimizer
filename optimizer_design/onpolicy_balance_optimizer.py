"""On-policy gauge optimizer for fused RLB rational FFNs.

RLB has an exact per-group homogeneity:

    W_in[g]  <- c W_in[g]
    W_out[g] <- W_out[g] / c

for c > 0. This leaves the represented layer function unchanged, but changes
the conditioning seen by the optimizer. This wrapper uses that rational-only
gauge freedom after each optimizer step. Unlike the previous static balance
rules, the target is partly on-policy: it uses the live gradient pressure on
the current batch for each layer and hidden group.

The method is not defined for SiLU/SwiGLU. It requires a single-branch RLB FFN
with trainable rational coefficients and grouped homogeneous hidden channels.
"""

from __future__ import annotations

import math

import torch


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


class RationalOnPolicyBalanceOptimizer:
    """Composite optimizer with live RLB gauge balancing.

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
        coeff_logits = group["coeff_logits"]
        centers = group["centers"]
        beta = group["beta"]
        coeff_limit = float(group["coeff_limit"])
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

        if coeff_logits.numel() > 0:
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
        width = hidden_dim // groups
        in_view = in_weight.view(groups, width, -1)
        out_view = out_weight.view(out_weight.shape[0], groups, width).permute(1, 2, 0)
        return in_view, out_view

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
                param = group[key]
                if param.grad is None:
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
    def _balance(self):
        if self.strength <= 0.0 or not self.balance_groups:
            return
        if self.step_index % self.every != 0:
            return
        schedule = self.strength * _smoothstep(self.start, self.end, self._progress())
        if schedule <= 0.0:
            return
        log_activity_min = math.log(max(self.activity_gain_min, self.eps))
        log_activity_max = math.log(max(self.activity_gain_max, self.eps))
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
                group["_functional_out_gain"] = out_gain
                group["_functional_deriv_gain"] = deriv_gain
                group["_functional_metric_step"] = self.step_index
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
            log_scale = 0.5 * (target_log_ratio - current_log_ratio)
            log_scale = log_scale * (schedule * self._layer_scale(group)) * activity_gain
            log_scale = log_scale.clamp(min=-self.max_log_step, max=self.max_log_step)
            scale = torch.exp(log_scale)
            in_view.mul_(scale.view(scale.numel(), 1, 1))
            out_inv_scale = scale.reciprocal()
            out_view.mul_(out_inv_scale.view(scale.numel(), 1, 1))
            self._scale_optimizer_state(group["in_weight"], scale, in_like=True)
            self._scale_optimizer_state(group["out_weight"], out_inv_scale, in_like=False)

    def step(self):
        self.step_index += 1
        self._update_onpolicy_stats()
        for optimizer in self.optimizers:
            optimizer.step()
        self._balance()
