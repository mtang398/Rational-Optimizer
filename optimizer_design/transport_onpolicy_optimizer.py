"""Curve-transport on-policy optimizer for fused RLB rational FFNs.

RLB exposes two exact positive scale gauges:

    W_in,g <- c W_in,g, W_out,g <- W_out,g / c
    R_g    <- a R_g,    W_out,g <- W_out,g / a

The existing on-policy optimizers use the first gauge. This wrapper adds the
second, rational-only gauge: it rescales each learned rational curve by scaling
its numerator and local atom coefficients, then compensates the output matrix so
the represented layer function is unchanged.
"""

from __future__ import annotations

import torch

from .adaptive_metric_onpolicy_optimizer import RationalAdaptiveMetricOnPolicyOptimizer
from .onpolicy_balance_optimizer import _smoothstep


class RationalTransportOnPolicyOptimizer(RationalAdaptiveMetricOnPolicyOptimizer):
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
        coeff_logits = group["coeff_logits"]
        if coeff_logits.numel() == 0:
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

        coeff_logits = group["coeff_logits"]
        if coeff_logits.numel() == 0:
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
        self._update_onpolicy_stats()
        self._project_gauge_gradients()
        self._precondition_coefficient_gradients()
        self._precondition_matrix_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        self._balance()
        self._transport_curve_amplitude()
