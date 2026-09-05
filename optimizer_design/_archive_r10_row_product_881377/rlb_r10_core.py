"""RLB-conditioned attention optimization in row product coordinates.

The complete R03 router and complete R02 attention transaction remain the
scientific parent.  This module changes only the coordinates supplied to the
attention child: exact row magnitudes use Adam while the existing learned-RLB
response transaction consumes the tangent direction gradient.
"""

from __future__ import annotations

import torch

from .rlb_r02_core import R02AttentionCore


class R10AttentionCore(R02AttentionCore):
    """Complete-R02 attention in implicit row-magnitude/direction coordinates."""

    component_code = 47
    checkpoint_schema = "r10_r03_rlb_conditioned_attention_row_product_v1"
    inherited_parent = "complete_r03_with_complete_r02_attention"
    new_scientific_components = (
        "rlb_conditioned_attention_row_magnitude_direction_product_geometry",
    )

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "attention_row_product_lr_scale": 1.0,
            "attention_direction_lr_scale": 1.0,
            "attention_magnitude_lr_scale": 1.0,
            "attention_effective_weight_decay_scale": 1.0,
        })
        return report

    @staticmethod
    def _row_norm(value):
        return torch.linalg.vector_norm(value, dim=1, keepdim=True)

    @staticmethod
    def _init_row_product_state(parameter, state):
        magnitude = R10AttentionCore._row_norm(parameter.detach()).float()
        torch._assert_async(torch.isfinite(magnitude).all())
        torch._assert_async((magnitude > 0.0).all())
        state["r10_magnitude"] = magnitude.clone()
        state["r10_direction_norm"] = magnitude.clone()
        state["r10_magnitude_first_moment"] = torch.zeros_like(magnitude)
        state["r10_magnitude_second_moment"] = torch.zeros_like(magnitude)
        state["r10_magnitude_step"] = 0

    @classmethod
    def _pullback(cls, parameter, gradient, state):
        if "r10_magnitude" not in state:
            cls._init_row_product_state(parameter, state)
        magnitude = state["r10_magnitude"]
        direction_norm = state["r10_direction_norm"]
        weight = parameter.detach().float()
        gradient_fp32 = gradient.detach().float()
        if magnitude.shape != (weight.shape[0], 1):
            raise RuntimeError("R10 attention magnitude inventory changed")
        if direction_norm.shape != magnitude.shape:
            raise RuntimeError("R10 attention direction-norm inventory changed")
        torch._assert_async(torch.isfinite(weight).all())
        torch._assert_async(torch.isfinite(gradient_fp32).all())
        torch._assert_async(torch.isfinite(magnitude).all())
        torch._assert_async(torch.isfinite(direction_norm).all())
        torch._assert_async((magnitude > 0.0).all())
        torch._assert_async((direction_norm > 0.0).all())

        unit = weight / magnitude
        direction = unit * direction_norm
        magnitude_gradient = (gradient_fp32 * unit).sum(dim=1, keepdim=True)
        direction_gradient = (magnitude / direction_norm) * (
            gradient_fp32 - unit * magnitude_gradient
        )
        torch._assert_async(torch.isfinite(direction).all())
        torch._assert_async(torch.isfinite(magnitude_gradient).all())
        torch._assert_async(torch.isfinite(direction_gradient).all())
        tangent_inner = (direction_gradient * direction).sum(dim=1)
        tangent_scale = (
            torch.linalg.vector_norm(direction_gradient, dim=1)
            * torch.linalg.vector_norm(direction, dim=1)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        tangent_relative = tangent_inner.abs() / tangent_scale
        reconstruction = magnitude * (
            direction / cls._row_norm(direction).clamp_min(
                torch.finfo(torch.float32).tiny
            )
        )
        reconstruction_relative = (
            torch.linalg.vector_norm(reconstruction - weight)
            / torch.linalg.vector_norm(weight).clamp_min(1.0)
        )
        return {
            "old_weight": weight.clone(),
            "original_gradient": gradient,
            "direction": direction,
            "direction_gradient": direction_gradient,
            "magnitude_gradient": magnitude_gradient,
            "tangent_relative_max": tangent_relative.amax(),
            "reconstruction_relative": reconstruction_relative,
            "old_magnitude": magnitude.clone(),
        }

    @staticmethod
    def _adam_magnitude(state, gradient, *, lr, beta1, beta2, eps):
        step = state["r10_magnitude_step"] + 1
        if type(step) is not int or step <= 0:
            raise RuntimeError("R10 attention magnitude step changed")
        state["r10_magnitude_step"] = step
        first = state["r10_magnitude_first_moment"]
        second = state["r10_magnitude_second_moment"]
        first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        corrected_first = first / (1.0 - beta1**step)
        corrected_second = second / (1.0 - beta2**step)
        denominator = corrected_second.sqrt().add_(eps)
        state["r10_magnitude"].addcdiv_(
            corrected_first, denominator, value=-lr
        )

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("R10 refuses a nonunit attention LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if weight_decay != 0.10:
            raise RuntimeError("R10 requires the matched effective-weight WD")
        beta1 = 0.90
        beta2 = self.beta2
        eps = self.adaptive_eps
        if beta2 != 0.95 or eps != 1.0e-8:
            raise RuntimeError("R10 magnitude Adam contract changed")

        records = {}
        parameters = [
            parameter
            for role in self._ROLES
            for parameter in self.role_parameters[role]
        ]
        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError("R10 attention matrix gradient is missing")
            state = self.state[parameter]
            record = self._pullback(parameter, parameter.grad, state)
            records[id(parameter)] = record
            parameter.copy_(record["direction"].to(parameter.dtype))
            parameter.grad = record["direction_gradient"].to(parameter.dtype)

        # The complete R02 direction transaction must not decay the latent
        # direction.  Decoupled WD is applied once to the reconstructed
        # effective weight below.  Restore the public group value even if the
        # parent raises; a failure aborts rather than silently changing method.
        group["weight_decay"] = 0.0
        try:
            loss = super().step(closure)
        finally:
            group["weight_decay"] = weight_decay

        tangent_values = []
        reconstruction_values = []
        magnitude_drift_values = []
        steps = []
        for parameter in parameters:
            record = records[id(parameter)]
            state = self.state[parameter]
            self._adam_magnitude(
                state,
                record["magnitude_gradient"],
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )
            direction_new = parameter.detach().float()
            direction_norm_new = self._row_norm(direction_new)
            torch._assert_async(torch.isfinite(direction_norm_new).all())
            torch._assert_async((direction_norm_new > 0.0).all())
            magnitude = state["r10_magnitude"]
            effective = magnitude * (direction_new / direction_norm_new)
            effective.add_(record["old_weight"], alpha=-lr * weight_decay)
            torch._assert_async(torch.isfinite(effective).all())
            effective_norm = self._row_norm(effective)
            torch._assert_async((effective_norm > 0.0).all())
            parameter.copy_(effective.to(parameter.dtype))
            state["r10_magnitude"].copy_(effective_norm)
            state["r10_direction_norm"] = direction_norm_new.clone()
            parameter.grad = record["original_gradient"]

            tangent_values.append(record["tangent_relative_max"])
            reconstruction_values.append(record["reconstruction_relative"])
            magnitude_drift_values.append(
                ((effective_norm - record["old_magnitude"]).abs()
                 / record["old_magnitude"].clamp_min(
                     torch.finfo(torch.float32).tiny
                 )).amax()
            )
            steps.append(state["r10_magnitude_step"])

        if len(set(steps)) != 1:
            raise RuntimeError("R10 attention magnitude steps diverged")
        if publish:
            inherited = dict(self._last_telemetry)
            inherited.update({
                "rlb_r10_component_code": self.component_code,
                "rlb_r10_parent_is_complete_r03_attention": 1,
                "rlb_r10_rlb_conditioned_attention_row_product_enabled": 1,
                "rlb_r10_attention_row_count": sum(
                    parameter.shape[0] for parameter in parameters
                ),
                "rlb_r10_attention_matrix_count": len(parameters),
                "rlb_r10_magnitude_step": steps[0],
                "rlb_r10_tangent_relative_residual_max": float(
                    torch.stack(tangent_values).amax().item()
                ),
                "rlb_r10_reconstruction_relative_residual_max": float(
                    torch.stack(reconstruction_values).amax().item()
                ),
                "rlb_r10_row_magnitude_relative_change_max": float(
                    torch.stack(magnitude_drift_values).amax().item()
                ),
                "rlb_r10_effective_weight_decay": weight_decay,
                "rlb_r10_internal_lr_wd_scalar_max": 1.0,
            })
            self._last_telemetry = inherited
        return loss


__all__ = ("R10AttentionCore",)
