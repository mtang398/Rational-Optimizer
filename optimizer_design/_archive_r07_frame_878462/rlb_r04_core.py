"""Opaque R04 block optimizer driven by learned Global-RLB morphology.

The RLB matrix pair executes the successful R05 generation-one B+C path.
The exact current-versus-frozen RLB response congruence is then reused as a
parameter-free block sensor for the two attention matrices.  It selects the
input to the unchanged NS5 polar map between ordinary Muon momentum and a
matched-beta2 factorized adaptive momentum.  External LR, WD, Nesterov,
Newton--Schulz steps, and Muon shape calibration remain unchanged.
"""

from __future__ import annotations

import math

import torch

from .rlb_group_muon_core import _batched_zero_power, _match_rms_adamw_scale
from .rlb_r05_core import R05Core
from .rlb_r08_core import R08RevisionCore
from .rlb_response_capture_core import RLBResponseCaptureCore


class R04RLBRouterCore(R05Core):
    """R05 generation-one parent plus a current-step block sensor."""

    def __init__(
        self,
        pairs,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta2: float,
        eps: float,
    ):
        self._r04_last_alignments = None
        super().__init__(
            pairs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            eps=eps,
            use_response_router=True,
            use_functional_selector=False,
        )

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        for name in (
            "endpoint_geometry_lr_scale",
            "functional_selector_lr_scale",
            "aligned_functional_tangent_lr_scale",
        ):
            report.pop(name, None)
        report.update({
            "r05_generation_one_parent_lr_scale": 1.0,
            "block_response_sensor_lr_scale": 1.0,
        })
        return report

    # The generation-two aligned functional selector is absent from R04.
    # These direct parent hooks retain exactly the B+C and response-statistic
    # inventories without recording dead x/z/h packets.
    def _make_input_hook(self, index):
        return R08RevisionCore._make_input_hook(self, index)

    def _make_feature_hook(self, index):
        return RLBResponseCaptureCore._make_feature_hook(self, index)

    def _consume_functional_samples(self):
        if any(value is not None for value in self._functional_pending_inputs):
            raise RuntimeError("R04 unexpectedly captured a functional input")
        if any(records for records in self._functional_records):
            raise RuntimeError("R04 unexpectedly captured functional samples")
        return None, None, None

    def _consume_router_alignments(self):
        alignments = super()._consume_router_alignments()
        self._r04_last_alignments = alignments
        return alignments

    @staticmethod
    def _descent_safe_endpoint(parent, adaptive, momentum, alignment):
        """Remove R05 generation two's dead endpoint computation.

        Generation one selects the already formed equal-budget parent.  The
        endpoint was never consumed when the functional selector was deleted,
        so returning the literal parent is trajectory preserving.
        """
        del adaptive, alignment
        descent = (momentum * parent).sum(dim=(-2, -1), keepdim=True)
        zeros = torch.zeros_like(descent)
        ones = torch.ones_like(descent)
        # R05's caller scales the parent and endpoint variables separately.
        # Keep an independent dead endpoint buffer so the literal parent is
        # scaled exactly once.
        return parent.clone(), {
            "half_angle": zeros.flatten(),
            "response_cap": zeros.flatten(),
            "branch_cap": zeros.flatten(),
            "descent_cap": zeros.flatten(),
            "gamma": ones.flatten(),
            "budget_residual": zeros.flatten(),
            "descent_margin": zeros.flatten(),
            "parent_descent": descent.flatten(),
            "endpoint_descent": descent.flatten(),
        }

    def _select_functional_corner(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        incoming_parent,
        incoming_endpoint,
        outgoing_parent_transpose,
        outgoing_endpoint_transpose,
        incoming_parent_descent,
        incoming_endpoint_descent,
        outgoing_parent_descent,
        outgoing_endpoint_descent,
        lr,
        *,
        force_parent,
    ):
        del (
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_endpoint,
            outgoing_endpoint_transpose,
            incoming_parent_descent,
            incoming_endpoint_descent,
            outgoing_parent_descent,
            outgoing_endpoint_descent,
            lr,
            force_parent,
        )
        layers = incoming_parent.shape[0]
        device = incoming_parent.device
        dtype = incoming_parent.dtype
        choices = torch.zeros(layers, device=device, dtype=torch.int64)
        zeros = torch.zeros(layers, device=device, dtype=dtype)
        return incoming_parent, outgoing_parent_transpose, {
            "choices": choices,
            "scores": torch.zeros((layers, 4), device=device, dtype=dtype),
            "score_margin": zeros,
            "energies": torch.zeros((layers, 4), device=device, dtype=dtype),
            "global_count": torch.zeros((), device=device, dtype=dtype),
        }

    def current_attention_alignments(self):
        if self._r04_last_alignments is None:
            raise RuntimeError("R04 attention requested a stale RLB sensor")
        if self._r04_last_alignments.shape != (len(self.pairs), 2):
            raise RuntimeError("R04 RLB sensor inventory changed")
        return self._r04_last_alignments

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r04_last_alignments = None
        loss = super().step(closure)
        if self._r04_last_alignments is None:
            raise RuntimeError("R04 did not publish its current-step RLB sensor")
        if publish:
            public = {}
            for key, value in self._last_telemetry.items():
                if not key.startswith("rlb_r05_"):
                    continue
                if "selector" in key or "functional" in key or "endpoint" in key:
                    continue
                public[key.replace("rlb_r05_", "rlb_r04_", 1)] = value
            public["rlb_r04_structural_matrix_elements"] = 245_366_784
            self._last_telemetry = public
        return loss


class R04AttentionCore(torch.optim.Optimizer):
    """RLB-morphology-selected factorized adaptive Muon for attention."""

    _ROLES = ("qkv", "attn_out")

    def __init__(
        self,
        blocks,
        router: R04RLBRouterCore,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta2: float,
        eps: float,
        adjust_lr_fn: str,
    ):
        if float(lr) != 3.0e-4 or float(weight_decay) != 0.10:
            raise ValueError("R04 requires the matched LR/WD contract")
        if float(momentum) != 0.95 or int(ns_steps) != 5:
            raise ValueError("R04 requires the matched Muon recurrence")
        if float(beta2) != 0.95 or float(eps) != 1.0e-8:
            raise ValueError("R04 requires matched beta2/epsilon")
        if adjust_lr_fn != "match_rms_adamw":
            raise ValueError("R04 requires matched Muon shape calibration")
        self.blocks = sorted(
            (dict(block) for block in blocks), key=lambda item: item["layer_index"]
        )
        if not self.blocks or len(self.blocks) != len(router.pairs):
            raise ValueError("R04 requires one attention pair per RLB router layer")
        if [int(block["layer_index"]) for block in self.blocks] != list(
            range(len(self.blocks))
        ):
            raise ValueError("R04 block inventory must be contiguous")
        self.router = router
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        self.beta2 = float(beta2)
        self.adaptive_eps = float(eps)
        self.adjust_lr_fn = adjust_lr_fn
        self.role_parameters = {
            "qkv": [block["qkv_weight"] for block in self.blocks],
            "attn_out": [block["attn_out_weight"] for block in self.blocks],
        }
        external = int(router.external_width)
        expected_shapes = {
            "qkv": (3 * external, external),
            "attn_out": (external, external),
        }
        parameters = []
        seen = {id(parameter) for parameter in router.incoming + router.outgoing}
        for role in self._ROLES:
            for parameter in self.role_parameters[role]:
                if tuple(parameter.shape) != expected_shapes[role]:
                    raise ValueError(f"R04 {role} matrix shape changed")
                if parameter.ndim != 2 or id(parameter) in seen:
                    raise ValueError("R04 structural matrix ownership changed")
                seen.add(id(parameter))
                parameters.append(parameter)
        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_scale": 1.0,
        }
        super().__init__([{"params": parameters}], defaults)
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "factorized_second_moment_lr_scale": 1.0,
            "equal_source_budget_lr_scale": 1.0,
            "rlb_morphology_router_lr_scale": 1.0,
            "polar_lr_scale": 1.0,
            "phase_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def set_telemetry_capture(self, enabled: bool = True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def _nesterov(self, parameter):
        if parameter.grad is None:
            raise RuntimeError("R04 attention matrix gradient is missing")
        state = self.state[parameter]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(parameter)
            state["momentum_buffer"] = buffer
        buffer.lerp_(parameter.grad, 1.0 - self.momentum)
        return parameter.grad.lerp(buffer, self.momentum)

    def _factorized_adaptive_source(self, role, gradients, momenta, step):
        anchor_state = self.state[self.role_parameters[role][0]]
        row_key = f"r04_{role}_row_second_moment"
        column_key = f"r04_{role}_column_second_moment"
        rows = anchor_state.get(row_key)
        columns = anchor_state.get(column_key)
        row_shape = gradients.shape[:-1]
        column_shape = (gradients.shape[0], gradients.shape[-1])
        if rows is None:
            rows = torch.zeros(row_shape, device=gradients.device, dtype=torch.float32)
            columns = torch.zeros(
                column_shape, device=gradients.device, dtype=torch.float32
            )
            anchor_state[row_key] = rows
            anchor_state[column_key] = columns
        if rows.shape != row_shape or columns is None or columns.shape != column_shape:
            raise RuntimeError("R04 factorized second-moment inventory changed")
        squared = gradients.square()
        rows.mul_(self.beta2).add_(
            squared.sum(dim=-1), alpha=1.0 - self.beta2
        )
        columns.mul_(self.beta2).add_(
            squared.sum(dim=-2), alpha=1.0 - self.beta2
        )
        correction = 1.0 - self.beta2 ** int(step)
        row_total = rows.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(rows.dtype).tiny
        )
        variance = rows[:, :, None] * columns[:, None, :] / row_total[:, :, None]
        variance.div_(correction)
        inverse_root = torch.reciprocal(variance.sqrt() + self.adaptive_eps)
        adaptive = momenta * inverse_root
        momentum_norm = torch.linalg.vector_norm(
            momenta, dim=(-2, -1), keepdim=True
        )
        adaptive_norm = torch.linalg.vector_norm(
            adaptive, dim=(-2, -1), keepdim=True
        )
        tiny = torch.finfo(momenta.dtype).tiny
        adaptive_equal = adaptive * (momentum_norm / adaptive_norm.clamp_min(tiny))
        condition = inverse_root.amax(dim=(-2, -1)) / inverse_root.amin(
            dim=(-2, -1)
        ).clamp_min(tiny)
        return adaptive_equal, condition

    @staticmethod
    def _route_source(momentum, adaptive_equal, alignment):
        if momentum.shape != adaptive_equal.shape:
            raise RuntimeError("R04 attention source shapes differ")
        if alignment.shape != (momentum.shape[0],):
            raise RuntimeError("R04 attention sensor shape changed")
        c = alignment[:, None, None]
        valid = torch.isfinite(c) & (c >= 0.0) & (c <= 1.0)
        torch._assert_async(valid.all())
        source = (
            torch.sqrt(c) * momentum
            + torch.sqrt((1.0 - c).clamp_min(0.0)) * adaptive_equal
        )
        return torch.where(c == 1.0, momentum, source)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("R04 refuses a nonunit attention LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        alignments = self.router.current_attention_alignments()
        incoming_alignment = alignments[:, 0]
        router_step = int(self.router._r05_step)
        anchor_state = self.state[self.role_parameters["qkv"][0]]
        previous_step = anchor_state.get("r04_attention_step", 0)
        if type(previous_step) is not int or router_step != previous_step + 1:
            raise RuntimeError("R04 attention did not consume exactly one current RLB sensor")
        anchor_state["r04_attention_step"] = router_step

        role_records = {}
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            gradients = torch.stack([parameter.grad for parameter in parameters]).float()
            momenta = torch.stack([self._nesterov(parameter) for parameter in parameters]).float()
            adaptive_equal, factor_condition = self._factorized_adaptive_source(
                role, gradients, momenta, router_step
            )
            source = self._route_source(momenta, adaptive_equal, incoming_alignment)
            direction = _batched_zero_power(source, self.ns_steps).float()
            direction.mul_(
                _match_rms_adamw_scale(momenta.shape[-2], momenta.shape[-1])
            )
            descent = (momenta * direction).sum(dim=(-2, -1))
            source_norm = torch.linalg.vector_norm(source, dim=(-2, -1))
            momentum_norm = torch.linalg.vector_norm(momenta, dim=(-2, -1))
            adaptive_norm = torch.linalg.vector_norm(adaptive_equal, dim=(-2, -1))
            budget_residual = (
                (adaptive_norm - momentum_norm).abs()
                / momentum_norm.clamp_min(1.0)
            )
            torch._assert_async(torch.isfinite(direction).all())
            torch._assert_async((descent > 0.0).all())
            torch._assert_async((budget_residual <= 64.0 * torch.finfo(torch.float32).eps).all())
            for index, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(direction[index].to(parameter.dtype), alpha=-lr)
            role_records[role] = {
                "momenta": momenta,
                "adaptive": adaptive_equal,
                "source": source,
                "source_norm": source_norm,
                "momentum_norm": momentum_norm,
                "direction": direction,
                "descent": descent,
                "budget_residual": budget_residual,
                "factor_condition": factor_condition,
            }

        if self._capture_telemetry_next_step:
            qkv = role_records["qkv"]
            out = role_records["attn_out"]
            c = incoming_alignment
            adaptive_amplitude = torch.sqrt((1.0 - c).clamp_min(0.0))
            tiny = torch.finfo(torch.float32).tiny
            qkv_cosine = (qkv["momenta"] * qkv["adaptive"]).sum(
                dim=(-2, -1)
            ) / (
                qkv["momentum_norm"]
                * torch.linalg.vector_norm(qkv["adaptive"], dim=(-2, -1))
            ).clamp_min(tiny)
            out_cosine = (out["momenta"] * out["adaptive"]).sum(
                dim=(-2, -1)
            ) / (
                out["momentum_norm"]
                * torch.linalg.vector_norm(out["adaptive"], dim=(-2, -1))
            ).clamp_min(tiny)
            self._last_telemetry = {
                "rlb_r04_attention_step": router_step,
                "rlb_r04_attention_alignment_min": float(c.amin().item()),
                "rlb_r04_attention_alignment_median": float(c.median().item()),
                "rlb_r04_attention_alignment_max": float(c.amax().item()),
                "rlb_r04_attention_router_activity_max": float(
                    adaptive_amplitude.amax().item()
                ),
                "rlb_r04_qkv_factor_condition_max": float(
                    qkv["factor_condition"].amax().item()
                ),
                "rlb_r04_attention_output_factor_condition_max": float(
                    out["factor_condition"].amax().item()
                ),
                "rlb_r04_qkv_branch_disagreement_max": float(
                    torch.sqrt((1.0 - qkv_cosine.square()).clamp_min(0.0)).amax().item()
                ),
                "rlb_r04_attention_output_branch_disagreement_max": float(
                    torch.sqrt((1.0 - out_cosine.square()).clamp_min(0.0)).amax().item()
                ),
                "rlb_r04_attention_budget_residual_max": float(
                    torch.maximum(
                        qkv["budget_residual"].amax(), out["budget_residual"].amax()
                    ).item()
                ),
                "rlb_r04_attention_descent_min": float(
                    torch.minimum(qkv["descent"].amin(), out["descent"].amin()).item()
                ),
            }
        self._capture_telemetry_next_step = False
        return loss
