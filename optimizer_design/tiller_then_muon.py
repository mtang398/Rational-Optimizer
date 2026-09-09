"""Two-stage TILLER-then-Muon optimizer candidate.

The first 1000 updates use the full TILLER optimizer.  The remaining updates
use ordinary Muon on the same structural GRAIN/attention matrices while
preserving compatible first-moment state and the AdamW state for all remaining
parameters.
"""

from __future__ import annotations

from typing import Iterable

import torch

from .tiller import TILLERAttentionOptimizer, TILLERRouter, configure_microbatch_count


OPTIMIZER_ID = "tiller_then_muon_v1"
FAMILY_ID = OPTIMIZER_ID
PREFIX = "tiller_then_muon_"
SWITCH_AFTER_STEP = 1000


def _sorted_blocks(blocks):
    ordered = sorted((dict(block) for block in blocks), key=lambda item: int(item["layer_index"]))
    if [int(block["layer_index"]) for block in ordered] != list(range(len(ordered))):
        raise ValueError("TILLER-then-Muon layer inventory must be contiguous")
    return ordered


def _ordinary_muon_named_parameters(blocks) -> list[tuple[str, torch.Tensor]]:
    named: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    for block in blocks:
        layer = int(block["layer_index"])
        entries = (
            (f"layers.{layer}.attn.qkv.weight", block["qkv_weight"]),
            (f"layers.{layer}.attn.out.weight", block["attn_out_weight"]),
            (f"layers.{layer}.mlp.in_proj.weight", block["in_weight"]),
            (f"layers.{layer}.mlp.out_proj.weight", block["out_weight"]),
        )
        for name, parameter in entries:
            if parameter.ndim != 2:
                raise ValueError(f"{name} is not a Muon matrix parameter")
            if id(parameter) in seen:
                raise ValueError("TILLER-then-Muon structural parameter ownership overlaps")
            seen.add(id(parameter))
            named.append((name, parameter))
    return named


def _extend_param_groups(optimizers: Iterable[object]) -> list[dict]:
    param_groups: list[dict] = []
    for optimizer in optimizers:
        param_groups.extend(optimizer.param_groups)
    return param_groups


class TillerThenMuonOptimizer:
    """Composite optimizer with a state-preserving TILLER-to-Muon switch."""

    _SCHEMA = "tiller_then_muon_composite_v1"

    def __init__(
        self,
        blocks,
        *,
        adam_groups,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta1: float,
        beta2: float,
        eps: float,
        adjust_lr_fn: str,
    ):
        self.blocks = _sorted_blocks(blocks)
        if not self.blocks:
            raise ValueError("TILLER-then-Muon requires GRAIN blocks")
        if int(SWITCH_AFTER_STEP) != 1000:
            raise RuntimeError("TILLER-then-Muon switch step is fixed at 1000")

        self.tiller_router = TILLERRouter(
            self.blocks,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            eps=eps,
        )
        self.tiller_attention = TILLERAttentionOptimizer(
            self.blocks,
            self.tiller_router,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        muon_named = _ordinary_muon_named_parameters(self.blocks)
        self._ordinary_muon = torch.optim.Muon(
            muon_named,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        if not adam_groups:
            raise ValueError("TILLER-then-Muon requires an AdamW child for non-structural parameters")
        self._adamw = torch.optim.AdamW(
            adam_groups,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )

        self._momentum_sources: list[tuple[torch.Tensor, object]] = []
        for block in self.blocks:
            self._momentum_sources.extend(
                (
                    (block["in_weight"], self.tiller_router),
                    (block["out_weight"], self.tiller_router),
                    (block["qkv_weight"], self.tiller_attention),
                    (block["attn_out_weight"], self.tiller_attention),
                )
            )

        self._completed_steps = 0
        self._transitioned = False
        self._capture_telemetry_next_step = False
        self._last_telemetry: dict[str, object] = {}
        self.optimizers = [self.tiller_router, self.tiller_attention, self._adamw]
        self.param_groups = _extend_param_groups(self.optimizers)

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def switch_after_step(self) -> int:
        return SWITCH_AFTER_STEP

    @property
    def stage(self) -> str:
        return "muon" if self._transitioned else "tiller"

    @property
    def ordinary_muon(self):
        return self._ordinary_muon

    @property
    def adamw(self):
        return self._adamw

    def lr_wd_fairness_audit(self):
        result = {
            "formal_candidate_lr_scale": 1.0,
            "stage1_full_tiller_lr_scale": 1.0,
            "stage2_ordinary_muon_lr_scale": 1.0,
            "state_preserving_switch_lr_scale": 1.0,
            "shared_schedule_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }
        for prefix, child in (
            ("stage1_router_", self.tiller_router),
            ("stage1_attention_", self.tiller_attention),
        ):
            provider = getattr(child, "lr_wd_fairness_audit", None)
            if provider is None:
                continue
            for key, value in provider().items():
                result[prefix + key] = value
        return result

    def set_telemetry_capture(self, enabled=True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def method_specific_sync_active(self) -> bool:
        return not self._transitioned

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def record_realized_clipping(self, preclip_norm, max_norm):
        if self._transitioned:
            return
        if preclip_norm is None:
            raise RuntimeError("TILLER-then-Muon requires the realized gradient norm before the switch")
        self.tiller_router.record_realized_clipping(preclip_norm, max_norm)

    def _snapshot_child_telemetry(self):
        telemetry: dict[str, object] = {}
        for child in (self.tiller_router, self.tiller_attention):
            getter = getattr(child, "telemetry", None)
            if getter is None:
                continue
            telemetry.update(getter())
        return telemetry

    def _copy_current_lr_to_ordinary_muon(self):
        reference = self.tiller_router.param_groups[0]
        for group in self._ordinary_muon.param_groups:
            group["lr"] = float(reference["lr"])
            group["weight_decay"] = float(reference["weight_decay"])

    def _transfer_momentum_buffers(self):
        missing = []
        for parameter, source in self._momentum_sources:
            source_state = source.state.get(parameter, {})
            buffer = source_state.get("momentum_buffer")
            if not torch.is_tensor(buffer):
                missing.append(tuple(parameter.shape))
                continue
            self._ordinary_muon.state[parameter]["momentum_buffer"] = buffer.detach().clone(
                memory_format=torch.preserve_format
            )
        if missing:
            raise RuntimeError(f"TILLER-then-Muon missing compatible momentum buffers: {missing[:4]}")

    def _remove_tiller_hooks_and_probe_storage(self):
        handles = list(getattr(self.tiller_router, "_hook_handles", ()))
        for handle in handles:
            handle.remove()
        self.tiller_router._hook_handles = []
        if hasattr(self.tiller_router, "_pending_inputs"):
            self.tiller_router._pending_inputs = [None for _ in self.tiller_router.pairs]
        if hasattr(self.tiller_router, "_functional_records"):
            self.tiller_router._functional_records = [[] for _ in self.tiller_router.pairs]
        if hasattr(self.tiller_router, "_cotangent_records"):
            self.tiller_router._cotangent_records = [[] for _ in self.tiller_router.pairs]
        if hasattr(self.tiller_router, "_clip_factor"):
            self.tiller_router._clip_factor = None

    def _activate_muon_stage(self):
        self._transitioned = True
        self._remove_tiller_hooks_and_probe_storage()
        self.optimizers = [self._ordinary_muon, self._adamw]
        self.param_groups = _extend_param_groups(self.optimizers)

    def _transition_to_muon(self):
        if self._transitioned:
            return
        if self._completed_steps != SWITCH_AFTER_STEP:
            raise RuntimeError("TILLER-then-Muon transition attempted at the wrong step")
        self._copy_current_lr_to_ordinary_muon()
        self._transfer_momentum_buffers()
        self._activate_muon_stage()

    @torch.no_grad()
    def step(self):
        self._last_telemetry = {}
        prior_stage = self.stage
        switched_now = False
        child_telemetry: dict[str, object] = {}
        for optimizer in list(self.optimizers):
            optimizer.step()
        self._completed_steps += 1
        if prior_stage == "tiller" and self._capture_telemetry_next_step:
            child_telemetry = self._snapshot_child_telemetry()
        if prior_stage == "tiller" and self._completed_steps == SWITCH_AFTER_STEP:
            self._transition_to_muon()
            switched_now = True
        if self._capture_telemetry_next_step:
            self._last_telemetry = {
                PREFIX + "family_id": FAMILY_ID,
                PREFIX + "formal_candidate": 1,
                PREFIX + "switch_after_step": SWITCH_AFTER_STEP,
                PREFIX + "completed_steps": self._completed_steps,
                PREFIX + "stage_before_step": prior_stage,
                PREFIX + "stage_after_step": self.stage,
                PREFIX + "switched_to_muon": int(switched_now),
                PREFIX + "state_transition": "preserve_compatible_state",
            }
            if switched_now:
                self._last_telemetry.update(child_telemetry)
        self._capture_telemetry_next_step = False

    def state_dict(self):
        return {
            "schema": self._SCHEMA,
            "optimizer_id": OPTIMIZER_ID,
            "switch_after_step": SWITCH_AFTER_STEP,
            "completed_steps": self._completed_steps,
            "stage": self.stage,
            "state_transition": "preserve_compatible_state",
            "children": {
                "tiller_router": self.tiller_router.state_dict(),
                "tiller_attention": self.tiller_attention.state_dict(),
                "ordinary_muon": self._ordinary_muon.state_dict(),
                "adamw": self._adamw.state_dict(),
            },
        }

    def load_state_dict(self, state_dict):
        if (
            not isinstance(state_dict, dict)
            or state_dict.get("schema") != self._SCHEMA
            or state_dict.get("optimizer_id") != OPTIMIZER_ID
            or int(state_dict.get("switch_after_step", -1)) != SWITCH_AFTER_STEP
            or state_dict.get("state_transition") != "preserve_compatible_state"
            or not isinstance(state_dict.get("children"), dict)
        ):
            raise RuntimeError("TILLER-then-Muon checkpoint schema changed")
        children = state_dict["children"]
        self.tiller_router.load_state_dict(children["tiller_router"])
        self.tiller_attention.load_state_dict(children["tiller_attention"])
        self._ordinary_muon.load_state_dict(children["ordinary_muon"])
        self._adamw.load_state_dict(children["adamw"])
        self._completed_steps = int(state_dict["completed_steps"])
        if self._completed_steps < 0:
            raise RuntimeError("TILLER-then-Muon checkpoint step is invalid")
        self._transitioned = bool(self._completed_steps >= SWITCH_AFTER_STEP)
        if self._transitioned:
            self._activate_muon_stage()
        else:
            self.optimizers = [self.tiller_router, self.tiller_attention, self._adamw]
            self.param_groups = _extend_param_groups(self.optimizers)
        self._last_telemetry = {}
        self._capture_telemetry_next_step = False


__all__ = (
    "FAMILY_ID",
    "OPTIMIZER_ID",
    "PREFIX",
    "SWITCH_AFTER_STEP",
    "TillerThenMuonOptimizer",
    "configure_microbatch_count",
)
