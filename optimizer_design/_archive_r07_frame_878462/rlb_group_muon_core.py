"""Shared numerical kernel for opaque RLB optimizer slots R01 and R02.

This file contains implementation infrastructure, not a separately numbered
scientific candidate. Both candidates use Muon's fixed momentum, quintic
Newton--Schulz coefficients, five iterations, and ``match_rms_adamw``
calibration. Their only difference is the RLB-derived direct-sum norm.
"""

from __future__ import annotations

import math

import torch


_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_NS_EPS = 1.0e-7


def _batched_zero_power(momentum: torch.Tensor, steps: int) -> torch.Tensor:
    """Muon quintic Newton--Schulz polar map for a batch of wide matrices."""
    if momentum.ndim != 3:
        raise RuntimeError(f"expected [batch, rows, cols], got {tuple(momentum.shape)}")
    transposed = momentum.shape[-2] > momentum.shape[-1]
    work = momentum.transpose(-2, -1) if transposed else momentum
    work = work.to(dtype=torch.bfloat16)
    norm = torch.linalg.vector_norm(
        work.float(), dim=(-2, -1), keepdim=True
    ).clamp_min(_NS_EPS)
    work = work / norm.to(dtype=work.dtype)
    a, b, c = _NS_COEFFICIENTS
    for _ in range(int(steps)):
        gram = torch.bmm(work, work.transpose(1, 2))
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        work = torch.baddbmm(work, polynomial, work, beta=a)
    return work.transpose(-2, -1) if transposed else work


def _match_rms_adamw_scale(rows: int, cols: int) -> float:
    """The exact shape rule used by the matched PyTorch Muon control."""
    return 0.2 * math.sqrt(max(int(rows), int(cols)))


class RLBGroupMuonCore(torch.optim.Optimizer):
    """Muon over the direct sum induced by Global-RLB activation groups."""

    def __init__(
        self,
        pairs,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        mode: str,
    ):
        if mode not in {"r01", "r02"}:
            raise ValueError("opaque RLB optimizer mode must be r01 or r02")
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("RLB group optimizer requires at least one matrix pair")
        self.mode = mode
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        if self.momentum != 0.95:
            raise ValueError("campaign Muon momentum must be 0.95")
        if self.ns_steps != 5:
            raise ValueError("campaign Muon Newton--Schulz step count must be 5")

        parameters = []
        seen = set()
        signature = None
        for pair_index, pair in enumerate(self.pairs):
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            groups = int(pair["groups"])
            hidden = int(pair["hidden_dim"])
            if incoming.ndim != 2 or outgoing.ndim != 2:
                raise ValueError("RLB optimizer pairs must contain two matrices")
            if hidden % groups != 0:
                raise ValueError("RLB hidden width must be divisible by groups")
            width = hidden // groups
            current = (groups, width, incoming.shape[1], outgoing.shape[0])
            if incoming.shape[0] != hidden or outgoing.shape[1] != hidden:
                raise ValueError(f"invalid RLB pair shape at index {pair_index}")
            if incoming.shape[1] != outgoing.shape[0]:
                raise ValueError("campaign RLB pair must have matched external width")
            if signature is None:
                signature = current
            elif current != signature:
                raise ValueError("all campaign RLB pairs must have the same structural shape")
            for parameter in (incoming, outgoing):
                if id(parameter) in seen:
                    raise ValueError("RLB matrix occurs in more than one optimizer pair")
                seen.add(id(parameter))
                parameters.append(parameter)
        self.groups, self.width, self.external_width, _ = signature
        defaults = {"lr": float(lr), "weight_decay": float(weight_decay), "lr_scale": 1.0}
        super().__init__([{"params": parameters}], defaults)

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "group_lr_scale": 1.0,
            "matrix_role_lr_scale": 1.0,
            "phase_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _nesterov_momentum(self, parameter: torch.Tensor) -> torch.Tensor:
        """Return the exact Nesterov momentum tensor used by torch.optim.Muon."""
        if parameter.grad is None:
            raise RuntimeError("RLB matrix gradient is missing")
        state = self.state[parameter]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(parameter)
            state["momentum_buffer"] = buffer
        buffer.lerp_(parameter.grad, 1.0 - self.momentum)
        return parameter.grad.lerp(buffer, self.momentum)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("RLB group optimizer refuses nonunit LR scale")

        incoming_momenta = []
        outgoing_momenta = []
        for pair in self.pairs:
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming_momenta.append(
                self._nesterov_momentum(incoming).view(
                    self.groups, self.width, self.external_width
                )
            )
            outgoing_momenta.append(
                self._nesterov_momentum(outgoing)
                .view(self.external_width, self.groups, self.width)
                .permute(1, 2, 0)
            )

        incoming_batch = torch.stack(incoming_momenta, dim=0).flatten(0, 1)
        outgoing_batch = torch.stack(outgoing_momenta, dim=0).flatten(0, 1)
        block_count = incoming_batch.shape[0]
        if self.mode == "r01":
            work = torch.cat((incoming_batch, outgoing_batch), dim=0)
            direction = _batched_zero_power(work, self.ns_steps)
            direction.mul_(
                _match_rms_adamw_scale(self.width, self.external_width)
            )
            incoming_direction = direction[:block_count]
            outgoing_direction = direction[block_count:]
        else:
            work = torch.cat((incoming_batch, outgoing_batch), dim=-1)
            direction = _batched_zero_power(work, self.ns_steps)
            direction.mul_(
                _match_rms_adamw_scale(
                    self.width, 2 * self.external_width
                )
            )
            incoming_direction, outgoing_direction = direction.split(
                self.external_width, dim=-1
            )

        incoming_direction = incoming_direction.view(
            len(self.pairs), self.groups, self.width, self.external_width
        )
        outgoing_direction = outgoing_direction.view(
            len(self.pairs), self.groups, self.width, self.external_width
        )
        for index, pair in enumerate(self.pairs):
            pair["in_weight"].add_(
                incoming_direction[index].reshape_as(pair["in_weight"]).to(
                    dtype=pair["in_weight"].dtype
                ),
                alpha=-lr,
            )
            outgoing_update = (
                outgoing_direction[index]
                .permute(2, 0, 1)
                .reshape_as(pair["out_weight"])
                .to(dtype=pair["out_weight"].dtype)
            )
            pair["out_weight"].add_(outgoing_update, alpha=-lr)
        return loss
