"""Compiled realization of the exact packed-INT4 owner transport.

The current local-apply executor launches the blockwise absolute value,
maximum, scaling, rounding, nibble conversion, and packing operations eagerly.
It likewise launches unpack, conversion, subtraction, and block scaling as
separate operations for every remote owner.  The programs below preserve the
literal floating-point operation order and exact integer nibble values while
writing into the same preallocated buffers; ``torch.compile`` may fuse their
fixed-shape CUDA realization.

This module deliberately inherits publication and local-apply semantics from
the already bitwise-gated parent.  It changes no quantization code, FP32
scale, decoded update, optimizer equation, cadence, LR/WD, or Newton--Schulz
iteration.  Quality inheritance is forbidden unless a complete multi-rank
CUDA transition gate proves bitwise equality to that parent.
"""

from __future__ import annotations

import torch

from .rlb_int4_owner_local_apply_collective import (
    Method1GlobalOwnerTransactionCachedFunctionalInt4LocalApplyComposite,
    Method1GlobalOwnerTransactionCachedFunctionalInt4PaddedComposite,
    R019150BatchedResponseInverseCompiledSpanInt4LocalApplyComposite,
    R019150BatchedResponseInverseCompiledSpanInt4PaddedComposite,
    _apply_decoded_owner_rows,
)
from .rlb_layer_owner_collectives import owner_layer_lists
from .rlb_layer_owner_int8_delta import BLOCK_ELEMENTS
from .rlb_method1_local_layer_owner import _owner_local_reductions


RAGGED_FAMILY_ID = "int4_owner_local_apply_compiled_transport_ragged_v1"
PADDED_FAMILY_ID = "int4_owner_local_apply_compiled_transport_padded_v1"
QUANTIZATION_MAX = 7.0


def _pack_program(
    delta_blocks: torch.Tensor,
    scratch_blocks: torch.Tensor,
    nibbles: torch.Tensor,
    send_packed: torch.Tensor,
    send_scales: torch.Tensor,
) -> torch.Tensor:
    """Execute the parent's literal block-256 pack into existing buffers."""

    torch.abs(delta_blocks, out=scratch_blocks)
    torch.amax(scratch_blocks, dim=-1, out=send_scales)
    send_scales.div_(QUANTIZATION_MAX)
    send_scales.clamp_(min=torch.finfo(torch.float32).tiny)
    torch.div(delta_blocks, send_scales.unsqueeze(-1), out=scratch_blocks)
    scratch_blocks.round_().clamp_(-7, 7)
    nibbles.copy_(scratch_blocks.view_as(nibbles)).add_(8)
    nibble_pairs = nibbles.view(nibbles.shape[0], -1, 2)
    torch.bitwise_left_shift(nibble_pairs[..., 1], 4, out=send_packed)
    torch.bitwise_or(send_packed, nibble_pairs[..., 0], out=send_packed)
    return scratch_blocks


def _decode_program(
    packed: torch.Tensor,
    scales: torch.Tensor,
    decode_codes: torch.Tensor,
    decoded_rows: torch.Tensor,
) -> torch.Tensor:
    """Execute the parent's literal nibble decode into existing buffers."""

    # Dynamo cannot capture ``out=decode_pairs[..., 0]`` because that view is
    # non-contiguous.  Stacking the same two exact integer nibbles is storage-
    # equivalent and lets Inductor fuse the fixed-shape unpack program.
    low = torch.bitwise_and(packed, 15)
    high = torch.bitwise_right_shift(packed, 4)
    decode_codes.copy_(
        torch.stack((low, high), dim=-1).view_as(decode_codes)
    )
    decoded_rows.copy_(decode_codes).sub_(8.0)
    decode_blocks = decoded_rows.view(
        decoded_rows.shape[0], scales.shape[1], BLOCK_ELEMENTS
    )
    torch.mul(decode_blocks, scales.unsqueeze(-1), out=decode_blocks)
    return decoded_rows


def _functional_pack_program(delta_blocks: torch.Tensor):
    """Return exact pack values without mutating aliased packet views.

    Production packed bytes and FP32 scales are two dtype views of one byte
    packet. AOTAutograd cannot compile mutations through such mixed-dtype
    aliases. Returning the literal values functionally and copying them into
    those same views after the compiled call removes only that compiler
    limitation.
    """

    scales = delta_blocks.abs().amax(dim=-1)
    scales = scales.div(QUANTIZATION_MAX)
    scales = scales.clamp(min=torch.finfo(torch.float32).tiny)
    codes = delta_blocks.div(scales.unsqueeze(-1)).round().clamp(-7, 7)
    nibbles = codes.to(torch.uint8).add(8)
    pairs = nibbles.view(nibbles.shape[0], -1, 2)
    packed = torch.bitwise_or(
        torch.bitwise_left_shift(pairs[..., 1], 4), pairs[..., 0]
    )
    return codes, packed, scales


_compiled_functional_pack_program = torch.compile(
    _functional_pack_program, fullgraph=True, dynamic=False
)
_compiled_decode_program = torch.compile(
    _decode_program, fullgraph=True, dynamic=False
)


class _CompiledInt4TransportMixin:
    """Replace eager packing/remote decoding with fixed-shape programs."""

    @torch.no_grad()
    def _step_int4_owner_local_apply(self):
        local_families = tuple(
            tuple(family[layer] for layer in self.owned_layers)
            for family in self._structural_families
        )
        offset = 0
        for family, count, shape in zip(
            local_families,
            self._int4_family_counts,
            self._int4_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                self._int4_before[
                    local_index, offset : offset + count
                ].view(shape).copy_(parameter)
            offset += count

        with _owner_local_reductions():
            self.router.step()
            self.attention.step()

        offset = 0
        for family, count, shape in zip(
            local_families,
            self._int4_family_counts,
            self._int4_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                original = self._int4_before[
                    local_index, offset : offset + count
                ].view(shape)
                target = self._int4_delta[
                    local_index, offset : offset + count
                ].view(shape)
                torch.sub(parameter, original, out=target)
            offset += count

        delta_blocks = self._int4_delta.view(
            self._int4_delta.shape[0],
            self._int4_blocks_per_layer,
            BLOCK_ELEMENTS,
        )
        scratch_blocks = self._int4_scratch[
            : self._int4_delta.shape[0]
        ].view_as(delta_blocks)
        send_packed, send_scales = self._int4_transport_send_views()
        if delta_blocks.is_cuda:
            codes, packed, scales = _compiled_functional_pack_program(
                delta_blocks
            )
            scratch_blocks.copy_(codes)
            send_packed.copy_(packed)
            send_scales.copy_(scales)
        else:
            _pack_program(
                delta_blocks,
                scratch_blocks,
                self._int4_nibbles,
                send_packed,
                send_scales,
            )

        self._publish_int4_owner_rows()

        inventories = owner_layer_lists()
        maxima = [None] * len(inventories)
        local_count = len(inventories[self.rank])
        local_rows = self._int4_scratch[:local_count]
        local_blocks = local_rows.view(
            local_count, self._int4_blocks_per_layer, BLOCK_ELEMENTS
        )
        torch.mul(local_blocks, send_scales.unsqueeze(-1), out=local_blocks)
        maxima[self.rank] = _apply_decoded_owner_rows(
            self,
            owner=self.rank,
            inventory=inventories[self.rank],
            decoded_rows=local_rows,
            local_owner=True,
        )

        decode = (
            _compiled_decode_program
            if self._int4_delta.is_cuda
            else _decode_program
        )
        for owner, inventory in enumerate(inventories):
            if owner == self.rank:
                continue
            owner_count = len(inventory)
            decode_codes = self._int4_decode_codes[:owner_count]
            decoded_rows = self._int4_scratch[:owner_count]
            packed, scales = self._int4_remote_views(owner)
            decode(packed, scales, decode_codes, decoded_rows)
            maxima[owner] = _apply_decoded_owner_rows(
                self,
                owner=owner,
                inventory=inventory,
                decoded_rows=decoded_rows,
                local_owner=False,
            )

        self.adamw.step()
        if any(value is None for value in maxima):
            raise RuntimeError("compiled INT4 maximum inventory changed")
        self._last_delta_max_abs = float(torch.stack(maxima).amax().item())
        value_bytes, scale_bytes = self._int4_actual_wire_bytes()
        self._int4_wire_value_bytes = int(value_bytes)
        self._int4_wire_scale_bytes = int(scale_bytes)
        self._last_wire_elements = self._int4_wire_value_bytes
        self._last_wire_bytes = value_bytes + scale_bytes

    def _compiled_transport_execution_report(self):
        return {
            "pack_realization": "fixed_shape_compiled_functional_literal_values",
            "remote_decode_realization": "fixed_shape_compiled_literal_program",
            "preallocated_transport_outputs_preserved": True,
            "quantization_codes_changed_vs_parent": False,
            "fp32_scales_changed_vs_parent": False,
            "decoded_updates_changed_vs_parent": False,
            "optimizer_equations_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "quality_inheritance_requires_complete_cuda_bitwise_preflight": True,
        }


class Method1Int4LocalApplyCompiledTransportComposite(
    _CompiledInt4TransportMixin,
    Method1GlobalOwnerTransactionCachedFunctionalInt4LocalApplyComposite,
):
    _SCHEMA = RAGGED_FAMILY_ID + "_method1_composite"

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._compiled_transport_execution_report())
        result["family_id"] = RAGGED_FAMILY_ID + "_method1"
        return result


class Method1Int4PaddedCompiledTransportComposite(
    _CompiledInt4TransportMixin,
    Method1GlobalOwnerTransactionCachedFunctionalInt4PaddedComposite,
):
    _SCHEMA = PADDED_FAMILY_ID + "_method1_composite"

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._compiled_transport_execution_report())
        result["family_id"] = PADDED_FAMILY_ID + "_method1"
        return result


class R01Int4LocalApplyCompiledTransportComposite(
    _CompiledInt4TransportMixin,
    R019150BatchedResponseInverseCompiledSpanInt4LocalApplyComposite,
):
    _SCHEMA = RAGGED_FAMILY_ID + "_r01_composite"

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._compiled_transport_execution_report())
        result["family_id"] = RAGGED_FAMILY_ID + "_r01"
        return result


class R01Int4PaddedCompiledTransportComposite(
    _CompiledInt4TransportMixin,
    R019150BatchedResponseInverseCompiledSpanInt4PaddedComposite,
):
    _SCHEMA = PADDED_FAMILY_ID + "_r01_composite"

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._compiled_transport_execution_report())
        result["family_id"] = PADDED_FAMILY_ID + "_r01"
        return result


__all__ = (
    "PADDED_FAMILY_ID",
    "RAGGED_FAMILY_ID",
    "Method1Int4LocalApplyCompiledTransportComposite",
    "Method1Int4PaddedCompiledTransportComposite",
    "R01Int4LocalApplyCompiledTransportComposite",
    "R01Int4PaddedCompiledTransportComposite",
)
