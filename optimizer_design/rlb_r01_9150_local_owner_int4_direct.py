"""Block-256 symmetric INT4 publication for original R01 local ownership.

The scientific R01 owner equations are unchanged.  Only the already
approximate owner-update wire changes from signed INT8 values to two packed
signed INT4 values per byte, with the same FP32 block scales.  Every rank,
including the owner, applies the same decoded FP32 update.  This is a numerical
approximation and requires matched timing followed by a fresh complete
9,150-step FineWeb-Edu quality trajectory if faster.
"""

from __future__ import annotations

import torch

from .rlb_layer_owner_collectives import owner_layer_lists
from .rlb_layer_owner_int8_delta import BLOCK_ELEMENTS
from .rlb_layer_owner_ragged_publication import publish_ragged_owner_rows
from .rlb_method1_local_layer_owner import (
    Method1LocalLayerOwnerComposite,
    _owner_local_reductions,
)
from .rlb_r01_9150_local_layer_owner import R019150LocalLayerOwnerComposite


FAMILY_ID = "r01_9150_local_owner_block256_int4_direct_apply_v1"
QUANTIZATION_MAX = 7.0


def _packed_int4_transport_views(
    packet: torch.Tensor,
    *,
    rows: int,
    packed_elements: int,
    blocks_per_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose packed INT4 bytes and aligned FP32 scales in one packet."""

    rows = int(rows)
    packed_elements = int(packed_elements)
    blocks_per_layer = int(blocks_per_layer)
    value_bytes = rows * packed_elements
    scale_bytes = 4 * rows * blocks_per_layer
    if (
        packet.dtype != torch.uint8
        or packet.ndim != 1
        or packet.numel() != value_bytes + scale_bytes
        or value_bytes % 4 != 0
    ):
        raise RuntimeError("direct INT4 packed transport layout changed")
    values = packet.narrow(0, 0, value_bytes).view(rows, packed_elements)
    scales = packet.narrow(0, value_bytes, scale_bytes).view(
        torch.float32
    ).view(rows, blocks_per_layer)
    return values, scales


def pack_signed_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack last-dimension signed codes in [-7, 7] into uint8 nibbles."""
    if codes.dtype is not torch.int8 or int(codes.shape[-1]) % 2:
        raise RuntimeError("signed INT4 pack inventory changed")
    if bool(((codes < -7) | (codes > 7)).any()):
        raise RuntimeError("signed INT4 code is outside [-7, 7]")
    nibbles = codes.to(torch.uint8).add(8).view(*codes.shape[:-1], -1, 2)
    high = torch.bitwise_left_shift(nibbles[..., 1], 4)
    return torch.bitwise_or(nibbles[..., 0], high)


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 nibbles to interleaved signed INT8 codes."""
    if packed.dtype is not torch.uint8:
        raise RuntimeError("signed INT4 unpack requires uint8 input")
    result = torch.empty(
        *packed.shape[:-1], int(packed.shape[-1]) * 2,
        device=packed.device,
        dtype=torch.uint8,
    )
    pairs = result.view(*packed.shape, 2)
    torch.bitwise_and(packed, 15, out=pairs[..., 0])
    torch.bitwise_right_shift(packed, 4, out=pairs[..., 1])
    return result.to(torch.int8).sub(8)


class Method1LocalLayerOwnerInt4DirectComposite(
    Method1LocalLayerOwnerComposite
):
    """Allocation-stable packed INT4 owner update publication."""

    _SCHEMA = FAMILY_ID + "_transport"

    def __init__(self, blocks, adamw, **kwargs):
        super().__init__(blocks, adamw, **kwargs)
        self._int4_family_counts = tuple(
            int(family[0].numel()) for family in self._structural_families
        )
        self._int4_family_shapes = tuple(
            tuple(int(item) for item in family[0].shape)
            for family in self._structural_families
        )
        for family, count, shape in zip(
            self._structural_families,
            self._int4_family_counts,
            self._int4_family_shapes,
        ):
            if any(
                int(parameter.numel()) != count
                or tuple(int(item) for item in parameter.shape) != shape
                for parameter in family
            ):
                raise RuntimeError("direct INT4 owner family shape changed")

        inventories = owner_layer_lists()
        owner_counts = tuple(map(len, inventories))
        local_count = len(self.owned_layers)
        if local_count != owner_counts[self.rank]:
            raise RuntimeError("direct INT4 owner inventory changed")
        elements = sum(self._int4_family_counts)
        padded = (
            (elements + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
        ) * BLOCK_ELEMENTS
        if padded % 2:
            raise RuntimeError("direct INT4 padded inventory must be even")
        blocks_per_layer = padded // BLOCK_ELEMENTS
        first = self._structural_families[0][0]
        device = first.device
        self._int4_elements_per_layer = elements
        self._int4_padded_elements = padded
        self._int4_blocks_per_layer = blocks_per_layer
        self._int4_owner_counts = owner_counts
        self._int4_before = torch.empty(
            local_count, padded, device=device, dtype=torch.float32
        )
        self._int4_delta = torch.zeros_like(self._int4_before)
        # Reuse this FP32 storage after quantization as one owner-batched
        # decode buffer.  Four batched decodes replace eighteen row decodes.
        self._int4_scratch = torch.empty(
            max(owner_counts), padded, device=device, dtype=torch.float32
        )
        self._int4_nibbles = torch.empty(
            local_count, padded, device=device, dtype=torch.uint8
        )
        # Use the literal [5, 5, 4, 4] owner inventory rather than forcing
        # ranks 2 and 3 to publish a fifth padded layer.  A grouped send/receive
        # publication supports those unequal tensors on both Gloo and NCCL.
        # The payload is byte-identical for every real layer while removing
        # 10% of value and scale traffic.
        self._int4_packets = [
            torch.empty(
                count * (padded // 2) + 4 * count * blocks_per_layer,
                device=device,
                dtype=torch.uint8,
            )
            for count in owner_counts
        ]
        self._int4_send_packet = torch.empty_like(
            self._int4_packets[self.rank]
        )
        self._int4_send_packed, self._int4_send_scales = (
            _packed_int4_transport_views(
                self._int4_send_packet,
                rows=local_count,
                packed_elements=padded // 2,
                blocks_per_layer=blocks_per_layer,
            )
        )
        packet_views = [
            _packed_int4_transport_views(
                packet,
                rows=count,
                packed_elements=padded // 2,
                blocks_per_layer=blocks_per_layer,
            )
            for packet, count in zip(self._int4_packets, owner_counts)
        ]
        self._int4_gathered_packed = [views[0] for views in packet_views]
        self._int4_gathered_scales = [views[1] for views in packet_views]
        self._int4_decode_codes = torch.empty(
            max(owner_counts), padded, device=device, dtype=torch.uint8
        )
        self._int4_wire_value_bytes = 0
        self._int4_wire_scale_bytes = 0

    @torch.no_grad()
    def step(self):
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
                parameter.copy_(original)
            offset += count

        delta_blocks = self._int4_delta.view(
            self._int4_delta.shape[0],
            self._int4_blocks_per_layer,
            BLOCK_ELEMENTS,
        )
        scratch_blocks = self._int4_scratch[
            : self._int4_delta.shape[0]
        ].view_as(delta_blocks)
        torch.abs(delta_blocks, out=scratch_blocks)
        torch.amax(scratch_blocks, dim=-1, out=self._int4_send_scales)
        self._int4_send_scales.div_(QUANTIZATION_MAX)
        self._int4_send_scales.clamp_(min=torch.finfo(torch.float32).tiny)
        torch.div(
            delta_blocks,
            self._int4_send_scales.unsqueeze(-1),
            out=scratch_blocks,
        )
        scratch_blocks.round_().clamp_(-7, 7)
        # Converting a negative integral float directly to uint8 gives the
        # same modulo-256 representation as float -> int8 -> uint8.  Adding
        # eight therefore produces the exact [1, 15] nibble inventory while
        # avoiding a complete model-sized INT8 intermediate and copy.
        self._int4_nibbles.copy_(
            self._int4_scratch[: self._int4_delta.shape[0]]
        ).add_(8)
        nibble_pairs = self._int4_nibbles.view(
            self._int4_nibbles.shape[0], -1, 2
        )
        torch.bitwise_left_shift(
            nibble_pairs[..., 1], 4, out=self._int4_send_packed
        )
        torch.bitwise_or(
            self._int4_send_packed,
            nibble_pairs[..., 0],
            out=self._int4_send_packed,
        )

        publish_ragged_owner_rows(
            self._int4_send_packet,
            self._int4_packets,
            rank=self.rank,
        )

        maxima = []
        inventories = owner_layer_lists()
        for owner, inventory in enumerate(inventories):
            owner_count = len(inventory)
            decode_codes = self._int4_decode_codes[:owner_count]
            decode_pairs = decode_codes.view(owner_count, -1, 2)
            packed = self._int4_gathered_packed[owner]
            torch.bitwise_and(packed, 15, out=decode_pairs[..., 0])
            torch.bitwise_right_shift(packed, 4, out=decode_pairs[..., 1])
            decoded_rows = self._int4_scratch[:owner_count]
            decoded_rows.copy_(decode_codes).sub_(8.0)
            decode_blocks = decoded_rows.view(
                owner_count,
                self._int4_blocks_per_layer,
                BLOCK_ELEMENTS,
            )
            torch.mul(
                decode_blocks,
                self._int4_gathered_scales[owner].unsqueeze(-1),
                out=decode_blocks,
            )
            maxima.append(
                decoded_rows[:, : self._int4_elements_per_layer].abs().amax()
            )
            parameters = []
            deltas = []
            for local_index, layer in enumerate(inventory):
                offset = 0
                for family, count, shape in zip(
                    self._structural_families,
                    self._int4_family_counts,
                    self._int4_family_shapes,
                ):
                    parameters.append(family[layer])
                    deltas.append(
                        decoded_rows[
                            local_index, offset : offset + count
                        ].view(shape)
                    )
                    offset += count
            torch._foreach_add_(parameters, deltas)

        self.adamw.step()
        self._last_delta_max_abs = float(torch.stack(maxima).amax().item())
        self._int4_wire_value_bytes = sum(
            value.numel() for value in self._int4_gathered_packed
        )
        self._int4_wire_scale_bytes = (
            4 * sum(value.numel() for value in self._int4_gathered_scales)
        )
        self._last_wire_elements = self._int4_wire_value_bytes
        self._last_wire_bytes = (
            self._int4_wire_value_bytes + self._int4_wire_scale_bytes
        )

    def telemetry(self):
        result = super().telemetry()
        result.update({
            "rlb_layer_owner_int4_block_elements": BLOCK_ELEMENTS,
            "rlb_layer_owner_int4_wire_bytes": self._last_wire_bytes,
            "rlb_layer_owner_int4_value_bytes": self._int4_wire_value_bytes,
            "rlb_layer_owner_int4_scale_bytes": self._int4_wire_scale_bytes,
        })
        return result

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "transport_execution_family_id": FAMILY_ID,
            "direction_wire": "block256_symmetric_packed_int4_plus_fp32_scales",
            "quantization_code_range": (-7, 7),
            "values_per_wire_byte": 2,
            "ragged_owner_counts": self._int4_owner_counts,
            "padded_owner_slots_removed": True,
            "packed_value_scale_publication": True,
            "grouped_publication_calls_per_step": 1,
            "owner_batched_decode": True,
            "owner_batched_foreach_apply": True,
            "persistent_collective_buffers": True,
            "max_owner_count_decode_buffer": True,
            "model_sized_int8_intermediate_removed": True,
            "scientific_equations_changed_vs_int8_parent": False,
            "floating_point_update_changed_vs_int8_parent": True,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required": True,
        })
        return result


class R019150LocalLayerOwnerInt4DirectComposite(
    Method1LocalLayerOwnerInt4DirectComposite,
    R019150LocalLayerOwnerComposite,
):
    """Original R01 owner equations with packed INT4 update publication."""

    _SCHEMA = FAMILY_ID + "_composite"


__all__ = (
    "FAMILY_ID",
    "QUANTIZATION_MAX",
    "Method1LocalLayerOwnerInt4DirectComposite",
    "R019150LocalLayerOwnerInt4DirectComposite",
    "_packed_int4_transport_views",
    "pack_signed_int4",
    "unpack_signed_int4",
)
