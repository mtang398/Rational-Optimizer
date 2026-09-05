"""Exact ragged publication for original R01's local-owner INT8 path.

The parent path publishes five layer rows from every rank even though the
literal owner inventory is [5, 5, 4, 4].  This implementation keeps the same
block-256 INT8 values, FP32 scales, decode order, and FP32 updates, but uses a
grouped point-to-point publication for the unequal row counts.  No numerical
or scientific equation changes.
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


FAMILY_ID = "r01_9150_local_owner_block256_int8_ragged_v1"


def _packed_transport_views(
    packet: torch.Tensor,
    *,
    rows: int,
    padded_elements: int,
    blocks_per_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose aligned INT8 values and FP32 scales inside one byte packet."""

    rows = int(rows)
    padded_elements = int(padded_elements)
    blocks_per_layer = int(blocks_per_layer)
    value_bytes = rows * padded_elements
    scale_bytes = 4 * rows * blocks_per_layer
    if (
        packet.dtype != torch.uint8
        or packet.ndim != 1
        or packet.numel() != value_bytes + scale_bytes
        or value_bytes % 4 != 0
    ):
        raise RuntimeError("ragged INT8 packed transport layout changed")
    values = packet.narrow(0, 0, value_bytes).view(torch.int8).view(
        rows, padded_elements
    )
    scales = packet.narrow(0, value_bytes, scale_bytes).view(
        torch.float32
    ).view(rows, blocks_per_layer)
    return values, scales


def _initialize_ragged_owner_transport(composite) -> None:
    composite._ragged_family_counts = tuple(
        int(family[0].numel()) for family in composite._structural_families
    )
    composite._ragged_family_shapes = tuple(
        tuple(int(item) for item in family[0].shape)
        for family in composite._structural_families
    )
    for family, count, shape in zip(
        composite._structural_families,
        composite._ragged_family_counts,
        composite._ragged_family_shapes,
    ):
        if any(
            int(parameter.numel()) != count
            or tuple(int(item) for item in parameter.shape) != shape
            for parameter in family
        ):
            raise RuntimeError("ragged INT8 owner family shape changed")

    inventories = owner_layer_lists()
    owner_counts = tuple(map(len, inventories))
    local_count = len(composite.owned_layers)
    if local_count != owner_counts[composite.rank]:
        raise RuntimeError("ragged INT8 owner inventory changed")
    elements = sum(composite._ragged_family_counts)
    padded = (
        (elements + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
    ) * BLOCK_ELEMENTS
    blocks_per_layer = padded // BLOCK_ELEMENTS
    first = composite._structural_families[0][0]
    device = first.device
    composite._ragged_elements_per_layer = elements
    composite._ragged_blocks_per_layer = blocks_per_layer
    composite._ragged_owner_counts = owner_counts
    composite._ragged_before = torch.empty(
        local_count, padded, device=device, dtype=torch.float32
    )
    composite._ragged_delta = torch.zeros_like(composite._ragged_before)
    # Quantization needs only the local row count.  After publication the same
    # storage becomes a max-owner-count decode batch, avoiding 18 one-row
    # decode kernels and a separate model-row-sized decode allocation.
    composite._ragged_scratch = torch.empty(
        max(owner_counts), padded, device=device, dtype=torch.float32
    )
    composite._ragged_packets = [
        torch.empty(
            count * padded + 4 * count * blocks_per_layer,
            device=device,
            dtype=torch.uint8,
        )
        for count in owner_counts
    ]
    composite._ragged_send_packet = torch.empty_like(
        composite._ragged_packets[composite.rank]
    )
    (
        composite._ragged_send_values,
        composite._ragged_send_scales,
    ) = _packed_transport_views(
        composite._ragged_send_packet,
        rows=local_count,
        padded_elements=padded,
        blocks_per_layer=blocks_per_layer,
    )
    packet_views = [
        _packed_transport_views(
            packet,
            rows=count,
            padded_elements=padded,
            blocks_per_layer=blocks_per_layer,
        )
        for packet, count in zip(composite._ragged_packets, owner_counts)
    ]
    composite._ragged_values = [views[0] for views in packet_views]
    composite._ragged_scales = [views[1] for views in packet_views]
    composite._ragged_wire_value_bytes = 0
    composite._ragged_wire_scale_bytes = 0


class Method1LocalLayerOwnerInt8RaggedComposite(
    Method1LocalLayerOwnerComposite
):
    """Bitwise-equivalent INT8 publication without padded owner rows."""

    _SCHEMA = FAMILY_ID + "_transport"

    def __init__(self, blocks, adamw, **kwargs):
        super().__init__(blocks, adamw, **kwargs)
        _initialize_ragged_owner_transport(self)

    @torch.no_grad()
    def step(self):
        local_families = tuple(
            tuple(family[layer] for layer in self.owned_layers)
            for family in self._structural_families
        )
        offset = 0
        for family, count, shape in zip(
            local_families,
            self._ragged_family_counts,
            self._ragged_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                self._ragged_before[
                    local_index, offset : offset + count
                ].view(shape).copy_(parameter)
            offset += count

        with _owner_local_reductions():
            self.router.step()
            self.attention.step()

        offset = 0
        for family, count, shape in zip(
            local_families,
            self._ragged_family_counts,
            self._ragged_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                original = self._ragged_before[
                    local_index, offset : offset + count
                ].view(shape)
                target = self._ragged_delta[
                    local_index, offset : offset + count
                ].view(shape)
                torch.sub(parameter, original, out=target)
                parameter.copy_(original)
            offset += count

        delta_blocks = self._ragged_delta.view(
            self._ragged_delta.shape[0],
            self._ragged_blocks_per_layer,
            BLOCK_ELEMENTS,
        )
        scratch_blocks = self._ragged_scratch[
            : self._ragged_delta.shape[0]
        ].view_as(delta_blocks)
        torch.abs(delta_blocks, out=scratch_blocks)
        torch.amax(scratch_blocks, dim=-1, out=self._ragged_send_scales)
        self._ragged_send_scales.div_(127.0)
        self._ragged_send_scales.clamp_(
            min=torch.finfo(torch.float32).tiny
        )
        torch.div(
            delta_blocks,
            self._ragged_send_scales.unsqueeze(-1),
            out=scratch_blocks,
        )
        scratch_blocks.round_().clamp_(-127, 127)
        self._ragged_send_values.copy_(
            self._ragged_scratch[: self._ragged_delta.shape[0]]
        )

        publish_ragged_owner_rows(
            self._ragged_send_packet, self._ragged_packets, rank=self.rank
        )

        maxima = []
        inventories = owner_layer_lists()
        for owner, inventory in enumerate(inventories):
            owner_count = len(inventory)
            decoded_rows = self._ragged_scratch[:owner_count]
            decode_blocks = decoded_rows.view(
                owner_count,
                self._ragged_blocks_per_layer,
                BLOCK_ELEMENTS,
            )
            torch.mul(
                self._ragged_values[owner].view_as(decode_blocks),
                self._ragged_scales[owner].unsqueeze(-1),
                out=decode_blocks,
            )
            maxima.append(
                decoded_rows[:, : self._ragged_elements_per_layer].abs().amax()
            )
            parameters = []
            deltas = []
            for local_index, layer in enumerate(inventory):
                offset = 0
                for family, count, shape in zip(
                    self._structural_families,
                    self._ragged_family_counts,
                    self._ragged_family_shapes,
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
        self._ragged_wire_value_bytes = sum(
            value.numel() for value in self._ragged_values
        )
        self._ragged_wire_scale_bytes = 4 * sum(
            value.numel() for value in self._ragged_scales
        )
        self._last_wire_elements = self._ragged_wire_value_bytes
        self._last_wire_bytes = (
            self._ragged_wire_value_bytes + self._ragged_wire_scale_bytes
        )

    def telemetry(self):
        result = super().telemetry()
        result.update({
            "rlb_layer_owner_int8_block_elements": BLOCK_ELEMENTS,
            "rlb_layer_owner_int8_wire_bytes": self._last_wire_bytes,
            "rlb_layer_owner_int8_value_bytes": self._ragged_wire_value_bytes,
            "rlb_layer_owner_int8_scale_bytes": self._ragged_wire_scale_bytes,
        })
        return result

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "transport_execution_family_id": FAMILY_ID,
            "direction_wire": "block256_symmetric_int8_plus_fp32_scales",
            "ragged_owner_counts": self._ragged_owner_counts,
            "padded_owner_slots_removed": True,
            "packed_value_scale_publication": True,
            "grouped_publication_calls_per_step": 1,
            "owner_batched_decode": True,
            "owner_batched_foreach_apply": True,
            "scientific_equations_changed_vs_int8_parent": False,
            "floating_point_update_changed_vs_int8_parent": False,
            "collective_payload_bits_changed_vs_int8_parent": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required": False,
        })
        return result


class R019150LocalLayerOwnerInt8RaggedComposite(
    Method1LocalLayerOwnerInt8RaggedComposite,
    R019150LocalLayerOwnerComposite,
):
    """Original R01 with exact ragged INT8 owner publication."""

    _SCHEMA = FAMILY_ID + "_composite"


__all__ = (
    "FAMILY_ID",
    "Method1LocalLayerOwnerInt8RaggedComposite",
    "R019150LocalLayerOwnerInt8RaggedComposite",
    "_initialize_ragged_owner_transport",
    "_packed_transport_views",
)
