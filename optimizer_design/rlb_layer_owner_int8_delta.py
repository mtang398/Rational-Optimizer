"""Blockwise INT8 transport for layer-owned RLB parameter deltas.

The layer-owner optimizer leaves FP32 model parameters and the local optimizer
equations untouched.  This module only replaces its BF16 direction wire with
one symmetric INT8 value per element and one FP32 scale per 256-element block.
Every rank, including the owner, applies the same reconstructed FP32 delta.

This is a numerical approximation and therefore requires fresh quality
evidence.  It does not alter Newton--Schulz, learning rates, weight decay,
optimizer cadence, or model precision.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from .rlb_layer_owner_collectives import (
    LAYERS,
    WORLD_SIZE,
    owner_layer_indices,
    owner_layer_lists,
)


FAMILY_ID = "rlb_layer_owner_block256_int8_delta_v1"
BLOCK_ELEMENTS = 256


def _require_world() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("RLB INT8 delta transport requires distributed")
    world = int(dist.get_world_size())
    if world != WORLD_SIZE:
        raise RuntimeError("RLB INT8 delta transport requires four ranks")
    return int(dist.get_rank()), world


def _flatten_owned_families(
    local_families: Sequence[torch.Tensor], *, owned_count: int
) -> tuple[torch.Tensor, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    if not local_families:
        raise RuntimeError("RLB INT8 delta family inventory is empty")
    first = local_families[0]
    if first.dtype != torch.float32 or int(first.shape[0]) != owned_count:
        raise RuntimeError("RLB INT8 deltas must be FP32 owner-layer tensors")
    for value in local_families:
        if (
            value.dtype != torch.float32
            or value.device != first.device
            or int(value.shape[0]) != owned_count
        ):
            raise RuntimeError("RLB INT8 delta families are incompatible")
    counts = tuple(int(value[0].numel()) for value in local_families)
    shapes = tuple(tuple(int(item) for item in value.shape[1:]) for value in local_families)
    flat = torch.cat(
        tuple(
            value.reshape(owned_count, count)
            for value, count in zip(local_families, counts)
        ),
        dim=-1,
    )
    return flat, counts, shapes


def _split_full_families(
    flat: torch.Tensor,
    counts: Sequence[int],
    shapes: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, ...]:
    pieces = flat.split(tuple(int(count) for count in counts), dim=-1)
    return tuple(
        piece.reshape(int(flat.shape[0]), *tuple(int(item) for item in shape))
        for piece, shape in zip(pieces, shapes)
    )


def gather_blockwise_int8_owner_direction_families(
    local_families: Sequence[torch.Tensor],
    *,
    layers: int = LAYERS,
    block_elements: int = BLOCK_ELEMENTS,
) -> tuple[tuple[torch.Tensor, ...], dict[str, float | int]]:
    """Quantize, gather, and identically reconstruct complete owner deltas."""

    rank, world = _require_world()
    layers = int(layers)
    block_elements = int(block_elements)
    if block_elements != BLOCK_ELEMENTS:
        raise RuntimeError("RLB INT8 direction blocks are fixed at 256 elements")
    inventories = owner_layer_lists(layers=layers, world=world)
    owned_count = len(inventories[rank])
    local_flat, family_counts, family_shapes = _flatten_owned_families(
        local_families, owned_count=owned_count
    )
    elements_per_layer = int(local_flat.shape[-1])
    padded_elements = (
        (elements_per_layer + block_elements - 1) // block_elements
    ) * block_elements
    if padded_elements == elements_per_layer:
        padded = local_flat
    else:
        padded = torch.nn.functional.pad(
            local_flat, (0, padded_elements - elements_per_layer)
        )
    blocks = padded.reshape(owned_count, -1, block_elements)
    scales = blocks.abs().amax(dim=-1).div(127.0)
    scales.clamp_(min=torch.finfo(torch.float32).tiny)
    quantized = torch.round(blocks / scales.unsqueeze(-1)).clamp_(-127, 127)
    quantized = quantized.to(dtype=torch.int8)

    maximum = max(len(item) for item in inventories)
    blocks_per_layer = int(scales.shape[-1])
    send_values = torch.zeros(
        maximum,
        padded_elements,
        device=local_flat.device,
        dtype=torch.int8,
    )
    send_scales = torch.ones(
        maximum,
        blocks_per_layer,
        device=local_flat.device,
        dtype=torch.float32,
    )
    send_values[:owned_count].copy_(quantized.reshape(owned_count, padded_elements))
    send_scales[:owned_count].copy_(scales)
    gathered_values_flat = torch.empty(
        world * send_values.numel(),
        device=local_flat.device,
        dtype=torch.int8,
    )
    gathered_scales_flat = torch.empty(
        world * send_scales.numel(),
        device=local_flat.device,
        dtype=torch.float32,
    )
    dist.all_gather_into_tensor(gathered_values_flat, send_values.reshape(-1))
    dist.all_gather_into_tensor(gathered_scales_flat, send_scales.reshape(-1))
    gathered_values = gathered_values_flat.view(
        world, maximum, padded_elements
    )
    gathered_scales = gathered_scales_flat.view(
        world, maximum, blocks_per_layer
    )
    canonical_values = torch.empty(
        layers, padded_elements, device=local_flat.device, dtype=torch.int8
    )
    canonical_scales = torch.empty(
        layers, blocks_per_layer, device=local_flat.device, dtype=torch.float32
    )
    for owner, inventory in enumerate(inventories):
        indices = owner_layer_indices(
            owner, local_flat.device, layers=layers, world=world
        )
        count = int(indices.numel())
        canonical_values.index_copy_(0, indices, gathered_values[owner, :count])
        canonical_scales.index_copy_(0, indices, gathered_scales[owner, :count])
    decoded = canonical_values.reshape(
        layers, blocks_per_layer, block_elements
    ).float()
    decoded.mul_(canonical_scales.unsqueeze(-1))
    decoded = decoded.reshape(layers, padded_elements)[..., :elements_per_layer]
    wire_value_elements = world * send_values.numel()
    wire_scale_elements = world * send_scales.numel()
    report: dict[str, float | int] = {
        "block_elements": block_elements,
        "elements_per_layer": elements_per_layer,
        "padded_elements_per_layer": padded_elements,
        "wire_value_bytes": wire_value_elements,
        "wire_scale_bytes": 4 * wire_scale_elements,
        "wire_bytes": wire_value_elements + 4 * wire_scale_elements,
    }
    return _split_full_families(
        decoded, family_counts, family_shapes
    ), report


def execution_report() -> dict[str, object]:
    return {
        "family_id": FAMILY_ID,
        "quantization": "symmetric_blockwise_int8",
        "block_elements": BLOCK_ELEMENTS,
        "scale_dtype": "float32",
        "parameters_remain_fp32": True,
        "owners_consume_reconstructed_delta": True,
        "newton_schulz_changed": False,
        "fresh_quality_required": True,
    }


__all__ = (
    "BLOCK_ELEMENTS",
    "FAMILY_ID",
    "execution_report",
    "gather_blockwise_int8_owner_direction_families",
)
