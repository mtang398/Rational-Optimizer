"""Collective primitives for four-rank layer-owned RLB execution.

These primitives do not define an optimizer and do not alter Newton--Schulz.
They provide the communication boundaries required to move a complete RLB
transaction from replicated execution to stable ``layer % 4`` ownership:

* reduce-scatter complete per-layer metric families to their owners;
* all-to-all rank-local functional rows to those same owners;
* reconstruct small owner-computed coordinate-score packets globally;
* reconstruct BF16-quantized owner directions on every DDP replica.

The final direction wire is deliberately numerical: every rank, including the
owner, consumes the same BF16-dequantized direction while FP32 parameters stay
FP32.  A router using this module therefore requires a fresh quality run.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist


FAMILY_ID = "rlb_four_rank_layer_owner_collectives_v1"
WORLD_SIZE = 4
LAYERS = 18


def _require_world() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("RLB layer ownership requires initialized distributed")
    world = int(dist.get_world_size())
    if world != WORLD_SIZE:
        raise RuntimeError("RLB layer ownership requires exactly four ranks")
    return int(dist.get_rank()), world


def owner_layer_lists(
    *, layers: int = LAYERS, world: int = WORLD_SIZE
) -> tuple[tuple[int, ...], ...]:
    layers = int(layers)
    world = int(world)
    if layers < 1 or world < 1:
        raise ValueError("layer-owner inventory must be positive")
    return tuple(
        tuple(layer for layer in range(layers) if layer % world == owner)
        for owner in range(world)
    )


def owner_layer_indices(
    owner: int,
    device: torch.device,
    *,
    layers: int = LAYERS,
    world: int = WORLD_SIZE,
) -> torch.Tensor:
    inventories = owner_layer_lists(layers=layers, world=world)
    owner = int(owner)
    if not 0 <= owner < len(inventories):
        raise ValueError("layer owner is outside the world")
    return torch.tensor(
        inventories[owner], device=device, dtype=torch.int64
    )


def _check_layer_families(
    families: Sequence[torch.Tensor], *, layers: int
) -> tuple[torch.device, torch.dtype]:
    if not families:
        raise RuntimeError("RLB layer-owner family inventory is empty")
    first = families[0]
    if first.ndim < 2 or int(first.shape[0]) != int(layers):
        raise RuntimeError("RLB layer-owner family lost its layer axis")
    for value in families:
        if (
            value.ndim < 2
            or int(value.shape[0]) != int(layers)
            or value.device != first.device
            or value.dtype != first.dtype
        ):
            raise RuntimeError("RLB layer-owner families are incompatible")
    return first.device, first.dtype


def _flatten_layer_families(
    families: Sequence[torch.Tensor], *, layers: int
) -> tuple[torch.Tensor, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    _check_layer_families(families, layers=layers)
    counts = tuple(int(value[0].numel()) for value in families)
    shapes = tuple(tuple(int(item) for item in value.shape[1:]) for value in families)
    flat = torch.cat(
        tuple(value.reshape(int(layers), count) for value, count in zip(families, counts)),
        dim=-1,
    )
    return flat, counts, shapes


def _split_layer_families(
    flat: torch.Tensor,
    counts: Sequence[int],
    shapes: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, ...]:
    if len(counts) != len(shapes):
        raise RuntimeError("RLB layer-owner split metadata changed")
    pieces = flat.split(tuple(int(count) for count in counts), dim=-1)
    prefix = tuple(int(item) for item in flat.shape[:-1])
    return tuple(
        piece.reshape(*prefix, *tuple(int(item) for item in shape))
        for piece, shape in zip(pieces, shapes)
    )


def reduce_scatter_layer_metric_families(
    families: Sequence[torch.Tensor], *, layers: int = LAYERS
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """SUM complete metric families and return only this rank's layers.

    The two production metric families have different matrix widths, so they
    are flattened per layer and submitted in one reduce-scatter packet.  Two
    padding layers make the 18-layer inventory evenly chunkable by four.
    """

    rank, world = _require_world()
    flat, family_counts, family_shapes = _flatten_layer_families(
        families, layers=layers
    )
    inventories = owner_layer_lists(layers=layers, world=world)
    maximum = max(len(item) for item in inventories)
    packet = torch.zeros(
        world,
        maximum,
        int(flat.shape[-1]),
        device=flat.device,
        dtype=flat.dtype,
    )
    for owner in range(world):
        indices = owner_layer_indices(
            owner, flat.device, layers=layers, world=world
        )
        torch.index_select(
            flat, 0, indices, out=packet[owner, : indices.numel()]
        )
    owned_packet = torch.empty(
        maximum,
        int(flat.shape[-1]),
        device=flat.device,
        dtype=flat.dtype,
    )
    dist.reduce_scatter_tensor(
        owned_packet,
        packet.reshape(world * maximum, int(flat.shape[-1])),
        op=dist.ReduceOp.SUM,
    )
    owned_indices = owner_layer_indices(
        rank, flat.device, layers=layers, world=world
    )
    owned = owned_packet[: owned_indices.numel()]
    return (
        _split_layer_families(owned, family_counts, family_shapes),
        owned_indices,
    )


def exchange_functional_row_families(
    families: Sequence[torch.Tensor], *, layers: int = LAYERS
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Send each rank's functional rows to stable layer owners.

    Inputs are ``[layer, row, ...]`` with a common row count.  Returned
    families are ``[origin_rank, owned_layer, row, ...]``.  The origin axis is
    explicit so cross-layer score rows are never accidentally mixed.
    """

    rank, world = _require_world()
    device, dtype = _check_layer_families(families, layers=layers)
    if any(value.ndim < 3 for value in families):
        raise RuntimeError("RLB functional family requires layer and row axes")
    rows = int(families[0].shape[1])
    if any(int(value.shape[1]) != rows for value in families):
        raise RuntimeError("RLB functional row counts differ")
    family_counts = tuple(int(value[0, 0].numel()) for value in families)
    family_shapes = tuple(
        tuple(int(item) for item in value.shape[2:]) for value in families
    )
    packet = torch.cat(
        tuple(
            value.reshape(int(layers), rows, count)
            for value, count in zip(families, family_counts)
        ),
        dim=-1,
    )
    inventories = owner_layer_lists(layers=layers, world=world)
    sends = []
    send_splits = []
    for owner, inventory in enumerate(inventories):
        indices = owner_layer_indices(
            owner, device, layers=layers, world=world
        )
        owned = packet.index_select(0, indices).contiguous().view(-1)
        sends.append(owned)
        send_splits.append(int(owned.numel()))
    send = torch.cat(sends)
    owned_indices = owner_layer_indices(
        rank, device, layers=layers, world=world
    )
    receive_per_origin = (
        int(owned_indices.numel()) * rows * int(packet.shape[-1])
    )
    receive_splits = [receive_per_origin for _ in range(world)]
    receive = torch.empty(
        sum(receive_splits), device=device, dtype=dtype
    )
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=receive_splits,
        input_split_sizes=send_splits,
    )
    shaped = receive.view(
        world,
        int(owned_indices.numel()),
        rows,
        int(packet.shape[-1]),
    )
    return (
        _split_layer_families(shaped, family_counts, family_shapes),
        owned_indices,
    )


def gather_owner_coordinate_scores(
    local_scores: torch.Tensor,
    *,
    coordinates_per_layer: int,
    layers: int = LAYERS,
) -> torch.Tensor:
    """Reconstruct globally ordered owner-computed coordinate scores."""

    rank, world = _require_world()
    coordinates_per_layer = int(coordinates_per_layer)
    if coordinates_per_layer < 1 or local_scores.ndim < 1:
        raise RuntimeError("RLB owner coordinate inventory is invalid")
    inventories = owner_layer_lists(layers=layers, world=world)
    owned = inventories[rank]
    expected = len(owned) * coordinates_per_layer
    if int(local_scores.shape[-1]) != expected:
        raise RuntimeError("RLB owner score width changed")
    maximum = max(len(item) for item in inventories) * coordinates_per_layer
    send = torch.zeros(
        *local_scores.shape[:-1],
        maximum,
        device=local_scores.device,
        dtype=local_scores.dtype,
    )
    send[..., :expected].copy_(local_scores)
    gathered_flat = torch.empty(
        world * send.numel(), device=send.device, dtype=send.dtype
    )
    dist.all_gather_into_tensor(gathered_flat, send.reshape(-1))
    gathered = gathered_flat.view(world, *send.shape)
    canonical = torch.empty(
        *local_scores.shape[:-1],
        int(layers) * coordinates_per_layer,
        device=local_scores.device,
        dtype=local_scores.dtype,
    )
    for owner, inventory in enumerate(inventories):
        global_indices = torch.tensor(
            [
                layer * coordinates_per_layer + coordinate
                for layer in inventory
                for coordinate in range(coordinates_per_layer)
            ],
            device=local_scores.device,
            dtype=torch.int64,
        )
        canonical.index_copy_(
            -1,
            global_indices,
            gathered[owner, ..., : global_indices.numel()],
        )
    return canonical


def gather_quantized_owner_direction_families(
    local_families: Sequence[torch.Tensor],
    *,
    layers: int = LAYERS,
    wire_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    """Return the same dequantized full direction on every rank.

    Each input is ``[owned_layer, ...]``.  All families must share the owned
    layer count, device, and FP32 dtype.  Owners also consume the reconstructed
    result, ensuring no rank applies a higher-precision private direction.
    """

    rank, world = _require_world()
    if wire_dtype is not torch.bfloat16:
        raise RuntimeError("RLB direction wire is fixed to bfloat16")
    if not local_families:
        raise RuntimeError("RLB owner direction inventory is empty")
    inventories = owner_layer_lists(layers=layers, world=world)
    owned_count = len(inventories[rank])
    first = local_families[0]
    if first.dtype != torch.float32 or int(first.shape[0]) != owned_count:
        raise RuntimeError("RLB owner direction must be FP32 owned layers")
    for value in local_families:
        if (
            value.dtype != first.dtype
            or value.device != first.device
            or int(value.shape[0]) != owned_count
        ):
            raise RuntimeError("RLB owner direction families differ")
    family_counts = tuple(int(value[0].numel()) for value in local_families)
    family_shapes = tuple(
        tuple(int(item) for item in value.shape[1:])
        for value in local_families
    )
    local_flat = torch.cat(
        tuple(
            value.reshape(owned_count, count)
            for value, count in zip(local_families, family_counts)
        ),
        dim=-1,
    ).to(dtype=wire_dtype)
    maximum = max(len(item) for item in inventories)
    send = torch.zeros(
        maximum,
        int(local_flat.shape[-1]),
        device=first.device,
        dtype=wire_dtype,
    )
    send[:owned_count].copy_(local_flat)
    gathered_flat = torch.empty(
        world * send.numel(), device=first.device, dtype=wire_dtype
    )
    dist.all_gather_into_tensor(gathered_flat, send.reshape(-1))
    gathered = gathered_flat.view(world, maximum, int(local_flat.shape[-1]))
    canonical = torch.empty(
        int(layers),
        int(local_flat.shape[-1]),
        device=first.device,
        dtype=wire_dtype,
    )
    for owner, inventory in enumerate(inventories):
        indices = owner_layer_indices(
            owner, first.device, layers=layers, world=world
        )
        canonical.index_copy_(
            0, indices, gathered[owner, : indices.numel()]
        )
    return _split_layer_families(
        canonical.float(), family_counts, family_shapes
    )


def layer_owner_execution_report() -> dict[str, object]:
    return {
        "family_id": FAMILY_ID,
        "world_size": WORLD_SIZE,
        "layers": LAYERS,
        "ownership": "layer_mod_world",
        "metric_collective": "one_padded_reduce_scatter_packet",
        "functional_collective": "origin_preserving_all_to_all",
        "score_collective": "padded_all_gather_exact_reconstruction",
        "direction_collective": "padded_bf16_all_gather_then_fp32_dequantize",
        "fp32_parameters_quantized": False,
        "newton_schulz_changed": False,
        "fresh_quality_required": True,
    }


__all__ = (
    "FAMILY_ID",
    "LAYERS",
    "WORLD_SIZE",
    "exchange_functional_row_families",
    "gather_owner_coordinate_scores",
    "gather_quantized_owner_direction_families",
    "layer_owner_execution_report",
    "owner_layer_indices",
    "owner_layer_lists",
    "reduce_scatter_layer_metric_families",
)
