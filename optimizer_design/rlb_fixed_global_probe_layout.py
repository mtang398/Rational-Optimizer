"""Topology-independent layout for a globally fixed loss-probe measure."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FixedGlobalProbeLayout:
    global_probe_count: int
    process_count: int
    process_rank: int
    global_start: int
    global_stop: int
    local_probe_count: int
    method_state_depends_on_total_tokens: bool
    method_state_depends_on_machine_count: bool


def fixed_global_probe_layout(
    global_probe_count: int,
    process_rank: int,
    process_count: int,
) -> FixedGlobalProbeLayout:
    """Partition fixed probe IDs evenly, allowing empty high-rank shards."""

    probes = int(global_probe_count)
    rank = int(process_rank)
    world = int(process_count)
    if probes < 2 or world < 1 or rank < 0 or rank >= world:
        raise RuntimeError("fixed global probe layout is invalid")
    start = (rank * probes) // world
    stop = ((rank + 1) * probes) // world
    return FixedGlobalProbeLayout(
        global_probe_count=probes,
        process_count=world,
        process_rank=rank,
        global_start=start,
        global_stop=stop,
        local_probe_count=stop - start,
        method_state_depends_on_total_tokens=False,
        method_state_depends_on_machine_count=False,
    )


def evenly_spaced_indices(
    row_count: int,
    selected_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Deterministically select 0, 1, or more rows without division hazards."""

    rows = int(row_count)
    selected = int(selected_count)
    if rows < 0 or selected < 0 or selected > rows:
        raise RuntimeError("fixed probe row selection is invalid")
    if selected == 0:
        return torch.empty(0, device=device, dtype=torch.int64)
    if selected == 1:
        return torch.zeros(1, device=device, dtype=torch.int64)
    numerators = torch.arange(
        selected, device=device, dtype=torch.int64
    ) * (rows - 1)
    return torch.div(numerators, selected - 1, rounding_mode="floor")


def required_capture_rows_per_microbatch(
    local_probe_count: int,
    microbatch_count: int,
    *,
    minimum_capture_rows: int,
) -> int:
    local = int(local_probe_count)
    microbatches = int(microbatch_count)
    minimum = int(minimum_capture_rows)
    if local < 0 or microbatches < 1 or minimum < 1:
        raise RuntimeError("fixed probe capture layout is invalid")
    needed = (local + microbatches - 1) // microbatches
    return max(minimum, needed)


__all__ = (
    "FixedGlobalProbeLayout",
    "evenly_spaced_indices",
    "fixed_global_probe_layout",
    "required_capture_rows_per_microbatch",
)
