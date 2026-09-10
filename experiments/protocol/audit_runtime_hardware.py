#!/usr/bin/env python3
"""Fail closed on the matched four-A6000, NVLink/P2P launch contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch


def run(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def slurm_value(description: str, key: str) -> str | None:
    prefix = f"{key}="
    token = next(
        (item for item in description.split() if item.startswith(prefix)), None
    )
    if token is None:
        raise RuntimeError(f"Slurm job description lacks {key}")
    value = token[len(prefix):]
    return None if value == "(null)" else value


def optional_slurm_value(description: str, key: str) -> str | None:
    prefix = f"{key}="
    token = next(
        (item for item in description.split() if item.startswith(prefix)), None
    )
    if token is None:
        return None
    value = token[len(prefix):]
    return None if value == "(null)" else value


def topology_peer_values(topology: str) -> dict[str, dict[str, str]]:
    topology_rows: dict[str, list[str]] = {}
    for line in topology.splitlines():
        fields = line.split()
        if not fields or re.fullmatch(r"GPU[0-9]+", fields[0]) is None:
            continue
        # A matrix data row contains its diagonal X entry; the header does not.
        if "X" not in fields[1:]:
            continue
        topology_rows[fields[0][3:]] = fields[1:]
    topology_columns = sorted(topology_rows, key=int)
    return {
        row_index: dict(zip(topology_columns, values))
        for row_index, values in topology_rows.items()
    }


def nvlink_peer_map(topology: str, selected_indices: list[str]) -> dict[str, bool]:
    """Return whether each selected visible GPU has a selected NVLink peer."""
    peer_values_by_index = topology_peer_values(topology)
    return {
        visible_index: any(
            peer_values_by_index.get(visible_index, {}).get(other, "").startswith("NV")
            for other in selected_indices
            if other != visible_index
        )
        for visible_index in selected_indices
    }


def nvlink_edges(topology: str, selected_indices: list[str]) -> list[tuple[str, str]]:
    peer_values_by_index = topology_peer_values(topology)
    edges: list[tuple[str, str]] = []
    for left_position, left in enumerate(selected_indices):
        for right in selected_indices[left_position + 1 :]:
            if peer_values_by_index.get(left, {}).get(right, "").startswith("NV"):
                edges.append((left, right))
    return edges


def has_two_disjoint_nvlink_pairs(edges: list[tuple[str, str]]) -> bool:
    for first_index, first in enumerate(edges):
        for second in edges[first_index + 1 :]:
            if len({*first, *second}) == 4:
                return True
    return False


def normalize_uuid(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value).strip()
    if not text:
        return ""
    if not text.upper().startswith("GPU-") and re.fullmatch(
        r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
        text,
    ):
        text = f"GPU-{text}"
    return text.upper()


def torch_visible_gpu_uuids(visible_count: int) -> list[str]:
    return [
        normalize_uuid(getattr(torch.cuda.get_device_properties(index), "uuid", ""))
        for index in range(visible_count)
    ]


def visible_uuid_to_nvidia_smi_index(identity_rows: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for identity_row in identity_rows:
        visible_index, uuid = (item.strip() for item in identity_row.split(",", 1))
        mapping[normalize_uuid(uuid)] = visible_index
    return mapping


def split_gpu_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_topology_indices_from_torch_uuids(
    torch_uuids: list[str], identity_rows: list[str]
) -> list[str]:
    uuid_to_index = visible_uuid_to_nvidia_smi_index(identity_rows)
    return [
        uuid_to_index.get(normalize_uuid(uuid), "")
        for uuid in torch_uuids
    ]


def resolve_env_gpu_indices(
    tokens: list[str], identity_rows: list[str]
) -> list[str]:
    uuid_to_index = visible_uuid_to_nvidia_smi_index(identity_rows)
    known_indices = set(uuid_to_index.values())
    result: list[str] = []
    for token in tokens:
        normalized = token.removeprefix("gpu:")
        if normalized in known_indices:
            result.append(normalized)
        else:
            result.append(uuid_to_index.get(normalize_uuid(normalized), ""))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST")
    if not job_id or not node:
        raise RuntimeError("missing Slurm allocation identity")
    description = run("scontrol", "show", "job", job_id, "-o")
    partition_name = slurm_value(description, "Partition")
    partition_description = run("scontrol", "show", "partition", "gpu", "-o")
    partition_over_subscribe = slurm_value(partition_description, "OverSubscribe")
    job_over_subscribe = slurm_value(description, "OverSubscribe")
    requested_tres_raw = slurm_value(description, "ReqTRES")
    if requested_tres_raw is None:
        raise RuntimeError("Slurm job description has null ReqTRES")
    requested_tres = set(requested_tres_raw.split(","))
    allocated_tres_raw = optional_slurm_value(description, "AllocTRES")
    allocated_tres = set(allocated_tres_raw.split(",")) if allocated_tres_raw else set()
    required_tres = {
        "cpu=16",
        "mem=128G",
        "node=1",
        "gres/gpu=4",
        "gres/gpu:nvidia_rtx_a6000=4",
    }
    feature = slurm_value(description, "Features")
    visible_count = torch.cuda.device_count()
    selected_names = [torch.cuda.get_device_name(i) for i in range(visible_count)]
    peer_matrix = [
        [i == j or torch.cuda.can_device_access_peer(i, j) for j in range(visible_count)]
        for i in range(visible_count)
    ]
    # A6000 NVLink is pairwise.  Every selected rank must have at least one
    # direct peer; the remaining paths may traverse PCIe under NCCL.
    every_rank_has_peer = all(
        any(peer_matrix[i][j] for j in range(visible_count) if i != j)
        for i in range(visible_count)
    )
    topology = run("nvidia-smi", "topo", "-m")
    identity_rows = run(
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"
    ).splitlines()
    torch_gpu_uuids = torch_visible_gpu_uuids(visible_count)
    selected_visible_indices = resolve_topology_indices_from_torch_uuids(
        torch_gpu_uuids, identity_rows
    )
    selected_visible_tokens = split_gpu_list(
        os.environ.get("CAMPAIGN_SELECTED_VISIBLE_GPU_IDS")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    selected_visible_token_indices = resolve_env_gpu_indices(
        selected_visible_tokens, identity_rows
    )
    visible_nvlink_edges = nvlink_edges(topology, selected_visible_indices)
    has_two_visible_nvlink_pairs = has_two_disjoint_nvlink_pairs(
        visible_nvlink_edges
    )
    nvlink_peer_by_visible_index = nvlink_peer_map(topology, selected_visible_indices)
    every_visible_gpu_has_nvlink_peer = bool(selected_visible_indices) and all(
        selected_visible_indices
    ) and all(nvlink_peer_by_visible_index.values())
    nvlink = subprocess.run(
        ("nvidia-smi", "nvlink", "--status"),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    passed = (
        slurm_value(description, "ReqNodeList") is None
        and slurm_value(description, "ExcNodeList") is None
        and partition_name == "gpu"
        and partition_over_subscribe == "NO"
        and job_over_subscribe == "NO"
        and feature is not None
        and "nvlink" in feature.lower()
        and required_tres <= requested_tres
        and "CpusPerTres=gres/gpu:4" in description
        and visible_count == 4
        and selected_names == ["NVIDIA RTX A6000"] * 4
        and len(torch_gpu_uuids) == 4
        and all(torch_gpu_uuids)
        and len(set(torch_gpu_uuids)) == 4
        and len(selected_visible_indices) == 4
        and all(selected_visible_indices)
        and len(set(selected_visible_indices)) == 4
        and every_rank_has_peer
        and every_visible_gpu_has_nvlink_peer
        and has_two_visible_nvlink_pairs
        and os.environ.get("NCCL_P2P_DISABLE") == "0"
        and os.environ.get("NCCL_P2P_LEVEL") == "NVL"
        and os.environ.get("NCCL_SHM_DISABLE") == "0"
        and os.environ.get("RATIONAL_OPT_TORCH_FALLBACK") == "0"
    )
    payload = {
        "schema": "four_a6000_nvlink_p2p_allocation_audit_v1",
        "passed": passed,
        "slurm_job_id": job_id,
        "node": node,
        "partition": partition_name,
        "partition_over_subscribe": partition_over_subscribe,
        "job_over_subscribe": job_over_subscribe,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "original_cuda_visible_devices": os.environ.get(
            "CAMPAIGN_ORIGINAL_CUDA_VISIBLE_DEVICES"
        ),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "campaign_slurm_job_gpus": os.environ.get("CAMPAIGN_SLURM_JOB_GPUS"),
        "requested_node": slurm_value(description, "ReqNodeList"),
        "excluded_node": slurm_value(description, "ExcNodeList"),
        "requested_features": feature,
        "requested_tres": sorted(requested_tres),
        "allocated_tres": sorted(allocated_tres),
        "visible_cuda_count": visible_count,
        "selected_local_rank_gpu_names": selected_names,
        "cuda_peer_access_matrix": peer_matrix,
        "every_selected_rank_has_a_direct_peer": every_rank_has_peer,
        "torch_visible_gpu_uuids": torch_gpu_uuids,
        "selected_visible_gpu_tokens": selected_visible_tokens,
        "selected_visible_gpu_token_indices": selected_visible_token_indices,
        "selected_visible_gpu_indices": selected_visible_indices,
        "selected_visible_gpu_uuids": torch_gpu_uuids,
        "selected_visible_nvlink_edges": visible_nvlink_edges,
        "selected_visible_has_two_disjoint_nvlink_pairs": (
            has_two_visible_nvlink_pairs
        ),
        "nvlink_peer_by_visible_index": nvlink_peer_by_visible_index,
        "every_selected_visible_gpu_has_an_nvlink_peer": (
            every_visible_gpu_has_nvlink_peer
        ),
        "selected_physical_gpu_indices": selected_visible_indices,
        "nvlink_peer_by_physical_index": nvlink_peer_by_visible_index,
        "every_selected_physical_gpu_has_an_nvlink_peer": (
            every_visible_gpu_has_nvlink_peer
        ),
        "topology_diagnostic": topology,
        "nvlink_status_diagnostic": nvlink.stdout.strip(),
        "nvlink_status_exit_code": nvlink.returncode,
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL"),
        "nccl_shm_disable": os.environ.get("NCCL_SHM_DISABLE"),
        "torch_fallback": os.environ.get("RATIONAL_OPT_TORCH_FALLBACK"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"four-A6000 NVLink/P2P contract failed: {payload}")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
