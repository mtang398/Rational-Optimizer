#!/usr/bin/env python3
"""Fail closed on the matched four-A6000, NVLink/P2P launch contract."""

from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST")
    if not job_id or not node:
        raise RuntimeError("missing Slurm allocation identity")
    description = run("scontrol", "show", "job", job_id, "-o")
    requested_tres_raw = slurm_value(description, "ReqTRES")
    if requested_tres_raw is None:
        raise RuntimeError("Slurm job description has null ReqTRES")
    requested_tres = set(requested_tres_raw.split(","))
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
    selected_tokens = os.environ.get(
        "CAMPAIGN_SELECTED_PHYSICAL_GPU_IDS", ""
    ).split(",")
    identity_rows = run(
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"
    ).splitlines()
    uuid_to_index = {}
    for identity_row in identity_rows:
        physical_index, uuid = (item.strip() for item in identity_row.split(",", 1))
        uuid_to_index[uuid] = physical_index
    selected_indices = [
        token if token.isdigit() else uuid_to_index.get(token, "")
        for token in selected_tokens
    ]
    topology_rows = {
        fields[0][3:]: fields[1:]
        for line in topology.splitlines()
        if (fields := line.split()) and fields[0].startswith("GPU")
    }
    topology_header = next(
        (line.split() for line in topology.splitlines() if line.lstrip().startswith("GPU0")),
        [],
    )
    topology_columns = [item[3:] for item in topology_header if item.startswith("GPU")]
    nvlink_peer_by_physical_index = {}
    for physical_index in selected_indices:
        values = topology_rows.get(physical_index, [])
        peer_values = dict(zip(topology_columns, values))
        nvlink_peer_by_physical_index[physical_index] = any(
            peer_values.get(other, "").startswith("NV")
            for other in selected_indices if other != physical_index
        )
    every_physical_gpu_has_nvlink_peer = bool(selected_indices) and all(
        selected_indices
    ) and all(nvlink_peer_by_physical_index.values())
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
        and feature is not None
        and "nvlink" in feature.lower()
        and required_tres <= requested_tres
        and "OverSubscribe=NO" in description
        and "CpusPerTres=gres/gpu:4" in description
        and visible_count == 4
        and selected_names == ["NVIDIA RTX A6000"] * 4
        and every_rank_has_peer
        and every_physical_gpu_has_nvlink_peer
        and os.environ.get("NCCL_P2P_DISABLE") == "0"
        and os.environ.get("NCCL_SHM_DISABLE") == "0"
        and os.environ.get("RATIONAL_OPT_TORCH_FALLBACK") == "0"
    )
    payload = {
        "schema": "four_a6000_nvlink_p2p_allocation_audit_v1",
        "passed": passed,
        "slurm_job_id": job_id,
        "node": node,
        "requested_node": slurm_value(description, "ReqNodeList"),
        "excluded_node": slurm_value(description, "ExcNodeList"),
        "requested_features": feature,
        "requested_tres": sorted(requested_tres),
        "visible_cuda_count": visible_count,
        "selected_local_rank_gpu_names": selected_names,
        "cuda_peer_access_matrix": peer_matrix,
        "every_selected_rank_has_a_direct_peer": every_rank_has_peer,
        "selected_physical_gpu_indices": selected_indices,
        "nvlink_peer_by_physical_index": nvlink_peer_by_physical_index,
        "every_selected_physical_gpu_has_an_nvlink_peer": (
            every_physical_gpu_has_nvlink_peer
        ),
        "topology_diagnostic": topology,
        "nvlink_status_diagnostic": nvlink.stdout.strip(),
        "nvlink_status_exit_code": nvlink.returncode,
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
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
