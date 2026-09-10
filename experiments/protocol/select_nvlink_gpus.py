#!/usr/bin/env python3
"""Select exactly four visible A6000 GPUs as two disjoint NVLink pairs."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


EXPECTED_GPU_NAME = "NVIDIA RTX A6000"


def run(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


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


def split_gpu_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_visible_gpus(raw: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi GPU identity row: {line!r}")
        index, uuid, name = fields
        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "normalized_uuid": normalize_uuid(uuid),
                "name": name,
            }
        )
    return devices


def topology_peer_values(topology: str) -> dict[str, dict[str, str]]:
    topology_rows: dict[str, list[str]] = {}
    for line in topology.splitlines():
        fields = line.split()
        if not fields or re.fullmatch(r"GPU[0-9]+", fields[0]) is None:
            continue
        if "X" not in fields[1:]:
            continue
        topology_rows[fields[0][3:]] = fields[1:]
    topology_columns = sorted(topology_rows, key=int)
    return {
        row_index: dict(zip(topology_columns, values))
        for row_index, values in topology_rows.items()
    }


def nvlink_edges(
    topology: str, selected_indices: tuple[str, ...] | list[str]
) -> list[tuple[str, str]]:
    peer_values_by_index = topology_peer_values(topology)
    edges: list[tuple[str, str]] = []
    selected = list(selected_indices)
    for left_position, left in enumerate(selected):
        for right in selected[left_position + 1 :]:
            if peer_values_by_index.get(left, {}).get(right, "").startswith("NV"):
                edges.append((left, right))
    return edges


def has_two_disjoint_nvlink_pairs(edges: list[tuple[str, str]]) -> bool:
    for first_index, first in enumerate(edges):
        for second in edges[first_index + 1 :]:
            if len({*first, *second}) == 4:
                return True
    return False


def resolve_tokens_to_indices(
    tokens: list[str], devices: list[dict[str, str]]
) -> list[str]:
    uuid_to_index = {
        device["normalized_uuid"]: device["index"]
        for device in devices
    }
    known_indices = {device["index"] for device in devices}
    indices: list[str] = []
    for token in tokens:
        normalized = token.removeprefix("gpu:")
        if normalized in known_indices:
            indices.append(normalized)
        else:
            index = uuid_to_index.get(normalize_uuid(normalized))
            if index is None:
                raise RuntimeError(f"CUDA_VISIBLE_DEVICES token is not visible: {token}")
            indices.append(index)
    return indices


def valid_selection(
    devices_by_index: dict[str, dict[str, str]],
    topology: str,
    selected_indices: tuple[str, ...] | list[str],
) -> bool:
    selected = list(selected_indices)
    if len(selected) != 4 or len(set(selected)) != 4:
        return False
    if any(index not in devices_by_index for index in selected):
        return False
    if any(devices_by_index[index]["name"] != EXPECTED_GPU_NAME for index in selected):
        return False
    if any(not devices_by_index[index]["normalized_uuid"] for index in selected):
        return False
    return has_two_disjoint_nvlink_pairs(nvlink_edges(topology, selected))


def choose_selection(
    devices: list[dict[str, str]],
    topology: str,
    original_cuda_visible_devices: str | None,
) -> dict[str, object]:
    devices_by_index = {device["index"]: device for device in devices}
    original_tokens = split_gpu_list(original_cuda_visible_devices)
    if original_tokens:
        available_indices = resolve_tokens_to_indices(original_tokens, devices)
    else:
        available_indices = [device["index"] for device in devices]
    available_indices = list(dict.fromkeys(available_indices))
    first_four = tuple(available_indices[:4])
    if valid_selection(devices_by_index, topology, first_four):
        selected_indices = first_four
        reason = "original_first_four"
    else:
        a6000_indices = [
            index
            for index in sorted(available_indices, key=int)
            if devices_by_index[index]["name"] == EXPECTED_GPU_NAME
        ]
        selected_indices = ()
        reason = ""
        for candidate in itertools.combinations(a6000_indices, 4):
            if valid_selection(devices_by_index, topology, candidate):
                selected_indices = candidate
                reason = "deterministic_valid_pairs"
                break
        if not selected_indices:
            raise RuntimeError("no visible four-A6000 selection has two disjoint NVLink pairs")
    selected_devices = [devices_by_index[index] for index in selected_indices]
    selected_uuids = [device["uuid"] for device in selected_devices]
    selected_edges = nvlink_edges(topology, selected_indices)
    return {
        "selected_indices": list(selected_indices),
        "selected_uuids": selected_uuids,
        "selected_normalized_uuids": [
            device["normalized_uuid"] for device in selected_devices
        ],
        "selected_nvlink_edges": selected_edges,
        "selection_reason": reason,
        "available_indices": available_indices,
        "original_tokens": original_tokens,
    }


def shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}\n"


def write_env(
    path: Path,
    selection: dict[str, object],
    *,
    original_cuda_visible_devices: str,
    slurm_job_gpus: str,
) -> None:
    selected_indices = ",".join(selection["selected_indices"])
    selected_uuids = ",".join(selection["selected_uuids"])
    lines = [
        shell_export("CUDA_VISIBLE_DEVICES", selected_uuids),
        shell_export(
            "CAMPAIGN_ORIGINAL_CUDA_VISIBLE_DEVICES",
            original_cuda_visible_devices,
        ),
        shell_export("CAMPAIGN_SLURM_JOB_GPUS", slurm_job_gpus),
        shell_export("CAMPAIGN_SELECTED_VISIBLE_GPU_IDS", selected_indices),
        shell_export("CAMPAIGN_SELECTED_VISIBLE_GPU_UUIDS", selected_uuids),
        shell_export(
            "CAMPAIGN_SELECTED_VISIBLE_GPU_NORMALIZED_UUIDS",
            ",".join(selection["selected_normalized_uuids"]),
        ),
        shell_export(
            "CAMPAIGN_SELECTED_VISIBLE_NVLINK_EDGES",
            json.dumps(selection["selected_nvlink_edges"], separators=(",", ":")),
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-env", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    slurm_job_gpus = os.environ.get("SLURM_JOB_GPUS", "")
    raw_identities = run(
        "nvidia-smi",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader,nounits",
    )
    topology = run("nvidia-smi", "topo", "-m")
    devices = parse_visible_gpus(raw_identities)
    payload: dict[str, object] = {
        "schema": "four_a6000_nvlink_pair_selection_v1",
        "original_cuda_visible_devices": original_cuda_visible_devices,
        "slurm_job_gpus": slurm_job_gpus,
        "visible_gpus": devices,
        "topology_diagnostic": topology,
    }
    try:
        selection = choose_selection(devices, topology, original_cuda_visible_devices)
        payload.update(selection)
        payload["passed"] = True
        write_env(
            args.output_env,
            selection,
            original_cuda_visible_devices=original_cuda_visible_devices,
            slurm_job_gpus=slurm_job_gpus,
        )
    except Exception as exc:
        payload["passed"] = False
        payload["error"] = str(exc)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(",".join(selection["selected_uuids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
