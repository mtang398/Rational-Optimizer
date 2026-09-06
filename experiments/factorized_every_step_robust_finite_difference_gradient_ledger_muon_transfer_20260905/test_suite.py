from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from optimizer_design.rlb_factorized_every_step_rfd_gradient_ledger_muon import (
    factorized_every_step_rfd_gradient_ledger_scaling_formula,
)
from training import transformer_lm_compare as trainer

from . import suite
from .audit_runtime_hardware import nvlink_peer_map
from .row_tools import terminal


def rows():
    return suite.matrix_payload()["rows"]


def test_exact_matrix_inventory_and_scale_specific_controls():
    assert len(rows()) == 24
    assert {(r["dataset"], r["seed"]) for r in rows() if r["model"] == "M0"} == {
        (dataset, seed)
        for dataset in ("dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en")
        for seed in (1337, 2027, 3407)
    }
    assert {(r["dataset"], r["seed"]) for r in rows() if r["model"] == "M1"} == {
        (dataset, seed)
        for dataset in ("dclm", "fineweb_edu", "c4_en")
        for seed in (1337, 2027, 3407)
    }
    for row in rows():
        assert row["lr"] == 3e-4
        assert row["min_lr"] == 3e-5
        assert row["warmup_steps"] == 200
        assert row["weight_decay"] == 0.1
        assert (row["beta1"], row["beta2"], row["eps"], row["grad_clip"]) == (
            0.9, 0.95, 1e-8, 1.0
        )
        assert row["telemetry_rlb_stat_every"] == 4
        if row["model"] == "M0":
            assert (row["steps"], row["max_train_tokens"]) == (3050, 100_000_000)
            assert row["control_optimizer"] == "adamw"
            assert (row["probe_batch_size"], row["matrix_spectrum_interval"]) == (1, 250)
        else:
            assert (row["steps"], row["max_train_tokens"]) == (9150, 300_000_000)
            assert row["control_optimizer"] == "muon"
            assert (row["probe_batch_size"], row["matrix_spectrum_interval"]) == (0, 0)


def parse(argv):
    old = sys.argv
    try:
        sys.argv = argv
        return trainer.parse_args()
    finally:
        sys.argv = old


def test_all_parsed_shared_arguments_are_pairwise_identical(tmp_path):
    trainer.ACTIVE_OPTIMIZERS = sorted(
        set(trainer.ACTIVE_OPTIMIZERS) | {suite.EXACT_OPTIMIZER_KEY}
    )
    allowed = {"activation", "optimizer", "run_name"}
    for row in rows():
        control = parse(suite.training_argv(row, "control", tmp_path))
        candidate = parse(suite.training_argv(row, "candidate", tmp_path))
        suite.verify_args(control, row, "control")
        suite.verify_args(candidate, row, "candidate")
        control_values = vars(control)
        candidate_values = vars(candidate)
        assert control_values.keys() == candidate_values.keys()
        differences = {
            key for key in control_values
            if control_values[key] != candidate_values[key]
        }
        assert differences == allowed


def test_all_48_trajectory_paths_are_unique(tmp_path):
    paths = {
        suite.jsonl_path(row, arm, tmp_path)
        for row in rows()
        for arm in ("control", "candidate")
    }
    assert len(paths) == 48


def test_terminal_requires_one_complete_trajectory(tmp_path):
    path = tmp_path / "run.jsonl"
    records = [
        {"event": "config"},
        {"event": "eval", "step": 3050},
        {"event": "summary", "completed_steps": 3050, "stopped_early": False},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert terminal(path, 3050)
    path.write_text(path.read_text() + json.dumps({"event": "config"}) + "\n")
    assert not terminal(path, 3050)


def test_launcher_is_unpinned_four_a6000_nvlink_p2p_endpoint_pair():
    launcher = (suite.PACKAGE / "run_quality_row.sbatch").read_text()
    required = (
        "#SBATCH --exclusive",
        "#SBATCH --constraint=nvlink",
        "#SBATCH --gres=gpu:nvidia_rtx_a6000:4",
        "#SBATCH --cpus-per-gpu=4",
        "NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0",
        "audit_runtime_hardware",
        "control_endpoint",
        "candidate_endpoint_step1000_screen",
    )
    assert all(marker in launcher for marker in required)
    assert "#SBATCH --nodelist=" not in launcher
    assert "#SBATCH --exclude=" not in launcher
    assert "benchmark_runtime" not in launcher
    assert "gate_runtime" not in launcher


def test_nvlink_audit_parses_ansi_header_and_pairwise_a6000_topology():
    topology = """\
\x1b[4mGPU0 GPU1 GPU2 GPU3 CPU Affinity\x1b[0m
GPU0 X NODE NODE NV4 0-31
GPU1 NODE X NV4 PIX 0-31
GPU2 NODE NV4 X PIX 0-31
GPU3 NV4 PIX PIX X 0-31
"""
    assert nvlink_peer_map(topology, ["0", "1", "2", "3"]) == {
        "0": True,
        "1": True,
        "2": True,
        "3": True,
    }


def test_scaling_is_activation_position_invariant_and_owner_free():
    dimensions = dict(
        total_layers=96,
        total_groups=64,
        intermediate_width=24576,
        model_width=8192,
    )
    small = factorized_every_step_rfd_gradient_ledger_scaling_formula(
        total_positions=1, **dimensions
    )
    large = factorized_every_step_rfd_gradient_ledger_scaling_formula(
        total_positions=1_050_000, **dimensions
    )
    assert small["persistent_state_elements"] == large["persistent_state_elements"]
    assert large["state_depends_on_total_activation_positions"] == 0
    assert large["owner_count"] == 0
    assert large["complete_layer_owners"] == 0
    assert large["owner_local_mathematics"] == 0
    assert large["dense_lg_by_lg_metric_elements"] == 0
    assert large["selected_update_elements_published"] == 0
    assert large["largest_dense_solve_dimension"] == 96
    assert large["state_scales_as"] == "O(LH + LGd + 64LG)"


def test_full_model_lr_wd_audit_passed():
    payload = json.loads((suite.PACKAGE / "FULL_MODEL_FAIRNESS.json").read_text())
    assert payload["passed"] is True
    assert {(item["model"], item["arm"]) for item in payload["reports"]} == {
        ("M0", "control"), ("M0", "candidate"),
        ("M1", "control"), ("M1", "candidate"),
    }


def test_source_port_manifest_is_exact_and_legacy_method_is_absent():
    root = suite.PACKAGE.parents[1]
    manifest = json.loads((suite.PACKAGE / "EXACT_METHOD_SOURCE.json").read_text())
    import hashlib

    for item in manifest["files"]:
        path = root / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    forbidden = ("matrix" + "policy").lower()
    forbidden_spaced = ("matrix" + " policy").lower()
    inventory = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    for relative in inventory:
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.suffix == ".pyc":
            continue
        assert forbidden not in path.name.lower()
        if path.stat().st_size <= 32 * 1024 * 1024:
            payload = path.read_bytes().lower()
            assert forbidden.encode() not in payload
            assert forbidden_spaced.encode() not in payload
