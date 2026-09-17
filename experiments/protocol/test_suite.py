"""CPU-only integrity tests for the public reproducibility package."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import analyze_all
from . import analyze_row
from . import audit_runtime_hardware
from . import build_source_freeze
from . import collect_results
from . import monitor_step1000
from . import row_tools
from . import run_activation_row
from . import select_nvlink_gpus
from . import verify_repository as verifier


class PublicReproducibilityTests(unittest.TestCase):
    def test_analysis_keeps_12_layer_token_budgets_in_separate_summaries(self) -> None:
        matrix = json.loads(analyze_all.MATRIX.read_text())
        selected = [matrix["rows"][index] for index in (0, 1, 15)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps({
                "schema": matrix["schema"], "matrix_rows": len(selected), "rows": selected,
            }))
            for source, loss, control_loss, elapsed, control_elapsed in zip(
                selected, (4.0, 6.0, 2.0), (5.0, 7.0, 2.5), (10.0, 30.0, 40.0), (20.0, 60.0, 50.0)
            ):
                result_root = root / "results" / f"row-{source['matrix_index']:02d}"
                result_root.mkdir(parents=True)
                report = {key: source[key] for key in ("matrix_index", "model", "dataset", "seed", "steps")}
                report.update(
                    schema="tiller_matched_endpoint_result_v1", status="complete",
                    candidate_endpoint_loss=loss, control_endpoint_loss=control_loss,
                    absolute_endpoint_lead=control_loss - loss, step1000_absolute_lead=0.1,
                    candidate_end_to_end_total_seconds=elapsed,
                    control_end_to_end_total_seconds=control_elapsed, passes_final_1_05_time_gate=True,
                )
                (result_root / "RESULT.json").write_text(json.dumps(report))
            output = root / "RESULTS.json"
            with (
                patch.object(analyze_all, "MATRIX", matrix_path),
                patch.object(analyze_all, "PACKAGE", root),
                patch.object(sys, "argv", ["analyze_all", "--results-root", str(root / "results"),
                                          "--output", str(output), "--require-complete"]),
                redirect_stdout(io.StringIO()),
            ):
                analyze_all.main()
            result = json.loads(output.read_text())
            summaries = {row["train_tokens"]: row for row in result["dataset_summaries"]}
            self.assertEqual(len(result["dataset_summaries"]), 2)
            self.assertEqual(set(summaries), {100000000, 300000000})
            small, large = summaries[100000000], summaries[300000000]
            self.assertEqual((small["model"], small["dataset"]), (large["model"], large["dataset"]))
            self.assertEqual((small["phase"], small["steps_required"]),
                             ("12l_768d_100m_tokens_3050_steps", 3050))
            self.assertEqual((large["phase"], large["steps_required"]),
                             ("12l_768d_300m_tokens_9150_steps", 9150))
            self.assertEqual(small["completed_seeds"], [1337, 2027])
            self.assertEqual(large["completed_seeds"], [1337])
            self.assertEqual((small["tiller_endpoint_loss_mean"], large["tiller_endpoint_loss_mean"]), (5.0, 2.0))
            self.assertEqual((small["control_endpoint_loss_mean"], large["control_endpoint_loss_mean"]), (6.0, 2.5))
            self.assertEqual(small["exact_matched_hardware_endpoint_total_time_ratio"], 0.5)
            self.assertEqual(large["exact_matched_hardware_endpoint_total_time_ratio"], 0.8)
            self.assertEqual(result["complete_rows"], 3)
            self.assertEqual(result["status"], "complete")


    def test_manifest_has_exact_matched_activation_pairs(self) -> None:
        rows = verifier.verify_manifest()
        self.assertEqual(len(rows), 640)
        self.assertEqual(
            {row["activation"] for row in rows}, {"silu", verifier.GRAIN_ID}
        )
        self.assertEqual(
            {row["model"] for row in rows},
            {verifier.SMALL_MODEL, verifier.LARGE_MODEL},
        )
        self.assertEqual(
            {row["train_tokens"] for row in rows},
            {"2621440", "100000000", "300000000"},
        )

    def test_transfer_matrix_is_exactly_derived_from_manifest(self) -> None:
        manifest = verifier.verify_manifest()
        payload = verifier.verify_matrix(manifest)
        self.assertEqual(len(payload["rows"]), 60)
        self.assertEqual(
            sum(row["phase"] == verifier.SUITE_100M for row in payload["rows"]), 15
        )
        self.assertEqual(
            sum(row["phase"] == verifier.SUITE_300M_SMALL for row in payload["rows"]), 15
        )
        self.assertEqual(
            sum(row["phase"] == verifier.SUITE_100M_LARGE for row in payload["rows"]), 30
        )

    def test_token_fingerprint_manifest_covers_each_dataset_and_budget(self) -> None:
        payload = verifier.verify_token_fingerprints()
        self.assertEqual(len(payload["cells"]), 10)

    def test_real_launcher_keeps_every_shared_argument_pairwise_identical(self) -> None:
        payload = verifier.verify_matrix(verifier.verify_manifest())
        verifier.verify_launcher_pairing(payload)

    def test_primary_tiller_arms_disable_only_grain_diagnostics(self) -> None:
        primary = analyze_row.suite.row_at(30)
        self.assertEqual(primary["phase"], analyze_row.suite.PRIMARY_PHASE)
        for arm in ("control", "candidate"):
            argv = analyze_row.suite.training_argv(primary, arm, Path("runs"))
            self.assertEqual(argv.count("--no-grain-training-telemetry"), 1)
            self.assertIs(
                analyze_row.suite.expected_args(primary, arm)["grain_training_telemetry"],
                False,
            )
        historical = analyze_row.suite.row_at(0)
        self.assertNotEqual(historical["phase"], analyze_row.suite.PRIMARY_PHASE)
        self.assertNotIn(
            "--no-grain-training-telemetry",
            analyze_row.suite.training_argv(historical, "candidate", Path("runs")),
        )
        self.assertIs(
            analyze_row.suite.expected_args(historical, "candidate")["grain_training_telemetry"],
            True,
        )

    def test_baseline_launcher_enables_the_row_aware_lr_wd_contract(self) -> None:
        rows = verifier.verify_manifest()
        verifier.verify_manifest_launcher_contract(rows)
        lion = next(
            row
            for row in rows
            if row["phase"] == verifier.SUITE_100M_LARGE
            and row["optimizer"] == "lion"
        )
        argv = run_activation_row.training_arguments(lion, Path("runs"))
        self.assertEqual(
            argv[argv.index("--fairness-contract") + 1],
            verifier.EXACT_LR_WD_CONTRACT,
        )
        self.assertEqual(argv[argv.index("--lr") + 1], lion["lr"])
        self.assertEqual(argv[argv.index("--min-lr") + 1], lion["min_lr"])
        self.assertEqual(
            argv[argv.index("--weight-decay") + 1], lion["weight_decay"]
        )

    def test_primary_campaign_stages_partition_the_full_matrix(self) -> None:
        verifier.verify_stage_launchers()
        phase = verifier.SUITE_100M_LARGE
        muon = row_tools.manifest_stage_indices(
            verifier.MANIFEST, phase, "muon"
        )
        adamw = row_tools.manifest_stage_indices(
            verifier.MANIFEST, phase, "adamw"
        )
        remaining = row_tools.manifest_stage_indices(
            verifier.MANIFEST, phase, "remaining"
        )
        all_rows = row_tools.manifest_stage_indices(
            verifier.MANIFEST, phase, "all"
        )
        self.assertEqual((len(muon), len(adamw), len(remaining)), (30, 30, 150))
        self.assertEqual(set(muon) | set(adamw) | set(remaining), set(all_rows))
        self.assertFalse(set(muon) & set(adamw))
        self.assertFalse(set(muon) & set(remaining))
        self.assertFalse(set(adamw) & set(remaining))
        self.assertEqual(len(row_tools.matrix_suite_indices(phase)), 15)
        hybrid = row_tools.matrix_suite_indices(phase, optimizer="tiller_then_muon_v1")
        self.assertEqual(len(hybrid), 15)
        self.assertFalse(set(hybrid) & set(row_tools.matrix_suite_indices(phase)))

    def test_published_remaining_stage_selects_recorded_baselines(self) -> None:
        indices = row_tools.manifest_stage_indices(
            verifier.MANIFEST, verifier.SUITE_100M_LARGE, "published_remaining"
        )
        rows = verifier.verify_manifest()
        selected = [row for row in rows if int(row["row_index"]) in indices]
        self.assertEqual(len(indices), 90)
        self.assertEqual(len(selected), 90)
        self.assertEqual(
            {row["optimizer"] for row in selected},
            {"soap_adamw", "adafactor_came", "schedule_free_adamw"},
        )

    @staticmethod
    def _gpu_topology(count: int, pairs: set[tuple[int, int]]) -> str:
        header = "        " + " ".join(f"GPU{index}" for index in range(count))
        normalized_pairs = {tuple(sorted(pair)) for pair in pairs}
        rows = []
        for left in range(count):
            values = []
            for right in range(count):
                if left == right:
                    values.append("X")
                elif tuple(sorted((left, right))) in normalized_pairs:
                    values.append("NV4")
                else:
                    values.append("PHB")
            rows.append(f"GPU{left}    " + "  ".join(values))
        return header + "\n" + "\n".join(rows) + "\n"

    @staticmethod
    def _selector_devices(count: int = 8) -> tuple[list[dict[str, str]], list[str]]:
        uuids = [
            f"GPU-00000000-0000-0000-0000-{10 + index:012x}"
            for index in range(count)
        ]
        rows = "\n".join(
            f"{index}, {uuid}, NVIDIA RTX A6000"
            for index, uuid in enumerate(uuids)
        )
        return select_nvlink_gpus.parse_visible_gpus(rows), uuids

    def test_nvlink_selector_prefers_shuffled_original_first_four_and_emits_exact_uuids(self) -> None:
        devices, uuids = self._selector_devices()
        topology = self._gpu_topology(8, {(0, 1), (2, 3), (4, 5), (6, 7)})
        original_order = [uuids[index] for index in (4, 5, 6, 7, 0, 1, 2, 3)]
        selection = select_nvlink_gpus.choose_selection(
            devices, topology, ",".join(original_order)
        )
        self.assertEqual(selection["selected_indices"], ["4", "5", "6", "7"])
        self.assertEqual(selection["selection_reason"], "original_first_four")
        self.assertEqual(selection["selected_uuids"], original_order[:4])
        self.assertEqual(
            selection["selected_normalized_uuids"],
            [uuid.upper() for uuid in original_order[:4]],
        )
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / "selection.env"
            select_nvlink_gpus.write_env(
                env_path,
                selection,
                original_cuda_visible_devices=",".join(original_order),
                slurm_job_gpus="0,1,2,3,4,5,6,7",
            )
            env_text = env_path.read_text()
        expected_visible = ",".join(original_order[:4])
        self.assertIn(f"export CUDA_VISIBLE_DEVICES={expected_visible}\n", env_text)
        self.assertIn(
            f"export CAMPAIGN_SELECTED_VISIBLE_GPU_UUIDS={expected_visible}\n",
            env_text,
        )

    def test_nvlink_selector_skips_bad_first_four_when_other_pairs_are_valid(self) -> None:
        devices, _ = self._selector_devices()
        topology = self._gpu_topology(8, {(0, 1), (4, 5)})
        selection = select_nvlink_gpus.choose_selection(
            devices, topology, "0,2,3,4,1,5,6,7"
        )
        self.assertEqual(selection["selected_indices"], ["0", "1", "4", "5"])
        self.assertEqual(selection["selection_reason"], "deterministic_valid_pairs")

    def test_nvlink_selector_rejects_one_pair_and_does_not_escape_original_cvd(self) -> None:
        devices, _ = self._selector_devices()
        topology = self._gpu_topology(8, {(0, 1), (4, 5), (6, 7)})
        with self.assertRaisesRegex(
            RuntimeError,
            "no visible four-A6000 selection has two disjoint NVLink pairs",
        ):
            select_nvlink_gpus.choose_selection(devices, topology, "0,1,2,3")

    def test_hardware_audit_requires_exclusive_job_and_two_nvlink_pairs(self) -> None:
        torch_uuids = [
            "18d5b80b-03bb-a57d-4f2e-c3102aea8463",
            "ec0281f6-aab7-d36a-ab7c-24bef411856c",
            "8ea58d5b-8e21-c9e3-677d-96794c546098",
            "4c08f6ff-0c8a-70a9-6736-68eca9a15311",
        ]
        nvidia_smi_identities = "\n".join(
            f"{index}, GPU-{uuid.upper()}"
            for index, uuid in enumerate(torch_uuids)
        )
        two_pair_topology = """\
        GPU0 GPU1 GPU2 GPU3
GPU0    X    NV4  PHB  PHB
GPU1    NV4  X    PHB  PHB
GPU2    PHB  PHB  X    NV4
GPU3    PHB  PHB  NV4  X
"""
        one_pair_topology = """\
        GPU0 GPU1 GPU2 GPU3
GPU0    X    NV4  PHB  PHB
GPU1    NV4  X    PHB  PHB
GPU2    PHB  PHB  X    PIX
GPU3    PHB  PHB  PIX  X
"""

        def run_case(
            job_over_subscribe: str,
            partition_over_subscribe: str,
            *,
            visible_tokens: str = "0,1,2,3",
            topology: str = two_pair_topology,
        ) -> dict:
            job_description = (
                "JobId=123 Partition=gpu ReqNodeList=(null) ExcNodeList=(null) "
                "Features=nvlink "
                "ReqTRES=cpu=16,mem=128G,node=1,billing=16,gres/gpu=4,"
                "gres/gpu:nvidia_rtx_a6000=4 "
                "AllocTRES=cpu=64,mem=512G,node=1,billing=64,gres/gpu=8,"
                "gres/gpu:nvidia_rtx_a6000=8 "
                f"OverSubscribe={job_over_subscribe} "
                "CpusPerTres=gres/gpu:4"
            )
            partition_description = (
                f"PartitionName=gpu OverSubscribe={partition_over_subscribe} State=UP"
            )

            def fake_run(*args: str) -> str:
                if args == ("scontrol", "show", "job", "123", "-o"):
                    return job_description
                if args == ("scontrol", "show", "partition", "gpu", "-o"):
                    return partition_description
                if args == ("nvidia-smi", "topo", "-m"):
                    return topology
                if args == (
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                ):
                    return nvidia_smi_identities
                raise AssertionError(args)

            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "hardware.json"
                environment = {
                    "SLURM_JOB_ID": "123",
                    "SLURMD_NODENAME": "node-a",
                    "CUDA_VISIBLE_DEVICES": visible_tokens,
                    "SLURM_JOB_GPUS": "2,3,4,5",
                    "CAMPAIGN_SLURM_JOB_GPUS": "2,3,4,5",
                    "CAMPAIGN_SELECTED_VISIBLE_GPU_IDS": "0,1,2,3",
                    "NCCL_P2P_DISABLE": "0",
                    "NCCL_P2P_LEVEL": "NVL",
                    "NCCL_SHM_DISABLE": "0",
                    "RATIONAL_OPT_TORCH_FALLBACK": "0",
                }
                with (
                    patch.object(sys, "argv", ["audit_runtime_hardware", "--output", str(output)]),
                    patch.dict(os.environ, environment, clear=False),
                    patch.object(audit_runtime_hardware, "run", side_effect=fake_run),
                    patch.object(
                        audit_runtime_hardware.subprocess,
                        "run",
                        return_value=SimpleNamespace(
                            stdout="NVLink status fixture", returncode=0
                        ),
                    ),
                    patch.object(
                        audit_runtime_hardware.torch.cuda,
                        "device_count",
                        return_value=4,
                    ),
                    patch.object(
                        audit_runtime_hardware.torch.cuda,
                        "get_device_name",
                        return_value="NVIDIA RTX A6000",
                    ),
                    patch.object(
                        audit_runtime_hardware.torch.cuda,
                        "get_device_properties",
                        side_effect=[
                            SimpleNamespace(uuid=uuid) for uuid in torch_uuids
                        ],
                    ),
                    patch.object(
                        audit_runtime_hardware.torch.cuda,
                        "can_device_access_peer",
                        side_effect=lambda i, j: (i, j)
                        in {(0, 1), (1, 0), (2, 3), (3, 2)},
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    audit_runtime_hardware.main()
                return json.loads(output.read_text())

        cases = ("0,1,2,3", ",".join(f"GPU-{uuid}" for uuid in torch_uuids))
        for visible_tokens in cases:
            with self.subTest(
                visible_tokens=visible_tokens,
            ):
                payload = run_case(
                    "NO", "NO", visible_tokens=visible_tokens
                )
                self.assertTrue(payload["passed"])
                self.assertEqual(payload["partition_over_subscribe"], "NO")
                self.assertEqual(payload["job_over_subscribe"], "NO")
                self.assertEqual(payload["slurm_job_gpus"], "2,3,4,5")
                self.assertIn("gres/gpu=8", payload["allocated_tres"])
                self.assertEqual(payload["selected_visible_gpu_indices"], ["0", "1", "2", "3"])
                self.assertEqual(
                    payload["selected_visible_gpu_token_indices"],
                    ["0", "1", "2", "3"],
                )
                self.assertEqual(
                    payload["selected_visible_gpu_uuids"],
                    [f"GPU-{uuid.upper()}" for uuid in torch_uuids],
                )
                self.assertTrue(payload["selected_visible_has_two_disjoint_nvlink_pairs"])

        with self.assertRaisesRegex(RuntimeError, "contract failed"):
            run_case("NO", "YES")
        with self.assertRaisesRegex(RuntimeError, "contract failed"):
            run_case("OK", "NO")
        with self.assertRaisesRegex(RuntimeError, "contract failed"):
            run_case("NO", "NO", topology=one_pair_topology)

    def test_activation_rerun_requires_exact_row_and_source_identity(self) -> None:
        row = run_activation_row.read_row(verifier.MANIFEST, 10)
        identity = {
            "source_manifest_sha256": "1" * 64,
            "source_manifest_row_sha256": run_activation_row.row_sha256(row),
            "source_manifest_row_id": row["row_id"],
            "source_manifest_row_index": row["row_index"],
            "source_freeze_sha256": "2" * 64,
        }
        config = {
            "event": "config",
            **identity,
            "experiment_identity": "none",
            "activation": row["activation"],
            "optimizer": row["optimizer"],
            "dataset": row["dataset_name"],
            "dataset_config": row["dataset_config"],
            "dataset_revision": row["dataset_revision"],
            "tokenizer": row["tokenizer"],
            "tokenizer_revision": row["tokenizer_revision"],
            "train_tokens": int(row["train_tokens"]),
            "val_tokens": int(row["val_tokens"]),
            "steps": int(row["steps"]),
            "layers": int(row["layers"]),
            "d_model": int(row["d_model"]),
            "heads": int(row["heads"]),
            "ffn_dim": int(row["ffn_dim"]),
            "batch_size_per_gpu": int(row["batch_size"]),
            "grad_accum": int(row["grad_accum"]),
            "global_tokens_per_step": int(row["global_tokens_per_step"]),
            "warmup_steps": int(row["warmup_steps"]),
            "optimizer_lr": float(row["lr"]),
            "optimizer_min_lr": float(row["min_lr"]),
            "optimizer_weight_decay": float(row["weight_decay"]),
            "optimizer_beta1": float(row["beta1"]),
            "optimizer_beta2": float(row["beta2"]),
            "optimizer_eps": float(row["eps"]),
            "grad_clip": float(row["grad_clip"]),
            "init_std": float(row["init_std"]),
            "optimizer_lr_wd_fairness": {
                "contract": run_activation_row.EXACT_LR_WD_CONTRACT,
                "passed": True,
                "base_lr": float(row["lr"]),
                "minimum_lr": float(row["min_lr"]),
                "base_weight_decay": float(row["weight_decay"]),
            },
        }
        records = (
            config,
            {"event": "eval", "step": int(row["steps"])},
            {
                "event": "summary",
                "completed_steps": int(row["steps"]),
                "stopped_early": False,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            self.assertTrue(
                run_activation_row.matches_source_identity(path, row, identity)
            )
            changed = dict(identity, source_freeze_sha256="3" * 64)
            self.assertFalse(
                run_activation_row.matches_source_identity(path, row, changed)
            )
            config["optimizer_lr"] = float(row["lr"]) * 2
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            self.assertFalse(
                run_activation_row.matches_source_identity(path, row, identity)
            )

    def test_tiller_completion_requires_its_exact_matrix_identity(self) -> None:
        row = analyze_row.suite.row_at(0)
        config = {
            "event": "config",
            **analyze_row.expected_config(row, "candidate"),
            "optimizer_lr_wd_fairness": {
                "contract": analyze_row.suite.contract(row),
                "passed": True,
                "base_lr": row["lr"],
                "minimum_lr": row["min_lr"],
                "base_weight_decay": row["weight_decay"],
                "groups": [],
                "internal_lr_wd_scalars": {},
            },
            "tiller_experiment_identity": {
                "passed": True,
                "matrix_index": row["matrix_index"],
                "source_manifest_row_index": row["source_manifest_row_index"],
                "source_manifest_row_id": row["source_manifest_row_id"],
            },
        }
        endpoint = int(row["steps"])
        run_records = [
            config,
            {"event": "eval", "step": 1000, "val_loss": 4.2},
            {"event": "eval", "step": endpoint, "val_loss": 4.0},
            {
                "event": "summary",
                "completed_steps": endpoint,
                "stopped_early": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in run_records)
            )
            _, evals, _ = analyze_row.audited(path, row, "candidate")
            self.assertEqual(evals[endpoint], 4.0)

            config["experiment_identity"] = "none"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in run_records)
            )
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                analyze_row.audited(path, row, "candidate")

    def test_primary_analyzers_reject_diagnostic_grain_telemetry(self) -> None:
        row = analyze_row.suite.row_at(30)
        config = {
            "event": "config",
            **analyze_row.expected_config(row, "candidate"),
            "grain_live_stats_scope": "telemetry_only",
            "optimizer_lr_wd_fairness": {
                "contract": analyze_row.suite.contract(row),
                "passed": True,
                "base_lr": row["lr"],
                "minimum_lr": row["min_lr"],
                "base_weight_decay": row["weight_decay"],
                "groups": [],
                "internal_lr_wd_scalars": {},
            },
            "tiller_experiment_identity": {
                "passed": True,
                "matrix_index": row["matrix_index"],
                "source_manifest_row_index": row["source_manifest_row_index"],
                "source_manifest_row_id": row["source_manifest_row_id"],
            },
        }
        endpoint = int(row["steps"])
        records = [
            config,
            {"event": "eval", "step": 1000, "val_loss": 4.2},
            {"event": "eval", "step": endpoint, "val_loss": 4.0},
            {"event": "summary", "completed_steps": endpoint, "stopped_early": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(RuntimeError, "grain_live_stats_scope"):
                analyze_row.audited(path, row, "candidate")
            with self.assertRaisesRegex(RuntimeError, "grain_live_stats_scope"):
                monitor_step1000.audited_config(path, row, "candidate")

    def test_result_tables_have_valid_schemas_statuses_and_unique_cells(self) -> None:
        results = verifier.verify_results()
        self.assertEqual(set(results), set(verifier.OPTIMIZERS))
        self.assertTrue(results["tiller"]["checkpoints"])

    def test_public_layout_contains_only_the_tiller_optimizer_surface(self) -> None:
        inventory = verifier.verify_public_layout()
        optimizer_roots = {
            path for path in inventory
            if path.startswith("optimizer_design/") and path.count("/") == 1
        }
        self.assertEqual(
            optimizer_roots,
            {
                "optimizer_design/README.md",
                "optimizer_design/__init__.py",
                "optimizer_design/tiller.py",
                "optimizer_design/tiller_then_muon.py",
            },
        )

    def test_tiller_scaling_matches_reported_models_and_is_position_invariant(self) -> None:
        from optimizer_design import tiller_scaling_formula

        cases = (
            ((12, 12, 3072, 768), (144, 9508, 366398, 13824)),
            ((18, 18, 4608, 1024), (324, 21388, 976394, 31104)),
            ((96, 64, 24576, 8192), (6144, 405508, 110788838, 589824)),
        )
        for shape, expected in cases:
            layers, groups, intermediate_width, model_width = shape
            observed = []
            for positions in (1, 1_050_000):
                report = tiller_scaling_formula(
                    total_positions=positions,
                    total_layers=layers,
                    total_groups=groups,
                    intermediate_width=intermediate_width,
                    model_width=model_width,
                )
                observed.append(
                    (
                        report["coordinate_count"],
                        report["ledger_checkpoint_tensor_elements"],
                        report["persistent_state_elements"],
                        report["maximum_live_factor_elements"],
                    )
                )
                self.assertEqual(report["largest_dense_solve_dimension"], 96)
                self.assertEqual(
                    report["state_depends_on_total_activation_positions"], 0
                )
            self.assertEqual(observed, [expected, expected])

    def test_public_files_do_not_embed_workspace_paths(self) -> None:
        inventory = build_source_freeze.public_inventory()
        verifier.verify_no_absolute_workspace_paths(inventory)

    def test_reader_facing_files_use_descriptive_suite_names(self) -> None:
        inventory = build_source_freeze.public_inventory()
        verifier.verify_public_suite_names(inventory)

    def test_quick_verification_does_not_depend_on_results_or_source_freeze(self) -> None:
        labels = dict(verifier.verify_repository(quick=True))
        self.assertEqual(labels["activation/optimizer manifest"], 640)
        self.assertEqual(labels["TILLER transfer matrix"], 60)
        self.assertNotIn("optimizer result directories", labels)
        self.assertNotIn("frozen source files", labels)

    def test_source_freeze_scope_excludes_results_logs_and_binaries(self) -> None:
        paths = build_source_freeze.frozen_paths()
        self.assertIn("experiments/protocol/activation_optimizer_manifest.csv", paths)
        self.assertIn("experiments/protocol/matrix.json", paths)
        self.assertIn("experiments/protocol/token_fingerprints.json", paths)
        self.assertIn("optimizer_design/tiller.py", paths)
        self.assertIn("training/train.py", paths)
        self.assertFalse(any(path.startswith("experiments/results/") for path in paths))
        self.assertFalse(
            any(
                path.endswith(tuple(build_source_freeze.EXCLUDED_SUFFIXES))
                for path in paths
            )
        )

    def test_source_freeze_hashes_match_public_sources(self) -> None:
        entries = verifier.verify_source_freeze()
        self.assertTrue(entries)
        self.assertEqual(
            verifier.FREEZE.read_text(encoding="utf-8"),
            build_source_freeze.render_freeze(),
        )

    def test_source_freeze_parser_rejects_malformed_and_duplicate_entries(self) -> None:
        malformed = "not-a-digest  training/train.py\n"
        with self.assertRaises(verifier.VerificationError):
            verifier.parse_freeze(malformed)

        digest = "0" * 64
        duplicate = f"{digest}  training/train.py\n{digest}  training/train.py\n"
        with self.assertRaises(verifier.VerificationError):
            verifier.parse_freeze(duplicate)

    def test_verifier_and_freeze_check_run_without_cuda_imports(self) -> None:
        commands = (
            [sys.executable, str(verifier.__file__)],
            [sys.executable, str(build_source_freeze.__file__), "--check"],
        )
        for command in commands:
            with self.subTest(command=command):
                completed = subprocess.run(
                    command,
                    cwd=verifier.ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )

    def test_direct_verifier_bootstraps_the_repository_package(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, str(verifier.__file__), "--quick"],
            cwd=verifier.ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode, 0, completed.stdout + completed.stderr
        )


if __name__ == "__main__":
    unittest.main()
