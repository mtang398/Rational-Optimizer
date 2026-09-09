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
from unittest.mock import patch

from . import analyze_all
from . import analyze_row
from . import build_source_freeze
from . import collect_results
from . import row_tools
from . import run_activation_row
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

    def factorial_tiller_fixture(self, root: Path, index: int = 30) -> dict:
        public = json.loads(collect_results.DEFAULT_MATRIX.read_text())["rows"]
        staged = []
        for source in public:
            if source["phase"] != collect_results.FACTORIAL_PHASE:
                continue
            number = int(source["matrix_index"]) + 180
            row = dict(
                source, matrix_index=number, row_id=f"factorial-{number}",
                optimizer="tiller_v1", activation=collect_results.GRAIN_ID,
                train_tokens=source["max_train_tokens"], val_tokens=source["max_val_tokens"],
                expected_parameter_count=296871080, scientific_cell_key=f"cell-{number}",
                token_fingerprints={"train_token_sample_sha256": "train-tokens",
                                    "validation_token_sample_sha256": "validation-tokens"},
            )
            staged.append(row)
            staged.append(dict(row, matrix_index=number - 200,
                               row_id=f"control-{number}", optimizer="muon", activation="silu"))
        (root / "matrix.json").write_text(json.dumps({"rows": staged}))
        freeze = root / "SOURCE_FREEZE.sha256"
        freeze.write_text("synthetic frozen source identity\n")
        target = next(row for row in staged if row["matrix_index"] == index + 180)
        artifact = root / "runs" / target["row_id"] / f"{target['activation']}.jsonl"
        artifact.parent.mkdir(parents=True)
        config = {
            key: target[key] for key in (
                "seed", "layers", "d_model", "heads", "ffn_dim", "seq_len", "steps",
                "train_tokens", "val_tokens", "activation", "dataset_config",
            )
        }
        config.update(
            event="config", dataset=target["dataset_name"],
            optimizer="factorized_every_step_rfd_gradient_ledger_muon_v1",
            params=target["expected_parameter_count"],
            train_token_sample_sha256="train-tokens", val_token_sample_sha256="validation-tokens",
            m1_300m_campaign_identity={"passed": True, "scientific_cell_key": target["scientific_cell_key"]},
            optimizer_lr_wd_fairness={"passed": True},
        )
        records = [config,
                   {"event": "eval", "step": 1000, "val_loss": 4.0},
                   {"event": "eval", "step": 9150, "val_loss": 3.0},
                   {"event": "summary", "completed_steps": 9150, "stopped_early": False,
                    "total_seconds": 20.0, "realized_lr_trace_sha256": "new-trace"}]
        artifact.write_text("".join(json.dumps(row) + "\n" for row in records))
        control = next(row for row in staged if row["row_id"] == f"control-{index + 180}")
        control_path = root / "runs" / control["row_id"] / "silu.jsonl"
        control_path.parent.mkdir(parents=True)
        control_path.write_text(json.dumps({"event": "eval", "step": 1000, "val_loss": 4.5}) + "\n")
        result = root / "results/02_tiller" / f"matrix-{index + 180}"
        result.mkdir(parents=True)
        timing = {
            "schema": "tiller_endpoint_process_wall_clock_v1", "process_exit_status": 0,
            "elapsed_seconds": 25.0, "stage": "02_tiller", "matrix_index": index + 180,
            "row_id": target["row_id"], "artifact_path": str(artifact), "artifact_exists": True,
            "artifact_bytes": artifact.stat().st_size, "artifact_sha256": collect_results.sha256(artifact),
            "source_freeze_sha256": collect_results.sha256(freeze),
        }
        timing_path = result / "WALL_CLOCK.json"
        timing_path.write_text(json.dumps(timing))
        decision = {
            "schema": "tiller_exact_matched_step1000_gate_v1", "passed": True,
            "scientific_stop": False, "dataset": target["dataset"], "seed": target["seed"],
            "candidate_matrix_index": index + 180, "control_matrix_index": control["matrix_index"],
            "candidate_path": str(artifact), "control_path": str(control_path),
            "candidate_step1000_loss": 4.0, "control_step1000_loss": 4.5, "candidate_lead": 0.5,
        }
        decision_path = result / "STEP1000.json"
        decision_path.write_text(json.dumps(decision))
        output = collect_results.REPOSITORY / "experiments/results/tiller"
        with (output / "runs.csv").open(newline="") as handle:
            published = list(csv.DictReader(handle))
        with (output / "checkpoints.csv").open(newline="") as handle:
            checkpoints = list(csv.DictReader(handle))
        return dict(staged=staged, target=target, artifact=artifact, records=records,
                    timing=timing, timing_path=timing_path, decision=decision,
                    decision_path=decision_path, published=published, checkpoints=checkpoints)

    def test_factorial_tiller_endpoint_replaces_quality_time_and_checkpoints_together(self) -> None:
        for index in (30, 35, 44):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.factorial_tiller_fixture(root, index)
                rows, checkpoints = collect_results.factorial_tiller_runs(
                    collect_results.DEFAULT_MATRIX, root, fixture["published"], fixture["checkpoints"]
                )
                result = next(row for row in rows if int(row["run_index"]) == index)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["final_validation_loss"], 3.0)
                self.assertEqual(result["step1000_validation_loss"], 4.0)
                self.assertEqual(result["total_seconds"], 25.0)
                self.assertEqual(result["training_loop_total_seconds"], 20.0)
                self.assertEqual(result["time_scope"], "end_to_end_process")
                self.assertEqual(result["source_jsonl"], "")
                self.assertEqual(result["source_jsonl_sha256"], collect_results.sha256(fixture["artifact"]))
                self.assertEqual([(row["step"], row["validation_loss"]) for row in checkpoints
                                  if int(row["run_index"]) == index], [(1000, 4.0), (9150, 3.0)])
                self.assertEqual([row for row in rows if int(row["run_index"]) != index],
                                 [row for row in fixture["published"] if int(row["run_index"]) != index])

    def test_factorial_tiller_partial_failed_or_stopped_retry_keeps_historical_attempt(self) -> None:
        for mode in ("partial", "failed", "missing_timing", "missing_decision", "scientific_stop"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.factorial_tiller_fixture(root)
                if mode == "partial":
                    fixture["artifact"].write_text(json.dumps({"event": "eval", "step": 1000, "val_loss": 8.0}))
                elif mode == "failed":
                    fixture["timing"]["process_exit_status"] = 1
                    fixture["timing_path"].write_text(json.dumps(fixture["timing"]))
                elif mode == "missing_timing":
                    fixture["timing_path"].unlink()
                elif mode == "missing_decision":
                    fixture["decision_path"].unlink()
                else:
                    fixture["decision"].update(passed=False, scientific_stop=True)
                    fixture["decision_path"].write_text(json.dumps(fixture["decision"]))
                observed = collect_results.factorial_tiller_runs(
                    collect_results.DEFAULT_MATRIX, root, fixture["published"], fixture["checkpoints"]
                )
                self.assertEqual(observed, (fixture["published"], fixture["checkpoints"]))

    def test_factorial_tiller_rejects_wrong_stage_or_stale_process_identity(self) -> None:
        for field, value in (("stage", "01_muon"), ("matrix_index", 211), ("row_id", "other"),
                             ("artifact_sha256", "stale"), ("source_freeze_sha256", "stale"),
                             ("elapsed_seconds", 0), ("artifact_path", "other.jsonl")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.factorial_tiller_fixture(root)
                fixture["timing"][field] = value
                fixture["timing_path"].write_text(json.dumps(fixture["timing"]))
                with self.assertRaisesRegex(RuntimeError, "invalid factorial process wall clock"):
                    collect_results.factorial_tiller_runs(
                        collect_results.DEFAULT_MATRIX, root, fixture["published"], fixture["checkpoints"]
                    )

    def test_factorial_tiller_validates_public_mapping_and_artifact_configuration(self) -> None:
        for field, value in (("dataset", "wrong"), ("seed", 999), ("model", "12l_768d"),
                             ("train_tokens", 100000000), ("steps", 3050)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.factorial_tiller_fixture(root)
                fixture["target"][field] = value
                (root / "matrix.json").write_text(json.dumps({"rows": fixture["staged"]}))
                with self.assertRaisesRegex(RuntimeError, "factorial/public TILLER identity mismatch"):
                    collect_results.factorial_tiller_runs(
                        collect_results.DEFAULT_MATRIX, root, fixture["published"], fixture["checkpoints"]
                    )
        for field, value in (("dataset", "wrong"), ("seed", 999), ("layers", 12),
                             ("train_tokens", 100000000), ("steps", 3050)):
            with self.subTest(config_field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.factorial_tiller_fixture(root)
                fixture["records"][0][field] = value
                fixture["artifact"].write_text("".join(json.dumps(row) + "\n" for row in fixture["records"]))
                with self.assertRaisesRegex(RuntimeError, "factorial TILLER configuration mismatch"):
                    collect_results.factorial_tiller_runs(
                        collect_results.DEFAULT_MATRIX, root, fixture["published"], fixture["checkpoints"]
                    )

    def test_factorial_tiller_preserves_both_published_12_layer_suites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.factorial_tiller_fixture(root)
            rows, checkpoints = collect_results.factorial_tiller_runs(
                collect_results.DEFAULT_MATRIX, root, fixture["published"], fixture["checkpoints"]
            )
            small = [row for row in rows if row["model_scale"] == "12l_768d"]
            self.assertEqual({int(row["train_tokens"]) for row in small}, {100000000, 300000000})
            self.assertEqual(small, [row for row in fixture["published"] if row["model_scale"] == "12l_768d"])
            self.assertEqual([row for row in checkpoints if row["model_scale"] == "12l_768d"],
                             [row for row in fixture["checkpoints"] if row["model_scale"] == "12l_768d"])
            output = root / "published"
            collect_results.write_csv(output / "tiller/runs.csv", collect_results.RUN_FIELDS, fixture["published"])
            collect_results.write_csv(output / "tiller/checkpoints.csv", collect_results.CHECKPOINT_FIELDS,
                                      fixture["checkpoints"])
            refreshed, _ = collect_results.refresh_tiller_controls(
                output, [], factorial_campaign=root,
            )
            self.assertEqual([row for row in refreshed if row["model_scale"] == "12l_768d"], small)

    def test_completed_rerun_supersedes_historical_result(self) -> None:
        manifest = collect_results.DEFAULT_MANIFEST
        with manifest.open(newline="") as handle:
            sources = [r for r in csv.DictReader(handle)
                       if r["phase"] == collect_results.FACTORIAL_PHASE
                       and r["optimizer"] == "muon"]
        staged = [dict(r, matrix_index=i) for i, r in enumerate(sources)]
        target = staged[0]
        target["existing_result"] = {
            "status": "complete", "endpoint": 9.0, "step1000": 10.0,
            "total_seconds": 999.0, "completed_steps": 9150,
            "realized_lr_trace_sha256": "old", "file_sha256": "old",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "matrix.json").write_text(json.dumps({"rows": staged}))
            artifact = root / "runs" / target["row_id"] / f"{target['activation']}.jsonl"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("\n".join(json.dumps(r) for r in [
                {"event": "config"},
                {"event": "eval", "step": 1000, "val_loss": 4.0},
                {"event": "eval", "step": 9150, "val_loss": 3.0},
                {"event": "summary", "completed_steps": 9150, "total_seconds": 20.0},
            ]))
            rows, checkpoints = collect_results.factorial_muon_runs(manifest, root)
            result = next(r for r in rows if r["run_index"] == int(target["row_index"]))
            self.assertEqual(result["final_validation_loss"], 3.0)
            self.assertEqual(result["step1000_validation_loss"], 4.0)
            self.assertEqual(result["training_loop_total_seconds"], 20.0)
            self.assertEqual(result["source_jsonl_sha256"], collect_results.sha256(artifact))
            endpoint = next(r for r in checkpoints if r["run_index"] == result["run_index"]
                            and r["step"] == 9150)
            self.assertEqual(endpoint["validation_loss"], result["final_validation_loss"])
            # An unfinished retry leaves the completed historical result intact.
            artifact.write_text(json.dumps({"event": "eval", "step": 1000,
                                           "val_loss": 5.0}))
            rows, checkpoints = collect_results.factorial_muon_runs(manifest, root)
            result = next(r for r in rows if r["run_index"] == int(target["row_index"]))
            self.assertEqual(result["final_validation_loss"], 9.0)
            self.assertEqual(result["training_loop_total_seconds"], 999.0)
            self.assertEqual(result["source_jsonl_sha256"], "old")
            self.assertEqual(checkpoints, [])
            historical = root / "completed.jsonl"
            historical.write_text(json.dumps({"event": "eval", "step": 9150,
                                             "val_loss": 9.0}))
            target["existing_result"]["canonical_path"] = str(historical)
            target["existing_result"]["file_sha256"] = collect_results.sha256(historical)
            (root / "matrix.json").write_text(json.dumps({"rows": staged}))
            rows, checkpoints = collect_results.factorial_muon_runs(manifest, root)
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0]["validation_loss"], 9.0)

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
        self.assertEqual(len(payload["rows"]), 45)
        self.assertEqual(
            sum(row["phase"] == verifier.SUITE_100M for row in payload["rows"]), 15
        )
        self.assertEqual(
            sum(row["phase"] == verifier.SUITE_300M_SMALL for row in payload["rows"]), 15
        )
        self.assertEqual(
            sum(row["phase"] == verifier.SUITE_300M_LARGE for row in payload["rows"]), 15
        )

    def test_token_fingerprint_manifest_covers_each_dataset_and_budget(self) -> None:
        payload = verifier.verify_token_fingerprints()
        self.assertEqual(len(payload["cells"]), 10)

    def test_real_launcher_keeps_every_shared_argument_pairwise_identical(self) -> None:
        payload = verifier.verify_matrix(verifier.verify_manifest())
        verifier.verify_launcher_pairing(payload)

    def test_baseline_launcher_enables_the_row_aware_lr_wd_contract(self) -> None:
        rows = verifier.verify_manifest()
        verifier.verify_manifest_launcher_contract(rows)
        lion = next(
            row
            for row in rows
            if row["phase"] == verifier.SUITE_300M_LARGE
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
        phase = verifier.SUITE_300M_LARGE
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
        self.assertEqual(labels["TILLER transfer matrix"], 45)
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
