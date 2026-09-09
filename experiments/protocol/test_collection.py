"""CPU regressions for independent endpoints and loss/timing provenance."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from . import build_activation_optimizer_manifest as manifest_builder
from . import build_matrix
from . import collect_results as collector


class IndependentEndpointCollectionTests(unittest.TestCase):
    def fixture(self, root: Path, optimizer: str = "tiller_v1") -> dict:
        manifest = manifest_builder.build_rows()
        baseline = next(row for row in manifest if row["phase"] == collector.PRIMARY_PHASE
                        and row["dataset"] == "dclm" and row["seed"] == "1337"
                        and row["method"] == "silu_muon")
        index = 30 if optimizer == "tiller_v1" else 45
        source = build_matrix.build_row(index, baseline, {}, candidate_optimizer=optimizer)
        stage = collector.CANDIDATE_STAGES[optimizer]
        matrix = root / "matrix.json"
        matrix.write_text(json.dumps({"rows": [source]}))
        muon_snapshot = root / "source_muon/experiments/protocol"
        tiller_snapshot = root / "source_tiller/experiments/protocol"
        muon_snapshot.mkdir(parents=True)
        tiller_snapshot.mkdir(parents=True)
        manifest_builder.write_manifest(manifest, muon_snapshot / "activation_optimizer_manifest.csv")
        for snapshot in (muon_snapshot, tiller_snapshot):
            (snapshot / "SOURCE_FREEZE.sha256").write_text("synthetic CPU fixture source identity\n")
        name = f"{source['phase']}-{source['dataset']}-seed{source['seed']}-{collector.CANDIDATE_METHODS[optimizer]}"
        candidate = root / "runs" / stage / name / f"{collector.GRAIN_ID}.jsonl"
        control = root / "runs/activation_optimizer" / baseline["phase"] / baseline["dataset"] / baseline["row_id"] / "silu.jsonl"
        candidate.parent.mkdir(parents=True)
        control.parent.mkdir(parents=True)
        result_root = root / "results" / stage / f"matrix-{index}"
        result_root.mkdir(parents=True)
        token_cell = next(row for row in json.loads((collector.PACKAGE / "token_fingerprints.json").read_text())["cells"]
                          if row["dataset"] == "dclm" and row["train_tokens"] == 100_000_000)
        common = {key: source[key] for key in (
            "seed", "layers", "d_model", "heads", "ffn_dim", "seq_len", "steps", "grad_accum",
            "dataset_config", "dataset_revision", "tokenizer", "tokenizer_revision", "train_skip_tokens",
            "validation_skip_tokens", "grad_clip",
        )}
        common.update(
            event="config", dataset=source["dataset_name"], train_tokens=100_000_000,
            val_tokens=4_000_000, batch_size_per_gpu=source["batch_size"], world_size=4,
            train_token_sample_sha256=token_cell["train_token_sample_sha256"],
            val_token_sample_sha256=token_cell["validation_token_sample_sha256"],
            optimizer_lr_wd_fairness={"passed": True, "contract": "exact_lr_wd_v1"},
        )
        for field, key in (("optimizer_lr", "lr"), ("optimizer_min_lr", "min_lr"),
                           ("optimizer_weight_decay", "weight_decay"), ("optimizer_beta1", "beta1"),
                           ("optimizer_beta2", "beta2"), ("optimizer_eps", "eps")):
            common[field] = source[key]
        control_config = dict(common, activation="silu", optimizer="muon", experiment_identity="none",
                              source_manifest_sha256=collector.sha256(muon_snapshot / "activation_optimizer_manifest.csv"),
                              source_manifest_row_sha256=hashlib.sha256(json.dumps(
                                  baseline, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                              ).encode()).hexdigest(), source_manifest_row_id=baseline["row_id"],
                              source_manifest_row_index=baseline["row_index"],
                              source_freeze_sha256=collector.sha256(muon_snapshot / "SOURCE_FREEZE.sha256"))
        candidate_config = dict(common, activation=collector.GRAIN_ID, optimizer=optimizer,
                                experiment_identity="tiller_matrix_v1" if optimizer == "tiller_v1" else "tiller_then_muon_matrix_v1",
                                tiller_experiment_identity=dict(passed=True, **{
                                    key: source[key] for key in ("matrix_index", "source_manifest_row_index", "source_manifest_row_id")
                                }))
        def trajectory(config, early, final):
            return [config,
                    {"event": "eval", "step": 1000, "val_loss": early, "val_ppl": math.exp(early), "active_seconds_at_val_loss": 30.0},
                    {"event": "eval", "step": 3050, "val_loss": final, "val_ppl": math.exp(final), "active_seconds_at_val_loss": 90.0},
                    {"event": "summary", "completed_steps": 3050, "stopped_early": False,
                     "total_seconds": 90.0, "realized_lr_trace_sha256": "c" * 64}]
        candidate_data = trajectory(candidate_config, 3.9, 3.0)
        control_data = trajectory(control_config, 4.0, 3.1)
        for path, data in ((candidate, candidate_data), (control, control_data)):
            self.write_trajectory(path, data)
        clock = result_root / "CANDIDATE_WALL_CLOCK.json"
        control_clock = control.with_suffix(".wall_clock.json")
        for path, arm, elapsed in ((clock, "candidate", 110.0), (control_clock, "silu_muon", 100.0)):
            path.write_text(json.dumps({"schema": "rationalopt_process_wall_clock_v1",
                                       "arm": arm, "elapsed_seconds": elapsed, "return_code": 0}))
        report = {
            "schema": "tiller_matched_endpoint_result_v1", "status": "complete", "matrix_index": index,
            "phase": source["phase"], "candidate_optimizer": optimizer, "model": source["model"],
            "dataset": source["dataset"], "seed": source["seed"], "steps": 3050,
            "candidate_step1000_loss": 3.9, "candidate_endpoint_loss": 3.0,
            "control_step1000_loss": 4.0, "control_endpoint_loss": 3.1,
            "step1000_absolute_lead": 4.0 - 3.9, "absolute_endpoint_lead": 3.1 - 3.0,
            "candidate_end_to_end_total_seconds": 110.0, "control_end_to_end_total_seconds": 100.0,
            "exact_matched_hardware_end_to_end_total_time_ratio": 1.1,
            "candidate_training_loop_total_seconds": 90.0,
            "source_freeze_manifest_sha256": collector.sha256(tiller_snapshot / "SOURCE_FREEZE.sha256"),
        }
        for key, path in (("candidate_jsonl", candidate), ("control_jsonl", control),
                          ("candidate_wall_clock", clock), ("control_wall_clock", control_clock)):
            report[key] = {"path": str(path), "sha256": collector.sha256(path)}
        report_path = result_root / "RESULT.json"
        report_path.write_text(json.dumps(report))
        report_path.with_suffix(".json.sha256").write_text(f"{collector.sha256(report_path)}  RESULT.json\n")
        return locals()

    @staticmethod
    def write_trajectory(path, records):
        path.write_text("".join(json.dumps(row) + "\n" for row in records))

    def collect(self, fixture):
        return collector.tiller_runs(fixture["matrix"], Path("unused"), Path("unused"),
                                     campaign_root=fixture["root"])

    def test_both_new_candidates_require_their_own_complete_endpoint(self):
        for optimizer in collector.CANDIDATE_METHODS:
            with self.subTest(optimizer=optimizer), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory), optimizer)
                rows, checkpoints = self.collect(fixture)
                self.assertEqual(rows[0]["optimizer"], optimizer)
                self.assertEqual((rows[0]["status"], rows[0]["steps_completed"]), ("complete", 3050))
                self.assertEqual((rows[0]["final_validation_loss"], rows[0]["total_seconds"]), (3.0, 110.0))
                self.assertEqual([row["step"] for row in checkpoints], [1000, 3050])

    def test_partial_failed_retry_retains_completed_quality_time_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            prior, prior_checkpoints = self.collect(fixture)
            output = fixture["root"] / "published"
            collector.publish(output, "tiller_v1", prior, prior_checkpoints)
            fixture["report_path"].unlink()
            partial = fixture["candidate_data"][:2]
            partial[1] = dict(partial[1], val_loss=8.0, active_seconds_at_val_loss=7.0)
            self.write_trajectory(fixture["candidate"], partial)
            fixture["clock"].write_text(json.dumps({"schema": "rationalopt_process_wall_clock_v1",
                                                   "arm": "candidate", "return_code": 1, "elapsed_seconds": 9.0}))
            fresh, checkpoints = self.collect(fixture)
            self.assertEqual(fresh[0]["status"], "incomplete")
            merged, kept = collector.preserve_published_attempts(output, "tiller_v1", fresh, checkpoints)
            self.assertEqual((merged[0]["final_validation_loss"], merged[0]["total_seconds"]), ("3.0", "110.0"))
            self.assertEqual([(row["step"], row["validation_loss"]) for row in kept], [("1000", "3.9"), ("3050", "3.0")])

    def test_new_complete_retry_replaces_the_entire_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            rows, checkpoints = self.collect(fixture)
            output = fixture["root"] / "published"
            previous = [dict(rows[0], final_validation_loss=9.0, total_seconds=99.0)]
            collector.publish(output, "tiller_v1", previous, [])
            merged, kept = collector.preserve_published_attempts(output, "tiller_v1", rows, checkpoints)
            self.assertEqual((merged[0]["final_validation_loss"], merged[0]["total_seconds"]), (3.0, 110.0))
            self.assertEqual(kept, checkpoints)

    def test_wrong_budget_optimizer_or_baseline_freeze_is_rejected(self):
        for field, value in (("steps", 9150), ("optimizer", "muon"), ("train_tokens", 300_000_000)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                fixture["candidate_data"][0][field] = value
                self.write_trajectory(fixture["candidate"], fixture["candidate_data"])
                with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                    self.collect(fixture)
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            (fixture["muon_snapshot"] / "SOURCE_FREEZE.sha256").write_text("changed fixture freeze\n")
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                self.collect(fixture)

    def test_failed_or_wrong_arm_process_clock_cannot_publish_an_endpoint(self):
        for change in ({"return_code": 1}, {"arm": "silu_muon"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                clock = json.loads(fixture["clock"].read_text())
                clock.update(change)
                fixture["clock"].write_text(json.dumps(clock))
                with self.assertRaisesRegex(RuntimeError, "successful process timing"):
                    self.collect(fixture)

    def test_reused_matrix_index_never_preserves_old_18_layer_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            rows, checkpoints = self.collect(fixture)
            old = dict(rows[0], train_tokens=300_000_000, steps_required=9150,
                       source_phase="18l_1024d_300m_tokens_9150_steps")
            output = fixture["root"] / "published"
            collector.publish(output, "tiller_v1", [old], checkpoints)
            pending = dict(rows[0], status="pending", steps_completed=0,
                           final_validation_loss="", total_seconds="")
            merged, kept = collector.preserve_published_attempts(output, "tiller_v1", [pending], [])
            self.assertEqual(merged, [pending])
            self.assertEqual(kept, [])

    def test_all_published_12_layer_rows_and_checkpoints_are_preserved(self):
        output = collector.REPOSITORY / "experiments/results"
        total_rows = total_checkpoints = 0
        for optimizer, folder in collector.OUTPUT_DIRECTORIES.items():
            runs_path = output / folder / "runs.csv"
            if not runs_path.is_file():
                continue
            with runs_path.open(newline="") as handle:
                previous = [row for row in csv.DictReader(handle) if row["model_scale"] == "12l_768d"]
            with (runs_path.parent / "checkpoints.csv").open(newline="") as handle:
                previous_checkpoints = [row for row in csv.DictReader(handle) if row["model_scale"] == "12l_768d"]
            pending = [dict(row, status="pending", steps_completed=0,
                            final_validation_loss="", total_seconds="") for row in previous]
            merged, checkpoints = collector.preserve_published_attempts(output, optimizer, pending, [])
            self.assertEqual(merged, previous, folder)
            self.assertEqual(checkpoints, previous_checkpoints, folder)
            total_rows += len(merged)
            total_checkpoints += len(checkpoints)
        self.assertEqual((total_rows, total_checkpoints), (450, 930))


if __name__ == "__main__":
    unittest.main()
