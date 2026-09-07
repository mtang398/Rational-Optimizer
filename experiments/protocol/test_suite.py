"""CPU-only integrity tests for the public reproducibility package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from . import analyze_row
from . import build_source_freeze
from . import row_tools
from . import run_activation_row
from . import verify_repository as verifier


class PublicReproducibilityTests(unittest.TestCase):
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
