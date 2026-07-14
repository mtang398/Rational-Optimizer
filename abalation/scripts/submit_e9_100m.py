#!/usr/bin/env python3
"""Freeze E9 inputs and submit one Slurm job per manifest row in two chains."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = ROOT / "abalation/submissions/e9_frozen_inputs.json"
LAUNCHER = "abalation/scripts/run_e9_100m_manifest_job.sh"
FROZEN_PATHS = [
    "experiments/E9_100M_ABLATION_PLAN.md",
    "abalation/manifests/e9_100m_manifest.csv",
    "abalation/manifests/e9_preflight_manifest.csv",
    "experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv",
    "abalation/scripts/build_e9_100m_manifest.py",
    "abalation/scripts/analyze_e9_100m.py",
    "abalation/scripts/submit_e9_100m.py",
    "abalation/scripts/run_e9_100m_manifest_job.sh",
    "abalation/scripts/run_matrixpolicy_ablation_manifest_job.sh",
    "training/run_lm_optimizer_sweep.sbatch",
    "training/transformer_lm_compare.py",
    "optimizer_design/matrix_policy_optimizer.py",
    "optimizer_design/transport_onpolicy_optimizer.py",
    "setup.py",
    "activation/csrc/rational_ext.cpp",
    "activation/csrc/rational_cuda_kernel.cu",
    "activation/rational_opt/_C.cpython-312-x86_64-linux-gnu.so",
    "activation/rational_opt/__init__.py",
    "activation/rational_opt/rational.py",
    "optimizer_design/__init__.py",
    "optimizer_design/baseline_optimizers.py",
    "optimizer_design/function_space_rational_optimizer.py",
    "experiments/cache/tokens_iclr26_main/dclm/mlfoundations_dclm_baseline_1_0_none_gpt2_train_train_stream_text_skipdocs0_skiptoks0_2621440.pt",
    "experiments/cache/tokens_iclr26_main/dclm/mlfoundations_dclm_baseline_1_0_none_gpt2_train_validation_stream_text_skipdocs0_skiptoks210000000_200000.pt",
    "experiments/cache/tokens_iclr26_main/dclm/mlfoundations_dclm_baseline_1_0_none_gpt2_train_train_stream_text_skipdocs0_skiptoks0_100000000.pt",
    "experiments/cache/tokens_iclr26_main/dclm/mlfoundations_dclm_baseline_1_0_none_gpt2_train_validation_stream_text_skipdocs0_skiptoks210000000_4000000.pt",
    "experiments/cache/tokens_iclr26_main/fineweb_edu/HuggingFaceFW_fineweb_edu_sample_10BT_gpt2_train_train_stream_text_skipdocs0_skiptoks0_100000000.pt",
    "experiments/cache/tokens_iclr26_main/fineweb_edu/HuggingFaceFW_fineweb_edu_sample_10BT_gpt2_train_validation_stream_text_skipdocs0_skiptoks210000000_4000000.pt",
    "experiments/cache/tokens_iclr26_main/fineweb/HuggingFaceFW_fineweb_sample_10BT_gpt2_train_train_stream_text_skipdocs0_skiptoks0_100000000.pt",
    "experiments/cache/tokens_iclr26_main/fineweb/HuggingFaceFW_fineweb_sample_10BT_gpt2_train_validation_stream_text_skipdocs0_skiptoks210000000_4000000.pt",
    "experiments/cache/tokens_iclr26_main/dolma_sample/allenai_dolma_v1_6_sample_gpt2_train_train_stream_text_skipdocs0_skiptoks0_100000000.pt",
    "experiments/cache/tokens_iclr26_main/dolma_sample/allenai_dolma_v1_6_sample_gpt2_train_validation_stream_text_skipdocs0_skiptoks210000000_4000000.pt",
    "experiments/cache/tokens_iclr26_main/c4_en/allenai_c4_en_gpt2_train_train_stream_text_skipdocs0_skiptoks0_100000000.pt",
    "experiments/cache/tokens_iclr26_main/c4_en/allenai_c4_en_gpt2_validation_validation_stream_text_skipdocs0_skiptoks0_4000000.pt",
]
RUNTIME_PATHS = [
    "abalation/scripts/submit_e9_100m.py",
    "abalation/scripts/run_e9_100m_manifest_job.sh",
    "abalation/scripts/run_matrixpolicy_ablation_manifest_job.sh",
    "training/run_lm_optimizer_sweep.sbatch",
    "training/transformer_lm_compare.py",
    "optimizer_design/__init__.py",
    "optimizer_design/baseline_optimizers.py",
    "optimizer_design/function_space_rational_optimizer.py",
    "optimizer_design/matrix_policy_optimizer.py",
    "optimizer_design/transport_onpolicy_optimizer.py",
    "activation/rational_opt/__init__.py",
    "activation/rational_opt/rational.py",
    "activation/rational_opt/_C.cpython-312-x86_64-linux-gnu.so",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def current_runtime_sha256() -> str:
    files = {}
    for relative in RUNTIME_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"required runtime input is missing: {relative}")
        files[relative] = sha256(path)
    return object_sha256({"git_head": git_head(), "files": files})


def current_freeze() -> dict[str, object]:
    files = {}
    for relative in FROZEN_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"required frozen input is missing: {relative}")
        files[relative] = sha256(path)
    head = git_head()
    core = {"git_head": head, "files": files}
    runtime = {"git_head": head, "files": {path: files[path] for path in RUNTIME_PATHS}}
    return {
        **core,
        "freeze_sha256": object_sha256(core),
        "runtime_sha256": object_sha256(runtime),
    }


def verify_or_write_freeze(refresh: bool) -> dict[str, object]:
    current = current_freeze()
    if FREEZE_PATH.exists() and not refresh:
        frozen = json.loads(FREEZE_PATH.read_text())
        comparable = {
            "git_head": frozen.get("git_head"),
            "files": frozen.get("files"),
            "freeze_sha256": frozen.get("freeze_sha256"),
            "runtime_sha256": frozen.get("runtime_sha256"),
        }
        if comparable != current:
            raise SystemExit(
                f"E9 frozen inputs changed after the freeze at {FREEZE_PATH}; "
                "audit the change and use --refresh-freeze deliberately"
            )
        return frozen
    current["frozen_unix_time"] = int(time.time())
    current["frozen_local_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


def verify_runtime_freeze(expected: str) -> None:
    observed = current_runtime_sha256()
    if observed != expected:
        raise SystemExit(
            f"runtime freeze mismatch: expected {expected}, observed {observed}"
        )
    print(observed)


def load_manifest(path: Path, expected_phase: str, expected_rows: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise SystemExit(f"expected {expected_rows} rows in {path}, found {len(rows)}")
    indices = [int(row["row_index"]) for row in rows]
    if indices != list(range(expected_rows)):
        raise SystemExit(f"row_index is not contiguous in {path}")
    if {row["phase"] for row in rows} != {expected_phase}:
        raise SystemExit(f"unexpected phases in {path}")
    if len({row["row_id"] for row in rows}) != expected_rows:
        raise SystemExit(f"duplicate row IDs in {path}")
    return rows


def verify_generated_manifests() -> None:
    paths = [
        ROOT / "abalation/manifests/e9_100m_manifest.csv",
        ROOT / "abalation/manifests/e9_preflight_manifest.csv",
    ]
    before = {path: path.read_bytes() if path.is_file() else None for path in paths}
    subprocess.run(
        [sys.executable, "abalation/scripts/build_e9_100m_manifest.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [str(path.relative_to(ROOT)) for path in paths if before[path] != path.read_bytes()]
    if changed:
        raise SystemExit(f"generated E9 manifests were stale and have been rebuilt; audit before submission: {changed}")


def user_has_jobs() -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def verify_preflight_gate(frozen: dict[str, object]) -> None:
    coverage_path = ROOT / "abalation/results/e9_preflight/coverage.csv"
    if not coverage_path.is_file():
        raise SystemExit(f"scientific submission requires preflight coverage: {coverage_path}")
    with coverage_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10 or any(row.get("status") != "pass" for row in rows):
        failures = [row.get("arm_id", "?") for row in rows if row.get("status") != "pass"]
        raise SystemExit(f"scientific submission blocked by E9 preflight failures: {failures}")
    pass_path = ROOT / "abalation/results/e9_preflight/preflight_pass.json"
    if not pass_path.is_file():
        raise SystemExit(f"scientific submission requires provenance-bound preflight evidence: {pass_path}")
    evidence = json.loads(pass_path.read_text())
    if evidence.get("freeze_sha256") != frozen.get("freeze_sha256"):
        raise SystemExit("preflight evidence was produced under a different E9 freeze")
    manifest_hash = frozen["files"]["abalation/manifests/e9_preflight_manifest.csv"]
    if evidence.get("preflight_manifest_sha256") != manifest_hash:
        raise SystemExit("preflight evidence manifest hash mismatch")
    expected_rows = {
        (row["arm_id"], row["row_id"], row["jsonl_sha256"])
        for row in evidence.get("rows", [])
    }
    observed_rows = set()
    manifest_rows = load_manifest(
        ROOT / "abalation/manifests/e9_preflight_manifest.csv",
        "E9_preflight_80step",
        10,
    )
    for row in manifest_rows:
        path = (
            ROOT
            / "abalation/runs/e9_preflight"
            / row["phase"]
            / row["dataset"]
            / row["row_id"]
            / f"{row['activation']}.jsonl"
        )
        if not path.is_file():
            raise SystemExit(f"preflight JSONL missing after validation: {path}")
        observed_rows.add((row["arm_id"], row["row_id"], sha256(path)))
    if observed_rows != expected_rows:
        raise SystemExit("preflight JSONL provenance changed after validation")


def submit(mode: str, refresh_freeze: bool, allow_existing_jobs: bool) -> Path:
    verify_generated_manifests()
    frozen = verify_or_write_freeze(refresh_freeze)
    (ROOT / "abalation/logs/e9").mkdir(parents=True, exist_ok=True)
    if user_has_jobs() and not allow_existing_jobs:
        raise SystemExit("existing Slurm jobs detected; refusing to exceed the two-job 4xA6000 envelope")

    if mode == "preflight":
        manifest = "abalation/manifests/e9_preflight_manifest.csv"
        output_root = "abalation/runs/e9_preflight"
        rows = load_manifest(ROOT / manifest, "E9_preflight_80step", 10)
        prefix = "e9-pf"
    else:
        verify_preflight_gate(frozen)
        manifest = "abalation/manifests/e9_100m_manifest.csv"
        output_root = "abalation/runs/e9_100m"
        rows = load_manifest(ROOT / manifest, "E9_100m", 150)
        prefix = "e9"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    ledger = ROOT / f"abalation/submissions/e9_{mode}_{stamp}.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    submitted: list[dict[str, object]] = []
    previous: list[str | None] = [None, None]
    with ledger.open("w", newline="") as handle:
        fields = ["row_index", "row_id", "arm_id", "dataset", "seed", "job_id", "chain", "dependency"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        for row in rows:
            index = int(row["row_index"])
            chain = index % 2
            dependency = None if previous[chain] is None else f"afterany:{previous[chain]}"
            exports = ",".join(
                [
                    "ALL",
                    f"MANIFEST={manifest}",
                    f"OUTPUT_ROOT={output_root}",
                    f"ROW_START={index}",
                    "ROW_LIMIT=1",
                    "BUILD_EXT=0",
                    "SKIP_PREPARE=1",
                    f"E9_FREEZE_SHA256={frozen['freeze_sha256']}",
                    f"E9_RUNTIME_FREEZE_SHA256={frozen['runtime_sha256']}",
                    f"E9_MANIFEST_SHA256={frozen['files'][manifest]}",
                ]
            )
            command = [
                "sbatch",
                "--parsable",
                "--constraint=nvlink",
                f"--job-name={prefix}-r{index:03d}",
                f"--export={exports}",
            ]
            if dependency is not None:
                command.append(f"--dependency={dependency}")
            command.append(LAUNCHER)
            result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise SystemExit(f"unexpected sbatch output for row {index}: {result.stdout!r}")
            previous[chain] = job_id
            submitted_row = {
                "row_index": index,
                "row_id": row["row_id"],
                "arm_id": row["arm_id"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "job_id": job_id,
                "chain": chain,
                "dependency": dependency or "",
            }
            submitted.append(submitted_row)
            writer.writerow(submitted_row)
            handle.flush()
            os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "mode": mode,
                "submitted": len(submitted),
                "first_jobs": [submitted[0]["job_id"], submitted[1]["job_id"]],
                "terminal_jobs": previous,
                "ledger": str(ledger.relative_to(ROOT)),
                "freeze": str(FREEZE_PATH.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "scientific", "verify"))
    parser.add_argument("--refresh-freeze", action="store_true")
    parser.add_argument("--allow-existing-jobs", action="store_true")
    parser.add_argument("--expected-runtime-freeze")
    args = parser.parse_args()
    if args.mode == "verify":
        if not args.expected_runtime_freeze:
            parser.error("verify requires --expected-runtime-freeze")
        verify_runtime_freeze(args.expected_runtime_freeze)
        return
    submit(args.mode, args.refresh_freeze, args.allow_existing_jobs)


if __name__ == "__main__":
    main()
