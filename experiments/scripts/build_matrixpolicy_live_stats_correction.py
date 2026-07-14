#!/usr/bin/env python3
"""Build and verify the frozen MatrixPolicy live-stat correction campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(
    os.environ.get("RATIONALOPT_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
CAMPAIGN_ID = "matrixpolicy_live_stats_20260712"
CAMPAIGN_ROOT = ROOT / "experiments/corrections" / CAMPAIGN_ID
MANIFEST_ROOT = CAMPAIGN_ROOT / "manifests"
RUNTIME_ROOT = CAMPAIGN_ROOT / "runtime"
FREEZE_PATH = CAMPAIGN_ROOT / "freeze.json"
SOURCE_TOKEN_CACHE_ROOT = ROOT / "experiments/cache/tokens_iclr26_main"
SOURCE_HF_CACHE_ROOT = ROOT / "experiments/cache/huggingface"
CACHE_ROOT = CAMPAIGN_ROOT / "cache"
TOKEN_CACHE_ROOT = CACHE_ROOT / "tokens_iclr26_main"
HF_CACHE_ROOT = CACHE_ROOT / "huggingface"

SOURCE_MANIFESTS = {
    "main": ROOT / "experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv",
    "e8": ROOT / "experiments/manifests/iclr26_e8_primary_manifest.csv",
    "e9_preflight": ROOT / "abalation/manifests/e9_preflight_manifest.csv",
    "e9": ROOT / "abalation/manifests/e9_100m_manifest.csv",
}
OUTPUT_MANIFESTS = {
    key: MANIFEST_ROOT / f"{CAMPAIGN_ID}_{key}.csv" for key in SOURCE_MANIFESTS
}

RUNTIME_TREES = ("activation", "optimizer_design", "training", "tests")
RUNTIME_FILES = (
    "setup.py",
    "experiments/scripts/build_matrixpolicy_live_stats_correction.py",
    "experiments/scripts/validate_matrixpolicy_live_stats_correction.py",
    "experiments/scripts/run_matrixpolicy_live_stats_correction_job.sh",
    "experiments/scripts/run_matrixpolicy_live_stats_nccl_gate.sh",
    "experiments/scripts/submit_matrixpolicy_live_stats_correction.py",
    "experiments/scripts/cancel_failed_matrixpolicy_live_stats_dependencies.py",
)
RUNTIME_IGNORES = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "build",
    "dist",
    "*.egg-info",
)

TRAINING_DISTRIBUTION_ROOTS = (
    "datasets",
    "huggingface-hub",
    "numpy",
    "packaging",
    "torch",
    "transformers",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit(f"manifest has no header: {path}")
        return list(reader.fieldnames), rows


def corrected_rows(kind: str) -> tuple[list[str], list[dict[str, str]]]:
    fields, source = read_rows(SOURCE_MANIFESTS[kind])
    if kind == "main":
        selected = source
        if len(selected) != 30:
            raise SystemExit(f"main source must contain 30 rows, found {len(selected)}")
        if {row["optimizer"] for row in selected} != {"rational_matrix_policy_onpolicy"}:
            raise SystemExit("main correction manifest contains a non-MatrixPolicy row")
        if sorted(int(row["steps"]) for row in selected).count(3050) != 15:
            raise SystemExit("main correction manifest is missing 100M rows")
        if sorted(int(row["steps"]) for row in selected).count(9150) != 15:
            raise SystemExit("main correction manifest is missing 300M rows")
        expected = {
            (phase, dataset, seed)
            for phase in ("E1_rational_only_100m", "E2_rational_only_300m")
            for dataset in ("dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en")
            for seed in ("1337", "2027", "3407")
        }
        observed = {(row["phase"], row["dataset"], row["seed"]) for row in selected}
        if observed != expected:
            raise SystemExit("main correction manifest is not the complete 2 x 5 x 3 matrix")
    elif kind == "e8":
        selected = [
            row for row in source if row["optimizer"] == "rational_matrix_policy_onpolicy"
        ]
        if len(source) != 240 or len(selected) != 80:
            raise SystemExit(
                f"E8 source/filter mismatch: source={len(source)}, MatrixPolicy={len(selected)}"
            )
        observed = {
            (row["dataset"], row["lr"], row["weight_decay"], row["seed"])
            for row in selected
        }
        expected = {
            (dataset, lr, weight_decay, "1337")
            for dataset in ("dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en")
            for lr in ("0.0001", "0.0002", "0.0003", "0.0005")
            for weight_decay in ("0.00", "0.05", "0.10", "0.20")
        }
        if observed != expected:
            raise SystemExit("E8 correction manifest is not the complete 5 x 4 x 4 grid")
    elif kind == "e9_preflight":
        selected = source
        if len(selected) != 10 or {row["arm_id"] for row in selected} != {
            f"A{index}" for index in range(10)
        }:
            raise SystemExit("E9 preflight must contain exactly A0-A9")
        if {(row["dataset"], row["seed"]) for row in selected} != {("dclm", "2479")}:
            raise SystemExit("E9 preflight must use the frozen DCLM/2479 block")
    elif kind == "e9":
        selected = source
        if len(selected) != 150:
            raise SystemExit(f"E9 source must contain 150 rows, found {len(selected)}")
        coverage = {
            (row["arm_id"], row["dataset"], row["seed"]) for row in selected
        }
        expected = {
            (f"A{arm}", dataset, seed)
            for arm in range(10)
            for dataset in ("dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en")
            for seed in ("2479", "5052", "8913")
        }
        if coverage != expected:
            raise SystemExit("E9 source does not provide complete 10 x 5 x 3 coverage")
    else:
        raise AssertionError(kind)

    extra_fields = [
        "correction_campaign",
        "correction_source_manifest",
        "correction_source_row_index",
    ]
    output_fields = fields + [field for field in extra_fields if field not in fields]
    output: list[dict[str, str]] = []
    if len({row["row_id"] for row in selected}) != len(selected):
        raise SystemExit(f"{kind} source contains duplicate row IDs")
    for new_index, source_row in enumerate(selected):
        row = dict(source_row)
        row["correction_campaign"] = CAMPAIGN_ID
        row["correction_source_manifest"] = str(
            SOURCE_MANIFESTS[kind].relative_to(ROOT)
        )
        row["correction_source_row_index"] = source_row["row_index"]
        row["row_index"] = str(new_index)
        if kind.startswith("e9"):
            row["design_version"] = "E9-live-stat-syncfix-2026-07-12"
        output.append(row)
    output_paths = {
        (row["phase"], row["dataset"], row["row_id"], row["activation"])
        for row in output
    }
    if len(output_paths) != len(output):
        raise SystemExit(f"{kind} correction manifest contains duplicate output paths")
    return output_fields, output


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def build_runtime_snapshot() -> None:
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True)
    for relative in RUNTIME_TREES:
        source = ROOT / relative
        if not source.is_dir():
            raise SystemExit(f"runtime source tree is missing: {relative}")
        shutil.copytree(
            source,
            RUNTIME_ROOT / relative,
            ignore=RUNTIME_IGNORES,
            copy_function=shutil.copy2,
        )
    for relative in RUNTIME_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise SystemExit(f"runtime source file is missing: {relative}")
        destination = RUNTIME_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_cache_snapshot() -> None:
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    for source, destination in (
        (SOURCE_TOKEN_CACHE_ROOT, TOKEN_CACHE_ROOT),
        (SOURCE_HF_CACHE_ROOT, HF_CACHE_ROOT),
    ):
        if not source.is_dir():
            raise SystemExit(f"cache source is missing: {source}")
        shutil.copytree(source, destination, copy_function=shutil.copy2)


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        else:
            executable = bool(mode & stat.S_IXUSR)
            path.chmod(0o555 if executable else 0o444)
    root.chmod(0o555)


def assert_read_only(root: Path) -> None:
    paths = [root, *root.rglob("*")]
    writable = [
        str(path)
        for path in paths
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ]
    if writable:
        raise SystemExit(f"frozen path has writable entries: {writable[:8]}")


def training_distribution_versions() -> dict[str, str]:
    site_packages = (
        ROOT
        / ".venv-cu128"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not site_packages.is_dir():
        raise SystemExit(f"campaign site-packages is missing: {site_packages}")
    versions: dict[str, str] = {}
    for distribution in importlib_metadata.distributions(path=[str(site_packages)]):
        installed_name = canonicalize_name(distribution.metadata["Name"])
        installed_version = distribution.version
        previous = versions.get(installed_name)
        if previous is not None and previous != installed_version:
            raise SystemExit(
                f"conflicting local distributions for {installed_name}: "
                f"{previous} and {installed_version}"
            )
        versions[installed_name] = installed_version

    pending = list(TRAINING_DISTRIBUTION_ROOTS)
    expanded: set[str] = set()
    while pending:
        requested_name = pending.pop()
        canonical_name = canonicalize_name(requested_name)
        if canonical_name in expanded:
            continue
        try:
            distribution = importlib_metadata.distribution(requested_name)
        except importlib_metadata.PackageNotFoundError as error:
            raise SystemExit(
                f"campaign dependency is missing: {requested_name}"
            ) from error
        installed_name = canonicalize_name(
            distribution.metadata.get("Name", requested_name)
        )
        expanded.add(installed_name)
        versions[installed_name] = distribution.version
        for requirement_text in distribution.requires or ():
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate(
                {"extra": ""}
            ):
                continue
            pending.append(requirement.name)
    return dict(sorted(versions.items()))


def environment_record() -> dict[str, object]:
    python = ROOT / ".venv-cu128/bin/python"
    if not python.is_file():
        raise SystemExit(f"campaign Python is missing: {python}")
    version = json.loads(
        subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import json, platform, sysconfig, torch; "
                    "print(json.dumps({"
                    "'python_version': platform.python_version(), "
                    "'python_implementation': platform.python_implementation(), "
                    "'python_soabi': sysconfig.get_config_var('SOABI'), "
                    "'torch_version': torch.__version__, "
                    "'torch_cuda_version': torch.version.cuda"
                    "}, sort_keys=True))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {
        "python_path": str(python),
        **version,
        "training_distributions": training_distribution_versions(),
    }


def build(*, reuse_cache: bool) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    for generated_root in (MANIFEST_ROOT, RUNTIME_ROOT):
        if generated_root.exists():
            shutil.rmtree(generated_root)
    if reuse_cache:
        if not CACHE_ROOT.is_dir():
            raise SystemExit(f"reusable campaign cache is missing: {CACHE_ROOT}")
        assert_read_only(CACHE_ROOT)
    elif CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    for kind, path in OUTPUT_MANIFESTS.items():
        fields, rows = corrected_rows(kind)
        write_manifest(path, fields, rows)
    build_runtime_snapshot()
    if not reuse_cache:
        build_cache_snapshot()
    cache_files = file_inventory(TOKEN_CACHE_ROOT)
    if not cache_files:
        raise SystemExit(f"token cache is empty: {TOKEN_CACHE_ROOT}")
    hf_cache_files = file_inventory(HF_CACHE_ROOT)
    if not hf_cache_files:
        raise SystemExit(f"Hugging Face cache is empty: {HF_CACHE_ROOT}")
    runtime_files = file_inventory(RUNTIME_ROOT)
    environment = environment_record()
    runtime_sha256 = object_sha256(
        {"runtime_files": runtime_files, "environment": environment}
    )
    core = {
        "campaign_id": CAMPAIGN_ID,
        "git_head": git_head(),
        "manifests": {
            kind: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "rows": len(read_rows(path)[1]),
            }
            for kind, path in OUTPUT_MANIFESTS.items()
        },
        "runtime_files": runtime_files,
        "runtime_sha256": runtime_sha256,
        "token_cache_root": str(TOKEN_CACHE_ROOT),
        "token_cache_files": cache_files,
        "hf_cache_root": str(HF_CACHE_ROOT),
        "hf_cache_files": hf_cache_files,
        "environment": environment,
    }
    freeze = {
        **core,
        "freeze_sha256": object_sha256(core),
        "frozen_unix_time": int(time.time()),
        "frozen_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    make_read_only(RUNTIME_ROOT)
    make_read_only(CACHE_ROOT)
    for path in OUTPUT_MANIFESTS.values():
        path.chmod(0o444)
    MANIFEST_ROOT.chmod(0o555)
    assert_read_only(RUNTIME_ROOT)
    assert_read_only(CACHE_ROOT)
    assert_read_only(MANIFEST_ROOT)
    temporary_freeze = CAMPAIGN_ROOT / f".{FREEZE_PATH.name}.tmp.{os.getpid()}"
    with temporary_freeze.open("w") as handle:
        handle.write(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_freeze, FREEZE_PATH)
    FREEZE_PATH.chmod(0o444)
    print(json.dumps({
        "campaign": CAMPAIGN_ID,
        "freeze_sha256": freeze["freeze_sha256"],
        "manifests": {kind: entry["rows"] for kind, entry in freeze["manifests"].items()},
        "runtime_files": len(freeze["runtime_files"]),
        "cache_files": len(cache_files),
        "hf_cache_files": len(hf_cache_files),
    }, sort_keys=True))


def verify(
    *,
    verify_cache: bool,
    verify_host_environment: bool = True,
) -> dict[str, object]:
    if not FREEZE_PATH.is_file():
        raise SystemExit(f"campaign freeze is missing: {FREEZE_PATH}")
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("campaign_id") != CAMPAIGN_ID:
        raise SystemExit("campaign ID mismatch")
    for entry in freeze["manifests"].values():
        path = ROOT / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise SystemExit(f"frozen manifest changed: {path}")
    observed_runtime = file_inventory(RUNTIME_ROOT)
    if observed_runtime != freeze["runtime_files"]:
        raise SystemExit("frozen runtime snapshot changed")
    if verify_host_environment:
        environment = environment_record()
        if environment != freeze["environment"]:
            raise SystemExit("campaign Python environment changed")
    else:
        # CPU validators inspect immutable artifacts and recorded provenance; only
        # GPU training launchers need to qualify the executing host environment.
        environment = freeze["environment"]
    expected_runtime_sha = object_sha256(
        {"runtime_files": observed_runtime, "environment": environment}
    )
    if expected_runtime_sha != freeze.get("runtime_sha256"):
        raise SystemExit("runtime digest is internally inconsistent")
    assert_read_only(RUNTIME_ROOT)
    assert_read_only(MANIFEST_ROOT)
    assert_read_only(CACHE_ROOT)
    if FREEZE_PATH.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SystemExit("campaign freeze file is writable")
    if verify_cache:
        observed_cache = file_inventory(TOKEN_CACHE_ROOT)
        if observed_cache != freeze["token_cache_files"]:
            raise SystemExit("frozen token cache changed")
        observed_hf_cache = file_inventory(HF_CACHE_ROOT)
        if observed_hf_cache != freeze["hf_cache_files"]:
            raise SystemExit("frozen Hugging Face cache changed")
    core = {
        key: freeze[key]
        for key in (
            "campaign_id",
            "git_head",
            "manifests",
            "runtime_files",
            "runtime_sha256",
            "token_cache_root",
            "token_cache_files",
            "hf_cache_root",
            "hf_cache_files",
            "environment",
        )
    }
    if object_sha256(core) != freeze["freeze_sha256"]:
        raise SystemExit("freeze digest is internally inconsistent")
    print(freeze["freeze_sha256"])
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "build",
            "verify",
            "verify-runtime",
            "verify-static",
            "verify-static-runtime",
        ),
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="reuse and re-inventory an existing read-only campaign cache",
    )
    args = parser.parse_args()
    if args.mode == "build":
        if FREEZE_PATH.exists():
            raise SystemExit(
                f"campaign is already frozen at {FREEZE_PATH}; remove the generated campaign deliberately before rebuilding"
            )
        build(reuse_cache=args.reuse_cache)
    else:
        if args.reuse_cache:
            parser.error("--reuse-cache is valid only with build")
        verify(
            verify_cache=args.mode in {"verify", "verify-static"},
            verify_host_environment=args.mode in {"verify", "verify-runtime"},
        )


if __name__ == "__main__":
    main()
