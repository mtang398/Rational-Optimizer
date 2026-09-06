#!/usr/bin/env python3
"""Prove the exact method port and freeze every campaign input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
AUTHORITATIVE = Path("/home/mt872/Global-RLB")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_method_paths() -> list[Path]:
    existing = PACKAGE / "EXACT_METHOD_SOURCE.json"
    if existing.is_file():
        payload = json.loads(existing.read_text())
        paths = sorted(
            ROOT / item["path"]
            for item in payload.get("files", ())
            if item["path"].startswith("optimizer_design/")
        )
    else:
        paths = sorted(
            path for path in (ROOT / "optimizer_design").rglob("*.py")
            if "__pycache__" not in path.parts
            and (AUTHORITATIVE / path.relative_to(ROOT)).is_file()
            and digest(path) == digest(AUTHORITATIVE / path.relative_to(ROOT))
        )
    if len(paths) < 140:
        raise RuntimeError(f"exact source inventory unexpectedly small: {len(paths)}")
    return paths


def main() -> None:
    exact_paths = exact_method_paths()
    entries = []
    for local in exact_paths:
        relative = local.relative_to(ROOT)
        source = AUTHORITATIVE / relative
        if not source.is_file():
            raise FileNotFoundError(f"authoritative source missing: {source}")
        local_hash = digest(local)
        source_hash = digest(source)
        if local_hash != source_hash:
            raise RuntimeError(f"method source differs from authoritative bytes: {relative}")
        entries.append({"path": str(relative), "sha256": local_hash})

    entrypoint_relatives = (
        "experiments/rlb_300m_4000_design_20260731/candidate_entrypoint_factorized_every_step_rfd_gradient_ledger_muon_v1.py",
        "experiments/rlb_300m_4000_design_20260731/candidate_entrypoint_global_response_transaction_muon_v1.py",
        "experiments/rlb_300m_4000_design_20260731/candidate_entrypoint_loss_weighted_four_role_response_homotopy_batched_muon_v2.py",
        "experiments/rlb_300m_4000_design_20260731/candidate_entrypoint_r01_9150_base.py",
        "experiments/rlb_300m_4000_design_20260731/candidate_entrypoint_scalable_group_muon_base.py",
    )
    for item in entrypoint_relatives:
        local = ROOT / item
        source = AUTHORITATIVE / item
        if digest(local) != digest(source):
            raise RuntimeError(f"entrypoint differs from authoritative bytes: {item}")
        entries.append({"path": item, "sha256": digest(local)})

    activation_relatives = (
        "activation/csrc/rational_ext.cpp",
        "activation/csrc/rational_cuda_kernel.cu",
        "activation/rational_opt/__init__.py",
        "activation/rational_opt/rational.py",
        "activation/rational_opt/_C.cpython-312-x86_64-linux-gnu.so",
    )
    for item in activation_relatives:
        local = ROOT / item
        source = AUTHORITATIVE / item
        if not source.is_file():
            raise FileNotFoundError(f"authoritative activation artifact missing: {source}")
        if digest(local) != digest(source):
            raise RuntimeError(
                f"activation artifact differs from authoritative bytes: {item}"
            )
        entries.append({"path": item, "sha256": digest(local)})

    port = {
        "schema": "exact_factorized_ledger_source_port_v1",
        "authoritative_root": str(AUTHORITATIVE),
        "exact_internal_optimizer_key": "factorized_every_step_rfd_gradient_ledger_muon_v1",
        "files": entries,
    }
    port_path = PACKAGE / "EXACT_METHOD_SOURCE.json"
    port_path.write_text(json.dumps(port, indent=2, sort_keys=True) + "\n")

    package_inputs = [
        path for path in PACKAGE.iterdir()
        if path.is_file()
        and path.suffix in {".py", ".json", ".md", ".sbatch"}
        and path.name not in {"RESULTS.json"}
    ]
    other_inputs = [
        ROOT / "training/transformer_lm_compare.py",
        ROOT / "training/fairness_exact_resume.py",
        ROOT / "experiments/manifests/iclr26_main_manifest.csv",
        ROOT / "experiments/rlb_300m_4000_design_20260731/test_factorized_every_step_rfd_gradient_ledger_muon.py",
        *[ROOT / item for item in entrypoint_relatives],
        *[ROOT / item for item in activation_relatives],
        *exact_paths,
    ]
    extension = ROOT / activation_relatives[-1]
    all_inputs = sorted(
        {path.resolve() for path in (*package_inputs, *other_inputs, extension)},
        key=str,
    )
    missing = [str(path) for path in all_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"campaign input missing: {missing}")
    lines = [
        f"{digest(path)}  {path.relative_to(ROOT.resolve())}"
        for path in all_inputs
    ]
    freeze_path = PACKAGE / "SOURCE_FREEZE.sha256"
    freeze_path.write_text("\n".join(lines) + "\n")
    print(f"verified {len(entries)} authoritative method files")
    print(f"froze {len(all_inputs)} campaign inputs")


if __name__ == "__main__":
    main()
