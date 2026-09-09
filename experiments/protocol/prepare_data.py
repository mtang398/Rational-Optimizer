#!/usr/bin/env python3
"""Prepare the revision-pinned token caches for one experiment suite."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

from experiments.protocol.run_activation_row import DEFAULT_MANIFEST, training_arguments


ROOT = Path(__file__).resolve().parents[2]


def selected_rows(manifest: Path, phase: str, datasets: list[str]) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == phase]
    if not rows:
        raise ValueError(f"suite not found: {phase}")
    available = {row["dataset"] for row in rows}
    unknown = set(datasets) - available
    if unknown:
        raise ValueError(f"datasets not in this suite: {sorted(unknown)}")
    chosen: dict[str, dict[str, str]] = {}
    for row in rows:
        if not datasets or row["dataset"] in datasets:
            chosen.setdefault(row["dataset"], row)
    return list(chosen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT / "activation"), str(ROOT), environment.get("PYTHONPATH"),
    )))
    for row in selected_rows(args.manifest, args.suite, args.dataset):
        command = [sys.executable, "-B", *training_arguments(
            row, ROOT / "experiments/runs/activation_optimizer"
        ), "--prepare-only"]
        print(f"Preparing {row['dataset']}: {row['train_tokens']} training / "
              f"{row['val_tokens']} validation tokens", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
