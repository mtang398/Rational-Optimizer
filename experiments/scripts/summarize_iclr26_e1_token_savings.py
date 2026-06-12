#!/usr/bin/env python3
"""Summarize E1 M0/100M token-to-target savings from completed JSONL eval records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


PHASE = "E1_m0_100m"
MANIFEST = Path("experiments/manifests/iclr26_main_manifest.csv")
RUN_ROOT = Path("experiments/runs/iclr26_main")
DEFAULT_OUTPUT = Path("experiments/results/iclr26_e1_token_savings_2026_06_12")
MATRIXPOLICY_METHOD = "rlb_matrixpolicy_original"
ADAMW_METHOD = "silu_adamw"

DATASETS = [
    ("dclm", "DCLM"),
    ("fineweb_edu", "FineWeb-Edu"),
    ("fineweb", "FineWeb"),
    ("dolma_sample", "Dolma-sample"),
    ("c4_en", "C4"),
]

TARGETS = {
    "dclm": [4.90, 4.70, 4.55, 4.45, 4.35, 4.30],
    "fineweb_edu": [4.80, 4.60, 4.40, 4.30, 4.20, 4.10],
    "fineweb": [5.00, 4.80, 4.60, 4.50, 4.40, 4.35],
    "dolma_sample": [5.00, 4.80, 4.60, 4.50, 4.40, 4.35],
    "c4_en": [5.00, 4.80, 4.60, 4.50, 4.40, 4.30],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_jsonl(path: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    evals: list[dict[str, object]] = []
    summary = None
    if not path.exists():
        return evals, summary
    with path.open("r", errors="replace") as handle:
        for raw in handle:
            if not raw.startswith("{"):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "eval":
                evals.append(record)
            elif event == "summary":
                summary = record
    return evals, summary


def load_rows(manifest: Path, run_root: Path) -> list[dict[str, object]]:
    wanted = {dataset for dataset, _ in DATASETS}
    rows: list[dict[str, object]] = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != PHASE or row["dataset"] not in wanted:
                continue
            jsonl_path = run_root / row["phase"] / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"
            evals, summary = read_jsonl(jsonl_path)
            rows.append(
                {
                    "dataset": row["dataset"],
                    "row": int(row["row_index"]),
                    "row_id": row["row_id"],
                    "seed": int(row["seed"]),
                    "method": row["method"],
                    "activation": row["activation"],
                    "optimizer": row["optimizer"],
                    "steps": int(row["steps"]),
                    "global_tokens_per_step": int(row["global_tokens_per_step"]),
                    "total_tokens": int(row["steps"]) * int(row["global_tokens_per_step"]),
                    "eval_interval": int(row["eval_interval"]),
                    "jsonl": str(jsonl_path),
                    "evals": evals,
                    "summary": summary,
                }
            )
    return sorted(rows, key=lambda row: int(row["row"]))


def first_hit_tokens(row: dict[str, object] | None, target: float) -> tuple[int | None, int | None]:
    if row is None:
        return None, None
    tokens_per_step = int(row["global_tokens_per_step"])
    for record in row["evals"]:  # type: ignore[index]
        step = record.get("step")
        loss = safe_float(record.get("val_loss"))
        if isinstance(step, int) and step > 0 and math.isfinite(loss) and loss <= target:
            return step, step * tokens_per_step
    return None, None


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_dataset_seed_method: dict[tuple[str, int, str], dict[str, object]] = {
        (str(row["dataset"]), int(row["seed"]), str(row["method"])): row for row in rows
    }
    aggregate: list[dict[str, object]] = []
    per_seed: list[dict[str, object]] = []

    for dataset, _label in DATASETS:
        seeds = sorted({int(row["seed"]) for row in rows if row["dataset"] == dataset})
        methods = sorted({str(row["method"]) for row in rows if row["dataset"] == dataset and row["method"] != MATRIXPOLICY_METHOD})
        for target in TARGETS[dataset]:
            mp_hits: dict[int, tuple[int | None, int | None]] = {}
            second_hits: dict[int, tuple[str | None, int | None, int | None]] = {}
            adamw_hits: dict[int, tuple[int | None, int | None]] = {}

            for seed in seeds:
                mp_row = by_dataset_seed_method.get((dataset, seed, MATRIXPOLICY_METHOD))
                mp_step, mp_tokens = first_hit_tokens(mp_row, target)
                mp_hits[seed] = (mp_step, mp_tokens)

                best_method = None
                best_step = None
                best_tokens = None
                for method in methods:
                    row = by_dataset_seed_method.get((dataset, seed, method))
                    step, tokens = first_hit_tokens(row, target)
                    if tokens is None:
                        continue
                    if best_tokens is None or tokens < best_tokens or (tokens == best_tokens and method < str(best_method)):
                        best_method = method
                        best_step = step
                        best_tokens = tokens
                second_hits[seed] = (best_method, best_step, best_tokens)

                adamw_row = by_dataset_seed_method.get((dataset, seed, ADAMW_METHOD))
                adamw_hits[seed] = first_hit_tokens(adamw_row, target)

                per_seed.append(
                    {
                        "dataset": dataset,
                        "target_loss": target,
                        "seed": seed,
                        "matrixpolicy_step": mp_step,
                        "matrixpolicy_tokens": mp_tokens,
                        "second_best_method": best_method,
                        "second_best_step": best_step,
                        "second_best_tokens": best_tokens,
                        "silu_adamw_step": adamw_hits[seed][0],
                        "silu_adamw_tokens": adamw_hits[seed][1],
                    }
                )

            mp_all = [tokens for _, tokens in mp_hits.values() if tokens is not None]
            second_common = [
                (mp_hits[seed][1], second_hits[seed][2])
                for seed in seeds
                if mp_hits[seed][1] is not None and second_hits[seed][2] is not None
            ]
            adamw_common = [
                (mp_hits[seed][1], adamw_hits[seed][1])
                for seed in seeds
                if mp_hits[seed][1] is not None and adamw_hits[seed][1] is not None
            ]

            mp_second_mean = mean([float(pair[0]) for pair in second_common]) if second_common else math.nan
            second_mean = mean([float(pair[1]) for pair in second_common]) if second_common else math.nan
            mp_adamw_mean = mean([float(pair[0]) for pair in adamw_common]) if adamw_common else math.nan
            adamw_mean = mean([float(pair[1]) for pair in adamw_common]) if adamw_common else math.nan
            saved_second = second_mean - mp_second_mean if second_common else math.nan
            saved_adamw = adamw_mean - mp_adamw_mean if adamw_common else math.nan

            aggregate.append(
                {
                    "dataset": dataset,
                    "target_loss": target,
                    "matrixpolicy_mean_tokens_all_hits": mean([float(tokens) for tokens in mp_all]) if len(mp_all) == len(seeds) else math.nan,
                    "matrixpolicy_hit_seeds": len(mp_all),
                    "matrixpolicy_mean_tokens_second_best_common": mp_second_mean,
                    "second_best_mean_tokens_common": second_mean,
                    "second_best_common_seeds": len(second_common),
                    "saved_tokens_vs_second_best": saved_second,
                    "saved_fraction_vs_second_best": saved_second / second_mean if second_common and second_mean else math.nan,
                    "matrixpolicy_mean_tokens_silu_adamw_common": mp_adamw_mean,
                    "silu_adamw_mean_tokens_common": adamw_mean,
                    "silu_adamw_common_seeds": len(adamw_common),
                    "saved_tokens_vs_silu_adamw": saved_adamw,
                    "saved_fraction_vs_silu_adamw": saved_adamw / adamw_mean if adamw_common and adamw_mean else math.nan,
                }
            )
    return aggregate, per_seed


def fmt_tokens(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "not reached"
    return f"{value / 1_000_000:.1f}M"


def fmt_percent(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def token_table(rows: list[dict[str, object]], dataset: str, seed_count: int = 3) -> str:
    lines = [
        "| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |",
        "| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in [item for item in rows if item["dataset"] == dataset]:
        second_seeds = int(row["second_best_common_seeds"])
        adamw_seeds = int(row["silu_adamw_common_seeds"])
        second_cmp = (
            f"{fmt_tokens(float(row['matrixpolicy_mean_tokens_second_best_common']))} -> "
            f"{fmt_tokens(float(row['second_best_mean_tokens_common']))} ({second_seeds}/{seed_count})"
            if second_seeds
            else f"not reached (0/{seed_count})"
        )
        adamw_cmp = (
            f"{fmt_tokens(float(row['matrixpolicy_mean_tokens_silu_adamw_common']))} -> "
            f"{fmt_tokens(float(row['silu_adamw_mean_tokens_common']))} ({adamw_seeds}/{seed_count})"
            if adamw_seeds
            else f"not reached (0/{seed_count})"
        )
        lines.append(
            "| {target:.2f} | {mp_all} | {second_cmp} | {second_saved} | {second_pct} | {adamw_cmp} | {adamw_saved} | {adamw_pct} |".format(
                target=float(row["target_loss"]),
                mp_all=fmt_tokens(float(row["matrixpolicy_mean_tokens_all_hits"])),
                second_cmp=second_cmp,
                second_saved=fmt_tokens(float(row["saved_tokens_vs_second_best"])),
                second_pct=fmt_percent(float(row["saved_fraction_vs_second_best"])),
                adamw_cmp=adamw_cmp,
                adamw_saved=fmt_tokens(float(row["saved_tokens_vs_silu_adamw"])),
                adamw_pct=fmt_percent(float(row["saved_fraction_vs_silu_adamw"])),
            )
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float) and math.isnan(value):
                    value = "nan"
                cleaned[field] = value
            writer.writerow(cleaned)


def write_readme(output_dir: Path, token_rows: list[dict[str, object]], source_rows: list[dict[str, object]]) -> None:
    tokens_per_step = int(source_rows[0]["global_tokens_per_step"])
    eval_interval = int(source_rows[0]["eval_interval"])
    total_tokens = int(source_rows[0]["total_tokens"])
    sections = []
    for dataset, label in DATASETS:
        sections.append(f"## {label}\n\n{token_table(token_rows, dataset)}")
    text = f"""# ICLR26 E1 Token-To-Target Savings

Generated from completed E1 M0/100M JSONL eval records. All rows still trained to the fixed budget of about `{total_tokens / 1_000_000:.1f}M` tokens; this is an early-stop/speed-to-target readout only.

Each row uses `{tokens_per_step}` global tokens/step and the native E1 eval cadence of {eval_interval} steps, or `{tokens_per_step * eval_interval / 1_000_000:.2f}M` tokens per readout interval.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target.

{"\n\n".join(sections)}

## Files

- `token_savings.csv`: aggregate token-to-target savings by dataset and target.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest, args.run_root)
    if not rows:
        raise SystemExit("No E1 rows found.")
    token_rows, token_seed_rows = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "token_savings.csv",
        token_rows,
        [
            "dataset",
            "target_loss",
            "matrixpolicy_mean_tokens_all_hits",
            "matrixpolicy_hit_seeds",
            "matrixpolicy_mean_tokens_second_best_common",
            "second_best_mean_tokens_common",
            "second_best_common_seeds",
            "saved_tokens_vs_second_best",
            "saved_fraction_vs_second_best",
            "matrixpolicy_mean_tokens_silu_adamw_common",
            "silu_adamw_mean_tokens_common",
            "silu_adamw_common_seeds",
            "saved_tokens_vs_silu_adamw",
            "saved_fraction_vs_silu_adamw",
        ],
    )
    write_csv(
        args.output_dir / "token_savings_per_seed.csv",
        token_seed_rows,
        [
            "dataset",
            "target_loss",
            "seed",
            "matrixpolicy_step",
            "matrixpolicy_tokens",
            "second_best_method",
            "second_best_step",
            "second_best_tokens",
            "silu_adamw_step",
            "silu_adamw_tokens",
        ],
    )
    write_readme(args.output_dir, token_rows, rows)


if __name__ == "__main__":
    main()
