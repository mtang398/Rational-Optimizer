#!/usr/bin/env python3
"""Write horizon-AUC and threshold metrics for dense synthetic curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASKS = [
    ("synthetic/code", "synthetic_code", "Code"),
    ("synthetic/symbolic", "synthetic_symbolic", "Symbolic"),
    ("synthetic/reasoning_mix", "synthetic_reasoning_mix", "Reasoning mix"),
]
METHODS = [
    ("SiLU/SwiGLU+AdamW", "adamw_controls", "silu"),
    ("RLB+AdamW", "adamw_controls", "rlb_fused_fixed_strong_ffn"),
    ("SiLU/SwiGLU+Muon", "muon_controls", "silu"),
    ("RLB+Muon", "muon_controls", "rlb_fused_fixed_strong_ffn"),
    ("RLB MatrixPolicy", "matrix_policy", "rlb_fused_fixed_strong_ffn"),
    ("RLB MatrixPolicy group-stat", "matrix_policy_groupstat", "rlb_fused_fixed_strong_ffn"),
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def events(rows: list[dict], event: str) -> list[dict]:
    return [row for row in rows if row.get("event") == event]


def last(rows: list[dict], event: str) -> dict | None:
    selected = events(rows, event)
    return selected[-1] if selected else None


def mean_auc(points: list[tuple[int, float]], horizon: int) -> float | str:
    selected = [(int(step), float(value)) for step, value in points if int(step) <= horizon]
    if len(selected) < 2:
        return ""
    area = 0.0
    for (s0, y0), (s1, y1) in zip(selected, selected[1:]):
        area += (s1 - s0) * (y0 + y1) / 2.0
    denom = selected[-1][0] - selected[0][0]
    return area / denom if denom > 0 else ""


def first_below(points: list[tuple[int, float]], threshold: float) -> int | str:
    for step, value in points:
        if float(value) <= threshold:
            return int(step)
    return ""


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict], fields: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(title for title, _ in fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        cells = []
        for _, key in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="experiments/runs/synthetic_dense_curves_20260529")
    parser.add_argument("--suffix", default="20260529_dense_curve")
    parser.add_argument("--result-dir", default="experiments/results/synthetic_dense_curves_2026_05_29")
    args = parser.parse_args()

    root = Path(args.run_root)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name, safe_name, task_label in TASKS:
        for method, tag, activation in METHODS:
            run_name = f"{safe_name}_{tag}_{args.suffix}"
            path = root / safe_name / run_name / f"{activation}.jsonl"
            records = read_jsonl(path)
            eval_points = [(int(row["step"]), float(row["val_loss"])) for row in events(records, "eval") if "val_loss" in row]
            train_points = [(int(row["step"]), float(row["loss"])) for row in events(records, "train") if "loss" in row]
            final_eval = last(records, "eval") or {}
            summary = last(records, "summary") or {}
            rows.append({
                "task": task_label,
                "task_safe": safe_name,
                "dataset_name": dataset_name,
                "method": method,
                "complete": bool(final_eval and summary and int(summary.get("steps", -1)) == int(final_eval.get("step", -2))),
                "val_auc_mean_100": mean_auc(eval_points, 100),
                "val_auc_mean_200": mean_auc(eval_points, 200),
                "val_auc_mean_400": mean_auc(eval_points, 400),
                "train_auc_mean_100": mean_auc(train_points, 100),
                "train_auc_mean_200": mean_auc(train_points, 200),
                "train_auc_mean_400": mean_auc(train_points, 400),
                "first_val_loss_le_0p2": first_below(eval_points, 0.2),
                "first_val_loss_le_0p15": first_below(eval_points, 0.15),
                "final_val_loss": final_eval.get("val_loss", ""),
                "final_val_ppl": final_eval.get("val_ppl", ""),
                "path": str(path),
            })
    fields = [
        "task", "task_safe", "dataset_name", "method", "complete",
        "val_auc_mean_100", "val_auc_mean_200", "val_auc_mean_400",
        "train_auc_mean_100", "train_auc_mean_200", "train_auc_mean_400",
        "first_val_loss_le_0p2", "first_val_loss_le_0p15", "final_val_loss", "final_val_ppl", "path",
    ]
    write_csv(result_dir / "curve_speed_metrics.csv", rows, fields)
    sorted_rows = sorted(rows, key=lambda row: (row["task"], float(row["val_auc_mean_200"] or 1e9)))
    md = [
        "# Dense Synthetic Curve Speed Metrics",
        "",
        "Lower AUC mean loss is better. Final rows are secondary because these synthetic tasks approach the loss floor.",
        "",
        table(sorted_rows, [
            ("task", "task"),
            ("method", "method"),
            ("val AUC mean 100", "val_auc_mean_100"),
            ("val AUC mean 200", "val_auc_mean_200"),
            ("val AUC mean 400", "val_auc_mean_400"),
            ("train AUC mean 200", "train_auc_mean_200"),
            ("first val <= 0.2", "first_val_loss_le_0p2"),
            ("final loss", "final_val_loss"),
            ("final PPL", "final_val_ppl"),
        ]),
        "",
        "Generated by `experiments/scripts/summarize_dense_curve_speed_20260529.py`.",
    ]
    (result_dir / "curve_speed_metrics.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
