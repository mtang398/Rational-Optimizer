#!/usr/bin/env python3
"""Generate dense E1 mean/std curve figures from E1 run JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PHASE = "E1_m0_100m"
MATRIXPOLICY_METHOD = "rlb_matrixpolicy_original"
REPLACEMENT_RLB_METHODS = {
    "rlb_adamw",
    "rlb_lion",
    "rlb_soap",
    "rlb_muon",
    "rlb_schedulefree",
    "rlb_came",
    "rlb_ademamix",
}

DATASETS = [
    ("dclm", "DCLM"),
    ("fineweb_edu", "FineWeb-Edu"),
    ("fineweb", "FineWeb"),
    ("dolma_sample", "Dolma-sample"),
    ("c4_en", "C4"),
]

ALL_METHODS = [
    ("rlb_matrixpolicy_original", "MatrixPolicy", "#111111", "-", 2.3, 0.16),
    ("rlb_adamw", "RLB+AdamW", "#1f77b4", "-", 1.45, 0.10),
    ("silu_adamw", "SiLU+AdamW", "#1f77b4", "--", 1.45, 0.08),
    ("rlb_lion", "RLB+Lion", "#2ca02c", "-", 1.45, 0.10),
    ("silu_lion", "SiLU+Lion", "#2ca02c", "--", 1.45, 0.08),
    ("rlb_soap", "RLB+SOAP", "#9467bd", "-", 1.35, 0.08),
    ("silu_soap", "SiLU+SOAP", "#9467bd", "--", 1.35, 0.08),
    ("rlb_muon", "RLB+Muon", "#ff7f0e", "-", 1.35, 0.08),
    ("silu_muon", "SiLU+Muon", "#ff7f0e", "--", 1.35, 0.08),
    ("rlb_schedulefree", "RLB+ScheduleFree", "#8c564b", "-", 1.25, 0.06),
    ("silu_schedulefree", "SiLU+ScheduleFree", "#8c564b", "--", 1.25, 0.06),
    ("rlb_came", "RLB+CAME", "#17becf", "-", 1.25, 0.06),
    ("silu_came", "SiLU+CAME", "#17becf", "--", 1.25, 0.06),
]

CLEAN_METHODS = [
    spec for spec in ALL_METHODS
    if spec[0] not in {"rlb_soap", "silu_soap"}
]

TABLE_METHODS = [
    ("rlb_matrixpolicy_original", "MatrixPolicy"),
    ("rlb_adamw", "RLB+AdamW"),
    ("silu_adamw", "SiLU+AdamW"),
    ("rlb_lion", "RLB+Lion"),
    ("silu_lion", "SiLU+Lion"),
    ("rlb_soap", "RLB+SOAP"),
    ("silu_soap", "SiLU+SOAP"),
    ("rlb_muon", "RLB+Muon"),
    ("silu_muon", "SiLU+Muon"),
    ("rlb_schedulefree", "RLB+ScheduleFree"),
    ("silu_schedulefree", "SiLU+ScheduleFree"),
    ("rlb_came", "RLB+CAME"),
    ("silu_came", "SiLU+CAME"),
    ("rlb_ademamix", "RLB+ADeMaMix"),
    ("silu_ademamix", "SiLU+ADeMaMix"),
]

START_STEP = 500
END_STEP = 3050
XTICKS = [500, 1000, 1500, 2000, 2500, 3000]
CHECKPOINT_STEPS = [500, 1000, 1500, 2000, 2500, 3050]


def _read_curve_jsonl(jsonl_path: Path, dataset: str, method: str, seed: int, curves) -> None:
    with jsonl_path.open("r", errors="replace") as jsonl:
        for raw in jsonl:
            if not raw.startswith("{"):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            step = record.get("step")
            if event not in {"train", "eval"} or not isinstance(step, int):
                continue
            if step < START_STEP or step > END_STEP:
                continue
            if event == "train":
                value = record.get("loss")
                if isinstance(value, (int, float)) and math.isfinite(value):
                    curves[dataset][method][seed]["train"][step] = float(value)
            elif event == "eval":
                loss = record.get("val_loss")
                ppl = record.get("val_ppl")
                if isinstance(loss, (int, float)) and math.isfinite(loss):
                    curves[dataset][method][seed]["eval"].setdefault(step, {})["val_loss"] = float(loss)
                if isinstance(ppl, (int, float)) and math.isfinite(ppl):
                    curves[dataset][method][seed]["eval"].setdefault(step, {})["val_ppl"] = float(ppl)


def _load_manifest_rows(
    manifest_path: Path,
    run_root: Path,
    curves,
    phase: str,
    wanted_methods: set[str],
) -> None:
    wanted_datasets = {dataset for dataset, _ in DATASETS}
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != phase:
                continue
            dataset = row.get("dataset", "")
            method = row.get("method", "")
            if dataset not in wanted_datasets or method not in wanted_methods:
                continue
            seed = int(row["seed"])
            run_dir = run_root / dataset / row["row_id"]
            jsonl_path = run_dir / f"{row['activation']}.jsonl"
            if not jsonl_path.exists():
                matches = sorted(path for path in run_dir.glob("*.jsonl") if ".incomplete_" not in path.name)
                if not matches:
                    continue
                jsonl_path = matches[0]
            _read_curve_jsonl(jsonl_path, dataset, method, seed, curves)


def parse_jsonl_runs(
    manifest_path: Path,
    run_root: Path,
    matrixpolicy_manifest: Path | None = None,
    matrixpolicy_run_root: Path | None = None,
    matrixpolicy_phase: str | None = None,
    replacement_manifest: Path | None = None,
    replacement_run_root: Path | None = None,
    replacement_phase: str | None = None,
):
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"train": {}, "eval": {}})))
    wanted_methods = {method for method, *_ in ALL_METHODS} | {method for method, _ in TABLE_METHODS}
    _load_manifest_rows(manifest_path, run_root, curves, PHASE, wanted_methods)
    if matrixpolicy_manifest is not None and matrixpolicy_run_root is not None and matrixpolicy_phase is not None:
        _load_manifest_rows(
            matrixpolicy_manifest,
            matrixpolicy_run_root,
            curves,
            matrixpolicy_phase,
            {MATRIXPOLICY_METHOD},
        )
    if replacement_manifest is not None and replacement_run_root is not None and replacement_phase is not None:
        _load_manifest_rows(
            replacement_manifest,
            replacement_run_root,
            curves,
            replacement_phase,
            REPLACEMENT_RLB_METHODS,
        )
    return curves

def aggregate_values(curves, dataset: str, method: str, metric: str):
    seed_data = curves.get(dataset, {}).get(method, {})
    by_step = defaultdict(list)
    for events in seed_data.values():
        if metric == "train_loss":
            for step, value in events["train"].items():
                by_step[step].append(value)
        else:
            key = "val_loss" if metric == "val_loss" else "val_ppl"
            for step, values in events["eval"].items():
                value = values.get(key)
                if value is not None and math.isfinite(value):
                    by_step[step].append(value)
    return by_step


def aggregate(curves, dataset: str, method: str, metric: str):
    by_step = aggregate_values(curves, dataset, method, metric)
    steps, means, stds = [], [], []
    for step in sorted(by_step):
        vals = np.asarray(by_step[step], dtype=float)
        if vals.size < 3:
            continue
        steps.append(step)
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)))
    return np.asarray(steps), np.asarray(means), np.asarray(stds)


def checkpoint_cell(values) -> str:
    vals = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if vals.size == 0:
        return "--"
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    suffix = f" (n={vals.size})" if vals.size < 3 else ""
    return f"{mean:.4f} +/- {std:.4f}{suffix}"


def checkpoint_table(curves, dataset: str, dataset_label: str) -> str:
    lines = [
        f"{dataset_label} validation-loss checkpoint table, mean +/- sample std:",
        "",
        "| Method | " + " | ".join(str(step) for step in CHECKPOINT_STEPS) + " |",
        "| --- | " + " | ".join("---:" for _ in CHECKPOINT_STEPS) + " |",
    ]
    for method, label in TABLE_METHODS:
        by_step = aggregate_values(curves, dataset, method, "val_loss")
        cells = [checkpoint_cell(by_step.get(step, [])) for step in CHECKPOINT_STEPS]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def final_snapshot(curves) -> str:
    lines = [
        "Final validation-loss overview across completed E1 datasets. Lower is better; cells are mean +/- sample std over three seeds.",
        "",
        "| Method | " + " | ".join(f"{label} final" for _, label in DATASETS) + " |",
        "| --- | " + " | ".join("---:" for _ in DATASETS) + " |",
    ]
    for method, label in TABLE_METHODS:
        cells = []
        for dataset, _dataset_label in DATASETS:
            by_step = aggregate_values(curves, dataset, method, "val_loss")
            cells.append(checkpoint_cell(by_step.get(3050, [])))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_status_tables(status_path: Path, curves) -> None:
    text = status_path.read_text()
    for dataset, label in DATASETS:
        heading = f"{label} validation-loss checkpoint table, mean +/- sample std:"
        start = text.index(heading)
        table_start = text.index("\n\n|", start)
        next_section_candidates = [
            idx for idx in (
                text.find("\n#### ", table_start + 1),
                text.find("\n### ", table_start + 1),
                text.find("\n## ", table_start + 1),
            )
            if idx != -1
        ]
        if not next_section_candidates:
            raise RuntimeError(f"could not find end of checkpoint table for {label}")
        end = min(next_section_candidates)
        replacement = checkpoint_table(curves, dataset, label).rstrip() + "\n"
        text = text[:start] + replacement + text[end:]
    status_path.write_text(text)


def plot_dataset(curves, dataset: str, dataset_label: str, metric: str, out_path: Path, methods, variant_label: str):
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=140)
    plotted = 0
    for method, label, color, linestyle, linewidth, alpha in methods:
        steps, means, stds = aggregate(curves, dataset, method, metric)
        if steps.size == 0:
            continue
        ax.plot(
            steps,
            means,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
        )
        ax.fill_between(steps, means - stds, means + stds, color=color, alpha=alpha, linewidth=0)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return False

    metric_label = {
        "val_loss": "validation loss",
        "val_ppl": "validation PPL",
        "train_loss": "training loss",
    }[metric]
    ax.set_title(f"{dataset_label} E1 {metric_label}, {variant_label}, mean +/- std", fontsize=12, pad=9)
    ax.set_xlabel("step")
    ax.set_ylabel(metric_label)
    ax.set_xlim(START_STEP, END_STEP)
    ax.set_xticks(XTICKS)
    ax.grid(True, color="#d8d8d8", linewidth=0.65, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
        title="Curve",
        loc="upper right",
        frameon=True,
        framealpha=0.90,
        fontsize=6.8,
        title_fontsize=7.4,
        ncol=2,
        borderpad=0.6,
        labelspacing=0.35,
        columnspacing=0.8,
        handlelength=2.5,
    )
    legend.get_frame().set_edgecolor("#bbbbbb")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    svg_text = out_path.read_text()
    out_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
    plt.close(fig)
    return True


def write_readme(out_dir: Path, curves, uses_safe_matrixpolicy: bool, uses_global_rational_controls: bool) -> None:
    if len(DATASETS) == 1:
        completed = DATASETS[0][1]
    else:
        completed = ", ".join(label for _, label in DATASETS[:-1]) + f", and {DATASETS[-1][1]}"
    matrixpolicy_note = (
        "MatrixPolicy curves and tables use the replacement JSONL rows passed with "
        "`--matrixpolicy-manifest`."
        if uses_safe_matrixpolicy
        else "MatrixPolicy curves and tables use the manifest rows passed to this generator."
    )
    control_note = (
        " Non-MatrixPolicy RLB optimizer controls use the global-rational/no-local-atom (`rlb_fused_global_rational`) "
        "replacement rows passed with `--replacement-manifest`; SiLU controls use the clean main E1 rows "
        "plus the completed FineWeb-Edu seed-2027 runtime repair overlay where applicable."
        if uses_global_rational_controls
        else " Other methods use the clean main E1 rows plus the completed FineWeb-Edu seed-2027 runtime repair overlay where applicable."
    )
    sections = [
        "# ICLR26 E1 Dense Curve Figures",
        "",
        f"Completed E1 M0/100M datasets: {completed}. Figures use every native JSONL log point from step 500 through 3050. Validation curves use every 50-step eval; training-loss curves use every 10-step train log. Shaded bands are mean +/- 1 sample std over three seeds.",
        "",
        matrixpolicy_note + control_note,
        "",
        final_snapshot(curves),
    ]
    for dataset, label in DATASETS:
        sections.extend(
            [
                "",
                f"## {label}",
                "",
                "All-method view:",
                "",
                f"![{label} E1 validation loss mean +/- std, all methods]({dataset}_core_validation_loss_mean_std.svg)",
                "",
                f"![{label} E1 validation PPL mean +/- std, all methods]({dataset}_core_validation_ppl_mean_std.svg)",
                "",
                f"![{label} E1 training loss mean +/- std, all methods]({dataset}_core_training_loss_mean_std.svg)",
                "",
                "Clean comparison view:",
                "",
                f"![{label} E1 validation loss mean +/- std, clean comparison]({dataset}_clean_validation_loss_mean_std.svg)",
                "",
                f"![{label} E1 validation PPL mean +/- std, clean comparison]({dataset}_clean_validation_ppl_mean_std.svg)",
                "",
                f"![{label} E1 training loss mean +/- std, clean comparison]({dataset}_clean_training_loss_mean_std.svg)",
                "",
                checkpoint_table(curves, dataset, label),
            ]
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(sections) + "\n"
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    (out_dir / "README.md").write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("experiments/manifests/iclr26_main_manifest.csv"))
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/iclr26_main/E1_m0_100m"))
    parser.add_argument("--matrixpolicy-manifest", type=Path, default=None)
    parser.add_argument("--matrixpolicy-run-root", type=Path, default=None)
    parser.add_argument("--matrixpolicy-phase", type=str, default=None)
    parser.add_argument("--replacement-manifest", type=Path, default=None)
    parser.add_argument("--replacement-run-root", type=Path, default=None)
    parser.add_argument("--replacement-phase", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/iclr26_e1_figures"))
    parser.add_argument("--status-md", type=Path, default=None)
    args = parser.parse_args()

    curves = parse_jsonl_runs(
        args.manifest,
        args.run_root,
        args.matrixpolicy_manifest,
        args.matrixpolicy_run_root,
        args.matrixpolicy_phase,
        args.replacement_manifest,
        args.replacement_run_root,
        args.replacement_phase,
    )
    if args.status_md is not None:
        update_status_tables(args.status_md, curves)

    suffixes = {
        "val_loss": "validation_loss_mean_std.svg",
        "val_ppl": "validation_ppl_mean_std.svg",
        "train_loss": "training_loss_mean_std.svg",
    }
    variants = [
        ("core", "all methods", ALL_METHODS),
        ("clean", "clean comparison", CLEAN_METHODS),
    ]
    made = []
    for variant, variant_label, methods in variants:
        for dataset, label in DATASETS:
            for metric, suffix in suffixes.items():
                out_path = args.out_dir / f"{dataset}_{variant}_{suffix}"
                if plot_dataset(curves, dataset, label, metric, out_path, methods, variant_label):
                    made.append(out_path)
    write_readme(
        args.out_dir,
        curves,
        args.matrixpolicy_manifest is not None
        and args.matrixpolicy_run_root is not None
        and args.matrixpolicy_phase is not None,
        args.replacement_manifest is not None
        and args.replacement_run_root is not None
        and args.replacement_phase is not None,
    )
    for path in made:
        print(path)
    print(args.out_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
