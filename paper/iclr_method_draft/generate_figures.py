#!/usr/bin/env python3
"""Generate manuscript figures from repository result artifacts."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = Path(__file__).resolve().parent
OUT_DIR = DRAFT_DIR / "figures"
TOKENS_PER_STEP = 32768

sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
import plot_iclr26_e1_curves as e1_curves  # noqa: E402
import plot_iclr26_e2_curves as e2_curves  # noqa: E402


DATASETS = [
    ("dclm", "DCLM", "iclr26_e2_dclm_2026_06_10"),
    ("fineweb_edu", "FineWeb-Edu", "iclr26_e2_fineweb_edu_2026_06_12"),
    ("fineweb", "FineWeb", "iclr26_e2_fineweb_2026_06_15"),
    ("dolma_sample", "Dolma", "iclr26_e2_dolma_sample_2026_06_17"),
    ("c4_en", "C4", "iclr26_e2_c4_2026_06_19"),
]

METHOD_STYLE = {
    "rlb_matrixpolicy_original": ("MatrixPolicy", "#111111", "-", 2.05),
    "rlb_adamw": ("RLB+AdamW", "#2a6fbb", "-", 1.45),
    "rlb_muon": ("RLB+Muon", "#198754", "-", 1.45),
    "silu_adamw": ("SiLU+AdamW", "#2a6fbb", "--", 1.35),
    "silu_muon": ("SiLU+Muon", "#198754", "--", 1.35),
}


def _box(ax, xy, width, height, text, fc, ec="#333333", fontsize=8.6, weight=None):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.05,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111111",
    )
    return patch


def _arrow(ax, start, end, color="#333333", lw=1.25, rad=0.0, scale=10):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def make_matrixpolicy_overview(out_path: Path) -> None:
    """Draw the activation/optimizer interface without hiding matrix roles."""

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    matrix = "#e7eef7"
    group = "#eaf5ef"
    curve = "#fff2d6"
    policy = "#f0e8f6"
    signal = "#f7f7f7"

    ax.text(0.04, 0.935, "RLB exposes a matrix-role interface", fontsize=11.2, weight="bold")
    ax.text(
        0.04,
        0.885,
        "MatrixPolicy updates only the RLB input/output matrices; rational coefficients and other tensors use AdamW.",
        fontsize=8.45,
        color="#333333",
    )

    _box(ax, (0.035, 0.61), 0.125, 0.145, "$x_l$", "#eeeeee", fontsize=9.2)
    _box(ax, (0.195, 0.59), 0.135, 0.185, "$A_l$\ninput\nmatrix", matrix, fontsize=8.5)
    _box(ax, (0.365, 0.59), 0.160, 0.185, "$z_{l,g}\\to u_{l,g}$\nRMS-normalized\ngroups", group, fontsize=8.0)
    _box(ax, (0.560, 0.59), 0.145, 0.185, "$R_{l,g}$\nP5/Q4\nglobal curve", curve, fontsize=8.1)
    _box(ax, (0.740, 0.59), 0.135, 0.185, "$B_l$\noutput\nmatrix", matrix, fontsize=8.5)
    _box(ax, (0.910, 0.61), 0.065, 0.145, "$y_l$", "#eeeeee", fontsize=9.2)

    for start, end in [
        ((0.160, 0.682), (0.195, 0.682)),
        ((0.330, 0.682), (0.365, 0.682)),
        ((0.525, 0.682), (0.560, 0.682)),
        ((0.705, 0.682), (0.740, 0.682)),
        ((0.875, 0.682), (0.910, 0.682)),
    ]:
        _arrow(ax, start, end)

    _box(ax, (0.140, 0.325), 0.175, 0.165, "relative\nrole pressure\n$\\pi_{l,g}$", signal)
    _box(ax, (0.365, 0.325), 0.175, 0.165, "curve gains\n$\\hat d_{l,g},\\hat o_{l,g}$", signal)
    _box(ax, (0.590, 0.325), 0.175, 0.165, "coefficient\nactivity\n$\\alpha_{l,g}$", signal)
    _box(ax, (0.785, 0.325), 0.145, 0.165, "pair ratio\n$\\gamma_{l,g}$", signal)

    _box(
        ax,
        (0.205, 0.065),
        0.590,
        0.145,
        "MatrixPolicy actions: centered group multipliers, early matrix-direction branch, bounded pair balancing",
        policy,
        fontsize=8.25,
    )

    _arrow(ax, (0.262, 0.590), (0.228, 0.490), color="#666666", rad=0.10)
    _arrow(ax, (0.445, 0.590), (0.452, 0.490), color="#666666", rad=-0.08)
    _arrow(ax, (0.632, 0.590), (0.675, 0.490), color="#666666", rad=0.08)
    _arrow(ax, (0.807, 0.590), (0.858, 0.490), color="#666666", rad=-0.08)
    for x in [0.228, 0.452, 0.675, 0.858]:
        _arrow(ax, (x, 0.325), (0.500, 0.210), color="#7250a3", lw=1.10, scale=9)
    _arrow(ax, (0.330, 0.210), (0.258, 0.590), color="#7250a3", rad=0.22, lw=1.10)
    _arrow(ax, (0.690, 0.210), (0.808, 0.590), color="#7250a3", rad=-0.22, lw=1.10)

    fig.tight_layout(pad=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def load_e1_curves():
    return e1_curves.parse_jsonl_runs(
        ROOT / "experiments" / "manifests" / "iclr26_main_manifest.csv",
        ROOT / "experiments" / "runs" / "iclr26_main" / "E1_m0_100m",
        matrixpolicy_manifest=ROOT
        / "experiments"
        / "manifests"
        / "iclr26_global_rational_matrixpolicy_manifest.csv",
        matrixpolicy_run_root=ROOT
        / "experiments"
        / "runs"
        / "iclr26_main"
        / "E1_rational_only_100m",
        matrixpolicy_phase="E1_rational_only_100m",
        replacement_manifest=ROOT
        / "experiments"
        / "manifests"
        / "iclr26_global_rational_optimizer_controls_manifest.csv",
        replacement_run_root=ROOT
        / "experiments"
        / "runs"
        / "iclr26_main"
        / "E1_global_rational_optimizers_100m",
        replacement_phase="E1_global_rational_optimizers_100m",
    )


def load_e2_curves():
    return e2_curves.parse_jsonl_runs(
        ROOT / "experiments" / "manifests" / "iclr26_main_manifest.csv",
        ROOT / "experiments" / "runs" / "iclr26_main" / "E2_m0_300m",
        matrixpolicy_manifest=ROOT
        / "experiments"
        / "manifests"
        / "iclr26_global_rational_matrixpolicy_manifest.csv",
        matrixpolicy_run_root=ROOT / "experiments" / "runs" / "iclr26_main" / "E2_rational_only_300m",
        matrixpolicy_phase="E2_rational_only_300m",
        replacement_manifest=ROOT
        / "experiments"
        / "manifests"
        / "iclr26_global_rational_optimizer_controls_manifest.csv",
        replacement_run_root=ROOT / "experiments" / "runs" / "iclr26_main" / "E2_global_rational_optimizers_300m",
        replacement_phase="E2_global_rational_optimizers_300m",
    )


def make_training_dynamics_examples(out_path: Path) -> None:
    e1 = load_e1_curves()
    panels = [("dclm", "DCLM"), ("fineweb_edu", "FineWeb-Edu")]
    metrics = [("val_loss", "validation loss"), ("val_ppl", "validation perplexity"), ("train_loss", "training loss")]
    methods = [
        "rlb_matrixpolicy_original",
        "rlb_adamw",
        "rlb_muon",
        "silu_adamw",
        "silu_muon",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8.3, 5.75), sharex=True)
    for col, (dataset, dataset_label) in enumerate(panels):
        for row, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for method in methods:
                label, color, linestyle, linewidth = METHOD_STYLE[method]
                steps, means, stds = e1_curves.aggregate(e1, dataset, method, metric)
                if steps.size == 0:
                    continue
                tokens = steps * TOKENS_PER_STEP / 1_000_000
                ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
                if method in {"rlb_matrixpolicy_original", "rlb_adamw", "silu_adamw"}:
                    ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.055, linewidth=0)
            if row == 0:
                ax.set_title(dataset_label, fontsize=10.3)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8.6)
            if row == 2:
                ax.set_xlabel("tokens (M)", fontsize=8.6)
            ax.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.75)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=7.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8.0)
    fig.tight_layout(rect=(0, 0, 1, 0.925), pad=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)



def make_e2_validation_all_datasets(out_path: Path) -> None:
    """Main-paper E2 validation-loss small multiples."""
    e2 = load_e2_curves()
    methods = [
        "rlb_matrixpolicy_original",
        "rlb_adamw",
        "rlb_muon",
        "silu_adamw",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(8.4, 2.35), sharex=True)
    for ax, (dataset, label, _dir_name) in zip(axes, DATASETS):
        for method in methods:
            method_label, color, linestyle, linewidth = METHOD_STYLE[method]
            if method == "rlb_muon":
                color = "#c87519"
            steps, means, stds = e2_curves.aggregate(e2, dataset, method, "val_loss")
            if steps.size == 0:
                continue
            tokens = steps * TOKENS_PER_STEP / 1_000_000
            ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, label=method_label)
            ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.065, linewidth=0)
        ax.set_title(label, fontsize=8.6)
        ax.set_xlabel("tokens (M)", fontsize=7.4)
        ax.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.72)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=6.8)
    axes[0].set_ylabel("validation loss", fontsize=7.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.6)
    fig.tight_layout(rect=(0, 0, 1, 0.86), pad=0.35, w_pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_e2_auxiliary_dynamics(out_path: Path) -> None:
    """Appendix E2 train-loss and validation-PPL small multiples."""
    e2 = load_e2_curves()
    methods = [
        "rlb_matrixpolicy_original",
        "rlb_adamw",
        "rlb_muon",
        "silu_adamw",
    ]
    metrics = [("val_ppl", "validation perplexity"), ("train_loss", "training loss")]
    fig, axes = plt.subplots(2, 5, figsize=(8.4, 3.85), sharex=True)
    for col, (dataset, label, _dir_name) in enumerate(DATASETS):
        for row, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for method in methods:
                method_label, color, linestyle, linewidth = METHOD_STYLE[method]
                if method == "rlb_muon":
                    color = "#c87519"
                steps, means, stds = e2_curves.aggregate(e2, dataset, method, metric)
                if steps.size == 0:
                    continue
                tokens = steps * TOKENS_PER_STEP / 1_000_000
                ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, label=method_label)
                ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.055, linewidth=0)
            if row == 0:
                ax.set_title(label, fontsize=8.2)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=7.5)
            if row == 1:
                ax.set_xlabel("tokens (M)", fontsize=7.2)
            ax.grid(True, color="#d8d8d8", linewidth=0.48, alpha=0.72)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=6.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.3)
    fig.tight_layout(rect=(0, 0, 1, 0.90), pad=0.35, w_pad=0.45, h_pad=0.65)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_runtime_tps() -> dict[tuple[str, str, str], float]:
    rows = _read_rows(
        ROOT
        / "experiments"
        / "results"
        / "iclr26_runtime_summary_2026_06_11"
        / "runtime_by_dataset_method_clean.csv"
    )
    tps: dict[tuple[str, str, str], float] = {}
    for row in rows:
        scope = row["scope"]
        if scope.startswith("E1_"):
            regime = "E1"
        elif scope.startswith("E2_"):
            regime = "E2"
        else:
            continue
        value = row.get("tokens_per_second_mean")
        if not value:
            continue
        tps[(regime, row["dataset"], row["method"])] = float(value)
    return tps


def _select_representative(rows: list[dict[str, str]], common_col: str, saved_col: str) -> dict[str, str] | None:
    candidates = []
    for row in rows:
        try:
            common = int(float(row[common_col]))
            target = float(row["target_loss"])
            saved = float(row[saved_col])
        except (KeyError, TypeError, ValueError):
            continue
        if common == 3 and saved > 0:
            candidates.append((target, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _mean_time_saving(
    regime: str,
    dataset: str,
    target: float,
    comparator: str,
    per_seed_rows: list[dict[str, str]],
    tps: dict[tuple[str, str, str], float],
) -> float | None:
    mp_tps = tps.get((regime, dataset, "rlb_matrixpolicy_original"))
    if not mp_tps:
        return None
    savings = []
    for row in per_seed_rows:
        if "dataset" in row and row["dataset"] != dataset:
            continue
        try:
            if not math.isclose(float(row["target_loss"]), target, rel_tol=0, abs_tol=1e-8):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        try:
            mp_tokens = float(row["matrixpolicy_tokens"])
        except (KeyError, TypeError, ValueError):
            continue
        if comparator == "silu_adamw":
            comp_method = "silu_adamw"
            comp_field = "silu_adamw_tokens"
        else:
            comp_method = row.get("second_best_method", "")
            comp_field = "second_best_tokens"
        comp_tps = tps.get((regime, dataset, comp_method))
        if not comp_tps:
            continue
        try:
            comp_tokens = float(row[comp_field])
        except (KeyError, TypeError, ValueError):
            continue
        savings.append(comp_tokens / comp_tps - mp_tokens / mp_tps)
    if not savings:
        return None
    return sum(savings) / len(savings) / 60.0


def _token_time_points() -> list[dict[str, object]]:
    tps = _load_runtime_tps()
    e1_dir = ROOT / "experiments" / "results" / "iclr26_e1_token_savings_2026_06_12"
    e1_agg = _read_rows(e1_dir / "token_savings.csv")
    e1_seed = _read_rows(e1_dir / "token_savings_per_seed.csv")
    e2_root = ROOT / "experiments" / "results"
    points: list[dict[str, object]] = []

    comparator_specs = [
        ("fastest non-MP", "second_best_common_seeds", "saved_tokens_vs_second_best", "saved_fraction_vs_second_best"),
        ("SiLU+AdamW", "silu_adamw_common_seeds", "saved_tokens_vs_silu_adamw", "saved_fraction_vs_silu_adamw"),
    ]

    for dataset, label, e2_name in DATASETS:
        e1_rows = [row for row in e1_agg if row["dataset"] == dataset]
        for comp_label, common_col, saved_col, frac_col in comparator_specs:
            row = _select_representative(e1_rows, common_col, saved_col)
            if row is None:
                continue
            target = float(row["target_loss"])
            time_saved = _mean_time_saving(
                "E1",
                dataset,
                target,
                "silu_adamw" if comp_label == "SiLU+AdamW" else "fastest",
                e1_seed,
                tps,
            )
            if time_saved is None:
                continue
            points.append(
                {
                    "regime": "E1",
                    "dataset": label,
                    "comparator": comp_label,
                    "target": target,
                    "tokens_saved_m": float(row[saved_col]) / 1_000_000,
                    "fraction": float(row[frac_col]),
                    "minutes_saved": time_saved,
                }
            )

        e2_dir = e2_root / e2_name
        e2_agg = _read_rows(e2_dir / "token_savings.csv")
        e2_seed = _read_rows(e2_dir / "token_savings_per_seed.csv")
        for comp_label, common_col, saved_col, frac_col in comparator_specs:
            row = _select_representative(e2_agg, common_col, saved_col)
            if row is None:
                continue
            target = float(row["target_loss"])
            time_saved = _mean_time_saving(
                "E2",
                dataset,
                target,
                "silu_adamw" if comp_label == "SiLU+AdamW" else "fastest",
                e2_seed,
                tps,
            )
            if time_saved is None:
                continue
            points.append(
                {
                    "regime": "E2",
                    "dataset": label,
                    "comparator": comp_label,
                    "target": target,
                    "tokens_saved_m": float(row[saved_col]) / 1_000_000,
                    "fraction": float(row[frac_col]),
                    "minutes_saved": time_saved,
                }
            )
    return points


def make_token_time_target_plot(out_path: Path) -> None:
    all_points = _token_time_points()
    points = [p for p in all_points if p["regime"] == "E2"]
    fig, ax = plt.subplots(figsize=(8.2, 3.45))
    colors = {"fastest non-MP": "#c05a28", "SiLU+AdamW": "#2a6fbb"}
    abbreviations = {"FineWeb-Edu": "FWE", "FineWeb": "FW", "DCLM": "DCLM", "Dolma": "Dolma", "C4": "C4"}

    for comp in ["fastest non-MP", "SiLU+AdamW"]:
        xs = [p["tokens_saved_m"] for p in points if p["comparator"] == comp]
        ys = [p["minutes_saved"] for p in points if p["comparator"] == comp]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=64,
            c=colors[comp],
            marker="o" if comp == "fastest non-MP" else "s",
            edgecolor="#222222",
            linewidth=0.6,
            alpha=0.94,
            label=f"vs {comp}",
        )

    for p in points:
        dx = 0.35 if p["comparator"] == "SiLU+AdamW" else -0.35
        ha = "left" if p["comparator"] == "SiLU+AdamW" else "right"
        ax.annotate(
            abbreviations[str(p["dataset"])],
            (p["tokens_saved_m"], p["minutes_saved"]),
            xytext=(dx, 0.12),
            textcoords="offset fontsize",
            fontsize=7.0,
            ha=ha,
            va="center",
            color="#333333",
        )

    ax.axhline(0, color="#888888", linewidth=0.85)
    ax.axvline(0, color="#888888", linewidth=0.85)
    ax.set_xlabel("tokens saved to matched E2 target (M)")
    ax.set_ylabel("estimated wall-clock saved (min)")
    ax.set_title("E2 target arrival: fewer tokens and less training time", fontsize=10.3)
    ax.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.7, frameon=True, framealpha=0.94, borderpad=0.45)
    fig.tight_layout(pad=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    outputs = [
        OUT_DIR / "matrixpolicy_overview.pdf",
        OUT_DIR / "e2_validation_all_datasets.pdf",
        OUT_DIR / "token_time_target_savings.pdf",
        OUT_DIR / "e2_auxiliary_dynamics.pdf",
        OUT_DIR / "e1_multimetric_examples.pdf",
    ]
    make_matrixpolicy_overview(outputs[0])
    make_e2_validation_all_datasets(outputs[1])
    make_token_time_target_plot(outputs[2])
    make_e2_auxiliary_dynamics(outputs[3])
    make_training_dynamics_examples(outputs[4])
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
