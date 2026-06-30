#!/usr/bin/env python3
"""Generate manuscript figures and tables from repository result artifacts."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = Path(__file__).resolve().parent
OUT_DIR = DRAFT_DIR / "figures"
TABLE_DIR = DRAFT_DIR / "tables"
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

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

METHOD_STYLE = {
    "rlb_matrixpolicy_original": ("MatrixPolicy", OKABE_ITO["black"], "-", 2.05, "o"),
    "silu_adamw": ("SiLU+AdamW", OKABE_ITO["blue"], "--", 1.35, "s"),
    "rlb_adamw": ("RLB+AdamW", OKABE_ITO["blue"], "-", 1.45, "D"),
    "silu_muon": ("SiLU+Muon", OKABE_ITO["green"], "--", 1.30, "^"),
    "rlb_muon": ("RLB+Muon", OKABE_ITO["green"], "-", 1.40, "v"),
    "rlb_lion": ("RLB+Lion", OKABE_ITO["orange"], "-", 1.25, "P"),
}

MAIN_METHODS = [
    "rlb_matrixpolicy_original",
    "silu_adamw",
    "rlb_adamw",
    "silu_muon",
    "rlb_muon",
]
TABLE_COMPARATORS = [
    ("silu_adamw", "SiLU+AdamW"),
    ("rlb_adamw", "RLB+AdamW"),
    ("rlb_muon", "RLB+Muon"),
]


def _box(ax, xy, width, height, text, fc, ec="#333333", fontsize=8.6, weight=None):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.020",
        linewidth=1.0,
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


def _arrow(ax, start, end, color="#333333", lw=1.2, rad=0.0, scale=10):
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

    fig, ax = plt.subplots(figsize=(7.2, 3.55), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    matrix = "#e7eef7"
    group = "#eaf5ef"
    curve = "#fff2d6"
    policy = "#f0e8f6"
    signal = "#f7f7f7"

    ax.text(0.04, 0.935, "RLB turns the nonlinear sublayer into a role-visible interface", fontsize=11.0, weight="bold")
    ax.text(
        0.04,
        0.885,
        "The optimizer sees the input matrix, normalized rational groups, and output matrix as different objects.",
        fontsize=8.35,
        color="#333333",
    )

    _box(ax, (0.035, 0.61), 0.125, 0.145, "$x_l$", "#eeeeee", fontsize=9.2)
    _box(ax, (0.195, 0.59), 0.135, 0.185, "$A_l$\ninput\nmatrix", matrix, fontsize=8.5)
    _box(ax, (0.365, 0.59), 0.160, 0.185, "$z_{l,g}\to u_{l,g}$\nRMS-normalized\ngroups", group, fontsize=8.0)
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
        "MatrixPolicy actions: centered group multipliers, gated early Muon substep, bounded pair balancing",
        policy,
        fontsize=8.15,
    )

    _arrow(ax, (0.262, 0.590), (0.228, 0.490), color="#666666", rad=0.10)
    _arrow(ax, (0.445, 0.590), (0.452, 0.490), color="#666666", rad=-0.08)
    _arrow(ax, (0.632, 0.590), (0.675, 0.490), color="#666666", rad=0.08)
    _arrow(ax, (0.807, 0.590), (0.858, 0.490), color="#666666", rad=-0.08)
    for x in [0.228, 0.452, 0.675, 0.858]:
        _arrow(ax, (x, 0.325), (0.500, 0.210), color="#7250a3", lw=1.05, scale=9)
    _arrow(ax, (0.330, 0.210), (0.258, 0.590), color="#7250a3", rad=0.22, lw=1.05)
    _arrow(ax, (0.690, 0.210), (0.808, 0.590), color="#7250a3", rad=-0.22, lw=1.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_matrixpolicy_signal_flow(out_path: Path) -> None:
    """Draw how the RLB statistics are converted into concrete optimizer actions."""

    fig, ax = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    obs = "#edf4fb"
    stat = "#f7f7f7"
    action = "#f1e7f6"
    guard = "#fff0d9"

    ax.text(0.045, 0.925, "MatrixPolicy uses RLB statistics as bounded control signals", fontsize=10.8, weight="bold")
    ax.text(
        0.045,
        0.872,
        "The rules are not full natural-gradient preconditioners; they are clipped, centered, and role-specific matrix updates.",
        fontsize=8.15,
        color="#333333",
    )

    xs = [0.055, 0.285, 0.515, 0.745]
    top_text = [
        "matrix-gradient\nRMS ratios",
        "live curve\nresponse samples",
        "rational-coefficient\ngradient energy",
        "fixed probe-grid\ncurve diagnostics",
    ]
    mid_text = [
        "$\\pi_{l,g}$\ninput vs output\npressure",
        "$\\hat d_{l,g},\\hat o_{l,g}$\nderivative and\nresponse gains",
        "$\\alpha_{l,g}$\ncoefficient\nactivity",
        "$\\tau_{l,g}$\nheuristic target\nratio",
    ]
    for x, text in zip(xs, top_text):
        _box(ax, (x, 0.64), 0.175, 0.145, text, obs, fontsize=8.0)
    for x, text in zip(xs, mid_text):
        _box(ax, (x, 0.385), 0.175, 0.170, text, stat, fontsize=7.75)
    for x in xs:
        _arrow(ax, (x + 0.0875, 0.64), (x + 0.0875, 0.555), color="#555555", lw=1.0, scale=8)

    _box(ax, (0.060, 0.105), 0.255, 0.155, "centered group\ngradient multipliers\nfor $A_l$ and $B_l$", action, fontsize=8.0)
    _box(ax, (0.372, 0.105), 0.255, 0.155, "early gated Muon\nsubstep on matrices\nwith separate state", action, fontsize=8.0)
    _box(ax, (0.685, 0.105), 0.255, 0.155, "bounded positive\npair-balancing move\n$A_{l,g},B_{l,g}$", action, fontsize=8.0)
    _box(ax, (0.372, 0.300), 0.255, 0.055, "clipping and schedules keep every rule local and bounded", guard, fontsize=7.4)

    for start, end in [
        ((0.142, 0.385), (0.188, 0.260)),
        ((0.372, 0.385), (0.188, 0.260)),
        ((0.602, 0.385), (0.500, 0.260)),
        ((0.832, 0.385), (0.812, 0.260)),
        ((0.455, 0.385), (0.500, 0.355)),
        ((0.603, 0.385), (0.500, 0.355)),
    ]:
        _arrow(ax, start, end, color="#7250a3", lw=1.05, scale=8, rad=0.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
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


def _style(method: str):
    return METHOD_STYLE[method]


def _finish_axis(ax, labelsize=7.8):
    ax.grid(True, color="#d8d8d8", linewidth=0.52, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=labelsize)


def _legend_panel(ax, handles, labels, fontsize=7.8):
    ax.axis("off")
    ax.legend(handles, labels, loc="center", frameon=False, fontsize=fontsize, handlelength=2.6)


def make_e1_validation_all_datasets(out_path: Path) -> None:
    """Main-paper E1 validation-loss trajectories on all datasets."""
    e1 = load_e1_curves()
    fig, axes_grid = plt.subplots(2, 3, figsize=(7.2, 4.25), sharex=True, constrained_layout=True)
    axes = axes_grid.ravel()
    for idx, (ax, (dataset, label, _dir_name)) in enumerate(zip(axes, DATASETS)):
        for method in MAIN_METHODS:
            method_label, color, linestyle, linewidth, marker = _style(method)
            steps, means, stds = e1_curves.aggregate(e1, dataset, method, "val_loss")
            if steps.size == 0:
                continue
            tokens = steps * TOKENS_PER_STEP / 1_000_000
            ax.plot(
                tokens,
                means,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=method_label,
                marker=marker,
                markevery=max(1, len(tokens) // 4),
                markersize=3.4,
            )
            ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.040, linewidth=0)
        ax.set_title(label, fontsize=9.2)
        if idx % 3 == 0:
            ax.set_ylabel("validation loss", fontsize=8.4)
        _finish_axis(ax, labelsize=7.8)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_panel(axes[-1], handles, labels, fontsize=7.8)
    fig.supxlabel("tokens (M)", fontsize=8.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _target_candidates(dataset: str) -> list[float]:
    rows = _read_rows(ROOT / "experiments" / "results" / "iclr26_e1_token_savings_2026_06_12" / "token_savings.csv")
    return sorted({float(row["target_loss"]) for row in rows if row["dataset"] == dataset}, reverse=True)


def _first_hit_tokens(curves, dataset: str, method: str, target_loss: float) -> list[float]:
    hits: list[float] = []
    seed_data = curves.get(dataset, {}).get(method, {})
    for events in seed_data.values():
        hit = None
        for step, values in sorted(events["eval"].items()):
            loss = values.get("val_loss")
            if isinstance(loss, (int, float)) and math.isfinite(loss) and loss <= target_loss:
                hit = float(step * TOKENS_PER_STEP)
                break
        if hit is not None:
            hits.append(hit)
    return hits


def _mean_hit_tokens(curves, dataset: str, method: str, target_loss: float) -> float | None:
    stats = _hit_token_stats(curves, dataset, method, target_loss)
    if stats is None:
        return None
    return stats[0]


def _hit_token_stats(curves, dataset: str, method: str, target_loss: float) -> tuple[float, float] | None:
    hits = _first_hit_tokens(curves, dataset, method, target_loss)
    if len(hits) < 3:
        return None
    mean = float(np.mean(hits))
    std = float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0
    return mean, std


def make_e1_target_frontiers(out_path: Path) -> None:
    """Plot target validation loss versus tokens to first hit that target."""
    e1 = load_e1_curves()
    fig, axes_grid = plt.subplots(2, 3, figsize=(7.2, 4.25), sharex=False, sharey=False, constrained_layout=True)
    axes = axes_grid.ravel()
    for idx, (ax, (dataset, label, _dir_name)) in enumerate(zip(axes, DATASETS)):
        targets = _target_candidates(dataset)
        for method in MAIN_METHODS:
            method_label, color, linestyle, linewidth, marker = _style(method)
            xs, ys, yerrs = [], [], []
            for target in targets:
                stats = _hit_token_stats(e1, dataset, method, target)
                if stats is None:
                    continue
                mean_tokens, std_tokens = stats
                xs.append(target)
                ys.append(mean_tokens / 1_000_000)
                yerrs.append(std_tokens / 1_000_000)
            if not xs:
                continue
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=3.6 if method == "rlb_matrixpolicy_original" else 3.2,
                label=method_label,
                alpha=0.96,
                elinewidth=0.55,
                capsize=1.8,
                capthick=0.55,
            )
        ax.invert_xaxis()
        ax.set_title(label, fontsize=9.2)
        if idx % 3 == 0:
            ax.set_ylabel("tokens to target (M)", fontsize=8.4)
        _finish_axis(ax, labelsize=7.8)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_panel(axes[-1], handles, labels, fontsize=7.8)
    fig.supxlabel("target validation loss", fontsize=8.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_e1_multimetric_examples(out_path: Path) -> None:
    e1 = load_e1_curves()
    panels = [("dclm", "DCLM"), ("fineweb_edu", "FineWeb-Edu")]
    metrics = [("val_loss", "validation loss"), ("val_ppl", "validation perplexity"), ("train_loss", "training loss")]

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.55), sharex=True, constrained_layout=True)
    for col, (dataset, dataset_label) in enumerate(panels):
        for row, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for method in MAIN_METHODS:
                label, color, linestyle, linewidth, marker = _style(method)
                steps, means, stds = e1_curves.aggregate(e1, dataset, method, metric)
                if steps.size == 0:
                    continue
                tokens = steps * TOKENS_PER_STEP / 1_000_000
                ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, label=label, marker=marker, markevery=max(1, len(tokens)//4), markersize=3.0)
                ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.040, linewidth=0)
            if row == 0:
                ax.set_title(dataset_label, fontsize=10.3)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8.6)
            if row == 2:
                ax.set_xlabel("tokens (M)", fontsize=8.6)
            _finish_axis(ax)
            ax.tick_params(axis="both", labelsize=7.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=7.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_e2_metric_dynamics(out_path: Path, metric: str, metric_label: str) -> None:
    """Appendix E2 dynamics for one metric across all datasets."""
    e2 = load_e2_curves()
    fig, axes_grid = plt.subplots(2, 3, figsize=(7.2, 4.15), sharex=True, constrained_layout=True)
    axes = axes_grid.ravel()
    for idx, (ax, (dataset, label, _dir_name)) in enumerate(zip(axes, DATASETS)):
        for method in MAIN_METHODS:
            method_label, color, linestyle, linewidth, marker = _style(method)
            steps, means, stds = e2_curves.aggregate(e2, dataset, method, metric)
            if steps.size == 0:
                continue
            tokens = steps * TOKENS_PER_STEP / 1_000_000
            ax.plot(
                tokens,
                means,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=method_label,
                marker=marker,
                markevery=max(1, len(tokens) // 4),
                markersize=3.1,
            )
            ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.035, linewidth=0)
        ax.set_title(label, fontsize=9.0)
        if idx % 3 == 0:
            ax.set_ylabel(metric_label, fontsize=8.3)
        _finish_axis(ax, labelsize=7.6)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_panel(axes[-1], handles, labels, fontsize=7.6)
    fig.supxlabel("tokens (M)", fontsize=8.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


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
        if value:
            tps[(regime, row["dataset"], row["method"])] = float(value)
    return tps


def _select_hard_common_target(curves, dataset: str, methods: list[str]) -> float | None:
    # Lower validation loss is a harder pre-specified target, so sort ascending.
    for target in sorted(_target_candidates(dataset)):
        if all(_mean_hit_tokens(curves, dataset, method, target) is not None for method in methods):
            return target
    return None


def _fmt_token_fraction(saved_tokens_m: float, saved_fraction: float) -> str:
    return f"{saved_tokens_m:.1f} ({100.0 * saved_fraction:.1f}\\%)"


def make_e1_target_time_table(out_path: Path) -> None:
    curves = load_e1_curves()
    tps = _load_runtime_tps()
    rows = []
    required_methods = ["rlb_matrixpolicy_original"] + [method for method, _ in TABLE_COMPARATORS]
    for dataset, label, _dir_name in DATASETS:
        target = _select_hard_common_target(curves, dataset, required_methods)
        if target is None:
            continue
        mp_stats = _hit_token_stats(curves, dataset, "rlb_matrixpolicy_original", target)
        mp_tps = tps.get(("E1", dataset, "rlb_matrixpolicy_original"))
        if mp_stats is None or mp_tps is None:
            continue
        mp_tokens, mp_std = mp_stats
        cells = []
        for method, _method_label in TABLE_COMPARATORS:
            comp_stats = _hit_token_stats(curves, dataset, method, target)
            comp_tps = tps.get(("E1", dataset, method))
            if comp_stats is None or comp_tps is None:
                cells.extend(["--", "--"])
                continue
            comp_tokens, _comp_std = comp_stats
            saved_tokens = comp_tokens - mp_tokens
            saved_minutes = (comp_tokens / comp_tps - mp_tokens / mp_tps) / 60.0
            saved_fraction = saved_tokens / comp_tokens
            cells.extend([_fmt_token_fraction(saved_tokens / 1_000_000, saved_fraction), f"{saved_minutes:.1f}"])
        rows.append((label, target, mp_tokens / 1_000_000, mp_std / 1_000_000, cells))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{E1 target-arrival savings at the hardest pre-specified candidate-grid validation-loss target reached by MatrixPolicy and all listed comparators in all three seeds. Token columns report saved tokens in millions with the percentage of comparator tokens in parentheses; time columns report estimated saved minutes.}",
        r"\label{tab:e1-target-time}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}lccrrrrrr@{}}",
        r"\toprule",
        r"Dataset & Target & MP tokens & \multicolumn{2}{c}{vs SiLU+AdamW} & \multicolumn{2}{c}{vs RLB+AdamW} & \multicolumn{2}{c}{vs RLB+Muon} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
        r" &  & mean$\pm$sd (M) & $\Delta$tok (M,\%) & $\Delta$t (min) & $\Delta$tok (M,\%) & $\Delta$t (min) & $\Delta$tok (M,\%) & $\Delta$t (min) \\",
        r"\midrule",
    ]
    for label, target, mp_tokens_m, mp_std_m, cells in rows:
        lines.append(
            f"{label} & {target:.2f} & {mp_tokens_m:.1f}$\\pm${mp_std_m:.1f} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {cells[4]} & {cells[5]} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{9}{@{}p{0.98\textwidth}@{}}{\footnotesize Tokens-to-target are read at the native 50-step evaluation cadence from the completed E1 JSONL logs, so arrivals are quantized in 1.64M-token increments. Time estimates use cleaned per-dataset/method training-throughput summaries.}",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    outputs = [
        OUT_DIR / "matrixpolicy_overview.pdf",
        OUT_DIR / "matrixpolicy_signal_flow.pdf",
        OUT_DIR / "e1_validation_all_datasets.pdf",
        OUT_DIR / "e1_target_frontiers.pdf",
        OUT_DIR / "e1_multimetric_examples.pdf",
        OUT_DIR / "e2_validation_dynamics.pdf",
        OUT_DIR / "e2_perplexity_dynamics.pdf",
        OUT_DIR / "e2_training_dynamics.pdf",
        TABLE_DIR / "e1_target_time_table.tex",
    ]
    make_matrixpolicy_overview(outputs[0])
    make_matrixpolicy_signal_flow(outputs[1])
    make_e1_validation_all_datasets(outputs[2])
    make_e1_target_frontiers(outputs[3])
    make_e1_multimetric_examples(outputs[4])
    make_e2_metric_dynamics(outputs[5], "val_loss", "validation loss")
    make_e2_metric_dynamics(outputs[6], "val_ppl", "validation perplexity")
    make_e2_metric_dynamics(outputs[7], "train_loss", "training loss")
    make_e1_target_time_table(outputs[8])
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
