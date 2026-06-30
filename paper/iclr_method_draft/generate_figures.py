#!/usr/bin/env python3
"""Generate manuscript figures and tables from repository result artifacts."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
DATASET_ABBR = {
    "DCLM": "DCLM",
    "FineWeb-Edu": "FWE",
    "FineWeb": "FW",
    "Dolma": "Dolma",
    "C4": "C4",
}

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
    "silu_muon": ("SiLU+Muon", OKABE_ITO["green"], "--", 1.35, "^"),
    "rlb_adamw": ("RLB+AdamW", OKABE_ITO["blue"], "-", 1.45, "D"),
    "rlb_muon": ("RLB+Muon", OKABE_ITO["green"], "-", 1.45, "v"),
    "rlb_lion": ("RLB+Lion", OKABE_ITO["orange"], "-", 1.25, "P"),
}
MAIN_SILU_METHODS = ["rlb_matrixpolicy_original", "silu_adamw", "silu_muon"]
FULL_METHODS = ["rlb_matrixpolicy_original", "silu_adamw", "silu_muon", "rlb_adamw", "rlb_muon"]
SILU_COMPARATORS = [("silu_adamw", "SiLU+AdamW"), ("silu_muon", "SiLU+Muon")]
TABLE_COMPARATORS = [("silu_adamw", "SiLU+AdamW"), ("rlb_adamw", "RLB+AdamW"), ("rlb_muon", "RLB+Muon")]


def _box(ax, xy, width, height, text, fc, ec="#333333", fontsize=8.0, weight=None, lw=0.9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=lw,
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
        linespacing=1.1,
    )
    return patch


def _arrow(ax, start, end, color="#333333", lw=1.05, rad=0.0, scale=9, style="-|>", linestyle="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def _matrix_glyph(ax, xy, width, height, label, orientation="rows"):
    ax.add_patch(Rectangle(xy, width, height, facecolor="#e7eef7", edgecolor="#2f4f6f", linewidth=1.0))
    colors = ["#cfe1f2", "#e7eef7", "#cfe1f2", "#e7eef7"]
    if orientation == "rows":
        band_h = height / 4
        for i, color in enumerate(colors):
            ax.add_patch(Rectangle((xy[0], xy[1] + i * band_h), width, band_h, facecolor=color, edgecolor="white", linewidth=0.45))
    else:
        band_w = width / 4
        for i, color in enumerate(colors):
            ax.add_patch(Rectangle((xy[0] + i * band_w, xy[1]), band_w, height, facecolor=color, edgecolor="white", linewidth=0.45))
    ax.text(xy[0] + width / 2, xy[1] + height + 0.028, label, ha="center", va="bottom", fontsize=8.2)


def make_matrixpolicy_overview(out_path: Path) -> None:
    """RLB forward map and optimizer-visible interface."""

    fig, ax = plt.subplots(figsize=(7.2, 4.05), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#202020"
    muted = "#666666"
    forward = "#202020"
    observe = "#7a7a7a"
    policy = "#6f4aa4"
    blue = "#dcebf7"
    green = "#e7f5ed"
    amber = "#fff1d1"
    gray = "#f2f2f2"
    lavender = "#eee7f7"

    ax.text(0.018, 0.955, "a", fontsize=11, weight="bold", va="top")
    ax.text(0.048, 0.955, "RLB forward map", fontsize=9.4, weight="bold", va="top", color=ink)
    ax.text(0.018, 0.430, "b", fontsize=11, weight="bold", va="top")
    ax.text(0.048, 0.430, "Detached summaries form the optimizer interface", fontsize=9.4, weight="bold", va="top", color=ink)

    # Forward computation lane.
    y = 0.705
    _box(ax, (0.030, y - 0.045), 0.055, 0.090, "$x_l$", gray, fontsize=9.2, lw=0.8)

    ax.add_patch(Rectangle((0.120, y - 0.105), 0.065, 0.210, facecolor=blue, edgecolor="#325d84", linewidth=1.0))
    for i, color in enumerate(["#c5dcf0", "#e8f1f9", "#c5dcf0", "#e8f1f9"]):
        ax.add_patch(Rectangle((0.120, y - 0.105 + i * 0.0525), 0.065, 0.0525, facecolor=color, edgecolor="white", linewidth=0.5))
    ax.text(0.1525, y + 0.132, "$A_l$", ha="center", fontsize=8.6, weight="bold")
    ax.text(0.1525, y - 0.142, "input matrix", ha="center", fontsize=6.6, color=muted)

    _box(ax, (0.222, y - 0.085), 0.122, 0.170, "$z_l=A_lx_l$\npartition into\n$z_{l,g}$", green, fontsize=7.1, lw=0.85)
    for yy in [y - 0.052, y - 0.018, y + 0.018, y + 0.052]:
        ax.plot([0.235, 0.331], [yy, yy], color="#8db7a0", linewidth=0.45)

    _box(ax, (0.380, y - 0.090), 0.135, 0.180, "$r_{l,g}=\\mathrm{RMS}(z_{l,g})$\n$u_{l,g}=z_{l,g}/r_{l,g}$", green, fontsize=7.1, lw=0.85)

    ax.add_patch(FancyBboxPatch((0.552, y - 0.108), 0.150, 0.216, boxstyle="round,pad=0.014,rounding_size=0.018", linewidth=0.9, facecolor=amber, edgecolor="#8a6d2a"))
    ax.text(0.627, y + 0.080, "global response", ha="center", fontsize=7.3, weight="bold")
    ax.text(0.627, y + 0.052, "$R_{l,g}(u)=P_5(u)/Q_4(u)$", ha="center", fontsize=6.8)
    ax.text(0.627, y + 0.027, "$Q_4(u)\\geq 1$", ha="center", fontsize=6.8)
    xs = np.linspace(-1, 1, 100)
    ys = y - 0.030 + 0.050 * (xs / (1.0 + 0.75 * np.abs(xs)) + 0.10 * xs**2)
    ax.plot(0.584 + 0.087 * (xs + 1) / 2, ys, color="#7a4b00", linewidth=1.25)
    ax.plot([0.584, 0.671], [y - 0.030, y - 0.030], color="#b89042", linewidth=0.45)

    _box(ax, (0.738, y - 0.072), 0.095, 0.144, "$h_{l,g}$\n$=r_{l,g}R_{l,g}(u_{l,g})$", green, fontsize=6.8, lw=0.85)

    ax.add_patch(Rectangle((0.870, y - 0.105), 0.065, 0.210, facecolor=blue, edgecolor="#325d84", linewidth=1.0))
    for i, color in enumerate(["#c5dcf0", "#e8f1f9", "#c5dcf0", "#e8f1f9"]):
        ax.add_patch(Rectangle((0.870 + i * 0.01625, y - 0.105), 0.01625, 0.210, facecolor=color, edgecolor="white", linewidth=0.45))
    ax.text(0.9025, y + 0.132, "$B_l$", ha="center", fontsize=8.6, weight="bold")
    ax.text(0.9025, y - 0.142, "output matrix", ha="center", fontsize=6.6, color=muted)
    _box(ax, (0.958, y - 0.045), 0.036, 0.090, "$y_l$", gray, fontsize=9.2, lw=0.8)

    for start, end in [
        ((0.085, y), (0.120, y)), ((0.185, y), (0.222, y)),
        ((0.344, y), (0.380, y)), ((0.515, y), (0.552, y)),
        ((0.702, y), (0.738, y)), ((0.833, y), (0.870, y)),
        ((0.935, y), (0.958, y)),
    ]:
        _arrow(ax, start, end, color=forward, lw=1.0, scale=8)

    # Interface summaries.
    chip_specs = [
        (0.060, 0.260, 0.150, 0.102, "role pressure", "$\\pi_{l,g}$\nfrom $\\nabla A,\\nabla B$", (0.153, y - 0.105)),
        (0.242, 0.260, 0.145, 0.102, "live gains", "$\\hat d^{live},\\hat o^{live}$\nfrom cached $u$", (0.450, y - 0.090)),
        (0.420, 0.260, 0.155, 0.102, "coefficient activity", "$\\alpha_{l,g}$\nfrom $\\nabla P,\\nabla Q$", (0.627, y - 0.108)),
        (0.608, 0.260, 0.155, 0.102, "pair target", "$\\gamma_{l,g},\\tau_{l,g}$\nfrom norms + probes", (0.903, y - 0.105)),
    ]
    for x0, y0, w, h, title, body, tap in chip_specs:
        _box(ax, (x0, y0), w, h, f"{title}\n{body}", "#f7f7f7", ec="#777777", fontsize=6.6, lw=0.75)
        _arrow(ax, tap, (x0 + w / 2, y0 + h), color=observe, lw=0.75, scale=6, linestyle="--")

    _box(ax, (0.800, 0.242), 0.165, 0.138, "MatrixPolicy\nupdates only\n$A_l,B_l$", lavender, ec=policy, fontsize=7.4, weight="bold", lw=0.9)
    _arrow(ax, (0.763, 0.310), (0.800, 0.310), color=policy, lw=1.0, scale=8)
    _arrow(ax, (0.842, 0.380), (0.155, y - 0.105), color=policy, lw=0.9, scale=7, rad=0.18)
    _arrow(ax, (0.908, 0.380), (0.902, y - 0.105), color=policy, lw=0.9, scale=7, rad=-0.12)

    ax.text(0.060, 0.140, "The rational coefficients define the curve and expose activity, but they are not MatrixPolicy parameters.", fontsize=7.0, color=muted)
    ax.text(0.060, 0.105, "All non-RLB-matrix tensors follow the ordinary AdamW path.", fontsize=7.0, color=muted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_matrixpolicy_signal_flow(out_path: Path) -> None:
    """MatrixPolicy optimizer-action schematic with explicit bypass lane."""

    fig, ax = plt.subplots(figsize=(7.2, 4.05), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#202020"
    muted = "#666666"
    purple = "#6f4aa4"
    lavender = "#eee7f7"
    pale_lav = "#f8f4fc"
    amber = "#fff1d1"
    gray = "#eeeeee"
    blue = "#dcebf7"

    ax.text(0.018, 0.952, "MatrixPolicy optimizer action", fontsize=9.8, weight="bold", color=ink, va="top")
    ax.text(0.018, 0.902, "RLB matrix lane: staged policy for $A_l$ and $B_l$ only", fontsize=8.3, color=muted)
    ax.plot([0.020, 0.980], [0.347, 0.347], color="#d2d2d2", linewidth=0.8, linestyle="--")
    ax.text(0.018, 0.303, "AdamW-only bypass lane", fontsize=8.3, color=muted)

    def stage(x0, y0, w, h, num, title, body, fc=lavender, ec=purple, fontsize=6.55):
        patch = _box(ax, (x0, y0), w, h, f"{title}\n{body}", fc, ec=ec, fontsize=fontsize, lw=0.9)
        circ = plt.Circle((x0 + 0.018, y0 + h - 0.018), 0.017, facecolor=ec, edgecolor="none", zorder=4)
        ax.add_patch(circ)
        ax.text(x0 + 0.018, y0 + h - 0.018, str(num), ha="center", va="center", fontsize=6.2, color="white", weight="bold", zorder=5)
        return patch

    y = 0.575
    h = 0.205
    stage(0.025, y, 0.150, h, 1, "RLB signals", "$\\pi,\\hat d^{live},\\hat o^{live}$\n$\\alpha\\to c$\n$\\gamma,\\tau\\to\\ell$", fc="#edf4fb", ec="#366b9a", fontsize=6.35)
    stage(0.205, y, 0.145, h, 2, "group multipliers", "$c_{l,g,\\mathrm{in/out}}$\nrole-wise center\nclip $[0.75,1.35]$", fontsize=6.25)
    stage(0.380, y, 0.125, h, 3, "scaled gradients", "$\\widetilde\\nabla A_{l,g}$\n$\\widetilde\\nabla B_{l,g}$", fontsize=6.55)

    _box(ax, (0.535, y), 0.180, h, "sequential matrix step", pale_lav, ec=purple, fontsize=7.0, weight="bold", lw=0.9)
    _box(ax, (0.552, y + 0.090), 0.066, 0.072, "AdamW\non $A,B$", "#ffffff", ec="#8b6bb0", fontsize=5.95, lw=0.7)
    _box(ax, (0.635, y + 0.090), 0.062, 0.072, "then\nMuon", "#ffffff", ec="#8b6bb0", fontsize=6.05, lw=0.7)
    ax.text(0.625, y + 0.044, "$\\eta_t s^{Adam}_{l,r}(1-\\mu)$; $\\eta_t\\mu_{l,r,t}$", ha="center", fontsize=5.9, color=muted)
    ax.text(0.625, y + 0.017, "early window; separate states", ha="center", fontsize=5.9, color=muted)
    _arrow(ax, (0.618, y + 0.126), (0.635, y + 0.126), color=purple, lw=0.75, scale=6)
    circ = plt.Circle((0.553, y + h - 0.018), 0.017, facecolor=purple, edgecolor="none", zorder=4)
    ax.add_patch(circ)
    ax.text(0.553, y + h - 0.018, "4", ha="center", va="center", fontsize=6.2, color="white", weight="bold", zorder=5)

    stage(0.745, y, 0.130, h, 5, "pair balance", "last; every 5 steps\n$\\ell\\in[-.030,.030]$\n$A_g\\gets e^\\ell A_g$\n$B_g\\gets e^{-\\ell}B_g$", fc=amber, ec="#9a6a00", fontsize=6.1)
    _box(ax, (0.910, y + 0.035), 0.070, 0.135, "updated\n$A_l^{t+1}$\n$B_l^{t+1}$", blue, ec="#325d84", fontsize=6.7, lw=0.85)

    for start, end in [
        ((0.175, y + 0.103), (0.205, y + 0.103)),
        ((0.350, y + 0.103), (0.380, y + 0.103)),
        ((0.505, y + 0.103), (0.535, y + 0.103)),
        ((0.715, y + 0.103), (0.745, y + 0.103)),
        ((0.875, y + 0.103), (0.910, y + 0.103)),
    ]:
        _arrow(ax, start, end, color=purple, lw=1.05, scale=8)

    _arrow(ax, (0.100, y), (0.745, y), color="#8a8a8a", lw=0.65, rad=-0.26, scale=6)
    ax.text(0.430, y - 0.030, "$\\gamma,\\tau$ route directly to balancing, not to the AdamW-only bypass", fontsize=6.15, color=muted, ha="center")

    _box(ax, (0.045, 0.122), 0.245, 0.118, "non-RLB-matrix parameters\nRLB rational coefficients, attention,\nembeddings, norms, ordinary weights", gray, ec="#999999", fontsize=6.35)
    _box(ax, (0.405, 0.130), 0.175, 0.102, "AdamW only\nno group multipliers\nno Muon", gray, ec="#999999", fontsize=6.65, weight="bold")
    _box(ax, (0.742, 0.130), 0.190, 0.102, "updated\n$\\theta_{\\neg AB}^{t+1}$", gray, ec="#999999", fontsize=7.0)
    _arrow(ax, (0.290, 0.181), (0.405, 0.181), color="#777777", lw=0.9, scale=8)
    _arrow(ax, (0.580, 0.181), (0.742, 0.181), color="#777777", lw=0.9, scale=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

def load_e1_curves():
    return e1_curves.parse_jsonl_runs(
        ROOT / "experiments" / "manifests" / "iclr26_main_manifest.csv",
        ROOT / "experiments" / "runs" / "iclr26_main" / "E1_m0_100m",
        matrixpolicy_manifest=ROOT / "experiments" / "manifests" / "iclr26_global_rational_matrixpolicy_manifest.csv",
        matrixpolicy_run_root=ROOT / "experiments" / "runs" / "iclr26_main" / "E1_rational_only_100m",
        matrixpolicy_phase="E1_rational_only_100m",
        replacement_manifest=ROOT / "experiments" / "manifests" / "iclr26_global_rational_optimizer_controls_manifest.csv",
        replacement_run_root=ROOT / "experiments" / "runs" / "iclr26_main" / "E1_global_rational_optimizers_100m",
        replacement_phase="E1_global_rational_optimizers_100m",
    )


def load_e2_curves():
    return e2_curves.parse_jsonl_runs(
        ROOT / "experiments" / "manifests" / "iclr26_main_manifest.csv",
        ROOT / "experiments" / "runs" / "iclr26_main" / "E2_m0_300m",
        matrixpolicy_manifest=ROOT / "experiments" / "manifests" / "iclr26_global_rational_matrixpolicy_manifest.csv",
        matrixpolicy_run_root=ROOT / "experiments" / "runs" / "iclr26_main" / "E2_rational_only_300m",
        matrixpolicy_phase="E2_rational_only_300m",
        replacement_manifest=ROOT / "experiments" / "manifests" / "iclr26_global_rational_optimizer_controls_manifest.csv",
        replacement_run_root=ROOT / "experiments" / "runs" / "iclr26_main" / "E2_global_rational_optimizers_300m",
        replacement_phase="E2_global_rational_optimizers_300m",
    )


def _style(method: str):
    return METHOD_STYLE[method]


def _finish_axis(ax, labelsize=7.5):
    ax.grid(True, color="#d8d8d8", linewidth=0.50, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=labelsize)


def _legend_panel(ax, handles, labels, fontsize=7.3):
    ax.axis("off")
    ax.legend(handles, labels, loc="center", frameon=False, fontsize=fontsize, handlelength=2.5)


def make_e1_multimetric_all_datasets(out_path: Path) -> None:
    e1 = load_e1_curves()
    metrics = [("val_loss", "validation loss"), ("val_ppl", "validation PPL"), ("train_loss", "training loss")]
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 5.95), sharex=True, constrained_layout=True)
    for col, (dataset, dataset_label, _dir_name) in enumerate(DATASETS):
        for row, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for method in MAIN_SILU_METHODS:
                label, color, linestyle, linewidth, marker = _style(method)
                steps, means, stds = e1_curves.aggregate(e1, dataset, method, metric)
                if steps.size == 0:
                    continue
                tokens = steps * TOKENS_PER_STEP / 1_000_000
                ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, marker=marker, markevery=max(1, len(tokens)//4), markersize=3.2, label=label)
                ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.030, linewidth=0)
            if row == 0:
                ax.set_title(dataset_label, fontsize=9.0)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8.2)
            if row == 2:
                ax.set_xlabel("tokens (M)", fontsize=8.0)
            _finish_axis(ax, labelsize=7.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_e1_validation_all_datasets(out_path: Path) -> None:
    e1 = load_e1_curves()
    fig, axes_grid = plt.subplots(2, 3, figsize=(7.2, 4.25), sharex=True, constrained_layout=True)
    axes = axes_grid.ravel()
    for idx, (ax, (dataset, label, _dir_name)) in enumerate(zip(axes, DATASETS)):
        for method in FULL_METHODS:
            method_label, color, linestyle, linewidth, marker = _style(method)
            steps, means, stds = e1_curves.aggregate(e1, dataset, method, "val_loss")
            if steps.size == 0:
                continue
            tokens = steps * TOKENS_PER_STEP / 1_000_000
            ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, label=method_label, marker=marker, markevery=max(1, len(tokens)//4), markersize=3.0)
            ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.030, linewidth=0)
        ax.set_title(label, fontsize=9.2)
        if idx % 3 == 0:
            ax.set_ylabel("validation loss", fontsize=8.4)
        _finish_axis(ax, labelsize=7.6)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_panel(axes[-1], handles, labels, fontsize=7.4)
    fig.supxlabel("tokens (M)", fontsize=8.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _target_candidates(regime: str, dataset: str, e2_dir_name: str | None = None) -> list[float]:
    if regime == "E1":
        rows = _read_rows(ROOT / "experiments" / "results" / "iclr26_e1_token_savings_2026_06_12" / "token_savings.csv")
        return sorted({float(row["target_loss"]) for row in rows if row.get("dataset") == dataset}, reverse=True)
    if not e2_dir_name:
        e2_dir_name = dict((d, e2) for d, _label, e2 in DATASETS)[dataset]
    rows = _read_rows(ROOT / "experiments" / "results" / e2_dir_name / "token_savings.csv")
    return sorted({float(row["target_loss"]) for row in rows}, reverse=True)


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


def _hit_token_stats(curves, dataset: str, method: str, target_loss: float) -> tuple[float, float] | None:
    hits = _first_hit_tokens(curves, dataset, method, target_loss)
    if len(hits) < 3:
        return None
    return float(np.mean(hits)), float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0


def _mean_hit_tokens(curves, dataset: str, method: str, target_loss: float) -> float | None:
    stats = _hit_token_stats(curves, dataset, method, target_loss)
    return None if stats is None else stats[0]


def _select_hard_common_target(curves, regime: str, dataset: str, methods: list[str], e2_dir_name: str | None = None) -> float | None:
    for target in sorted(_target_candidates(regime, dataset, e2_dir_name)):
        if all(_hit_token_stats(curves, dataset, method, target) is not None for method in methods):
            return target
    return None


def _final_eval_mean(curves, dataset: str, method: str, metric: str = "val_loss") -> tuple[float, float] | None:
    vals = []
    for events in curves.get(dataset, {}).get(method, {}).values():
        evals = events.get("eval", {})
        if not evals:
            continue
        last_step = max(evals)
        value = evals[last_step].get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            vals.append(float(value))
    if len(vals) < 3:
        return None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def make_e1_target_frontiers(out_path: Path) -> None:
    e1 = load_e1_curves()
    fig, axes_grid = plt.subplots(2, 3, figsize=(7.2, 4.25), sharex=False, sharey=False, constrained_layout=True)
    axes = axes_grid.ravel()
    for idx, (ax, (dataset, label, _dir_name)) in enumerate(zip(axes, DATASETS)):
        targets = _target_candidates("E1", dataset)
        for method in FULL_METHODS:
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
            ax.errorbar(xs, ys, yerr=yerrs, color=color, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=3.0, label=method_label, alpha=0.95, elinewidth=0.5, capsize=1.5)
        ax.invert_xaxis()
        ax.set_title(label, fontsize=9.0)
        if idx % 3 == 0:
            ax.set_ylabel("tokens to target (M)", fontsize=8.2)
        _finish_axis(ax, labelsize=7.4)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_panel(axes[-1], handles, labels, fontsize=7.2)
    fig.supxlabel("target validation loss", fontsize=8.5)
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
            for method in FULL_METHODS:
                label, color, linestyle, linewidth, marker = _style(method)
                steps, means, stds = e1_curves.aggregate(e1, dataset, method, metric)
                if steps.size == 0:
                    continue
                tokens = steps * TOKENS_PER_STEP / 1_000_000
                ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, label=label, marker=marker, markevery=max(1, len(tokens)//4), markersize=3.0)
                ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.035, linewidth=0)
            if row == 0:
                ax.set_title(dataset_label, fontsize=10.1)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8.4)
            if row == 2:
                ax.set_xlabel("tokens (M)", fontsize=8.4)
            _finish_axis(ax, labelsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=7.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_e2_metric_dynamics(out_path: Path, metric: str, metric_label: str) -> None:
    e2 = load_e2_curves()
    fig, axes_grid = plt.subplots(2, 3, figsize=(7.2, 4.15), sharex=True, constrained_layout=True)
    axes = axes_grid.ravel()
    for idx, (ax, (dataset, label, _dir_name)) in enumerate(zip(axes, DATASETS)):
        for method in FULL_METHODS:
            method_label, color, linestyle, linewidth, marker = _style(method)
            steps, means, stds = e2_curves.aggregate(e2, dataset, method, metric)
            if steps.size == 0:
                continue
            tokens = steps * TOKENS_PER_STEP / 1_000_000
            ax.plot(tokens, means, color=color, linestyle=linestyle, linewidth=linewidth, label=method_label, marker=marker, markevery=max(1, len(tokens)//4), markersize=3.0)
            ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=0.030, linewidth=0)
        ax.set_title(label, fontsize=8.9)
        if idx % 3 == 0:
            ax.set_ylabel(metric_label, fontsize=8.1)
        _finish_axis(ax, labelsize=7.3)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_panel(axes[-1], handles, labels, fontsize=7.2)
    fig.supxlabel("tokens (M)", fontsize=8.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _load_runtime_tps() -> dict[tuple[str, str, str], float]:
    rows = _read_rows(ROOT / "experiments" / "results" / "iclr26_runtime_summary_2026_06_11" / "runtime_by_dataset_method_clean.csv")
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


def _summary_rows() -> list[dict[str, object]]:
    curves_by_regime = {"E1": load_e1_curves(), "E2": load_e2_curves()}
    tps = _load_runtime_tps()
    rows: list[dict[str, object]] = []
    for regime, curves in curves_by_regime.items():
        for dataset, label, e2_dir_name in DATASETS:
            target = _select_hard_common_target(curves, regime, dataset, ["rlb_matrixpolicy_original", "silu_adamw", "silu_muon"], e2_dir_name)
            if target is None:
                continue
            mp_stats = _hit_token_stats(curves, dataset, "rlb_matrixpolicy_original", target)
            mp_final = _final_eval_mean(curves, dataset, "rlb_matrixpolicy_original", "val_loss")
            mp_tps = tps.get((regime, dataset, "rlb_matrixpolicy_original"))
            if mp_stats is None or mp_final is None or mp_tps is None:
                continue
            mp_tokens, mp_token_std = mp_stats
            for method, method_label in SILU_COMPARATORS:
                comp_stats = _hit_token_stats(curves, dataset, method, target)
                comp_final = _final_eval_mean(curves, dataset, method, "val_loss")
                comp_tps = tps.get((regime, dataset, method))
                if comp_stats is None or comp_final is None or comp_tps is None:
                    continue
                comp_tokens, comp_token_std = comp_stats
                saved_tokens = comp_tokens - mp_tokens
                saved_minutes = (comp_tokens / comp_tps - mp_tokens / mp_tps) / 60.0
                endpoint_gap = comp_final[0] - mp_final[0]
                rows.append({
                    "regime": regime,
                    "dataset": dataset,
                    "dataset_label": label,
                    "target": target,
                    "method": method,
                    "method_label": method_label,
                    "mp_tokens_m": mp_tokens / 1_000_000,
                    "mp_token_std_m": mp_token_std / 1_000_000,
                    "comp_tokens_m": comp_tokens / 1_000_000,
                    "saved_tokens_m": saved_tokens / 1_000_000,
                    "saved_minutes": saved_minutes,
                    "saved_fraction": saved_tokens / comp_tokens,
                    "mp_final_val": mp_final[0],
                    "comp_final_val": comp_final[0],
                    "endpoint_gap": endpoint_gap,
                })
    return rows


def make_combined_result_dotplot(out_path: Path) -> None:
    curves_by_regime = {"E1": load_e1_curves(), "E2": load_e2_curves()}
    methods = ["rlb_matrixpolicy_original", "silu_adamw", "silu_muon"]
    dataset_markers = {"dclm": "o", "fineweb_edu": "s", "fineweb": "^", "dolma_sample": "D", "c4_en": "P"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=False, constrained_layout=True)
    for ax, regime in zip(axes, ["E1", "E2"]):
        curves = curves_by_regime[regime]
        for dataset, dataset_label, e2_dir_name in DATASETS:
            targets = _target_candidates(regime, dataset, e2_dir_name)
            for method in methods:
                label, color, linestyle, linewidth, marker = _style(method)
                xs, ys = [], []
                for target in targets:
                    stats = _hit_token_stats(curves, dataset, method, target)
                    if stats is None:
                        continue
                    mean_tokens, _std_tokens = stats
                    xs.append(mean_tokens / 1_000_000)
                    ys.append(target)
                if not xs:
                    continue
                order = np.argsort(xs)
                xs_arr = np.asarray(xs)[order]
                ys_arr = np.asarray(ys)[order]
                ax.plot(xs_arr, ys_arr, color=color, linestyle=linestyle, linewidth=linewidth * 0.78, alpha=0.72)
                ax.scatter(xs_arr, ys_arr, color=color, marker=dataset_markers[dataset], s=18, edgecolor="#202020", linewidth=0.35, alpha=0.90, zorder=3)
                if method == "rlb_matrixpolicy_original" and len(xs_arr):
                    ax.annotate(DATASET_ABBR[dataset_label], (xs_arr[-1], ys_arr[-1]), xytext=(2, 1), textcoords="offset points", fontsize=6.4, color="#202020")
        ax.invert_yaxis()
        ax.set_title(f"{regime}: loss-token target plane", fontsize=8.8)
        ax.set_xlabel("tokens to target (M)", fontsize=8.0)
        _finish_axis(ax, labelsize=7.2)
    axes[0].set_ylabel("target validation loss", fontsize=8.0)
    method_handles = []
    method_labels = []
    for method in methods:
        label, color, linestyle, linewidth, marker = _style(method)
        method_handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=4.5))
        method_labels.append(label)
    dataset_handles = [plt.Line2D([0], [0], color="#777777", linestyle="", marker=dataset_markers[d], markersize=4.4, markerfacecolor="#dddddd", markeredgecolor="#222222") for d, _lab, _e2 in DATASETS]
    dataset_labels = [DATASET_ABBR[lab] for _d, lab, _e2 in DATASETS]
    fig.legend(method_handles + dataset_handles, method_labels + dataset_labels, loc="upper center", ncol=8, frameon=False, fontsize=6.9, handlelength=1.55, columnspacing=0.85)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _fmt_signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}"


def make_e1_e2_silu_summary_table(out_path: Path) -> None:
    rows = _summary_rows()
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["regime"]), str(row["dataset_label"]))
        by_key.setdefault(key, {"target": row["target"], "mp_tokens_m": row["mp_tokens_m"], "values": {}})
        by_key[key]["values"][row["method"]] = row
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{E1/E2 all-dataset comparison of MatrixPolicy against the two SiLU baselines. For each dataset, regime, and comparator, the table reports token savings at the common validation-loss target, estimated wall-clock time savings from measured throughput, and the endpoint validation-loss gap, computed as comparator minus MatrixPolicy; positive values favor MatrixPolicy.}",
        r"\label{tab:e1e2-silu-comparison}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.3pt}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"Regime & Dataset & Target & MP tok & \multicolumn{3}{c}{vs SiLU+AdamW} & \multicolumn{3}{c}{vs SiLU+Muon} \\",
        r"\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r" & & val. loss & (M) & $\Delta$tok & $\Delta$t & $\Delta$val & $\Delta$tok & $\Delta$t & $\Delta$val \\",
        r"\midrule",
    ]
    for regime in ["E1", "E2"]:
        for _dataset, label, _e2 in DATASETS:
            key = (regime, label)
            if key not in by_key:
                continue
            item = by_key[key]
            vals = item["values"]
            adam = vals.get("silu_adamw")
            muon = vals.get("silu_muon")
            if adam is None or muon is None:
                continue
            lines.append(
                f"{regime} & {label} & {float(item['target']):.2f} & {float(item['mp_tokens_m']):.1f} & "
                f"{_fmt_signed(float(adam['saved_tokens_m']))} & {_fmt_signed(float(adam['saved_minutes']))} & {_fmt_signed(float(adam['endpoint_gap']), 3)} & "
                f"{_fmt_signed(float(muon['saved_tokens_m']))} & {_fmt_signed(float(muon['saved_minutes']))} & {_fmt_signed(float(muon['endpoint_gap']), 3)} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{10}{@{}p{0.98\textwidth}@{}}{\footnotesize $\Delta$tok is in millions of tokens and $\Delta$t is in minutes. Token arrivals are evaluated every 50 steps, so target times are quantized by 1.64M tokens; time estimates use measured per-dataset/method throughput. Broader RLB optimizer controls are reported in Appendix~\ref{app:experiment-protocol}.}",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    out_path.write_text("\n".join(lines) + "\n")

def _fmt_token_fraction(saved_tokens_m: float, saved_fraction: float) -> str:
    return f"{saved_tokens_m:.1f} ({100.0 * saved_fraction:.1f}\\%)"


def make_e1_target_time_table(out_path: Path) -> None:
    curves = load_e1_curves()
    tps = _load_runtime_tps()
    rows = []
    required_methods = ["rlb_matrixpolicy_original"] + [method for method, _ in TABLE_COMPARATORS]
    for dataset, label, _dir_name in DATASETS:
        target = _select_hard_common_target(curves, "E1", dataset, required_methods)
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
        r"\caption{E1 target-arrival savings for activation/optimizer controls. Token columns report saved tokens in millions with the percentage of comparator tokens in parentheses; time columns report estimated saved minutes.}",
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
        lines.append(f"{label} & {target:.2f} & {mp_tokens_m:.1f}$\\pm${mp_std_m:.1f} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {cells[4]} & {cells[5]} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{9}{@{}p{0.98\textwidth}@{}}{\footnotesize Tokens-to-target are read at the native 50-step evaluation cadence from the completed E1 JSONL logs. Time estimates use cleaned per-dataset/method training-throughput summaries.}",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    outputs = [
        OUT_DIR / "e1_multimetric_all_datasets.pdf",
        OUT_DIR / "combined_result_dotplot.pdf",
        OUT_DIR / "e1_validation_all_datasets.pdf",
        OUT_DIR / "e1_target_frontiers.pdf",
        OUT_DIR / "e1_multimetric_examples.pdf",
        OUT_DIR / "e2_validation_dynamics.pdf",
        OUT_DIR / "e2_perplexity_dynamics.pdf",
        OUT_DIR / "e2_training_dynamics.pdf",
        TABLE_DIR / "e1_e2_silu_summary_table.tex",
        TABLE_DIR / "e1_target_time_table.tex",
    ]
    make_e1_multimetric_all_datasets(outputs[0])
    make_combined_result_dotplot(outputs[1])
    make_e1_validation_all_datasets(outputs[2])
    make_e1_target_frontiers(outputs[3])
    make_e1_multimetric_examples(outputs[4])
    make_e2_metric_dynamics(outputs[5], "val_loss", "validation loss")
    make_e2_metric_dynamics(outputs[6], "val_ppl", "validation perplexity")
    make_e2_metric_dynamics(outputs[7], "train_loss", "training loss")
    make_e1_e2_silu_summary_table(outputs[8])
    make_e1_target_time_table(outputs[9])
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
