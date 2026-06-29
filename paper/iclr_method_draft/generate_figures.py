#!/usr/bin/env python3
"""Generate the figures used by the ICLR manuscript.

The validation panel reuses the E1 manifest/JSONL parser from
``experiments/scripts/plot_iclr26_e1_curves.py`` so the plotted means/std bands
come from the same raw evaluation events as the repository E1 result figures.
"""

from __future__ import annotations

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


def _box(ax, xy, width, height, text, fc, ec="#333333", fontsize=9.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.15,
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
        color="#111111",
    )
    return patch


def _arrow(ax, start, end, color="#333333", lw=1.4, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def make_matrixpolicy_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "matrix": "#e8eef7",
        "group": "#edf7ee",
        "curve": "#fff3d9",
        "policy": "#f2ebf8",
        "signal": "#f7f7f7",
    }

    _box(ax, (0.04, 0.56), 0.13, 0.16, "$A_l$\ninput matrix", colors["matrix"])
    _box(ax, (0.22, 0.56), 0.16, 0.16, "$z_{l,g}$\nRMS groups", colors["group"])
    _box(ax, (0.43, 0.56), 0.15, 0.16, "$R_{l,g}$\nP5/Q4", colors["curve"])
    _box(ax, (0.63, 0.56), 0.13, 0.16, "$B_l$\noutput matrix", colors["matrix"])
    _box(ax, (0.82, 0.56), 0.12, 0.16, "$y_l$\nresidual", "#eeeeee")

    _arrow(ax, (0.17, 0.64), (0.22, 0.64))
    _arrow(ax, (0.38, 0.64), (0.43, 0.64))
    _arrow(ax, (0.58, 0.64), (0.63, 0.64))
    _arrow(ax, (0.76, 0.64), (0.82, 0.64))

    _box(
        ax,
        (0.17, 0.22),
        0.20,
        0.20,
        "role scales\n$p^{in}, p^{out}$",
        colors["signal"],
        fontsize=8.7,
    )
    _box(
        ax,
        (0.42, 0.22),
        0.18,
        0.20,
        "curve gains\n$\\hat d, \\hat o$",
        colors["signal"],
        fontsize=8.7,
    )
    _box(
        ax,
        (0.65, 0.22),
        0.17,
        0.20,
        "matrix ratio\n$\\gamma$",
        colors["signal"],
        fontsize=8.7,
    )
    _box(
        ax,
        (0.30, 0.02),
        0.42,
        0.13,
        "MatrixPolicy: centered group scaling + transient matrix-direction branch + bounded pair balancing",
        colors["policy"],
        fontsize=8.2,
    )

    _arrow(ax, (0.285, 0.56), (0.27, 0.42), color="#666666", rad=0.12)
    _arrow(ax, (0.50, 0.56), (0.51, 0.42), color="#666666", rad=-0.10)
    _arrow(ax, (0.70, 0.56), (0.735, 0.42), color="#666666", rad=-0.10)
    _arrow(ax, (0.27, 0.22), (0.41, 0.15), color="#6f4aa5")
    _arrow(ax, (0.51, 0.22), (0.51, 0.15), color="#6f4aa5")
    _arrow(ax, (0.735, 0.22), (0.61, 0.15), color="#6f4aa5")
    _arrow(ax, (0.36, 0.15), (0.12, 0.56), color="#6f4aa5", rad=0.18)
    _arrow(ax, (0.66, 0.15), (0.70, 0.56), color="#6f4aa5", rad=-0.18)

    ax.text(0.04, 0.89, "RLB exposes optimizer-visible roles", fontsize=11, weight="bold")
    ax.text(0.04, 0.84, "Only the RLB input/output matrices take MatrixPolicy updates; other parameters follow AdamW.", fontsize=8.5)
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


def make_e1_validation_examples(out_path: Path) -> None:
    curves = load_e1_curves()
    datasets = [("dclm", "DCLM"), ("fineweb_edu", "FineWeb-Edu")]
    methods = [
        ("rlb_matrixpolicy_original", "MatrixPolicy", "#111111", "-", 2.1, 0.15),
        ("rlb_lion", "RLB+Lion", "#2ca02c", "-", 1.7, 0.11),
        ("rlb_adamw", "RLB+AdamW", "#1f77b4", "-", 1.45, 0.09),
        ("silu_adamw", "SiLU+AdamW", "#1f77b4", "--", 1.45, 0.08),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.05), sharex=True)
    for ax, (dataset, title) in zip(axes, datasets):
        for method, label, color, linestyle, linewidth, alpha in methods:
            steps, means, stds = e1_curves.aggregate(curves, dataset, method, "val_loss")
            if steps.size == 0:
                continue
            tokens = steps * TOKENS_PER_STEP / 1_000_000
            ax.plot(tokens, means, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            ax.fill_between(tokens, means - stds, means + stds, color=color, alpha=alpha, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tokens (M)")
        ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("validation loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90), pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    make_matrixpolicy_overview(OUT_DIR / "matrixpolicy_overview.pdf")
    make_e1_validation_examples(OUT_DIR / "e1_validation_examples.pdf")
    print(OUT_DIR / "matrixpolicy_overview.pdf")
    print(OUT_DIR / "e1_validation_examples.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
