#!/usr/bin/env python3
"""Generate manuscript figures and tables from repository result artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
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
FIG_SRC_DIR = DRAFT_DIR / "figures_src"
TABLE_DIR = DRAFT_DIR / "tables"
TOKENS_PER_STEP = 32768

sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
import plot_iclr26_e1_curves as e1_curves  # noqa: E402
import plot_iclr26_e2_curves as e2_curves  # noqa: E402


def _latexmk_command() -> str:
    local = ROOT / ".TinyTeX" / "bin" / "x86_64-linux" / "latexmk"
    if local.exists():
        return str(local)
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required to build the native TeX illustration figures")
    return latexmk


def _latex_env() -> dict[str, str]:
    env = os.environ.copy()
    local_bin = ROOT / ".TinyTeX" / "bin" / "x86_64-linux"
    if local_bin.exists():
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
    return env


def compile_tikz_figure(source_name: str, out_path: Path) -> None:
    source = FIG_SRC_DIR / source_name
    if not source.exists():
        raise FileNotFoundError(source)
    subprocess.run(
        [
            _latexmk_command(),
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            source.name,
        ],
        cwd=FIG_SRC_DIR,
        env=_latex_env(),
        check=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.with_suffix(".pdf"), out_path)

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
BUDGET_LABELS = {
    "E1": "100M tokens",
    "E2": "300M tokens",
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
TARGET_OPTIMIZER_COLORS = {
    "AdamW": OKABE_ITO["blue"],
    "Lion": "#B07D00",
    "Muon": OKABE_ITO["green"],
    "SOAP": "#B45F93",
    "ScheduleFree": "#C66525",
    "CAME": "#3C9DCA",
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
MAIN_COMPARATORS = [
    ("silu_adamw", "SiLU+AdamW"),
    ("silu_muon", "SiLU+Muon"),
    ("rlb_adamw", "RLB+AdamW"),
    ("rlb_muon", "RLB+Muon"),
]
TABLE_COMPARATORS = [("silu_adamw", "SiLU+AdamW"), ("rlb_adamw", "RLB+AdamW"), ("rlb_muon", "RLB+Muon")]
BUDGET_LABEL = BUDGET_LABELS
BUDGET_SHORT = BUDGET_LABELS
BROAD_CONTROL_METHODS = [
    ("silu_adamw", "SiLU + AdamW", "SiLU", "AdamW"),
    ("silu_lion", "SiLU + Lion", "SiLU", "Lion"),
    ("silu_muon", "SiLU + Muon", "SiLU", "Muon"),
    ("silu_soap", "SiLU + SOAP", "SiLU", "SOAP"),
    ("silu_schedulefree", "SiLU + ScheduleFree", "SiLU", "ScheduleFree"),
    ("silu_came", "SiLU + CAME", "SiLU", "CAME"),
    ("rlb_adamw", "RLB + AdamW", "RLB", "AdamW"),
    ("rlb_lion", "RLB + Lion", "RLB", "Lion"),
    ("rlb_muon", "RLB + Muon", "RLB", "Muon"),
    ("rlb_soap", "RLB + SOAP", "RLB", "SOAP"),
    ("rlb_schedulefree", "RLB + ScheduleFree", "RLB", "ScheduleFree"),
    ("rlb_came", "RLB + CAME", "RLB", "CAME"),
]
BROAD_METHODS = [("rlb_matrixpolicy_original", "MatrixPolicy", "MatrixPolicy", "MatrixPolicy")] + BROAD_CONTROL_METHODS
BROAD_METHOD_INFO = {method: (label, family, optimizer) for method, label, family, optimizer in BROAD_METHODS}


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

    fig, ax = plt.subplots(figsize=(7.2, 4.05), constrained_layout=False)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#202124"
    muted = "#5f6368"
    forward = "#202124"
    observe = "#9aa3ad"
    policy = "#6f4aa4"
    policy_light = "#b4a7c9"
    blue = "#dcebf7"
    blue_edge = "#325d84"
    green = "#e7f5ed"
    green_edge = "#2f7d54"
    amber = "#fff1d1"
    amber_edge = "#9a6a00"
    lavender = "#eee7f7"
    gray = "#f4f4f4"

    def panel(x0, y0, w, h, fc, ec, label):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            facecolor=fc, edgecolor=ec, linewidth=0.70, alpha=0.72, zorder=0,
        ))
        if label:
            ax.text(x0 + 0.015, y0 + h - 0.030, label, ha="left", va="top", fontsize=7.2, weight="bold", color=ec)

    def labeled_box(x0, y0, w, h, title, body, fc, ec, title_size=7.3, body_size=6.45, lw=0.85):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        ))
        if body:
            ax.text(x0 + w / 2, y0 + h * 0.64, title, ha="center", va="center", fontsize=title_size, weight="bold", color=ink, zorder=3)
            ax.text(x0 + w / 2, y0 + h * 0.34, body, ha="center", va="center", fontsize=body_size, color=ink, linespacing=1.05, zorder=3)
        else:
            ax.text(x0 + w / 2, y0 + h / 2, title, ha="center", va="center", fontsize=title_size, weight="bold", color=ink, zorder=3)

    def matrix_block(x0, y0, w, h, label, orientation="rows"):
        ax.add_patch(FancyBboxPatch(
            (x0 - 0.010, y0 - 0.018), w + 0.020, h + 0.052,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor="#f7fbff", edgecolor=policy, linewidth=0.85, zorder=1,
        ))
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=blue, edgecolor=blue_edge, linewidth=1.0, zorder=2))
        if orientation == "rows":
            band_h = h / 4
            for i, color in enumerate(["#c5dcf0", "#e8f1f9", "#c5dcf0", "#e8f1f9"]):
                ax.add_patch(Rectangle((x0, y0 + i * band_h), w, band_h, facecolor=color, edgecolor="white", linewidth=0.45, zorder=3))
        else:
            band_w = w / 4
            for i, color in enumerate(["#c5dcf0", "#e8f1f9", "#c5dcf0", "#e8f1f9"]):
                ax.add_patch(Rectangle((x0 + i * band_w, y0), band_w, h, facecolor=color, edgecolor="white", linewidth=0.45, zorder=3))
        ax.text(x0 + w / 2, y0 + h + 0.024, label, ha="center", va="bottom", fontsize=8.4, weight="bold", color=ink)

    def chip(x0, y0, title, body, width=0.118):
        labeled_box(x0, y0, width, 0.072, title, body, "#ffffff", "#a0a0a0", title_size=6.3, body_size=5.9, lw=0.65)

    def poly_arrow(points, color, lw=0.95, scale=7.0, linestyle="-"):
        for p0, p1 in zip(points[:-2], points[1:-1]):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, linestyle=linestyle, zorder=1.5)
        _arrow(ax, points[-2], points[-1], color=color, lw=lw, scale=scale, linestyle=linestyle)

    ax.text(0.022, 0.954, "Global-rational RLB interface", fontsize=9.8, weight="bold", va="top", color=ink)
    ax.text(0.022, 0.914, "Forward computation, detached statistics, and policy targets are separated visually.", fontsize=7.2, va="top", color=muted)

    panel(0.025, 0.565, 0.950, 0.300, "#f8fbfd", "#607d8b", "forward map")
    y = 0.710
    labeled_box(0.045, y - 0.043, 0.052, 0.086, "$x_l$", "", gray, "#9aa0a6", title_size=9.4, body_size=0)
    matrix_block(0.133, y - 0.090, 0.068, 0.180, "$A_l$", orientation="rows")
    labeled_box(0.248, y - 0.078, 0.142, 0.156, "group RMS", "$z_{l,g}=A_{l,g}x_l$\n$u=z/r$", green, green_edge, title_size=7.3, body_size=6.45)

    ax.add_patch(FancyBboxPatch(
        (0.410, y - 0.110), 0.250, 0.220,
        boxstyle="round,pad=0.014,rounding_size=0.020",
        linewidth=0.95, facecolor=amber, edgecolor=amber_edge, zorder=2,
    ))
    ax.text(0.535, y + 0.087, "shared global\nrational response", ha="center", va="center", fontsize=6.95, weight="bold", color=ink, linespacing=0.90, zorder=3)
    ax.text(0.535, y + 0.038, "$R_{l,g}(u)=P_5(u)/Q_4(u)$", ha="center", fontsize=6.75, color=ink, zorder=3)
    ax.text(0.535, y + 0.011, "$Q_4(u)\\geq 1$", ha="center", fontsize=6.55, color=muted, zorder=3)
    xs = np.linspace(-1, 1, 120)
    ys = y - 0.053 + 0.058 * (xs / (1.0 + 0.65 * np.abs(xs)) + 0.08 * xs**2)
    ax.plot(0.473 + 0.124 * (xs + 1) / 2, ys, color="#7a4b00", linewidth=1.30, zorder=4)
    ax.plot([0.473, 0.597], [y - 0.053, y - 0.053], color="#b89042", linewidth=0.45, zorder=3)

    labeled_box(0.680, y - 0.060, 0.090, 0.120, "restore", "$h=rR(u)$", green, green_edge, title_size=7.0, body_size=6.35)
    matrix_block(0.812, y - 0.090, 0.068, 0.180, "$B_l$", orientation="cols")
    labeled_box(0.922, y - 0.043, 0.052, 0.086, "$y_l$", "", gray, "#9aa0a6", title_size=9.4, body_size=0)

    for start, end in [
        ((0.097, y), (0.133, y)),
        ((0.201, y), (0.248, y)),
        ((0.390, y), (0.410, y)),
        ((0.660, y), (0.680, y)),
        ((0.770, y), (0.812, y)),
        ((0.880, y), (0.922, y)),
    ]:
        _arrow(ax, start, end, color=forward, lw=1.15, scale=9)

    panel(0.045, 0.320, 0.910, 0.190, "#fafafa", "#777777", "")
    ax.text(0.062, 0.496, "detached statistic bank", fontsize=7.0, weight="bold", color="#777777", ha="left", va="top", zorder=8)
    chips = [
        (0.108, "pressure", "$\\pi_{l,g}$", 0.167),
        (0.290, "live gains", "$\\hat d,\\hat o$", 0.319),
        (0.472, "curve activity", "$\\alpha_{l,g}$", 0.535),
        (0.784, "pair target", "$\\gamma,\\tau$", 0.846),
    ]
    for x0, title, body, source_x in chips:
        chip(x0, 0.382, title, body, width=0.124)
        if title == "pressure":
            elbow = [(source_x, 0.620), (source_x, 0.555), (0.260, 0.555), (0.260, 0.452)]
            for p0, p1 in zip(elbow[:-1], elbow[1:]):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=observe, linewidth=0.42, linestyle=(0, (2.0, 2.0)), zorder=1)
            _arrow(ax, (0.260, 0.452), (0.225, 0.452), color=observe, lw=0.42, scale=4.3, linestyle=(0, (2.0, 2.0)))
        else:
            _arrow(ax, (source_x, 0.620), (x0 + 0.062, 0.452), color=observe, lw=0.45, scale=4.4, linestyle=(0, (2.0, 2.0)))

    bus_y = 0.352
    for x0, _title, _body, _source_x in chips:
        cx = x0 + 0.062
        ax.plot([cx, cx], [0.382, bus_y], color="#b8b8b8", linewidth=0.42, linestyle="--", zorder=1)
    ax.plot([0.170, 0.846], [bus_y, bus_y], color="#b8b8b8", linewidth=0.42, linestyle="--", zorder=1)
    labeled_box(0.378, 0.240, 0.230, 0.078, "detached summaries", "detached", "#ffffff", "#8a8a8a", title_size=6.85, body_size=6.25, lw=0.70)
    _arrow(ax, (0.508, bus_y), (0.508, 0.318), color="#a8a8a8", lw=0.54, scale=4.8, linestyle="--")

    labeled_box(0.370, 0.115, 0.250, 0.100, "MatrixPolicy", "outputs $\\Delta A_l,\\Delta B_l$ only", lavender, policy, title_size=8.0, body_size=6.6, lw=0.95)
    _arrow(ax, (0.493, 0.240), (0.493, 0.215), color=policy, lw=0.82, scale=6.4)
    poly_arrow([(0.390, 0.115), (0.390, 0.070), (0.018, 0.070), (0.018, 0.600), (0.133, 0.600)], color=policy, lw=0.72, scale=7.2)
    poly_arrow([(0.600, 0.115), (0.600, 0.070), (0.982, 0.070), (0.982, 0.600), (0.880, 0.600)], color=policy, lw=0.72, scale=7.2)
    ax.text(0.095, 0.576, "$\\Delta A_l$", fontsize=6.45, color=policy, ha="center", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.94))
    ax.text(0.918, 0.576, "$\\Delta B_l$", fontsize=6.45, color=policy, ha="center", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.94))

    ax.plot([0.620, 0.640], [0.165, 0.165], color="#9aa0a6", linewidth=0.55)
    ax.text(0.650, 0.186, "coefficients are observed signals and follow AdamW", fontsize=6.15, color=muted, ha="left")
    ax.text(0.650, 0.158, "RLB response: grouped P5/Q4 only", fontsize=6.15, color=muted, ha="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, facecolor="white", transparent=False)
    plt.close(fig)


def make_matrixpolicy_signal_flow(out_path: Path) -> None:
    """MatrixPolicy optimizer-action schematic with explicit bypass lane."""

    fig, ax = plt.subplots(figsize=(7.2, 4.05), constrained_layout=False)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#202020"
    muted = "#666666"
    purple = "#6f4aa4"
    lavender = "#eee7f7"
    pale_lav = "#f8f4fc"
    amber = "#fff4dc"
    amber_edge = "#9a6a00"
    gray = "#eeeeee"
    blue = "#dcebf7"

    def lane(x0, y0, w, h, label, fc, ec):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            facecolor=fc, edgecolor=ec, linewidth=0.70, alpha=0.72, zorder=0,
        ))
        if label:
            ax.text(x0 + 0.018, y0 + h - 0.032, label, fontsize=7.4, weight="bold", color=ec, ha="left", va="top")

    def stage(x0, y0, w, h, num, title, body, fc=lavender, ec=purple, title_size=6.8, body_size=6.0):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=fc, edgecolor=ec, linewidth=0.88, zorder=2,
        ))
        circ = plt.Circle((x0 + 0.024, y0 + h - 0.029), 0.0165, facecolor=ec, edgecolor="white", linewidth=0.45, zorder=4)
        ax.add_patch(circ)
        ax.text(x0 + 0.024, y0 + h - 0.029, str(num), ha="center", va="center", fontsize=5.8, color="white", weight="bold", zorder=5)
        ax.text(x0 + w / 2, y0 + h * 0.65, title, ha="center", va="center", fontsize=title_size, weight="bold", color=ink, zorder=3)
        ax.text(x0 + w / 2, y0 + h * 0.34, body, ha="center", va="center", fontsize=body_size, color=ink, linespacing=1.18, zorder=3)

    def bypass_box(x0, y0, w, h, title, body):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=gray, edgecolor="#8a8a8a", linewidth=0.78, zorder=2,
        ))
        ax.text(x0 + w / 2, y0 + h * 0.66, title, ha="center", va="center", fontsize=6.9, weight="bold", color=ink, zorder=3)
        ax.text(x0 + w / 2, y0 + h * 0.34, body, ha="center", va="center", fontsize=6.05, color=ink, linespacing=1.15, zorder=3)

    ax.text(0.022, 0.952, "MatrixPolicy optimizer action", fontsize=9.8, weight="bold", color=ink, va="top")
    ax.text(0.022, 0.912, "The policy lane is reserved for $A_l,B_l$; every other tensor takes the compact AdamW bypass.", fontsize=7.2, color=muted, va="top")

    lane(0.030, 0.505, 0.940, 0.345, "RLB matrix lane: $A_l$ and $B_l$", "#fbf8ff", purple)
    y = 0.605
    h = 0.176
    stage(0.055, y, 0.120, h, 1, "signals", "$\\pi,\\alpha$\n$\\hat d,\\hat o$\n$\\gamma,\\tau$", fc="#edf4fb", ec="#366b9a", title_size=6.55, body_size=5.95)
    stage(0.205, y, 0.120, h, 2, "group gates", "$c_{l,g,\\sigma}$\ncenter + clip", title_size=6.55, body_size=6.05)
    stage(0.355, y, 0.120, h, 3, "scale grads", "$\\widetilde\\nabla A_{l,g}$\n$\\widetilde\\nabla B_{l,g}$", title_size=6.55, body_size=6.05)

    x4, w4, h4 = 0.512, 0.194, 0.212
    ax.add_patch(FancyBboxPatch(
        (x4, y - 0.010), w4, h4,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        facecolor=pale_lav, edgecolor=purple, linewidth=1.05, zorder=2,
    ))
    circ = plt.Circle((x4 + 0.026, y - 0.010 + h4 - 0.030), 0.0165, facecolor=purple, edgecolor="white", linewidth=0.45, zorder=4)
    ax.add_patch(circ)
    ax.text(x4 + 0.026, y - 0.010 + h4 - 0.030, "4", ha="center", va="center", fontsize=5.8, color="white", weight="bold", zorder=5)
    ax.text(x4 + w4 / 2, y + 0.139, "staged matrix step", ha="center", va="center", fontsize=7.0, weight="bold", color=ink)
    ax.add_patch(FancyBboxPatch((x4 + 0.030, y + 0.060), 0.064, 0.052, boxstyle="round,pad=0.006,rounding_size=0.010", facecolor="#ffffff", edgecolor="#8b6bb0", linewidth=0.70, zorder=3))
    ax.add_patch(FancyBboxPatch((x4 + 0.138, y + 0.060), 0.056, 0.052, boxstyle="round,pad=0.006,rounding_size=0.010", facecolor="#ffffff", edgecolor="#8b6bb0", linewidth=0.70, zorder=3))
    ax.text(x4 + 0.062, y + 0.086, "AdamW\non A,B", fontsize=5.55, ha="center", va="center", color=ink, zorder=4, linespacing=1.05)
    ax.text(x4 + 0.166, y + 0.086, "Muon\nmatrix", fontsize=5.55, ha="center", va="center", color=ink, zorder=4, linespacing=1.05)
    internal_arrow = _arrow(ax, (x4 + 0.097, y + 0.086), (x4 + 0.138, y + 0.086), color=purple, lw=0.95, scale=10.5)
    internal_arrow.set_zorder(7)
    ax.text(x4 + w4 / 2, y + 0.026, "separate optimizer states", ha="center", va="center", fontsize=5.8, color=muted)

    stage(0.728, y, 0.114, h, 5, "pair balance", "every 5 steps\n$A_{l,g}\\leftarrow e^\\ell A_{l,g}$\n$B_{l,g}\\leftarrow e^{-\\ell}B_{l,g}$", fc=amber, ec=amber_edge, title_size=6.10, body_size=5.50)
    stage(0.884, y + 0.010, 0.080, h - 0.020, 6, "updated", "$A_l^{t+1}$\n$B_l^{t+1}$", fc=blue, ec="#325d84", title_size=5.80, body_size=6.20)

    for start, end in [
        ((0.181, y + h / 2), (0.199, y + h / 2)),
        ((0.331, y + h / 2), (0.349, y + h / 2)),
        ((0.481, y + h / 2), (0.506, y + h / 2)),
        ((0.712, y + h / 2), (0.722, y + h / 2)),
        ((0.848, y + h / 2), (0.878, y + h / 2)),
    ]:
        _arrow(ax, start, end, color=purple, lw=0.88, scale=10.0)

    ax.text(0.505, 0.535, "Rational coefficients are observed signals, not MatrixPolicy update targets.", fontsize=5.95, color="#777777", ha="center")

    lane(0.055, 0.170, 0.890, 0.230, "", "#f7f7f7", "#777777")
    ax.text(0.074, 0.366, "AdamW-only bypass", fontsize=7.2, weight="bold", color="#777777", ha="left", va="top")
    bypass_box(0.090, 0.232, 0.285, 0.105, "non-matrix tensors", "incl. rational coefficients,\nattention, embeddings, norms")
    bypass_box(0.485, 0.232, 0.165, 0.105, "AdamW only", "no gates or Muon\nno pair rescale")
    bypass_box(0.735, 0.240, 0.170, 0.090, "updated", "$\\theta_{\\neg AB}^{t+1}$")
    _arrow(ax, (0.391, 0.285), (0.469, 0.285), color="#777777", lw=1.10, scale=12.0)
    _arrow(ax, (0.666, 0.285), (0.719, 0.285), color="#777777", lw=1.10, scale=12.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, facecolor="white", transparent=False)
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


def _saturation_zoom_bounds(regime: str, steps: np.ndarray) -> tuple[float, float]:
    finite_steps = steps[np.isfinite(steps)]
    if finite_steps.size == 0:
        return 0.0, 1.0
    x_min = float(np.min(finite_steps))
    x_max = float(np.max(finite_steps))
    span = max(x_max - x_min, 1.0)
    if regime == "E2":
        return x_min + 0.30 * span, x_min + 0.62 * span
    return x_min + 0.72 * span, x_max


def _add_saturation_inset(
    ax,
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, object, float, str]],
    regime: str,
) -> None:
    if not traces:
        return
    all_steps = np.concatenate([steps for steps, _means, _stds, _color, _linestyle, _linewidth, _marker in traces])
    x0, x1 = _saturation_zoom_bounds(regime, all_steps)
    inset_traces = []
    y_values: list[np.ndarray] = []
    for steps, means, stds, color, linestyle, linewidth, marker in traces:
        mask = np.isfinite(steps) & np.isfinite(means) & (steps >= x0) & (steps <= x1)
        if np.count_nonzero(mask) < 2:
            finite = np.flatnonzero(np.isfinite(steps) & np.isfinite(means))
            if finite.size < 2:
                continue
            take = finite[-min(4, finite.size):]
            mask = np.zeros_like(steps, dtype=bool)
            mask[take] = True
        xs = steps[mask]
        ys = means[mask]
        es = stds[mask] if stds.size == means.size else np.zeros_like(ys)
        if xs.size < 2:
            continue
        inset_traces.append((xs, ys, es, color, linestyle, linewidth, marker))
        y_values.append(ys)
    if not inset_traces or not y_values:
        return

    y_concat = np.concatenate(y_values)
    y_min = float(np.min(y_concat))
    y_max = float(np.max(y_concat))
    y_span = max(y_max - y_min, abs(y_max) * 0.012, 1e-6)
    y_pad = 0.18 * y_span

    inset = ax.inset_axes([0.555, 0.565, 0.405, 0.380])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.94))
    for xs, ys, es, color, linestyle, linewidth, marker in inset_traces:
        inset.plot(
            xs,
            ys,
            color=color,
            linestyle=linestyle,
            linewidth=max(0.82, 0.78 * linewidth),
            marker=marker,
            markersize=1.55,
            markevery=max(1, xs.size // 2),
        )
        inset.fill_between(xs, ys - es, ys + es, color=color, alpha=0.025, linewidth=0)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y_min - y_pad, y_max + y_pad)
    inset.grid(True, color="#dddddd", linewidth=0.28, alpha=0.58)
    inset.tick_params(axis="both", labelsize=3.7, pad=0.5, length=1.0, width=0.35)
    inset.locator_params(axis="x", nbins=2)
    inset.locator_params(axis="y", nbins=2)
    for spine in inset.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor("#444444")
    rect = Rectangle(
        (x0, y_min - y_pad),
        x1 - x0,
        (y_max - y_min) + 2.0 * y_pad,
        fill=False,
        edgecolor="#666666",
        linewidth=0.50,
        linestyle=(0, (2.2, 1.8)),
        alpha=0.75,
        zorder=5,
    )
    ax.add_patch(rect)


def _make_multimetric_all_datasets(curves, aggregate_fn, out_path: Path, title: str) -> None:
    metrics = [("val_loss", "validation loss"), ("val_ppl", "validation PPL"), ("train_loss", "training loss")]
    regime = "E2" if "300M" in title else "E1"
    fig, axes = plt.subplots(len(DATASETS), len(metrics), figsize=(7.2, 8.35), sharex=True)
    fig.subplots_adjust(left=0.082, right=0.992, bottom=0.072, top=0.920, hspace=0.335, wspace=0.215)
    for row, (dataset, dataset_label, _dir_name) in enumerate(DATASETS):
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            traces: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, object, float, str]] = []
            for method in MAIN_SILU_METHODS:
                label, color, linestyle, linewidth, marker = _style(method)
                steps, means, stds = aggregate_fn(curves, dataset, method, metric)
                if steps.size == 0:
                    continue
                traces.append((steps, means, stds, color, linestyle, linewidth, marker))
                mark_every = max(1, len(steps) // 5)
                ax.plot(
                    steps,
                    means,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker=marker,
                    markevery=mark_every,
                    markersize=2.2,
                    label=label,
                )
                ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.035, linewidth=0)
            _add_saturation_inset(ax, traces, regime)
            if row == 0:
                ax.set_title(metric_label, fontsize=8.5, pad=4)
            if col == 0:
                ax.set_ylabel(dataset_label, fontsize=8.0)
            if row == len(DATASETS) - 1:
                ax.set_xlabel("optimizer step", fontsize=7.6)
            ax.margins(x=0.035, y=0.10)
            _finish_axis(ax, labelsize=6.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=3,
        frameon=False,
        fontsize=7.6,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.suptitle(title, fontsize=9.0, y=0.999)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.045, facecolor="white", transparent=False)
    plt.close(fig)


def make_e1_multimetric_all_datasets(out_path: Path) -> None:
    _make_multimetric_all_datasets(
        load_e1_curves(),
        e1_curves.aggregate,
        out_path,
        "100M-token budget: five datasets by three training metrics",
    )


def make_e2_multimetric_all_datasets(out_path: Path) -> None:
    _make_multimetric_all_datasets(
        load_e2_curves(),
        e2_curves.aggregate,
        out_path,
        "300M-token budget: five datasets by three training metrics",
    )

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


def _first_hit_steps(curves, dataset: str, method: str, target_loss: float) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    seed_data = curves.get(dataset, {}).get(method, {})
    for seed, events in seed_data.items():
        hit = None
        for step, values in sorted(events["eval"].items()):
            loss = values.get("val_loss")
            if isinstance(loss, (int, float)) and math.isfinite(loss) and loss <= target_loss:
                hit = int(step)
                break
        if hit is not None:
            hits.append((int(seed), hit))
    return hits


def _first_hit_tokens(curves, dataset: str, method: str, target_loss: float) -> list[float]:
    hits: list[float] = []
    for _seed, step in _first_hit_steps(curves, dataset, method, target_loss):
        hits.append(float(step * TOKENS_PER_STEP))
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


def _completed_broad_methods(curves) -> list[tuple[str, str, str, str]]:
    methods: list[tuple[str, str, str, str]] = []
    for method, label, family, optimizer in BROAD_METHODS:
        if all(_final_eval_mean(curves, dataset, method, "val_loss") is not None for dataset, _label, _dir_name in DATASETS):
            methods.append((method, label, family, optimizer))
    return methods


def _target_line_style(method: str) -> tuple[str, object, float, float]:
    if method == "rlb_matrixpolicy_original":
        return OKABE_ITO["black"], "-", 2.55, 1.0
    _label, family, optimizer = BROAD_METHOD_INFO[method]
    linestyle = "-" if family == "RLB" else (0, (2.4, 1.8))
    if optimizer in {"AdamW", "Muon"}:
        linewidth = 1.45 if family == "RLB" else 1.28
        alpha = 0.88 if family == "RLB" else 0.76
    else:
        linewidth = 0.82 if family == "RLB" else 0.74
        alpha = 0.38 if family == "RLB" else 0.29
    return TARGET_OPTIMIZER_COLORS.get(optimizer, "#777777"), linestyle, linewidth, alpha


def _fmt_mean_std(stats: tuple[float, float] | None, digits: int = 4) -> str:
    if stats is None:
        return "--"
    mean, std = stats
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def make_e1_representative_silu_dynamics(out_path: Path) -> None:
    e1 = load_e1_curves()
    panels = [("dclm", "DCLM"), ("fineweb_edu", "FineWeb-Edu")]
    metrics = [("val_loss", "validation loss"), ("val_ppl", "validation perplexity"), ("train_loss", "training loss")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.38), sharex=True)
    fig.subplots_adjust(left=0.074, right=0.994, bottom=0.145, top=0.780, hspace=0.285, wspace=0.250)
    for row, (dataset, dataset_label) in enumerate(panels):
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            traces: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, object, float, str]] = []
            for method in MAIN_SILU_METHODS:
                label, color, linestyle, linewidth, marker = _style(method)
                steps, means, stds = e1_curves.aggregate(e1, dataset, method, metric)
                if steps.size == 0:
                    continue
                traces.append((steps, means, stds, color, linestyle, linewidth, marker))
                mark_every = max(1, len(steps) // 5)
                ax.plot(steps, means, color=color, linestyle=linestyle, linewidth=linewidth, label=label, marker=marker, markevery=mark_every, markersize=2.8)
                ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.045, linewidth=0)
            _add_saturation_inset(ax, traces, "E1")
            if row == 0:
                ax.set_title(metric_label, fontsize=8.8, pad=4.5)
            if col == 0:
                ax.text(
                    -0.205,
                    0.5,
                    dataset_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#202020",
                )
            ax.margins(x=0.035, y=0.10)
            _finish_axis(ax, labelsize=6.9)
    fig.text(0.540, 0.045, "optimizer step", ha="center", va="center", fontsize=8.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=3, frameon=False, fontsize=7.8, handlelength=1.95, columnspacing=1.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, facecolor="white", transparent=False)
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


def _runtime_regime(scope: str) -> str | None:
    if scope.startswith("E1_"):
        return "E1"
    if scope.startswith("E2_"):
        return "E2"
    return None


def _load_runtime_seconds_per_step() -> dict[tuple[str, str, str, int], float]:
    rows = _read_rows(ROOT / "experiments" / "results" / "iclr26_runtime_summary_2026_06_11" / "runtime_per_row.csv")
    seconds_per_step: dict[tuple[str, str, str, int], float] = {}
    for row in rows:
        regime = _runtime_regime(row["scope"])
        if regime is None:
            continue
        completed_steps = float(row.get("completed_steps") or row.get("steps") or 0.0)
        total_seconds = float(row.get("total_seconds") or 0.0)
        if completed_steps <= 0.0 or total_seconds <= 0.0:
            continue
        key = (regime, row["dataset"], row["method"], int(row["seed"]))
        if key in seconds_per_step:
            raise RuntimeError(f"duplicate runtime row for {key}")
        seconds_per_step[key] = total_seconds / completed_steps
    return seconds_per_step


def _hit_time_stats(
    curves,
    regime: str,
    dataset: str,
    method: str,
    target_loss: float,
    seconds_per_step: dict[tuple[str, str, str, int], float],
) -> tuple[float, float] | None:
    hit_steps = _first_hit_steps(curves, dataset, method, target_loss)
    if len(hit_steps) < 3:
        return None
    hit_minutes = []
    for seed, step in hit_steps:
        key = (regime, dataset, method, seed)
        if key not in seconds_per_step:
            raise RuntimeError(f"missing runtime row for {key}")
        hit_minutes.append(step * seconds_per_step[key] / 60.0)
    return float(np.mean(hit_minutes)), float(np.std(hit_minutes, ddof=1)) if len(hit_minutes) > 1 else 0.0


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


def _target_arrival_rows() -> list[dict[str, object]]:
    curves_by_regime = {"E1": load_e1_curves(), "E2": load_e2_curves()}
    seconds_per_step = _load_runtime_seconds_per_step()
    rows: list[dict[str, object]] = []
    required_methods = ["rlb_matrixpolicy_original"] + [method for method, _label in MAIN_COMPARATORS]
    for regime in ["E1", "E2"]:
        curves = curves_by_regime[regime]
        for dataset, label, e2_dir_name in DATASETS:
            target = _select_hard_common_target(curves, regime, dataset, required_methods, e2_dir_name)
            if target is None:
                continue
            mp_stats = _hit_token_stats(curves, dataset, "rlb_matrixpolicy_original", target)
            mp_time_stats = _hit_time_stats(curves, regime, dataset, "rlb_matrixpolicy_original", target, seconds_per_step)
            mp_final = _final_eval_mean(curves, dataset, "rlb_matrixpolicy_original", "val_loss")
            if mp_stats is None or mp_time_stats is None or mp_final is None:
                continue
            mp_tokens, mp_token_std = mp_stats
            mp_time_min, mp_time_std_min = mp_time_stats
            comparators = []
            complete = True
            for method, method_label in MAIN_COMPARATORS:
                comp_stats = _hit_token_stats(curves, dataset, method, target)
                comp_time_stats = _hit_time_stats(curves, regime, dataset, method, target, seconds_per_step)
                comp_final = _final_eval_mean(curves, dataset, method, "val_loss")
                if comp_stats is None or comp_time_stats is None or comp_final is None:
                    complete = False
                    break
                comp_tokens, comp_token_std = comp_stats
                comp_time_min, comp_time_std_min = comp_time_stats
                saved_tokens = comp_tokens - mp_tokens
                comparators.append({
                    "method": method,
                    "method_label": method_label,
                    "tokens_m": comp_tokens / 1_000_000,
                    "token_std_m": comp_token_std / 1_000_000,
                    "time_min": comp_time_min,
                    "time_std_min": comp_time_std_min,
                    "saved_tokens_m": saved_tokens / 1_000_000,
                    "saved_fraction": saved_tokens / comp_tokens,
                    "saved_minutes": comp_time_min - mp_time_min,
                    "endpoint_gap": comp_final[0] - mp_final[0],
                })
            if not complete:
                continue
            rows.append({
                "regime": regime,
                "dataset": dataset,
                "dataset_label": label,
                "target": target,
                "mp_tokens_m": mp_tokens / 1_000_000,
                "mp_token_std_m": mp_token_std / 1_000_000,
                "mp_time_min": mp_time_min,
                "mp_time_std_min": mp_time_std_min,
                "mp_final_val": mp_final[0],
                "comparators": comparators,
            })
    return rows

def make_target_arrival_evidence_matrix(out_path: Path) -> None:
    curves_by_regime = {"E1": load_e1_curves(), "E2": load_e2_curves()}
    method_sets = {regime: _completed_broad_methods(curves) for regime, curves in curves_by_regime.items()}
    panel_data: dict[tuple[str, str], list[tuple[str, str, str, list[float], list[float]]]] = {}
    panel_ranges: dict[tuple[str, str], tuple[float, float, float, float] | None] = {}
    for regime in ["E1", "E2"]:
        curves = curves_by_regime[regime]
        methods = method_sets[regime]
        for dataset, dataset_label, e2_dir_name in DATASETS:
            targets = sorted(_target_candidates(regime, dataset, e2_dir_name), reverse=True)
            if not targets:
                panel_data[(regime, dataset)] = []
                panel_ranges[(regime, dataset)] = None
                continue
            traces: list[tuple[str, str, str, list[float], list[float]]] = []
            panel_x_values: list[float] = []
            panel_y_values: list[float] = []
            for method, _label, family, optimizer in methods:
                xs, ys = [], []
                for target in targets:
                    stats = _hit_token_stats(curves, dataset, method, target)
                    if stats is None:
                        continue
                    xs.append(stats[0] / 1_000_000)
                    ys.append(target)
                    panel_x_values.append(xs[-1])
                    panel_y_values.append(target)
                if not xs:
                    continue
                traces.append((method, family, optimizer, xs, ys))
            panel_data[(regime, dataset)] = traces
            if panel_x_values and panel_y_values:
                x_span = max(panel_x_values) - min(panel_x_values)
                y_span = max(panel_y_values) - min(panel_y_values)
                x_pad = max(2.6, 0.065 * x_span)
                y_pad = max(0.030, 0.095 * y_span)
                xmin = max(0.0, math.floor((min(panel_x_values) - x_pad) / 5.0) * 5.0)
                xmax = math.ceil((max(panel_x_values) + x_pad) / 5.0) * 5.0
                ymin = math.floor((min(panel_y_values) - y_pad) * 20.0) / 20.0
                ymax = math.ceil((max(panel_y_values) + y_pad) * 20.0) / 20.0
                if xmax <= xmin:
                    xmax = xmin + 5.0
                if ymax <= ymin:
                    ymax = ymin + 0.05
                panel_ranges[(regime, dataset)] = (xmin, xmax, ymin, ymax)
            else:
                panel_ranges[(regime, dataset)] = None

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(7.2, 4.18), sharex=False, sharey=False)
    fig.subplots_adjust(left=0.060, right=0.996, bottom=0.114, top=0.855, wspace=0.235, hspace=0.255)

    for row_idx, regime in enumerate(["E1", "E2"]):
        for col_idx, (dataset, dataset_label, _e2_dir_name) in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            traces = panel_data[(regime, dataset)]
            for method, family, optimizer, xs, ys in traces:
                color, linestyle, linewidth, alpha = _target_line_style(method)
                is_matrix_policy = method == "rlb_matrixpolicy_original"
                is_primary_optimizer = optimizer in {"AdamW", "Muon"}
                marker = "o" if is_matrix_policy else ("s" if family == "RLB" else "^")
                base_size = 8.5 if is_matrix_policy else (5.4 if is_primary_optimizer else 3.2)
                final_size = 28 if is_matrix_policy else (13.0 if is_primary_optimizer else 6.0)
                sizes = [base_size] * len(xs)
                sizes[-1] = final_size
                line_z = 7 if is_matrix_policy else (5 if is_primary_optimizer else 2)
                point_z = 9 if is_matrix_policy else (6 if is_primary_optimizer else 3)
                scatter_alpha = 1.0 if is_matrix_policy else (
                    min(0.95, alpha + 0.10) if is_primary_optimizer else min(0.32, alpha + 0.08)
                )
                scatter_lw = 0.42 if is_matrix_policy or is_primary_optimizer else 0.12
                ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=line_z)
                ax.scatter(
                    xs,
                    ys,
                    color=color,
                    marker=marker,
                    s=sizes,
                    alpha=scatter_alpha,
                    edgecolor="white",
                    linewidth=scatter_lw,
                    zorder=point_z,
                )
            limits = panel_ranges[(regime, dataset)]
            if limits is not None:
                xmin, xmax, ymin, ymax = limits
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            if not traces:
                ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color="#777777")
            if row_idx == 0:
                ax.set_title(dataset_label, fontsize=6.55, pad=3.0, weight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{BUDGET_LABELS[regime]}\ntarget val. loss", fontsize=6.6, labelpad=2.0)
            ax.grid(True, color="#dddddd", linewidth=0.36, alpha=0.72)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=5.15, pad=1.0, length=1.9)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
    fig.text(0.535, 0.043, "tokens to validation-loss target (M)", ha="center", va="center", fontsize=7.1)

    present_optimizers = []
    for methods in method_sets.values():
        for _method, _label, family, optimizer in methods:
            if family != "MatrixPolicy" and optimizer not in present_optimizers:
                present_optimizers.append(optimizer)
    optimizer_order = [name for name in ["AdamW", "Muon", "Lion", "SOAP", "ScheduleFree", "CAME"] if name in present_optimizers]
    method_handles = [
        plt.Line2D([0], [0], color=OKABE_ITO["black"], linestyle="-", linewidth=2.05, marker="o", markersize=4.3),
        plt.Line2D([0], [0], color="#303030", linestyle="-", linewidth=1.20, marker="s", markersize=3.4),
        plt.Line2D([0], [0], color="#303030", linestyle=(0, (2.4, 1.8)), linewidth=1.20, marker="^", markersize=3.5),
    ]
    method_labels = ["MatrixPolicy", "RLB + optimizer", "SiLU + optimizer"]
    optimizer_handles = []
    for optimizer in optimizer_order:
        is_primary = optimizer in {"AdamW", "Muon"}
        optimizer_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=TARGET_OPTIMIZER_COLORS[optimizer],
                linestyle="-",
                linewidth=1.75 if is_primary else 1.00,
                alpha=0.96 if is_primary else 0.48,
            )
        )
    method_legend = fig.legend(
        method_handles,
        method_labels,
        loc="upper left",
        bbox_to_anchor=(0.070, 0.985),
        ncol=3,
        frameon=False,
        fontsize=6.05,
        title="Activation / method",
        title_fontsize=6.20,
        handlelength=1.45,
        columnspacing=0.70,
        handletextpad=0.35,
    )
    fig.add_artist(method_legend)
    fig.legend(
        optimizer_handles,
        optimizer_order,
        loc="upper right",
        bbox_to_anchor=(0.994, 0.985),
        ncol=len(optimizer_order),
        frameon=False,
        fontsize=5.95,
        title="RLB/SiLU optimizer color",
        title_fontsize=6.15,
        handlelength=1.28,
        columnspacing=0.58,
        handletextpad=0.28,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
    plt.close(fig)

def _fmt_signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}"


def _budget_table_label(regime: str) -> str:
    return {"E1": "100M", "E2": "300M"}[regime]


def make_e1_e2_silu_summary_table(out_path: Path) -> None:
    rows = _target_arrival_rows()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Target-arrival efficiency for the fixed 12-layer, width-768 Transformer. RLB denotes the same rational latent-basis activation without MatrixPolicy. At the hardest validation-loss target reached by every listed method in all three seeds, MatrixPolicy is always fastest; each token-saving column reports MatrixPolicy's savings relative to the method named below it.}",
        r"\label{tab:e1e2-silu-comparison}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.9pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{tabular}{@{}llc*{4}{r}@{\hspace{0.55em}}*{5}{r}@{}}",
        r"\toprule",
        r"& & & \multicolumn{4}{c}{MatrixPolicy token savings (\%)} & \multicolumn{5}{c}{Time to target (min)} \\",
        r"\cmidrule(lr){4-7}\cmidrule(l){8-12}",
        r"Budget & Dataset & \shortstack{Target\\loss} & \shortstack{vs. SiLU\\AdamW} & \shortstack{vs. SiLU\\Muon} & \shortstack{vs. RLB\\AdamW} & \shortstack{vs. RLB\\Muon} & \shortstack{Matrix\\Policy} & \shortstack{SiLU\\AdamW} & \shortstack{SiLU\\Muon} & \shortstack{RLB\\AdamW} & \shortstack{RLB\\Muon} \\",
        r"\midrule",
    ]
    for row_idx, row in enumerate(rows):
        by_method = {comp["method"]: comp for comp in row["comparators"]}
        saved_cells = []
        time_cells = []
        for method, _label in MAIN_COMPARATORS:
            comp = by_method[method]
            saved_cells.append(f"{100.0 * float(comp['saved_fraction']):.1f}")
            time_cells.append(f"{float(comp['time_min']):.1f}")
        budget_cell = _budget_table_label(str(row["regime"])) if row_idx in {0, 5} else ""
        lines.append(
            f"{budget_cell} & {row['dataset_label']} & {float(row['target']):.2f} & "
            + " & ".join(saved_cells)
            + f" & {float(row['mp_time_min']):.1f} & "
            + " & ".join(time_cells)
            + r" \\")
        if row_idx == 4:
            lines.append(r"\addlinespace[1pt]\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\endgroup",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n")

def _fmt_token_fraction(saved_tokens_m: float, saved_fraction: float) -> str:
    return f"{saved_tokens_m:.1f} ({100.0 * saved_fraction:.1f}\\%)"


def _final_cell(curves, dataset: str, method: str) -> str:
    return _fmt_mean_std(_final_eval_mean(curves, dataset, method, "val_loss"), digits=3)


def make_broad_final_validation_table(out_path: Path) -> None:
    curves_by_regime = {"E1": load_e1_curves(), "E2": load_e2_curves()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Final validation loss for the broader optimizer sweep on the fixed 12-layer, width-768 language model. Values are mean $\pm$ sample standard deviation over three seeds; lower is better. RLB rows use the same rational latent-basis activation without MatrixPolicy.}",
        r"\label{tab:broad-final-validation}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccc@{}}",
        r"\toprule",
        r"Budget & Method & DCLM & FineWeb-Edu & FineWeb & Dolma & C4 \\",
        r"\midrule",
    ]
    for panel_idx, regime in enumerate(["E1", "E2"]):
        if panel_idx:
            lines.append(r"\midrule")
        curves = curves_by_regime[regime]
        methods = _completed_broad_methods(curves)
        for method_idx, (method, label, _family, _optimizer) in enumerate(methods):
            cells = [_final_cell(curves, dataset, method) for dataset, _dataset_label, _e2 in DATASETS]
            budget_cell = BUDGET_LABEL[regime] if method_idx == 0 else ""
            lines.append(f"{budget_cell} & {label} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\endgroup",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n")

def _load_e8_sensitivity_records() -> dict[tuple[str, str, str, str], dict[str, object]]:
    manifest = ROOT / "experiments" / "manifests" / "iclr26_e8_primary_manifest.csv"
    run_root = ROOT / "experiments" / "runs" / "iclr26_main" / "E8_primary_100m"
    records: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in _read_rows(manifest):
        if row.get("phase") != "E8_primary_100m":
            continue
        jsonl_path = run_root / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(jsonl_path)
        evals: list[dict[str, object]] = []
        summary: dict[str, object] | None = None
        with jsonl_path.open("r", errors="replace") as handle:
            for raw in handle:
                if not raw.startswith("{"):
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "eval":
                    evals.append(record)
                elif record.get("event") == "summary":
                    summary = record
        if summary is None or int(summary.get("completed_steps", -1)) < int(row["steps"]):
            raise RuntimeError(f"incomplete E8 sensitivity JSONL: {jsonl_path}")
        key = (row["dataset"], row["lr"], row["weight_decay"], row["method"])
        records[key] = {"row": row, "evals": evals, "summary": summary}
    return records


def _e8_best_loss(record: dict[str, object]) -> float:
    losses = [
        float(entry["val_loss"])
        for entry in record["evals"]  # type: ignore[index]
        if isinstance(entry.get("val_loss"), (int, float)) and math.isfinite(float(entry["val_loss"]))
    ]
    if not losses:
        raise RuntimeError("E8 sensitivity record has no finite validation losses")
    return min(losses)


def _e8_first_hit(record: dict[str, object], target: float) -> tuple[int, float, float] | None:
    row = record["row"]  # type: ignore[index]
    summary = record["summary"]  # type: ignore[index]
    completed_steps = float(summary.get("completed_steps", 0.0))
    total_seconds = float(summary.get("total_seconds", 0.0))
    if completed_steps <= 0.0 or total_seconds <= 0.0:
        raise RuntimeError(f"E8 sensitivity record has invalid timing summary: {row['row_id']}")
    seconds_per_step = total_seconds / completed_steps
    for entry in record["evals"]:  # type: ignore[index]
        loss = entry.get("val_loss")
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)) and float(loss) <= target:
            step = int(entry["step"])
            tokens = step * int(row["global_tokens_per_step"])
            minutes = step * seconds_per_step / 60.0
            return step, float(tokens), float(minutes)
    return None


def _fmt_median_range(values: list[float], digits: int = 1) -> str:
    if not values:
        raise RuntimeError("cannot format empty value list")
    arr = np.asarray(values, dtype=float)
    return f"{float(np.median(arr)):.{digits}f} [{float(np.min(arr)):.{digits}f},{float(np.max(arr)):.{digits}f}]"


def _fmt_median(values: list[float], digits: int = 1) -> str:
    if not values:
        raise RuntimeError("cannot format empty value list")
    return f"{float(np.median(np.asarray(values, dtype=float))):.{digits}f}"


def _e8_sensitivity_rows() -> list[dict[str, object]]:
    records = _load_e8_sensitivity_records()
    datasets = [(dataset, label) for dataset, label, _e2 in DATASETS]
    methods = ["rlb_matrixpolicy_original", "silu_adamw", "silu_muon"]
    rows: list[dict[str, object]] = []
    for dataset, label in datasets:
        dataset_records = [
            record
            for (record_dataset, _lr, _wd, method), record in records.items()
            if record_dataset == dataset and method in methods
        ]
        if len(dataset_records) != 48:
            raise RuntimeError(f"unexpected E8 sensitivity coverage for {dataset}: {len(dataset_records)}")
        raw_target = max(_e8_best_loss(record) for record in dataset_records)
        hard_target = math.ceil(raw_target * 20.0) / 20.0
        targets = [hard_target + 0.10, hard_target + 0.05, hard_target]
        for target_index, target in enumerate(targets):
            target = round(target, 2)
            savings_vs_adamw: list[float] = []
            savings_vs_muon: list[float] = []
            mp_times: list[float] = []
            adamw_times: list[float] = []
            muon_times: list[float] = []
            wins_adamw = wins_muon = 0
            for lr in sorted({key[1] for key in records if key[0] == dataset}):
                for wd in sorted({key[2] for key in records if key[0] == dataset}):
                    mp_hit = _e8_first_hit(records[(dataset, lr, wd, "rlb_matrixpolicy_original")], target)
                    adamw_hit = _e8_first_hit(records[(dataset, lr, wd, "silu_adamw")], target)
                    muon_hit = _e8_first_hit(records[(dataset, lr, wd, "silu_muon")], target)
                    if mp_hit is None or adamw_hit is None or muon_hit is None:
                        raise RuntimeError(f"E8 sensitivity common target was not reached: {dataset} {lr} {wd} {target}")
                    _mp_step, mp_tokens, mp_minutes = mp_hit
                    _adamw_step, adamw_tokens, adamw_minutes = adamw_hit
                    _muon_step, muon_tokens, muon_minutes = muon_hit
                    savings_vs_adamw.append(100.0 * (adamw_tokens - mp_tokens) / adamw_tokens)
                    savings_vs_muon.append(100.0 * (muon_tokens - mp_tokens) / muon_tokens)
                    mp_times.append(mp_minutes)
                    adamw_times.append(adamw_minutes)
                    muon_times.append(muon_minutes)
                    wins_adamw += int(mp_tokens < adamw_tokens)
                    wins_muon += int(mp_tokens < muon_tokens)
            rows.append({
                "dataset": label,
                "show_dataset": target_index == 0,
                "target": target,
                "cells": 16,
                "wins_adamw": wins_adamw,
                "wins_muon": wins_muon,
                "savings_vs_adamw": savings_vs_adamw,
                "savings_vs_muon": savings_vs_muon,
                "mp_times": mp_times,
                "adamw_times": adamw_times,
                "muon_times": muon_times,
            })
    return rows


def make_e8_sensitivity_target_arrival_table(out_path: Path) -> None:
    rows = _e8_sensitivity_rows()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Target-arrival efficiency under learning-rate and weight-decay sensitivity at the 100M-token budget. Entries are medians over 16 paired hyperparameter cells.}",
        r"\label{tab:e8-sensitivity-target-arrival}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.9pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular}{@{}llc*{2}{r}@{\hspace{0.55em}}*{3}{r}@{}}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{MatrixPolicy token savings (\%)} & \multicolumn{3}{c}{Time to target (min)} \\",
        r"\cmidrule(lr){4-5}\cmidrule(l){6-8}",
        r"Budget & Dataset & \shortstack{Target\\loss} & \shortstack{vs. SiLU\\AdamW} & \shortstack{vs. SiLU\\Muon} & \shortstack{Matrix\\Policy} & \shortstack{SiLU\\AdamW} & \shortstack{SiLU\\Muon} \\",
        r"\midrule",
    ]
    for row_index, row in enumerate(rows):
        if row_index and row["show_dataset"]:
            lines.append(r"\addlinespace[1pt]\midrule")
        budget_cell = "100M" if row_index == 0 else ""
        dataset_cell = str(row["dataset"]) if row["show_dataset"] else ""
        lines.append(
            f"{budget_cell} & {dataset_cell} & {float(row['target']):.2f} & "
            f"{_fmt_median(row['savings_vs_adamw'])} & "
            f"{_fmt_median(row['savings_vs_muon'])} & "
            f"{_fmt_median(row['mp_times'])} & "
            f"{_fmt_median(row['adamw_times'])} & "
            f"{_fmt_median(row['muon_times'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\endgroup",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n")

def main() -> int:
    outputs = [
        OUT_DIR / "matrixpolicy_overview.pdf",
        OUT_DIR / "matrixpolicy_signal_flow.pdf",
        OUT_DIR / "e1_multimetric_all_datasets.pdf",
        OUT_DIR / "target_arrival_evidence_matrix.pdf",
        OUT_DIR / "e1_representative_silu_dynamics.pdf",
        OUT_DIR / "e2_multimetric_all_datasets.pdf",
        TABLE_DIR / "e1_e2_silu_summary_table.tex",
        TABLE_DIR / "final_validation_broad_optimizer_table.tex",
        TABLE_DIR / "e8_sensitivity_target_arrival_table.tex",
    ]
    compile_tikz_figure("matrixpolicy_overview.tex", outputs[0])
    compile_tikz_figure("matrixpolicy_signal_flow.tex", outputs[1])
    make_e1_multimetric_all_datasets(outputs[2])
    make_target_arrival_evidence_matrix(outputs[3])
    make_e1_representative_silu_dynamics(outputs[4])
    make_e2_multimetric_all_datasets(outputs[5])
    make_e1_e2_silu_summary_table(outputs[6])
    make_broad_final_validation_table(outputs[7])
    make_e8_sensitivity_target_arrival_table(outputs[8])
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
