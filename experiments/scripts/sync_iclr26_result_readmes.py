#!/usr/bin/env python3
"""Synchronize active Markdown result mirrors from generated result packages."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
RUNTIME_README = RESULTS / "iclr26_runtime_summary_2026_06_11" / "README.md"
E1_TOKEN_README = RESULTS / "iclr26_e1_token_savings_2026_06_12" / "README.md"
E1_FIGURE_README = RESULTS / "iclr26_e1_figures" / "README.md"
E2_FIGURE_README = RESULTS / "iclr26_e2_figures" / "README.md"
CORRECTION_MANIFEST = (
    ROOT
    / "experiments"
    / "corrections"
    / "matrixpolicy_live_stats_20260712"
    / "manifests"
    / "matrixpolicy_live_stats_20260712_main.csv"
)

E2_PACKAGES = [
    ("DCLM", RESULTS / "iclr26_e2_dclm_2026_06_10" / "README.md"),
    ("FineWeb-Edu", RESULTS / "iclr26_e2_fineweb_edu_2026_06_12" / "README.md"),
    ("FineWeb", RESULTS / "iclr26_e2_fineweb_2026_06_15" / "README.md"),
    ("Dolma-sample", RESULTS / "iclr26_e2_dolma_sample_2026_06_17" / "README.md"),
    ("C4", RESULTS / "iclr26_e2_c4_2026_06_19" / "README.md"),
]

TARGETS = [
    ROOT / "README.md",
    ROOT / "experiments" / "README.md",
    ROOT / "experiments" / "ICLR_RUN_STATUS.md",
]


def heading_level(line: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else None


def section(text: str, heading: str) -> str:
    lines = text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError as exc:
        raise RuntimeError(f"missing Markdown heading: {heading}") from exc
    level = heading_level(heading)
    if level is None:
        raise RuntimeError(f"not a Markdown heading: {heading}")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        candidate_level = heading_level(lines[index])
        if candidate_level is not None and candidate_level <= level:
            end = index
            break
    return "\n".join(lines[start:end]).strip() + "\n"


def replace_section(text: str, heading: str, replacement: str) -> str:
    old = section(text, heading)
    return text.replace(old, replacement.strip() + "\n", 1)


def body_without_title(text: str) -> str:
    lines = text.splitlines()
    if not lines or heading_level(lines[0]) != 1:
        raise RuntimeError("generated README does not start with a level-one title")
    return "\n".join(lines[1:]).strip()


def shift_headings(text: str, amount: int) -> str:
    shifted = []
    for line in text.splitlines():
        level = heading_level(line)
        if level is None:
            shifted.append(line)
            continue
        new_level = level + amount
        if new_level > 6:
            raise RuntimeError(f"heading shift exceeds Markdown depth: {line}")
        shifted.append("#" * new_level + line[level:])
    return "\n".join(shifted)


def relpath(path: Path, target: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix() if target.parent == ROOT else Path(
        os.path.relpath(path.resolve(), target.parent.resolve())
    ).as_posix()


def rewrite_links(text: str, source: Path, target: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if re.match(r"^[a-z]+://", url) or url.startswith("#"):
            return match.group(0)
        absolute = (source.parent / url).resolve()
        return f"[{label}]({relpath(absolute, target)})"

    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", replace, text)


def source_path(path: Path, target: Path) -> str:
    return f"`{relpath(path, target)}`"


def runtime_intro(target: Path, runtime_text: str) -> str:
    first_table = section(runtime_text, "## E1 M0/100M All Datasets")
    body = body_without_title(runtime_text)
    intro = body[: body.index(first_table.strip())].strip()
    return "## Runtime Accounting\n\n" + rewrite_links(intro, RUNTIME_README, target)


def e1_section(target: Path, runtime_text: str, token_text: str, figure_text: str) -> str:
    runtime_table = section(runtime_text, "## E1 M0/100M All Datasets")
    runtime_table = runtime_table.replace("## E1 M0/100M All Datasets", "### E1 Runtime Table", 1)
    token_body = shift_headings(body_without_title(token_text), 2)
    figure_body = shift_headings(body_without_title(figure_text), 2)
    correction = source_path(CORRECTION_MANIFEST, target)
    controls = source_path(
        ROOT / "experiments" / "manifests" / "iclr26_global_rational_optimizer_controls_manifest.csv",
        target,
    )
    intro = (
        "## Current E1 M0/100M Results\n\n"
        "E1 is complete across DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4 "
        "with three seeds, 15 matched methods per dataset/seed cell, validation every "
        "50 steps, and final evaluation at step `3050`. MatrixPolicy rows use the "
        f"validated live-statistic correction manifest {correction}; non-MatrixPolicy "
        f"RLB controls use {controls}; SiLU controls remain the completed main-manifest rows. "
        "All 15 corrected MatrixPolicy rows completed without restart.\n\n"
    )
    return (
        intro
        + runtime_table.strip()
        + "\n\n### E1 Token-To-Target Savings\n\n"
        + rewrite_links(token_body, E1_TOKEN_README, target).strip()
        + "\n\n### E1 Dense Curves And Checkpoint Tables\n\n"
        + rewrite_links(figure_body, E1_FIGURE_README, target).strip()
    )


def e2_runtime_tables(target: Path, runtime_text: str) -> str:
    blocks = ["### E2 Runtime Tables"]
    for label, _package in E2_PACKAGES:
        source_heading = f"## E2 M0/300M {label}"
        block = section(runtime_text, source_heading)
        block = block.replace(source_heading, f"#### {label}", 1)
        blocks.append(rewrite_links(block, RUNTIME_README, target).strip())
    return "\n\n".join(blocks)


def e2_section(target: Path, runtime_text: str, figure_text: str) -> str:
    correction = source_path(CORRECTION_MANIFEST, target)
    controls = source_path(
        ROOT / "experiments" / "manifests" / "iclr26_global_rational_optimizer_controls_manifest.csv",
        target,
    )
    blocks = [
        "## Current E2 M0/300M Results\n\n"
        "E2 is complete across the same five datasets with three seeds and 15 fixed "
        "methods per seed. MatrixPolicy rows use the validated live-statistic correction "
        f"manifest {correction}; non-MatrixPolicy RLB controls use {controls}; SiLU controls "
        "remain the completed main-manifest rows. All 15 corrected MatrixPolicy rows "
        "completed without restart. RLB+ADeMaMix stopped early with non-finite loss in "
        "every seed and remains reported rather than excluded."
    ]
    for label, package in E2_PACKAGES:
        package_body = shift_headings(body_without_title(package.read_text()), 2)
        blocks.append(
            f"### {label}\n\nTracked package: {source_path(package.parent, target)}.\n\n"
            + rewrite_links(package_body, package, target).strip()
        )
    blocks.append(e2_runtime_tables(target, runtime_text))
    figure_body = shift_headings(body_without_title(figure_text), 2)
    blocks.append(
        "### E2 Dense Curves And Checkpoint Tables\n\n"
        + rewrite_links(figure_body, E2_FIGURE_README, target).strip()
    )
    return "\n\n".join(blocks)


def main() -> int:
    runtime_text = RUNTIME_README.read_text()
    token_text = E1_TOKEN_README.read_text()
    e1_figure_text = E1_FIGURE_README.read_text()
    e2_figure_text = E2_FIGURE_README.read_text()
    for target in TARGETS:
        text = target.read_text()
        text = replace_section(text, "## Runtime Accounting", runtime_intro(target, runtime_text))
        text = replace_section(
            text,
            "## Current E1 M0/100M Results",
            e1_section(target, runtime_text, token_text, e1_figure_text),
        )
        text = replace_section(
            text,
            "## Current E2 M0/300M Results",
            e2_section(target, runtime_text, e2_figure_text),
        )
        if target.name == "ICLR_RUN_STATUS.md":
            text = re.sub(r"^Updated: .*?$", f"Updated: {date.today().isoformat()} EDT", text, count=1, flags=re.M)
        target.write_text(text.rstrip() + "\n")
        print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
