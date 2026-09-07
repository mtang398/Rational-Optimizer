#!/usr/bin/env python3
"""Build or check the hash inventory for public reproducibility sources."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE / "SOURCE_FREEZE.sha256"

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".py",
    ".pyi",
    ".sbatch",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
SOURCE_NAMES = {"pytest.ini", "requirements.txt", "setup.py"}
MANIFESTS = {
    "experiments/protocol/activation_optimizer_manifest.csv",
    "experiments/protocol/matrix.json",
    "experiments/protocol/token_fingerprints.json",
}
EXCLUDED_PREFIXES = (
    "experiments/cache/",
    "experiments/failed_attempts/",
    "experiments/logs/",
    "experiments/results/",
    "experiments/runs/",
)
EXCLUDED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".dll",
    ".dylib",
    ".egg",
    ".gif",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".o",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".so",
    ".tar",
    ".tgz",
    ".whl",
    ".zip",
}


class FreezeError(RuntimeError):
    """Raised when the public source inventory cannot be constructed."""


def public_inventory(root: Path = ROOT) -> tuple[str, ...]:
    """Return existing, non-ignored files exactly as a clean clone would expose."""
    command = [
        "git",
        "-C",
        str(root),
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise FreezeError("a Git work tree is required to enumerate public files") from exc

    inventory: list[str] = []
    for relative in completed.stdout.splitlines():
        path = root / relative
        if path.is_file() and not path.is_symlink():
            inventory.append(Path(relative).as_posix())
    return tuple(sorted(set(inventory)))


def is_frozen_source(relative: str) -> bool:
    """Select source and experiment manifests, excluding generated artifacts."""
    if relative == OUTPUT.relative_to(ROOT).as_posix():
        return False
    if relative.startswith(EXCLUDED_PREFIXES):
        return False
    path = Path(relative)
    if "__pycache__" in path.parts or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return (
        relative in MANIFESTS
        or path.name in SOURCE_NAMES
        or path.suffix.lower() in SOURCE_SUFFIXES
    )


def frozen_paths(root: Path = ROOT) -> tuple[str, ...]:
    paths = tuple(
        relative for relative in public_inventory(root) if is_frozen_source(relative)
    )
    missing = sorted(MANIFESTS - set(paths))
    if missing:
        raise FreezeError(f"required manifest is not public: {', '.join(missing)}")
    required_sources = {
        "experiments/protocol/build_source_freeze.py",
        "experiments/protocol/test_suite.py",
        "experiments/protocol/verify_repository.py",
        "optimizer_design/tiller.py",
        "training/train.py",
    }
    missing_sources = sorted(required_sources - set(paths))
    if missing_sources:
        raise FreezeError(
            f"required source is not public: {', '.join(missing_sources)}"
        )
    return paths


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_freeze(root: Path = ROOT) -> str:
    return "".join(
        f"{sha256(root / relative)}  {relative}\n"
        for relative in frozen_paths(root)
    )


def write_freeze(output: Path = OUTPUT, root: Path = ROOT) -> None:
    payload = render_freeze(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def check_freeze(output: Path = OUTPUT, root: Path = ROOT) -> None:
    if not output.is_file():
        raise FreezeError(f"source freeze is missing: {output.relative_to(root)}")
    expected = render_freeze(root)
    actual = output.read_text(encoding="utf-8")
    if actual != expected:
        raise FreezeError("source freeze is stale; rebuild it with build_source_freeze.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the committed freeze without modifying it",
    )
    args = parser.parse_args()
    try:
        if args.check:
            check_freeze()
            print(f"PASS source freeze ({len(frozen_paths())} files)")
        else:
            write_freeze()
            print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(frozen_paths())} files)")
    except FreezeError as exc:
        parser.exit(1, f"FAIL {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
