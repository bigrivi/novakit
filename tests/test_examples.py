"""Compile the real-provider quickstart and smoke-test offline examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _discover_examples() -> list[Path]:
    """Return sorted offline ``examples/NN_*.py`` files, or empty if none exist.

    ``01_quickstart.py`` is a real-provider tutorial requiring credentials and
    network access, so it is excluded from offline smoke.
    """
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        p for p in EXAMPLES_DIR.glob("[0-9][0-9]_*.py") if p.name != "01_quickstart.py"
    )


def test_examples_discovered_non_empty() -> None:
    """Fail loudly when no offline examples exist; silent smoke is a bug."""
    assert _discover_examples(), (
        f"no offline examples/[0-9][0-9]_*.py found under {EXAMPLES_DIR}"
    )


def test_quickstart_compiles() -> None:
    """Compile ``01_quickstart.py`` without executing it (no network)."""
    path = EXAMPLES_DIR / "01_quickstart.py"
    assert path.is_file(), f"missing real-provider quickstart: {path}"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("example", _discover_examples(), ids=lambda p: p.name)
def test_example_runs_offline(example: Path) -> None:
    """Run one example offline in a subprocess with a 45s timeout."""
    proc = subprocess.run(
        [sys.executable, str(example)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert proc.returncode == 0, (
        f"{example.name} exited with {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
