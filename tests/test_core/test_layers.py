"""The import contracts ``pyproject.toml`` declares hold: the layers of ``conductor._internal``, and that a package
is imported only through its ``__init__`` (``tests/import_contracts.py``). The contracts say what they allow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_import_contracts_hold() -> None:
    # in a process of its own: import-linter sets up logging for its output, which the other tests read
    found = subprocess.run([Path(sys.executable).with_name("lint-imports"), "--no-cache"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert found.returncode == 0, found.stdout
