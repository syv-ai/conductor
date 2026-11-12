"""The layers contract ``pyproject.toml`` declares holds: each package of ``conductor._internal`` imports only the
packages below it. The contract names the one import it allows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_layers_contract_holds() -> None:
    # in a process of its own: import-linter sets up logging for its output, which the other tests read
    found = subprocess.run([Path(sys.executable).with_name("lint-imports"), "--no-cache"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert found.returncode == 0, found.stdout
