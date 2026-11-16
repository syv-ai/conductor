"""The packages inside ``conductor._internal`` import only the packages below them.

The layers are written down once, as an import-linter contract in ``pyproject.toml``. This test runs import-linter
on it. The contract also lists the one upward import it allows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_layers_contract_holds() -> None:
    # Run it in its own process. import-linter sets up logging for its output, and other tests read the logs.
    found = subprocess.run(
        [Path(sys.executable).with_name("lint-imports"), "--no-cache"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert found.returncode == 0, found.stdout
