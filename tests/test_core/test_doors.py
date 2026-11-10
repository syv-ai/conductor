"""The public doors: ``conductor``, ``conductor.widgets``, ``conductor.events`` and ``conductor.testing``.

Everything a caller outside the library needs is exported from one of
these, and each lists its names in ``__all__``. The lists are pinned here
so that adding or dropping a public name is a visible, reviewed change; a
type checker accepts every name (``mypy --strict`` and pyright each refuse
an import a package does not export explicitly).
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mypy import api as mypy_api

ROOT = Path(__file__).resolve().parents[2]

DOORS: dict[str, list[str]] = {
    "conductor": [
        "Ask", "Choice", "Deprecation", "ExternalFailure", "FromRun", "Graph", "GraphInput", "GraphRecord", "InvalidCallError", "NodeDeclarationError",
        "NodeDefinition", "NodeError", "NodeRecord", "NodeRegistry", "PROBLEM_CODES", "Param", "Policy", "Problem", "Question", "Receives", "Run", "RunStatus", "RunsOn",
        "SKIPPED", "Series", "Table", "Upgrade", "Waiting", "make_graph", "node",
    ],
    "conductor.widgets": [
        "DatePicker", "Dropdown", "FileUpload", "ListInput", "NumberInput", "Slider", "Switch",
        "TextInput", "Textarea", "Widget",
    ],
    "conductor.events": [
        "Event", "NodeComplete", "NodeFailed", "NodeProgress", "NodeRetry", "NodeSkipped", "NodeStart", "RunCheckpoint",
        "RunEnd", "RunStart", "UnitWaiting",
    ],
    "conductor.testing": ["check_registry"],
}


def test_the_package_holds_the_doors_and_one_private_directory():
    package = ROOT / "packages" / "conductor" / "src" / "conductor"
    entries = {entry.name for entry in package.iterdir() if entry.name != "__pycache__"}
    assert entries == {"__init__.py", "widgets.py", "events.py", "testing.py", "about", "_internal", "py.typed"}


@pytest.mark.parametrize("door", sorted(DOORS))
def test_a_door_exports_exactly_its_pinned_names(door):
    module = importlib.import_module(door)
    assert sorted(module.__all__) == DOORS[door]
    for name in module.__all__:
        getattr(module, name)


def test_a_type_checker_accepts_every_name_of_every_door(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("".join(f"from {door} import {', '.join(names)}\n" for door, names in DOORS.items()))
    stdout, _, status = mypy_api.run(["--strict", "--no-incremental", "--follow-imports=silent", str(probe)])
    assert "explicitly export" not in stdout, stdout
    assert status == 0, stdout


def test_pyright_accepts_every_name_of_every_door(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("".join(f"from {door} import {', '.join(names)}\n" for door, names in DOORS.items()))
    pyright = Path(sys.executable).with_name("pyright")
    found = subprocess.run(
        [str(pyright), "--outputjson", "--pythonpath", sys.executable, str(probe)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    report = json.loads(found.stdout)
    assert report["generalDiagnostics"] == [], report["generalDiagnostics"]
