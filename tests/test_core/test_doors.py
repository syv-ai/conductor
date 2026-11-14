"""The public modules: ``conductor``, ``conductor.widgets``, ``conductor.events``, ``conductor.testing`` and
``conductor.about``.

Everything a caller outside the library needs is exported from one of these, and each lists its names in
``__all__``. Every name listed exists, and a type checker accepts every one (``mypy --strict`` and pyright each
refuse an import a package does not export explicitly).
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mypy import api as mypy_api

ROOT = Path(__file__).resolve().parents[2]

#: The modules a caller outside the library imports from.
PUBLIC_MODULES = ("conductor", "conductor.widgets", "conductor.events", "conductor.testing", "conductor.about")


def _imports_of_every_name() -> str:
    """A module that imports every name each public module lists."""
    modules = {name: importlib.import_module(name).__all__ for name in PUBLIC_MODULES}
    return "".join(f"from {module} import {', '.join(names)}\n" for module, names in modules.items() if names)


def test_the_package_has_only_the_public_modules_and_one_private_directory():
    package = ROOT / "packages" / "conductor" / "src" / "conductor"
    entries = {entry.name for entry in package.iterdir() if entry.name != "__pycache__"}
    assert entries == {"__init__.py", "widgets.py", "events.py", "testing", "about", "_internal", "py.typed"}


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_every_name_a_public_module_lists_exists_once(module_name):
    module = importlib.import_module(module_name)
    assert len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        getattr(module, name)


def test_a_type_checker_accepts_every_name_of_every_public_module(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(_imports_of_every_name())
    stdout, _, status = mypy_api.run(["--strict", "--no-incremental", "--follow-imports=silent", str(probe)])
    assert "explicitly export" not in stdout, stdout
    assert status == 0, stdout


def test_pyright_accepts_every_name_of_every_public_module(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(_imports_of_every_name())
    pyright = Path(sys.executable).with_name("pyright")
    found = subprocess.run(
        [str(pyright), "--outputjson", "--pythonpath", sys.executable, str(probe)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    report = json.loads(found.stdout)
    assert report["generalDiagnostics"] == [], report["generalDiagnostics"]
