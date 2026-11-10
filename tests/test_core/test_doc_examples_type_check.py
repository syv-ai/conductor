"""Every Python example a reader copies passes mypy (default settings) and pyright (standard mode).

The examples are every document's, as ``tests.test_core.doc_examples`` reads them: each one module.

pyright is a development dependency; where it is not installed the pyright test is skipped, saying so.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import warnings

import pytest

from tests.test_core.doc_examples import ROOT, documents


@pytest.fixture(scope="module")
def modules(tmp_path_factory: pytest.TempPathFactory) -> list[pathlib.Path]:
    directory = tmp_path_factory.mktemp("doc_examples")
    written = []
    for name, source in documents().items():
        path = directory / f"{name}.py"
        path.write_text(source)
        written.append(path)
    return written


def test_every_document_with_python_is_read() -> None:
    names = set(documents())
    assert {"doc_README_md", "doc_docs_writing_nodes_md", "doc_skills_add_node_REFERENCE_md", "doc_examples_03_what_the_run_supplies_ipynb"} <= names, names


def test_the_doc_examples_pass_mypy(modules: list[pathlib.Path], tmp_path: pathlib.Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path), "--no-error-summary", *map(str, modules)],
        capture_output=True, text=True, cwd=modules[0].parent,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_doc_examples_pass_pyright(modules: list[pathlib.Path]) -> None:
    pyright = pathlib.Path(sys.executable).with_name("pyright")
    command = str(pyright) if pyright.exists() else shutil.which("pyright")
    if command is None:
        warnings.warn("pyright is not installed, so the doc examples were not checked by it", stacklevel=1)
        pytest.skip("pyright is not installed: the doc examples were not checked by it")
    directory = modules[0].parent
    (directory / "pyrightconfig.json").write_text(json.dumps({
        "extraPaths": [str(ROOT / "packages" / "conductor" / "src")],
        "typeCheckingMode": "standard",
        "venvPath": str(pathlib.Path(sys.prefix).parent),
        "venv": pathlib.Path(sys.prefix).name,
    }))
    done = subprocess.run([command, "--outputjson", *map(str, modules)], capture_output=True, text=True, cwd=directory)
    report = json.loads(done.stdout)
    errors = [f"{pathlib.Path(d['file']).name}:{d['range']['start']['line'] + 1}: {d['message']}" for d in report["generalDiagnostics"] if d["severity"] == "error"]
    assert not errors, "\n".join(errors)
