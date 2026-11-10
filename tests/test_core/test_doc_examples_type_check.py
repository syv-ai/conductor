"""Every Python example a reader copies passes mypy (default settings) and pyright (standard mode).

The examples are every ``python`` block of the README files, ``docs/``, ``skills/`` and ``about/llms.txt``, and every
code cell of the notebooks; each document is one module, its blocks in order, and a notebook cell that awaits at the top
is read as a coroutine. ``llms.txt`` names conductor's public
names without importing them, so it is read after a star import of ``conductor`` and ``conductor.widgets``.

pyright is a development dependency; where it is not installed the pyright test is skipped, saying so.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys
import warnings

import pytest

_ROOT = pathlib.Path(__file__).parents[2]
_STAR_IMPORTS = "from typing import Annotated\nfrom conductor import *\nfrom conductor.widgets import *\n"


def _documents() -> dict[str, str]:
    """Each document's Python as one module, by a module name made from its path."""
    found: dict[str, str] = {}
    markdown = [*_ROOT.glob("*.md"), *_ROOT.glob("docs/*.md"), *_ROOT.glob("skills/*/*.md"), *_ROOT.glob("packages/*/README.md")]
    for path in sorted(markdown):
        blocks = [_doctest_free(block) for block in re.findall(r"```python\n(.*?)```", path.read_text(), re.S)]
        if blocks:
            found[_module_name(path)] = "\n".join(blocks)
    llms = _ROOT / "packages" / "conductor" / "src" / "conductor" / "about" / "llms.txt"
    found[_module_name(llms)] = _STAR_IMPORTS + "\n".join(_doctest_free(b) for b in re.findall(r"```python\n(.*?)```", llms.read_text(), re.S))
    for path in sorted(_ROOT.glob("examples/*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
        found[_module_name(path)] = "\n".join(_awaitable(source, n) for n, source in enumerate(code))
    return found


def _awaitable(cell: str, number: int) -> str:
    """A notebook cell that awaits at the top, as a notebook allows, as the body of a coroutine of its own."""
    if not re.search(r"\bawait\b|\basync (for|with)\b", cell):
        return cell
    return f"async def _cell_{number}() -> None:\n" + "\n".join(f"    {line}" if line else "" for line in cell.splitlines())


def _doctest_free(block: str) -> str:
    """A block written as a session (``>>> ...``) as the code it runs."""
    if not block.lstrip().startswith(">>>"):
        return block
    return "\n".join(line[4:] for line in block.splitlines() if line.startswith((">>> ", "... ")))


def _module_name(path: pathlib.Path) -> str:
    return "doc_" + re.sub(r"\W", "_", str(path.relative_to(_ROOT)))


@pytest.fixture(scope="module")
def modules(tmp_path_factory: pytest.TempPathFactory) -> list[pathlib.Path]:
    directory = tmp_path_factory.mktemp("doc_examples")
    written = []
    for name, source in _documents().items():
        path = directory / f"{name}.py"
        path.write_text(source)
        written.append(path)
    return written


def test_every_document_with_python_is_read() -> None:
    names = set(_documents())
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
        "extraPaths": [str(_ROOT / "packages" / "conductor" / "src")],
        "typeCheckingMode": "standard",
        "venvPath": str(pathlib.Path(sys.prefix).parent),
        "venv": pathlib.Path(sys.prefix).name,
    }))
    done = subprocess.run([command, "--outputjson", *map(str, modules)], capture_output=True, text=True, cwd=directory)
    report = json.loads(done.stdout)
    errors = [f"{pathlib.Path(d['file']).name}:{d['range']['start']['line'] + 1}: {d['message']}" for d in report["generalDiagnostics"] if d["severity"] == "error"]
    assert not errors, "\n".join(errors)
