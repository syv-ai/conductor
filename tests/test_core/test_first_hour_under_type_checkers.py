"""The README's programs and the other first-hour programs pass pyright and mypy strict, and run."""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

_README = pathlib.Path(__file__).parents[2] / "README.md"

_A_FRAME_IN_AND_OUT = """
from typing import TypedDict

import polars as pl
from conductor import CompiledGraph, NodeRegistry, Table, make_graph

registry = NodeRegistry()


class Sale(TypedDict):
    region: str
    amount: int


@registry.node("sales")
def sales() -> Table[Sale]:
    return pl.DataFrame({"region": ["north", "south"], "amount": [3, 9]})


@registry.node("first-row")
def first_row(table: Table[Sale]) -> Table[Sale]:
    return table.head(1)


@registry.node("rows")
def rows(table: Table) -> int:
    return table.height


graph = make_graph(sales, first_row, rows)
compiled = CompiledGraph.from_graph(graph, registry)
results = compiled.results(compiled.run_sync())
assert results["first-row"]["result"]["region"].to_list() == ["north"]
assert results["rows"]["result"] == 1
"""

_AN_EMBEDDED_GRAPH = """
from conductor import CompiledGraph, NodeDefinition, NodeRegistry, make_graph, node


@node("trim")
def trim(text: str) -> str:
    return text.strip()


@node("upper")
def upper(text: str) -> str:
    return text.upper()


base = NodeRegistry()
base.add(trim, upper)
inner = CompiledGraph.from_graph(make_graph(trim, upper), base)
clean = NodeDefinition.from_compiled_graphs("clean", versions=inner)
outer = make_graph(clean, upper, values={"clean.text": "  hi  "})
base.add(clean)
compiled = CompiledGraph.from_graph(outer, base)
assert compiled.outputs(compiled.run_sync()) == {"upper.result": "HI"}
"""


def _programs() -> dict[str, str]:
    blocks = re.findall(r"```python\n(.*?)```", _README.read_text(), re.S)
    first = blocks.index(next(block for block in blocks if '@registry.node("exclaim")' in block))
    return {
        "readme_first_program.py": blocks[first] + blocks[first + 1],
        "readme_csv.py": blocks[first + 2],
        "a_frame_in_and_out.py": _A_FRAME_IN_AND_OUT,
        "an_embedded_graph.py": _AN_EMBEDDED_GRAPH,
    }


@pytest.fixture(scope="module")
def programs(tmp_path_factory: pytest.TempPathFactory) -> list[pathlib.Path]:
    directory = tmp_path_factory.mktemp("first_hour")
    written = []
    for name, source in _programs().items():
        path = directory / name
        path.write_text(source)
        written.append(path)
    return written


@pytest.mark.parametrize("program", list(_programs()))
def test_a_first_hour_program_runs(program: str) -> None:
    namespace: dict[str, object] = {"__name__": "not_main"}
    exec(compile(_programs()[program], program, "exec"), namespace)


def test_the_first_hour_programs_pass_mypy_strict(programs: list[pathlib.Path], tmp_path: pathlib.Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--cache-dir", str(tmp_path), *map(str, programs)],
        capture_output=True, text=True, cwd=programs[0].parent,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_first_hour_programs_pass_pyright_strict(programs: list[pathlib.Path]) -> None:
    pyright = pathlib.Path(sys.executable).with_name("pyright")
    if not pyright.exists() and shutil.which("pyright") is None:
        pytest.skip("pyright is not installed")
    directory = programs[0].parent
    (directory / "pyrightconfig.json").write_text(json.dumps({
        "extraPaths": [str(pathlib.Path(__file__).parents[2] / "packages" / package / "src") for package in ("conductor",)],
        "typeCheckingMode": "strict",
        "venvPath": str(pathlib.Path(sys.prefix).parent),
        "venv": pathlib.Path(sys.prefix).name,
    }))
    done = subprocess.run([str(pyright if pyright.exists() else "pyright"), "--outputjson", *map(str, programs)], capture_output=True, text=True, cwd=directory)
    report = json.loads(done.stdout)
    assert report["summary"]["errorCount"] == 0, json.dumps([d for d in report["generalDiagnostics"] if d["severity"] == "error"], indent=1)
