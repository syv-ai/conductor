"""The README's program and the other first-hour programs pass pyright and mypy strict, and run.

The first hour includes the custom widget ``docs/writing-nodes.md`` shows, and a ``Table[R]`` whose row type a
checker keeps.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any

import pytest

_README = pathlib.Path(__file__).parents[2] / "README.md"
_WRITING_NODES = pathlib.Path(__file__).parents[2] / "docs" / "writing-nodes.md"

_A_FRAME_IN_AND_OUT = """
from typing import TypedDict

import polars as pl
from conductor import NodeRegistry, Table, make_graph

registry = NodeRegistry()


class Sale(TypedDict):
    region: str
    amount: int


@registry.node("sales", title="Sales", description="")
def sales() -> Table[Sale]:
    return pl.DataFrame({"region": ["north", "south"], "amount": [3, 9]})


@registry.node("first-row", title="First row", description="")
def first_row(table: Table[Sale]) -> Table[Sale]:
    return table.head(1)


@registry.node("rows", title="Rows", description="")
def rows(table: Table) -> int:
    return table.height


graph = registry.load(make_graph(sales, first_row, rows))
results = graph.run().results
assert results["first-row"]["result"]["region"].to_list() == ["north"]
assert results["rows"]["result"] == 1
"""

_AN_EMBEDDED_GRAPH = """
from conductor import NodeRegistry, make_graph, node


@node("trim", title="Trim", description="")
def trim(text: str) -> str:
    return text.strip()


@node("upper", title="Upper", description="")
def upper(text: str) -> str:
    return text.upper()


base = NodeRegistry()
base.add(trim, upper)
clean = node("clean", title="Clean", description="")(make_graph(trim, upper))
base.add(clean)
outer = make_graph(clean, upper, edges=[("clean.result", "upper.text")], values={"clean.text": "  hi  "})
assert base.load(outer).run().outputs == {"upper.result": "HI"}
"""


_A_TABLE_KEEPS_ITS_ROW_TYPE = """
from typing import TypedDict, assert_type

import polars as pl
from conductor import Table, node


class Sale(TypedDict):
    region: str
    amount: int


@node("sales", title="Sales", description="")
def sales() -> Table[Sale]:
    return pl.DataFrame({"region": ["north", "south"], "amount": [3, 9]})


@node("first-row", title="First row", description="")
def first_row(table: Table[Sale]) -> Table[Sale]:
    assert_type(table, Table[Sale])
    return table.head(1)


assert first_row(sales()).height == 1
"""

#: Two row types a checker tells apart: pyright refuses the ``assert_type``, mypy shows ``Sale`` on ``reveal_type``.
_TWO_ROW_TYPES = """
from typing import TypedDict, assert_type, reveal_type

from conductor import Table


class Sale(TypedDict):
    amount: int


class Order(TypedDict):
    id: int


def first(sales: Table[Sale]) -> None:
    reveal_type(sales)
    assert_type(sales, Table[Order])
"""


def _programs() -> dict[str, str]:
    return {
        "readme.py": "".join(re.findall(r"```python\n(.*?)```", _README.read_text(), re.S)),
        "a_frame_in_and_out.py": _A_FRAME_IN_AND_OUT,
        "an_embedded_graph.py": _AN_EMBEDDED_GRAPH,
        "a_table_keeps_its_row_type.py": _A_TABLE_KEEPS_ITS_ROW_TYPE,
        "docs_custom_widget.py": next(block for block in re.findall(r"```python\n(.*?)```", _WRITING_NODES.read_text(), re.S) if "(Widget):" in block),
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


#: The README still reads ``Waiting.row``; its program runs and type-checks again once the docs follow the engine.
_README_IS_STALE = pytest.mark.xfail(reason="docs: README.md reads Waiting.row, which is Waiting.item; the docs pass rewrites it", strict=True)

_HELD_BACK = {
    "readme.py": _README_IS_STALE,
    "an_embedded_graph.py": pytest.mark.skip(reason="embedding: nested run not built"),
}


@pytest.mark.parametrize("program", [pytest.param(name, marks=_HELD_BACK.get(name, ())) for name in _programs()])
def test_a_first_hour_program_runs(program: str) -> None:
    namespace: dict[str, object] = {"__name__": "not_main"}
    exec(compile(_programs()[program], program, "exec"), namespace)


@_README_IS_STALE
def test_the_first_hour_programs_pass_mypy_strict(programs: list[pathlib.Path], tmp_path: pathlib.Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--cache-dir", str(tmp_path), *map(str, programs)],
        capture_output=True, text=True, cwd=programs[0].parent,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def _pyright_strict(programs: list[pathlib.Path]) -> dict[str, Any]:
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
    report: dict[str, Any] = json.loads(done.stdout)
    return report


@_README_IS_STALE
def test_the_first_hour_programs_pass_pyright_strict(programs: list[pathlib.Path]) -> None:
    report = _pyright_strict(programs)
    assert report["summary"]["errorCount"] == 0, json.dumps([d for d in report["generalDiagnostics"] if d["severity"] == "error"], indent=1)


def test_a_checker_keeps_a_tables_row_type(tmp_path: pathlib.Path) -> None:
    program = tmp_path / "two_row_types.py"
    program.write_text(_TWO_ROW_TYPES)
    found = _pyright_strict([program])["generalDiagnostics"]
    assert [d.get("rule") for d in found if d["severity"] == "error"] == ["reportAssertTypeFailure"], found
    assert any("_RowsOf[Sale]" in d["message"] for d in found if d["severity"] == "information"), found
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--cache-dir", str(tmp_path / "cache"), str(program)], capture_output=True, text=True, cwd=tmp_path,
    )
    assert "_RowsOf[TypedDict(two_row_types.Sale" in done.stdout, done.stdout


_WRONG_KEYWORDS = """
from conductor import Ask, Param, node
from conductor.widgets import Textarea

times = Param(gee=1)
text = Param(widget=Textarea)
asked = Ask("?", choices="ab")
called = node("inner", title="Inner", description="A stored graph.")({"nodes": []})("a")
"""


def test_a_checker_refuses_an_unknown_limit_a_widget_class_passed_bare_choices_spelt_as_text_and_a_graph_called(tmp_path: pathlib.Path) -> None:
    """mypy reads a ``Param`` written outside ``Annotated``, an ``Ask``'s keywords, and a call of a node that is a stored
    graph, which has no function: what conductor doesn't check when they run."""
    program = tmp_path / "wrong_keywords.py"
    program.write_text(_WRONG_KEYWORDS)
    done = subprocess.run([sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "cache"), str(program)], capture_output=True, text=True, cwd=tmp_path)
    assert [line.split(": error: ")[0].rsplit(":", 1)[1] for line in done.stdout.splitlines() if ": error: " in line] == ["5", "6", "7", "8"], done.stdout
