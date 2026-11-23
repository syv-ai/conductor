"""The programs a new user writes in their first hour run, and pass mypy and pyright in strict mode.

These are the README's program, the custom widget from ``docs/writing-nodes.md``, and a few small programs below.
A new user's editor runs a type checker, so a red squiggle on the first example would be a bad start. One of the
programs checks that a ``Table[R]`` keeps its row type ``R`` for the checker. Another, a host's file, calls every
public signature whose values depend on the graph, and passes mypy's ``--disallow-any-expr`` too: conductor types
those values ``object``, so no ``Any`` reaches a host's code.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pytest import TempPathFactory

_README = Path(__file__).parents[2] / "README.md"
_WRITING_NODES = Path(__file__).parents[2] / "docs" / "writing-nodes.md"

_A_FRAME_IN_AND_OUT = """
from typing import TypedDict

import polars as pl
from conductor import Graph, NodeRegistry, Table, make_graph

registry = NodeRegistry()


class Sale(TypedDict):
    region: str
    amount: int


@registry.node("sales", title="Sales", description="For a test.")
def sales() -> Table[Sale]:
    return pl.DataFrame({"region": ["north", "south"], "amount": [3, 9]})


@registry.node("first-row", title="First row", description="For a test.")
def first_row(table: Table[Sale]) -> Table[Sale]:
    return table.head(1)


@registry.node("rows", title="Rows", description="For a test.")
def rows(table: Table) -> int:
    return table.height


graph = Graph.compile(make_graph(sales, first_row, rows), registry)
results = graph.run().results
first = results["first-row"]["result"]
assert isinstance(first, pl.DataFrame)
assert first["region"].to_list() == ["north"]
assert results["rows"]["result"] == 1
"""

_A_SUBGRAPH = """
from conductor import Graph, NodeRegistry, make_graph, node, subgraph


@node("trim", title="Trim", description="For a test.")
def trim(text: str) -> str:
    return text.strip()


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


base = NodeRegistry()
base.add(trim, upper)
clean = subgraph("clean", title="Clean", description="For a test.")(make_graph(trim, upper))
base.add(clean)
outer = make_graph(clean, upper, edges=[("clean.result", "upper.text")], constants={"clean.text": "  hi  "})
assert Graph.compile(outer, base).run().outputs == {"result": "HI"}
"""


#: A host's file that calls every public signature whose values depend on the graph: what it gives in, and what it
#: reads back, which it narrows with ``isinstance`` before using a value as its type.
_A_HOST = """
import asyncio
from typing import Annotated, TypedDict

from conductor import (
    Ask,
    Choice,
    Graph,
    NodeRegistry,
    Param,
    Run,
    Series,
    Upgrade,
    make_graph,
    node,
    read_json,
    type_fields,
)
from conductor.events import RunEnd
from conductor.testing import check_registry
from conductor_providers.fastapi import Worker

registry = NodeRegistry()
answers = [True, False]


class Invoice(TypedDict):
    name: str
    amount: int


@registry.node("read", title="Read", description="Reads invoices from a CSV text.")
def read(csv: str) -> Series[Invoice]:
    found = [line.split(",") for line in csv.splitlines()[1:]]
    return [Invoice(name=name, amount=int(amount)) for name, amount in found]


@registry.node("approve", title="Approve", description="Approves an invoice, or asks a person to.", asks=True)
def approve(name: str, amount: int) -> bool:
    if amount < 1000:
        return True
    raise Ask(f"Approve {name}, {amount} kr.?", choices=answers, default=False)


# A form asks per output, each with choices of its own type, and choices may be a tuple
form = Ask("Which tier, and how many?", choices={"tier": ["gold", "silver"], "count": [1, 2]})
pair = Ask("Approve?", choices=(True, False))


@registry.node("summary", title="Summary", description="Tells how many invoices were approved.")
def summary(approved: Series[bool]) -> str:
    return f"{sum(approved)} of {len(approved)} approved"


class Contact(TypedDict):
    name: str


def fields_of(value: type) -> dict[str, object]:
    return type_fields(value)


@node("fields", title="Fields", description="Gives each field of a value.", compute_outputs=fields_of)
def fields(value: Contact) -> dict[str, str]:
    return {"name": value["name"]}


@node("repeat", title="Repeat", description="Repeats a text.")
def repeat_v1(text: str, times: int = 1) -> str:
    return text * times


counts = [1, 2, 3]


@node(
    "repeat",
    title="Repeat",
    description="Repeats a text.",
    replaces=repeat_v1,
    upgrade=Upgrade(rewrite=lambda constants: {"times": constants.get("times", 1)}),
)
def repeat(text: str, times: Annotated[int, Param(choices=counts)] = 1) -> str:
    return text * times


record = make_graph(
    read,
    approve,
    summary,
    edges=[
        ("read.result.name", "approve.name"),
        ("read.result.amount", "approve.amount"),
        ("approve.result", "summary.approved"),
    ],
    defaults={"read.csv": "name,amount\\n"},
)
make_graph(repeat, constants={"repeat.text": "ho", "repeat.times": 2})
graph = Graph.compile(record.model_dump_json(), registry)
assert isinstance(read_json(record.model_dump_json()), dict)

csv: dict[str, str] = {"csv": "name,amount\\nIda,100\\nBo,5000\\n"}
run = graph.run(csv)
given = run.inputs["csv"]
assert isinstance(given, str)
for waiting in run.waiting:
    shown = waiting.inputs["name"]
    assert isinstance(shown, str)

queued = graph.answers_as_json({run.waiting[0].id: True})
Run.declines_as_json({run.waiting[0].id: "Not today"})
run.with_answers(queued).resume()
run.with_decline(run.waiting[0], reason="No").resume()
answered = Run.from_json(run.with_answers(run.waiting[0], True).resume().to_json(), graph)
for asked in answered.asked:
    if asked.answer is not None:
        assert asked.answer["result"] is True

summed = answered.outputs["result"]
assert isinstance(summed, str)
assert summed.startswith("2 of 2")
assert answered.results["approve"]["result"] == [True, True]
answered.again({"csv": "name,amount\\n"})
answered.with_inputs({"csv": "name,amount\\n"}, max_items=10)

for port in registry.describe().nodes[0].inputs:
    assert port.json_schema and port.widget is None
for definition in registry:
    assert registry[definition.id] is definition
assert Param(ge=0).limits
assert not check_registry(registry, examples={"read": [csv]})
assert Choice(True, "Yes", colour="green").id is True


async def streamed() -> None:
    async for event in graph.with_inputs(csv).stream():
        if isinstance(event, RunEnd):
            assert event.status == "waiting"


asyncio.run(streamed())


def everyone(metadata: dict[str, str]) -> bool:
    return True


async def served(worker: Worker[dict[str, str], None]) -> None:
    view = await worker.create(record.model_dump_json(), csv, {"by": "ida"}, allows=everyone)
    await worker.estimate(record, csv, {"by": "ida"}, allows=everyone)
    view = await worker.again(view.id, csv, {"by": "ida"}, allows=everyone)
    for waiting in view.waiting:
        view = await worker.answer(view.id, {waiting.id: True}, allows=everyone)
    if view.outputs is not None:
        print(view.outputs["result"], view.metadata["by"])
"""


_A_TABLE_KEEPS_ITS_ROW_TYPE = """
from typing import TypedDict, assert_type

import polars as pl
from conductor import Table, node


class Sale(TypedDict):
    region: str
    amount: int


@node("sales", title="Sales", description="For a test.")
def sales() -> Table[Sale]:
    return pl.DataFrame({"region": ["north", "south"], "amount": [3, 9]})


@node("first-row", title="First row", description="For a test.")
def first_row(table: Table[Sale]) -> Table[Sale]:
    assert_type(table, Table[Sale])
    return table.head(1)


assert first_row(sales()).height == 1
"""

#: A table of one row type, asserted to be a table of another. pyright refuses the ``assert_type``, and mypy shows
#: ``Sale`` for ``reveal_type``.
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
        "a_subgraph.py": _A_SUBGRAPH,
        "a_table_keeps_its_row_type.py": _A_TABLE_KEEPS_ITS_ROW_TYPE,
        "a_host.py": _A_HOST,
        "docs_custom_widget.py": next(
            block
            for block in re.findall(r"```python\n(.*?)```", _WRITING_NODES.read_text(), re.S)
            if "(Widget):" in block
        ),
    }


@pytest.fixture(scope="module")
def programs(tmp_path_factory: TempPathFactory) -> list[Path]:
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


def test_the_first_hour_programs_pass_mypy_strict(programs: list[Path], tmp_path: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--cache-dir",
            str(tmp_path),
            *map(str, programs),
        ],
        capture_output=True,
        text=True,
        cwd=programs[0].parent,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_host_meets_no_any(programs: list[Path], tmp_path: Path) -> None:
    host = next(path for path in programs if path.name == "a_host.py")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--disallow-any-expr",
            "--cache-dir",
            str(tmp_path),
            str(host),
        ],
        capture_output=True,
        text=True,
        cwd=host.parent,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def _pyright_strict(programs: list[Path]) -> dict[str, Any]:
    pyright = Path(sys.executable).with_name("pyright")
    if not pyright.exists() and shutil.which("pyright") is None:
        pytest.skip("pyright is not installed")
    directory = programs[0].parent
    (directory / "pyrightconfig.json").write_text(
        json.dumps(
            {
                "extraPaths": [
                    str(Path(__file__).parents[2] / "packages" / package / "src")
                    for package in ("conductor",)
                ],
                "typeCheckingMode": "strict",
                "venvPath": str(Path(sys.prefix).parent),
                "venv": Path(sys.prefix).name,
            }
        )
    )
    done = subprocess.run(
        [str(pyright if pyright.exists() else "pyright"), "--outputjson", *map(str, programs)],
        capture_output=True,
        text=True,
        cwd=directory,
    )
    report: dict[str, Any] = json.loads(done.stdout)
    return report


def test_the_first_hour_programs_pass_pyright_strict(programs: list[Path]) -> None:
    report = _pyright_strict(programs)
    assert report["summary"]["errorCount"] == 0, json.dumps(
        [d for d in report["generalDiagnostics"] if d["severity"] == "error"], indent=1
    )


def test_a_checker_keeps_a_tables_row_type(tmp_path: Path) -> None:
    program = tmp_path / "two_row_types.py"
    program.write_text(_TWO_ROW_TYPES)
    found = _pyright_strict([program])["generalDiagnostics"]
    assert [d.get("rule") for d in found if d["severity"] == "error"] == [
        "reportAssertTypeFailure"
    ], found
    assert any("_RowsOf[Sale]" in d["message"] for d in found if d["severity"] == "information"), (
        found
    )
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--cache-dir",
            str(tmp_path / "cache"),
            str(program),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "_RowsOf[TypedDict(two_row_types.Sale" in done.stdout, done.stdout


_WRONG_KEYWORDS = """
from conductor import GraphRecord, Param, subgraph
from conductor.widgets import Textarea

times = Param(gee=1)
text = Param(widget=Textarea)
called = subgraph("inner", title="Inner", description="A subgraph.")(GraphRecord())("a")
"""


def test_a_checker_refuses_an_unknown_limit_a_widget_class_passed_bare_and_a_graph_called(
    tmp_path: Path,
) -> None:
    """mypy catches three mistakes that conductor doesn't check when the code runs.

    They are an unknown keyword on ``Param``, a widget class passed where an instance goes, and a call of a
    subgraph, which has no function to call.
    """
    program = tmp_path / "wrong_keywords.py"
    program.write_text(_WRONG_KEYWORDS)
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "cache"), str(program)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert [
        line.split(": error: ")[0].rsplit(":", 1)[1]
        for line in done.stdout.splitlines()
        if ": error: " in line
    ] == ["5", "6", "7"], done.stdout
