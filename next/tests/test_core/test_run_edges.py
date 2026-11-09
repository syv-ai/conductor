"""What a run does at its edges: a node's own frame, values JSON can't spell, a damaged state, a person's odd answer."""

import datetime
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    ALWAYS,
    NEVER,
    ROW_ID_COLUMN,
    SKIPPED,
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Output,
    Param,
    Ref,
    Result,
    RunState,
    Series,
    Static,
    Table,
    node,
)
from conductor.widgets import Textarea
from helpers import read_table


class Txt(DType, str):
    id = "edges-txt"
    display_title = "Text"


class Num(DType, float):
    id = "edges-num"
    display_title = "Number"


class Contact(TypedDict):
    name: Txt
    email: Txt


Contacts = Table[Contact]
In = Annotated[Txt, Param(title="In", widget=Textarea())]
Wire = Annotated[Txt, Param(title="In")]
Out = Annotated[Txt, Result(title="result")]
Parts = Annotated[Series[Txt], Result(title="Parts")]
TableOut = Annotated[Contacts, Result(title="Table")]
CALLS: list[Any] = []
CANNED: dict[str, Any] = {}


def n(id: str, type: str | None = None, **bindings: Any) -> GraphNode:
    return GraphNode(id=id, type=type or id, version=1, bindings=bindings)


@node("source", title="s", description="d", category="t")
def source(text: In = Txt("")) -> Out:
    return Txt(text)


@node("split", title="s", description="d", category="t")
def split(text: Wire) -> Parts:
    return [Txt(part) for part in text.split(",") if part]


@node("upper", title="s", description="d", category="t")
def upper(text: Wire) -> Out:
    CALLS.append(("upper", str(text)))
    return Txt(text.upper())


@node("join", title="s", description="d", category="t")
def join(texts: Annotated[Series[Txt], Param(title="T")]) -> Out:
    CALLS.append(("join", list(texts)))
    return Txt(",".join(texts))


@node("two", title="s", description="d", category="t")
def two(a: Wire, b: Wire) -> Out:
    CALLS.append(("two", str(a), str(b)))
    return Txt(f"{a}+{b}")


@node("lister", title="s", description="d", category="t")
def lister(text: Wire, tag: Wire) -> Out:
    return Txt(f"{text}:{tag}")


@dataclass(frozen=True)
class Branch:
    yes: Annotated[Txt, Result(title="Yes", exclusive_group="c")]
    no: Annotated[Txt, Result(title="No", exclusive_group="c")]


@node("route", title="s", description="d", category="t")
def route(text: Wire) -> Branch:
    if text == "slow-no":
        time.sleep(0.05)
    return Branch(yes=SKIPPED, no=text) if text.endswith("no") else Branch(yes=text, no=SKIPPED)


@node("num", title="s", description="d", category="t")
def num(k: In = Txt("")) -> Annotated[Num, Result(title="N")]:
    return CANNED[k]


@node("num_in", title="s", description="d", category="t")
def num_in(v: Annotated[Num, Param(title="N")]) -> Annotated[Num, Result(title="N")]:
    CALLS.append(("num_in", repr(v)))
    return v


@node("frame", title="s", description="d", category="t")
def frame_node(k: In = Txt("")) -> Annotated[Table, Result(title="T")]:
    return CANNED[k]


@node("frame_in", title="s", description="d", category="t")
def frame_in(table: Annotated[Table, Param(title="T")]) -> Out:
    CALLS.append(("frame_in", table.schema, table.rows()))
    return Txt(str(table.height))


def _any_outputs() -> list[Output]:
    return [Output("result", Series[Any], title="A")]


@node("anything", title="s", description="d", category="t", compute_outputs=_any_outputs)
def anything(k: In = Txt("")) -> Mapping[str, Any]:
    return {"result": [CANNED[k]]}


@node("pass_table", title="s", description="d", category="t")
def pass_table(x: Wire, t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    return t


@node("merged", title="s", description="d", category="t")
def merged(table: Annotated[Table, Param(title="T")]) -> Annotated[Table, Result(title="T")]:
    return table


MODULE_FRAME = pl.DataFrame({"name": ["a", "b"], "email": ["x", "y"]})


@node("module_frame", title="s", description="d", category="t")
def module_frame(x: Wire) -> TableOut:
    return MODULE_FRAME  # type: ignore[return-value]


@node("csv", title="s", description="d", category="t")
def csv(k: In = Txt("")) -> TableOut:
    names = [x for x in k.split(",") if x]
    return read_table(Contacts, pl.DataFrame({"name": names, "email": [f"{x}@m" for x in names]}, schema={"name": pl.String, "email": pl.String}))


@node("rebuild", title="s", description="d", category="t")
def rebuild(t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    names = t["name"].to_list() or ["placeholder"]
    return read_table(Contacts, [{"name": x, "email": "e"} for x in names])


@node("none_left", title="s", description="d", category="t")
def none_left(t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    return read_table(Contacts, [r for r in t.iter_rows(named=True) if r["name"] == "nobody"])


@node("ask", title="s", description="d", category="t")
def ask(text: Wire) -> Out | AskPerson:
    return AskPerson(prompt=text)


@node("ask_table", title="s", description="d", category="t")
def ask_table(t: Annotated[Contacts, Param(title="T")]) -> TableOut | AskPerson:
    return AskPerson(defaults={"result": t})


@node("ask_rows", title="s", description="d", category="t")
def ask_rows(k: In = Txt("")) -> TableOut | AskPerson:
    return AskPerson()


@node("ask_choice_default", title="s", description="d", category="t")
def ask_choice_default(k: In = Txt("")) -> Branch | AskPerson:
    return AskPerson(defaults={"yes": Txt("x")})


@node("ask_choice", title="s", description="d", category="t")
def ask_choice(k: In = Txt("")) -> Branch | AskPerson:
    return AskPerson()


@node("ask_default_skipped", title="s", description="d", category="t")
def ask_default_skipped(k: In = Txt("")) -> Parts | AskPerson:
    return AskPerson(defaults={"result": [Txt("a"), SKIPPED]})


@node("ask_bad_widget", title="s", description="d", category="t")
def ask_bad_widget(k: In = Txt("")) -> Out | AskPerson:
    return AskPerson(widgets={"result": "not a widget"})  # type: ignore[dict-item]


@node("ask_slow_first", title="s", description="d", category="t")
def ask_slow_first(text: Wire) -> Out | AskPerson:
    time.sleep(0.05 if text == "a" else 0)
    return AskPerson(prompt=text)


@node("reference", title="s", description="d", category="t")
def reference(k: In = Txt("")) -> TableOut:
    return read_table(Contacts, [{"name": "a", "email": "x"}, {"name": "b", "email": "y"}])


@node("lookup", title="s", description="d", category="t")
def lookup(x: Wire, t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    return t.filter(pl.col("name") == "a")  # type: ignore[return-value]


REGISTRY = NodeRegistry([
    source, split, upper, join, two, lister, route, num, num_in, frame_node, frame_in, anything, pass_table, merged, module_frame,
    csv, rebuild, none_left, reference, lookup, ask, ask_table, ask_rows, ask_choice_default, ask_choice, ask_default_skipped, ask_bad_widget, ask_slow_first,
])


def compiled_of(*nodes: GraphNode, registry: NodeRegistry = REGISTRY) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=nodes), registry)


def through_json(state: RunState) -> RunState:
    return RunState.model_validate_json(state.model_dump_json())


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    CALLS.clear()


# -- the frame a node returned ------------------------------------------------------


# -- values JSON has no spelling for -------------------------------------------------


# -- a table's JSON form --------------------------------------------------------------


UTC = datetime.timezone.utc


def test_a_json_form_cannot_name_anything_but_a_column_type():
    with pytest.raises(ValueError, match="does not line up"):
        read_table(Table, {"columns": [{"name": "a", "dtype": {"type": "read_csv", "source": "/etc/hosts"}}], "data": {"a": []}, "rows": None})


# -- zero rows ------------------------------------------------------------------------


def test_a_typed_table_of_no_records_is_a_table_with_its_columns_and_no_rows():
    assert read_table(Contacts, []).columns == [ROW_ID_COLUMN, "name", "email"] and read_table(Contacts, []).height == 0


# -- row ids --------------------------------------------------------------------------


# -- lists set on a node over no rows --------------------------------------------------


# -- reading a state -------------------------------------------------------------------


def _waiting_run() -> tuple[CompiledGraph, RunState]:
    compiled = compiled_of(n("s", "source", text=Static("a,b")), n("sp", "split", text=From("s.result")), n("a", "ask", text=From("sp.result")), n("j", "join", texts=From("a.result")))
    return compiled, compiled.run_sync()


def _damage(change: Any) -> tuple[CompiledGraph, RunState]:
    compiled, state = _waiting_run()
    dumped = json.loads(state.model_dump_json())
    change(dumped)
    return compiled, RunState.model_validate(dumped)


def _drop(d: dict[str, Any], key: str) -> None:
    del d[key]


# -- what can be written and asked -----------------------------------------------------


def test_a_node_id_with_the_separator_of_a_unit_id_is_refused():
    assert "invalid_node_id" in {p.type for p in compiled_of(n("a@0", "source", text=Static("x"))).problems}


# -- skips, blanks, without ------------------------------------------------------------


def test_an_alternative_taking_both_ways_of_one_decision_can_never_hold():
    compiled = compiled_of(
        n("g", "route", text=Static("x")), n("a", "upper", text=From("g.yes")), n("b", "upper", text=From("g.no")), n("m", "two", a=From("a.result"), b=From("b.result")),
    )
    assert compiled.field(Ref("a.result")).condition != ALWAYS
    assert compiled.field(Ref("m.result")).condition == NEVER


# -- rows that repeat an id under different parents -------------------------------------


# -- an answer that keeps a table's ids; results straight after an answer -----------------


@node("ask_list", title="s", description="d", category="t")
def ask_list(k: In = Txt("")) -> Parts | AskPerson:
    return AskPerson()


