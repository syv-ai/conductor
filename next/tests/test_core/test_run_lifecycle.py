"""A run is its ``RunState``: the scenarios a host writes, through the public API only."""

import pathlib
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeDefinition,
    NodeError,
    NodeRegistry,
    Param,
    Result,
    RunState,
    Series,
    Static,
    node,
    table_with_columns,
)
from conductor.widgets import Textarea
from helpers import read_table


class Txt(DType, str):
    id = "lifecycle-txt"
    display_title = "Tekst"


class Whole(DType, int):
    id = "lifecycle-int"
    display_title = "Tal"


Contacts = table_with_columns({"name": Txt, "email": Txt})
Out = Annotated[Txt, Result(title="Svar")]
calls: dict[str, int] = {}


def _called(name: str) -> None:
    calls[name] = calls.get(name, 0) + 1


@pytest.fixture(autouse=True)
def _no_calls():
    calls.clear()


def graph_node(cls_id, run_fn, **attrs):
    return type(cls_id.title().replace("-", ""), (NodeDefinition,), {"id": cls_id, "title": cls_id, "description": "d", "category": "test", "run": run_fn, **attrs})


@node("split", title="Split", description="d", category="test")
def split_node(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    _called("split")
    return [Txt(line) for line in text.split("\n")]


@node("words", title="Words", description="d", category="test")
def words(text: Annotated[Txt, Param(title="T")]) -> Annotated[Whole, Result(title="N")]:
    _called("words")
    return Whole(len(text.split()))


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Out:
    _called("upper")
    return Txt(text.upper())


@node("approve", title="Approve", description="d", category="test")
def approve(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Out | AskPerson:
    _called("approve")
    return AskPerson(prompt=f"ok? {text}", defaults={"result": text})


@dataclass(frozen=True)
class Verdict:
    yes: Annotated[Txt, Result(title="Ja", exclusive_group="verdict")]
    no: Annotated[Txt, Result(title="Nej", exclusive_group="verdict")]
    note: Annotated[Txt, Result(title="Note")]


@node("decide", title="Decide", description="d", category="test")
def decide() -> Verdict | AskPerson:
    _called("decide")
    return AskPerson()


@node("ask-table", title="Ask table", description="d", category="test")
def ask_table() -> Annotated[Contacts, Result(title="Rettelser")] | AskPerson:
    _called("ask-table")
    return AskPerson(defaults={"result": read_table(Contacts, [{"name": "a", "email": "a@x"}])})


@node("ask-many", title="Ask many", description="d", category="test")
def ask_many(text: Annotated[Txt, Param(title="T")]) -> Annotated[Series[Txt], Result(title="Dele")] | AskPerson:
    return AskPerson()


@node("read-emails", title="Read", description="d", category="test")
def read_emails(email: Annotated[Txt, Param(title="E")]) -> Out:
    return Txt(email.upper())


@node("boom", title="Boom", description="d", category="test")
def boom(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Out:
    raise NodeError("det gik galt")


@node("wrong-type", title="Wrong", description="d", category="test")
def wrong_type() -> Out:
    return 42  # type: ignore[return-value]


@node("ask-stray", title="Ask stray", description="d", category="test")
def ask_stray() -> Out | AskPerson:
    return AskPerson(defaults={"nope": Txt("x")})


@node("slow", title="Slow", description="d", category="test")
def slow() -> Out:
    import time

    time.sleep(0.4)
    return Txt("late")


REGISTRY = NodeRegistry()
for cls in (split_node, words, upper_node, approve, decide, ask_table, ask_many, read_emails, boom, wrong_type, ask_stray, slow):
    REGISTRY = REGISTRY.extended_with([cls])


def compiled_of(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), REGISTRY)
    assert compiled.is_runnable, compiled.problems
    return compiled


def n(id: str, type: str | None = None, **bindings: Any) -> GraphNode:
    return GraphNode(id=id, type=type or id, version=1, bindings=bindings)


def roundtrip(state: RunState) -> RunState:
    return RunState.model_validate_json(state.model_dump_json())


LINES = (n("split", text=Static("one two\nthree")), n("count", "words", text=From("split.result")))


# -- scenario 1: run to the end, read the outputs -------------------------------------


# -- a failed run is a status, reading it is loud --------------------------------------


# -- scenario 2: stream events -------------------------------------------------------


# -- scenario 3: a run stops for a person; another process goes on -----------------------


PER_ROW = (n("split", text=Static("a\nb\nc")), n("ask", "approve", text=From("split.result")), n("up", "upper", text=From("ask.result")))


# -- scenario 4: re-run with one input changed ------------------------------------------


CHAIN = (n("a", "upper", text=Static("x")), n("b", "upper", text=From("a.result")))


# -- restoring reads every value at the door ---------------------------------------------


# -- with_inputs --------------------------------------------------------------------------


def test_with_inputs_refuses_every_name_it_does_not_offer():
    compiled = compiled_of(n("a", "upper", text=Static("x")))
    with pytest.raises(InvalidCallError) as raised:
        compiled.with_inputs(nope=1, ghost=2)
    assert [(fault.type, fault.loc) for fault in raised.value.errors()] == [
        ("not_offered", ("inputs", "nope")), ("not_offered", ("inputs", "ghost")),
    ]


# -- staleness reaches a reader of a table column; a partly answered node reads so far ---


@node("extract", title="Extract", description="d", category="test")
def extract(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Annotated[Contacts, Result(title="R")]:
    return read_table(Contacts, [{"name": text, "email": "a@x"}])


REGISTRY = REGISTRY.extended_with([extract])


def test_the_state_module_imports_nothing_from_the_graph():
    """A state loads, and a form renders from it, with no graph or registry in sight."""
    import conductor._internal.execution.run_state as module

    assert "conductor._internal.graph" not in pathlib.Path(module.__file__).read_text()


