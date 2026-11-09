"""A leg ends with a typed ``RunState``, and a node the graph changed runs again.

The state is what a host stores between legs and hands back: every value, what is done, and a fingerprint per node of
how the graph uses it. It survives JSON, and the next leg's results are
typed like the first's. On restore, a node whose fingerprint differs from
the graph it is restored into — a static edited, a version bumped, a
binding moved — is dropped with everything downstream of it, so those
units run again; a node the graph no longer has is dropped the same way,
and ``without_nodes`` drops one on purpose (a host's "run from here").
"""

import json
from typing import Annotated

import pytest
from conductor import (
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Param,
    Result,
    RunState,
    Series,
    Static,
    Upgrade,
    node,
)
from conductor.widgets import Textarea
from pydantic import ValidationError


class Txt(DType, str):
    id = "state-test-text"
    display_title = "Text"


In = Annotated[Txt, Param(title="In", widget=Textarea())]
Out = Annotated[Txt, Result(title="Out")]

calls: list[str] = []


@node("split", title="Split", description="d", category="test")
def split_node(text: In = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
    calls.append("split")
    return [Txt(part) for part in text.split(",")]


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: In = Txt("")) -> Out:
    calls.append(f"upper:{text}")
    return Txt(text.upper())


def wrap_v1(text: In = Txt("")) -> Out:
    calls.append(f"wrap:{text}")
    return Txt(f"[{text}]")

@node("wrap", title="Wrap", description="d", category="test", replaces=node("wrap")(wrap_v1), upgrade=Upgrade())
def wrap(text: In = Txt("")) -> Out:
    calls.append(f"wrap2:{text}")
    return Txt(f"<{text}>")


@node("echo", title="Echo", description="d", category="test")
def echo(text: In = Txt("")) -> Out:
    calls.append(f"echo:{text}")
    return text


@node("ask", title="Ask", description="d", category="test")
def ask(text: In = Txt("")) -> Out | AskPerson:
    calls.append(f"ask:{text}")
    return AskPerson(defaults={"result": text}, widgets={"result": Textarea()})


def _registry() -> NodeRegistry:
    reg = NodeRegistry()
    for cls in (split_node, upper_node, wrap, echo, ask):
        reg = reg.extended_with([cls])
    return reg


def _compiled(nodes: list[GraphNode]) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled


def _edge(graph_node: str, field: str = "result") -> From:
    return From(f"{graph_node}.{field}")


def _through_json(state: RunState) -> RunState:
    return RunState.model_validate(json.loads(state.model_dump_json()))


# -- the state survives JSON ---------------------------------------------------------


# -- a changed node runs again -------------------------------------------------------


# -- what a node returns and what a person answers are checked ------------------------


@pytest.mark.parametrize("column", [
    {"rows": [[]], "values": []},
    {"rows": None, "values": []},
    {"rows": [[]], "values": ["A"], "stray": 1},
    {"rows": "x", "values": ["A"]},
])
def test_a_column_that_does_not_line_up_is_refused_where_it_arrives(column):
    with pytest.raises(ValidationError):
        RunState.model_validate({"status": "complete", "nodes": {"e": {"outputs": {"result": column}}}})


# -- edits between legs, and a state read after the ledger moved on ----------------------


def test_a_static_hashes_as_its_type_writes_it_not_as_the_author_spelled_it():
    class Num(DType, float):
        id = "state-test-number"
        display_title = "Number"

    @node("half", title="Half", description="d", category="test")
    def half(n: Annotated[Num, Param(title="N", widget=Textarea())] = Num(0)) -> Annotated[Num, Result(title="Out")]:
        return Num(n / 2)

    def fingerprint(value):
        reg = NodeRegistry([half])
        return CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="h", type="half", version=1, bindings={"n": Static(value)})]), reg).node("h").fingerprint

    assert fingerprint(2) == fingerprint(2.0) == fingerprint(Num(2))
    assert fingerprint(2) != fingerprint(3)


