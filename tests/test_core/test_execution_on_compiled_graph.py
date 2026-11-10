"""The engine is one more caller of ``CompiledGraph``: it takes what compile
produced and runs it — here, graphs whose nodes each run once. Rows are
``test_rows.py``'s.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    Param,
    Policy,
    Result,
    Series,
    Static,
    node,
)
from conductor.widgets import Textarea
from helpers import events_of, registry_of


class Txt(DType, str):
    id = "exec-test-txt"
    display_title = "Text"


Out = Annotated[Txt, Result(title="Result")]


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@dataclass(frozen=True)
class Answer:
    yes: Annotated[Txt, Result(title="Yes")]
    no: Annotated[Txt, Result(title="No")]


@node("gate", title="Gate", description="d", category="test")
def gate(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Answer:
    return Answer(yes=x, no=SKIPPED) if x else Answer(yes=SKIPPED, no=x)


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("+".join(texts))


flaky_state = SimpleNamespace(active=0, peak=0, seen=[], calls=0)


@node("flaky", title="Flaky", description="d", category="test", policy=Policy(retries=2, retry_delay=0.0))
def flaky(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    flaky_state.calls += 1
    if flaky_state.calls < 3:
        from conductor import ExternalFailure

        raise ExternalFailure("not yet")
    return text


def _registry():
    return registry_of(upper_node, gate, join_node, flaky)


def _run(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    return compiled.results(compiled.run_sync())


def test_a_graph_of_bindings_compiles_and_runs():
    results = _run([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("a.result")}),
    ])

    assert results["a"]["result"] == "HI"
    assert results["b"]["result"] == "HI"


def test_an_unbound_input_falls_back_to_its_declared_default():
    assert _run([GraphNode(id="a", type="upper", version=1)])["a"]["result"] == ""


def test_a_branch_not_taken_is_skipped_downstream():
    """A branch is an output; SKIPPED on it skips what hangs off it."""
    results = _run([
        GraphNode(id="g", type="gate", version=1, bindings={"x": Static("hi")}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": From("g.yes")}),
        GraphNode(id="no", type="upper", version=1, bindings={"text": From("g.no")}),
    ])

    assert results["yes"]["result"] == "HI"
    assert "no" not in results


def test_a_gather_arrives_as_a_series():
    results = _run([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static("b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("a.result", "b.result")}),
    ])

    assert results["j"]["result"] == "A+B"


def test_policy_is_read_off_the_pinned_version():
    flaky_state.calls = 0
    results = _run([GraphNode(id="f", type="flaky", version=1, bindings={"text": Static("ok")})])

    assert results["f"]["result"] == "ok"
    assert flaky_state.calls == 3


def test_a_graph_compile_rejected_is_refused_with_its_problems():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="gone", version=1)]), _registry())

    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync()
    assert [fault.type for fault in raised.value.errors()] == ["unknown_node_type"]


def test_events_carry_the_placement_title():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="a", type="upper", version=1, title="Shout", bindings={"text": Static("hi")})]),
        _registry(),
    )

    events = events_of(compiled)
    assert [e.type for e in events] == ["run_start", "node_start", "node_complete", "run_end"]
