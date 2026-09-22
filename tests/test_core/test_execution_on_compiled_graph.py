"""The engine is one more caller of ``CompiledGraph``: it takes what compile
produced and runs it — here, graphs whose nodes each run once. Rows are
``test_rows.py``'s.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import NodeRegistry, Param, run_sync
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import CompilationError
from conductor.execution.engine import execute
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Result
from conductor.node import NodeDefinition, Policy, version
from conductor.ref import Ref
from conductor.series import Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "exec-test-txt"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(text.upper())


@dataclass(frozen=True)
class Answer:
    yes: Annotated[Txt, Result(title="Yes")]
    no: Annotated[Txt, Result(title="No")]


class Gate(NodeDefinition):
    id = "gate"
    title = "Gate"
    description = "d"
    category = "test"

    def run(self, x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Answer:
        return Answer(yes=x, no=SKIPPED) if x else Answer(yes=SKIPPED, no=x)


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class Flaky(NodeDefinition):
    id = "flaky"
    title = "Flaky"
    description = "d"
    category = "test"
    calls = 0

    @version(1, policy=Policy(retries=2, delay=0.0))
    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        Flaky.calls += 1
        if Flaky.calls < 3:
            from conductor.errors import ExternalFailure

            raise ExternalFailure("not yet")
        return text


def _registry():
    registry = NodeRegistry()
    for node_cls in (Upper, Gate, Join, Flaky):
        registry.register(node_cls)
    return registry


def _run(nodes):
    return run_sync(CompiledGraph.from_graph(Graph(nodes=nodes), _registry()))["results"]


def test_a_graph_of_bindings_compiles_and_runs():
    results = _run([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="hi")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Edges(refs=(Ref("a", "result"),))}),
    ])

    assert results["a"]["result"] == "HI"
    assert results["b"]["result"] == "HI"


def test_an_unbound_input_falls_back_to_its_declared_default():
    assert _run([GraphNode(id="a", type="upper", version=1)])["a"]["result"] == ""


def test_a_branch_not_taken_is_skipped_downstream():
    """A branch is an output; SKIPPED on it skips what hangs off it."""
    results = _run([
        GraphNode(id="g", type="gate", version=1, bindings={"x": Static(value="hi")}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": Edges(refs=(Ref("g", "yes"),))}),
        GraphNode(id="no", type="upper", version=1, bindings={"text": Edges(refs=(Ref("g", "no"),))}),
    ])

    assert results["yes"]["result"] == "HI"
    assert "no" not in results


def test_a_gather_arrives_as_a_series():
    results = _run([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static(value="b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": Edges(refs=(Ref("a", "result"), Ref("b", "result")))}),
    ])

    assert results["j"]["result"] == "A+B"


def test_policy_is_read_off_the_pinned_version():
    Flaky.calls = 0
    results = _run([GraphNode(id="f", type="flaky", version=1, bindings={"text": Static(value="ok")})])

    assert results["f"]["result"] == "ok"
    assert Flaky.calls == 3


def test_a_graph_compile_rejected_is_refused_with_its_problems():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="gone", version=1)]), _registry())

    with pytest.raises(CompilationError) as raised:
        run_sync(compiled)
    assert [p.code for p in raised.value.problems] == ["unknown_node_type"]


def test_events_carry_the_placement_title():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="a", type="upper", version=1, title="Shout", bindings={"text": Static(value="hi")})]),
        _registry(),
    )

    async def gathered():
        return [e async for e in execute(compiled)]

    events = asyncio.run(gathered())
    assert [e["type"] for e in events] == ["node_start", "node_complete", "graph_complete"]
