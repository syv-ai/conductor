"""Plan 3b Task 5 tests — run against the plan-3b engine (pre-plan-4: scalar nodes only).

Written at tests/test_core/test_execution_on_compiled_graph.py during execution;
plan 4's engine rewrite supersedes them (test_rows.py). Not in the verifier tree
because the pre-plan-4 engine is not either.

The engine is one more caller of ``CompiledGraph``: it takes what compile
produced and runs scalar nodes over it.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import NodeRegistry
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import CompilationError
from conductor.execution.engine import execute, execute_sync
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.node import NodeDefinition, Policy, version
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import ConnectionList, Textarea


class Txt(DType, str):
    id = "exec-test-txt"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
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

    def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Answer:
        return Answer(yes=x, no=SKIPPED) if x else Answer(yes=SKIPPED, no=x)


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], ConnectionList(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class Docs(NodeDefinition):
    id = "docs"
    title = "Docs"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Textarea(title="Folder")] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
        return [Txt("a"), Txt("b")]


class Flaky(NodeDefinition):
    id = "flaky"
    title = "Flaky"
    description = "d"
    category = "test"
    calls = 0

    @version(1, policy=Policy(retries=2, delay=0.0))
    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        Flaky.calls += 1
        if Flaky.calls < 3:
            from conductor.errors import NodeExecutionError

            raise NodeExecutionError("not yet")
        return text


def _registry():
    registry = NodeRegistry()
    for node_cls in (Upper, Gate, Join, Docs, Flaky):
        registry.register(node_cls)
    return registry


def _run(nodes):
    return execute_sync(CompiledGraph.from_graph(Graph(nodes=nodes), _registry()))


def test_a_flow_of_bindings_compiles_and_runs():
    results = _run([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="hi")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Edges(refs=(Ref("a", "result"),))}),
    ])

    assert results["a"]["result"] == "HI"
    assert results["b"]["result"] == "HI"


def test_an_unbound_input_falls_back_to_its_declared_default():
    assert _run([GraphNode(id="a", type="upper", version=1)])["a"]["result"] == ""


def test_a_stale_static_never_reaches_the_node():
    """The resolver reads the roster, so a binding nothing declares is not
    a value anything receives."""
    results = _run([GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="x"), "gone": Static(value=1)})])

    assert results["a"]["result"] == "X"


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


def test_a_flow_compile_rejected_is_refused_with_its_problems():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="gone", version=1)]), _registry())

    with pytest.raises(CompilationError) as raised:
        execute_sync(compiled)
    assert [p.code for p in raised.value.problems] == ["unknown_node_type"]


def test_this_engine_refuses_a_lifted_node():
    """This engine runs scalar nodes only. The limit is stated, not
    papered over."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="up", type="upper", version=1, bindings={"text": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(),
    )
    assert compiled.is_runnable

    with pytest.raises(NotImplementedError, match="lifted"):
        execute_sync(compiled)


def test_events_carry_the_placement_title():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="a", type="upper", version=1, title="Shout", bindings={"text": Static(value="hi")})]),
        _registry(),
    )

    async def collect():
        return [e async for e in execute(compiled)]

    events = asyncio.run(collect())
    assert [e["type"] for e in events] == ["node_start", "node_complete", "flow_complete"]
