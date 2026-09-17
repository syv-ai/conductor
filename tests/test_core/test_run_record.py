"""A leg ends with a typed run record, and a node the graph changed runs again.

The record is what a host stores between legs and hands back: every cell
in wire form, the rows born, what is done, and a fingerprint per node of
how the graph placed it. It survives JSON, and the next leg's results are
typed like the first's. On restore, a node whose fingerprint differs from
the graph it is restored into — a static edited, a version bumped, a
binding moved — is dropped with everything downstream of it, so those
units run again; a node the graph no longer has is dropped the same way,
and ``without`` drops one on purpose (a host's "run from here").
"""

import asyncio
import json
from typing import Annotated

from conductor import Asks, CompiledGraph, GraphNode, NodeRegistry
from conductor.dtype import DType
from conductor.execution.engine import execute
from conductor.execution.record import RunRecord
from conductor.graph.binding import Edges, Static
from conductor.graph.model import Graph
from conductor.metadata import Input
from conductor.node import NodeDefinition, version
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "record-test-text"
    title = "Text"


In = Annotated[Txt, Textarea(title="In")]
Out = Annotated[Txt, Result(title="Out")]

calls: list[str] = []


class Split(NodeDefinition):
    id = "split"
    title = "Split"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
        calls.append("split")
        return [Txt(part) for part in text.split(",")]


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        calls.append(f"upper:{text}")
        return Txt(text.upper())


class Wrap(NodeDefinition):
    id = "wrap"
    title = "Wrap"
    description = "d"
    category = "test"

    @version(1)
    def run_v1(self, text: In = Txt("")) -> Out:
        calls.append(f"wrap:{text}")
        return Txt(f"[{text}]")

    @version(2)
    def run(self, text: In = Txt("")) -> Out:
        calls.append(f"wrap2:{text}")
        return Txt(f"<{text}>")


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        calls.append(f"echo:{text}")
        return text


class Ask(NodeDefinition):
    id = "ask"
    title = "Ask"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Out | Asks:
        calls.append(f"ask:{text}")
        return Asks(questions=(Input(name="result", dtype=Txt, title="Answer", widget=Textarea(title="Answer"), default=text, optional=True),))


def _registry() -> NodeRegistry:
    reg = NodeRegistry()
    for cls in (Split, Upper, Wrap, Echo, Ask):
        reg.register(cls)
    return reg


def _compiled(nodes: list[GraphNode]) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled


def _leg(compiled: CompiledGraph, **kw) -> list[dict]:
    async def run() -> list[dict]:
        return [event async for event in execute(compiled, **kw)]

    return asyncio.run(run())


def _edge(node: str, field: str = "result") -> Edges:
    return Edges(refs=(Ref(node, field),))


def _through_json(record: RunRecord) -> RunRecord:
    return RunRecord.model_validate(json.loads(json.dumps(record.model_dump())))


# -- the record survives JSON ---------------------------------------------------------


def test_a_paused_legs_record_survives_json_and_the_next_leg_returns_typed_results():
    calls.clear()
    compiled = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="ask", type="ask", version=1, bindings={"text": _edge("split")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("ask")}),
    ])

    first = _leg(compiled)
    assert first[-1]["type"] == "graph_pending"
    record = first[-1]["record"]
    assert isinstance(record, RunRecord)

    restored = _through_json(record)
    second = _leg(compiled, record=restored, cache={"ask": {"result": Series(compiled.field(Ref("split", "result")).index, [Txt("x"), Txt("y")])}})

    assert second[-1]["type"] == "graph_complete"
    parts = second[-1]["results"]["split"]["result"]
    assert isinstance(parts, Series) and all(type(v) is Txt for v in parts)
    assert list(second[-1]["results"]["up"]["result"]) == ["X", "Y"]
    assert calls.count("split") == 1


def test_the_record_carries_a_fingerprint_per_node():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
    ])

    record = _leg(compiled)[-1]["record"]

    assert set(record.fingerprints) == {"a", "b"}
    assert record.fingerprints["a"] == compiled.node("a").fingerprint
    assert all(len(fp) == 64 for fp in record.fingerprints.values())


# -- a changed node runs again -------------------------------------------------------


def test_a_static_edited_between_legs_reruns_that_node_and_its_readers_and_nothing_else():
    calls.clear()
    nodes = [
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="c", type="wrap", version=1, bindings={"text": _edge("b")}),
        GraphNode(id="d", type="echo", version=1, bindings={"text": Static(value="alone")}),
    ]
    record = _leg(_compiled(nodes))[-1]["record"]
    calls.clear()

    edited = _compiled([GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="y")}), *nodes[1:]])
    ending = _leg(edited, record=_through_json(record))[-1]

    assert ending["type"] == "graph_complete"
    assert ending["results"]["c"]["result"] == "[Y]"
    assert ending["results"]["d"]["result"] == "alone"
    assert sorted(calls) == ["echo:y", "upper:y", "wrap:Y"]


def test_a_node_removed_between_legs_is_dropped_with_its_readers():
    calls.clear()
    record = _leg(_compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="c", type="echo", version=1, bindings={"text": Static(value="c")}),
        GraphNode(id="d", type="wrap", version=1, bindings={"text": _edge("c")}),
    ]))[-1]["record"]
    calls.clear()

    # ``c`` is gone and ``d`` now reads ``a``: ``c``'s cells are dropped, ``d`` runs again, ``a`` and ``b`` do not.
    smaller = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="d", type="wrap", version=1, bindings={"text": _edge("a")}),
    ])
    ending = _leg(smaller, record=_through_json(record))[-1]

    assert ending["type"] == "graph_complete"
    assert set(ending["results"]) == {"a", "b", "d"}
    assert ending["results"]["d"]["result"] == "[x]"
    assert calls == ["wrap:x"]


def test_a_nodes_shape_changed_between_legs_reruns_it():
    calls.clear()
    nodes = [
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="w", type="wrap", version=1, bindings={"text": _edge("a")}),
    ]
    record = _leg(_compiled(nodes))[-1]["record"]
    calls.clear()

    bumped = _compiled([nodes[0], GraphNode(id="w", type="wrap", version=2, bindings={"text": _edge("a")})])
    ending = _leg(bumped, record=_through_json(record))[-1]

    assert ending["results"]["w"]["result"] == "<x>"
    assert calls == ["wrap2:x"]


def test_without_drops_a_node_on_purpose_with_everything_downstream():
    """Run from here: the host drops one node from the record, and the
    restore drops what reads it, so they run again while the rest is kept."""
    calls.clear()
    compiled = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("split")}),
        GraphNode(id="w", type="wrap", version=1, bindings={"text": _edge("up")}),
        GraphNode(id="d", type="echo", version=1, bindings={"text": Static(value="alone")}),
    ])
    record = _leg(compiled)[-1]["record"]
    calls.clear()

    ending = _leg(compiled, record=_through_json(record.without("up")))[-1]

    assert ending["type"] == "graph_complete"
    assert list(ending["results"]["w"]["result"]) == ["[A]", "[B]"]
    assert sorted(calls) == ["upper:a", "upper:b", "wrap:A", "wrap:B"]
    assert "up" not in record.without("up").fingerprints and "split" in record.without("up").fingerprints
