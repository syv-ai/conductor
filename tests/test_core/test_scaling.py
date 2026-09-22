"""A run's time grows with its rows, not with their square.

One graph at 250 and at 1,000 rows: a node unfolds a text into rows, the
rows run through a chain that reads a scalar beside them, a decision
splits them, and three reductions fold them back. Four times the rows
should take about four times as long. Were readiness recomputed after
every unit, it would be sixteen.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Annotated

from conductor import NodeRegistry, Param, execute
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "scaling-test-txt"
    title = "Tekst"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    names: Annotated[Series[Txt], Result(title="Navne")]


@dataclass(frozen=True)
class Length:
    long: Annotated[Txt, Result(title="If long")]
    short: Annotated[Txt, Result(title="If short")]


class Docs(NodeDefinition):
    id = "docs"
    title = "Docs"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Texts", widget=Textarea())] = Txt("")) -> Documents:
        parts = [Txt(p) for p in text.split(",")]
        return Documents(texts=parts, names=[Txt(f"doc{i}") for i in range(len(parts))])


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(text.upper())


class Pair(NodeDefinition):
    id = "pair"
    title = "Pair"
    description = "d"
    category = "test"

    def run(self, a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""), b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt("")) -> Out:
        return Txt(f"{a}:{b}")


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class LongOnly(NodeDefinition):
    id = "long-only"
    title = "Long only"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Length:
        return Length(long=text, short=SKIPPED) if len(text) > 4 else Length(long=SKIPPED, short=text)


def _edge(node_id: str, field: str) -> From:
    return From(Ref(node_id, field))


def _compiled(rows: int) -> CompiledGraph:
    """Ten nodes over ``rows`` rows; every third row takes the short branch."""
    registry = NodeRegistry()
    for node_cls in (Docs, Upper, Pair, Join, LongOnly):
        registry.register(node_cls)
    text = ",".join(f"row{i}" if i % 3 else "r" for i in range(rows))
    graph = Graph(nodes=[
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(text)}),
        GraphNode(id="prefix", type="upper", version=1, bindings={"text": Static("p")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("docs", "texts")}),
        GraphNode(id="pair", type="pair", version=1, bindings={"a": _edge("prefix", "result"), "b": _edge("up", "result")}),
        GraphNode(id="gate", type="long-only", version=1, bindings={"text": _edge("pair", "result")}),
        GraphNode(id="long", type="upper", version=1, bindings={"text": _edge("gate", "long")}),
        GraphNode(id="short", type="upper", version=1, bindings={"text": _edge("gate", "short")}),
        GraphNode(id="long-joined", type="join", version=1, bindings={"texts": _edge("long", "result")}),
        GraphNode(id="short-joined", type="join", version=1, bindings={"texts": _edge("short", "result")}),
        GraphNode(id="names", type="join", version=1, bindings={"texts": _edge("docs", "names")}),
    ])
    compiled = CompiledGraph.from_graph(graph, registry)
    assert compiled.is_runnable, compiled.problems
    return compiled


async def _drain(compiled: CompiledGraph) -> dict:
    """Run one leg through ``execute``, the way a host does, and return its ending."""
    ending = None
    async for ending in execute(compiled):
        pass
    return ending


def _seconds(rows: int) -> float:
    """The best of three runs over ``rows`` rows, through ``execute``, checking the results."""
    compiled = _compiled(rows)
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        results = asyncio.run(_drain(compiled))["results"]
        best = min(best, time.perf_counter() - started)
    short = len(range(0, rows, 3))
    assert len(results["long-joined"]["result"].split("+")) == rows - short
    assert len(results["short-joined"]["result"].split("+")) == short
    assert len(list(results["pair"]["result"])) == rows
    return best


def test_four_times_the_rows_takes_at_most_eight_times_as_long():
    """Measured as a host runs a graph, through ``execute``, with nothing
    patched. Linear is fourfold and quadratic sixteen; the best of three
    runs at each size, and a bound of eight, leave room for a busy machine."""
    _seconds(50)  # warm imports and caches before timing

    small, large = _seconds(250), _seconds(1000)

    assert large <= 8 * small, (small, large)
