"""A run's time grows with its rows, not with their square.

One graph at 250 and at 1,000 rows: a node unfolds a text into rows, the
rows run through a chain that reads a scalar beside them, a decision
splits them, and three reductions fold them back. Four times the rows
should be about four times the work. Were readiness recomputed after
every unit, it would be sixteen.
"""

import os
import time
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import NodeRegistry, Param
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.execution.engine import execute_sync
from conductor.execution.ledger import Ledger
from conductor.graph.binding import Edges, Static
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


def _edge(node_id: str, field: str) -> Edges:
    return Edges(refs=(Ref(node_id, field),))


def _compiled(rows: int) -> CompiledGraph:
    """Ten nodes over ``rows`` rows; every third row takes the short branch."""
    registry = NodeRegistry()
    for node_cls in (Docs, Upper, Pair, Join, LongOnly):
        registry.register(node_cls)
    text = ",".join(f"row{i}" if i % 3 else "r" for i in range(rows))
    graph = Graph(nodes=[
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(value=text)}),
        GraphNode(id="prefix", type="upper", version=1, bindings={"text": Static(value="p")}),
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


def _work(rows: int, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    """``(readiness checks, cells read)`` for one run over ``rows`` rows, checking the results."""
    counts = {"ready": 0, "lookup": 0}
    ready, lookup = Ledger.ready, Ledger._lookup

    def counted_ready(self, unit):
        counts["ready"] += 1
        return ready(self, unit)

    def counted_lookup(self, ref, key):
        counts["lookup"] += 1
        return lookup(self, ref, key)

    compiled = _compiled(rows)
    with monkeypatch.context() as patched:
        patched.setattr(Ledger, "ready", counted_ready)
        patched.setattr(Ledger, "_lookup", counted_lookup)
        results = execute_sync(compiled)

    short = len(range(0, rows, 3))
    assert len(results["long-joined"]["result"].split("+")) == rows - short
    assert len(results["short-joined"]["result"].split("+")) == short
    assert len(list(results["pair"]["result"])) == rows
    return counts["ready"], counts["lookup"]


def test_four_times_the_rows_is_about_four_times_the_work(monkeypatch):
    """Counted, not timed, so it holds on any machine: readiness checks and
    the cells they read grow with the rows. Quadratic would be sixteenfold."""
    small_checks, small_reads = _work(250, monkeypatch)
    large_checks, large_reads = _work(1000, monkeypatch)

    assert large_checks <= 5 * small_checks, (small_checks, large_checks)
    assert large_reads <= 5 * small_reads, (small_reads, large_reads)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="wall-clock timing is noise on a shared CI runner")
def test_four_times_the_rows_takes_at_most_eight_times_as_long():
    """The wall clock, with room for noise: linear is fourfold, quadratic sixteen."""
    execute_sync(_compiled(50))  # warm imports and caches before timing

    def timed(rows: int) -> float:
        compiled = _compiled(rows)
        started = time.perf_counter()
        execute_sync(compiled)
        return time.perf_counter() - started

    small, large = timed(250), timed(1000)

    assert large <= 8 * small, (small, large)


def test_a_ready_unit_nobody_started_stops_the_leg_loudly(monkeypatch):
    """What the engine starts is what a write reports ready. A write that
    reports nothing leaves units ready and unstarted, and the leg raises
    when it goes quiet rather than ending short."""
    record = Ledger.record

    def silent(self, unit, outputs):
        record(self, unit, outputs)
        return []

    monkeypatch.setattr(Ledger, "record", silent)

    with pytest.raises(RuntimeError, match="missed wake"):
        execute_sync(_compiled(3))
