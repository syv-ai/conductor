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

from conductor import SKIPPED, CompiledGraph, DType, Param, Result, Series, node
from conductor._internal.graph.binding import From, Static
from conductor.widgets import Textarea
from helpers import placed, record, registry_of, stream_of


class Txt(DType, str):
    id = "scaling-test-txt"
    display_title = "Tekst"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    names: Annotated[Series[Txt], Result(title="Navne")]


@dataclass(frozen=True)
class Length:
    long: Annotated[Txt, Result(title="If long")]
    short: Annotated[Txt, Result(title="If short")]


@node("docs", title="Docs", description="d", category="test")
def docs(text: Annotated[Txt, Param(title="Texts", widget=Textarea())] = Txt("")) -> Documents:
    parts = [Txt(p) for p in text.split(",")]
    return Documents(texts=parts, names=[Txt(f"doc{i}") for i in range(len(parts))])


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("pair", title="Pair", description="d", category="test")
def pair(a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""), b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt("")) -> Out:
    return Txt(f"{a}:{b}")


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("+".join(texts))


@node("long-only", title="Long only", description="d", category="test")
def long_only(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Length:
    return Length(long=text, short=SKIPPED) if len(text) > 4 else Length(long=SKIPPED, short=text)


def _edge(node_id: str, field: str) -> From:
    return From(f"{node_id}.{field}")


def _compiled(rows: int) -> CompiledGraph:
    """Ten nodes over ``rows`` rows; every third row takes the short branch."""
    registry = registry_of(docs, upper_node, pair, join_node, long_only)
    text = ",".join(f"row{i}" if i % 3 else "r" for i in range(rows))
    graph = record(placed("docs", "docs", text=Static(text)), placed("prefix", "upper", text=Static("p")), placed("up", "upper", text=_edge("docs", "texts")), placed("pair", "pair", a=_edge("prefix", "result"), b=_edge("up", "result")), placed("gate", "long-only", text=_edge("pair", "result")), placed("long", "upper", text=_edge("gate", "long")), placed("short", "upper", text=_edge("gate", "short")), placed("long-joined", "join", texts=_edge("long", "result")), placed("short-joined", "join", texts=_edge("short", "result")), placed("names", "join", texts=_edge("docs", "names")))
    compiled = CompiledGraph.from_graph(graph, registry)
    assert compiled.is_runnable, compiled.problems
    return compiled


async def _drain(compiled: CompiledGraph) -> dict:
    """Run one leg through ``stream``, the way a host does, and return its ending."""
    ending = None
    async for ending in stream_of(compiled):
        pass
    return ending.state


def _seconds(rows: int) -> float:
    """The best of three runs over ``rows`` rows, through ``stream``, checking the results."""
    compiled = _compiled(rows)
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        ending = asyncio.run(_drain(compiled))
        best = min(best, time.perf_counter() - started)
        results = compiled.results(ending)
    short = len(range(0, rows, 3))
    assert len(results["long-joined"]["result"].split("+")) == rows - short
    assert len(results["short-joined"]["result"].split("+")) == short
    assert len(results["pair"]["result"]["values"]) == rows
    return best


def test_four_times_the_rows_takes_at_most_eight_times_as_long():
    """Measured as a host runs a graph, through ``stream``, with nothing
    patched. Linear is fourfold and quadratic sixteen; the best of three
    runs at each size, and a bound of eight, leave room for a busy machine."""
    _seconds(50)  # warm imports and caches before timing

    small, large = _seconds(250), _seconds(1000)

    assert large <= 8 * small, (small, large)
