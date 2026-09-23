"""The leg owns its work.

The leg owns the threads its units run on and the tasks that wait on them:
closing the event stream stops every unit, a node's timeout counts only the
time its thread ran, a timed-out attempt is final and its thread keeps the
node's concurrency slot until it returns, and the leg has no deadline
unless the caller sets one.
"""

import asyncio
import inspect
import threading
import time
from typing import Annotated

from conductor import CompiledGraph, GraphNode, NodeRegistry, Param, run_sync
from conductor.dtype import DType
from conductor.execution.engine import execute
from conductor.graph.binding import From, Static
from conductor.graph.model import Graph
from conductor.metadata import Result
from conductor.node import NodeDefinition, Policy, version
from conductor.series import Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "leg-test-text"
    title = "Text"


In = Annotated[Txt, Param(title="In", widget=Textarea())]
Out = Annotated[Txt, Result(title="Out")]
Parts = Annotated[Series[Txt], Result(title="Parts")]


class Split(NodeDefinition):
    """``n`` rows, so a node fed by it runs ``n`` times."""

    id = "split"
    title = "Split"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Parts:
        return [Txt(part) for part in text.split(",")]


def _rows(count: int) -> str:
    return ",".join(str(i) for i in range(count))


def _per_row(slow_cls: type[NodeDefinition], rows: int, *more: type[NodeDefinition]) -> CompiledGraph:
    """``split`` feeding ``rows`` rows into ``slow_cls`` as ``slow``, plus ``more`` as root nodes."""
    reg = NodeRegistry()
    for cls in (Split, slow_cls, *more):
        reg.register(cls)
    nodes = [
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(_rows(rows))}),
        GraphNode(id="slow", type=slow_cls.id, version=1, bindings={"text": From("split.result")}),
        *[GraphNode(id=cls.id, type=cls.id, version=1, bindings={"text": Static("x")}) for cls in more],
    ]
    return CompiledGraph.from_graph(Graph(nodes=nodes), reg)


def _single(cls: type[NodeDefinition]) -> CompiledGraph:
    reg = NodeRegistry()
    reg.register(cls)
    return CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="n1", type=cls.id, version=1, bindings={"text": Static("x")})]), reg
    )


def _events(compiled: CompiledGraph, **kw) -> list[dict]:
    async def run() -> list[dict]:
        return [event async for event in execute(compiled, **kw)]

    return asyncio.run(run())


# -- closing the stream -------------------------------------------------------------


def _sleeper(seconds: float, calls: list[str], concurrency: int = 1) -> type[NodeDefinition]:
    class Sleeper(NodeDefinition):
        id = "sleeper"
        title = "Sleeper"
        description = "d"
        category = "test"

        @version(1, policy=Policy(concurrency=concurrency))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(str(text))
            time.sleep(seconds)
            return Txt(text.upper())

    return Sleeper


def test_closing_the_stream_after_one_event_stops_the_rest():
    """``aclose()`` on the generator ends every unit: the rows still to run
    never start, and the call count stops moving."""
    calls: list[str] = []
    compiled = _per_row(_sleeper(0.05, calls), 20)

    async def run() -> tuple[int, int]:
        events = execute(compiled)
        async for event in events:
            if event["type"] == "node_start" and event["node_id"] == "slow":
                break  # the twenty rows are started; one is in its thread
        await events.aclose()
        await asyncio.sleep(0.3)
        seen = len(calls)
        await asyncio.sleep(0.3)
        return seen, len(calls)

    seen, later = asyncio.run(run())

    assert later == seen
    assert seen <= 2, f"{seen} rows ran after the stream was closed"


def test_cancelling_the_consumer_stops_the_rest():
    calls: list[str] = []
    compiled = _per_row(_sleeper(0.05, calls), 20)

    async def run() -> tuple[int, int]:
        started = asyncio.Event()

        async def consume() -> None:
            async for event in execute(compiled):
                if event["type"] == "node_start" and event["node_id"] == "slow":
                    started.set()

        consumer = asyncio.create_task(consume())
        await started.wait()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await asyncio.sleep(0.3)
        seen = len(calls)
        await asyncio.sleep(0.3)
        return seen, len(calls)

    seen, later = asyncio.run(run())

    assert later == seen
    assert seen <= 2, f"{seen} rows ran after the consumer was cancelled"


# -- the clock starts when the thread does -----------------------------------------


def test_an_instant_node_beside_forty_slow_rows_does_not_time_out():
    """Forty rows of a slow node start at once, and an instant node with a
    short timeout becomes ready in the same moment; it gets its own thread
    at once, so its clock never counts time spent waiting for a worker."""
    calls: list[str] = []

    class Instant(NodeDefinition):
        id = "instant"
        title = "Instant"
        description = "d"
        category = "test"

        @version(1, policy=Policy(timeout=0.1))
        def run(self, texts: Annotated[Series[Txt], Param(title="In", widget=Textarea())]) -> Out:
            return Txt("now")

    reg = NodeRegistry()
    for cls in (Split, _sleeper(0.5, calls, concurrency=40), Instant):
        reg.register(cls)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(_rows(40))}),
        GraphNode(id="slow", type="sleeper", version=1, bindings={"text": From("split.result")}),
        GraphNode(id="instant", type="instant", version=1, bindings={"texts": From("split.result")}),
    ]), reg)

    events = _events(compiled)

    assert events[-1]["type"] == "graph_complete", events[-1]
    assert events[-1]["results"]["instant"]["result"] == "now"
    assert len(calls) == 40


# -- a timed-out attempt ---------------------------------------------------------


def test_concurrency_one_with_a_timeout_never_has_two_calls_in_flight():
    """The thread of a timed-out attempt keeps the node's slot until it
    returns, so ``concurrency=1`` is one thread whatever the clock says; the
    timeout is final, so nothing retries beside it; and the leg returns at
    the timeout, not when the abandoned thread does."""
    in_flight = 0
    most = 0
    lock = threading.Lock()

    class Slow(NodeDefinition):
        id = "slow-rows"
        title = "Slow"
        description = "d"
        category = "test"

        @version(1, policy=Policy(concurrency=1, timeout=0.2, retries=2, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            nonlocal in_flight, most
            with lock:
                in_flight += 1
                most = max(most, in_flight)
            try:
                time.sleep(0.5)
            finally:
                with lock:
                    in_flight -= 1
            return Txt(text)

    compiled = _per_row(Slow, 3)

    start = time.monotonic()
    events = _events(compiled)
    elapsed = time.monotonic() - start
    time.sleep(0.6)  # let the abandoned thread finish before reading the counters

    assert most == 1
    assert [e["type"] for e in events if e["type"] == "node_retry"] == []
    assert events[-1]["type"] == "graph_error"
    assert events[-1]["cause"].code == "timeout"
    assert elapsed < 0.45, f"the leg waited {elapsed:.2f}s for an abandoned thread"


def test_a_timed_out_attempt_is_final_and_its_cause_is_timeout():
    calls: list[str] = []

    class Slow(NodeDefinition):
        id = "slow-once"
        title = "Slow"
        description = "d"
        category = "test"

        @version(1, policy=Policy(timeout=0.1, retries=3, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(str(text))
            time.sleep(0.3)
            return Txt("late")

    events = _events(_single(Slow))
    time.sleep(0.4)

    assert len(calls) == 1
    error = next(e for e in events if e["type"] == "node_error")
    assert error["cause"].code == "timeout"
    assert error["cause"].message == "The node did not answer in time."
    assert error["cause"].details == {"seconds": 0.1}
    assert "node_retry" not in [e["type"] for e in events]


# -- the leg's own bound ------------------------------------------------------------


def test_execute_has_no_deadline_by_default_and_runs_a_slow_node_to_the_end():
    class Slow(NodeDefinition):
        id = "slow-second"
        title = "Slow"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Out:
            time.sleep(1.0)
            return Txt("done")

    assert inspect.signature(execute).parameters["timeout"].default is None
    assert "timeout_seconds" not in inspect.signature(execute).parameters

    assert run_sync(_single(Slow))["results"]["n1"]["result"] == "done"


def test_a_leg_deadline_ends_the_leg_at_once_with_graph_timeout():
    class Slow(NodeDefinition):
        id = "slow-second"
        title = "Slow"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Out:
            time.sleep(1.0)
            return Txt("done")

    start = time.monotonic()
    events = _events(_single(Slow), timeout=0.2)
    elapsed = time.monotonic() - start

    assert events[-1]["type"] == "graph_timeout"
    assert events[-1]["timeout_seconds"] == 0.2
    assert 0.2 <= events[-1]["elapsed_seconds"] < 0.5
    assert elapsed < 0.5


def test_a_cancel_set_mid_run_ends_the_leg_within_one_event():
    """The cancel event is awaited, not polled: the ending follows it at once."""
    calls: list[str] = []
    compiled = _per_row(_sleeper(0.05, calls), 20)

    async def run() -> tuple[list[str], float]:
        cancel = asyncio.Event()
        types: list[str] = []
        set_at = 0.0
        async for event in execute(compiled, cancel=cancel):
            types.append(event["type"])
            if event["type"] == "node_start" and event["node_id"] == "slow":
                cancel.set()
                set_at = time.monotonic()
        return types, time.monotonic() - set_at

    types, after = asyncio.run(run())

    assert types[-1] == "graph_cancelled"
    assert types[types.index("graph_cancelled") - 1] == "node_start"
    assert after < 0.2, f"the cancel took {after:.2f}s to end the leg"


def test_a_ready_unit_nobody_started_stops_the_leg_loudly(monkeypatch):
    """What the engine starts is what a write reports ready. A ledger whose
    write reports nothing leaves units ready and unstarted, and the leg
    raises when it goes quiet rather than ending short."""
    import pytest
    from conductor.execution.ledger import Ledger

    class Echo(NodeDefinition):
        id = "echo-wake"
        title = "Echo"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Out:
            return text

    record = Ledger.record

    def silent(self, unit, outputs):
        record(self, unit, outputs)
        return []

    monkeypatch.setattr(Ledger, "record", silent)
    graph = Graph(nodes=[
        GraphNode(id="a", type="echo-wake", version=1),
        GraphNode(id="b", type="echo-wake", version=1, bindings={"text": From("a.result")}),
    ])
    with pytest.raises(RuntimeError, match="missed wake"):
        run_sync(CompiledGraph.from_graph(graph, NodeRegistry(nodes=(Echo,))))
