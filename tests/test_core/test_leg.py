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
from typing import Annotated, Any

import pytest
from conductor import (
    CompiledGraph,
    DType,
    ExternalFailure,
    From,
    Graph,
    GraphNode,
    NodeDefinition,
    NodeRegistry,
    Param,
    Policy,
    Result,
    RunState,
    Series,
    Static,
    node,
)
from conductor.widgets import Textarea
from helpers import stream_of


class Txt(DType, str):
    id = "leg-test-text"
    display_title = "Text"


In = Annotated[Txt, Param(title="In", widget=Textarea())]
Out = Annotated[Txt, Result(title="Out")]
Parts = Annotated[Series[Txt], Result(title="Parts")]


@node("split", title="Split", description="d", category="test")
def split_node(text: In = Txt("")) -> Parts:
    """``n`` rows, so a node fed by it runs ``n`` times."""
    return [Txt(part) for part in text.split(",")]


def _rows(count: int) -> str:
    return ",".join(str(i) for i in range(count))


def _per_row(slow_cls: type[NodeDefinition], rows: int, *more: type[NodeDefinition]) -> CompiledGraph:
    """``split`` feeding ``rows`` rows into ``slow_cls`` as ``slow``, plus ``more`` as root nodes."""
    reg = NodeRegistry()
    for cls in (split_node, slow_cls, *more):
        reg = reg.extended_with([cls])
    nodes = [
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(_rows(rows))}),
        GraphNode(id="slow", type=slow_cls.id, version=1, bindings={"text": From("split.result")}),
        *[GraphNode(id=cls.id, type=cls.id, version=1, bindings={"text": Static("x")}) for cls in more],
    ]
    return CompiledGraph.from_graph(Graph(nodes=nodes), reg)


def _single(cls: type[NodeDefinition]) -> CompiledGraph:
    reg = NodeRegistry([cls])
    return CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="n1", type=cls.id, version=1, bindings={"text": Static("x")})]), reg
    )


def _events(compiled: CompiledGraph, **kw) -> list[dict]:
    async def run() -> list[dict]:
        return [event async for event in stream_of(compiled, **kw)]

    return asyncio.run(run())


# -- closing the stream -------------------------------------------------------------


def _sleeper(seconds: float, calls: list[str], concurrency: int = 1) -> type[NodeDefinition]:
    @node("sleeper", title="Sleeper", description="d", category="test", policy=Policy(concurrency=concurrency))
    def sleeper(text: In = Txt("")) -> Out:
        calls.append(str(text))
        time.sleep(seconds)
        return Txt(text.upper())

    return sleeper


def test_closing_the_stream_after_one_event_stops_the_rest():
    """``aclose()`` on the generator ends every unit: the rows still to run
    never start, and the call count stops moving."""
    calls: list[str] = []
    compiled = _per_row(_sleeper(0.05, calls), 20)

    async def run() -> tuple[int, int]:
        events = stream_of(compiled)
        async for event in events:
            if event.type == "node_start" and event.node_id == "slow":
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
            async for event in stream_of(compiled):
                if event.type == "node_start" and event.node_id == "slow":
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

    @node("instant", title="Instant", description="d", category="test", policy=Policy(timeout=0.1))
    def instant(texts: Annotated[Series[Txt], Param(title="In", widget=Textarea())]) -> Out:
        return Txt("now")

    reg = NodeRegistry()
    for cls in (split_node, _sleeper(0.5, calls, concurrency=40), instant):
        reg = reg.extended_with([cls])
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(_rows(40))}),
        GraphNode(id="slow", type="sleeper", version=1, bindings={"text": From("split.result")}),
        GraphNode(id="instant", type="instant", version=1, bindings={"texts": From("split.result")}),
    ]), reg)

    events = _events(compiled)

    assert events[-1].state.status == "complete", events[-1]
    assert compiled.results(events[-1].state)["instant"]["result"] == "now"
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

    @node("slow-rows", title="Slow", description="d", category="test", policy=Policy(concurrency=1, timeout=0.2, retries=2, delay=0.01))
    def slow(text: In = Txt("")) -> Out:
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

    compiled = _per_row(slow, 3)

    start = time.monotonic()
    events = _events(compiled)
    elapsed = time.monotonic() - start
    time.sleep(0.6)  # let the abandoned thread finish before reading the counters

    assert most == 1
    assert [e.type for e in events if e.type == "node_retry"] == []
    assert events[-1].state.status == "failed"
    assert events[-1].state.failure.type == "timeout"
    assert elapsed < 0.45, f"the leg waited {elapsed:.2f}s for an abandoned thread"


def test_a_timed_out_attempt_is_final_and_its_cause_is_timeout():
    calls: list[str] = []

    @node("slow-once", title="Slow", description="d", category="test", policy=Policy(timeout=0.1, retries=3, delay=0.01))
    def slow(text: In = Txt("")) -> Out:
        calls.append(str(text))
        time.sleep(0.3)
        return Txt("late")

    events = _events(_single(slow))
    time.sleep(0.4)

    assert len(calls) == 1
    error = next(e for e in events if e.type == "node_failed")
    assert error.cause.type == "timeout"
    assert error.cause.msg == "The node did not answer in time."
    assert error.cause.ctx == {"seconds": 0.1}
    assert "node_retry" not in [e.type for e in events]


# -- the leg's own bound ------------------------------------------------------------


def test_execute_has_no_deadline_by_default_and_runs_a_slow_node_to_the_end():
    @node("slow-second", title="Slow", description="d", category="test")
    def slow(text: In = Txt("")) -> Out:
        time.sleep(1.0)
        return Txt("done")

    assert inspect.signature(CompiledGraph.run).parameters["timeout"].default is None

    compiled = _single(slow)
    assert compiled.results(compiled.run_sync())["n1"]["result"] == "done"


def test_a_leg_deadline_ends_the_leg_at_once_timed_out():
    @node("slow-second", title="Slow", description="d", category="test")
    def slow(text: In = Txt("")) -> Out:
        time.sleep(1.0)
        return Txt("done")

    start = time.monotonic()
    events = _events(_single(slow), timeout=0.2)
    elapsed = time.monotonic() - start

    assert events[-1].state.status == "timed_out"
    assert events[-1].state.failure.ctx["timeout_seconds"] == 0.2
    assert 0.2 <= events[-1].state.failure.ctx["elapsed_seconds"] < 0.5
    assert elapsed < 0.5


def test_a_cancel_set_mid_run_ends_the_leg_within_one_event():
    """The cancel event is awaited, not polled: the ending follows it at once."""
    calls: list[str] = []
    compiled = _per_row(_sleeper(0.05, calls), 20)

    async def run() -> tuple[list[str], float]:
        cancel = asyncio.Event()
        types: list[str] = []
        set_at = 0.0
        async for event in stream_of(compiled, cancel=cancel):
            types.append(event.type)
            if event.type == "node_start" and event.node_id == "slow":
                cancel.set()
                set_at = time.monotonic()
        return types, time.monotonic() - set_at

    types, after = asyncio.run(run())

    assert types[-1] == "run_end"
    assert types[-2] == "node_start"
    assert after < 0.2, f"the cancel took {after:.2f}s to end the leg"


def test_a_ready_unit_nobody_started_stops_the_leg_loudly(monkeypatch):
    """What the engine starts is what a write reports ready. A ledger whose
    write reports nothing leaves units ready and unstarted, and the leg
    raises when it goes quiet rather than ending short."""
    import pytest
    from conductor._internal.execution.run_ledger import RunLedger

    @node("echo-wake", title="Echo", description="d", category="test")
    def echo(text: In = Txt("")) -> Out:
        return text

    record = RunLedger.record

    def silent(self, unit, outputs):
        record(self, unit, outputs)
        return []

    monkeypatch.setattr(RunLedger, "record", silent)
    graph = Graph(nodes=[
        GraphNode(id="a", type="echo-wake", version=1),
        GraphNode(id="b", type="echo-wake", version=1, bindings={"text": From("a.result")}),
    ])
    with pytest.raises(RuntimeError, match="missed wake"):
        CompiledGraph.from_graph(graph, NodeRegistry((echo,))).run_sync()


# -- what a node may raise, and how a unit gets its turn -------------------------------


@pytest.mark.parametrize(
    "raised",
    [asyncio.CancelledError(), GeneratorExit(), BaseException("b"), SystemExit(3), KeyboardInterrupt(), StopIteration()],
    ids=lambda raised: type(raised).__name__,
)
def test_anything_a_node_raises_fails_its_unit_and_the_leg_ends(raised):
    @node("raises", title="R", description="d", category="test")
    def raises(text: In = Txt("")) -> Out:
        raise raised

    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="r", type="raises", version=1)]), NodeRegistry([raises]))

    async def bounded() -> RunState:
        return await asyncio.wait_for(compiled.run(), 3)

    state = asyncio.run(bounded())
    assert (state.status, state.failure.type, state.failure.loc[0]) == ("failed", "execution_failed", "r")


def _wide_graph(rows: int, **definition: Any) -> CompiledGraph:
    @node("source", title="S", description="d", category="test")
    def source(text: In = Txt("")) -> Parts:
        return [Txt(str(n)) for n in range(rows)]

    @node("one", title="O", description="d", category="test", **definition)
    def one(text: Annotated[Txt, Param(title="T")]) -> Out:
        return text

    nodes = [GraphNode(id="s", type="source", version=1), GraphNode(id="o", type="one", version=1, bindings={"text": From("s.result")})]
    return CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry([source, one]))


def test_a_cancel_set_while_the_queue_holds_messages_ends_the_leg_at_the_next_event():
    cancel = asyncio.Event()

    async def go() -> list[Any]:
        seen = []
        async with _wide_graph(2000).stream(cancel=cancel) as events:
            async for event in events:
                seen.append(event)
                if len(seen) == 20:
                    cancel.set()
        return seen

    seen = asyncio.run(go())
    assert len(seen) == 21 and seen[-1].state.status == "cancelled"


def test_a_cancel_from_another_task_lands_within_a_moment():
    cancel = asyncio.Event()

    async def go() -> tuple[str, float]:
        running = asyncio.create_task(_wide_graph(20000).run(cancel=cancel))
        await asyncio.sleep(0.05)
        asked = time.monotonic()
        cancel.set()
        state = await running
        return state.status, time.monotonic() - asked

    status, took = asyncio.run(go())
    assert status == "cancelled" and took < 0.5


def test_a_unit_waiting_for_a_slot_has_no_task():
    release = threading.Event()
    peak: list[int] = []

    @node("source", title="S", description="d", category="test")
    def source(text: In = Txt("")) -> Parts:
        return [Txt(str(n)) for n in range(500)]

    @node("held", title="H", description="d", category="test", policy=Policy(concurrency=2))
    def held(text: Annotated[Txt, Param(title="T")]) -> Out:
        release.wait(5)
        return text

    nodes = [GraphNode(id="s", type="source", version=1), GraphNode(id="h", type="held", version=1, bindings={"text": From("s.result")})]
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry([source, held]))

    async def go() -> RunState:
        running = asyncio.create_task(compiled.run())
        await asyncio.sleep(0.2)
        peak.append(len(asyncio.all_tasks()))
        release.set()
        return await running

    assert asyncio.run(go()).status == "complete"
    assert peak[0] < 10


def test_the_retry_delay_doubles_each_time_and_a_retry_holds_no_slot_while_it_waits():
    tries: dict[str, int] = {}

    @node("source", title="S", description="d", category="test")
    def source(text: In = Txt("")) -> Parts:
        return [Txt("flaky"), Txt("steady")]

    @node("flaky", title="F", description="d", category="test", policy=Policy(concurrency=1, retries=3, delay=0.02))
    def flaky(text: Annotated[Txt, Param(title="T")]) -> Out:
        tries[text] = tries.get(text, 0) + 1
        if text == "flaky" and tries[text] <= 3:
            raise ExternalFailure("try again")
        finished.append(str(text))
        return text

    finished: list[str] = []
    nodes = [GraphNode(id="s", type="source", version=1), GraphNode(id="f", type="flaky", version=1, bindings={"text": From("s.result")})]
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry([source, flaky]))

    async def go() -> list[Any]:
        return [event async for event in stream_of(compiled)]

    events = asyncio.run(go())
    assert [event.delay for event in events if event.type == "node_retry"] == [0.02, 0.04, 0.08]
    assert finished == ["steady", "flaky"]


def test_a_leg_makes_no_more_threads_than_calls_in_flight():
    seen: set[int] = set()

    @node("source", title="S", description="d", category="test")
    def source(text: In = Txt("")) -> Parts:
        return [Txt(str(n)) for n in range(20)]

    @node("step", title="T", description="d", category="test")
    def step(text: Annotated[Txt, Param(title="T")]) -> Out:
        seen.add(threading.get_ident())
        return text

    nodes = [GraphNode(id="s", type="source", version=1), GraphNode(id="n0", type="step", version=1, bindings={"text": From("s.result")})]
    nodes += [GraphNode(id=f"n{i}", type="step", version=1, bindings={"text": From(f"n{i - 1}.result")}) for i in range(1, 100)]
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry([source, step]))
    assert compiled.run_sync().status == "complete"
    assert len(seen) <= 21
