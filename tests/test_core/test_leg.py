"""The leg owns its work.

The leg owns the threads its units run on:
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
    AskPerson,
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

    @node("instant", title="Instant", description="d", category="test", policy=Policy(attempt_timeout=0.1))
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

    @node("slow-rows", title="Slow", description="d", category="test", policy=Policy(concurrency=1, attempt_timeout=0.2, retries=2, retry_delay=0.01))
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

    @node("slow-once", title="Slow", description="d", category="test", policy=Policy(attempt_timeout=0.1, retries=3, retry_delay=0.01))
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


def _events_of(compiled: CompiledGraph) -> list[Any]:
    async def go() -> list[Any]:
        async with compiled.stream() as events:
            return [event async for event in events]

    return asyncio.run(go())


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
                if event.type == "node_start" and event.node_id == "o":
                    cancel.set()
        return seen

    seen = asyncio.run(go())
    types = [event.type for event in seen]
    after_the_cancel = types[next(i for i, event in enumerate(seen) if event.type == "node_start" and event.node_id == "o") + 1 :]
    assert after_the_cancel[-1] == "run_end" and seen[-1].state.status == "cancelled"
    assert "node_complete" not in after_the_cancel[:-1]


def test_a_fast_node_over_many_rows_reports_a_few_times_and_ends_on_its_last_count():
    started = time.monotonic()
    events = _events_of(_wide_graph(20_000))
    took = time.monotonic() - started

    progress = [event for event in events if event.type == "node_progress"]
    assert len(progress) <= took / 0.25 + 2
    assert (progress[-1].done, progress[-1].total) == (20_000, 20_000)
    assert [event.type for event in events].index("node_complete", [event.type for event in events].index("node_progress")) > events.index(progress[-1])


def test_a_leg_that_goes_quiet_reports_the_last_count_of_a_node_still_waiting():
    @node("source", title="S", description="d", category="test")
    def source() -> Parts:
        return [Txt("ask"), Txt("b"), Txt("c")]

    @node("maybe", title="M", description="d", category="test")
    def maybe(text: Annotated[Txt, Param(title="T")]) -> Out | AskPerson:
        return AskPerson() if text == "ask" else text

    nodes = [GraphNode(id="s", type="source", version=1), GraphNode(id="m", type="maybe", version=1, bindings={"text": From("s.result")})]
    events = _events_of(CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry([source, maybe])))

    last = [event for event in events if event.type == "node_progress"][-1]
    assert (last.done, last.total) == (2, 3)
    assert events[-1].state.status == "waiting"


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


def test_units_waiting_for_a_slot_hold_no_tasks_and_the_leg_is_one_task():
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

    @node("flaky", title="F", description="d", category="test", policy=Policy(concurrency=1, retries=3, retry_delay=0.02))
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


# -- a failed leg keeps the calls it already made --------------------------------------


def _one_row_fails_at_once(rows: int, concurrency: int, **policy: Any) -> tuple[CompiledGraph, list[str], threading.Event]:
    """Row 0 fails at once; every other row waits for ``release`` before it counts its call."""
    calls: list[str] = []
    release = threading.Event()

    @node("mail", title="M", description="d", category="test", policy=Policy(concurrency=concurrency, **policy))
    def mail(text: In = Txt("")) -> Out:
        calls.append(str(text))
        if text == "0":
            raise ValueError("this row can't be sent")
        release.wait(10)
        return Txt(text)

    return _per_row(mail, rows), calls, release


def _run_to_the_end_releasing_on_failure(compiled: CompiledGraph, release: threading.Event, state: RunState | None = None) -> tuple[list[Any], RunState]:
    """A leg that lets the waiting calls go only when it has reported the failure."""
    async def go() -> list[Any]:
        seen = []
        async with compiled.stream(state) as events:
            async for event in events:
                seen.append(event)
                if event.type == "node_failed":
                    release.set()
        return seen

    seen = asyncio.run(go())
    return seen, seen[-1].state


def test_one_row_failing_at_concurrency_eight_keeps_the_seven_calls_already_running():
    compiled, calls, release = _one_row_fails_at_once(rows=30, concurrency=8)

    events, state = _run_to_the_end_releasing_on_failure(compiled, release)

    assert state.status == "failed" and state.failure.type == "execution_failed"
    assert sorted(calls, key=int) == [str(i) for i in range(8)], "no unit starts after the failure"
    assert [row[0] for row, _ in state.nodes["slow"].finished_units] == [1, 2, 3, 4, 5, 6, 7]
    assert compiled.results(state)["slow"]["result"]["values"] == [str(i) for i in range(1, 8)]
    assert events[-1].type == "run_end" and "node_failed" in [event.type for event in events]


def test_a_resume_after_a_failed_row_does_not_call_the_seven_again():
    compiled, calls, release = _one_row_fails_at_once(rows=30, concurrency=8)
    _, failed = _run_to_the_end_releasing_on_failure(compiled, release)
    calls.clear()

    state = compiled.run_sync(failed)

    assert "0" in calls, "the failed row runs again"
    assert not set(calls) & {str(i) for i in range(1, 8)}, "the seven kept calls are not made again"
    assert state.status == "failed", "row 0 fails on every leg"
    assert [row[0] for row, _ in state.nodes["slow"].finished_units][:7] == [1, 2, 3, 4, 5, 6, 7]


def test_a_row_failing_at_concurrency_one_starts_nothing_more():
    compiled, calls, release = _one_row_fails_at_once(rows=5, concurrency=1)

    _, state = _run_to_the_end_releasing_on_failure(compiled, release)
    time.sleep(0.2)

    assert calls == ["0"]
    assert state.status == "failed"
    assert not state.nodes["slow"].finished_units


def test_a_failed_leg_waits_for_a_call_no_longer_than_its_attempt_timeout():
    release = threading.Event()

    @node("hangs-or-fails", title="H", description="d", category="test", policy=Policy(concurrency=2, attempt_timeout=0.2))
    def hangs_or_fails(text: In = Txt("")) -> Out:
        if text == "0":
            raise ValueError("no")
        release.wait(10)
        return Txt(text)

    started = time.monotonic()
    state = _per_row(hangs_or_fails, 2).run_sync()
    took = time.monotonic() - started
    release.set()

    assert state.status == "failed" and state.failure.type == "execution_failed", "the first failure stands"
    assert 0.15 < took < 1.0, f"the failed leg took {took:.2f}s"


def test_an_ask_or_a_second_failure_while_a_failed_leg_drains():
    """A person's question is kept like any finished call; a later failure is only in the log."""
    release = threading.Event()

    @node("mixed", title="M", description="d", category="test", policy=Policy(concurrency=4))
    def mixed(text: In = Txt("")) -> Out | AskPerson:
        if text == "0":
            raise ValueError("first")
        release.wait(10)
        if text == "1":
            return AskPerson()
        if text == "2":
            raise ValueError("second")
        return Txt(text)

    events, state = _run_to_the_end_releasing_on_failure(_per_row(mixed, 4), release)

    assert state.status == "failed" and state.failure.loc[1] == 0, "the first failure"
    assert [event.type for event in events].count("node_failed") == 1
    assert [unit.row for unit in state.nodes["slow"].waiting] == [(1,)]
    assert [row for row, _ in state.nodes["slow"].finished_units] == [(3,)]


def test_a_retry_is_not_made_after_the_leg_has_failed():
    release = threading.Event()
    calls: list[str] = []

    @node("flaky-beside-a-failure", title="F", description="d", category="test", policy=Policy(concurrency=2, retries=3, retry_delay=0.01))
    def flaky(text: In = Txt("")) -> Out:
        calls.append(str(text))
        if text == "0":
            raise ValueError("no")
        release.wait(10)
        raise ExternalFailure("try again")

    events, state = _run_to_the_end_releasing_on_failure(_per_row(flaky, 2), release)
    time.sleep(0.1)

    assert state.failure.type == "execution_failed"
    assert sorted(calls) == ["0", "1"] and "node_retry" not in [event.type for event in events]


def test_a_cancel_while_a_failed_leg_drains_ends_it_failed_at_once():
    cancel = asyncio.Event()
    compiled, _, release = _one_row_fails_at_once(rows=4, concurrency=4)

    async def go() -> tuple[RunState, float]:
        cancel_at = 0.0
        async with compiled.stream(cancel=cancel) as events:
            async for event in events:
                if event.type == "node_failed":
                    cancel_at = time.monotonic()
                    cancel.set()
                if event.type == "run_end":
                    return event.state, time.monotonic() - cancel_at
        raise AssertionError("no run_end")

    state, took = asyncio.run(go())
    release.set()

    assert state.status == "failed" and took < 0.2


# -- the event loop stays the host's ------------------------------------------------


def _leg_where_every_unit_is_reused(rows: int) -> tuple[CompiledGraph, RunState]:
    """A graph with its source edited and the state of a first leg: every row of ``m`` is reused."""
    @node("many", title="S", description="d", category="test")
    def many(tag: Annotated[int, Param(title="C")] = 0) -> Parts:
        return [Txt(str(n)) for n in range(rows)]

    @node("echo", title="E", description="d", category="test")
    def echo(text: Annotated[Txt, Param(title="T")]) -> Out:
        return text

    registry = NodeRegistry([many, echo])

    def graph(tag: int) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[
            GraphNode(id="s", type="many", version=1, bindings={"tag": Static(tag)}),
            GraphNode(id="m", type="echo", version=1, bindings={"text": From("s.result")}),
        ]), registry)

    return graph(1), graph(0).run_sync()


def test_another_coroutine_gets_turns_during_a_leg_of_thousands_of_units_that_need_no_call():
    """A unit that needs no thread never lets the event loop go on its own. The
    bound counts turns, not milliseconds, and a slower machine only gives more."""
    edited, earlier = _leg_where_every_unit_is_reused(20_000)
    turns = 0
    running = True

    async def go() -> RunState:
        nonlocal turns, running

        async def other_coroutine() -> None:
            nonlocal turns
            while running:
                await asyncio.sleep(0)
                turns += 1

        beside = asyncio.create_task(other_coroutine())
        ending = None
        async with edited.stream(earlier) as events:
            async for event in events:
                if event.type == "node_start" and event.node_id == "m":
                    turns = 0  # the source's thread is back: from here no unit needs one
                if event.type == "run_end":
                    ending = event.state
        running = False
        await beside
        assert ending is not None
        return ending

    state = asyncio.run(go())

    assert state.status == "complete"
    assert state.nodes["m"].reused_units == 20_000
    assert turns >= 10, turns


def test_a_cancel_from_a_coroutine_lands_during_a_leg_that_needs_no_thread():
    edited, earlier = _leg_where_every_unit_is_reused(20_000)
    cancel = asyncio.Event()

    async def go() -> RunState:
        async def cancel_after_a_few_turns() -> None:
            for _ in range(3):
                await asyncio.sleep(0)
            cancel.set()

        beside = asyncio.create_task(cancel_after_a_few_turns())
        state = await edited.run(earlier, cancel=cancel)
        await beside
        return state

    state = asyncio.run(go())

    assert state.status == "cancelled"
    assert state.nodes["m"].reused_units < 20_000


def test_a_call_returning_after_a_cancel_changes_nothing_and_no_event_follows_the_end():
    release = threading.Event()

    @node("late", title="L", description="d", category="test", policy=Policy(concurrency=2))
    def late(text: In = Txt("")) -> Out:
        release.wait(10)
        return Txt(text)

    compiled = _per_row(late, 2)
    cancel = asyncio.Event()

    async def go() -> list[Any]:
        seen: list[Any] = []
        async with compiled.stream(cancel=cancel) as events:
            async for event in events:
                seen.append(event)
                if event.type == "node_start" and event.node_id == "slow":
                    cancel.set()
            release.set()
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)
        return seen

    seen = asyncio.run(go())

    assert [event.type for event in seen].count("run_end") == 1 and seen[-1].type == "run_end"
    assert seen[-1].state.status == "cancelled"
    assert seen[-1].state.nodes["slow"].finished_units in (None, [])


def test_a_cancel_records_what_threads_have_already_handed_back():
    """A call that has returned by the moment a cancel lands is kept, as any finished call is."""
    release = threading.Event()
    calls: list[str] = []

    @node("fast-then-cancel", title="F", description="d", category="test", policy=Policy(concurrency=4))
    def fast(text: In = Txt("")) -> Out:
        calls.append(str(text))
        if text == "3":
            release.wait(10)
        return Txt(text)

    compiled = _per_row(fast, 4)
    cancel = asyncio.Event()

    async def go() -> RunState:
        ending: RunState | None = None
        async with compiled.stream(cancel=cancel) as events:
            async for event in events:
                if event.type == "node_start" and event.node_id == "slow":
                    while len(calls) < 4:
                        await asyncio.sleep(0.01)
                    await asyncio.sleep(0.05)  # the three fast calls have long been handed back
                    cancel.set()
                if event.type == "run_end":
                    ending = event.state
        assert ending is not None
        return ending

    state = asyncio.run(go())
    release.set()

    assert state.status == "cancelled"
    assert sorted(row[0] for row, _ in state.nodes["slow"].finished_units) == [0, 1, 2]
