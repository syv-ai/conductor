"""How a run goes: what its events say, how a leg ends (done, failed, cancelled, timed out), and what a saved run keeps.

A leg starts every unit whose inputs exist and records what comes of each.
``run.stream()`` says so as events, the last one ``run_end``; leaving the
loop, cancelling the task that reads it, or setting ``cancel`` ends the leg.
A node's ``Policy`` decides how it is called: on the leg's own thread one item
at a time, or on threads, with retries and a clock on each attempt.
"""

import asyncio
import json
import math
import threading
import time
from collections.abc import Callable
from typing import Any, ClassVar, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    FromRun,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    NotCalled,
    Policy,
    Series,
    Table,
    make_graph,
    node,
    stored_graph,
)
from pydantic import BaseModel, model_serializer

from tests.test_core.nodes import place, registry_of

#: The longest any test here waits for a leg, a thread or an event.
BOUND = 10.0


@node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",") if text else []


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


def _numbers(count: int) -> str:
    return ",".join(str(i) for i in range(count))


def _per_item(definition: Any, items: int, *more: Any) -> Any:
    """``split`` making ``items`` items, read by ``definition`` as node ``slow``."""
    return registry_of(split, definition, *more).load(GraphRecord(
        nodes=[NodeRecord(id="split", node="split", values={"text": _numbers(items)}), NodeRecord(id="slow", node=definition.id)],
        edges=[("split.result", "slow.text")],
    ))


def _events(run: Any, **kwargs: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream(**kwargs)]

    return asyncio.run(asyncio.wait_for(collect(), BOUND))


def _bounded(coroutine: Any) -> Any:
    return asyncio.run(asyncio.wait_for(coroutine, BOUND))


def _settle(calls: list[Any]) -> tuple[int, int]:
    """How many calls there were, and how many a moment later: equal once nothing runs any more."""
    time.sleep(0.3)
    seen = len(calls)
    time.sleep(0.3)
    return seen, len(calls)


# -- what the events say ---------------------------------------------------------------------


@node("addresses", title="Addresses", description="For a test.")
def addresses(text: str) -> Series[str | None]:
    return [part or None for part in text.split(",")]


@node("send", title="Send", description="For a test.")
def send(email: str) -> str:
    return f"sent {email}"


def test_node_complete_counts_the_items_a_blank_kept_the_node_from_and_a_saved_run_names_them():
    graph = registry_of(addresses, send).load(make_graph(addresses, send))

    events = _events(graph.with_inputs(text="ida@a.dk,,al@a.dk"))

    (mail,) = [e for e in events if e.type == "node_complete" and e.loc[0] == "send"]
    assert (mail.units, mail.not_called, mail.reused_units) == (3, 1, 0)
    loaded = graph.load_run(events[-1].run.to_json())
    assert loaded.not_called == [NotCalled(loc=("send", 1), input="email")]
    assert loaded.results["send"]["result"] == ["sent ida@a.dk", None, "sent al@a.dk"]
    assert loaded.status == "complete"


def test_a_diamond_runs_each_branch_once_and_joins_them():
    calls: list[str] = []

    @node("source", title="Source", description="For a test.")
    def source(text: str) -> str:
        calls.append("source")
        return text

    @node("left", title="Left", description="For a test.")
    def left(text: str) -> str:
        calls.append("left")
        return text.upper()

    @node("right", title="Right", description="For a test.")
    def right(text: str) -> str:
        calls.append("right")
        return text[::-1]

    @node("meet", title="Meet", description="For a test.")
    def meet(a: str, b: str) -> str:
        return f"{a}|{b}"

    graph = registry_of(source, left, right, meet).load(GraphRecord(
        nodes=[NodeRecord(id="s", node="source"), NodeRecord(id="l", node="left"), NodeRecord(id="r", node="right"), NodeRecord(id="m", node="meet")],
        edges=[("s.result", "l.text"), ("s.result", "r.text"), ("l.result", "m.a"), ("r.result", "m.b")],
    ))

    events = _events(graph.with_inputs(text="abc"))

    assert events[-1].run.outputs == {"m.result": "ABC|cba"}
    assert sorted(calls) == ["left", "right", "source"]
    assert [e.loc[0] for e in events if e.type == "node_complete"][-1] == "m"


def test_a_failed_item_fails_the_node_naming_the_item_and_the_gather_below_never_starts():
    @node("picky", title="Picky", description="For a test.")
    def picky(text: str) -> str:
        if text == "1":
            raise ValueError("not this one")
        return text

    graph = registry_of(split, picky, join).load(make_graph(split, picky, join, values={"split.text": "0,1,2"}))

    events = _events(graph.with_inputs())

    (failed,) = [e for e in events if e.type == "node_failed"]
    assert (failed.loc[0], failed.cause.type, failed.cause.loc) == ("picky", "execution_failed", ("picky", 1))
    assert not [e for e in events if getattr(e, "loc", None) == ("join",)]
    assert events[-1].run.status == "failed" and events[-1].run.results["join"]["result"] is None


def test_outputs_leave_out_a_skipped_output_and_a_failed_run_keeps_what_it_made():
    @node("only-short", title="Only short", description="For a test.")
    def only_short(text: str) -> str:
        return text if len(text) < 3 else SKIPPED

    @node("fails-on-c", title="Fails on c", description="For a test.")
    def fails_on_c(text: str) -> str:
        if text == "c":
            raise ValueError("no c")
        return text

    skipping = registry_of(only_short).load(GraphRecord(nodes=[NodeRecord(id="o", node="only-short")]))
    assert skipping.run(text="long").outputs == {"o.result": None}
    assert skipping.run(text="ab").outputs == {"o.result": "ab"}

    failing = registry_of(split, fails_on_c).load(make_graph(split, fails_on_c))
    failed = failing.run(text="a,b,c")
    assert failed.status == "failed" and failed.results["fails-on-c"]["result"] == ["a", "b", None]


def test_progress_counts_items_between_start_and_complete_with_the_total_once_the_items_are_known():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    events = _events(_per_item(echo, 3).with_inputs())

    kinds = [(e.type, getattr(e, "loc", None)) for e in events]
    progress = [e for e in events if e.type == "node_progress" and e.loc[0] == "slow"]
    assert progress and all(e.total == 3 for e in progress) and progress[-1].done >= 1
    assert kinds.index(("node_start", ("slow",))) < kinds.index(("node_progress", ("slow",))) < kinds.index(("node_complete", ("slow",)))


def test_a_fast_node_over_many_items_reports_progress_a_few_times_a_second_ending_on_its_last_count():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    started = time.monotonic()
    events = _events(_per_item(echo, 20_000).with_inputs())
    took = time.monotonic() - started

    progress = [e for e in events if e.type == "node_progress" and e.loc[0] == "slow"]
    assert len(progress) <= took / 0.25 + 2
    assert (progress[-1].done, progress[-1].total) == (20_000, 20_000)
    complete = next(i for i, e in enumerate(events) if e.type == "node_complete" and e.loc[0] == "slow")
    assert events.index(progress[-1]) < complete



def test_a_leg_that_ends_waiting_reports_the_last_count_of_the_node_still_waiting():
    @node("maybe", title="Maybe", description="For a test.")
    def maybe(text: str) -> str:
        if text == "ask":
            raise Ask("Is this right?")
        return text

    graph = registry_of(split, maybe).load(make_graph(split, maybe, values={"split.text": "ask,b,c"}))

    events = _events(graph.with_inputs())

    last = [e for e in events if e.type == "node_progress" and e.loc[0] == "maybe"][-1]
    assert (last.done, last.total) == (2, 3)
    assert events[-1].status == "waiting"


def test_a_node_above_items_that_were_never_made_reports_no_progress():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    events = _events(_per_item(echo, 0).with_inputs())

    assert not [e for e in events if e.type == "node_progress" and e.loc[0] == "slow"]
    assert events[-1].status == "complete" and events[-1].run.results["slow"]["result"] == []


# -- the policy a node is called by ------------------------------------------------------------


def test_the_policy_of_the_version_the_graph_is_pinned_at_is_the_one_a_run_obeys():
    calls: list[int] = []

    @node("fetch", title="Fetch one", description="For a test.", policy=Policy(retries=2, retry_delay=0.01))
    def fetch_one(text: str) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ExternalError("try again")
        return "one"

    @node("fetch", title="Fetch two", description="For a test.", replaces=fetch_one, policy=Policy(retries=0))
    def fetch_two(text: str) -> str:
        calls.append(2)
        raise ExternalError("never retried")

    registry = registry_of(fetch_two)
    pinned = registry.load(GraphRecord(nodes=[NodeRecord(id="f", node="fetch", version=1)]))
    newest = registry.load(GraphRecord(nodes=[NodeRecord(id="f", node="fetch", version=2)]))

    assert pinned.run(text="x").outputs == {"f.result": "one"} and calls == [1, 1, 1]
    calls.clear()
    assert newest.run(text="x").status == "failed" and calls == [2]


def test_concurrency_one_calls_one_item_at_a_time():
    in_flight, most = 0, 0
    lock = threading.Lock()

    @node("counted", title="Counted", description="For a test.")
    def counted(text: str) -> str:
        nonlocal in_flight, most
        with lock:
            in_flight += 1
            most = max(most, in_flight)
        time.sleep(0.005)
        with lock:
            in_flight -= 1
        return text

    assert _per_item(counted, 10).run().status == "complete"
    assert most == 1


def test_a_leg_of_nodes_at_concurrency_one_makes_no_threads_of_its_own():
    """Called in order on the leg's own thread: a 100-node chain over 20 items runs on one thread."""
    seen: set[int] = set()

    @node("step", title="Step", description="For a test.")
    def step(text: str) -> str:
        seen.add(threading.get_ident())
        return text

    nodes = [NodeRecord(id="s", node="split", values={"text": _numbers(20)})] + [NodeRecord(id=f"n{i}", node="step") for i in range(100)]
    edges = [("s.result", "n0.text")] + [(f"n{i - 1}.result", f"n{i}.text") for i in range(1, 100)]
    graph = registry_of(split, step).load(GraphRecord(nodes=nodes, edges=edges))

    assert graph.run().status == "complete"
    assert seen == {threading.get_ident()}


def test_the_retry_delay_doubles_each_time_and_another_item_takes_the_slot_while_a_retry_waits():
    tries: dict[str, int] = {}
    finished: list[str] = []

    @node("flaky", title="Flaky", description="For a test.", policy=Policy(concurrency=1, retries=3, retry_delay=0.02))
    def flaky(text: str) -> str:
        tries[text] = tries.get(text, 0) + 1
        if text == "flaky" and tries[text] <= 3:
            raise ExternalError("try again")
        finished.append(text)
        return text

    graph = registry_of(split, flaky).load(make_graph(split, flaky, values={"split.text": "flaky,steady"}))

    events = _events(graph.with_inputs())

    assert [e.delay for e in events if e.type == "node_retry"] == [0.02, 0.04, 0.08]
    assert finished == ["steady", "flaky"]


def test_an_instant_node_beside_forty_slow_items_does_not_time_out():
    """An attempt's clock starts when its call does, never while it waits for a thread."""
    calls: list[str] = []

    @node("sleeper", title="Sleeper", description="For a test.", policy=Policy(concurrency=40))
    def sleeper(text: str) -> str:
        calls.append(text)
        time.sleep(0.5)
        return text

    @node("instant", title="Instant", description="For a test.", policy=Policy(attempt_timeout=0.1))
    def instant(texts: Series[str]) -> str:
        return "now"

    graph = registry_of(split, sleeper, instant).load(GraphRecord(
        nodes=[NodeRecord(id="split", node="split", values={"text": _numbers(40)}), NodeRecord(id="slow", node="sleeper"), NodeRecord(id="i", node="instant")],
        edges=[("split.result", "slow.text"), ("split.result", "i.texts")],
    ))

    run = graph.run()

    assert run.status == "complete" and run.outputs["i.result"] == "now" and len(calls) == 40


def test_units_recorded_as_threads_return_are_read_in_item_order_live_in_a_checkpoint_and_after_a_load():
    """Calls on threads finish when they finish; what a run says of a node's units is in item order all the same."""

    @node("first-is-last", title="First is last", description="For a test.", policy=Policy(concurrency=4))
    def first_is_last(text: str) -> str | None:
        if text == "0":
            time.sleep(0.3)  # the first item returns after every other
        return None

    @node("needs-text", title="Needs text", description="For a test.")
    def needs_text(text: str) -> str:
        return text

    graph = registry_of(split, first_is_last, needs_text).load(GraphRecord(
        nodes=[NodeRecord(id="split", node="split", values={"text": _numbers(4)}), NodeRecord(id="slow", node="first-is-last"), NodeRecord(id="after", node="needs-text")],
        edges=[("split.result", "slow.text"), ("slow.result", "after.text")],
    ))
    in_item_order = [("after", 0), ("after", 1), ("after", 2), ("after", 3)]

    async def streamed() -> list[Any]:
        return [event async for event in graph.with_inputs().stream(checkpoint_every=0.0001)]

    events = asyncio.run(asyncio.wait_for(streamed(), BOUND))
    run = events[-1].run
    checkpoints = [event.run for event in events if event.type == "run_checkpoint"]

    assert [call.loc for call in run.not_called] == in_item_order
    assert [call.loc for call in graph.load_run(run.to_json()).not_called] == in_item_order
    assert checkpoints and all([call.loc for call in held.not_called] == sorted(call.loc for call in held.not_called) for held in checkpoints)
    assert any(len(held.not_called) > 1 for held in checkpoints)  # some checkpoint held the later items before the first


@pytest.mark.parametrize("items", [None, 3])
def test_a_reader_made_ready_by_each_of_two_nodes_it_reads_is_called_once(items: int | None):
    """A call that comes back from a thread makes its readers ready; a unit made ready twice is still one unit."""
    calls: list[tuple[str, str]] = []

    @node("from-a-thread", title="From a thread", description="For a test.", policy=Policy(concurrency=2))
    def from_a_thread(text: str) -> str:
        time.sleep(0.01)
        return text

    @node("left", title="Left", description="For a test.")
    def left(text: str) -> str:
        return text + "a"

    @node("right", title="Right", description="For a test.")
    def right(text: str) -> str:
        return text + "b"

    @node("both", title="Both", description="For a test.")
    def both(left: str, right: str) -> str:
        calls.append((left, right))
        return left + right

    first = [NodeRecord(id="s", node="from-a-thread", values={"text": "x"})] if items is None else [
        NodeRecord(id="split", node="split", values={"text": _numbers(items)}), NodeRecord(id="s", node="from-a-thread"),
    ]
    graph = registry_of(split, from_a_thread, left, right, both).load(GraphRecord(
        nodes=[*first, NodeRecord(id="a", node="left"), NodeRecord(id="b", node="right"), NodeRecord(id="r", node="both")],
        edges=[*([] if items is None else [("split.result", "s.text")]), ("s.result", "a.text"), ("s.result", "b.text"), ("a.result", "r.left"), ("b.result", "r.right")],
    ))

    run = graph.run()

    assert run.status == "complete"
    assert sorted(calls) == ([("xa", "xb")] if items is None else [(f"{i}a", f"{i}b") for i in range(items)])


@pytest.mark.parametrize("concurrency", [1, 4])
@pytest.mark.parametrize("shape", ["chain", "per_item_with_a_gather"])
def test_one_leg_starts_everything_its_own_units_make_ready_and_a_further_resume_calls_nothing(shape: str, concurrency: int):
    """A leg tries each node after the nodes it reads, so one leg finishes a graph with nobody to ask, whether
    the first node is called on the leg's own thread or on others."""
    calls: list[str] = []

    @node("first", title="First", description="For a test.", policy=Policy(concurrency=concurrency))
    def first(text: str) -> str:
        calls.append(f"first {text}")
        return text

    @node("then", title="Then", description="For a test.")
    def then(text: str) -> str:
        calls.append(f"then {text}")
        return text + "!"

    @node("last", title="Last", description="For a test.")
    def last(texts: Series[str]) -> str:
        calls.append("last")
        return "+".join(texts)

    if shape == "chain":
        nodes = [NodeRecord(id="f", node="first", values={"text": "x"}), *(NodeRecord(id=f"t{i}", node="then") for i in range(5))]
        edges = [("f.result", "t0.text"), *((f"t{i}.result", f"t{i + 1}.text") for i in range(4))]
        expected, outputs = 6, {"t4.result": "x!!!!!"}
    else:
        nodes = [NodeRecord(id="split", node="split", values={"text": _numbers(4)}), NodeRecord(id="f", node="first"), NodeRecord(id="t", node="then"), NodeRecord(id="l", node="last")]
        edges = [("split.result", "f.text"), ("f.result", "t.text"), ("t.result", "l.texts")]
        expected, outputs = 9, {"l.result": "0!+1!+2!+3!"}
    graph = registry_of(split, first, then, last).load(GraphRecord(nodes=nodes, edges=edges))

    run = graph.run()

    assert run.status == "complete" and run.outputs == outputs and len(calls) == expected
    assert all(value is not None for values in run.results.values() for value in values.values())
    assert run.resume().outputs == outputs and graph.load_run(run.to_json()).resume().outputs == outputs and len(calls) == expected


def test_concurrency_one_with_an_attempt_timeout_never_has_two_calls_in_flight():
    """A timed-out attempt is final and its thread keeps the node's slot until it returns."""
    in_flight, most = 0, 0
    lock = threading.Lock()

    @node("slow-items", title="Slow items", description="For a test.", policy=Policy(concurrency=1, attempt_timeout=0.2, retries=2, retry_delay=0.01))
    def slow_items(text: str) -> str:
        nonlocal in_flight, most
        with lock:
            in_flight += 1
            most = max(most, in_flight)
        try:
            time.sleep(0.5)
        finally:
            with lock:
                in_flight -= 1
        return text

    events = _events(_per_item(slow_items, 3).with_inputs())
    time.sleep(0.6)  # the abandoned call returns

    assert most == 1
    assert not [e for e in events if e.type == "node_retry"]
    assert events[-1].run.status == "failed" and events[-1].run.state.cause.type == "timeout"


def test_a_run_has_no_deadline_by_default_and_a_slow_node_runs_to_the_end():
    @node("slow-second", title="Slow second", description="For a test.")
    def slow_second(text: str) -> str:
        time.sleep(1.0)
        return "done"

    graph = registry_of(slow_second).load(GraphRecord(nodes=[NodeRecord(id="s", node="slow-second")]))

    assert graph.run(text="x").outputs == {"s.result": "done"}


# -- what a node may raise ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "raised",
    [asyncio.CancelledError(), GeneratorExit(), BaseException("b"), StopIteration()],
    ids=lambda raised: type(raised).__name__,
)
def test_anything_a_node_raises_fails_its_unit_and_the_leg_ends(raised):
    @node("raises", title="Raises", description="For a test.")
    def raises(text: str) -> str:
        raise raised

    graph = registry_of(raises).load(GraphRecord(nodes=[NodeRecord(id="r", node="raises")]))

    run = graph.run(text="x")
    streamed = _events(graph.with_inputs(text="x"))[-1].run

    for ended in (run, streamed):
        assert (ended.status, ended.state.cause.type, ended.state.cause.loc) == ("failed", "execution_failed", ("r",))


def test_a_defect_in_the_engine_reaches_the_caller_rather_than_hanging_the_stream(monkeypatch):
    from conductor._internal.run.call_inputs import ItemReader

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    graph = registry_of(echo).load(GraphRecord(nodes=[NodeRecord(id="e", node="echo")]))
    prepared = graph.with_inputs(text="x")

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("an engine defect")

    monkeypatch.setattr(ItemReader, "comes_to", broken)
    with pytest.raises(RuntimeError, match="an engine defect"):
        _events(prepared)


# -- a failed leg keeps the calls it already made ----------------------------------------------


def _item_zero_fails_at_once(concurrency: int, **policy: Any) -> tuple[Any, list[str], threading.Event]:
    """``mail`` as a node: item 0 fails at once, every other item waits for ``release``."""
    calls: list[str] = []
    release = threading.Event()

    @node("mail", title="Mail", description="For a test.", policy=Policy(concurrency=concurrency, **policy))
    def mail(text: str) -> str:
        calls.append(text)
        if text == "0":
            raise ValueError("this item can't be sent")
        release.wait(BOUND)
        return text

    return mail, calls, release


def _releasing_on_failure(prepared: Any, release: threading.Event, cancel: asyncio.Event | None = None) -> list[Any]:
    async def collect() -> list[Any]:
        seen = []
        async for event in prepared.stream(cancel=cancel):
            seen.append(event)
            if event.type == "node_failed":
                release.set()
        return seen

    try:
        return _bounded(collect())
    finally:
        release.set()


def test_an_item_failing_at_concurrency_one_starts_no_later_item():
    mail, calls, release = _item_zero_fails_at_once(concurrency=1)

    events = _releasing_on_failure(_per_item(mail, 5).with_inputs(), release)

    assert _settle(calls) == (1, 1) and calls == ["0"]
    assert events[-1].run.status == "failed" and events[-1].run.results["slow"]["result"] == [None] * 5


def test_a_failed_leg_waits_for_a_call_still_running_no_longer_than_its_attempt_timeout():
    release = threading.Event()

    @node("hangs-or-fails", title="Hangs or fails", description="For a test.", policy=Policy(concurrency=2, attempt_timeout=0.2))
    def hangs_or_fails(text: str) -> str:
        if text == "0":
            raise ValueError("no")
        release.wait(BOUND)
        return text

    started = time.monotonic()
    try:
        run = _per_item(hangs_or_fails, 2).run()
    finally:
        release.set()
    took = time.monotonic() - started

    assert run.status == "failed" and run.state.cause.type == "execution_failed", "the first failure stands"
    assert 0.15 < took < 1.0, took


def test_an_ask_or_a_second_failure_while_a_failed_leg_drains_is_kept_or_logged():
    """A person's question is kept like any finished call; a second failure goes to the log, not onto the run."""
    release = threading.Event()

    @node("mixed", title="Mixed", description="For a test.", policy=Policy(concurrency=4))
    def mixed(text: str) -> str:
        if text == "0":
            raise ValueError("first")
        release.wait(BOUND)
        if text == "1":
            raise Ask("Send it?")
        if text == "2":
            raise ValueError("second")
        return text

    events = _releasing_on_failure(_per_item(mixed, 4).with_inputs(), release)

    run = events[-1].run
    assert run.status == "failed" and run.state.cause.loc == ("slow", 0)
    assert [e.type for e in events].count("node_failed") == 1
    assert [place(w) for w in run.waiting] == ["slow@1"]
    assert run.results["slow"]["result"] == [None, None, None, "3"]


def test_no_retry_starts_once_the_leg_has_failed():
    release = threading.Event()
    calls: list[str] = []

    @node("flaky-beside-a-failure", title="Flaky", description="For a test.", policy=Policy(concurrency=2, retries=3, retry_delay=0.01))
    def flaky(text: str) -> str:
        calls.append(text)
        if text == "0":
            raise ValueError("no")
        release.wait(BOUND)
        raise ExternalError("try again")

    events = _releasing_on_failure(_per_item(flaky, 2).with_inputs(), release)

    assert _settle(calls)[0] == 2 and sorted(calls) == ["0", "1"]
    assert events[-1].run.state.cause.type == "execution_failed" and not [e for e in events if e.type == "node_retry"]


# -- leaving, cancelling, timing out -----------------------------------------------------------


def _sleeper(seconds: float, calls: list[str]) -> Any:
    @node("sleeper", title="Sleeper", description="For a test.")
    def sleeper(text: str) -> str:
        calls.append(text)
        time.sleep(seconds)
        return text

    return sleeper


def test_closing_the_stream_after_one_event_stops_the_rest():
    calls: list[str] = []
    prepared = _per_item(_sleeper(0.05, calls), 20).with_inputs()

    async def go() -> None:
        events = prepared.stream()
        async for event in events:
            if event.type == "node_start" and event.loc[0] == "slow":
                break
        await events.aclose()  # type: ignore[attr-defined]

    _bounded(go())
    seen, later = _settle(calls)

    assert later == seen and seen <= 2, seen


def test_cancelling_the_task_that_reads_the_stream_stops_the_rest():
    calls: list[str] = []
    prepared = _per_item(_sleeper(0.05, calls), 20).with_inputs()

    async def go() -> None:
        started = asyncio.Event()

        async def consume() -> None:
            async for event in prepared.stream():
                if event.type == "node_start" and event.loc[0] == "slow":
                    started.set()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), BOUND)
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)

    _bounded(go())
    seen, later = _settle(calls)

    assert later == seen and seen <= 2, seen


def test_a_cancel_set_mid_run_ends_the_leg_cancelled_at_once():
    calls: list[str] = []
    prepared = _per_item(_sleeper(0.05, calls), 20).with_inputs()

    async def go() -> tuple[list[Any], float]:
        cancel, seen, set_at = asyncio.Event(), [], 0.0
        async for event in prepared.stream(cancel=cancel):
            seen.append(event)
            if event.type == "node_start" and event.loc[0] == "slow":
                cancel.set()
                set_at = time.monotonic()
        return seen, time.monotonic() - set_at

    seen, after = _bounded(go())

    assert seen[-1].type == "run_end" and seen[-1].status == "cancelled"
    assert not [e for e in seen if e.type == "node_complete" and e.loc[0] == "slow"]
    assert after < 0.3 and len(calls) < 20, (after, len(calls))


def test_a_cancel_set_while_a_leg_reuses_earlier_results_ends_the_leg_before_the_remaining_items():
    """The second run calls ``echo`` for no item: each one's earlier result is reused, which reads the item to
    compare it with the earlier one. The test sets the cancel from inside the tenth such read, and holds the
    leg there until the event loop has set it, so the cancel is set while 90 items are still to go."""
    reads: list[Callable[[], None]] = []

    class Line(BaseModel):
        type_id: ClassVar[str] = "Line"
        type_title: ClassVar[str] = "Line"
        text: str

        @model_serializer
        def written(self) -> dict[str, str]:
            for read in reads:
                read()
            return {"text": self.text}

    @node("echo", title="Echo", description="For a test.")
    def echo(line: Line) -> str:
        return line.text

    @node("source", title="Source", description="For a test.")
    def source(tag: int = 0) -> Series[Line]:
        return [Line(text=str(n)) for n in range(100)]

    graph = registry_of(source, echo).load(make_graph(source, echo))
    prepared = graph.run(tag=0).with_inputs(tag=1)

    async def go() -> Any:
        loop, cancel, cancel_is_set = asyncio.get_running_loop(), asyncio.Event(), threading.Event()
        read_so_far = 0

        def set_the_cancel() -> None:
            cancel.set()
            cancel_is_set.set()

        def cancel_at_the_tenth_read() -> None:
            nonlocal read_so_far
            read_so_far += 1
            if read_so_far == 10:
                loop.call_soon_threadsafe(set_the_cancel)
                assert cancel_is_set.wait(5)

        reads.append(cancel_at_the_tenth_read)
        last = None
        async for event in prepared.stream(cancel=cancel):
            last = event
        return last

    ended = _bounded(go())

    assert ended.status == "cancelled"
    assert len([r for r in ended.run.results["echo"]["result"] if r is not None]) == 10  # the item being read, and none after it


def test_exactly_one_run_end_comes_last_when_a_call_returns_after_a_cancel():
    release = threading.Event()

    @node("late", title="Late", description="For a test.", policy=Policy(concurrency=2))
    def late(text: str) -> str:
        release.wait(BOUND)
        return text

    prepared = _per_item(late, 2).with_inputs()

    async def go() -> list[Any]:
        cancel, seen = asyncio.Event(), []
        async for event in prepared.stream(cancel=cancel):
            seen.append(event)
            if event.type == "node_start" and event.loc[0] == "slow":
                cancel.set()
                release.set()
        await asyncio.sleep(0.1)
        return seen

    try:
        seen = _bounded(go())
    finally:
        release.set()

    assert [e.type for e in seen].count("run_end") == 1 and seen[-1].type == "run_end"
    assert seen[-1].status == "cancelled"


def test_a_cancel_during_a_retry_delay_ends_the_leg_before_every_attempt_has_run():
    calls: list[int] = []

    @node("flaky", title="Flaky", description="For a test.", policy=Policy(retries=5, retry_delay=0.2))
    def flaky(text: str) -> str:
        calls.append(1)
        raise ExternalError("try again")

    prepared = registry_of(flaky).load(GraphRecord(nodes=[NodeRecord(id="f", node="flaky")])).with_inputs(text="x")

    async def go() -> Any:
        cancel, last = asyncio.Event(), None
        async for event in prepared.stream(cancel=cancel):
            if event.type == "node_retry":
                cancel.set()
            last = event
        return last

    ended = _bounded(go())

    assert ended.status == "cancelled" and len(calls) < 6


# -- calling the door --------------------------------------------------------------------------


def test_a_stream_that_cannot_start_raises_at_the_call_before_any_event():
    class Caller:
        pass

    @node("signed", title="Signed", description="For a test.")
    def signed(text: str, who: FromRun[Caller]) -> str:
        return text

    prepared = registry_of(signed).load(GraphRecord(nodes=[NodeRecord(id="s", node="signed")])).with_inputs(text="x")

    with pytest.raises(InvalidCallError) as refused:
        prepared.stream()

    assert [p.type for p in refused.value.errors()] == ["from_run_missing"]
    assert refused.value.errors()[0].ctx == {"node_id": "s", "parameter": "who", "needs": "Caller", "given": []}


def test_a_type_no_node_needs_from_the_run_is_fine():
    class Caller:
        pass

    class Unused:
        pass

    @node("signed", title="Signed", description="For a test.")
    def signed(text: str, who: FromRun[Caller]) -> str:
        return text

    graph = registry_of(signed).load(GraphRecord(nodes=[NodeRecord(id="s", node="signed")]))

    assert graph.run(text="x", from_run={Caller: Caller(), Unused: Unused()}).outputs == {"s.result": "x"}


def test_running_a_graph_under_a_running_event_loop_ends_as_it_does_outside_one():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    graph = registry_of(split, echo, join).load(make_graph(split, echo, join))

    async def inside() -> Any:
        return graph.run(text="a,b")

    assert _bounded(inside()).outputs == graph.run(text="a,b").outputs == {"join.result": "a+b"}


# -- what a saved run keeps --------------------------------------------------------------------


def test_a_float_that_is_not_finite_inside_a_model_survives_a_saved_run_and_the_run_goes_on():
    class Reading(BaseModel):
        type_id: ClassVar[str] = "Reading"
        type_title: ClassVar[str] = "Reading"
        value: float
        low: float
        high: float

    @node("measure", title="Measure", description="For a test.")
    def measure(text: str) -> Reading:
        return Reading(value=math.nan, low=-math.inf, high=math.inf)

    @node("check", title="Check", description="For a test.")
    def check(reading: Reading) -> str:
        raise Ask(f"Is {reading.low}..{reading.high} right?")

    @node("width", title="Width", description="For a test.")
    def width(reading: Reading, checked: str) -> str:
        return f"{reading.low}..{reading.high} {checked}"

    graph = registry_of(measure, check, width).load(GraphRecord(
        nodes=[NodeRecord(id="m", node="measure"), NodeRecord(id="c", node="check"), NodeRecord(id="w", node="width")],
        edges=[("m.result", "c.reading"), ("m.result", "w.reading"), ("c.result", "w.checked")],
    ))
    paused = graph.run(text="x")

    loaded = graph.load_run(json.loads(paused.to_json()))

    reading = loaded.results["m"]["result"]
    assert math.isnan(reading.value) and (reading.low, reading.high) == (-math.inf, math.inf)
    assert loaded.answer("c", "yes").resume().outputs == {"w.result": "-inf..inf yes"}


class Contact(TypedDict):
    name: str
    email: str


@node("contact-records", title="Contact records", description="For a test.")
def contact_records(table: Table[Contact]) -> Series[Contact]:
    return table.to_dicts()


def test_a_table_missing_a_declared_column_at_run_time_fails_the_node_that_made_it_naming_the_column():
    @node("bare", title="Bare", description="For a test.")
    def bare(text: str) -> Table[Contact]:
        return pl.DataFrame({"name": ["ida"]})

    @node("greet", title="Greet", description="For a test.")
    def greet(email: str) -> str:
        return email

    graph = registry_of(bare, contact_records, greet).load(GraphRecord(
        nodes=[NodeRecord(id="b", node="bare"), NodeRecord(id="r", node="contact-records"), NodeRecord(id="g", node="greet")],
        edges=[("b.result", "r.table"), ("r.result.email", "g.email")],
    ))

    run = graph.run(text="x")

    assert run.status == "failed" and (run.state.cause.type, run.state.cause.loc) == ("invalid_output", ("b",))
    assert (run.state.cause.ctx["name"], run.state.cause.ctx["column"]) == ("result", "email")


def test_a_table_with_a_blank_cell_survives_a_saved_run():
    class Reachable(TypedDict):
        name: str
        email: str | None

    @node("contacts", title="Contacts", description="For a test.")
    def contacts(text: str) -> Table[Reachable]:
        return pl.DataFrame({"name": ["ida", "bo"], "email": ["ida@a.dk", None]})

    @node("records", title="Records", description="For a test.")
    def records(table: Table[Reachable]) -> Series[Reachable]:
        return table.to_dicts()

    graph = registry_of(contacts, records, send).load(GraphRecord(
        nodes=[NodeRecord(id="c", node="contacts"), NodeRecord(id="r", node="records"), NodeRecord(id="m", node="send")],
        edges=[("c.result", "r.table"), ("r.result.email", "m.email")],
    ))
    run = graph.run(text="x")

    loaded = graph.load_run(run.to_json())

    assert loaded.results["c"]["result"].to_dicts() == [{"name": "ida", "email": "ida@a.dk"}, {"name": "bo", "email": None}]
    assert loaded.not_called == [NotCalled(loc=("m", 1), input="email")]


def test_a_node_starts_before_any_other_event_about_it_when_its_items_are_taken_from_earlier_work():
    """A leg going on from an answered run takes the earlier items before calling the new one; ``node_start`` comes first."""

    @node("read", title="Read", description="For a test.")
    def read(text: str) -> Series[str]:
        return text.split(",")

    @node("approve", title="Approve", description="For a test.")
    def approve(name: str) -> bool:
        if name == "ida":
            return True
        raise Ask(f"Approve {name}?")

    graph = registry_of(read, approve).load(make_graph(read, approve))
    paused = graph.run(text="ida,bo")
    answered = paused.answer(paused.waiting[0], True).resume()

    events = _events(answered.with_inputs(text="ida,bo,eva"))
    about_approve = [e.type for e in events if getattr(e, "loc", None) == ("approve",)]
    assert about_approve[0] == "node_start", about_approve
    assert about_approve.count("node_start") == 1


def test_a_series_over_the_item_ceiling_coming_back_after_a_failure_leaves_the_failure_the_cause():
    @node("boom", title="Boom", description="For a test.", policy=Policy(concurrency=2))
    def boom(text: str) -> str:
        raise ValueError("no")

    @node("many", title="Many", description="For a test.", policy=Policy(concurrency=2))
    def many(text: str) -> Series[str]:
        time.sleep(0.2)
        return ["a", "b", "c", "d"]

    graph = registry_of(boom, many).load(GraphRecord(nodes=[NodeRecord(id="b", node="boom"), NodeRecord(id="m", node="many")]))
    run = graph.run({"b.text": "x", "m.text": "y"}, max_items=2)

    assert (run.status, run.state.cause.type) == ("failed", "execution_failed")


def test_a_call_past_its_timeout_that_comes_back_while_the_leg_drains_is_kept_and_never_called_again():
    effects: list[str] = []

    @node("late", title="Late", description="For a test.", policy=Policy(attempt_timeout=0.2, concurrency=2))
    def late(text: str) -> str:
        time.sleep(0.4)
        effects.append(text)
        return text

    @node("long", title="Long", description="For a test.", policy=Policy(concurrency=2))
    def long(text: str) -> str:
        time.sleep(0.9)
        return text

    graph = registry_of(late, long).load(GraphRecord(nodes=[NodeRecord(id="l", node="late"), NodeRecord(id="o", node="long")]))
    run = graph.run({"l.text": "mail", "o.text": "x"})
    assert (run.status, run.state.cause.type) == ("failed", "timeout") and effects == ["mail"]

    resumed = run.resume()

    assert resumed.status == "complete" and resumed.results["l"]["result"] == "mail" and effects == ["mail"]


@node("numbers", title="Numbers", description="For a test.")
def numbers(n: list[int]) -> Series[int]:
    """The caller's list as a Series, so what reads it runs once per item."""
    return n


def _counting_llm(calls: list[int], seconds: float) -> Any:
    @node("llm", title="Llm", description="For a test.", policy=Policy(concurrency=4))
    def llm(n: int) -> int:
        time.sleep(seconds)
        calls.append(n)
        return n * 2

    return llm


def test_a_stream_offers_checkpoints_and_resuming_the_last_one_repeats_only_the_calls_in_flight():
    calls: list[int] = []
    llm = _counting_llm(calls, 0.005)
    graph = registry_of(numbers, llm).load(make_graph(numbers, llm))

    async def until_killed() -> tuple[str, int]:
        saved, start = None, time.monotonic()
        async for event in graph.with_inputs(n=list(range(400))).stream(checkpoint_every=0.05):
            if event.type == "run_checkpoint":
                assert event.run.status == "running" and event.model_dump(mode="json") == {"type": "run_checkpoint"}
                saved = event.run.to_json()
            if time.monotonic() - start > 0.25 and saved is not None:
                break  # the process dies here
        return saved, len(calls)

    saved, before = _bounded(until_killed())
    time.sleep(0.1)  # the abandoned leg's threads come back
    calls.clear()
    held = sum(r is not None for r in graph.load_run(saved).results["llm"]["result"])
    resumed = graph.load_run(saved).resume()

    assert 0 < held <= before < 400
    assert resumed.status == "complete" and resumed.results["llm"]["result"] == [n * 2 for n in range(400)]
    assert len(calls) == 400 - held  # only what the checkpoint holds no record of


def test_a_stopped_leg_with_calls_on_threads_offers_a_checkpoint_before_it_drains():
    calls: list[int] = []
    llm = _counting_llm(calls, 0.3)
    graph = registry_of(numbers, llm).load(make_graph(numbers, llm))

    async def cancelled() -> list[Any]:
        cancel, seen = asyncio.Event(), []
        async for event in graph.with_inputs(n=[1, 2, 3, 4, 5, 6]).stream(cancel=cancel, checkpoint_every=60):
            seen.append(event)
            if event.type == "node_start" and event.loc[0] == "llm":
                cancel.set()
        return seen

    seen = _bounded(cancelled())
    kinds = [e.type for e in seen]
    (checkpoint,) = [e for e in seen if e.type == "run_checkpoint"]
    assert kinds[-1] == "run_end" and seen[-1].run.status == "cancelled"
    assert checkpoint.run.status == "cancelled" and checkpoint.run.results["llm"]["result"] == [None] * 6  # taken as the cancel came, before the calls in flight came back
    assert 0 < sum(v is not None for v in seen[-1].run.results["llm"]["result"]) == len(calls) < 6


def test_a_leg_handed_back_keeps_the_calls_in_flight_and_ends_with_a_checkpoint_of_a_running_run_and_no_run_end():
    calls: list[int] = []
    llm = _counting_llm(calls, 0.3)
    graph = registry_of(numbers, llm).load(make_graph(numbers, llm))

    async def handed_back() -> list[Any]:
        hand_back, seen = asyncio.Event(), []
        async for event in graph.with_inputs(n=[1, 2, 3, 4, 5, 6]).stream(hand_back=hand_back):
            seen.append(event)
            if event.type == "node_start" and event.loc[0] == "llm":
                hand_back.set()
        return seen

    seen = _bounded(handed_back())
    assert "run_end" not in [e.type for e in seen] and seen[-1].type == "run_checkpoint"
    left = seen[-1].run
    assert left.status == "running" and 0 < sum(v is not None for v in left.results["llm"]["result"]) == len(calls) < 6  # the calls in flight came back and are kept

    calls.clear()
    resumed = graph.load_run(left.to_json()).resume()  # another process goes on, repeating no call
    assert resumed.status == "complete" and resumed.results["llm"]["result"] == [2, 4, 6, 8, 10, 12]
    assert len(calls) == 6 - sum(v is not None for v in left.results["llm"]["result"])


def test_a_cancel_set_with_a_hand_back_ends_the_run_cancelled():
    graph = registry_of(numbers).load(make_graph(numbers))

    async def both() -> Any:
        cancel, hand_back = asyncio.Event(), asyncio.Event()
        cancel.set()
        hand_back.set()
        return [event async for event in graph.with_inputs(n=[1]).stream(cancel=cancel, hand_back=hand_back)][-1]

    last = _bounded(both())
    assert last.type == "run_end" and last.run.status == "cancelled"


@pytest.mark.parametrize("every", [0, -1])
def test_checkpoint_every_is_a_number_of_seconds_above_zero(every: Any) -> None:
    graph = registry_of(split).load(make_graph(split))
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(text="a").stream(checkpoint_every=every)
    assert [p.type for p in refused.value.errors()] == ["invalid_checkpoint_every"]


@pytest.mark.parametrize("timeout", [0, -1, math.nan])
def test_a_legs_timeout_is_a_number_of_seconds_above_zero_refused_at_the_call(timeout: Any) -> None:
    graph = registry_of(split).load(make_graph(split))
    for call in (lambda run: run.resume(timeout=timeout), lambda run: run.stream(timeout=timeout)):
        with pytest.raises(InvalidCallError) as refused:
            call(graph.with_inputs(text="a"))
        assert [p.type for p in refused.value.errors()] == ["invalid_timeout"]


def test_each_checkpoint_of_a_leg_that_reuses_earlier_work_resumes_without_a_call():
    """A checkpoint keeps the earlier work not yet taken as it stood, written after the stream ended or at once."""
    calls: list[int] = []
    llm = _counting_llm(calls, 0.0005)
    graph = registry_of(numbers, llm).load(make_graph(numbers, llm))
    first = graph.run(n=list(range(300)))
    calls.clear()

    async def checkpoints() -> list[Any]:
        return [e.run async for e in first.with_inputs().stream(checkpoint_every=0.0001) if e.type == "run_checkpoint"]

    taken = _bounded(checkpoints())
    assert len(taken) > 5 and calls == []
    for snapshot in taken:
        assert graph.load_run(snapshot.to_json()).resume().results == first.results
    assert calls == []


def test_a_failed_leg_with_calls_on_threads_offers_a_checkpoint_saying_it_failed():
    @node("flaky", title="Flaky", description="For a test.", policy=Policy(concurrency=4))
    def flaky(n: int) -> int:
        if n == 0:
            raise RuntimeError("boom")
        time.sleep(0.3)
        return n

    graph = registry_of(numbers, flaky).load(make_graph(numbers, flaky))

    async def leg() -> list[Any]:
        return [e async for e in graph.with_inputs(n=[1, 2, 0, 3, 4, 5]).stream(checkpoint_every=60)]

    seen = _bounded(leg())
    (checkpoint,) = [e for e in seen if e.type == "run_checkpoint"]
    assert (checkpoint.run.status, checkpoint.run.state.cause.type) == ("failed", "execution_failed")
    assert seen[-1].type == "run_end" and seen[-1].status == "failed"


def test_a_blank_into_an_embedding_leaves_the_nodes_inside_uncalled_as_it_would_drawn_flat():
    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str | None]:
        return [None if w == "_" else w for w in text.split()]

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    inner = stored_graph("twice", title="Twice", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="v", node="upper")], edges=[("u.result", "v.text")]))
    graph = registry_of(words, upper, inner).load(GraphRecord(
        nodes=[NodeRecord(id="w", node="words"), NodeRecord(id="emb", node="twice")], edges=[("w.result", "emb.text")],
    ))
    events = _events(graph.with_inputs(text="a _ b _"))

    complete = {e.loc: (e.units, e.not_called) for e in events if e.type == "node_complete"}
    assert complete == {("w",): (1, 0), ("emb", "u"): (4, 2), ("emb", "v"): (4, 2), ("emb",): (4, 0)}  # the embedding is never called: the nodes inside are
    assert events[-1].run.not_called == [
        NotCalled(loc=("emb", 1, "u"), input="text"), NotCalled(loc=("emb", 3, "u"), input="text"),
        NotCalled(loc=("emb", 1, "v"), input="text"), NotCalled(loc=("emb", 3, "v"), input="text"),
    ]
    assert events[-1].run.results["emb"] == {"v.result": ["A", None, "B", None]}
