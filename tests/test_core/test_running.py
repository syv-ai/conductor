"""How a run goes: what its events say, how ``resume()`` and ``stream()`` end (complete, failed, cancelled, timed
out), and what a saved run keeps.

``resume()`` calls every node that is ready, until the run is complete, waits for a person, fails or stops, and keeps
the outcome of each call. ``run.stream()`` does the same and yields events, the last one ``run_end``; breaking out of
the loop, cancelling the task that reads it, or setting ``cancel`` stops it. A node's ``Policy`` decides how it is
called: one item at a time on the thread that called ``resume()``, or on threads, with retries and a timeout on each
attempt.
"""

import asyncio
import math
import threading
import time
from asyncio import CancelledError
from asyncio import Event as AsyncioEvent
from collections.abc import Callable, Mapping, Sequence
from threading import Event as ThreadEvent
from typing import Any, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    NotCalled,
    Policy,
    Run,
    Series,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import BaseModel, model_serializer

from tests.test_core.nodes import place, registry_of

#: The longest any test here waits for a run, a thread or an event.
BOUND = 10.0


@node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",") if text else []


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


def _numbers(count: int) -> str:
    return ",".join(str(i) for i in range(count))


def _record(registry: NodeRegistry, nodes: Mapping[str, str], edges: Sequence[tuple[str, str]], constants: Mapping[str, Any]) -> GraphRecord:
    """``make_graph`` of ``nodes`` (node id to the id of its definition in ``registry``), joined by exactly ``edges``,
    with ``constants`` by ``"node.input"``."""
    return make_graph(*((node_id, registry[definition_id]) for node_id, definition_id in nodes.items()), edges=list(edges), constants=constants)


def _per_item(definition: Any, items: int, *more: Any) -> Any:
    """A graph in which ``split`` returns ``items`` items and ``definition``, as node ``slow``, is called once per item."""
    return Graph.compile(GraphRecord.model_validate({"nodes": {"split": {"node": "split", "inputs": {"text": {"constant": _numbers(items)}}}, "slow": {"node": definition.id, "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}}}}}), registry_of(split, definition, *more))


def _events(run: Any, **kwargs: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream(**kwargs)]

    return asyncio.run(asyncio.wait_for(collect(), BOUND))


def _bounded(coroutine: Any) -> Any:
    return asyncio.run(asyncio.wait_for(coroutine, BOUND))


def _count_twice(calls: list[Any]) -> tuple[int, int]:
    """The number of calls after 0.3 seconds and after 0.6: equal when no call was still running."""
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


def test_node_complete_counts_the_items_not_called_for_a_none_and_the_saved_run_lists_them():
    graph = Graph.compile(make_graph(addresses, send), registry_of(addresses, send))

    events = _events(graph.with_inputs(text="ida@a.dk,,al@a.dk"))

    (mail,) = [e for e in events if e.type == "node_complete" and e.loc[0] == "send"]
    assert (mail.outcomes, mail.not_called, mail.reused) == (3, 1, 0)
    read_back = Run.from_json(events[-1].run.to_json(), graph)
    assert read_back.not_called == [NotCalled(loc=("send", 1), input="email")]
    assert read_back.results["send"]["result"] == ["sent ida@a.dk", None, "sent al@a.dk"]
    assert read_back.status == "complete"


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

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "source", "inputs": {"text": {"graph_input": "text"}}}, "l": {"node": "left", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}, "r": {"node": "right", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}, "m": {"node": "meet", "inputs": {"a": {"sources": [{"node": "l", "output": "result"}]}, "b": {"sources": [{"node": "r", "output": "result"}]}}}}, "outputs": {"result": {"node": "m", "output": "result"}}}), registry_of(source, left, right, meet))

    events = _events(graph.with_inputs(text="abc"))

    assert events[-1].run.outputs == {"result": "ABC|cba"}
    assert sorted(calls) == ["left", "right", "source"]
    assert [e.loc[0] for e in events if e.type == "node_complete"][-1] == "m"


def test_a_failed_item_fails_the_node_naming_the_item_and_the_node_collecting_it_never_starts():
    @node("picky", title="Picky", description="For a test.")
    def picky(text: str) -> str:
        if text == "1":
            raise ValueError("not this one")
        return text

    graph = Graph.compile(make_graph(split, picky, join, constants={"split.text": "0,1,2"}), registry_of(split, picky, join))

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

    skipping = Graph.compile(GraphRecord.model_validate({"nodes": {"o": {"node": "only-short", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "o", "output": "result"}}}), registry_of(only_short))
    assert skipping.run(text="long").outputs == {"result": None}
    assert skipping.run(text="ab").outputs == {"result": "ab"}

    failing = Graph.compile(make_graph(split, fails_on_c), registry_of(split, fails_on_c))
    failed = failing.run(text="a,b,c")
    assert failed.status == "failed" and failed.results["fails-on-c"]["result"] == ["a", "b", None]


def test_progress_counts_items_between_start_and_complete_with_the_total_once_the_items_are_known():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    events = _events(_per_item(echo, 3).with_inputs())

    kinds = [(e.type, getattr(e, "loc", None)) for e in events]
    progress = [e for e in events if e.type == "node_progress" and e.loc[0] == "slow"]
    assert progress and all(e.total == 3 for e in progress) and progress[-1].finished >= 1
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
    assert (progress[-1].finished, progress[-1].total) == (20_000, 20_000)
    complete = next(i for i, e in enumerate(events) if e.type == "node_complete" and e.loc[0] == "slow")
    assert events.index(progress[-1]) < complete



def test_a_run_that_ends_waiting_reports_the_last_progress_of_the_node_that_waits():
    @node("maybe", title="Maybe", description="For a test.", asks=True)
    def maybe(text: str) -> str:
        if text == "ask":
            raise Ask("Is this right?")
        return text

    graph = Graph.compile(make_graph(split, maybe, constants={"split.text": "ask,b,c"}), registry_of(split, maybe))

    events = _events(graph.with_inputs())

    last = [e for e in events if e.type == "node_progress" and e.loc[0] == "maybe"][-1]
    assert (last.finished, last.total) == (2, 3)
    assert events[-1].status == "waiting"


def test_a_node_in_a_loop_over_an_empty_series_reports_no_progress():
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
    pinned = Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "fetch", "version": 1, "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "f", "output": "result"}}}), registry)
    newest = Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "fetch", "version": 2, "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "f", "output": "result"}}}), registry)

    assert pinned.run(text="x").outputs == {"result": "one"} and calls == [1, 1, 1]
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


def test_nodes_at_concurrency_one_are_all_called_on_the_callers_thread():
    """A chain of 100 nodes over 20 items is called on one thread, the one that called ``run``."""
    seen: set[int] = set()

    @node("step", title="Step", description="For a test.")
    def step(text: str) -> str:
        seen.add(threading.get_ident())
        return text

    nodes = {"s": "split", **{f"n{i}": "step" for i in range(100)}}
    edges = [("s.result", "n0.text")] + [(f"n{i - 1}.result", f"n{i}.text") for i in range(1, 100)]
    registry = registry_of(split, step)
    graph = Graph.compile(_record(registry, nodes, edges, {"s.text": _numbers(20)}), registry)

    assert graph.run().status == "complete"
    assert seen == {threading.get_ident()}


def test_the_retry_delay_doubles_each_time_and_another_item_is_called_while_a_retry_waits():
    tries: dict[str, int] = {}
    finished: list[str] = []

    @node("flaky", title="Flaky", description="For a test.", policy=Policy(concurrency=1, retries=3, retry_delay=0.02))
    def flaky(text: str) -> str:
        tries[text] = tries.get(text, 0) + 1
        if text == "flaky" and tries[text] <= 3:
            raise ExternalError("try again")
        finished.append(text)
        return text

    graph = Graph.compile(make_graph(split, flaky, constants={"split.text": "flaky,steady"}), registry_of(split, flaky))

    events = _events(graph.with_inputs())

    assert [e.delay for e in events if e.type == "node_retry"] == [0.02, 0.04, 0.08]
    assert finished == ["steady", "flaky"]


def test_an_instant_node_beside_forty_slow_items_does_not_time_out():
    """An attempt's timeout counts from when its call starts, never while it waits for a thread."""
    calls: list[str] = []

    @node("sleeper", title="Sleeper", description="For a test.", policy=Policy(concurrency=40))
    def sleeper(text: str) -> str:
        calls.append(text)
        time.sleep(0.5)
        return text

    @node("instant", title="Instant", description="For a test.", policy=Policy(attempt_timeout=0.1))
    def instant(texts: Series[str]) -> str:
        return "now"

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"split": {"node": "split", "inputs": {"text": {"constant": _numbers(40)}}}, "slow": {"node": "sleeper", "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}}}, "i": {"node": "instant", "inputs": {"texts": {"sources": [{"node": "split", "output": "result"}]}}}}, "outputs": {"result": {"node": "i", "output": "result"}}}), registry_of(split, sleeper, instant))

    run = graph.run()

    assert run.status == "complete" and run.outputs["result"] == "now" and len(calls) == 40


def test_calls_returning_out_of_order_on_threads_are_listed_in_item_order_in_the_run_a_checkpoint_and_a_saved_run():
    """Calls on threads return in any order; what a run says of a node's calls is in item order all the same."""

    @node("first-is-last", title="First is last", description="For a test.", policy=Policy(concurrency=4))
    def first_is_last(text: str) -> str | None:
        if text == "0":
            time.sleep(0.3)  # the first item returns after every other
        return None

    @node("needs-text", title="Needs text", description="For a test.")
    def needs_text(text: str) -> str:
        return text

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"split": {"node": "split", "inputs": {"text": {"constant": _numbers(4)}}}, "slow": {"node": "first-is-last", "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}}}, "after": {"node": "needs-text", "inputs": {"text": {"sources": [{"node": "slow", "output": "result"}]}}}}}), registry_of(split, first_is_last, needs_text))
    in_item_order = [("after", 0), ("after", 1), ("after", 2), ("after", 3)]

    async def streamed() -> list[Any]:
        return [event async for event in graph.with_inputs().stream(checkpoint_every=0.0001)]

    events = asyncio.run(asyncio.wait_for(streamed(), BOUND))
    run = events[-1].run
    checkpoints = [event.run for event in events if event.type == "run_checkpoint"]

    assert [call.loc for call in run.not_called] == in_item_order
    assert [call.loc for call in Run.from_json(run.to_json(), graph).not_called] == in_item_order
    assert checkpoints and all([call.loc for call in checkpoint.not_called] == sorted(call.loc for call in checkpoint.not_called) for checkpoint in checkpoints)
    assert any(len(checkpoint.not_called) > 1 for checkpoint in checkpoints)  # some checkpoint has later items before the first


@pytest.mark.parametrize("items", [None, 3])
def test_a_node_made_ready_by_each_of_its_two_upstream_nodes_is_called_once(items: int | None):
    """A call that returns on a thread makes its downstream nodes ready; a call made ready twice is still one call."""
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

    first, constants = ({}, {"s.text": "x"}) if items is None else ({"split": "split"}, {"split.text": _numbers(items)})
    registry = registry_of(split, from_a_thread, left, right, both)
    graph = Graph.compile(_record(
        registry,
        {**first, "s": "from-a-thread", "a": "left", "b": "right", "r": "both"},
        [*([] if items is None else [("split.result", "s.text")]), ("s.result", "a.text"), ("s.result", "b.text"), ("a.result", "r.left"), ("b.result", "r.right")],
        constants,
    ), registry)

    run = graph.run()

    assert run.status == "complete"
    assert sorted(calls) == ([("xa", "xb")] if items is None else [(f"{i}a", f"{i}b") for i in range(items)])


@pytest.mark.parametrize("concurrency", [1, 4])
@pytest.mark.parametrize("shape", ["chain", "per_item_then_collected"])
def test_one_resume_calls_every_node_its_own_calls_make_ready_and_another_resume_calls_nothing(shape: str, concurrency: int):
    """``resume()`` tries each node after its upstream nodes, so one call of it completes a graph that asks no person,
    whether the first node is called on the caller's thread or on others."""
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
        nodes, constants = {"f": "first", **{f"t{i}": "then" for i in range(5)}}, {"f.text": "x"}
        edges = [("f.result", "t0.text"), *((f"t{i}.result", f"t{i + 1}.text") for i in range(4))]
        expected, outputs = 6, {"result": "x!!!!!"}
    else:
        nodes, constants = {"split": "split", "f": "first", "t": "then", "l": "last"}, {"split.text": _numbers(4)}
        edges = [("split.result", "f.text"), ("f.result", "t.text"), ("t.result", "l.texts")]
        expected, outputs = 9, {"result": "0!+1!+2!+3!"}
    registry = registry_of(split, first, then, last)
    graph = Graph.compile(_record(registry, nodes, edges, constants), registry)

    run = graph.run()

    assert run.status == "complete" and run.outputs == outputs and len(calls) == expected
    assert all(value is not None for values in run.results.values() for value in values.values())
    assert run.resume().outputs == outputs and Run.from_json(run.to_json(), graph).resume().outputs == outputs and len(calls) == expected


def test_concurrency_one_with_an_attempt_timeout_never_has_two_calls_in_flight():
    """A timed-out attempt is not retried, and its thread counts against the node's concurrency until it returns."""
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

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "slow-second", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "s", "output": "result"}}}), registry_of(slow_second))

    assert graph.run(text="x").outputs == {"result": "done"}


# -- what a node may raise ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "raised",
    [CancelledError(), GeneratorExit(), BaseException("b"), StopIteration()],
    ids=lambda raised: type(raised).__name__,
)
def test_anything_a_node_raises_fails_its_call_and_ends_the_run(raised):
    @node("raises", title="Raises", description="For a test.")
    def raises(text: str) -> str:
        raise raised

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"r": {"node": "raises", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "r", "output": "result"}}}), registry_of(raises))

    run = graph.run(text="x")
    streamed = _events(graph.with_inputs(text="x"))[-1].run

    for ended in (run, streamed):
        assert (ended.status, ended.state.cause.type, ended.state.cause.loc) == ("failed", "execution_failed", ("r",))


def test_a_defect_in_the_engine_reaches_the_caller_rather_than_hanging_the_stream(monkeypatch):
    from conductor._internal.run.arguments import ResultReader

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "echo", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), registry_of(echo))
    prepared = graph.with_inputs(text="x")

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("an engine defect")

    monkeypatch.setattr(ResultReader, "arguments_for", broken)
    with pytest.raises(RuntimeError, match="an engine defect"):
        _events(prepared)


# -- a failed run keeps the calls it already made ----------------------------------------------


def _item_zero_fails_at_once(concurrency: int, **policy: Any) -> tuple[Any, list[str], ThreadEvent]:
    """The node ``mail``, the list of texts it was called with, and the event that releases it: item 0 fails at
    once, every other item waits for ``release``."""
    calls: list[str] = []
    release = ThreadEvent()

    @node("mail", title="Mail", description="For a test.", policy=Policy(concurrency=concurrency, **policy))
    def mail(text: str) -> str:
        calls.append(text)
        if text == "0":
            raise ValueError("this item can't be sent")
        release.wait(BOUND)
        return text

    return mail, calls, release


def _releasing_on_failure(prepared: Any, release: ThreadEvent, cancel: AsyncioEvent | None = None) -> list[Any]:
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

    assert _count_twice(calls) == (1, 1) and calls == ["0"]
    assert events[-1].run.status == "failed" and events[-1].run.results["slow"]["result"] == [None] * 5


def test_a_failed_run_waits_for_a_call_still_running_no_longer_than_its_attempt_timeout():
    release = ThreadEvent()

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


def test_a_question_or_a_second_failure_from_a_call_still_running_after_a_failure_is_kept_or_logged():
    """A person's question is kept like any finished call; a second failure goes to the log, not onto the run."""
    release = ThreadEvent()

    @node("mixed", title="Mixed", description="For a test.", policy=Policy(concurrency=4), asks=True)
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


def test_no_retry_starts_once_the_run_has_failed():
    release = ThreadEvent()
    calls: list[str] = []

    @node("flaky-beside-a-failure", title="Flaky", description="For a test.", policy=Policy(concurrency=2, retries=3, retry_delay=0.01))
    def flaky(text: str) -> str:
        calls.append(text)
        if text == "0":
            raise ValueError("no")
        release.wait(BOUND)
        raise ExternalError("try again")

    events = _releasing_on_failure(_per_item(flaky, 2).with_inputs(), release)

    assert _count_twice(calls)[0] == 2 and sorted(calls) == ["0", "1"]
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
    seen, later = _count_twice(calls)

    assert later == seen and seen <= 2, seen


def test_cancelling_the_task_that_reads_the_stream_stops_the_rest():
    calls: list[str] = []
    prepared = _per_item(_sleeper(0.05, calls), 20).with_inputs()

    async def go() -> None:
        started = AsyncioEvent()

        async def consume() -> None:
            async for event in prepared.stream():
                if event.type == "node_start" and event.loc[0] == "slow":
                    started.set()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), BOUND)
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)

    _bounded(go())
    seen, later = _count_twice(calls)

    assert later == seen and seen <= 2, seen


def test_a_cancel_set_mid_run_ends_the_run_cancelled_at_once():
    calls: list[str] = []
    prepared = _per_item(_sleeper(0.05, calls), 20).with_inputs()

    async def go() -> tuple[list[Any], float]:
        cancel, seen, set_at = AsyncioEvent(), [], 0.0
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


def test_a_cancel_set_while_outcomes_of_the_previous_run_are_reused_ends_the_run_before_the_remaining_items():
    """The second run calls ``echo`` for no item: each item's outcome from the previous run is reused, and deciding
    that writes the item to compare it with the previous one. The test sets the cancel from inside the tenth such
    write and blocks there until the event loop has set it, so the cancel is set while 90 items are still to go."""
    writes: list[Callable[[], None]] = []

    @host_type("Line", title="Line")
    class Line(BaseModel):
        text: str

        @model_serializer
        def written(self) -> dict[str, str]:
            for write in writes:
                write()
            return {"text": self.text}

    @node("echo", title="Echo", description="For a test.")
    def echo(line: Line) -> str:
        return line.text

    @node("source", title="Source", description="For a test.")
    def source(tag: int = 0) -> Series[Line]:
        return [Line(text=str(n)) for n in range(100)]

    graph = Graph.compile(make_graph(source, echo, defaults={"source.tag": 0}), registry_of(source, echo))
    prepared = graph.run(tag=0).with_inputs(tag=1)

    async def go() -> Any:
        loop, cancel, cancel_is_set = asyncio.get_running_loop(), AsyncioEvent(), ThreadEvent()
        written_so_far = 0

        def set_the_cancel() -> None:
            cancel.set()
            cancel_is_set.set()

        def cancel_at_the_tenth_write() -> None:
            nonlocal written_so_far
            written_so_far += 1
            if written_so_far == 10:
                loop.call_soon_threadsafe(set_the_cancel)
                assert cancel_is_set.wait(5)

        writes.append(cancel_at_the_tenth_write)
        last = None
        async for event in prepared.stream(cancel=cancel):
            last = event
        return last

    ended = _bounded(go())

    assert ended.status == "cancelled"
    assert len([r for r in ended.run.results["echo"]["result"] if r is not None]) == 10  # the item being written, and none after it


def test_exactly_one_run_end_comes_last_when_a_call_returns_after_a_cancel():
    release = ThreadEvent()

    @node("late", title="Late", description="For a test.", policy=Policy(concurrency=2))
    def late(text: str) -> str:
        release.wait(BOUND)
        return text

    prepared = _per_item(late, 2).with_inputs()

    async def go() -> list[Any]:
        cancel, seen = AsyncioEvent(), []
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


def test_a_cancel_during_a_retry_delay_ends_the_run_before_every_attempt_has_run():
    calls: list[int] = []

    @node("flaky", title="Flaky", description="For a test.", policy=Policy(retries=5, retry_delay=0.2))
    def flaky(text: str) -> str:
        calls.append(1)
        raise ExternalError("try again")

    prepared = Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "flaky", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "f", "output": "result"}}}), registry_of(flaky)).with_inputs(text="x")

    async def go() -> Any:
        cancel, last = AsyncioEvent(), None
        async for event in prepared.stream(cancel=cancel):
            if event.type == "node_retry":
                cancel.set()
            last = event
        return last

    ended = _bounded(go())

    assert ended.status == "cancelled" and len(calls) < 6


def test_running_a_graph_under_a_running_event_loop_ends_as_it_does_outside_one():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    graph = Graph.compile(make_graph(split, echo, join), registry_of(split, echo, join))

    async def inside() -> Any:
        return graph.run(text="a,b")

    assert _bounded(inside()).outputs == graph.run(text="a,b").outputs == {"result": "a+b"}


# -- what a saved run keeps --------------------------------------------------------------------


def test_a_float_that_is_not_finite_inside_a_model_reads_back_from_a_saved_run_that_then_resumes():
    @host_type("Reading", title="Reading")
    class Reading(BaseModel):
        value: float
        low: float
        high: float

    @node("measure", title="Measure", description="For a test.")
    def measure(text: str) -> Reading:
        return Reading(value=math.nan, low=-math.inf, high=math.inf)

    @node("check", title="Check", description="For a test.", asks=True)
    def check(reading: Reading) -> str:
        raise Ask(f"Is {reading.low}..{reading.high} right?")

    @node("width", title="Width", description="For a test.")
    def width(reading: Reading, checked: str) -> str:
        return f"{reading.low}..{reading.high} {checked}"

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"m": {"node": "measure", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "check", "inputs": {"reading": {"sources": [{"node": "m", "output": "result"}]}}}, "w": {"node": "width", "inputs": {"reading": {"sources": [{"node": "m", "output": "result"}]}, "checked": {"sources": [{"node": "c", "output": "result"}]}}}}, "outputs": {"result": {"node": "w", "output": "result"}}}), registry_of(measure, check, width))
    paused = graph.run(text="x")

    read_back = Run.from_json(paused.to_json(), graph)

    reading = read_back.results["m"]["result"]
    assert math.isnan(reading.value) and (reading.low, reading.high) == (-math.inf, math.inf)
    assert read_back.answer("c", "yes").resume().outputs == {"result": "-inf..inf yes"}


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

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "bare", "inputs": {"text": {"graph_input": "text"}}}, "r": {"node": "contact-records", "inputs": {"table": {"sources": [{"node": "b", "output": "result"}]}}}, "g": {"node": "greet", "inputs": {"email": {"sources": [{"node": "r", "output": "result", "field": "email"}]}}}}, "outputs": {"result": {"node": "g", "output": "result"}}}), registry_of(bare, contact_records, greet))

    run = graph.run(text="x")

    assert run.status == "failed" and (run.state.cause.type, run.state.cause.loc) == ("invalid_output", ("b",))
    assert (run.state.cause.ctx["name"], run.state.cause.ctx["column"]) == ("result", "email")


def test_a_table_with_a_null_cell_reads_back_from_a_saved_run():
    class Reachable(TypedDict):
        name: str
        email: str | None

    @node("contacts", title="Contacts", description="For a test.")
    def contacts(text: str) -> Table[Reachable]:
        return pl.DataFrame({"name": ["ida", "bo"], "email": ["ida@a.dk", None]})

    @node("records", title="Records", description="For a test.")
    def records(table: Table[Reachable]) -> Series[Reachable]:
        return table.to_dicts()

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"c": {"node": "contacts", "inputs": {"text": {"graph_input": "text"}}}, "r": {"node": "records", "inputs": {"table": {"sources": [{"node": "c", "output": "result"}]}}}, "m": {"node": "send", "inputs": {"email": {"sources": [{"node": "r", "output": "result", "field": "email"}]}}}}, "outputs": {"result": {"node": "m", "output": "result"}}}), registry_of(contacts, records, send))
    run = graph.run(text="x")

    read_back = Run.from_json(run.to_json(), graph)

    assert read_back.results["c"]["result"].to_dicts() == [{"name": "ida", "email": "ida@a.dk"}, {"name": "bo", "email": None}]
    assert read_back.not_called == [NotCalled(loc=("m", 1), input="email")]


def test_node_start_comes_first_for_a_node_whose_items_are_reused_from_the_previous_run():
    """A run given new inputs after an answered run reuses the previous items before calling the new one;
    ``node_start`` is still the first event about the node."""

    @node("read", title="Read", description="For a test.")
    def read(text: str) -> Series[str]:
        return text.split(",")

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        if name == "ida":
            return True
        raise Ask(f"Approve {name}?")

    graph = Graph.compile(make_graph(read, approve), registry_of(read, approve))
    paused = graph.run(text="ida,bo")
    answered = paused.answer(paused.waiting[0], True).resume()

    events = _events(answered.with_inputs(text="ida,bo,eva"))
    about_approve = [e.type for e in events if getattr(e, "loc", None) == ("approve",)]
    assert about_approve[0] == "node_start", about_approve
    assert about_approve.count("node_start") == 1


def test_a_series_over_the_item_limit_returned_after_a_failure_leaves_the_failure_the_cause():
    @node("boom", title="Boom", description="For a test.", policy=Policy(concurrency=2))
    def boom(text: str) -> str:
        raise ValueError("no")

    @node("many", title="Many", description="For a test.", policy=Policy(concurrency=2))
    def many(text: str) -> Series[str]:
        time.sleep(0.2)
        return ["a", "b", "c", "d"]

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "boom", "inputs": {"text": {"graph_input": "b_text"}}}, "m": {"node": "many", "inputs": {"text": {"graph_input": "m_text"}}}}, "outputs": {"b_result": {"node": "b", "output": "result"}, "m_result": {"node": "m", "output": "result"}}}), registry_of(boom, many))
    run = graph.run(b_text="x", m_text="y", max_items=2)

    assert (run.status, run.state.cause.type) == ("failed", "execution_failed")


def test_a_call_past_its_timeout_that_returns_before_the_run_ends_is_kept_and_never_called_again():
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

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"l": {"node": "late", "inputs": {"text": {"graph_input": "l_text"}}}, "o": {"node": "long", "inputs": {"text": {"graph_input": "o_text"}}}}, "outputs": {"l_result": {"node": "l", "output": "result"}, "o_result": {"node": "o", "output": "result"}}}), registry_of(late, long))
    run = graph.run(l_text="mail", o_text="x")
    assert (run.status, run.state.cause.type) == ("failed", "timeout") and effects == ["mail"]

    resumed = run.resume()

    assert resumed.status == "complete" and resumed.results["l"]["result"] == "mail" and effects == ["mail"]


@node("numbers", title="Numbers", description="For a test.")
def numbers(n: list[int]) -> Series[int]:
    """The graph input as a series, so a node wired to it is called once per item."""
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
    graph = Graph.compile(make_graph(numbers, llm), registry_of(numbers, llm))

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
    time.sleep(0.1)  # the calls on the abandoned stream's threads return
    calls.clear()
    kept = sum(r is not None for r in Run.from_json(saved, graph).results["llm"]["result"])
    resumed = Run.from_json(saved, graph).resume()

    assert 0 < kept <= before < 400
    assert resumed.status == "complete" and resumed.results["llm"]["result"] == [n * 2 for n in range(400)]
    assert len(calls) == 400 - kept  # only the calls the checkpoint has no outcome for


def test_a_cancelled_stream_with_calls_on_threads_yields_a_checkpoint_before_waiting_for_them():
    calls: list[int] = []
    llm = _counting_llm(calls, 0.3)
    graph = Graph.compile(make_graph(numbers, llm), registry_of(numbers, llm))

    async def cancelled() -> list[Any]:
        cancel, seen = AsyncioEvent(), []
        async for event in graph.with_inputs(n=[1, 2, 3, 4, 5, 6]).stream(cancel=cancel, checkpoint_every=60):
            seen.append(event)
            if event.type == "node_start" and event.loc[0] == "llm":
                cancel.set()
        return seen

    seen = _bounded(cancelled())
    kinds = [e.type for e in seen]
    (checkpoint,) = [e for e in seen if e.type == "run_checkpoint"]
    assert kinds[-1] == "run_end" and seen[-1].run.status == "cancelled"
    assert checkpoint.run.status == "cancelled" and checkpoint.run.results["llm"]["result"] == [None] * 6  # taken when the cancel was set, before the calls in flight returned
    assert 0 < sum(v is not None for v in seen[-1].run.results["llm"]["result"]) == len(calls) < 6


def test_a_stream_handed_back_keeps_the_calls_in_flight_and_ends_with_a_checkpoint_of_a_running_run_and_no_run_end():
    calls: list[int] = []
    llm = _counting_llm(calls, 0.3)
    graph = Graph.compile(make_graph(numbers, llm), registry_of(numbers, llm))

    async def handed_back() -> list[Any]:
        hand_back, seen = AsyncioEvent(), []
        async for event in graph.with_inputs(n=[1, 2, 3, 4, 5, 6]).stream(hand_back=hand_back):
            seen.append(event)
            if event.type == "node_start" and event.loc[0] == "llm":
                hand_back.set()
        return seen

    seen = _bounded(handed_back())
    assert "run_end" not in [e.type for e in seen] and seen[-1].type == "run_checkpoint"
    left = seen[-1].run
    assert left.status == "running" and 0 < sum(v is not None for v in left.results["llm"]["result"]) == len(calls) < 6  # the calls in flight returned and are kept

    calls.clear()
    resumed = Run.from_json(left.to_json(), graph).resume()  # another process resumes it, repeating no call
    assert resumed.status == "complete" and resumed.results["llm"]["result"] == [2, 4, 6, 8, 10, 12]
    assert len(calls) == 6 - sum(v is not None for v in left.results["llm"]["result"])


def test_a_cancel_set_with_a_hand_back_ends_the_run_cancelled():
    graph = Graph.compile(make_graph(numbers), registry_of(numbers))

    async def both() -> Any:
        cancel, hand_back = AsyncioEvent(), AsyncioEvent()
        cancel.set()
        hand_back.set()
        return [event async for event in graph.with_inputs(n=[1]).stream(cancel=cancel, hand_back=hand_back)][-1]

    last = _bounded(both())
    assert last.type == "run_end" and last.run.status == "cancelled"


@pytest.mark.parametrize("every", [0, -1, math.inf, True, "5"])
def test_checkpoint_every_is_a_number_of_seconds_above_zero(every: Any) -> None:
    graph = Graph.compile(make_graph(split), registry_of(split))
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(text="a").stream(checkpoint_every=every)
    assert [p.type for p in refused.value.errors()] == ["invalid_checkpoint_every"]


@pytest.mark.parametrize("timeout", [0, -1, math.nan, True, "5"])
def test_a_timeout_of_resume_or_stream_that_is_not_a_number_of_seconds_above_zero_is_refused(timeout: Any) -> None:
    graph = Graph.compile(make_graph(split), registry_of(split))
    for call in (lambda run: run.resume(timeout=timeout), lambda run: run.stream(timeout=timeout)):
        with pytest.raises(InvalidCallError) as refused:
            call(graph.with_inputs(text="a"))
        assert [p.type for p in refused.value.errors()] == ["invalid_timeout"]


def test_each_checkpoint_of_a_run_reusing_the_previous_runs_outcomes_resumes_without_a_call():
    """A checkpoint keeps the reusable outcomes not yet reused, whether it is saved after the stream ended or at once."""
    calls: list[int] = []
    llm = _counting_llm(calls, 0.0005)
    graph = Graph.compile(make_graph(numbers, llm), registry_of(numbers, llm))
    first = graph.run(n=list(range(300)))
    calls.clear()

    async def checkpoints() -> list[Any]:
        return [e.run async for e in first.with_inputs().stream(checkpoint_every=0.0001) if e.type == "run_checkpoint"]

    taken = _bounded(checkpoints())
    assert len(taken) > 5 and calls == []
    for snapshot in taken:
        assert Run.from_json(snapshot.to_json(), graph).resume().results == first.results
    assert calls == []


def test_a_failed_stream_with_calls_on_threads_yields_a_checkpoint_saying_it_failed():
    @node("flaky", title="Flaky", description="For a test.", policy=Policy(concurrency=4))
    def flaky(n: int) -> int:
        if n == 0:
            raise RuntimeError("boom")
        time.sleep(0.3)
        return n

    graph = Graph.compile(make_graph(numbers, flaky), registry_of(numbers, flaky))

    async def streamed() -> list[Any]:
        return [e async for e in graph.with_inputs(n=[1, 2, 0, 3, 4, 5]).stream(checkpoint_every=60)]

    seen = _bounded(streamed())
    (checkpoint,) = [e for e in seen if e.type == "run_checkpoint"]
    assert (checkpoint.run.status, checkpoint.run.state.cause.type) == ("failed", "execution_failed")
    assert seen[-1].type == "run_end" and seen[-1].status == "failed"


def test_a_none_into_a_subgraph_node_leaves_its_inner_nodes_not_called_as_in_a_flat_graph():
    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str | None]:
        return [None if w == "_" else w for w in text.split()]

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    inner = subgraph("twice", title="Twice", description="For a test.")(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "v": {"node": "upper", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "v", "output": "result"}}}))
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}}, "sub": {"node": "twice", "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}}}}, "outputs": {"result": {"node": "sub", "output": "result"}}}), registry_of(words, upper, inner))
    events = _events(graph.with_inputs(text="a _ b _"))

    complete = {e.loc: (e.outcomes, e.not_called) for e in events if e.type == "node_complete"}
    assert complete == {("w",): (1, 0), ("sub", "u"): (4, 2), ("sub", "v"): (4, 2), ("sub",): (4, 0)}  # the subgraph node is never called: its inner nodes are
    assert events[-1].run.not_called == [
        NotCalled(loc=("sub", 1, "u"), input="text"), NotCalled(loc=("sub", 3, "u"), input="text"),
        NotCalled(loc=("sub", 1, "v"), input="text"), NotCalled(loc=("sub", 3, "v"), input="text"),
    ]
    assert events[-1].run.results["sub"] == {"result": ["A", None, "B", None]}
