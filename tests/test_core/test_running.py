"""Tests for how a run goes.

They cover what a run's events say, how ``resume()`` and ``stream()`` end (complete, failed, cancelled or timed
out), and what a saved run keeps.

``resume()`` calls every node that is ready, and keeps the outcome of each call. It goes on until the run is
complete, waits for a person, fails or stops. ``run.stream()`` does the same and yields events, the last of them
``run_end``. Breaking out of the loop, cancelling the task that reads it, or setting ``cancel`` stops it.

A node's ``Policy`` decides how it is called. Either it is called one item at a time on the thread that called
``resume()``, or on threads of its own, with retries and a timeout on each attempt.
"""

import asyncio
import contextlib
import gc
import math
import threading
import time
from asyncio import CancelledError
from asyncio import Event as AsyncioEvent
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import Barrier, Condition, Thread
from threading import Event as ThreadEvent
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeError,
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
from conductor.events import NodeComplete, NodeStart
from pydantic import AfterValidator, BaseModel, model_serializer
from pytest import MonkeyPatch

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


def _record(
    registry: NodeRegistry,
    nodes: Mapping[str, str],
    edges: Sequence[tuple[str, str]],
    constants: Mapping[str, Any],
) -> GraphRecord:
    """Returns ``make_graph`` of ``nodes``, joined by exactly ``edges``, with these ``constants``.

    ``nodes`` maps each node id to the id of its definition in ``registry``. ``constants`` are keyed by
    ``"node.input"``.
    """
    return make_graph(
        *((node_id, registry[definition_id]) for node_id, definition_id in nodes.items()),
        edges=list(edges),
        constants=constants,
    )


def _per_item(definition: Any, items: int, *more: Any) -> Any:
    """Returns a graph in which ``split`` returns ``items`` items, and ``definition``, as node ``slow``, is called once
    per item."""
    return Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "split": {"node": "split", "inputs": {"text": {"constant": _numbers(items)}}},
                    "slow": {
                        "node": definition.id,
                        "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                }
            }
        ),
        registry_of(split, definition, *more),
    )


def _events(run: Any, **kwargs: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream(**kwargs)]

    return asyncio.run(asyncio.wait_for(collect(), BOUND))


def _bounded(coroutine: Any) -> Any:
    return asyncio.run(asyncio.wait_for(coroutine, BOUND))


def _count_twice(calls: list[Any]) -> tuple[int, int]:
    """Counts the calls after 0.3 seconds, and again after 0.6.

    The two counts are equal when no call was still running.
    """
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
    assert (mail.count.finished, mail.count.not_called, mail.reused) == (3, 1, 0)
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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "source", "inputs": {"text": {"graph_input": "text"}}},
                    "l": {
                        "node": "left",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "r": {
                        "node": "right",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "m": {
                        "node": "meet",
                        "inputs": {
                            "a": {"sources": [{"node": "l", "output": "result"}]},
                            "b": {"sources": [{"node": "r", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "m", "output": "result"}},
            }
        ),
        registry_of(source, left, right, meet),
    )

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

    graph = Graph.compile(
        make_graph(split, picky, join, constants={"split.text": "0,1,2"}),
        registry_of(split, picky, join),
    )

    events = _events(graph.with_inputs())

    (failed,) = [e for e in events if e.type == "node_failed"]
    assert (failed.loc[0], failed.cause.type, failed.cause.loc) == (
        "picky",
        "execution_failed",
        ("picky", 1),
    )
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

    skipping = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"o": {"node": "only-short", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "o", "output": "result"}},
            }
        ),
        registry_of(only_short),
    )
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
    assert (
        progress and all(e.count.total == 3 for e in progress) and progress[-1].count.finished >= 1
    )
    assert (
        kinds.index(("node_start", ("slow",)))
        < kinds.index(("node_progress", ("slow",)))
        < kinds.index(("node_complete", ("slow",)))
    )


def test_a_fast_node_over_many_items_reports_progress_a_few_times_a_second_ending_on_its_last_count():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    started = time.monotonic()
    events = _events(_per_item(echo, 20_000).with_inputs())
    took = time.monotonic() - started

    progress = [e for e in events if e.type == "node_progress" and e.loc[0] == "slow"]
    assert len(progress) <= took / 0.25 + 2
    assert (progress[-1].count.finished, progress[-1].count.total) == (20_000, 20_000)
    complete = next(
        i for i, e in enumerate(events) if e.type == "node_complete" and e.loc[0] == "slow"
    )
    assert events.index(progress[-1]) < complete


def test_a_run_that_ends_waiting_reports_the_last_progress_of_the_node_that_waits():
    @node("maybe", title="Maybe", description="For a test.", asks=True)
    def maybe(text: str) -> str:
        if text == "ask":
            raise Ask("Is this right?")
        return text

    graph = Graph.compile(
        make_graph(split, maybe, constants={"split.text": "ask,b,c"}), registry_of(split, maybe)
    )

    events = _events(graph.with_inputs())

    last = [e for e in events if e.type == "node_progress" and e.loc[0] == "maybe"][-1]
    assert (last.count.finished, last.count.total) == (2, 3)
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

    @node(
        "fetch",
        title="Fetch one",
        description="For a test.",
        policy=Policy(retries=2, retry_delay=0.01),
    )
    def fetch_one(text: str) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ExternalError("try again")
        return "one"

    @node(
        "fetch",
        title="Fetch two",
        description="For a test.",
        replaces=fetch_one,
        policy=Policy(retries=0),
    )
    def fetch_two(text: str) -> str:
        calls.append(2)
        raise ExternalError("never retried")

    registry = registry_of(fetch_two)
    pinned = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "f": {
                        "node": "fetch",
                        "version": 1,
                        "inputs": {"text": {"graph_input": "text"}},
                    }
                },
                "outputs": {"result": {"node": "f", "output": "result"}},
            }
        ),
        registry,
    )
    newest = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "f": {
                        "node": "fetch",
                        "version": 2,
                        "inputs": {"text": {"graph_input": "text"}},
                    }
                },
                "outputs": {"result": {"node": "f", "output": "result"}},
            }
        ),
        registry,
    )

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

    @node(
        "flaky",
        title="Flaky",
        description="For a test.",
        policy=Policy(concurrency=1, retries=3, retry_delay=0.02),
    )
    def flaky(text: str) -> str:
        tries[text] = tries.get(text, 0) + 1
        if text == "flaky" and tries[text] <= 3:
            raise ExternalError("try again")
        finished.append(text)
        return text

    graph = Graph.compile(
        make_graph(split, flaky, constants={"split.text": "flaky,steady"}),
        registry_of(split, flaky),
    )

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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "split": {"node": "split", "inputs": {"text": {"constant": _numbers(40)}}},
                    "slow": {
                        "node": "sleeper",
                        "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                    "i": {
                        "node": "instant",
                        "inputs": {"texts": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "i", "output": "result"}},
            }
        ),
        registry_of(split, sleeper, instant),
    )

    run = graph.run()

    assert run.status == "complete" and run.outputs["result"] == "now" and len(calls) == 40


def test_calls_returning_out_of_order_on_threads_are_listed_in_item_order_in_the_run_a_checkpoint_and_a_saved_run():
    """Calls on threads may return in any order, but a run always lists a node's calls in item order."""

    @node(
        "first-is-last",
        title="First is last",
        description="For a test.",
        policy=Policy(concurrency=4),
    )
    def first_is_last(text: str) -> str | None:
        if text == "0":
            time.sleep(0.3)  # the first item returns after all the others
        return None

    @node("needs-text", title="Needs text", description="For a test.")
    def needs_text(text: str) -> str:
        return text

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "split": {"node": "split", "inputs": {"text": {"constant": _numbers(4)}}},
                    "slow": {
                        "node": "first-is-last",
                        "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                    "after": {
                        "node": "needs-text",
                        "inputs": {"text": {"sources": [{"node": "slow", "output": "result"}]}},
                    },
                }
            }
        ),
        registry_of(split, first_is_last, needs_text),
    )
    in_item_order = [("after", 0), ("after", 1), ("after", 2), ("after", 3)]

    async def streamed() -> list[Any]:
        return [event async for event in graph.with_inputs().stream(checkpoint_every=0.0001)]

    events = asyncio.run(asyncio.wait_for(streamed(), BOUND))
    run = events[-1].run
    checkpoints = [event.run for event in events if event.type == "run_checkpoint"]

    assert [call.loc for call in run.not_called] == in_item_order
    assert [call.loc for call in Run.from_json(run.to_json(), graph).not_called] == in_item_order
    assert checkpoints and all(
        [call.loc for call in checkpoint.not_called]
        == sorted(call.loc for call in checkpoint.not_called)
        for checkpoint in checkpoints
    )
    assert any(
        len(checkpoint.not_called) > 1 for checkpoint in checkpoints
    )  # some checkpoint has later items before the first


@pytest.mark.parametrize("items", [None, 3])
def test_a_node_made_ready_by_each_of_its_two_upstream_nodes_is_called_once(items: int | None):
    """A node that each of two upstream nodes makes ready is still called only once.

    A call that returns on a thread makes the nodes after it ready. A call made ready twice is still one call.
    """
    calls: list[tuple[str, str]] = []

    @node(
        "from-a-thread",
        title="From a thread",
        description="For a test.",
        policy=Policy(concurrency=2),
    )
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

    first, constants = (
        ({}, {"s.text": "x"})
        if items is None
        else ({"split": "split"}, {"split.text": _numbers(items)})
    )
    registry = registry_of(split, from_a_thread, left, right, both)
    graph = Graph.compile(
        _record(
            registry,
            {**first, "s": "from-a-thread", "a": "left", "b": "right", "r": "both"},
            [
                *([] if items is None else [("split.result", "s.text")]),
                ("s.result", "a.text"),
                ("s.result", "b.text"),
                ("a.result", "r.left"),
                ("b.result", "r.right"),
            ],
            constants,
        ),
        registry,
    )

    run = graph.run()

    assert run.status == "complete"
    assert sorted(calls) == (
        [("xa", "xb")] if items is None else [(f"{i}a", f"{i}b") for i in range(items)]
    )


@pytest.mark.parametrize("concurrency", [1, 4])
@pytest.mark.parametrize("shape", ["chain", "per_item_then_collected"])
def test_one_resume_calls_every_node_its_own_calls_make_ready_and_another_resume_calls_nothing(
    shape: str, concurrency: int
):
    """One ``resume()`` completes a graph that asks no person, and a second ``resume()`` calls nothing.

    ``resume()`` tries each node after the nodes before it. So one call is enough, whether the first node is called on
    the caller's thread or on others.
    """
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
        nodes, constants = (
            {"split": "split", "f": "first", "t": "then", "l": "last"},
            {"split.text": _numbers(4)},
        )
        edges = [("split.result", "f.text"), ("f.result", "t.text"), ("t.result", "l.texts")]
        expected, outputs = 9, {"result": "0!+1!+2!+3!"}
    registry = registry_of(split, first, then, last)
    graph = Graph.compile(_record(registry, nodes, edges, constants), registry)

    run = graph.run()

    assert run.status == "complete" and run.outputs == outputs and len(calls) == expected
    assert all(value is not None for values in run.results.values() for value in values.values())
    assert (
        run.resume().outputs == outputs
        and Run.from_json(run.to_json(), graph).resume().outputs == outputs
        and len(calls) == expected
    )


def test_concurrency_one_with_an_attempt_timeout_never_has_two_calls_in_flight():
    """A timed-out attempt isn't retried. Its thread still counts against the node's concurrency until it returns."""
    in_flight, most = 0, 0
    lock = threading.Lock()

    @node(
        "slow-items",
        title="Slow items",
        description="For a test.",
        policy=Policy(concurrency=1, attempt_timeout=0.2, retries=2, retry_delay=0.01),
    )
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
    assert events[-1].run.status == "failed" and events[-1].run.cause.type == "timeout"


def test_a_run_has_no_deadline_by_default_and_a_slow_node_runs_to_the_end():
    @node("slow-second", title="Slow second", description="For a test.")
    def slow_second(text: str) -> str:
        time.sleep(1.0)
        return "done"

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "slow-second", "inputs": {"text": {"graph_input": "text"}}}
                },
                "outputs": {"result": {"node": "s", "output": "result"}},
            }
        ),
        registry_of(slow_second),
    )

    assert graph.run(text="x").outputs == {"result": "done"}


class _AtOnce:
    """Counts how many calls of a node run at once.

    Each call waits, for at most ``seconds``, until more than ``cap`` calls are running. So calls that are allowed to
    overlap really do overlap. Calls that aren't allowed to show it by never getting there.

    Attributes:
        most: The most calls that ran at once.
        calls: How many calls were made.
    """

    def __init__(self, cap: int, seconds: float = 0.1) -> None:
        self.cap, self.seconds = cap, seconds
        self.now, self.most, self.calls = 0, 0, 0
        self.changed = Condition()

    def call(self, release: ThreadEvent | None = None) -> None:
        """Makes one call.

        It runs until more than ``cap`` calls run or ``seconds`` pass, then until ``release`` is set.
        """
        with self.changed:
            self.now, self.calls = self.now + 1, self.calls + 1
            self.most = max(self.most, self.now)
            self.changed.notify_all()
            self.changed.wait_for(lambda: self.now > self.cap, timeout=self.seconds)
        if release is not None:
            release.wait(BOUND)
        with self.changed:
            self.now -= 1
            self.changed.notify_all()

    def wait_until_running(self, calls: int) -> None:
        with self.changed:
            assert self.changed.wait_for(lambda: self.now == calls, timeout=BOUND), self.now


def _in_two_places(definition: Any, items: int) -> Graph:
    """Returns a graph in which ``split`` returns ``items`` items, and ``definition`` is called once per item in two
    places, ``a`` and ``b``."""
    return Graph.compile(
        make_graph(
            ("split", split),
            ("a", definition),
            ("b", definition),
            edges=[("split.result", "a.text"), ("split.result", "b.text")],
            constants={"split.text": _numbers(items)},
            outputs={"a": "a.result", "b": "b.result"},
        ),
        registry_of(split, definition),
    )


def _in_threads(*functions: Callable[[], Any]) -> list[Any]:
    """Calls each of ``functions`` on a thread of its own, all at once, and returns what each returned.

    If one of them raised, the error is raised again here.
    """
    returned: list[Any] = [None] * len(functions)
    raised: list[BaseException] = []

    def go(index: int) -> None:
        try:
            returned[index] = functions[index]()
        except BaseException as error:  # noqa: BLE001  raised again on the test's thread
            raised.append(error)

    threads = [Thread(target=go, args=(index,)) for index in range(len(functions))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(BOUND)
        assert not thread.is_alive()
    if raised:
        raise raised[0]
    return returned


@pytest.mark.parametrize("concurrency", [1, 2])
def test_concurrency_caps_the_calls_of_a_node_in_two_places_of_two_runs_at_once(concurrency: int):
    """``concurrency`` counts every call of the node in the whole process.

    That includes calls from two places in one graph, and from two runs at once.
    """
    at_once = _AtOnce(concurrency)

    @node(
        "capped", title="Capped", description="For a test.", policy=Policy(concurrency=concurrency)
    )
    def capped(text: str) -> str:
        at_once.call()
        return text

    graph = _in_two_places(capped, 3)

    def run() -> str:
        return graph.with_inputs().resume(timeout=BOUND).status

    assert run() == "complete" and at_once.most == concurrency  # two places in one run
    at_once.most = 0
    assert _in_threads(run, run) == ["complete", "complete"]
    assert at_once.most == concurrency and at_once.calls == 3 * 2 + 3 * 2 * 2


def test_a_node_that_sets_no_concurrency_is_called_one_item_at_a_time_in_each_run_whatever_other_runs_do():
    """``Policy()`` sets no limit across runs.

    Each run calls the node one item at a time, but two runs at once may each be in a call of it.
    """
    at_once = _AtOnce(1)

    @node("free", title="Free", description="For a test.")
    def free(text: str) -> str:
        at_once.call()
        return text

    graph = _in_two_places(free, 3)

    def run() -> str:
        return graph.with_inputs().resume(timeout=BOUND).status

    assert (
        run() == "complete" and at_once.most == 1
    )  # one item at a time in one run, in both places
    at_once.most = 0
    assert _in_threads(run, run) == ["complete", "complete"] and at_once.most == 2


@pytest.mark.parametrize("ends", ["cancelled", "timed_out"])
def test_a_run_waiting_for_calls_another_run_makes_starts_once_that_run_ends_early_and_its_calls_return(
    ends: str,
):
    """A run waiting for another run's calls starts once that run ends early and its calls return.

    Run ``first`` takes both calls that ``concurrency=2`` allows, and ends early while they run. Run ``second`` waits
    for them to return, then makes its own calls, two at a time.
    """
    at_once, release = _AtOnce(2, seconds=0.2), ThreadEvent()

    @node("held", title="Held", description="For a test.", policy=Policy(concurrency=2))
    def held(text: str) -> str:
        at_once.call(release)
        return text

    graph = _per_item(held, 4)
    first, second = graph.with_inputs(), graph.with_inputs()
    cancel: list[Callable[[], None]] = []

    async def stream_first() -> Any:
        stop, loop = AsyncioEvent(), asyncio.get_running_loop()
        cancel.append(lambda: loop.call_soon_threadsafe(stop.set))
        return [event async for event in first.stream(cancel=stop)][-1].run

    def run_first() -> Any:
        return asyncio.run(stream_first()) if ends == "cancelled" else first.resume(timeout=0.05)

    def run_second() -> Any:
        at_once.wait_until_running(2)  # once first holds both calls
        return second.resume(timeout=BOUND)

    def end_first() -> None:
        try:
            at_once.wait_until_running(2)
            if ends == "cancelled":
                cancel[0]()
            with at_once.changed:  # first ends, and neither run makes a third call
                assert not at_once.changed.wait_for(lambda: at_once.now > 2, timeout=0.2)
        finally:
            release.set()

    first_run, second_run, _ = _in_threads(run_first, run_second, end_first)

    assert (
        first_run.status == ends
        and sum(text is not None for text in first_run.results["slow"]["result"]) == 2
    )
    assert second_run.status == "complete" and second_run.results["slow"]["result"] == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert at_once.most == 2 and at_once.calls == 2 + 4


def test_a_run_waiting_for_a_node_another_run_calls_calls_its_other_nodes_and_can_end_while_it_waits():
    """A run waiting for a node that another run is calling goes on with its other nodes, and can end while it waits.

    ``gate`` has ``concurrency=1`` and is called on each run's own thread. While run ``holder`` is in a call of
    ``gate``, run ``waiter`` calls ``other``. It then times out, still waiting for ``gate``, and its thread is never
    blocked. Once ``holder`` is done, ``waiter`` resumes and calls ``gate``.
    """
    holding, release = ThreadEvent(), ThreadEvent()
    gate_calls: list[str] = []

    @node("gate", title="Gate", description="For a test.", policy=Policy(concurrency=1))
    def gate(text: str) -> str:
        gate_calls.append(text)
        if text == "holder":
            holding.set()
            release.wait(BOUND)
        return text

    @node("other", title="Other", description="For a test.")
    def other(text: str) -> str:
        return text

    def graph_for(name: str) -> Graph:
        return Graph.compile(
            make_graph(
                ("gate", gate),
                ("other", other),
                edges=[],
                constants={"gate.text": name, "other.text": name},
                outputs={"gate": "gate.result", "other": "other.result"},
            ),
            registry_of(gate, other),
        )

    holder, waiter = graph_for("holder").with_inputs(), graph_for("waiter").with_inputs()

    def hold() -> Any:
        return holder.resume()

    def wait() -> Any:
        assert holding.wait(BOUND)
        returned = waiter.resume(timeout=0.1)
        release.set()
        return returned

    held, waited = _in_threads(hold, wait)

    assert (
        waited.status == "timed_out"
        and waited.results["other"]["result"] == "waiter"
        and waited.results["gate"]["result"] is None
    )
    assert held.status == "complete" and gate_calls == ["holder"]
    assert waited.resume().status == "complete" and gate_calls == ["holder", "waiter"]


def _one_node_per_item(definition: Any) -> Graph:
    """Returns a graph in which ``split`` returns the items of the graph input ``text``, and ``definition``, as node
    ``slow``, is called once per item."""
    return Graph.compile(
        make_graph(
            ("split", split),
            ("slow", definition),
            edges=[("split.result", "slow.text")],
            outputs={"result": "slow.result"},
        ),
        registry_of(split, definition),
    )


def test_a_run_that_raises_out_of_resume_leaves_no_call_of_a_capped_node_counted():
    """When ``resume()`` raises, no call of a capped node is left counted.

    Calls handed to the pool but not yet on a thread are cancelled, and each is counted out. So a later run of the
    node isn't kept waiting for them.
    """

    @node("exits", title="Exits", description="For a test.", policy=Policy(concurrency=2))
    def exits(text: str) -> str:
        if text == "7":
            raise SystemExit("the node exits")
        return text

    graph = _one_node_per_item(exits)
    for _ in range(10):
        with pytest.raises(SystemExit):
            graph.with_inputs(text=_numbers(40)).resume(timeout=BOUND)
    assert graph.with_inputs(text="a,b,c").resume(timeout=2).status == "complete"


def test_a_run_woken_by_another_runs_call_stops_at_its_timeout_while_it_works_through_its_waiting_calls():
    """A run woken with 100 calls waiting still ends at its timeout, not after all 100.

    Whether to stop is checked before each waiting call starts.
    """
    holding, made = ThreadEvent(), []

    @node(
        "one-at-a-time",
        title="One at a time",
        description="For a test.",
        policy=Policy(concurrency=1),
    )
    def one_at_a_time(text: str) -> str:
        if text == "hold":
            holding.set()
            ThreadEvent().wait(0.3)  # holds the only call allowed, for 0.3 s
        else:
            made.append(text)
            ThreadEvent().wait(0.02)
        return text

    graph = _one_node_per_item(one_at_a_time)

    def second() -> Any:
        assert holding.wait(BOUND)
        return graph.with_inputs(text=_numbers(100)).resume(timeout=0.6)

    _, second_run = _in_threads(lambda: graph.with_inputs(text="hold").resume(), second)

    assert second_run.status == "timed_out" and 0 < len(made) < 50, len(made)


def test_a_run_that_calls_a_capped_node_on_its_own_thread_lets_a_run_waiting_for_it_have_a_turn():
    """Run ``busy`` calls the node 200 times, one at a time on its own thread. Run ``late`` asks for one call after
    ``busy``'s first and gets it within a few of ``busy``'s calls, not after all of them."""
    busy_calls, busy_started, late_done = [], ThreadEvent(), ThreadEvent()
    calls_before_late: list[int] = []

    @node("shared", title="Shared", description="For a test.", policy=Policy(concurrency=1))
    def shared(text: str) -> str:
        if text == "late":
            calls_before_late.append(len(busy_calls))
            late_done.set()
        else:
            busy_calls.append(text)
            busy_started.set()
            late_done.wait(0.002)  # each of busy's calls takes 2 ms until late's call is made
        return text

    graph = _one_node_per_item(shared)

    def late() -> Any:
        assert busy_started.wait(BOUND)
        return graph.with_inputs(text="late").resume(timeout=BOUND)

    busy, late_run = _in_threads(
        lambda: graph.with_inputs(text=_numbers(200)).resume(timeout=BOUND), late
    )

    assert busy.status == late_run.status == "complete"
    assert calls_before_late[0] < 10, calls_before_late


def test_a_run_cancelled_while_it_waits_for_another_runs_call_says_no_node_start_for_that_node():
    """``NodeStart`` comes only when a call of the node really starts.

    ``svc`` waits behind another run, and ``after`` sets the cancel. The stream then ends with no ``NodeStart`` for
    ``svc``.
    """
    holding, release = ThreadEvent(), ThreadEvent()
    cancel: list[Callable[[], None]] = []

    @node("svc", title="Svc", description="For a test.", policy=Policy(concurrency=1))
    def svc(text: str) -> str:
        if text == "hold":
            holding.set()
            release.wait(BOUND)
        return text

    @node("after", title="After", description="For a test.")
    def after(text: str) -> str:
        cancel[
            0
        ]()  # svc was tried before this, in graph order, and waits behind the run holding it
        return text

    holder = Graph.compile(
        make_graph(("svc", svc), outputs={"result": "svc.result"}), registry_of(svc)
    ).with_inputs(text="hold")
    waiter = Graph.compile(
        make_graph(
            ("svc", svc),
            ("after", after),
            edges=[],
            constants={"svc.text": "wait", "after.text": "x"},
            outputs={"svc": "svc.result", "after": "after.result"},
        ),
        registry_of(svc, after),
    ).with_inputs()

    async def stream() -> list[Any]:
        stop, loop = AsyncioEvent(), asyncio.get_running_loop()
        cancel.append(lambda: loop.call_soon_threadsafe(stop.set))
        return [event async for event in waiter.stream(cancel=stop)]

    def wait() -> list[Any]:
        try:
            assert holding.wait(BOUND)
            return asyncio.run(stream())
        finally:
            release.set()

    _, events = _in_threads(holder.resume, wait)

    assert events[-1].run.status == "cancelled"
    assert [event.loc for event in events if event.type == "node_start"] == [("after",)]


def _held_x() -> tuple[Any, ThreadEvent, ThreadEvent]:
    """Returns the node ``x`` and two events, ``holding`` and ``release``.

    ``x`` has ``concurrency=1``. Its call with the text ``"hold"`` sets ``holding``, then keeps the node busy until
    ``release`` is set.
    """
    holding, release = ThreadEvent(), ThreadEvent()

    @node("x", title="X", description="For a test.", policy=Policy(concurrency=1))
    def x(text: str) -> str:
        if text == "hold":
            holding.set()
            release.wait(BOUND)
        return text

    return x, holding, release


def _x_then(x: Any, y: Any) -> Graph:
    """Returns a graph of ``x`` and ``y`` side by side, with no edge. A run tries ``x`` first, then calls ``y``."""
    return Graph.compile(
        make_graph(
            ("x", x),
            ("y", y),
            edges=[],
            constants={"x.text": "first", "y.text": "y"},
            outputs={"x": "x.result", "y": "y.result"},
        ),
        registry_of(x, y),
    )


def test_a_run_woken_while_its_thread_is_busy_elsewhere_delays_no_run_that_asked_after_it():
    """A run woken while its thread is busy elsewhere doesn't delay a run that asked after it.

    Run ``first`` waits for ``x``, and meanwhile calls ``y`` on its own thread. ``y`` waits for run ``last`` to be
    done. ``last`` asks for ``x`` after ``first``, and gets it as soon as it is free.
    """
    x, holding, release = _held_x()
    first_in_y, last_done = ThreadEvent(), ThreadEvent()

    @node("y", title="Y", description="For a test.")
    def y(text: str) -> str:
        first_in_y.set()
        return "after last" if last_done.wait(3) else "last kept waiting"

    only_x = Graph.compile(make_graph(("x", x), outputs={"result": "x.result"}), registry_of(x))

    def run_first() -> Any:
        assert holding.wait(BOUND)
        return _x_then(x, y).with_inputs().resume(timeout=BOUND)

    def run_last() -> Any:
        assert first_in_y.wait(BOUND)
        try:
            return only_x.with_inputs(text="last").resume(timeout=BOUND)
        finally:
            last_done.set()

    def let_go() -> None:
        assert first_in_y.wait(BOUND)
        ThreadEvent().wait(0.1)  # last has asked for x by now
        release.set()

    _, first, last, _ = _in_threads(
        lambda: only_x.with_inputs(text="hold").resume(timeout=BOUND), run_first, run_last, let_go
    )

    assert last.status == first.status == "complete" and first.outputs["y"] == "after last"


def test_a_node_whose_function_runs_a_graph_using_a_capped_node_is_not_kept_from_it_by_the_run_it_was_called_by():
    """A node whose function runs a graph using a capped node isn't kept from it by its own outer run.

    Run ``first`` waits for ``x``, and meanwhile calls ``y``. ``y``'s function runs a graph that uses ``x``. That inner
    run gets ``x`` as soon as it is free, even though ``first`` asked before it.
    """
    x, holding, release = _held_x()
    inner_starts = ThreadEvent()
    only_x = Graph.compile(make_graph(("x", x), outputs={"result": "x.result"}), registry_of(x))

    @node("y", title="Y", description="For a test.")
    def y(text: str) -> str:
        inner_starts.set()
        return only_x.with_inputs(text="inner").resume(timeout=3).status

    def run_first() -> Any:
        assert holding.wait(BOUND)
        return _x_then(x, y).with_inputs().resume(timeout=BOUND)

    def let_go() -> None:
        assert inner_starts.wait(BOUND)
        ThreadEvent().wait(0.1)  # the inner run has asked for x by now
        release.set()

    _, first, _ = _in_threads(
        lambda: only_x.with_inputs(text="hold").resume(timeout=BOUND), run_first, let_go
    )

    assert first.status == "complete" and first.outputs == {"x": "first", "y": "complete"}


def test_an_interrupt_inside_the_pools_submit_counts_the_call_out_once(monkeypatch: MonkeyPatch):
    """A ``KeyboardInterrupt`` inside ``ThreadPoolExecutor.submit`` counts the call out exactly once.

    The interrupt comes after the call was handed to a thread. Since the call is counted out once, a later run still
    has at most ``concurrency`` calls at once.
    """
    started = ThreadEvent()
    at_once = _AtOnce(1)

    @node(
        "counted-once",
        title="Counted once",
        description="For a test.",
        policy=Policy(concurrency=1, attempt_timeout=BOUND),
    )
    def counted_once(text: str) -> str:
        started.set()
        at_once.call()
        return text

    submit = ThreadPoolExecutor.submit

    def interrupted(self: ThreadPoolExecutor, *args: Any, **kwargs: Any) -> Any:
        submit(self, *args, **kwargs)
        assert started.wait(BOUND)
        raise KeyboardInterrupt

    graph = _one_node_per_item(counted_once)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        graph.with_inputs(text="a").resume()
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    ThreadEvent().wait(0.2)  # the interrupted call returns on its thread
    at_once.most = 0

    def run() -> str:
        return graph.with_inputs(text="b").resume(timeout=BOUND).status

    assert (
        _in_threads(run, run) == ["complete", "complete"] and at_once.most == 1
    )  # one pool thread each: the count decides


@pytest.mark.slow
def test_a_run_kept_waiting_for_a_capped_node_while_its_thread_is_busy_elsewhere_slows_no_other_run_of_it():
    """A run kept waiting for a capped node, while its thread is busy elsewhere, slows no other run of that node.

    Run ``hog`` makes 200 calls of ``x`` on its own thread. Run ``busy`` asks for ``x`` while ``hog`` has it, and then
    spends the rest of ``hog``'s run in ``y``. ``hog`` doesn't wait for ``busy`` after each call, so it takes about as
    long as it would alone.
    """
    hog_started, hog_done = ThreadEvent(), ThreadEvent()

    @node("x", title="X", description="For a test.", policy=Policy(concurrency=1))
    def x(text: str) -> str:
        hog_started.set()
        ThreadEvent().wait(0.001)
        return text

    @node("y", title="Y", description="For a test.")
    def y(text: str) -> str:
        hog_done.wait(BOUND)
        return text

    hog = _one_node_per_item(x)
    busy = _x_then(x, y)

    def timed_hog() -> float:
        started = time.monotonic()
        assert hog.with_inputs(text=_numbers(200)).resume(timeout=BOUND).status == "complete"
        return time.monotonic() - started

    alone = timed_hog()
    hog_started.clear()

    def run_hog() -> float:
        try:
            return timed_hog()
        finally:
            hog_done.set()

    def run_busy() -> Any:
        assert hog_started.wait(BOUND)
        return busy.with_inputs().resume(timeout=BOUND)

    with_busy, busy_run = _in_threads(run_hog, run_busy)

    assert busy_run.status == "complete"
    assert with_busy < alone + 0.5, (alone, with_busy)


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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"r": {"node": "raises", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "r", "output": "result"}},
            }
        ),
        registry_of(raises),
    )

    run = graph.run(text="x")
    streamed = _events(graph.with_inputs(text="x"))[-1].run

    for ended in (run, streamed):
        assert (ended.status, ended.cause.type, ended.cause.loc) == (
            "failed",
            "execution_failed",
            ("r",),
        )


def _broken_check(text: str) -> str:
    raise RuntimeError("a defect in a type's own check")


@pytest.mark.parametrize(
    "policy", [Policy(), Policy(concurrency=2)], ids=["on the run's thread", "on a pool thread"]
)
def test_a_defect_outside_the_nodes_function_reaches_the_caller_rather_than_hanging_the_stream(
    policy: Policy,
) -> None:
    """A type whose own check raises something other than a refusal is a bug, not the node's failure.

    The error ends the stream by raising, wherever the node is called. It never leaves the stream hanging.
    """

    @node("echo", title="Echo", description="For a test.", policy=policy)
    def echo(text: str) -> Annotated[str, AfterValidator(_broken_check)]:
        return text

    prepared = Graph.compile(make_graph(echo), registry_of(echo)).with_inputs(text="x")
    with pytest.raises(RuntimeError, match="a defect in a type's own check"):
        _events(prepared)


# -- a failed run keeps the calls it already made ----------------------------------------------


def _item_zero_fails_at_once(concurrency: int, **policy: Any) -> tuple[Any, list[str], ThreadEvent]:
    """Returns the node ``mail``, the list of texts it was called with, and the event that releases it.

    Item 0 fails at once. Every other item waits for ``release``.
    """
    calls: list[str] = []
    release = ThreadEvent()

    @node(
        "mail",
        title="Mail",
        description="For a test.",
        policy=Policy(concurrency=concurrency, **policy),
    )
    def mail(text: str) -> str:
        calls.append(text)
        if text == "0":
            raise ValueError("this item can't be sent")
        release.wait(BOUND)
        return text

    return mail, calls, release


def _releasing_on_failure(
    prepared: Any, release: ThreadEvent, cancel: AsyncioEvent | None = None
) -> list[Any]:
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
    assert (
        events[-1].run.status == "failed" and events[-1].run.results["slow"]["result"] == [None] * 5
    )


@pytest.mark.slow
def _beside_a_call_on_the_scheduler_thread(
    threaded: Callable[[ThreadEvent], int], policy: Policy, own: Callable[[ThreadEvent], int]
) -> Run:
    """Runs a node called on threads beside one called on the scheduler's own thread, and returns the run.

    ``a_threaded`` comes first by id, so it is on its thread when ``b_own`` blocks the scheduler. Each function gets
    one shared event, so the test can order what the two do without timing them.
    """
    shared = ThreadEvent()

    @node("a_threaded", title="Threaded", description="For a test.", policy=policy)
    def a_threaded(n: int) -> int:
        return threaded(shared)

    @node("b_own", title="Own", description="For a test.")
    def b_own(m: int) -> int:
        return own(shared)

    graph = make_graph(
        a_threaded, b_own, edges=[], outputs={"a": "a_threaded.result", "b": "b_own.result"}
    )
    return Graph.compile(graph, registry_of(a_threaded, b_own)).run(n=1, m=2)


def test_a_call_that_runs_past_its_attempt_timeout_times_out_while_the_scheduler_runs_another_call():
    """The scheduler sees the threaded call back only after ``b_own`` returns, but it came back after its deadline."""

    def late(back: ThreadEvent) -> int:
        time.sleep(0.3)
        back.set()
        return 1

    def own(back: ThreadEvent) -> int:
        assert back.wait(BOUND)
        time.sleep(0.05)  # the threaded call is on the queue by now
        return 2

    run = _beside_a_call_on_the_scheduler_thread(
        late, Policy(attempt_timeout=0.1, concurrency=2), own
    )

    assert (run.status, run.cause.type, run.cause.loc) == ("failed", "timeout", ("a_threaded",))


def test_a_call_back_before_its_attempt_timeout_isnt_timed_out_when_the_scheduler_sees_it_after():
    def in_time(back: ThreadEvent) -> int:
        back.set()
        return 1

    def own(back: ThreadEvent) -> int:
        assert back.wait(BOUND)
        time.sleep(0.3)  # past the threaded call's deadline
        return 2

    run = _beside_a_call_on_the_scheduler_thread(
        in_time, Policy(attempt_timeout=0.1, concurrency=2), own
    )

    assert (run.status, run.outputs) == ("complete", {"a": 1, "b": 2})


def test_a_retry_that_falls_due_while_the_scheduler_makes_its_own_calls_is_made_between_them():
    """``c_own`` is called for 20 items on the scheduler's own thread, one after another, so something is always
    ready. The retry of ``a_flaky`` falls due during them, and is made then, not once they are all done."""
    tries, retried = [], ThreadEvent()

    @node(
        "a_flaky",
        title="Flaky",
        description="For a test.",
        policy=Policy(retries=1, retry_delay=0.05, concurrency=2),
    )
    def a_flaky(n: int) -> int:
        tries.append(n)
        if len(tries) == 1:
            raise ExternalError("not yet")
        retried.set()
        return n

    @node("b_items", title="Items", description="For a test.")
    def b_items(count: int) -> Series[int]:
        return list(range(count))

    @node("c_own", title="Own", description="For a test.")
    def c_own(i: int) -> bool:
        time.sleep(0.03)
        return retried.is_set()

    graph = make_graph(
        a_flaky,
        b_items,
        c_own,
        edges=[("b_items.result", "c_own.i")],
        outputs={"seen": "c_own.result"},
    )
    run = Graph.compile(graph, registry_of(a_flaky, b_items, c_own)).run(n=1, count=20)

    assert run.status == "complete" and any(run.outputs["seen"])


def test_the_arguments_of_a_call_that_returned_arent_kept_until_its_attempt_timeout():
    """``count`` is given a list collected for its call alone, and returns long before its ``attempt_timeout`` of 30
    seconds. Once it has, nothing holds that list, so ``after``, called next in the same run, finds no list equal
    to it.

    The run's outcomes hold each item, never that list, so only something kept from the call could hold it. A weak
    reference can't be used: a list can't have one, and every argument that can is held by the outcomes.
    """

    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Series[str]:
        return text.split(",")

    @node("shout", title="Shout", description="For a test.")
    def shout(line: str) -> str:
        return line.upper()

    @node(
        "count",
        title="Count",
        description="For a test.",
        policy=Policy(concurrency=2, attempt_timeout=30),
    )
    def count(texts: Series[str]) -> int:
        return len(texts)

    given = [f"timed-{letter}" for letter in "abcdefghij"]

    @node("after", title="After", description="For a test.")
    def after(n: int) -> int:
        gc.collect()
        counted = [line.upper() for line in given]
        return sum(
            type(kept) is list and kept is not counted and kept == counted
            for kept in gc.get_objects()
        )

    graph = Graph.compile(
        make_graph(lines, shout, count, after), registry_of(lines, shout, count, after)
    )

    assert graph.run(text=",".join(given)).outputs == {"result": 0}


def test_a_call_that_times_out_after_a_cancel_ends_the_run_failed_with_timeout():
    """A failure after a cancel is the cancel's doing, but a timeout never is: the attempt's thread is abandoned,
    not stopped. So the run ends ``failed`` with ``timeout``, as it would without the cancel, and the call isn't
    made again in that ``resume()``."""
    calls, release = [], ThreadEvent()

    @node("charge", title="Charge", description="For a test.", policy=Policy(attempt_timeout=0.1))
    def charge(x: int) -> int:
        calls.append(x)
        release.wait(BOUND)
        return x

    prepared = Graph.compile(make_graph(charge), registry_of(charge)).with_inputs(x=1)

    async def cancelled_at_once() -> Any:
        cancel = AsyncioEvent()
        cancel.set()
        return [event async for event in prepared.stream(cancel=cancel)][-1]

    async def cancelled_once_it_runs() -> Any:
        cancel, last = AsyncioEvent(), None
        async for event in prepared.stream(cancel=cancel):
            if event.type == "node_start":
                cancel.set()
            last = event
        return last

    try:
        ended = _bounded(cancelled_once_it_runs())
    finally:
        release.set()

    assert (ended.status, ended.cause.type, ended.cause.loc) == ("failed", "timeout", ("charge",))
    assert calls == [1]
    assert (
        _bounded(cancelled_at_once()).status == "cancelled"
    )  # a cancel before the call still cancels


def test_a_failed_run_waits_for_a_call_still_running_no_longer_than_its_attempt_timeout():
    release, both_started = ThreadEvent(), Barrier(2)

    @node(
        "hangs-or-fails",
        title="Hangs or fails",
        description="For a test.",
        policy=Policy(concurrency=2, attempt_timeout=0.2),
    )
    def hangs_or_fails(text: str) -> str:
        both_started.wait(BOUND)
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

    assert run.status == "failed" and run.cause.type == "execution_failed", (
        "the first failure stands"
    )
    assert 0.15 < took < 1.0, took


def test_a_question_or_a_second_failure_from_a_call_still_running_after_a_failure_is_kept_or_logged():
    """A call still running after a failure keeps its question, or logs its own failure.

    A person's question is kept like any finished call. A second failure goes to the log, not onto the run.
    """
    release, all_started = ThreadEvent(), Barrier(4)

    @node(
        "mixed", title="Mixed", description="For a test.", policy=Policy(concurrency=4), asks=True
    )
    def mixed(text: str) -> str:
        all_started.wait(BOUND)
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
    assert run.status == "failed" and run.cause.loc == ("slow", 0)
    assert [e.type for e in events].count("node_failed") == 1
    assert [place(w) for w in run.waiting] == ["slow@1"]
    assert run.results["slow"]["result"] == [None, None, None, "3"]


def test_no_retry_starts_once_the_run_has_failed():
    release, both_started = ThreadEvent(), Barrier(2)
    calls: list[str] = []

    @node(
        "flaky-beside-a-failure",
        title="Flaky",
        description="For a test.",
        policy=Policy(concurrency=2, retries=3, retry_delay=0.01),
    )
    def flaky(text: str) -> str:
        calls.append(text)
        both_started.wait(BOUND)
        if text == "0":
            raise ValueError("no")
        release.wait(BOUND)
        raise ExternalError("try again")

    events = _releasing_on_failure(_per_item(flaky, 2).with_inputs(), release)

    assert _count_twice(calls)[0] == 2 and sorted(calls) == ["0", "1"]
    assert events[-1].run.cause.type == "execution_failed" and not [
        e for e in events if e.type == "node_retry"
    ]


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


@pytest.mark.slow
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
    """A cancel set while the previous run's outcomes are being reused ends the run before the remaining items.

    The second run calls ``echo`` for no item, since each item's outcome from the previous run is reused. To decide
    that, it writes each item out to compare it with the previous one. The test sets the cancel from inside the
    tenth such write, and blocks there until the event loop has set it. So the cancel is set while 90 items are
    still to go.
    """
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

    graph = Graph.compile(
        make_graph(source, echo, defaults={"source.tag": 0}), registry_of(source, echo)
    )
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
    assert (
        len([r for r in ended.run.results["echo"]["result"] if r is not None]) == 10
    )  # the item being written, and none after it


def test_exactly_one_run_end_comes_last_when_a_call_returns_after_a_cancel():
    release = ThreadEvent()

    @node("late", title="Late", description="For a test.", policy=Policy(concurrency=2))
    def late(text: str) -> str:
        release.wait(BOUND)
        return text

    prepared = _per_item(
        late, 4
    ).with_inputs()  # two calls on threads when the cancel comes, two never started

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

    @node(
        "flaky", title="Flaky", description="For a test.", policy=Policy(retries=5, retry_delay=0.2)
    )
    def flaky(text: str) -> str:
        calls.append(1)
        raise ExternalError("try again")

    prepared = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"f": {"node": "flaky", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "f", "output": "result"}},
            }
        ),
        registry_of(flaky),
    ).with_inputs(text="x")

    async def go() -> Any:
        cancel, last = AsyncioEvent(), None
        async for event in prepared.stream(cancel=cancel):
            if event.type == "node_retry":
                cancel.set()
            last = event
        return last

    ended = _bounded(go())

    assert ended.status == "cancelled" and len(calls) < 6


def _stopped_in_the_call(stop: str, then: Callable[[], Any]) -> Any:
    """Runs ``lines`` then ``echo`` with ``max_items=2``, and returns the run as it ended.

    In its first call, ``lines`` stops the run and then does ``then``. When ``stop`` is ``"cancel"`` it sets
    ``cancel``. Otherwise it sleeps past ``timeout``. The call runs on the scheduler's own thread, so whether to stop
    isn't checked again before the call ends.
    """
    stop_now: list[Callable[[], None]] = []

    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Series[str]:
        if stop_now:  # the first call only
            stop_now.pop()()
        return then()

    @node("echo", title="Echo", description="For a test.")
    def echo(line: str) -> str:
        return line

    prepared = Graph.compile(make_graph(lines, echo), registry_of(lines, echo)).with_inputs(
        text="x", max_items=2
    )

    async def stream_with_a_cancel() -> Any:
        loop, cancel, cancel_is_set = asyncio.get_running_loop(), AsyncioEvent(), ThreadEvent()

        def set_the_cancel() -> None:
            cancel.set()
            cancel_is_set.set()

        def cancel_and_wait() -> None:
            loop.call_soon_threadsafe(set_the_cancel)
            assert cancel_is_set.wait(BOUND)

        stop_now.append(cancel_and_wait)
        return [event async for event in prepared.stream(cancel=cancel)][-1].run

    if stop == "cancel":
        return _bounded(stream_with_a_cancel())
    stop_now.append(lambda: time.sleep(0.05))
    return prepared.resume(timeout=0.01)


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_a_series_over_the_item_limit_returned_by_a_call_that_ran_while_a_cancel_or_a_timeout_came_leaves_the_run_stopped(
    stop: str,
):
    """The call's outcome is kept, and ends the run at the item limit before the cancel or timeout is checked again.

    The first reason to stop is the one that stands.
    """
    ended = _stopped_in_the_call(stop, lambda: ["a", "b", "c"])

    assert (ended.status, ended.cause.type) == ("stopped", "item_limit")


def test_a_node_looped_over_an_empty_series_completes_with_no_node_start():
    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Series[str]:
        return text.splitlines()

    @node("shout", title="Shout", description="For a test.")
    def shout(line: str) -> str:
        return line.upper()

    prepared = Graph.compile(make_graph(lines, shout), registry_of(lines, shout)).with_inputs(
        text=""
    )

    async def events() -> list[Any]:
        return [event async for event in prepared.stream()]

    started = [event.loc for event in _bounded(events()) if isinstance(event, NodeStart)]
    assert started == [("lines",)]


def test_a_series_reused_over_the_item_limit_starts_its_node_before_it_completes():
    """A reused outcome comes without a call, so it starts its node when it is written, even when its series is
    over the item limit and stops the run. Only a node never called completes with no ``node_start``."""

    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Series[str]:
        return text.splitlines()

    @node("shout", title="Shout", description="For a test.")
    def shout(line: str) -> str:
        return line.upper()

    graph = Graph.compile(make_graph(lines, shout), registry_of(lines, shout))
    again = graph.run(text="a\nb\nc").with_inputs(max_items=2)

    async def events() -> list[Any]:
        return [event async for event in again.stream()]

    streamed = _bounded(events())
    assert [
        (type(event).__name__, event.loc)
        for event in streamed
        if isinstance(event, NodeStart | NodeComplete)
    ] == [
        ("NodeStart", ("lines",)),
        ("NodeComplete", ("lines",)),
    ]
    assert (streamed[-1].run.status, streamed[-1].run.cause.type) == ("stopped", "item_limit")


@pytest.mark.parametrize(
    ("stop", "ends", "cause"),
    [("cancel", "cancelled", "run_cancelled"), ("timeout", "failed", "no_lines")],
)
def test_a_call_that_fails_after_a_cancel_ends_the_run_cancelled_and_after_the_timeout_failed_and_is_made_again_on_resume(
    stop: str, ends: str, cause: str
):
    """A call that fails after a cancel ends the run ``cancelled``, and one that fails after the timeout ends it
    ``failed``.

    A node can be told of a cancel, and a node that stops a slow call then raises. So a failure after a cancel ends
    the run ``cancelled``. Nothing tells a node that the timeout passed, so a failure after it is the node's own. It
    ends the run ``failed``, with the node's cause. Either way the call has no outcome, and a resume makes it again.
    """
    calls: list[str] = []

    def fail_the_first_time() -> list[str]:
        calls.append("lines")
        if len(calls) == 1:
            raise NodeError("no lines", code="no_lines")
        return ["a"]

    ended = _stopped_in_the_call(stop, fail_the_first_time)

    assert (ended.status, ended.cause.type) == (ends, cause)
    resumed = ended.resume()
    assert (resumed.status, resumed.outputs, calls) == (
        "complete",
        {"result": ["a"]},
        ["lines", "lines"],
    )


def test_a_call_longer_than_the_timeout_that_fails_ends_the_run_failed_on_every_resume():
    """Each resume makes the call again, and it runs past the timeout again. The run reports the node's own failure."""
    calls: list[str] = []

    @node("slow", title="Slow", description="For a test.")
    def slow(text: str) -> str:
        calls.append(text)
        time.sleep(0.05)
        raise NodeError("refused", code="refused")

    ended = Graph.compile(make_graph(slow), registry_of(slow)).run(text="x", timeout=0.01)
    again = ended.resume(timeout=0.01)

    assert [(run.status, run.cause.type) for run in (ended, again)] == [("failed", "refused")] * 2
    assert calls == ["x", "x"]


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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "m": {"node": "measure", "inputs": {"text": {"graph_input": "text"}}},
                    "c": {
                        "node": "check",
                        "inputs": {"reading": {"sources": [{"node": "m", "output": "result"}]}},
                    },
                    "w": {
                        "node": "width",
                        "inputs": {
                            "reading": {"sources": [{"node": "m", "output": "result"}]},
                            "checked": {"sources": [{"node": "c", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "w", "output": "result"}},
            }
        ),
        registry_of(measure, check, width),
    )
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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "b": {"node": "bare", "inputs": {"text": {"graph_input": "text"}}},
                    "r": {
                        "node": "contact-records",
                        "inputs": {"table": {"sources": [{"node": "b", "output": "result"}]}},
                    },
                    "g": {
                        "node": "greet",
                        "inputs": {
                            "email": {
                                "sources": [{"node": "r", "output": "result", "field": "email"}]
                            }
                        },
                    },
                },
                "outputs": {"result": {"node": "g", "output": "result"}},
            }
        ),
        registry_of(bare, contact_records, greet),
    )

    run = graph.run(text="x")

    assert run.status == "failed" and (run.cause.type, run.cause.loc) == ("invalid_output", ("b",))
    assert (run.cause.ctx["name"], run.cause.ctx["column"]) == ("result", "email")


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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "c": {"node": "contacts", "inputs": {"text": {"graph_input": "text"}}},
                    "r": {
                        "node": "records",
                        "inputs": {"table": {"sources": [{"node": "c", "output": "result"}]}},
                    },
                    "m": {
                        "node": "send",
                        "inputs": {
                            "email": {
                                "sources": [{"node": "r", "output": "result", "field": "email"}]
                            }
                        },
                    },
                },
                "outputs": {"result": {"node": "m", "output": "result"}},
            }
        ),
        registry_of(contacts, records, send),
    )
    run = graph.run(text="x")

    read_back = Run.from_json(run.to_json(), graph)

    assert read_back.results["c"]["result"].to_dicts() == [
        {"name": "ida", "email": "ida@a.dk"},
        {"name": "bo", "email": None},
    ]
    assert read_back.not_called == [NotCalled(loc=("m", 1), input="email")]


def test_node_start_comes_first_for_a_node_whose_items_are_reused_from_the_previous_run():
    """``node_start`` is still the first event about a node whose items are reused from the previous run.

    A run given new inputs after an answered run reuses the earlier items before it calls the new one.
    """

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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "b": {"node": "boom", "inputs": {"text": {"graph_input": "b_text"}}},
                    "m": {"node": "many", "inputs": {"text": {"graph_input": "m_text"}}},
                },
                "outputs": {
                    "b_result": {"node": "b", "output": "result"},
                    "m_result": {"node": "m", "output": "result"},
                },
            }
        ),
        registry_of(boom, many),
    )
    run = graph.run(b_text="x", m_text="y", max_items=2)

    assert (run.status, run.cause.type) == ("failed", "execution_failed")


def test_a_call_past_its_timeout_that_returns_before_the_run_ends_is_kept_and_never_called_again():
    effects: list[str] = []

    @node(
        "late",
        title="Late",
        description="For a test.",
        policy=Policy(attempt_timeout=0.2, concurrency=2),
    )
    def late(text: str) -> str:
        time.sleep(0.4)
        effects.append(text)
        return text

    @node("long", title="Long", description="For a test.", policy=Policy(concurrency=2))
    def long(text: str) -> str:
        time.sleep(0.9)
        return text

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "l": {"node": "late", "inputs": {"text": {"graph_input": "l_text"}}},
                    "o": {"node": "long", "inputs": {"text": {"graph_input": "o_text"}}},
                },
                "outputs": {
                    "l_result": {"node": "l", "output": "result"},
                    "o_result": {"node": "o", "output": "result"},
                },
            }
        ),
        registry_of(late, long),
    )
    run = graph.run(l_text="mail", o_text="x")
    assert (run.status, run.cause, run.outputs) == (
        "complete",
        None,
        {"l_result": "mail", "o_result": "x"},
    ) and effects == ["mail"]

    resumed = run.resume()

    assert resumed.status == "complete" and effects == ["mail"]


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
                assert event.run.status == "running" and event.model_dump(mode="json") == {
                    "type": "run_checkpoint"
                }
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
    assert resumed.status == "complete" and resumed.results["llm"]["result"] == [
        n * 2 for n in range(400)
    ]
    assert len(calls) == 400 - kept  # only the calls the checkpoint has no outcome for


def test_a_run_read_back_from_a_checkpoint_announces_complete_only_the_nodes_not_announced_before_it():
    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Series[str]:
        return text.split(",")

    @node("shout", title="Shout", description="For a test.", policy=Policy(concurrency=2))
    def shout(line: str) -> str:
        time.sleep(0.05)
        return line.upper()

    graph = Graph.compile(make_graph(lines, shout), registry_of(lines, shout))

    async def completed(run: Run) -> list[tuple[list[str], Run | None]]:
        """Returns the nodes announced complete before each checkpoint, with the checkpoint, and then in all."""
        announced: list[str] = []
        seen: list[tuple[list[str], Run | None]] = []
        async for event in run.stream(checkpoint_every=0.01):
            if event.type == "node_complete":
                announced.append(event.loc[0])
            elif event.type == "run_checkpoint":
                seen.append((list(announced), event.run))
        return [*seen, (announced, None)]

    *checkpoints, (in_all, _) = _bounded(completed(graph.with_inputs(text="a,b,c")))

    assert any(before for before, _ in checkpoints)  # a node is announced before some checkpoint
    for before, checkpoint in checkpoints:
        read_back = Run.from_json(checkpoint.to_json(), graph)
        ((after, _),) = _bounded(completed(read_back))[-1:]
        assert sorted(before + after) == sorted(in_all) == ["lines", "shout"], (before, after)


def test_a_checkpoint_is_running_while_calls_go_on_and_read_back_is_what_its_outcomes_say():
    @node("q", title="Q", description="For a test.", asks=True)
    def q(n: int) -> int:
        if n == 0:
            raise Ask("?")
        time.sleep(0.02)
        return n

    graph = Graph.compile(make_graph(numbers, q), registry_of(numbers, q))

    async def copies() -> list[Run[Any]]:
        return [
            event.run
            async for event in graph.with_inputs(n=[0, 1, 2, 3]).stream(checkpoint_every=0.005)
            if event.type in ("run_checkpoint", "run_end")
        ]

    *checkpoints, end = _bounded(copies())
    assert (
        checkpoints and {c.status for c in checkpoints} == {"running"} and end.status == "waiting"
    )
    read_back = [
        Run.from_json(c.to_json(), graph) for c in checkpoints
    ]  # the process that made them is gone
    assert {r.status for r in read_back} == {"waiting"} and '"running"' not in checkpoints[
        -1
    ].to_json()


def test_a_run_whose_every_call_returned_is_complete_though_the_timeout_passed_while_it_waited_for_a_thread():
    calls: list[int] = []
    llm = _counting_llm(calls, 0.1)
    graph = Graph.compile(make_graph(numbers, llm), registry_of(numbers, llm))
    run = graph.with_inputs(n=[1]).resume(timeout=0.02)
    assert (run.status, run.cause, run.outputs) == (
        "complete",
        None,
        {"result": [2]},
    ) and calls == [1]


def test_a_cancelled_stream_with_calls_on_threads_yields_a_checkpoint_before_waiting_for_them():
    calls: list[int] = []
    llm = _counting_llm(calls, 0.3)
    graph = Graph.compile(make_graph(numbers, llm), registry_of(numbers, llm))

    async def cancelled() -> list[Any]:
        cancel, seen = AsyncioEvent(), []
        async for event in graph.with_inputs(n=[1, 2, 3, 4, 5, 6]).stream(
            cancel=cancel, checkpoint_every=60
        ):
            seen.append(event)
            if event.type == "node_start" and event.loc[0] == "llm":
                cancel.set()
        return seen

    seen = _bounded(cancelled())
    kinds = [e.type for e in seen]
    (checkpoint,) = [e for e in seen if e.type == "run_checkpoint"]
    assert kinds[-1] == "run_end" and seen[-1].run.status == "cancelled"
    assert (
        checkpoint.run.status == "cancelled"
        and checkpoint.run.results["llm"]["result"] == [None] * 6
    )  # taken when the cancel was set, before the calls in flight returned
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
    assert (
        left.status == "running"
        and 0 < sum(v is not None for v in left.results["llm"]["result"]) == len(calls) < 6
    )  # the calls in flight returned and are kept

    calls.clear()
    resumed = Run.from_json(
        left.to_json(), graph
    ).resume()  # another process resumes it, repeating no call
    assert resumed.status == "complete" and resumed.results["llm"]["result"] == [2, 4, 6, 8, 10, 12]
    assert len(calls) == 6 - sum(v is not None for v in left.results["llm"]["result"])


def test_a_cancel_set_with_a_hand_back_ends_the_run_cancelled():
    graph = Graph.compile(make_graph(numbers), registry_of(numbers))

    async def both() -> Any:
        cancel, hand_back = AsyncioEvent(), AsyncioEvent()
        cancel.set()
        hand_back.set()
        return [
            event
            async for event in graph.with_inputs(n=[1]).stream(cancel=cancel, hand_back=hand_back)
        ][-1]

    last = _bounded(both())
    assert last.type == "run_end" and last.run.status == "cancelled"


@pytest.mark.parametrize("every", [0, -1, math.inf, True, "5"])
def test_checkpoint_every_is_a_number_of_seconds_above_zero(every: Any) -> None:
    graph = Graph.compile(make_graph(split), registry_of(split))
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(text="a").stream(checkpoint_every=every)
    assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [
        (
            "invalid_limit",
            ("checkpoint_every",),
            {"limit": "checkpoint_every", "value": every, "variant": "seconds"},
        ),
    ]


@pytest.mark.parametrize("timeout", [0, -1, math.nan, True, "5"])
def test_a_timeout_of_resume_or_stream_that_is_not_a_number_of_seconds_above_zero_is_refused(
    timeout: Any,
) -> None:
    graph = Graph.compile(make_graph(split), registry_of(split))
    for call in (lambda run: run.resume(timeout=timeout), lambda run: run.stream(timeout=timeout)):
        with pytest.raises(InvalidCallError) as refused:
            call(graph.with_inputs(text="a"))
        assert [
            (p.type, p.loc, p.ctx["limit"], p.ctx["variant"]) for p in refused.value.problems
        ] == [("invalid_limit", ("timeout",), "timeout", "seconds")]
        if not (isinstance(timeout, float) and math.isnan(timeout)):
            assert refused.value.problems[0].ctx["value"] == timeout


def test_every_limit_refused_at_one_call_is_an_invalid_limit_of_its_own_in_the_order_of_the_limits():
    graph = Graph.compile(make_graph(split), registry_of(split))
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(text="a").stream(checkpoint_every=True, timeout=0)
    assert [(p.type, p.ctx) for p in refused.value.problems] == [
        ("invalid_limit", {"limit": "timeout", "value": 0, "variant": "seconds"}),
        ("invalid_limit", {"limit": "checkpoint_every", "value": True, "variant": "seconds"}),
    ]


def test_seconds_given_as_a_decimal_are_read_as_a_float_and_the_run_keeps_that_float():
    graph = Graph.compile(make_graph(split), registry_of(split))
    assert graph.with_inputs(text="a").resume(timeout=Decimal("1.5")).status == "complete"

    async def streamed() -> list[Any]:
        async with contextlib.aclosing(
            graph.with_inputs(text="a").stream(
                timeout=Decimal("2"), checkpoint_every=Decimal("0.01")
            )
        ) as events:
            return [event async for event in events]

    assert asyncio.run(streamed())[-1].run.status == "complete"


def test_each_checkpoint_of_a_run_reusing_the_previous_runs_outcomes_resumes_without_a_call():
    """A checkpoint keeps the reusable outcomes that aren't reused yet.

    That holds whether the checkpoint is saved after the stream ended or straight away.
    """
    calls: list[int] = []
    llm = _counting_llm(calls, 0.0005)
    graph = Graph.compile(make_graph(numbers, llm), registry_of(numbers, llm))
    first = graph.run(n=list(range(300)))
    calls.clear()

    async def checkpoints() -> list[Any]:
        return [
            e.run
            async for e in first.with_inputs().stream(checkpoint_every=0.0001)
            if e.type == "run_checkpoint"
        ]

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
        return [
            e async for e in graph.with_inputs(n=[1, 2, 0, 3, 4, 5]).stream(checkpoint_every=60)
        ]

    seen = _bounded(streamed())
    (checkpoint,) = [e for e in seen if e.type == "run_checkpoint"]
    assert (checkpoint.run.status, checkpoint.run.cause.type) == ("failed", "execution_failed")
    assert seen[-1].type == "run_end" and seen[-1].status == "failed"


def test_a_none_into_a_subgraph_node_leaves_its_inner_nodes_not_called_as_in_one_graph():
    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str | None]:
        return [None if w == "_" else w for w in text.split()]

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    inner = subgraph("twice", title="Twice", description="For a test.")(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
                    "v": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "v", "output": "result"}},
            }
        )
    )
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
                    "sub": {
                        "node": "twice",
                        "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "sub", "output": "result"}},
            }
        ),
        registry_of(words, upper, inner),
    )
    events = _events(graph.with_inputs(text="a _ b _"))

    complete = {
        e.loc: (e.count.finished, e.count.not_called) for e in events if e.type == "node_complete"
    }
    assert complete == {
        ("w",): (1, 0),
        ("sub", "u"): (4, 2),
        ("sub", "v"): (4, 2),
        ("sub",): (8, 4),
    }  # the subgraph node is never called: its count is its inner nodes' added up
    assert events[-1].run.not_called == [
        NotCalled(loc=("sub", 1, "u"), input="text"),
        NotCalled(loc=("sub", 3, "u"), input="text"),
        NotCalled(loc=("sub", 1, "v"), input="text"),
        NotCalled(loc=("sub", 3, "v"), input="text"),
    ]
    assert events[-1].run.results["sub"] == {"result": ["A", None, "B", None]}
