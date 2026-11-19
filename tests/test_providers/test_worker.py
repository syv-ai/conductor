"""Tests of the ``Worker`` that ``run_router`` serves runs through.

The tests call the worker directly, the way the routes do, with no HTTP in between. The store is kept in memory, and
a test waits for the store's own writes where it can, rather than sleeping.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from asyncio import Event as AsyncioEvent
from asyncio import Task
from collections.abc import Awaitable, Callable, Mapping
from threading import Event as ThreadEvent
from typing import Any, TypedDict

import polars as pl
import pytest
from conductor import (
    Ask,
    Context,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeError,
    NodeRegistry,
    Policy,
    Problem,
    Run,
    Series,
    Table,
    host_type,
    make_graph,
)
from conductor_providers.fastapi import RunChanges, StoredRun, Worker
from conductor_providers.fastapi.store import StoreFailedError
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import core_schema
from pytest import LogCaptureFixture, MonkeyPatch

from tests.test_providers.in_memory import (
    BEGUN_RUN_STATE,
    EVERY_RUN,
    MemoryStore,
    Stoppable,
    asking,
    gated,
    left_running,
    stoppable,
    stopping,
)

#: The metadata of the runs the tests start: the user who started each one.
ANNA = {"user": "anna"}


def _worker(store: MemoryStore, registry: NodeRegistry[Any], **settings: Any) -> Worker[Any, Any]:
    """Returns a worker on ``store`` whose runs keep a dict of metadata.

    Any setting that ``settings`` leaves out keeps the worker's own default.
    """
    worker: Worker[Any, Any] = Worker(store, nodes_for=lambda metadata: registry, **settings)
    return worker


async def _started(worker: Worker[Any, Any], graph: GraphRecord, inputs: Mapping[str, Any]) -> str:
    return (await worker.begin(graph, inputs, ANNA)).id


async def _resume_calls(store: MemoryStore, run_id: str) -> int:
    """Counts the resume calls that have started on the run. Each one stores a ``run_start`` event."""
    return [event["type"] for event in await store.stored_events(run_id)].count("run_start")


# -- beginning a run ------------------------------------------------------------------------------


async def test_a_run_begun_from_a_previous_one_of_an_earlier_version_of_its_graph_calls_only_what_the_edit_changed() -> (
    None
):
    registry, store, calls = NodeRegistry(), MemoryStore(), []

    @registry.node("split", title="Split", description="For a test.")
    def split(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("square", title="Square", description="For a test.")
    def square(i: int) -> int:
        calls.append(f"square {i}")
        return i * i

    @registry.node("add", title="Add", description="For a test.")
    def add(i: int, amount: int) -> int:
        calls.append(f"add {i}")
        return i + amount

    worker = _worker(store, registry)
    async with worker.serving():
        first = await _started(worker, make_graph(split, square), {"n": 3})
        assert (await store.ended(first)).status == "complete"

        edited = make_graph(split, square, add, constants={"add.amount": 1})
        second = await worker.begin(edited, {"n": 3}, ANNA, previous=first, allows=EVERY_RUN)
        ended = await store.ended(second.id)
        assert (ended.status, ended.outputs) == ("complete", {"result": [1, 2, 5]})
        assert calls == ["square 0", "square 1", "square 2", "add 0", "add 1", "add 4"]


def _registry_that_drops_old() -> tuple[
    NodeRegistry[Any], NodeRegistry[Any], GraphRecord, GraphRecord
]:
    """Returns two registries and two graphs, for a node that a later version drops.

    The first registry has the nodes ``split``, ``square`` and ``old``, and the second is the same without ``old``.
    The first graph uses all three nodes, and the second is that graph edited to drop ``old``.
    """
    later: NodeRegistry[Any] = NodeRegistry()

    @later.node("split", title="Split", description="For a test.")
    def split(n: int) -> Series[int]:
        return list(range(n))

    @later.node("square", title="Square", description="For a test.")
    def square(i: int) -> int:
        return i * i

    earlier: NodeRegistry[Any] = NodeRegistry()
    earlier.add_subregistry(later)

    @earlier.node("old", title="Old", description="For a test.")
    def old(i: int) -> int:
        return i

    return earlier, later, make_graph(split, square, old), make_graph(split, square)


async def test_a_run_begun_from_a_previous_run_that_cant_be_loaded_on_the_callers_nodes_is_refused_with_its_graphs_problems() -> (
    None
):
    earlier, later, uses_old, edited = _registry_that_drops_old()
    store = MemoryStore()
    starter = _worker(store, earlier)
    async with starter.serving():
        first = await _started(starter, uses_old, {"n": 2})
        assert (await store.ended(first)).status == "complete"
    worker = _worker(store, later)
    async with worker.serving():
        with pytest.raises(InvalidCallError) as refused:
            await worker.begin(edited, {"n": 2}, ANNA, previous=first, allows=EVERY_RUN)

        # the previous run's own graph, the one it was saved from, uses a node the caller's nodes don't have
        assert refused.value.problems == Graph.compile(uses_old, later).problems
        assert [(problem.type, problem.ctx["node"]) for problem in refused.value.problems] == [
            ("unknown_node", "old")
        ]
        assert list(store.rows) == [first]  # nothing is stored
        begun = await worker.begin(edited, {"n": 2}, ANNA)  # without previous=, it goes through
        assert (await store.ended(begun.id)).outputs == {"result": [0, 1]}


async def test_a_graph_begun_from_a_previous_run_is_refused_for_its_own_problems_and_its_inputs_before_the_previous_runs() -> (
    None
):
    earlier, later, uses_old, edited = _registry_that_drops_old()
    store = MemoryStore()
    starter = _worker(store, earlier)
    async with starter.serving():
        first = await _started(starter, uses_old, {"n": 2})
        await store.ended(first)
    worker = _worker(store, later)
    async with worker.serving():
        previous = first
        with pytest.raises(InvalidCallError) as broken:
            await worker.begin(
                {"nodes": {"x": {"node": "nonesuch"}}},
                {},
                ANNA,
                previous=previous,
                allows=EVERY_RUN,
            )
        with pytest.raises(InvalidCallError) as wrong_input:
            await worker.begin(edited, {"n": "two"}, ANNA, previous=previous, allows=EVERY_RUN)

    assert [problem.type for problem in broken.value.problems] == ["unknown_node"]
    assert [problem.type for problem in wrong_input.value.problems] == ["invalid_input"]


async def test_a_worker_that_never_served_refuses_to_make_a_run_running_and_one_that_left_hands_the_run_back() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)

    with pytest.raises(RuntimeError, match=r"serving\(\)"):
        await _started(worker, graph, inputs)
    assert store.rows == {}

    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (await store.ended(run_id)).status == "waiting"

    assert (await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)).status == "running"
    begun = await _started(worker, graph, inputs)
    for handed_back in (run_id, begun):  # handed back, so the next worker to serve takes them over
        row = await store.until(handed_back, lambda row: row.owner is None)
        assert (row.status, row.lease_until) == ("running", 0.0)

    async with _worker(store, registry).serving():
        assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}
        assert (await store.ended(begun)).status == "waiting"


# -- how a run's values are read back -------------------------------------------------------------


async def test_an_output_that_isnt_finite_is_stored_as_its_text() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("edges", title="Edges", description="For a test.")
    def edges(x: int) -> list[float]:
        return [float("inf"), float("-inf"), float("nan"), x]

    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, make_graph(edges), {"x": 1})

        assert (await store.ended(run_id)).outputs == {
            "result": ["Infinity", "-Infinity", "NaN", 1.0]
        }


@host_type("cents", title="Cents")
class Cents:
    """A host's own type, which only its own pydantic schema can write as JSON."""

    def __init__(self, cents: int) -> None:
        self.cents = cents

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, cls) else cls(int(value)),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda price: price.cents
            ),
        )


async def test_an_output_only_its_own_type_can_write_completes_as_it_does_in_the_engine() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("price", title="Price", description="For a test.")
    def price(cents: int) -> Cents:
        return Cents(cents)

    @registry.node("prices", title="Prices", description="For a test.")
    def prices(cents: int) -> Series[Cents]:
        return [Cents(cents), Cents(cents + 1)]

    graph = make_graph(
        price,
        prices,
        edges=[],
        inputs={"price": "price.cents", "prices": "prices.cents"},
        outputs={"price": "price.result", "prices": "prices.result"},
    )
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, {"price": 5, "prices": 7})

        ended = await store.ended(run_id)
        assert (ended.status, ended.outputs, ended.cause) == (
            "complete",
            {"price": 5, "prices": [7, 8]},
            None,
        )


class Part(TypedDict):
    text: str


async def test_a_table_made_for_each_item_is_stored_as_its_columns_per_item() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("as-table", title="As table", description="For a test.")
    def as_table(text: str) -> Table[Part]:
        return pl.DataFrame({"text": [text]})

    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, make_graph(split, as_table), {"text": "a,b"})

        ended = await store.ended(run_id)
        columns = [
            {"columns": [{"name": "text", "type": "str", "values": [text]}]} for text in ("a", "b")
        ]
        assert (ended.status, ended.outputs) == ("complete", {"result": columns})


# -- an item limit, a resume ----------------------------------------------------------------------


async def test_a_run_stopped_at_its_item_limit_says_why_and_is_resumed_at_the_same_or_a_higher_limit() -> (
    None
):
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("up", title="Up", description="For a test.")
    def up(text: str) -> str:
        return text.upper()

    worker = _worker(store, registry, max_items=1)
    async with worker.serving():

        async def last_end() -> dict[str, Any]:
            return [
                event for event in await store.stored_events(run_id) if event["type"] == "run_end"
            ][-1]

        run_id = await _started(worker, make_graph(split, up), {"text": "a,b"})
        stopped = await store.ended(run_id)
        assert stopped.status == "stopped" and stopped.cause is not None
        assert stopped.cause.type == (await last_end())["cause"]["type"] == "item_limit"
        assert stopped.cause.ctx == {"series": "split.result", "items": 2, "max_items": 1}

        await worker.resume(run_id, allows=EVERY_RUN)
        assert (await store.ended(run_id)).status == "stopped"
        await worker.resume(run_id, max_items=2, allows=EVERY_RUN)
        completed = await store.ended(run_id)
        assert completed.outputs == {"result": ["A", "B"]}
        assert completed.cause is None and (await last_end())["cause"] is None


async def test_an_item_limit_raised_by_a_resume_applies_to_every_resume_call_after_it() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="For a test.")
    def split(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("ask", title="Ask", description="For a test.", asks=True)
    def ask(x: int) -> int:
        raise Ask("ok?")

    worker = _worker(store, registry, max_items=3)
    async with worker.serving():
        run_id = await _started(
            worker,
            make_graph(split, ask, edges=[("split.result", "ask.x")], constants={"split.n": 5}),
            {},
        )
        assert (await store.ended(run_id)).status == "stopped"

        await worker.resume(run_id, max_items=10, allows=EVERY_RUN)
        assert (
            json.loads(store.rows[run_id].run)["max_items"] == 10
        )  # stored with the claim, so a call given up before its first checkpoint is resumed with it
        waiting = (await store.ended(run_id)).waiting
        assert len(waiting) == 5
        assert (
            await worker.answer(run_id, {waiting[0].id: 7}, allows=EVERY_RUN)
        ).status == "running"
        after = await store.ended(run_id)
        assert (after.status, len(after.waiting)) == ("waiting", 4)


async def test_a_resume_with_a_higher_item_limit_of_an_unreadable_saved_run_is_refused_and_leaves_the_row() -> (
    None
):
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("up", title="Up", description="For a test.")
    def up(text: str) -> str:
        return text.upper()

    worker = _worker(store, registry, max_items=1)
    async with worker.serving():
        run_id = await _started(worker, make_graph(split, up), {"text": "a,b"})
        assert (await store.ended(run_id)).status == "stopped"
        store.rows[run_id] = unreadable = dataclasses.replace(store.rows[run_id], run="{")

        with pytest.raises(InvalidCallError) as refused:
            await worker.resume(run_id, max_items=5, allows=EVERY_RUN)

        assert [problem.type for problem in refused.value.problems] == [
            "saved_run_unreadable"
        ] and store.rows[run_id] == unreadable


@pytest.mark.parametrize("status", ["running", "waiting", "complete"])
async def test_only_a_run_whose_last_resume_call_ended_early_is_resumed(status: str) -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        if status != "running":
            gate.set()
            await store.ended(run_id)
        if status == "complete":
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
            await store.ended(run_id)

        with pytest.raises(InvalidCallError) as refused:
            await worker.resume(run_id, allows=EVERY_RUN)

        assert [(problem.type, problem.ctx) for problem in refused.value.problems] == [
            ("run_not_resumable", {"status": status})
        ]
        gate.set()


# -- a cancel -------------------------------------------------------------------------------------


async def test_a_cancel_reaches_the_resume_call_that_an_answer_queued_during_the_previous_one_started() -> (
    None
):
    registry, graph, gates, calls = gated()
    inputs = {"csv": "a,b", "text": "x", "question": "y"}
    for gate in ("first", "second"):
        gates[gate].set()
    (waiting,) = Graph.compile(graph, registry).run(inputs).waiting
    gates["first"].clear()
    gates["second"].clear()
    calls.clear()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting.id: True}, allows=EVERY_RUN)
        ).status == "running"
        gates["first"].set()
        await asyncio.to_thread(gates["per_item_called"].wait, 5)
        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "running"
        gates["second"].set()

        assert (await store.ended(run_id)).status == "cancelled" and calls == ["a"]


async def test_a_cancel_sent_while_a_dead_worker_held_the_run_is_kept_when_another_takes_it_over() -> (
    None
):
    registry, graph, gates, calls = gated()
    for gate in gates.values():
        gate.set()
    run = Graph.compile(graph, registry).run({"csv": "a,b", "text": "x", "question": "y"})
    answered = run.with_answers({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    worker = _worker(store, registry)
    await store.create(left_running("taken", graph, answered, owner="a-dead-worker"))

    assert (await worker.cancel("taken", allows=EVERY_RUN)).status == "running"
    async with worker.serving():  # the worker takes over the run whose lease ran out straight away
        assert (await store.ended("taken")).status == "cancelled" and calls == []

        await worker.resume("taken", allows=EVERY_RUN)
        assert (await store.ended("taken")).status == "complete" and calls == ["a", "b"]


async def test_a_cancel_of_a_waiting_run_ends_it_cancelled_with_its_question_waiting_and_an_answer_resumes_it() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (await store.ended(run_id)).status == "waiting"
        events_while_waiting = len(await store.stored_events(run_id))

        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "running"
        cancelled = await store.ended(run_id)
        assert cancelled.cause is not None
        assert (
            cancelled.status,
            cancelled.cause.type,
            [waiting.id for waiting in cancelled.waiting],
        ) == (
            "cancelled",
            "run_cancelled",
            [waiting_id],
        )
        assert [
            (event["type"], event.get("status"))
            for event in (await store.stored_events(run_id))[events_while_waiting:]
        ] == [("run_start", None), ("run_end", "cancelled")]
        assert (
            await worker.cancel(run_id, allows=EVERY_RUN)
        ).status != "running"  # it ended early: nothing to cancel

        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}
        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status != "running"


async def test_the_worker_goes_by_a_runs_state_and_never_reads_the_status_column() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()

    class DropsStatus(MemoryStore):
        """Writes every column except ``status``.

        That is what a host's store does if its ``update`` forgets the one column it only uses in queries.
        """

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            return await super().update(
                run_id, rev, {name: value for name, value in changes.items() if name != "status"}
            )

    store = DropsStatus()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        row = await store.until(run_id, lambda row: row.owner is None)
        waiting = await worker.read(run_id, allows=EVERY_RUN)
        assert (row.status, waiting.status) == ("running", "waiting")

        with pytest.raises(InvalidCallError) as refused:
            await worker.resume(run_id, allows=EVERY_RUN)
        assert [problem.ctx for problem in refused.value.problems] == [{"status": "waiting"}]
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        await store.until(run_id, lambda row: row.owner is None)
        completed = await worker.read(run_id, allows=EVERY_RUN)
        assert completed.outputs == {"busy": "x", "after": True}
        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "complete"


# -- the stop event a node reads ------------------------------------------------------------------


async def test_a_node_sees_the_stop_event_set_when_its_run_is_cancelled_and_a_later_resume_call_gets_one_not_set() -> (
    None
):
    """``stream`` raises once it sees the stop event, so the run ends ``cancelled``.

    A later resume call gets a new stop event that isn't set, and calls ``stream`` again.
    """
    graph = stopping()
    store = MemoryStore()
    worker = _worker(store, graph.registry, context=stoppable)
    async with worker.serving():
        run_id = await _started(
            worker, graph.graph, {"question": "y", "text": "x", "seconds": 60.0}
        )
        await asyncio.to_thread(graph.started.wait, 60)
        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "running"
        assert (await store.ended(run_id)).status == "cancelled"

        assert (
            await worker.answer(run_id, {graph.waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        assert (await store.ended(run_id)).status == "complete"
    assert graph.seen == [True, False, False]


async def test_a_resume_call_an_answer_queued_during_a_cancelled_one_starts_gets_a_stop_event_not_set() -> (
    None
):
    graph = stopping()
    store = MemoryStore()
    worker = _worker(store, graph.registry, context=stoppable)
    async with worker.serving():
        run_id = await _started(
            worker, graph.graph, {"question": "y", "text": "x", "seconds": 60.0}
        )
        await asyncio.to_thread(graph.started.wait, 60)
        assert (
            await worker.answer(run_id, {graph.waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "running"

        completed = await store.until(run_id, lambda row: row.status == "complete")
        assert completed.status == "complete"
        ends = [
            event["status"]
            for event in await store.stored_events(run_id)
            if event["type"] == "run_end"
        ]
    assert (ends, graph.seen) == (["cancelled", "complete"], [True, False, False])


async def test_a_node_sees_the_stop_event_set_when_another_worker_cancels_its_run() -> None:
    graph = stopping()
    store = MemoryStore()
    holder = _worker(store, graph.registry, context=stoppable, lease=0.6)
    other = _worker(store, graph.registry, context=stoppable)
    async with holder.serving():
        run_id = await _started(
            holder, graph.graph, {"question": "y", "text": "x", "seconds": 60.0}
        )
        await asyncio.to_thread(graph.started.wait, 60)
        assert (
            await other.cancel(run_id, allows=EVERY_RUN)
        ).status == "running"  # the worker holding the run hears it at its next lease renewal

        assert (await store.ended(run_id)).status == "cancelled"
    assert graph.seen == [True]


async def test_a_worker_hears_a_cancel_sent_to_another_at_its_first_lease_renewal_after_the_cancel() -> (
    None
):
    """The run's one node waits for its stop event, which the worker holding the run sets when it hears the cancel.
    The store keeps whether each renewal tried since the cancel was written landed, so nothing rests on a clock. The
    first one finds the row moved on by the cancel, and the holder reads the row again and hears it. The node makes
    no checkpoint, since it never returns, so that renewal is the only read."""
    registry, started, heard = NodeRegistry(context_type=Stoppable), ThreadEvent(), []

    class CountsRenewals(MemoryStore):
        """Keeps whether each lease renewal tried once a cancel is written landed."""

        def __init__(self) -> None:
            super().__init__()
            self.cancelled = False
            self.renewals: list[bool] = []

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            written = await super().update(run_id, rev, changes, event)
            if changes.keys() == {"lease_until"} and self.cancelled:
                self.renewals.append(written)
            if written and "state" in changes and json.loads(changes["state"])["cancel_asked"]:
                self.cancelled = True
            return written

    store = CountsRenewals()

    @registry.node("listen", title="Listen", description="Waits until it is told to stop.")
    def listen(ctx: Context[Stoppable], text: str) -> str:
        started.set()
        assert ctx.stop.wait(10)
        heard.append(list(store.renewals))
        raise NodeError("The call was stopped.", code="stopped")

    holder = _worker(store, registry, context=stoppable, lease=0.6)
    other = _worker(store, registry, context=stoppable)
    async with holder.serving():
        run_id = await _started(holder, make_graph(listen), {"text": "x"})
        await asyncio.to_thread(started.wait, 10)
        assert (await other.cancel(run_id, allows=EVERY_RUN)).status == "running"

        assert (await store.ended(run_id)).status == "cancelled"
    assert heard[0][:1] == [False]


async def test_a_cancel_sent_to_another_worker_ends_the_run_cancelled_though_it_completes_before_its_holder_hears_it() -> (
    None
):
    """The worker holding a run hears a cancel sent to another worker only at its next checkpoint or lease renewal.
    A run that completes before then still ends ``cancelled``, as the cancel's answer said. Every call's outcome is
    kept, so a resume makes no call and completes the run."""
    registry, started, gate, calls = NodeRegistry(), ThreadEvent(), ThreadEvent(), []

    @registry.node("parts", title="Parts", description="Splits a text at its commas.")
    def parts(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("work", title="Work", description="Waits on a gate, then gives its part back.")
    def work(part: str) -> str:
        calls.append(part)
        started.set()
        gate.wait(10)
        return part

    store = MemoryStore()
    graph = make_graph(parts, work, edges=[("parts.result", "work.part")])
    holder, other = _worker(store, registry), _worker(store, registry)
    async with holder.serving(), other.serving():
        run_id = await _started(holder, graph, {"text": "p,q,r"})
        await asyncio.to_thread(started.wait, 10)
        assert (await other.cancel(run_id, allows=EVERY_RUN)).status == "running"
        gate.set()

        cancelled = await store.ended(run_id)
        assert (cancelled.status, cancelled.outputs, sorted(calls)) == (
            "cancelled",
            None,
            ["p", "q", "r"],
        )
        assert cancelled.cause is not None and cancelled.cause.type == "run_cancelled"
        ends = [
            event["status"]
            for event in await store.stored_events(run_id)
            if event["type"] == "run_end"
        ]
        assert ends == ["cancelled"]

        await other.resume(run_id, allows=EVERY_RUN)
        completed = await store.ended(run_id)
    assert (completed.status, completed.outputs, len(calls)) == (
        "complete",
        {"result": ["p", "q", "r"]},
        3,
    )


async def test_the_stop_event_is_not_set_without_a_cancel() -> None:
    graph = stopping()
    store = MemoryStore()
    worker = _worker(store, graph.registry, context=stoppable)
    async with worker.serving():
        run_id = await _started(worker, graph.graph, {"question": "y", "text": "x", "seconds": 0.0})
        assert (await store.ended(run_id)).status == "waiting"
        await worker.answer(run_id, {graph.waiting_id: True}, allows=EVERY_RUN)
        assert (await store.ended(run_id)).status == "complete"
    assert graph.seen == [False, False]


async def test_the_stop_event_is_not_set_when_the_worker_hands_the_run_back() -> None:
    """A worker that shuts down waits for the call in progress and keeps its result, so the node isn't told to stop."""
    registry, started, release, seen = (
        NodeRegistry(context_type=Stoppable),
        ThreadEvent(),
        ThreadEvent(),
        [],
    )

    @registry.node("held", title="Held", description="For a test.")
    def held(ctx: Context[Stoppable], text: str) -> str:
        started.set()
        release.wait(60)
        seen.append(ctx.stop.is_set())
        return text

    store = MemoryStore()
    worker = _worker(store, registry, context=stoppable)
    serving = worker.serving()
    await serving.__aenter__()
    run_id = await _started(worker, make_graph(held), {"text": "x"})
    await asyncio.to_thread(started.wait, 60)
    leaving = asyncio.create_task(serving.__aexit__(None, None, None))
    await asyncio.sleep(0)  # the worker is handing its runs back, and waits for the call
    release.set()
    await leaving

    assert (await worker.read(run_id, allows=EVERY_RUN)).status == "running"
    assert seen == [False]


async def test_a_node_sees_the_stop_event_set_when_another_worker_takes_its_run_over() -> None:
    graph = stopping()
    store = MemoryStore()
    worker = _worker(store, graph.registry, context=stoppable, lease=0.6)
    async with worker.serving():
        run_id = await _started(
            worker, graph.graph, {"question": "y", "text": "x", "seconds": 60.0}
        )
        await asyncio.to_thread(graph.started.wait, 60)
        row = store.rows[run_id]
        # write the row the way another worker does when it takes over a run whose lease ran out
        assert await store.update(
            run_id, row.rev, {"owner": "another-worker", "lease_until": time.time() + 60}
        )

        await asyncio.to_thread(graph.returned.wait, 60)  # the holder learns it at its next renewal
    assert graph.seen == [True]


async def test_a_node_sees_the_stop_event_set_when_the_store_raises_and_the_worker_gives_the_run_up() -> (
    None
):
    """A worker that gives a run up tells the call in progress to stop.

    Once the lease runs out, another worker goes on with the run, so this call shouldn't keep working on it.
    """
    graph = stopping()

    class DownForEvents(MemoryStore):
        """Raises when the ``node_start`` event of ``stream`` is written, once ``stream`` has been called."""

        async def add_event(self, run_id: str, owner: str, body: str) -> bool:
            if json.loads(body)["type"] == "node_start" and "stream" in body:
                await asyncio.to_thread(graph.started.wait, 60)
                raise ConnectionError("the store is down")
            return await super().add_event(run_id, owner, body)

    store = DownForEvents()
    worker = _worker(store, graph.registry, context=stoppable)
    async with worker.serving():
        await _started(worker, graph.graph, {"question": "y", "text": "x", "seconds": 60.0})
        await asyncio.to_thread(graph.returned.wait, 60)
    assert graph.seen == [True]


async def test_an_attempt_that_timed_out_keeps_seeing_the_stop_event_of_its_resume_call_set_after_the_next_one_starts() -> (
    None
):
    """The old call's stop event stays set, even after the next resume call starts with a new one.

    ``slow`` sets its own resume call's stop event, as a cancel would, and that cancels the call. ``slow`` then keeps
    running on its thread past its ``attempt_timeout``, so the first resume call only ends once that attempt has
    timed out.

    The store holds back the write that ends the first resume call until an answer is queued. That answer starts the
    next resume call, which gets a new stop event. The first call's event stays set.

    Each step waits on the one before it, never on a clock.
    """
    registry, release, events = NodeRegistry(context_type=Stoppable), ThreadEvent(), []

    @registry.node("ask", title="Ask", description="For a test.", asks=True)
    def ask(question: str) -> bool:
        raise Ask("Go on?")

    @registry.node(
        "slow", title="Slow", description="For a test.", policy=Policy(attempt_timeout=0.2)
    )
    def slow(ctx: Context[Stoppable], text: str) -> str:
        events.append(ctx.stop)
        if len(events) == 1:
            ctx.stop.set()  # as a cancel sets it, before the attempt can time out
            release.wait(60)  # ignores the stop event, like a call that can't be stopped
        return text

    @registry.node("after", title="After", description="For a test.")
    def after(text: str, ok: bool) -> str:
        return text

    graph = make_graph(
        ask,
        slow,
        after,
        edges=[("slow.result", "after.text"), ("ask.result", "after.ok")],
        outputs={"after": "after.result"},
    )
    inputs = {"question": "y", "text": "x"}
    release.set()
    (waiting,) = (
        Graph.compile(graph, registry).run(inputs, context=Stoppable(ThreadEvent())).waiting
    )
    release.clear()
    events.clear()

    class HoldsTheFirstEnd(MemoryStore):
        """Holds back the write that ends the first resume call until ``let_end`` is set.

        ``ending`` is set as soon as that write arrives.
        """

        ending, let_end = AsyncioEvent(), AsyncioEvent()

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if changes.get("status") not in (None, "running") and not self.let_end.is_set():
                self.ending.set()
                await self.let_end.wait()
            return await super().update(run_id, rev, changes, event)

    store = HoldsTheFirstEnd()
    worker = _worker(store, registry, context=stoppable)
    try:
        async with worker.serving():
            run_id = await _started(worker, graph, inputs)
            await store.ending.wait()  # the attempt timed out, so the first resume call ended failed, though cancelled first
            assert (
                await worker.answer(run_id, {waiting.id: True}, allows=EVERY_RUN)
            ).status == "running"
            store.let_end.set()

            completed = await store.until(run_id, lambda row: row.status == "complete")
            ends = [
                event["status"]
                for event in await store.stored_events(run_id)
                if event["type"] == "run_end"
            ]
            assert (completed.status, ends) == ("complete", ["failed", "complete"])
    finally:
        release.set()
    first, second = events
    assert first.is_set() and not second.is_set()


# -- answers --------------------------------------------------------------------------------------


async def test_two_workers_answering_one_run_at_once_start_one_resume_call() -> None:
    """Both workers answer ``running``, but only one resume call starts for the two answers.

    The worker that loses the race to claim the run reads it again and finds it running, so it queues its answer.
    The run then refuses that answer as ``not_waiting``, since the other worker already answered the question.
    """
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    one, other = _worker(store, registry), _worker(store, registry)
    async with one.serving(), other.serving():
        run_id = await _started(one, graph, inputs)
        await store.ended(run_id)
        outcomes = await asyncio.gather(
            one.answer(run_id, {waiting_id: True}, allows=EVERY_RUN),
            other.answer(run_id, {waiting_id: True}, allows=EVERY_RUN),
        )

        assert [outcome.status for outcome in outcomes] == ["running", "running"]
        ended = await store.ended(run_id)
        assert (ended.status, list(ended.refused)) == ("complete", [waiting_id])
        assert [problem.type for problem in ended.refused[waiting_id]] == ["not_waiting"]
        assert await _resume_calls(store, run_id) == 2


async def test_a_second_answer_queued_for_a_question_is_refused_as_not_waiting_when_the_run_stops() -> (
    None
):
    """The first answer is taken. When the second is applied, nothing waits on the question any more.

    The run still has calls to make at that point, so the refusal's ``ctx.status`` is ``ready``.
    """
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)

        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        assert (
            await worker.answer(run_id, {waiting_id: False}, allows=EVERY_RUN)
        ).status == "running"
        gate.set()
        ended = await store.ended(run_id)

        assert ended.outputs == {"busy": "x", "after": True}
        assert [(problem.type, problem.ctx["status"]) for problem in ended.refused[waiting_id]] == [
            ("not_waiting", "ready")
        ]


async def test_a_wrong_first_answer_and_a_right_second_leave_nothing_refused() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await worker.answer(run_id, {waiting_id: "not a bool"}, allows=EVERY_RUN)
        await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        gate.set()
        ended = await store.ended(run_id)

    assert (ended.status, ended.refused, ended.outputs) == (
        "complete",
        {},
        {"busy": "x", "after": True},
    )


async def test_an_answer_to_a_complete_run_is_refused_by_conductor_as_not_waiting() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await store.ended(run_id)
        await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        assert (await store.ended(run_id)).status == "complete"

        with pytest.raises(InvalidCallError) as refused:
            await worker.answer(run_id, {waiting_id: False}, allows=EVERY_RUN)

    assert [(problem.type, problem.ctx["status"]) for problem in refused.value.problems] == [
        ("not_waiting", "complete")
    ]


# -- declines -------------------------------------------------------------------------------------


class _UpdatesKept(MemoryStore):
    """A ``MemoryStore`` that keeps the changes of every ``update`` it took, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[RunChanges] = []

    async def update(
        self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
    ) -> bool:
        written = await super().update(run_id, rev, changes, event)
        if written:
            self.updates.append(changes)
        return written


async def test_a_decline_of_a_waiting_run_ends_it_declined_for_good_with_one_run_end_and_no_claim() -> (
    None
):
    """The run is given the decline once and written as it ends: never ``running``, no lease, no resume call.

    Its events get one ``run_end`` that says what the row says, so a host that follows them learns the run ended.
    The run has ended for good: a resume is ``run_not_resumable``, and an answer or a decline is ``not_waiting``.
    """
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = _UpdatesKept()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await store.ended(run_id)
        events, updates = len(await store.stored_events(run_id)), len(store.updates)

        written = await worker.decline(run_id, waiting_id, reason="Over budget", allows=EVERY_RUN)

        row = store.rows[run_id]
        assert written == await store.view(run_id)
        assert (written.status, written.cause.type, written.cause.ctx, written.waiting) == (
            "declined",
            "answer_declined",
            {"reason": "Over budget"},
            [],
        )
        assert (row.status, row.owner, row.lease_until) == ("declined", None, None)
        assert [changes.get("status") for changes in store.updates[updates:]] == ["declined"]
        assert [
            (event["type"], event["status"], event["cause"]["type"])
            for event in (await store.stored_events(run_id))[events:]
        ] == [("run_end", "declined", "answer_declined")]
        assert await _resume_calls(store, run_id) == 1

        refused = []
        for later in (
            worker.resume(run_id, allows=EVERY_RUN),
            worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN),
            worker.decline(run_id, waiting_id, allows=EVERY_RUN),
        ):
            with pytest.raises(InvalidCallError) as refusal:
                await later
            refused.extend(
                (problem.type, problem.ctx["status"]) for problem in refusal.value.problems
            )
        assert refused == [
            ("run_not_resumable", "declined"),
            ("not_waiting", "declined"),
            ("not_waiting", "declined"),
        ]
        assert store.rows[run_id] == row


async def test_a_worker_that_never_served_declines_a_waiting_run() -> None:
    """A declined run is never resumed, so declining it needs no worker that hands runs back."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    served = _worker(store, registry)
    async with served.serving():
        run_id = await _started(served, graph, inputs)
        await store.ended(run_id)

    written = await _worker(store, registry).decline(run_id, waiting_id, allows=EVERY_RUN)

    assert (written.status, written.cause.type) == ("declined", "answer_declined")


def _queued_decline(waiting_id: str) -> dict[str, Any]:
    return {"kind": "decline", "waiting_id": waiting_id, "reason": "First"}


def _event_types(events: list[dict[str, Any]]) -> list[str]:
    return [
        ":".join(
            [event["type"], event.get("status") or "", (event.get("cause") or {}).get("type", "")]
        )
        for event in events
    ]


@pytest.mark.parametrize(
    "kind",
    ["answer", "decline"],
)
def _two_questions(
    busy_fails_once: bool,
) -> tuple[NodeRegistry, GraphRecord, dict[str, Any], dict[str, str]]:
    """A graph that asks ``a_first`` at once and ``c_second`` only once ``b_busy`` has its result.

    Nodes no edge puts in order are called in the order of their ids. With ``busy_fails_once``, ``b_busy``'s next
    call fails, so the run as it then stops asks only ``a_first``. Returns the registry, the graph, its inputs, and
    the waiting id of each question, by node id, as a run asks them once ``b_busy`` has its result.
    """
    registry, fails = NodeRegistry(), [False]

    @registry.node("b_busy", title="Busy", description="Fails once when told to.")
    def busy(text: str) -> str:
        if fails[0]:
            fails[0] = False
            raise NodeError("not yet")
        return text

    @registry.node("a_first", title="First", description="Asks.", asks=True)
    def first(question: str) -> bool:
        raise Ask("first?")

    @registry.node(
        "c_second", title="Second", description="Asks once busy has its result.", asks=True
    )
    def second(text: str) -> bool:
        raise Ask("second?")

    @registry.node("z_after", title="After", description="Joins the answers.")
    def after(a: bool, b: bool) -> str:
        return f"{a}/{b}"

    graph = make_graph(
        busy,
        first,
        second,
        after,
        edges=[
            ("b_busy.result", "c_second.text"),
            ("a_first.result", "z_after.a"),
            ("c_second.result", "z_after.b"),
        ],
        outputs={"after": "z_after.result"},
    )
    inputs = {"text": "x", "question": "y"}
    ids = {
        str(waiting.loc[0]): waiting.id
        for waiting in Graph.compile(graph, registry).run(inputs).waiting
    }
    fails[0] = busy_fails_once
    return registry, graph, inputs, ids


async def test_a_decline_ends_every_question_so_a_decline_of_another_is_not_waiting() -> None:
    """A run that waits on two questions is declined on one. The run has ended, so the other waits no more, and a
    decline of it is refused with nothing written."""
    registry, graph, inputs, ids = _two_questions(busy_fails_once=False)
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await store.ended(run_id)
        written = await worker.decline(run_id, ids["a_first"], allows=EVERY_RUN)
        row = store.rows[run_id]

        with pytest.raises(InvalidCallError) as refused:
            await worker.decline(run_id, ids["c_second"], reason="No", allows=EVERY_RUN)

    assert (written.status, written.waiting) == ("declined", [])
    assert [(problem.type, problem.ctx["status"]) for problem in refused.value.problems] == [
        ("not_waiting", "declined")
    ]
    assert store.rows[run_id] == row and await _resume_calls(store, run_id) == 1


async def test_one_call_that_answers_one_waiting_call_twice_is_refused_on_a_stopped_run() -> None:
    """``run.with_answers`` refuses a mapping that answers one call by its id and by its node id, and nothing is
    answered; the worker gives a stopped run's answers to ``run.with_answers`` as they came."""
    registry, graph, inputs, ids = _two_questions(busy_fails_once=False)
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await store.ended(run_id)
        before = store.rows[run_id]

        with pytest.raises(InvalidCallError) as refused:
            await worker.answer(run_id, {ids["a_first"]: True, "a_first": False}, allows=EVERY_RUN)

    assert [problem.type for problem in refused.value.problems] == ["answer_given_twice"]
    assert store.rows[run_id] == before


async def test_a_decline_queued_during_a_resume_call_that_raised_ends_the_run_then_so_no_later_reply_calls_a_node() -> (
    None
):
    """A person declines ``a_first`` while the run runs, and the resume call then raises outside any node. The run
    as the row has it takes the queue as the call fails, so it ends ``declined`` at once, as it would on a running
    run. A later answer to ``c_second`` is ``not_waiting`` and a resume is ``run_not_resumable``: no node is called
    after the person said stop."""
    registry, graph, inputs, ids = _two_questions(busy_fails_once=False)
    checkpoint = Graph.compile(graph, registry).run(inputs)
    assert sorted(waiting.loc[0] for waiting in checkpoint.waiting) == ["a_first", "c_second"]
    store, calls = MemoryStore(), []

    def raises(metadata: object, stop: ThreadEvent) -> None:
        calls.append(metadata)
        raise ConnectionError("the secrets service is down")

    worker = _worker(store, registry, context=raises)
    await store.create(left_running("left", graph, checkpoint, owner="a-dead-worker"))
    queued = await worker.decline("left", ids["a_first"], reason="No", allows=EVERY_RUN)
    assert queued.status == "running"
    async with worker.serving():
        ended = await store.ended("left")
    row = store.rows["left"]

    refused = []
    for later in (
        worker.answer("left", {ids["c_second"]: True}, allows=EVERY_RUN),
        worker.resume("left", allows=EVERY_RUN),
    ):
        with pytest.raises(InvalidCallError) as refusal:
            await later
        refused.extend((problem.type, problem.ctx["status"]) for problem in refusal.value.problems)

    assert (ended.status, ended.cause.type, ended.cause.ctx, ended.waiting) == (
        "declined",
        "answer_declined",
        {"reason": "No"},
        [],
    )
    assert store.state("left")["queued"] == []
    assert _event_types(await store.stored_events("left"))[-1] == "run_end:declined:answer_declined"
    assert refused == [("not_waiting", "declined"), ("run_not_resumable", "declined")]
    assert store.rows["left"] == row and len(calls) == 1


class _FailsAnUpdateWithAnEvent(MemoryStore):
    """A ``MemoryStore`` whose ``update`` raises when it is given an event once ``failing`` is set, as a database that
    goes away would."""

    failing = False

    async def update(
        self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
    ) -> bool:
        if self.failing and event is not None:
            raise ConnectionError("the database went away")
        return await super().update(run_id, rev, changes, event)


async def test_a_decline_whose_write_the_store_fails_writes_neither_the_row_nor_its_run_end() -> (
    None
):
    """The failed row and its ``run_end`` are one write, so an outage leaves the run waiting, as it was, with no
    ``run_end`` to repair. The decline raises the outage."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = _FailsAnUpdateWithAnEvent()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await store.ended(run_id)
    before, events = store.rows[run_id], await store.stored_events(run_id)
    store.failing = True

    with pytest.raises(StoreFailedError):
        await worker.decline(run_id, waiting_id, allows=EVERY_RUN)

    assert (store.rows[run_id], await store.stored_events(run_id)) == (before, events)


class _FollowedAsItEnds(MemoryStore):
    """A ``MemoryStore`` that awaits ``hook`` once, right after an ``update`` with an event lands. ``read`` is set
    each time ``events`` is called."""

    def __init__(self) -> None:
        super().__init__()
        self.hook: Callable[[], Awaitable[None]] | None = None
        self.read = AsyncioEvent()

    async def update(
        self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
    ) -> bool:
        written = await super().update(run_id, rev, changes, event)
        if written and event is not None and self.hook is not None:
            hook, self.hook = self.hook, None
            await hook()
        return written

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        self.read.set()
        return await super().events(run_id, after)


async def test_a_follower_that_connects_the_moment_a_decline_is_written_gets_its_run_end() -> None:
    """The row and its ``run_end`` are written together, so a follower that reads the row stopped has the event."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = _FollowedAsItEnds()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await store.ended(run_id)
        last = len(await store.stored_events(run_id)) - 1
        got: list[str] = []

        async def follow() -> None:
            async for _, body in await worker.events(run_id, last, allows=EVERY_RUN):
                got.append(json.loads(body)["type"])

        following: list[Task[None]] = []

        async def connect() -> None:
            store.read.clear()
            following.append(asyncio.create_task(follow()))
            await store.read.wait()  # the follower has read the row and asked for its events

        store.hook = connect
        await worker.decline(run_id, waiting_id, allows=EVERY_RUN)
        await asyncio.wait_for(following[0], 5)

    assert got == ["run_end"]


class _EventsByMethod(MemoryStore):
    """A ``MemoryStore`` that keeps, for each event, whether it came through ``update`` or ``add_event``."""

    def __init__(self) -> None:
        super().__init__()
        self.came: list[tuple[str, str]] = []

    async def update(
        self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
    ) -> bool:
        written = await super().update(run_id, rev, changes, event)
        if written and event is not None:
            self.came.append(("update", json.loads(event)["type"]))
        return written

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        added = await super().add_event(run_id, owner, body)
        if added:
            self.came.append(("add_event", json.loads(body)["type"]))
        return added


async def test_a_held_runs_run_end_is_written_in_the_update_that_writes_how_it_stopped() -> None:
    """A run that ends while a worker holds it gets its ``run_end`` the one way any run does: with its row."""
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = _EventsByMethod()
    worker = _worker(store, registry)
    async with worker.serving():
        await store.ended(await _started(worker, graph, inputs))

    assert [came for came in store.came if came[1] == "run_end"] == [("update", "run_end")]


async def test_a_decline_sent_while_a_resume_call_runs_is_queued_and_ends_the_run_when_it_stops() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)

        assert (await worker.decline(run_id, waiting_id, allows=EVERY_RUN)).status == "running"
        gate.set()
        ended = await store.ended(run_id)

    assert (ended.status, ended.cause.type, ended.outputs, ended.refused) == (
        "declined",
        "answer_declined",
        None,
        {},
    )
    assert await _resume_calls(store, run_id) == 1


@pytest.mark.parametrize("cancel", [False, True], ids=["alone", "with-a-cancel"])
async def test_a_decline_sent_while_another_call_runs_ends_the_runs_events_declined(
    cancel: bool,
) -> None:
    """The last ``run_end`` stored says what the row says: ``declined`` with ``answer_declined``.

    ``approve`` asks, then ``busy`` waits on the gate, so the decline is queued while the question waits. The resume
    call's own ``run_end`` says ``waiting``, or ``cancelled`` with a cancel asked too, and a host that follows the
    events would never learn that the run was declined if it were the last.
    """
    registry, gate = NodeRegistry(), ThreadEvent()

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(amount: int) -> bool:
        raise Ask("Approve?")

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    graph = make_graph(
        approve, busy, edges=[], outputs={"ok": "approve.result", "text": "busy.result"}
    )
    inputs = {"amount": 5, "text": "x"}
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        while not (
            asked := [
                event["waiting"]["id"]
                for event in await store.stored_events(run_id)
                if event["type"] == "node_waiting"
            ]
        ):
            await asyncio.sleep(0.01)
        await worker.decline(run_id, asked[0], allows=EVERY_RUN)
        if cancel:
            await worker.cancel(run_id, allows=EVERY_RUN)
        gate.set()
        ended = await store.ended(run_id)

    last = (await store.stored_events(run_id))[-1]
    assert (ended.status, ended.cause.type) == ("declined", "answer_declined")
    assert (last["type"], last["status"], last["cause"]["type"]) == (
        "run_end",
        "declined",
        "answer_declined",
    )


async def test_an_answer_or_a_reason_json_cant_write_is_refused_alike_whether_or_not_the_run_runs() -> (
    None
):
    """An answer or a reason JSON can't write is refused with the problem the run gives it, whether the run is running
    (and the answer would be queued) or not. Either way the row is as it was.
    """
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)

    async def refusals(run_id: str) -> list[list[Problem]]:
        refused = []
        before = store.rows[run_id]
        for sent in (
            worker.answer(run_id, {waiting_id: "\ud800"}, allows=EVERY_RUN),
            worker.decline(run_id, waiting_id, reason="\ud800", allows=EVERY_RUN),
        ):
            with pytest.raises(InvalidCallError) as refusal:
                await sent
            refused.append(refusal.value.problems)
        assert store.rows[run_id] == before
        return refused

    async with worker.serving():
        running = await _started(worker, graph, inputs)
        while_running = await refusals(running)
        gate.set()
        await store.ended(running)
        stopped = await refusals(running)

    assert while_running == stopped
    assert [[problem.type for problem in each] for each in stopped] == [
        ["invalid_answer"],
        ["invalid_decline_reason"],
    ]


async def test_an_answer_pydantic_cant_write_itself_is_queued_as_its_json_and_taken_when_the_run_stops() -> (
    None
):
    registry, gate = NodeRegistry(), ThreadEvent()

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    @registry.node("parts", title="Parts", description="For a test.", asks=True)
    def parts(question: str) -> Table[Part]:
        raise Ask("Which parts?")

    graph = make_graph(
        busy, parts, edges=[], outputs={"busy": "busy.result", "parts": "parts.result"}
    )
    inputs = {"text": "x", "question": "y"}
    gate.set()
    (waiting,) = Graph.compile(graph, registry).run(inputs).waiting
    gate.clear()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        answer = pl.DataFrame({"text": ["bolt", "nut"]})
        await worker.answer(run_id, {waiting.id: answer}, allows=EVERY_RUN)
        gate.set()
        ended = await store.ended(run_id)

    assert (ended.status, ended.outputs) == (
        "complete",
        {
            "busy": "x",
            "parts": {"columns": [{"name": "text", "type": "str", "values": ["bolt", "nut"]}]},
        },
    )


async def test_a_decline_to_a_complete_run_is_not_waiting_and_to_a_stale_id_answer_stale() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        waits = await _started(worker, graph, inputs)
        await store.ended(waits)
        with pytest.raises(InvalidCallError) as stale:
            await worker.decline(waits, waiting_id.split("~")[0] + "~00000000", allows=EVERY_RUN)
        assert (await store.view(waits)).status == "waiting"

        await worker.answer(waits, {waiting_id: True}, allows=EVERY_RUN)
        assert (await store.ended(waits)).status == "complete"
        with pytest.raises(InvalidCallError) as ended:
            await worker.decline(waits, waiting_id, allows=EVERY_RUN)

    assert [problem.type for problem in stale.value.problems] == ["answer_stale"]
    assert [(problem.type, problem.ctx["status"]) for problem in ended.value.problems] == [
        ("not_waiting", "complete")
    ]


async def test_a_run_taken_over_with_a_decline_it_cant_take_yet_is_resumed_and_given_the_decline_when_it_stops() -> (
    None
):
    """A worker died while the run ran, before it asked the question that a decline queued meanwhile names.

    The worker that takes the run over resumes it, since the run refuses the decline as it is. The run asks the
    question, stops, and takes the decline then. A decline for an id that never waits is refused once the run
    stops, and the run ends as its resume call left it.
    """
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    never_asked = waiting_id.replace("confirm", "after", 1)
    for run_id, declined in (("asked", waiting_id), ("never", never_asked)):
        ready = Graph.compile(graph, registry).with_inputs(inputs)
        await store.create(left_running(run_id, graph, ready, owner="a-dead-worker"))
        assert (await worker.decline(run_id, declined, allows=EVERY_RUN)).status == "running"

    async with worker.serving():
        asked, never = await store.ended("asked"), await store.ended("never")

    assert (asked.status, asked.cause.type, asked.refused) == ("declined", "answer_declined", {})
    assert (never.status, [waiting.id for waiting in never.waiting]) == ("waiting", [waiting_id])
    assert [problem.type for problem in never.refused[never_asked]] == ["not_waiting"]


@pytest.mark.parametrize("owner", ["a-dead-worker", None])
async def test_a_running_run_taken_over_with_a_queued_decline_ends_declined_when_its_resume_call_stops(
    owner: str | None,
) -> None:
    """A worker died, or handed the run back, after the run asked its question and a decline of it was queued. The
    worker that takes the run over resumes it, and gives it the queue when that call stops, as the worker that died
    would have: the run ends ``declined``, and its last ``run_end`` says so."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    checkpoint = Graph.compile(graph, registry).run(inputs)
    store = MemoryStore()
    row = left_running("run", graph, checkpoint, owner=owner)
    state = {**json.loads(row.state), "queued": [_queued_decline(waiting_id)]}
    await store.create(dataclasses.replace(row, state=json.dumps(state)))

    async with _worker(store, registry).serving():
        ended = await store.ended("run")

    assert (ended.status, ended.cause and ended.cause.type) == ("declined", "answer_declined")
    assert _event_types(await store.stored_events("run"))[-1] == "run_end:declined:answer_declined"
    assert (store.rows["run"].owner, store.rows["run"].lease_until) == (None, None)


async def test_an_answer_queued_during_a_resume_call_starts_the_next_however_long_the_first_takes_to_write_its_end() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()

    class SlowEnd(MemoryStore):
        """Delays the write that ends a resume call by 0.3 seconds, the time of three lease renewals.

        That is the write that takes the queued answers.
        """

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if {"run", "state"} <= changes.keys():
                await asyncio.sleep(0.3)
            return await super().update(run_id, rev, changes, event)

    store = SlowEnd()
    worker = _worker(store, registry, lease=0.3)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        gate.set()
        ended = await store.ended(run_id)

    assert (ended.status, ended.refused, ended.outputs) == (
        "complete",
        {},
        {"busy": "x", "after": True},
    )
    assert await _resume_calls(store, run_id) == 2


async def test_the_end_of_a_resume_call_costs_one_read_one_event_and_one_write_also_with_an_answer_queued() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()

    class Counts(MemoryStore):
        """Counts the reads, and the attempts to write, whether they succeed or not."""

        def __init__(self) -> None:
            super().__init__()
            self.reads = self.writes = 0

        async def get(self, run_id: str) -> StoredRun | None:
            self.reads += 1
            return await super().get(run_id)

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            self.writes += 1
            return await super().update(run_id, rev, changes, event)

    store = Counts()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        store.reads = store.writes = 0
        gate.set()

        assert (await store.ended(run_id)).status == "complete"
        assert (store.reads, store.writes) == (
            2,
            2,
        )  # one of each for the end of each of the two resume calls
        assert [event["type"] for event in await store.stored_events(run_id)].count("run_end") == 2


async def test_an_answer_queued_just_before_a_resume_call_writes_its_end_starts_the_next_resume_call() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()

    class AnswerAtTheEnd(MemoryStore):
        """Has another worker answer the run just before the write that ends the running resume call.

        Since that resume call is still running, the answer is queued for it.
        """

        answered = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if not self.answered and changes.get("status") == "waiting":
                self.answered = True
                assert (
                    await other.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
                ).status == "running"
            return await super().update(run_id, rev, changes, event)

    store = AnswerAtTheEnd()
    worker, other = _worker(store, registry), _worker(store, registry)
    async with worker.serving(), other.serving():
        run_id = await _started(worker, graph, inputs)
        ended = await store.ended(run_id)

        assert store.answered and (ended.status, ended.refused, ended.outputs) == (
            "complete",
            {},
            {"busy": "x", "after": True},
        )
        assert [
            event["status"]
            for event in await store.stored_events(run_id)
            if event["type"] == "run_end"
        ] == ["waiting", "complete"]


async def test_a_run_that_takes_longer_than_the_lease_to_write_stays_with_its_worker_and_is_written_once(
    monkeypatch: MonkeyPatch,
) -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    slow_writes = []
    to_json = Run.to_json

    def slow_to_json(self: Run) -> str:
        if self.status == "waiting":  # the run as written at the end of the resume call
            slow_writes.append(self.status)
            time.sleep(0.5)  # like a large run, it takes longer than the lease
        return to_json(self)

    monkeypatch.setattr(Run, "to_json", slow_to_json)
    store = MemoryStore()
    one, other = _worker(store, registry, lease=0.3), _worker(store, registry, lease=0.3)
    async with one.serving(), other.serving():
        run_id = await _started(one, graph, inputs)
        ended = await store.ended(run_id)
        # a second worker that took the run would be running it again by now
        await asyncio.sleep(0.4)

    assert [waiting_call.id for waiting_call in ended.waiting] == [waiting_id] and (
        await one.read(run_id, allows=EVERY_RUN)
    ) == ended
    assert await _resume_calls(store, run_id) == 1
    assert slow_writes == ["waiting"]


async def test_an_answer_queued_during_a_resume_call_stopped_at_the_item_limit_starts_the_next_resume_call() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking(items=True)
    store = MemoryStore()
    worker = _worker(store, registry, max_items=1)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        gate.set()
        ended = await store.ended(run_id)  # the next resume call stops at the same item limit

        assert (ended.status, ended.waiting, ended.refused) == ("stopped", [], {})
        assert await _resume_calls(store, run_id) == 2


async def test_an_answer_queued_for_a_question_a_timed_out_resume_call_never_asked_is_refused_as_not_waiting() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking(items=True)
    store = MemoryStore()
    worker = _worker(store, registry, resume_timeout=0.2)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        while await _resume_calls(store, run_id) == 0:  # its timeout counts from when it starts
            await asyncio.sleep(0.01)
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        # wait past the resume call's timeout, then let the node call it waits for finish
        await asyncio.sleep(0.3)
        gate.set()
        ended = await store.ended(run_id)

        assert ended.status == "timed_out"
        assert [(problem.type, problem.ctx["status"]) for problem in ended.refused[waiting_id]] == [
            ("not_waiting", "timed_out")
        ]
        assert await _resume_calls(store, run_id) == 1


async def test_an_answer_queued_before_a_cancelled_run_asks_its_question_is_refused_as_not_waiting_and_starts_nothing() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "running"
        gate.set()
        ended = await store.ended(run_id)

        assert ended.status == "cancelled"
        assert [(problem.type, problem.ctx["status"]) for problem in ended.refused[waiting_id]] == [
            ("not_waiting", "cancelled")
        ]
        assert await _resume_calls(store, run_id) == 1


async def test_an_answer_queued_for_a_question_the_checkpoint_hasnt_asked_is_refused_loudly_when_the_resume_call_raises() -> (
    None
):
    """The run as the row has it takes the queue as the call fails, so a row that stops holds none. An answer to a
    question the run asked after its last checkpoint finds nothing waiting there: it is ``not_waiting`` in
    ``refused``, never dropped unseen, and the question waits again on the next resume."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    resume_calls = []

    def fails_on_the_first_resume_call(metadata: object, stop: ThreadEvent) -> None:
        resume_calls.append(metadata)
        if len(resume_calls) == 1:
            raise ConnectionError("the secrets service is down")

    store = MemoryStore()
    worker = _worker(store, registry, context=fails_on_the_first_resume_call)
    left = left_running(
        "left", graph, Graph.compile(graph, registry).with_inputs(inputs), owner="a-dead-worker"
    )
    await store.create(left)
    assert (await worker.answer("left", {waiting_id: True}, allows=EVERY_RUN)).status == "running"

    async with worker.serving():
        failed = await store.ended("left")
        assert store.state("left")["queued"] == []
        assert (failed.status, failed.cause.type) == ("failed", "raised_outside_a_node")
        assert {key: [p.type for p in value] for key, value in failed.refused.items()} == {
            waiting_id: ["not_waiting"]
        }

        await worker.resume("left", allows=EVERY_RUN)
        asked_again = await store.ended("left")
    assert (asked_again.status, [waiting.id for waiting in asked_again.waiting]) == (
        "waiting",
        [waiting_id],
    )


async def test_an_answer_that_is_taken_leaves_nothing_refused_or_queued_for_its_question() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting_id: "not a bool"}, allows=EVERY_RUN)
        ).status == "running"
        gate.set()
        waiting = await store.ended(run_id)
        assert (waiting.status, list(waiting.refused)) == ("waiting", [waiting_id])

        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        completed = await store.ended(run_id)
        assert (completed.status, completed.refused, completed.outputs) == (
            "complete",
            {},
            {"busy": "x", "after": True},
        )


@host_type("fragile", title="Fragile")
class Fragile:
    """A host's type with a bug in its own check: it raises a ``KeyError`` instead of refusing the value."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def check(value: Any) -> Fragile:
            raise KeyError("a lookup the host's type forgot to guard")

        return core_schema.no_info_plain_validator_function(
            check,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _value: "fragile"
            ),
        )


async def test_a_queued_answer_whose_check_raises_is_refused_and_starts_no_second_resume_call() -> (
    None
):
    registry, store, gate = NodeRegistry(), MemoryStore(), ThreadEvent()

    @registry.node("ask", title="Ask", description="For a test.", asks=True)
    def ask(question: str) -> Fragile:
        raise Ask("Which?")

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    graph = make_graph(ask, busy, edges=[], outputs={"ask": "ask.result", "busy": "busy.result"})
    inputs = {"question": "y", "text": "x"}
    gate.set()
    (waiting,) = Graph.compile(graph, registry).run(inputs).waiting
    gate.clear()
    worker = _worker(store, registry, lease=0.15)

    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting.id: "anything"}, allows=EVERY_RUN)
        ).status == "running"
        gate.set()
        ended = await store.ended(run_id)
        await asyncio.sleep(0.4)  # long enough for a resume call given up to be started again

    assert (
        ended.status,
        [(problem.type, problem.ctx) for problem in ended.refused[waiting.id]],
    ) == (
        "waiting",
        [
            (
                "invalid_answer",
                {
                    "waiting_id": waiting.id,
                    "reason": "The value did not pass its type's own check",
                    "errors": [
                        {
                            "type": "value_error",
                            "loc": [],
                            "msg": "The value did not pass its type's own check",
                        }
                    ],
                },
            )
        ],
    )
    assert await _resume_calls(store, run_id) == 1
    # No host reads the queue, and an answer left in it would only show at the run's next stop. This run never stops
    # again, since only a fixed type could answer its question. So the test reads the stored state directly.
    assert store.state(run_id)["queued"] == []


async def test_a_cancel_of_a_run_an_answer_resumed_stops_the_answered_resume_call() -> None:
    registry, graph, gates, calls = gated()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, {"csv": "a,b", "text": "x", "question": "y"})
        gates["first"].set()
        waiting = await store.ended(run_id)
        assert (
            await worker.answer(run_id, {waiting.waiting[0].id: True}, allows=EVERY_RUN)
        ).status == "running"
        await asyncio.to_thread(gates["per_item_called"].wait, 5)

        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "running"
        gates["second"].set()
        cancelled = await store.ended(run_id)
        assert (cancelled.status, cancelled.waiting, calls) == ("cancelled", [], ["a"])

        await worker.resume(run_id, allows=EVERY_RUN)  # the answer is kept: nothing is asked again
        completed = await store.ended(run_id)
        assert (completed.status, completed.outputs, calls) == (
            "complete",
            {"busy": "x", "per_item": ["a", "b"]},
            ["a", "b"],
        )


async def test_a_cancel_of_a_run_an_answer_completed_leaves_it_complete() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (await store.ended(run_id)).status == "waiting"
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        assert (await store.ended(run_id)).status == "complete"

        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "complete"
        assert (await worker.read(run_id, allows=EVERY_RUN)).status == "complete"


def _asked_beside_a_chain() -> tuple[
    NodeRegistry, GraphRecord, dict[str, Any], dict[str, ThreadEvent], list[str], str
]:
    """Builds a graph with a question beside a chain of three steps.

    Step ``one`` waits on the gate ``first``. Step ``two`` sets ``in_second``, then waits on the gate ``second``.

    Returns:
        The registry, the graph, its inputs, the gates by name, ``calls`` (the steps called, in order), and the
        waiting id of the question.
    """
    registry, gates, calls = (
        NodeRegistry(),
        {name: ThreadEvent() for name in ("first", "in_second", "second")},
        [],
    )

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(question: str) -> bool:
        raise Ask("Go on?")

    @registry.node("one", title="One", description="For a test.")
    def one(text: str) -> str:
        calls.append("one")
        gates["first"].wait(5)
        return text

    @registry.node("two", title="Two", description="For a test.")
    def two(text: str) -> str:
        calls.append("two")
        gates["in_second"].set()
        gates["second"].wait(5)
        return text

    @registry.node("three", title="Three", description="For a test.")
    def three(text: str) -> str:
        calls.append("three")
        return text

    graph = make_graph(
        approve,
        one,
        two,
        three,
        edges=[("one.result", "two.text"), ("two.result", "three.text")],
        outputs={"approve": "approve.result", "three": "three.result"},
    )
    inputs = {"question": "y", "text": "x"}
    for gate in gates.values():
        gate.set()
    (waiting,) = Graph.compile(graph, registry).run(inputs).waiting
    for gate in gates.values():
        gate.clear()
    calls.clear()
    return registry, graph, inputs, gates, calls, waiting.id


async def test_a_cancel_ends_the_resume_call_it_reached_and_not_the_one_an_answer_queued_meanwhile_starts() -> (
    None
):
    registry, graph, inputs, gates, calls, waiting_id = _asked_beside_a_chain()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        gates["first"].set()
        await asyncio.to_thread(gates["in_second"].wait, 5)
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        assert (await worker.cancel(run_id, allows=EVERY_RUN)).status == "running"
        gates["second"].set()

        completed = await store.until(
            run_id, lambda row: row.status not in ("running", "cancelled")
        )
        ends = [
            event["status"]
            for event in await store.stored_events(run_id)
            if event["type"] == "run_end"
        ]
        assert (completed.status, ends, calls) == (
            "complete",
            ["cancelled", "complete"],
            ["one", "two", "three"],
        )


async def test_a_cancel_sent_as_a_resume_call_ends_cancels_the_one_an_answer_queued_meanwhile_starts() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()

    class CancelAtTheEnd(MemoryStore):
        """Has another worker cancel the run just before the write that ends the first resume call."""

        cancelled = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if not self.cancelled and "outcome" in changes:
                self.cancelled = True
                assert (await other.cancel(run_id, allows=EVERY_RUN)).status == "running"
            return await super().update(run_id, rev, changes, event)

    store = CancelAtTheEnd()
    worker, other = _worker(store, registry), _worker(store, registry)
    async with worker.serving(), other.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(run_id, {waiting_id: True}, allows=EVERY_RUN)
        ).status == "running"
        gate.set()

        ended = await store.ended(run_id)
        ends = [
            event["status"]
            for event in await store.stored_events(run_id)
            if event["type"] == "run_end"
        ]
        assert store.cancelled and (ended.status, ends, ended.outputs) == (
            "cancelled",
            ["waiting", "cancelled"],
            None,
        )
        await worker.resume(
            run_id, allows=EVERY_RUN
        )  # the cancel that ended the last resume call doesn't carry over
        assert (await store.ended(run_id)).status == "complete"


@pytest.mark.parametrize("meanwhile", ["declined", "taken"])
async def test_a_decline_or_a_takeover_that_lands_just_before_a_resume_call_writes_its_end_decides_that_end(
    meanwhile: str,
) -> None:
    """The write that ends a resume call loses to a write that lands just before it, and the end is worked out again
    from the row as it is then.

    A decline queued then ends the run ``declined``: the call's own ``run_end`` said ``waiting``, but the one stored
    says what the row says. A run another worker took then is left as the taker has it, with no ``run_end`` from the
    worker that lost it.
    """
    registry, graph, gate, inputs, waiting_id = asking()

    class LandsFirst(MemoryStore):
        """Has another worker decline or take the run just before the write that ends the first resume call."""

        landed = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if not self.landed and event is not None:
                self.landed = True
                if meanwhile == "declined":
                    declined = await other.decline(run_id, waiting_id, allows=EVERY_RUN)
                    assert declined.status == "running"
                else:
                    taken = self.rows[run_id]
                    lease = time.time() + 30
                    changed = {"owner": "another-worker", "lease_until": lease}
                    assert await super().update(run_id, taken.rev, changed)
            return await super().update(run_id, rev, changes, event)

    store = LandsFirst()
    worker, other = _worker(store, registry), _worker(store, registry)
    async with worker.serving(), other.serving():
        run_id = await _started(worker, graph, inputs)
        gate.set()
        row = await store.until(
            run_id,
            lambda row: store.landed and (row.status != "running" or row.owner == "another-worker"),
        )
        run_ends = [
            f"{event['status']}:{event['cause']['type'] if event['cause'] else ''}"
            for event in await store.stored_events(run_id)
            if event["type"] == "run_end"
        ]
        if meanwhile == "declined":
            assert (row.status, row.owner, run_ends) == (
                "declined",
                None,
                ["declined:answer_declined"],
            )
        else:
            await asyncio.sleep(
                0.05
            )  # what the worker that lost the run would write has landed by now
            assert store.rows[run_id] == row
            assert (row.status, row.owner, run_ends) == ("running", "another-worker", [])


async def test_a_question_that_waits_in_a_failed_run_is_answered_without_a_resume() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("many", title="Many", description="For a test.")
    def many(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(x: int) -> bool:
        if x == 2:
            raise NodeError("item 2 fails")
        raise Ask(f"approve {x}?")

    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, make_graph(many, approve), {"n": 3})
        failed = await store.ended(run_id)
        assert (failed.status, len(failed.waiting)) == ("failed", 2)

        assert (
            await worker.answer(run_id, {failed.waiting[0].id: True}, allows=EVERY_RUN)
        ).status == "running"
        after = await store.ended(run_id)
        assert (after.status, [waiting_call.id for waiting_call in after.waiting]) == (
            "failed",
            [failed.waiting[1].id],
        )


# -- a resume call that raised --------------------------------------------------------------------


@pytest.mark.parametrize("fails", ["the write of an event", "the write of a checkpoint"])
async def test_a_resume_call_the_store_raises_in_is_given_up_and_the_run_is_resumed_once_its_lease_runs_out(
    fails: str,
) -> None:
    registry, graph, called, calls = _slow_items()

    class RaisesOnce(MemoryStore):
        """Raises once in the middle of a resume call, like a database that drops one statement."""

        def __init__(self) -> None:
            super().__init__()
            self.raised = AsyncioEvent()

        def down(self) -> bool:
            if not self.raised.is_set() and len(calls) >= 5:
                self.raised.set()
                return True
            return False

        async def add_event(self, run_id: str, owner: str, body: str) -> bool:
            if fails == "the write of an event" and self.down():
                raise ConnectionError("the store is down for one statement")
            return await super().add_event(run_id, owner, body)

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if (
                fails == "the write of a checkpoint"
                and changes.keys() == {"run", "outcome"}
                and self.down()
            ):
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes, event)

    store = RaisesOnce()
    worker = _worker(store, registry, lease=0.3, checkpoint_every=0.01)
    async with worker.serving():
        run_id = await _started(worker, graph, {"n": 40})
        async with asyncio.timeout(5):
            await store.raised.wait()
        at_the_raise = len(calls)
        # a resume call that was still going would have called more by now, and the lease hasn't run out yet
        await asyncio.sleep(0.1)
        assert len(calls) <= at_the_raise + 1 and store.rows[run_id].status == "running"
        ended = await store.ended(run_id)  # once the lease ran out and a sweep took the run

    ends = [
        event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"
    ]
    assert (ended.status, ended.cause, ends, ended.outputs) == (
        "complete",
        None,
        ["complete"],
        {"result": list(range(40))},
    )
    assert await _resume_calls(store, run_id) == 2


def _shout() -> tuple[NodeRegistry, GraphRecord]:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    return registry, make_graph(shout)


def _context_raises(metadata: object, stop: ThreadEvent) -> None:
    raise ConnectionError("the secrets service is down")


async def test_a_resume_call_that_raised_outside_a_node_fails_the_run_with_a_cause_its_run_end_has_too() -> (
    None
):
    registry, graph = _shout()
    store = MemoryStore()

    worker = _worker(store, registry, context=_context_raises)
    async with worker.serving():
        run_id = await _started(worker, graph, {"text": "a"})
        failed = await store.ended(run_id)

        assert failed.cause is not None and (
            failed.status,
            failed.cause.type,
            failed.cause.loc,
            failed.cause.ctx,
        ) == (
            "failed",
            "raised_outside_a_node",
            (),
            {"error": "ConnectionError"},
        )
        assert await store.stored_events(run_id) == [
            {"type": "run_end", "status": "failed", "cause": failed.cause.model_dump(mode="json")}
        ]


class Tenant:
    """A context class the registry expects, which this test's worker never gives."""


async def test_a_run_conductor_refuses_to_resume_fails_with_its_first_problem_and_logs_them_all(
    caplog: LogCaptureFixture,
) -> None:
    registry, store = NodeRegistry(context_type=Tenant), MemoryStore()

    @registry.node("stamp", title="Stamp", description="For a test.")
    def stamp(tenant: Context[Tenant], text: str) -> str:
        return text

    @registry.node("seal", title="Seal", description="For a test.")
    def seal(tenant: Context[Tenant], text: str) -> str:
        return text

    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, make_graph(stamp, seal), {"text": "a"})
        failed = await store.ended(run_id)

        assert failed.cause is not None and (
            failed.status,
            failed.cause.type,
            failed.cause.loc,
        ) == (
            "failed",
            "context_missing",
            ("stamp", "tenant"),
        )
        assert failed.cause.ctx["needs"] == "Tenant"
        assert await store.stored_events(run_id) == [
            {"type": "run_end", "status": "failed", "cause": failed.cause.model_dump(mode="json")}
        ]
        (logged,) = [
            record.getMessage() for record in caplog.records if run_id in record.getMessage()
        ]
        assert "stamp.tenant" in logged and "seal.tenant" in logged and "\n" not in logged


async def test_a_run_whose_resume_call_raised_after_a_checkpoint_lists_the_calls_waiting_in_that_checkpoint() -> (
    None
):
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(question: str) -> bool:
        raise Ask("Go on?")

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        # longer than checkpoint_every, so a checkpoint is due when this call's outcome is written
        time.sleep(0.02)
        return text

    @registry.node("opaque", title="Opaque", description="For a test.")
    def opaque(text: str) -> Opaque:
        return Opaque()

    graph = make_graph(
        approve,
        busy,
        opaque,
        edges=[("busy.result", "opaque.text")],
        outputs={"approve": "approve.result", "opaque": "opaque.result"},
    )
    worker = _worker(store, registry, checkpoint_every=0.01)
    async with worker.serving():
        run_id = await _started(worker, graph, {"question": "y", "text": "x"})
        failed = await store.ended(run_id)

        assert failed.cause is not None and (failed.status, failed.cause.type) == (
            "failed",
            "run_unwritable",
        )
        stored = Run.from_json(store.rows[run_id].run, Graph.compile(graph, registry))
        assert [waiting_call.id for waiting_call in failed.waiting] == [
            waiting_call.id for waiting_call in stored.waiting
        ] and len(failed.waiting) == 1


async def test_a_run_whose_resume_call_raised_lists_the_calls_waiting_as_stored_and_takes_answers_without_a_resume() -> (
    None
):
    registry, store, resume_calls = NodeRegistry(), MemoryStore(), []

    @registry.node("many", title="Many", description="For a test.")
    def many(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(x: int) -> bool:
        raise Ask(f"approve {x}?")

    def fails_on_the_second_resume_call(metadata: object, stop: ThreadEvent) -> None:
        resume_calls.append(metadata)
        if len(resume_calls) == 2:
            raise ConnectionError("the secrets service is down")

    worker = _worker(store, registry, context=fails_on_the_second_resume_call)
    async with worker.serving():
        run_id = await _started(worker, make_graph(many, approve), {"n": 3})
        first, second, third = (
            waiting_call.id for waiting_call in (await store.ended(run_id)).waiting
        )

        assert (await worker.answer(run_id, {first: True}, allows=EVERY_RUN)).status == "running"
        failed = await store.ended(run_id)
        assert failed.cause is not None and (
            failed.status,
            failed.cause.type,
            [waiting_call.id for waiting_call in failed.waiting],
        ) == ("failed", "raised_outside_a_node", [second, third])

        assert (await worker.answer(run_id, {second: True}, allows=EVERY_RUN)).status == "running"
        assert [waiting_call.id for waiting_call in (await store.ended(run_id)).waiting] == [third]


async def test_a_saved_run_that_cant_be_read_fails_the_run_and_lists_nothing_as_waiting() -> None:
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    await store.create(
        dataclasses.replace(
            left_running(
                "left", graph, Graph.compile(graph, registry).run(inputs), owner="a-dead-worker"
            ),
            run="{",
        )
    )

    async with worker.serving():
        failed = await store.ended("left")
    assert failed.cause is not None and (failed.status, failed.cause.type, failed.waiting) == (
        "failed",
        "saved_run_unreadable",
        [],
    )


async def test_a_graph_with_no_node_to_call_begins_running_with_no_outputs_and_shows_them_once_complete() -> (
    None
):
    """The run is complete before its first resume call, but its row says ``running`` until that call ends, and a
    row shows outputs only once it says ``complete``."""
    store = MemoryStore()
    worker = _worker(store, NodeRegistry())
    async with worker.serving():
        begun = await worker.begin(GraphRecord(nodes={}, outputs={}), {}, ANNA)
        completed = await store.ended(begun.id)

    assert (begun.status, begun.outputs) == ("running", None)
    assert (completed.status, completed.outputs) == ("complete", {})


async def test_a_queued_answer_that_completes_a_run_whose_resume_call_failed_shows_no_outputs() -> (
    None
):
    """The run as the row has it takes the answer as the call fails, and every node is then complete. The row says
    ``failed``, so it shows no outputs; a resume makes no call and completes it."""
    registry = NodeRegistry()

    @registry.node("sign", title="Sign", description="Asks for a signature.", asks=True)
    def sign(document: str) -> bool:
        raise Ask("Sign?")

    graph = make_graph(sign, outputs={"signed": "sign.result"})
    checkpoint = Graph.compile(graph, registry).run({"document": "x"})
    (waiting,) = checkpoint.waiting
    contexts: list[object] = []

    def fails_once(metadata: object, stop: ThreadEvent) -> None:
        contexts.append(metadata)
        if len(contexts) == 1:
            raise ConnectionError("the secrets service is down")

    store = MemoryStore()
    worker = _worker(store, registry, context=fails_once)
    await store.create(left_running("left", graph, checkpoint, owner="a-dead-worker"))
    await worker.answer("left", {waiting.id: True}, allows=EVERY_RUN)
    async with worker.serving():
        failed = await store.ended("left")
        await worker.resume("left", allows=EVERY_RUN)
        completed = await store.ended("left")

    assert (failed.status, failed.outputs, failed.refused) == ("failed", None, {})
    assert (completed.status, completed.outputs) == ("complete", {"signed": True})


async def test_a_run_cancelled_after_it_completed_whose_resume_call_fails_before_any_node_call_shows_no_outputs() -> (
    None
):
    """The run as the row has it is complete, but the row says ``failed``, so it shows no outputs."""
    registry, started, gate, contexts = NodeRegistry(), ThreadEvent(), ThreadEvent(), []

    @registry.node("pieces", title="Pieces", description="Splits a text at its commas.")
    def pieces(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("hold", title="Hold", description="Waits on a gate, then gives its piece back.")
    def hold(piece: str) -> str:
        started.set()
        gate.wait(10)
        return piece

    def fails_after_the_first(metadata: object, stop: ThreadEvent) -> None:
        contexts.append(metadata)
        if len(contexts) > 1:
            raise ConnectionError("the secrets service is down")

    store = MemoryStore()
    graph = make_graph(pieces, hold, edges=[("pieces.result", "hold.piece")])
    holder = _worker(store, registry, context=fails_after_the_first)
    other = _worker(store, registry, context=fails_after_the_first)
    async with holder.serving(), other.serving():
        run_id = await _started(holder, graph, {"text": "p,q"})
        await asyncio.to_thread(started.wait, 10)
        await other.cancel(run_id, allows=EVERY_RUN)
        gate.set()
        cancelled = await store.ended(run_id)
        await other.resume(run_id, allows=EVERY_RUN)
        failed = await store.ended(run_id)

    assert (cancelled.status, cancelled.outputs) == ("cancelled", None)
    assert failed.cause is not None
    assert (failed.status, failed.cause.type, failed.outputs) == (
        "failed",
        "raised_outside_a_node",
        None,
    )


async def test_a_run_whose_end_the_store_fails_to_write_stays_running_and_is_resumed_once_its_lease_runs_out() -> (
    None
):
    registry, graph = _shout()

    class EndFailsOnce(MemoryStore):
        """Raises once on the write that ends a resume call, which carries that call's ``run_end``, so neither lands."""

        raised = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if changes.get("status") == "complete" and not self.raised:
                self.raised = True
                raise ConnectionError("the store went away for one statement")
            return await super().update(run_id, rev, changes, event)

    store = EndFailsOnce()
    worker = _worker(store, registry, lease=0.15)
    async with worker.serving():
        run_id = await _started(worker, graph, {"text": "a"})
        ended = await store.ended(run_id)  # once the lease ran out and a sweep took the run

    ends = [
        event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"
    ]
    assert store.raised and (ended.status, ended.cause, ends) == (
        "complete",
        None,
        ["complete"],
    )
    assert await _resume_calls(store, run_id) == 2


@host_type("opaque", title="Opaque")
class Opaque:
    """A host's type that pydantic can carry but can't write as JSON."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls)


@pytest.mark.parametrize(
    ("graph_of", "at"),
    [(("opaque",), "opaque.result"), (("opaque", "shown"), "opaque.result")],
    ids=["an output of the graph", "a value inside the run"],
)
async def test_a_run_that_cant_be_written_fails_with_run_unwritable_the_first_time_and_is_not_resumed_again(
    graph_of: tuple[str, ...], at: str
) -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("opaque", title="Opaque", description="For a test.")
    def opaque(text: str) -> Opaque:
        return Opaque()

    @registry.node("shown", title="Shown", description="For a test.")
    def shown(value: Opaque) -> str:
        return "shown"

    graph = make_graph(*(registry[node_id] for node_id in graph_of))
    worker = _worker(store, registry, lease=0.15)
    async with worker.serving():
        run_id = await _started(worker, graph, {"text": "a"})
        ended = await store.ended(run_id)
        await asyncio.sleep(0.4)  # long enough for a retried resume call to start

    ends = [
        (event["status"], event["cause"]["type"])
        for event in await store.stored_events(run_id)
        if event["type"] == "run_end"
    ]
    assert ended.cause is not None
    assert (ended.status, ended.cause.type, ended.cause.ctx["where"], ended.waiting, ends) == (
        "failed",
        "run_unwritable",
        at,
        [],
        [("failed", "run_unwritable")],
    )
    assert (
        await _resume_calls(store, run_id) == 1
        and (await worker.read(run_id, allows=EVERY_RUN)).status == "failed"
    )


# -- a follower -----------------------------------------------------------------------------------


async def test_a_follower_reads_the_row_only_to_learn_whether_a_run_end_was_the_last() -> None:
    registry, graph, gate, inputs, _ = asking()

    class CountsReads(MemoryStore):
        reads = 0

        async def get(self, run_id: str) -> StoredRun | None:
            self.reads += 1
            return await super().get(run_id)

    store = CountsReads()
    worker = _worker(store, registry, poll=0.01)
    async with worker.serving():
        run = await worker.read(await _started(worker, graph, inputs), allows=EVERY_RUN)

        async def followed(after: int = -1) -> list[tuple[int, str]]:
            return [event async for event in await worker.events(run.id, after, allows=EVERY_RUN)]

        following = asyncio.create_task(followed())
        await asyncio.sleep(0.1)  # the follower has the resume call's first events by now
        before = store.reads
        await asyncio.sleep(0.3)  # about thirty polls, none of which finds an event
        during = store.reads - before
        gate.set()
        events = await following

        assert during == 0 and [number for number, _ in events] == list(range(len(events)))
        assert [json.loads(body) for _, body in events] == await store.stored_events(run.id)
        assert (
            await followed(len(events) - 1) == []
        )  # a follower that already has every event of a waiting run gets an answer at once
        assert await followed(len(events) - 2) == events[-1:]


async def test_a_follower_that_got_a_run_end_while_the_row_still_reads_running_keeps_asking_until_it_changes() -> (
    None
):
    registry, graph = _shout()

    class RowLags(MemoryStore):
        """Reads a completed run as still running, the first five times.

        That is what a reader sees between the stored ``run_end`` event and the write of the row.
        """

        lagged = 0

        async def get(self, run_id: str) -> StoredRun | None:
            row = self.rows[run_id]
            if row.status == "complete" and self.lagged < 5:
                self.lagged += 1
                return dataclasses.replace(row, status="running", state=BEGUN_RUN_STATE)
            return row

    store = RowLags()
    worker = _worker(store, registry, poll=0.01)
    async with worker.serving():
        run_id = await _started(worker, graph, {"text": "a"})
        await store.ended(run_id)

        async with asyncio.timeout(5):
            events = [
                json.loads(body) async for _, body in await worker.events(run_id, allows=EVERY_RUN)
            ]

        assert [event["type"] for event in events].count("run_end") == 1 and events[-1][
            "type"
        ] == "run_end"
        assert store.lagged >= 5


# -- several workers, a shutdown ------------------------------------------------------------------


def _slow_items() -> tuple[NodeRegistry, GraphRecord, ThreadEvent, list[int]]:
    """Builds a graph that works through a Series one slow item at a time.

    Returns:
        The registry, the graph, ``called`` (set once the first item is called), and ``calls`` (every item called).
    """
    registry, called, calls = NodeRegistry(), ThreadEvent(), []

    @registry.node("many", title="Many", description="Gives several items.")
    def many(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("slow", title="Slow", description="Waits before it answers.")
    def slow(x: int) -> int:
        calls.append(x)
        called.set()
        time.sleep(0.02)
        return x

    return registry, make_graph(many, slow, edges=[("many.result", "slow.x")]), called, calls


async def test_a_checkpoint_writes_the_runs_progress_with_the_outcome_as_the_run_last_stopped() -> (
    None
):
    registry, graph, _, _ = _slow_items()
    written: list[dict[str, Any]] = []

    class Recording(MemoryStore):
        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if changes.keys() == {"run", "outcome"}:
                written.append(json.loads(changes["outcome"]))
            return await super().update(run_id, rev, changes, event)

    store = Recording()
    worker = _worker(store, registry, checkpoint_every=0.01)
    async with worker.serving():
        run_id = await _started(worker, graph, {"n": 20})
        await store.ended(run_id)

        assert written and {
            outcome["waiting"] == [] and outcome["outputs"] is None for outcome in written
        } == {True}
        finished = [outcome["progress"]["nodes"]["slow"]["finished"] for outcome in written]
        assert (
            finished == sorted(finished)
            and finished[0] < 20
            and written[0]["progress"]["nodes"]["slow"]["total"] == 20
        )


async def test_a_run_handed_back_by_a_worker_shutting_down_is_running_with_no_cause_and_its_saved_run_is_ready() -> (
    None
):
    registry, graph, called, _ = _slow_items()
    store = MemoryStore()
    worker = _worker(store, registry)

    async with worker.serving():
        run_id = await _started(worker, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)

    row, view = store.rows[run_id], (await worker.read(run_id, allows=EVERY_RUN))
    assert (view.status, view.cause, row.owner, row.lease_until) == (
        "running",
        None,
        None,
        0.0,
    )
    assert (
        Run.from_json(row.run, Graph.compile(row.graph, registry)).status == "ready"
    )  # the saved run is only ready: it's the row that says a worker should go on


async def test_a_run_handed_back_at_shutdown_stores_no_run_end_and_the_worker_that_takes_it_repeats_no_call() -> (
    None
):
    registry, graph, called, calls = _slow_items()
    store = MemoryStore()
    first, second = _worker(store, registry), _worker(store, registry)

    async with first.serving():
        run_id = await _started(first, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)
    assert "run_end" not in [event["type"] for event in await store.stored_events(run_id)]

    async with second.serving():
        assert (await store.ended(run_id)).outputs == {"result": list(range(40))}
    events = [event["type"] for event in await store.stored_events(run_id)]
    assert (events.count("run_start"), events.count("run_end"), events[-1]) == (2, 1, "run_end")
    assert sorted(calls) == list(range(40))


async def test_a_cancel_from_another_worker_that_arrives_as_a_worker_hands_the_run_back_is_kept_for_the_worker_that_takes_it() -> (
    None
):
    registry, graph, gate, inputs, _ = asking()

    class CancelAtHandBack(MemoryStore):
        """Has another worker cancel the run just before the write that hands the run back."""

        cancelled = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if not self.cancelled and changes.get("lease_until") == 0.0:
                self.cancelled = True
                assert (await other.cancel(run_id, allows=EVERY_RUN)).status == "running"
            return await super().update(run_id, rev, changes, event)

    store = CancelAtHandBack()
    worker, other = _worker(store, registry), _worker(store, registry)

    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        asyncio.get_running_loop().call_later(0.05, gate.set)

    handed_back = store.rows[run_id]
    assert store.cancelled  # and the worker that takes the run hears the cancel, below
    assert (handed_back.status, handed_back.owner, handed_back.lease_until) == (
        "running",
        None,
        0.0,
    )
    assert "run_end" not in [event["type"] for event in await store.stored_events(run_id)]

    async with other.serving():
        row = await store.until(run_id, lambda row: row.status != "running")
    assert (row.status, row.owner, row.lease_until) == ("cancelled", None, None)
    assert [
        event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"
    ] == ["cancelled"]


async def test_one_renewal_the_store_never_answers_is_followed_at_once_by_the_next_so_the_worker_keeps_the_run() -> (
    None
):
    registry, graph, called, calls = _slow_items()

    class OneRenewalHangs(MemoryStore):
        """Never answers the first lease renewal, like one statement on a connection that went quiet."""

        hung = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if changes.keys() == {"lease_until"} and not self.hung:
                self.hung = True
                await AsyncioEvent().wait()
            return await super().update(run_id, rev, changes, event)

    store = OneRenewalHangs()
    worker, other = _worker(store, registry, lease=0.45), _worker(store, registry, lease=0.45)

    async with worker.serving(), other.serving():
        run_id = await _started(worker, graph, {"n": 40})
        ended = await store.ended(run_id)

    assert store.hung and (ended.status, sorted(calls)) == ("complete", list(range(40)))
    assert await _resume_calls(store, run_id) == 1


async def test_a_worker_whose_renewal_never_returns_stops_once_another_took_the_run_and_can_shut_down() -> (
    None
):
    registry, graph, called, calls = _slow_items()

    class RenewalHangs(MemoryStore):
        """Never answers the lease renewals of the worker that began the run, like a connection that went quiet."""

        of: str | None = None

        async def create(self, run: StoredRun) -> None:
            if self.of is None:
                self.of = run.owner  # the worker that begins the run
            await super().create(run)

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if changes.keys() == {"lease_until"} and self.rows[run_id].owner == self.of:
                await AsyncioEvent().wait()
            return await super().update(run_id, rev, changes, event)

    store = RenewalHangs()
    hung, other = (
        _worker(store, registry, lease=0.15, checkpoint_every=0.02),
        _worker(store, registry, lease=0.15, checkpoint_every=0.02),
    )

    async with asyncio.timeout(8):
        async with hung.serving():
            run_id = await _started(hung, graph, {"n": 40})
            async with other.serving():
                ended = await store.ended(run_id)
        # leaving ``hung.serving()`` waits for its resume call, which ends only once it learns it lost the run

    ends = [
        event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"
    ]
    assert (ended.status, ends, ended.outputs) == (
        "complete",
        ["complete"],
        {"result": list(range(40))},
    )
    assert (
        len(calls) < 2 * 40
    )  # the resume call that lost the run stopped before it had called everything again


async def test_a_deploy_whose_last_checkpoint_the_store_fails_to_take_leaves_the_run_running_for_the_worker_that_takes_it() -> (
    None
):
    registry, graph, called, calls = _slow_items()

    class HandBackFails(MemoryStore):
        """Once ``armed``, raises on one checkpoint write.

        The test arms it as the worker shuts down, so the write that fails is the last checkpoint, written as the
        run is handed back.
        """

        armed = raised = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if self.armed and not self.raised and changes.keys() == {"run", "outcome"}:
                self.raised = True
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes, event)

    store = HandBackFails()
    deployed, other = _worker(store, registry, lease=0.15), _worker(store, registry, lease=0.15)

    async with deployed.serving():
        run_id = await _started(deployed, graph, {"n": 40})
        owner = store.rows[run_id].owner
        await asyncio.to_thread(called.wait, 5)
        store.armed = True
    row = store.rows[run_id]
    assert store.raised and (row.status, row.owner) == ("running", owner)

    async with other.serving():
        ended = await store.ended(run_id)  # once the lease the deployed worker left ran out

    ends = [
        event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"
    ]
    assert (ended.status, ended.cause, ends, ended.outputs) == (
        "complete",
        None,
        ["complete"],
        {"result": list(range(40))},
    )


async def test_a_run_begun_as_a_worker_shuts_down_is_handed_back_before_the_worker_has_shut_down() -> (
    None
):
    registry, graph, called, _ = _slow_items()

    class BeginsAtHandBack(MemoryStore):
        """Starts a second run on the worker just as the worker hands its first run back."""

        late: str | None = None

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if self.late is None and changes.get("lease_until") == 0.0:
                self.late = await _started(worker, graph, {"n": 40})
            return await super().update(run_id, rev, changes, event)

    store = BeginsAtHandBack()
    worker = _worker(store, registry)

    async with worker.serving():
        await _started(worker, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)

    assert store.late is not None
    late = store.rows[store.late]
    assert (late.status, late.owner, late.lease_until) == ("running", None, 0.0)


async def test_a_run_answered_while_its_last_resume_call_still_writes_its_end_is_handed_back_at_shutdown() -> (
    None
):
    registry, graph, gates, _ = gated()

    class AnswersItsEndLate(MemoryStore):
        """Writes a row at once, but returns from the write that ends the first resume call only when told to."""

        def __init__(self) -> None:
            super().__init__()
            self.end_written, self.answer_end = AsyncioEvent(), AsyncioEvent()

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            written = await super().update(run_id, rev, changes, event)
            if written and changes.get("owner", "kept") is None and not self.end_written.is_set():
                self.end_written.set()
                await self.answer_end.wait()
            return written

    store, errors = AnswersItsEndLate(), []
    asyncio.get_running_loop().set_exception_handler(lambda _, context: errors.append(context))
    worker = _worker(store, registry)
    gates["first"].set()
    async with worker.serving():
        run_id = await _started(worker, graph, {"csv": "a,b", "text": "x", "question": "y"})
        await store.end_written.wait()
        (waiting,) = (await worker.read(run_id, allows=EVERY_RUN)).waiting
        await worker.answer(run_id, {waiting.id: True}, allows=EVERY_RUN)
        store.answer_end.set()
        await asyncio.to_thread(gates["per_item_called"].wait, 5)
    row = store.rows[run_id]
    gates["second"].set()
    # The answered resume call is handed back like any other, though the first one's end came back after it began.
    assert (row.status, row.owner, row.lease_until) == ("running", None, 0.0)
    assert errors == []


@pytest.mark.parametrize(
    "fails", ["the read of the expired runs", "the claim of a run", "a row that can't be read"]
)
async def test_a_sweep_that_raises_is_logged_and_the_next_one_takes_the_run(
    fails: str, caplog: LogCaptureFixture
) -> None:
    registry, graph = _shout()

    class SweepFailsOnce(MemoryStore):
        raised = False

        async def expired(self, now: float) -> list[StoredRun]:
            if fails == "the read of the expired runs" and not self.raised:
                self.raised = True
                raise ConnectionError("the store is down for one statement")
            return await super().expired(now)

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if fails == "the claim of a run" and not self.raised:
                self.raised = True
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes, event)

    store = SweepFailsOnce()
    worker = _worker(store, registry, lease=0.15)
    left = left_running(
        "left", graph, Graph.compile(graph, registry).with_inputs({"text": "a"}), owner=None
    )
    if fails == "a row that can't be read":
        await store.create(dataclasses.replace(left, id="broken", state="{"))
    await store.create(left)

    async with worker.serving():
        ended = await store.ended("left")

    assert ended.outputs == {"result": "A"}
    logged = [
        record.exc_info is not None
        for record in caplog.records
        if record.name == "conductor_providers.fastapi"
    ]
    assert logged and all(logged)


async def test_a_renewal_that_raises_is_logged_and_the_next_one_renews_the_lease(
    caplog: LogCaptureFixture,
) -> None:
    """The first renewal of the run's lease raises. The run's one node waits until the next renewal lands, so nothing
    rests on a clock. That renewal is written by the worker that holds the run, and moves the lease on past the one
    the run was claimed with. A sweep takes only a run whose lease is past, as the sweep tests show."""
    registry, renewed = NodeRegistry(), ThreadEvent()

    @registry.node("wait", title="Wait", description="Waits for the lease to be renewed.")
    def wait(text: str) -> str:
        assert renewed.wait(5)
        return text

    class RenewalFailsOnce(MemoryStore):
        def __init__(self) -> None:
            super().__init__()
            self.raised = False
            self.renewals: list[tuple[str | None, float | None]] = []

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if changes.keys() == {"lease_until"} and not self.raised:
                self.raised = True
                raise ConnectionError("the store is down for one statement")
            written = await super().update(run_id, rev, changes, event)
            if written and changes.keys() == {"lease_until"}:
                self.renewals.append((self.rows[run_id].owner, changes["lease_until"]))
                renewed.set()
            return written

    store = RenewalFailsOnce()
    worker = _worker(store, registry, lease=0.3)
    async with worker.serving():
        run_id = await _started(worker, make_graph(wait), {"text": "x"})
        claimed = store.rows[run_id]
        ended = await store.ended(run_id)

    (holder, renewed_until), *_ = store.renewals
    assert store.raised and ended.status == "complete"
    assert holder == claimed.owner and renewed_until > claimed.lease_until
    assert await _resume_calls(store, run_id) == 1
    logged = [
        record.exc_info is not None
        for record in caplog.records
        if record.name == "conductor_providers.fastapi"
    ]
    assert logged and all(logged)


@pytest.mark.parametrize("ends", ["complete", "waiting"])
async def test_a_run_that_stopped_running_has_no_lease_so_no_sweep_takes_it_again(
    ends: str,
) -> None:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    store = MemoryStore()
    worker = _worker(store, registry, lease=0.3)  # a sweep each tenth of a second
    async with worker.serving():
        run_id = await _started(
            worker, make_graph(shout if ends == "complete" else approve), {"text": "a"}
        )
        assert (await store.ended(run_id)).status == ends

        row = store.rows[run_id]
        assert (row.owner, row.lease_until) == (None, None)
        assert await store.expired(time.time()) == []
        store.swept.clear()
        await store.swept.wait()
    assert store.rows[run_id] == row and await _resume_calls(store, run_id) == 1


async def test_a_run_whose_worker_died_is_taken_by_one_of_two_workers_and_finished_from_its_checkpoint() -> (
    None
):
    registry, graph, gates, calls = gated()
    for gate in gates.values():
        gate.set()
    run = Graph.compile(graph, registry).run({"csv": "a,b", "text": "x", "question": "y"})
    checkpoint = run.with_answers({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    one, other = _worker(store, registry), _worker(store, registry)
    await store.create(left_running("left", graph, checkpoint, owner="a-dead-worker"))

    async with one.serving(), other.serving():
        ended = await store.ended("left")

    assert ended.outputs == {"busy": "x", "per_item": ["a", "b"]}
    assert calls == ["a", "b"] and await _resume_calls(store, "left") == 1


async def test_a_worker_whose_run_another_took_stops_resuming_it_and_writes_nothing_more() -> None:
    registry, graph, gates, calls = gated()
    inputs = {"csv": "a,b", "text": "x", "question": "y"}
    store = MemoryStore()
    slow, taker = _worker(store, registry, lease=0.15), _worker(store, registry)

    async with slow.serving():  # leaving it waits for the resume call the slow worker still runs
        run_id = await _started(slow, graph, inputs)
        row = store.rows[run_id]
        assert await store.update(
            run_id, row.rev, {"lease_until": 0.0}
        )  # as if its lease had run out without being renewed
        async with taker.serving():
            taken = await store.until(run_id, lambda now: now.owner not in (None, row.owner))
            await asyncio.sleep(0.2)  # the slow worker's renewals would have landed by now
            assert store.rows[run_id] == taken
            gates["first"].set()
            ended = await store.ended(run_id)

    assert ended.status == "waiting" and await store.view(run_id) == ended
    assert [event["type"] for event in await store.stored_events(run_id)].count("run_end") == 1
    for gate in gates.values():
        gate.set()


async def test_a_call_on_a_run_by_id_names_its_run_filter_and_a_start_from_a_previous_run_does_too() -> (
    None
):
    """``allows=`` is required, so a host's own route can't skip the access check by leaving it out."""
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        await store.ended(run_id)

        with pytest.raises(TypeError):
            await worker.read(run_id)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="allows="):
            await worker.begin(graph, inputs, ANNA, previous=run_id)
        assert (await worker.read(run_id, allows=EVERY_RUN)).id == run_id


async def test_a_run_taken_over_whose_metadata_the_type_refuses_is_failed_once_with_its_run_end(
    caplog: LogCaptureFixture,
) -> None:
    """The sweep takes the run without reading its metadata, and the resume call fails it, as any call that raises."""
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()
    left = left_running(
        "bad", graph, Graph.compile(graph, registry).with_inputs(inputs), owner="dead"
    )
    await store.create(dataclasses.replace(left, metadata='{"user": 5}'))
    worker = Worker(store, nodes_for=lambda metadata: registry, poll=0.01, lease=0.15)

    async with worker.serving():
        # a reader's view reads the metadata too, so the row is read as the store has it
        row = await asyncio.wait_for(store.until("bad", lambda row: row.status != "running"), 5)
        await asyncio.sleep(0.3)  # two more sweeps, which must leave the failed run alone

    assert (row.status, store.state("bad")["cause"]["type"]) == ("failed", "raised_outside_a_node")
    assert [event["type"] for event in await store.stored_events("bad")] == ["run_end"]
    assert sum("bad" in record.getMessage() for record in caplog.records) == 1


class _Owner(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user: str
    team: str = "none"


async def test_a_views_metadata_is_the_json_the_row_keeps_not_as_the_type_reads_it_now() -> None:
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()
    worker = Worker(store, nodes_for=lambda metadata: registry, poll=0.01, metadata_type=_Owner)
    async with worker.serving():
        run_id = (await worker.begin(graph, inputs, _Owner(user="anna"))).id
        await store.ended(run_id)
        store.rows[run_id] = dataclasses.replace(
            store.rows[run_id], metadata='{"user": "anna", "legacy": 7}'
        )

        view = await worker.read(run_id, allows=EVERY_RUN)

    assert view.metadata == {"user": "anna", "legacy": 7}


async def test_a_held_run_that_stops_on_a_row_it_cant_read_logs_it_with_the_runs_id(
    caplog: LogCaptureFixture,
) -> None:
    """A row the worker can't read can't fail the run, since the failure would be written to that row. The held run
    stops, and the log says which run and why, rather than the task dying unseen."""
    registry, graph, gate, inputs, _ = asking()

    class Unreadable(MemoryStore):
        """Writes a ``state`` the worker can't read just before the write that ends the first resume call."""

        broken = False

        async def update(
            self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
        ) -> bool:
            if not self.broken and event is not None:
                self.broken = True
                row = self.rows[run_id]
                self.rows[run_id] = dataclasses.replace(row, state="{}", rev=row.rev + 1)
            return await super().update(run_id, rev, changes, event)

    store = Unreadable()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        gate.set()
        async with asyncio.timeout(5):
            while not any("stopped resuming run" in record.message for record in caplog.records):
                await asyncio.sleep(0.01)

    assert any(
        f"the worker stopped resuming run {run_id}" in record.message and record.exc_info
        for record in caplog.records
    )
