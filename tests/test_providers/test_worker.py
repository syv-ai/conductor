"""The ``Worker`` behind ``run_router``, called as the routes call it, on a store in memory: no HTTP, and each wait is
on the store's own writes."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from asyncio import Event as AsyncioEvent
from collections.abc import Mapping
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
    Run,
    Series,
    Table,
    host_type,
    make_graph,
)
from conductor_providers.fastapi import RefusedError, RunChanges, StoredRun, Worker
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pytest import LogCaptureFixture, MonkeyPatch

from tests.test_providers.in_memory import BEGUN_RUN_STATE, MemoryStore, asking, gated, left_running


def _worker(store: MemoryStore, registry: NodeRegistry[Any], **settings: Any) -> Worker[str]:
    """A worker on ``store`` whose callers are names, with the worker's own settings where ``settings`` name none."""
    worker: Worker[str] = Worker(
        store=store, caller_type=str, nodes_for=lambda caller: registry, **settings
    )
    return worker


async def _started(worker: Worker[str], graph: GraphRecord, inputs: Mapping[str, Any]) -> str:
    return await worker.begin(graph, inputs, "anna")


async def _resume_calls(store: MemoryStore, run_id: str) -> int:
    """How many resume calls of the run have started: one ``run_start`` event each."""
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
        second = await worker.begin(edited, {"n": 3}, "anna", previous=await worker.read(first))
        ended = await store.ended(second)
        assert (ended.status, ended.outputs) == ("complete", {"result": [1, 2, 5]})
        assert calls == ["square 0", "square 1", "square 2", "add 0", "add 1", "add 4"]


def _registry_that_drops_old() -> tuple[
    NodeRegistry[Any], NodeRegistry[Any], GraphRecord, GraphRecord
]:
    """A registry with the nodes ``split``, ``square`` and ``old``, the same registry without ``old``, a graph that
    uses ``old`` and its edit without it."""
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


async def test_a_run_begun_from_a_previous_run_that_cant_be_loaded_on_the_callers_nodes_is_refused_saying_so() -> (
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
            await worker.begin(edited, {"n": 2}, "anna", previous=await worker.read(first))

        (problem,) = refused.value.problems
        assert (problem.type, problem.ctx["run_id"]) == ("previous_run_unloadable", first)
        assert "'old'" in str(problem.ctx["reason"])
        assert list(store.rows) == [first]  # nothing is stored
        begun = await worker.begin(edited, {"n": 2}, "anna")  # without it, the start is taken
        assert (await store.ended(begun)).outputs == {"result": [0, 1]}


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
        previous = await worker.read(first)
        with pytest.raises(InvalidCallError) as broken:
            await worker.begin(
                {"nodes": {"x": {"node": "nonesuch"}}}, {}, "anna", previous=previous
            )
        with pytest.raises(InvalidCallError) as wrong_input:
            await worker.begin(edited, {"n": "two"}, "anna", previous=previous)

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

    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "started"
    begun = await _started(worker, graph, inputs)
    for handed_back in (run_id, begun):  # for a worker that serves to take over
        row = await store.until(handed_back, lambda row: row.owner is None)
        assert (row.status, row.lease_until) == ("running", 0.0)

    async with _worker(store, registry).serving():
        assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}
        assert (await store.ended(begun)).status == "waiting"


# -- what a run reads as --------------------------------------------------------------------------


async def test_an_output_that_isnt_finite_reads_as_its_text() -> None:
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
    """A host's type that only its own schema writes as JSON."""

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


async def test_a_table_made_for_each_item_reads_as_its_columns_per_item() -> None:
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

        await worker.resume(await worker.read(run_id))
        assert (await store.ended(run_id)).status == "stopped"
        await worker.resume(await worker.read(run_id), max_items=2)
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

        await worker.resume(await worker.read(run_id), max_items=10)
        assert (
            json.loads(store.rows[run_id].run)["max_items"] == 10
        )  # written with the claim, so a resume call given up before its first checkpoint is resumed under it
        waiting = (await store.ended(run_id)).waiting
        assert len(waiting) == 5
        assert await worker.answer(await worker.read(run_id), {waiting[0].id: 7}) == "started"
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
            await worker.resume(await worker.read(run_id), max_items=5)

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
            await worker.answer(await worker.read(run_id), {waiting_id: True})
            await store.ended(run_id)

        with pytest.raises(RefusedError) as refused:
            await worker.resume(await worker.read(run_id))

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
        assert await worker.answer(await worker.read(run_id), {waiting.id: True}) == "queued"
        gates["first"].set()
        await asyncio.to_thread(gates["per_item_called"].wait, 5)
        assert await worker.cancel(await worker.read(run_id))
        gates["second"].set()

        assert (await store.ended(run_id)).status == "cancelled" and calls == ["a"]


async def test_a_cancel_sent_while_a_dead_worker_held_the_run_is_kept_when_another_takes_it_over() -> (
    None
):
    registry, graph, gates, calls = gated()
    for gate in gates.values():
        gate.set()
    run = Graph.compile(graph, registry).run({"csv": "a,b", "text": "x", "question": "y"})
    answered = run.answer({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    worker = _worker(store, registry)
    await store.create(left_running("taken", graph, answered, owner="a-dead-worker"))

    assert await worker.cancel(await worker.read("taken"))
    async with worker.serving():  # which takes over at once the run whose lease ran out
        assert (await store.ended("taken")).status == "cancelled" and calls == []

        await worker.resume(await worker.read("taken"))
        assert (await store.ended("taken")).status == "complete" and calls == ["a", "b"]


async def test_a_cancel_of_a_waiting_run_ends_it_cancelled_with_its_question_waiting_and_an_answer_carries_it_on() -> (
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

        assert await worker.cancel(await worker.read(run_id))
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
        assert not await worker.cancel(
            await worker.read(run_id)
        )  # it ended early: nothing to cancel

        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "started"
        assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}
        assert not await worker.cancel(await worker.read(run_id))


async def test_the_worker_goes_by_a_runs_state_and_never_reads_the_status_column() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()

    class DropsStatus(MemoryStore):
        """Writes every column but ``status``, as a host's store whose ``update`` misses the one it only queries."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            return await super().update(
                run_id, rev, {name: value for name, value in changes.items() if name != "status"}
            )

    store = DropsStatus()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        row = await store.until(run_id, lambda row: row.owner is None)
        waiting = await worker.read(run_id)
        assert (row.status, waiting.view().status) == ("running", "waiting")

        with pytest.raises(RefusedError) as refused:
            await worker.resume(waiting)
        assert [problem.ctx for problem in refused.value.problems] == [{"status": "waiting"}]
        assert await worker.answer(waiting, {waiting_id: True}) == "started"
        await store.until(run_id, lambda row: row.owner is None)
        completed = await worker.read(run_id)
        assert completed.view().outputs == {"busy": "x", "after": True}
        assert not await worker.cancel(completed)


# -- answers --------------------------------------------------------------------------------------


async def test_two_workers_answering_one_run_at_once_start_one_resume_call() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    one, other = _worker(store, registry), _worker(store, registry)
    async with one.serving(), other.serving():
        run_id = await _started(one, graph, inputs)
        await store.ended(run_id)
        row = await one.read(run_id)

        outcomes = await asyncio.gather(
            one.answer(row, {waiting_id: True}),
            other.answer(row, {waiting_id: True}),
            return_exceptions=True,
        )

        lost = [outcome for outcome in outcomes if isinstance(outcome, RefusedError)]
        assert [outcome for outcome in outcomes if not isinstance(outcome, RefusedError)] == [
            "started"
        ]
        assert [[problem.type for problem in refused.problems] for refused in lost] == [
            ["run_changed"]
        ]
        assert (await store.ended(run_id)).status == "complete" and await _resume_calls(
            store, run_id
        ) == 2


async def test_a_second_answer_queued_for_a_question_is_refused_and_a_complete_run_takes_none() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)

        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
        with pytest.raises(RefusedError) as twice:
            await worker.answer(await worker.read(run_id), {waiting_id: False})
        assert [(problem.type, problem.ctx) for problem in twice.value.problems] == [
            ("already_answered", {"waiting_id": waiting_id})
        ]

        gate.set()
        assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}
        with pytest.raises(RefusedError) as complete:
            await worker.answer(await worker.read(run_id), {})
        assert [problem.type for problem in complete.value.problems] == ["run_complete"]


async def test_an_answer_queued_during_a_resume_call_starts_the_next_however_long_the_first_takes_to_write_its_end() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()

    class SlowEnd(MemoryStore):
        """Writes the end of a resume call (the write that takes the queued answers) after three lease renewals."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if {"run", "state"} <= changes.keys():
                await asyncio.sleep(0.3)
            return await super().update(run_id, rev, changes)

    store = SlowEnd()
    worker = _worker(store, registry, lease=0.3)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
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
        """Counts the reads and the tries to write, taken or lost."""

        def __init__(self) -> None:
            super().__init__()
            self.reads = self.writes = 0

        async def get(self, run_id: str) -> StoredRun | None:
            self.reads += 1
            return await super().get(run_id)

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            self.writes += 1
            return await super().update(run_id, rev, changes)

    store = Counts()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        row = await worker.read(run_id)
        assert await worker.answer(row, {waiting_id: True}) == "queued"
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
        """Takes an answer from another worker, queued for the running resume call, just before the write that ends it."""

        answered = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if not self.answered and changes.get("status") == "waiting":
                self.answered = True
                assert await other.answer(await other.read(run_id), {waiting_id: True}) == "queued"
            return await super().update(run_id, rev, changes)

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
        if self.status == "waiting":  # the run as the end of the resume call writes it
            slow_writes.append(self.status)
            time.sleep(0.5)  # a large run, longer than the lease
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
        await one.read(run_id)
    ).view() == ended
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
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
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
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
        # past the resume call's timeout; the node call it waits for then finishes
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
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
        assert await worker.cancel(await worker.read(run_id))
        gate.set()
        ended = await store.ended(run_id)

        assert ended.status == "cancelled"
        assert [(problem.type, problem.ctx["status"]) for problem in ended.refused[waiting_id]] == [
            ("not_waiting", "cancelled")
        ]
        assert await _resume_calls(store, run_id) == 1


async def test_an_answer_queued_during_a_resume_call_that_raised_stays_queued_and_is_taken_after_a_resume() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    resume_calls = []

    def fails_on_the_first_resume_call(caller: str) -> None:
        resume_calls.append(caller)
        if len(resume_calls) == 1:
            raise ConnectionError("the secrets service is down")

    store = MemoryStore()
    worker = _worker(store, registry, context=fails_on_the_first_resume_call)
    left = left_running(
        "left", graph, Graph.compile(graph, registry).with_inputs(inputs), owner="a-dead-worker"
    )
    await store.create(left)
    assert await worker.answer(await worker.read("left"), {waiting_id: True}) == "queued"

    async with worker.serving():
        failed = await store.ended("left")
        assert (failed.status, failed.refused) == (
            "failed",
            {},
        )  # the answer stays queued, as the resume below shows

        await worker.resume(await worker.read("left"))
        # the resume call after the resume ended waiting, and took the answer queued two resume calls before
        completed = await store.ended("left")
    assert (completed.status, completed.refused, completed.outputs) == (
        "complete",
        {},
        {"busy": "x", "after": True},
    )


async def test_an_answer_that_is_taken_leaves_nothing_refused_or_queued_for_its_question() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (
            await worker.answer(await worker.read(run_id), {waiting_id: "not a bool"}) == "queued"
        )
        gate.set()
        waiting = await store.ended(run_id)
        assert (waiting.status, list(waiting.refused)) == ("waiting", [waiting_id])
        store.rows[run_id] = dataclasses.replace(
            store.rows[run_id],
            state=json.dumps({**store.state(run_id), "queued_answers": {waiting_id: False}}),
        )  # as a resume call that raised leaves an answer queued; the test above reaches that through a raise

        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "started"
        completed = await store.ended(run_id)
        assert (completed.status, completed.refused, completed.outputs) == (
            "complete",
            {},
            {"busy": "x", "after": True},
        )


@host_type("fragile", title="Fragile")
class Fragile:
    """A host's type whose own check has a bug: it raises a ``KeyError`` instead of refusing the value."""

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
        assert await worker.answer(await worker.read(run_id), {waiting.id: "anything"}) == "queued"
        gate.set()
        ended = await store.ended(run_id)
        await asyncio.sleep(0.4)  # long enough for a resume call given up to be started again

    assert (
        ended.status,
        [(problem.type, problem.ctx) for problem in ended.refused[waiting.id]],
    ) == (
        "waiting",
        [("answer_check_failed", {"waiting_id": waiting.id, "error": "KeyError"})],
    )
    assert await _resume_calls(store, run_id) == 1
    # no host reads the queue, and an answer left in it would show only at the run's next stop, which this run, waiting
    # on a question only a fixed type can answer, never has; so the stored state is read here
    assert store.state(run_id)["queued_answers"] == {}


async def test_a_cancel_given_a_waiting_run_an_answer_resumed_first_stops_the_answered_resume_call() -> (
    None
):
    registry, graph, gates, calls = gated()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, {"csv": "a,b", "text": "x", "question": "y"})
        gates["first"].set()
        waiting = await store.ended(run_id)
        read_while_it_waited = await worker.read(run_id)
        assert (
            await worker.answer(await worker.read(run_id), {waiting.waiting[0].id: True})
            == "started"
        )
        await asyncio.to_thread(gates["per_item_called"].wait, 5)

        assert await worker.cancel(read_while_it_waited)
        gates["second"].set()
        cancelled = await store.ended(run_id)
        assert (cancelled.status, cancelled.waiting, calls) == ("cancelled", [], ["a"])

        await worker.resume(await worker.read(run_id))  # the answer is kept: nothing is asked again
        completed = await store.ended(run_id)
        assert (completed.status, completed.outputs, calls) == (
            "complete",
            {"busy": "x", "per_item": ["a", "b"]},
            ["a", "b"],
        )


async def test_a_cancel_given_a_waiting_run_that_an_answer_completed_first_answers_false() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        assert (await store.ended(run_id)).status == "waiting"
        read_while_it_waited = await worker.read(run_id)
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "started"
        assert (await store.ended(run_id)).status == "complete"

        assert not await worker.cancel(read_while_it_waited)
        assert (await worker.read(run_id)).view().status == "complete"


def _asked_beside_a_chain() -> tuple[
    NodeRegistry, GraphRecord, dict[str, Any], dict[str, ThreadEvent], list[str], str
]:
    """A graph with a question beside a chain of three steps, its inputs, and the waiting id the question gets: the
    first step waits on ``first`` and the second on ``second``, having set ``in_second``; ``calls`` lists the steps
    called."""
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
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
        assert await worker.cancel(await worker.read(run_id))
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
        """Takes a cancel from another worker just before the write that ends the first resume call."""

        cancelled = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if not self.cancelled and "outcome" in changes:
                self.cancelled = True
                assert await other.cancel(await other.read(run_id))
            return await super().update(run_id, rev, changes)

    store = CancelAtTheEnd()
    worker, other = _worker(store, registry), _worker(store, registry)
    async with worker.serving(), other.serving():
        run_id = await _started(worker, graph, inputs)
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
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
            await worker.read(run_id)
        )  # the cancel that ended it does not end this one
        assert (await store.ended(run_id)).status == "complete"


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
            await worker.answer(await worker.read(run_id), {failed.waiting[0].id: True})
            == "started"
        )
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
        """Raises once in the middle of a resume call, as a database that drops one statement does."""

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

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if (
                fails == "the write of a checkpoint"
                and changes.keys() == {"run", "outcome"}
                and self.down()
            ):
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes)

    store = RaisesOnce()
    worker = _worker(store, registry, lease=0.3, checkpoint_every=0.01)
    async with worker.serving():
        run_id = await _started(worker, graph, {"n": 40})
        async with asyncio.timeout(5):
            await store.raised.wait()
        at_the_raise = len(calls)
        # a resume call still running would have called more by now, and no lease has run out
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


def _context_raises(caller: str) -> None:
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
        ) == (
            "failed",
            "resume_failed",
            (),
        )
        assert await store.stored_events(run_id) == [
            {"type": "run_end", "status": "failed", "cause": failed.cause.model_dump(mode="json")}
        ]


class Tenant:
    """The run's context, which the worker's ``context`` does not give."""


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
    registry, store, gate = NodeRegistry(), MemoryStore(), ThreadEvent()

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(question: str) -> bool:
        raise Ask("Go on?")

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gate.wait(5)
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
        await asyncio.sleep(0.05)  # a checkpoint is due by the time the busy step ends
        gate.set()
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

    def fails_on_the_second_resume_call(caller: str) -> None:
        resume_calls.append(caller)
        if len(resume_calls) == 2:
            raise ConnectionError("the secrets service is down")

    worker = _worker(store, registry, context=fails_on_the_second_resume_call)
    async with worker.serving():
        run_id = await _started(worker, make_graph(many, approve), {"n": 3})
        first, second, third = (
            waiting_call.id for waiting_call in (await store.ended(run_id)).waiting
        )

        assert await worker.answer(await worker.read(run_id), {first: True}) == "started"
        failed = await store.ended(run_id)
        assert failed.cause is not None and (
            failed.status,
            failed.cause.type,
            [waiting_call.id for waiting_call in failed.waiting],
        ) == ("failed", "resume_failed", [second, third])

        assert await worker.answer(await worker.read(run_id), {second: True}) == "started"
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


async def test_a_run_whose_end_the_store_fails_to_write_stays_running_and_is_resumed_once_its_lease_runs_out() -> (
    None
):
    registry, graph = _shout()

    class EndFailsOnce(MemoryStore):
        """Raises once in the write that ends a resume call, after its ``run_end`` is stored."""

        raised = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.get("status") == "complete" and not self.raised:
                self.raised = True
                raise ConnectionError("the store went away for one statement")
            return await super().update(run_id, rev, changes)

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
        ["complete", "complete"],
    )
    assert await _resume_calls(store, run_id) == 2


@host_type("opaque", title="Opaque")
class Opaque:
    """A value pydantic carries but has no JSON form for."""

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
        and (await worker.read(run_id)).view().status == "failed"
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
        run = await worker.read(await _started(worker, graph, inputs))

        async def followed(after: int = -1) -> list[tuple[int, str]]:
            return [event async for event in worker.events(run, after)]

        following = asyncio.create_task(followed())
        await asyncio.sleep(0.1)  # the follower has the resume call's first events by now
        before = store.reads
        await asyncio.sleep(0.3)  # thirty polls that find no event
        during = store.reads - before
        gate.set()
        events = await following

        assert during == 0 and [number for number, _ in events] == list(range(len(events)))
        assert [json.loads(body) for _, body in events] == await store.stored_events(run.id)
        assert (
            await followed(len(events) - 1) == []
        )  # a follower that has every event of a run that waits is told so at once
        assert await followed(len(events) - 2) == events[-1:]


async def test_a_follower_that_got_a_run_end_while_the_row_still_reads_running_keeps_asking_until_it_changes() -> (
    None
):
    registry, graph = _shout()

    class RowLags(MemoryStore):
        """Reads a run that has ended as still running a few times, as a reader between the ``run_end`` and the row's write does."""

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
                json.loads(body) async for _, body in worker.events(await worker.read(run_id))
            ]

        assert [event["type"] for event in events].count("run_end") == 1 and events[-1][
            "type"
        ] == "run_end"
        assert store.lagged >= 5


# -- several workers, a shutdown ------------------------------------------------------------------


def _slow_items() -> tuple[NodeRegistry, GraphRecord, ThreadEvent, list[int]]:
    """A graph that works through a Series one slow item at a time; ``called`` is set once the first item is, and
    ``calls`` lists every item called."""
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
        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.keys() == {"run", "outcome"}:
                written.append(json.loads(changes["outcome"]))
            return await super().update(run_id, rev, changes)

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


async def test_a_run_handed_back_by_a_worker_shutting_down_reads_as_running_with_no_cause_and_its_saved_run_as_its_outcomes_say() -> (
    None
):
    registry, graph, called, _ = _slow_items()
    store = MemoryStore()
    worker = _worker(store, registry)

    async with worker.serving():
        run_id = await _started(worker, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)

    row, view = store.rows[run_id], (await worker.read(run_id)).view()
    assert (view.status, view.cause, row.owner, row.lease_until) == (
        "running",
        None,
        None,
        0.0,
    )
    assert (
        Run.from_json(row.run, Graph.compile(row.graph, registry)).status == "ready"
    )  # only the row says a worker is to go on


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


async def test_a_cancel_from_another_worker_that_lands_as_a_worker_hands_the_run_back_is_kept_for_the_worker_that_takes_it() -> (
    None
):
    registry, graph, gate, inputs, _ = asking()

    class CancelAtHandBack(MemoryStore):
        """Takes a cancel from another worker just before the write handing the run back."""

        cancelled = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if not self.cancelled and changes.get("lease_until") == 0.0:
                self.cancelled = True
                assert await other.cancel(await other.read(run_id))
            return await super().update(run_id, rev, changes)

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
        """Never answers the first renewal, as one statement on a connection that went quiet."""

        hung = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.keys() == {"lease_until"} and not self.hung:
                self.hung = True
                await AsyncioEvent().wait()
            return await super().update(run_id, rev, changes)

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
        """Never answers the renewals of one worker, as a connection that went quiet."""

        of: str | None = None

        async def create(self, run: StoredRun) -> None:
            if self.of is None:
                self.of = run.owner  # the worker that begins the run
            await super().create(run)

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.keys() == {"lease_until"} and self.rows[run_id].owner == self.of:
                await AsyncioEvent().wait()
            return await super().update(run_id, rev, changes)

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
        """Once ``armed``, raises once in the write of a checkpoint: the last one, written as the run is handed back."""

        armed = raised = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if self.armed and not self.raised and changes.keys() == {"run", "outcome"}:
                self.raised = True
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes)

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
        """Starts a second run on the worker as it hands its first back."""

        late: str | None = None

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if self.late is None and changes.get("lease_until") == 0.0:
                self.late = await _started(worker, graph, {"n": 40})
            return await super().update(run_id, rev, changes)

    store = BeginsAtHandBack()
    worker = _worker(store, registry)

    async with worker.serving():
        await _started(worker, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)

    assert store.late is not None
    late = store.rows[store.late]
    assert (late.status, late.owner, late.lease_until) == ("running", None, 0.0)


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

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if fails == "the claim of a run" and not self.raised:
                self.raised = True
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes)

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


async def test_a_renewal_that_raises_is_logged_and_the_next_one_keeps_the_lease_so_no_other_worker_takes_the_run(
    caplog: LogCaptureFixture,
) -> None:
    registry, graph, called, calls = _slow_items()

    class RenewalFailsOnce(MemoryStore):
        raised = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.keys() == {"lease_until"} and not self.raised:
                self.raised = True
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes)

    store = RenewalFailsOnce()
    holder, other = _worker(store, registry, lease=0.3), _worker(store, registry, lease=0.3)
    async with holder.serving(), other.serving():
        run_id = await _started(holder, graph, {"n": 40})
        ended = await store.ended(run_id)

    assert store.raised and ended.status == "complete"
    assert sorted(calls) == list(range(40)) and await _resume_calls(store, run_id) == 1
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
    checkpoint = run.answer({run.waiting[0].id: True})
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
        )  # as when its lease ran out unrenewed
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
