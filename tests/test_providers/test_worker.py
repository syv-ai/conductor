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
from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.router import RunView
from conductor_providers.fastapi.router_problems import RefusedError
from conductor_providers.fastapi.stored_run import RunAsRead, StoredState
from conductor_providers.fastapi.worker import Worker
from pydantic import GetCoreSchemaHandler, TypeAdapter
from pydantic_core import core_schema
from pytest import LogCaptureFixture, MonkeyPatch

from tests.test_providers.in_memory import MemoryStore, asking, gated, left_running


def _worker(store: MemoryStore, registry: NodeRegistry[Any], **settings: Any) -> Worker[str]:
    """A worker on ``store`` whose callers are names, with the router's own defaults where ``settings`` name none."""
    defaults: dict[str, Any] = {"context": lambda who: None, "max_items": None, "lease": 30.0, "checkpoint_every": 5.0, "resume_timeout": 300.0, "poll": 0.1}
    return Worker(store=store, nodes_for=lambda who: registry, callers=TypeAdapter(str), **{**defaults, **settings})


async def _started(worker: Worker[str], graph: GraphRecord, inputs: Mapping[str, Any]) -> str:
    return await worker.begin(graph, inputs, "anna")


async def _resume_calls(store: MemoryStore, run_id: str) -> int:
    """How many resume calls of the run have started: one ``run_start`` event each."""
    return [event["type"] for event in await store.stored_events(run_id)].count("run_start")


# -- what a run reads as --------------------------------------------------------------------------


async def test_an_output_that_isnt_finite_reads_as_its_text() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("edges", title="Edges", description="For a test.")
    def edges(x: int) -> list[float]:
        return [float("inf"), float("-inf"), float("nan"), x]

    run_id = await _started(_worker(store, registry), make_graph(edges), {"x": 1})

    assert (await store.ended(run_id)).outputs == {"result": ["Infinity", "-Infinity", "NaN", 1.0]}


@host_type("cents", title="Cents")
class Cents:
    """A host's type that only its own schema writes as JSON."""


    def __init__(self, cents: int) -> None:
        self.cents = cents

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, cls) else cls(int(value)),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda price: price.cents),
        )


async def test_an_output_only_its_own_type_can_write_completes_as_it_does_in_the_engine() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("price", title="Price", description="For a test.")
    def price(cents: int) -> Cents:
        return Cents(cents)

    @registry.node("prices", title="Prices", description="For a test.")
    def prices(cents: int) -> Series[Cents]:
        return [Cents(cents), Cents(cents + 1)]

    graph = make_graph(price, prices, edges=[], inputs={"price": "price.cents", "prices": "prices.cents"}, outputs={"price": "price.result", "prices": "prices.result"})
    run_id = await _started(_worker(store, registry), graph, {"price": 5, "prices": 7})

    ended = await store.ended(run_id)
    assert (ended.status, ended.outputs, ended.cause) == ("complete", {"price": 5, "prices": [7, 8]}, None)


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

    run_id = await _started(_worker(store, registry), make_graph(split, as_table), {"text": "a,b"})

    ended = await store.ended(run_id)
    columns = [{"columns": [{"name": "text", "type": "str", "values": [text]}]} for text in ("a", "b")]
    assert (ended.status, ended.outputs) == ("complete", {"result": columns})


# -- an item limit, a resume ----------------------------------------------------------------------


async def test_a_run_stopped_at_its_item_limit_says_why_and_is_resumed_at_the_same_or_a_higher_limit() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("up", title="Up", description="For a test.")
    def up(text: str) -> str:
        return text.upper()

    worker = _worker(store, registry, max_items=1)

    async def last_end() -> dict[str, Any]:
        return [event for event in await store.stored_events(run_id) if event["type"] == "run_end"][-1]

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
    run_id = await _started(worker, make_graph(split, ask, edges=[("split.result", "ask.x")], constants={"split.n": 5}), {})
    assert (await store.ended(run_id)).status == "stopped"

    await worker.resume(await worker.read(run_id), max_items=10)
    assert json.loads(store.rows[run_id].run)["max_items"] == 10  # written with the claim, so a resume call given up before its first checkpoint is resumed under it
    waiting = (await store.ended(run_id)).waiting
    assert len(waiting) == 5
    assert await worker.answer(await worker.read(run_id), {waiting[0].id: 7}) == "started"
    after = await store.ended(run_id)
    assert (after.status, len(after.waiting)) == ("waiting", 4)


async def test_a_resume_with_a_higher_item_limit_of_an_unreadable_saved_run_is_refused_and_leaves_the_row() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("up", title="Up", description="For a test.")
    def up(text: str) -> str:
        return text.upper()

    worker = _worker(store, registry, max_items=1)
    run_id = await _started(worker, make_graph(split, up), {"text": "a,b"})
    assert (await store.ended(run_id)).status == "stopped"
    store.rows[run_id] = unreadable = dataclasses.replace(store.rows[run_id], run="{")

    with pytest.raises(InvalidCallError) as refused:
        await worker.resume(await worker.read(run_id), max_items=5)

    assert [problem.type for problem in refused.value.problems] == ["saved_run_unreadable"] and store.rows[run_id] == unreadable


@pytest.mark.parametrize("status", ["running", "waiting", "complete"])
async def test_only_a_run_whose_last_resume_call_ended_early_is_resumed(status: str) -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    run_id = await _started(worker, graph, inputs)
    if status != "running":
        gate.set()
        await store.ended(run_id)
    if status == "complete":
        await worker.answer(await worker.read(run_id), {waiting_id: True})
        await store.ended(run_id)

    with pytest.raises(RefusedError) as refused:
        await worker.resume(await worker.read(run_id))

    assert [(problem.type, problem.ctx) for problem in refused.value.problems] == [("run_not_resumable", {"status": status})]
    gate.set()


# -- a cancel -------------------------------------------------------------------------------------


async def test_a_cancel_reaches_the_resume_call_that_an_answer_queued_during_the_previous_one_started() -> None:
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

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting.id: True}) == "queued"
    gates["first"].set()
    await asyncio.to_thread(gates["per_item_called"].wait, 5)
    assert await worker.cancel(await worker.read(run_id))
    gates["second"].set()

    assert (await store.ended(run_id)).status == "cancelled" and calls == ["a"]


async def test_a_cancel_sent_while_a_dead_worker_held_the_run_is_kept_when_another_takes_it_over() -> None:
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
    await worker.sweep()
    assert (await store.ended("taken")).status == "cancelled" and calls == []

    await worker.resume(await worker.read("taken"))
    assert (await store.ended("taken")).status == "complete" and calls == ["a", "b"]


async def test_a_cancel_of_a_run_that_isnt_running_cancels_nothing() -> None:
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    run_id = await _started(worker, graph, inputs)
    await store.ended(run_id)

    assert not await worker.cancel(await worker.read(run_id))
    assert (await store.ended(run_id)).status == "waiting"


async def test_the_worker_goes_by_a_runs_state_and_never_reads_the_status_column() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()

    class DropsStatus(MemoryStore):
        """Writes every column but ``status``, as a host's store whose ``update`` misses the one it only queries."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            return await super().update(run_id, rev, {name: value for name, value in changes.items() if name != "status"})

    store = DropsStatus()
    worker = _worker(store, registry)
    run_id = await _started(worker, graph, inputs)
    waiting = RunAsRead.of(await store.until(run_id, lambda row: row.owner is None))
    assert (waiting.row.status, RunView.of(waiting).status) == ("running", "waiting")

    assert not await worker.cancel(waiting)
    with pytest.raises(RefusedError) as refused:
        await worker.resume(waiting)
    assert [problem.ctx for problem in refused.value.problems] == [{"status": "waiting"}]
    assert await worker.answer(waiting, {waiting_id: True}) == "started"
    completed = RunAsRead.of(await store.until(run_id, lambda row: row.owner is None))
    assert RunView.of(completed).outputs == {"busy": "x", "after": True}


# -- answers --------------------------------------------------------------------------------------


async def test_two_workers_answering_one_run_at_once_start_one_resume_call() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    one, other = _worker(store, registry), _worker(store, registry)
    run_id = await _started(one, graph, inputs)
    await store.ended(run_id)
    row = await one.read(run_id)

    outcomes = await asyncio.gather(one.answer(row, {waiting_id: True}), other.answer(row, {waiting_id: True}), return_exceptions=True)

    lost = [outcome for outcome in outcomes if isinstance(outcome, RefusedError)]
    assert [outcome for outcome in outcomes if not isinstance(outcome, RefusedError)] == ["started"]
    assert [[problem.type for problem in refused.problems] for refused in lost] == [["run_changed"]]
    assert (await store.ended(run_id)).status == "complete" and await _resume_calls(store, run_id) == 2


async def test_a_second_answer_queued_for_a_question_is_refused_and_a_complete_run_takes_none() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)
    run_id = await _started(worker, graph, inputs)

    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    with pytest.raises(RefusedError) as twice:
        await worker.answer(await worker.read(run_id), {waiting_id: False})
    assert [(problem.type, problem.ctx) for problem in twice.value.problems] == [("already_answered", {"waiting_id": waiting_id})]

    gate.set()
    assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}
    with pytest.raises(RefusedError) as complete:
        await worker.answer(await worker.read(run_id), {})
    assert [problem.type for problem in complete.value.problems] == ["run_complete"]


async def test_an_answer_queued_during_a_resume_call_starts_the_next_however_long_the_first_takes_to_write_its_end() -> None:
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

    assert (ended.status, ended.refused, ended.outputs) == ("complete", {}, {"busy": "x", "after": True})
    assert await _resume_calls(store, run_id) == 2


async def test_the_end_of_a_resume_call_costs_one_read_one_event_and_one_write_also_with_an_answer_queued() -> None:
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
    run_id = await _started(worker, graph, inputs)
    row = await worker.read(run_id)
    assert await worker.answer(row, {waiting_id: True}) == "queued"
    store.reads = store.writes = 0
    gate.set()

    assert (await store.ended(run_id)).status == "complete"
    assert (store.reads, store.writes) == (2, 2)  # one of each for the end of each of the two resume calls
    assert [event["type"] for event in await store.stored_events(run_id)].count("run_end") == 2


async def test_an_answer_queued_just_before_a_resume_call_writes_its_end_starts_the_next_resume_call() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()

    class AnswerAtTheEnd(MemoryStore):
        """Takes an answer from another worker, queued for the running resume call, just before the write that ends it."""

        answered = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if not self.answered and changes.get("status") == "waiting":
                self.answered = True
                assert await other.answer(RunAsRead.of(self.rows[run_id]), {waiting_id: True}) == "queued"
            return await super().update(run_id, rev, changes)

    store = AnswerAtTheEnd()
    worker, other = _worker(store, registry), _worker(store, registry)
    run_id = await _started(worker, graph, inputs)
    ended = await store.ended(run_id)

    assert store.answered and (ended.status, ended.refused, ended.outputs) == ("complete", {}, {"busy": "x", "after": True})
    assert [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"] == ["waiting", "complete"]


async def test_a_run_that_takes_longer_than_the_lease_to_write_stays_with_its_worker_and_is_written_once(monkeypatch: MonkeyPatch) -> None:
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
        await asyncio.sleep(0.4)  # a second worker that took the run would be running it again by now

    assert [waiting_call.id for waiting_call in ended.waiting] == [waiting_id] and RunView.of(await one.read(run_id)) == ended
    assert await _resume_calls(store, run_id) == 1
    assert slow_writes == ["waiting"]


async def test_an_answer_queued_during_a_resume_call_stopped_at_the_item_limit_starts_the_next_resume_call() -> None:
    registry, graph, gate, inputs, waiting_id = asking(items=True)
    store = MemoryStore()
    worker = _worker(store, registry, max_items=1)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    gate.set()
    ended = await store.ended(run_id)  # the next resume call stops at the same item limit

    assert (ended.status, ended.waiting, ended.refused) == ("stopped", [], {})
    assert await _resume_calls(store, run_id) == 2


async def test_an_answer_queued_for_a_question_a_timed_out_resume_call_never_asked_is_refused_as_not_waiting() -> None:
    registry, graph, gate, inputs, waiting_id = asking(items=True)
    store = MemoryStore()
    worker = _worker(store, registry, resume_timeout=0.2)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    await asyncio.sleep(0.3)  # past the resume call's timeout; the node call it waits for then finishes
    gate.set()
    ended = await store.ended(run_id)

    assert ended.status == "timed_out"
    assert [(problem.type, problem.ctx["status"]) for problem in ended.refused[waiting_id]] == [("not_waiting", "timed_out")]
    assert await _resume_calls(store, run_id) == 1


async def test_an_answer_queued_before_a_cancelled_run_asks_its_question_is_refused_as_not_waiting_and_starts_nothing() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    assert await worker.cancel(await worker.read(run_id))
    gate.set()
    ended = await store.ended(run_id)

    assert ended.status == "cancelled"
    assert [(problem.type, problem.ctx["status"]) for problem in ended.refused[waiting_id]] == [("not_waiting", "cancelled")]
    assert await _resume_calls(store, run_id) == 1


async def test_an_answer_queued_during_a_resume_call_that_raised_stays_queued_and_is_taken_after_a_resume() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    resume_calls = []

    def fails_on_the_first_resume_call(caller: str) -> None:
        resume_calls.append(caller)
        if len(resume_calls) == 1:
            raise ConnectionError("the secrets service is down")

    store = MemoryStore()
    worker = _worker(store, registry, context=fails_on_the_first_resume_call)
    left = left_running("left", graph, Graph.compile(graph, registry).with_inputs(inputs), owner="a-dead-worker")
    await store.create(left)
    assert await worker.answer(await worker.read("left"), {waiting_id: True}) == "queued"

    await worker.sweep()
    failed = await store.ended("left")
    assert (failed.status, failed.refused, (await worker.read("left")).state.queued_answers) == ("failed", {}, {waiting_id: True})

    await worker.resume(await worker.read("left"))
    completed = await store.ended("left")  # the resume call after the resume ended waiting, and took the answer queued two resume calls before
    assert (completed.status, completed.refused, completed.outputs) == ("complete", {}, {"busy": "x", "after": True})


async def test_an_answer_that_is_taken_leaves_nothing_refused_or_queued_for_its_question() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: "not a bool"}) == "queued"
    gate.set()
    waiting = await store.ended(run_id)
    assert (waiting.status, list(waiting.refused)) == ("waiting", [waiting_id])
    queued_before = (await worker.read(run_id)).state.model_copy(update={"queued_answers": {waiting_id: False}})
    store.rows[run_id] = dataclasses.replace(store.rows[run_id], **queued_before.columns())  # as a resume call that raised leaves an answer queued

    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "started"
    completed = await store.ended(run_id)
    assert (completed.status, completed.refused, completed.outputs) == ("complete", {}, {"busy": "x", "after": True})


@host_type("fragile", title="Fragile")
class Fragile:
    """A host's type whose own check has a bug: it raises a ``KeyError`` instead of refusing the value."""


    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def check(value: Any) -> Fragile:
            raise KeyError("a lookup the host's type forgot to guard")

        return core_schema.no_info_plain_validator_function(check, serialization=core_schema.plain_serializer_function_ser_schema(lambda _value: "fragile"))


async def test_a_queued_answer_whose_check_raises_is_refused_and_starts_no_second_resume_call() -> None:
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

    assert (ended.status, [(problem.type, problem.ctx) for problem in ended.refused[waiting.id]]) == (
        "waiting", [("answer_check_failed", {"waiting_id": waiting.id, "error": "KeyError"})],
    )
    assert await _resume_calls(store, run_id) == 1 and (await worker.read(run_id)).state.queued_answers == {}


async def test_a_cancel_given_a_run_read_before_its_resume_call_started_cancels_nothing() -> None:
    registry, graph, gates, calls = gated()
    store = MemoryStore()
    worker = _worker(store, registry)

    run_id = await _started(worker, graph, {"csv": "a,b", "text": "x", "question": "y"})
    gates["first"].set()
    waiting = await store.ended(run_id)
    read_while_it_waited = await worker.read(run_id)
    assert await worker.answer(await worker.read(run_id), {waiting.waiting[0].id: True}) == "started"
    await asyncio.to_thread(gates["per_item_called"].wait, 5)

    assert not await worker.cancel(read_while_it_waited)
    gates["second"].set()
    assert (await store.ended(run_id)).status == "complete" and calls == ["a", "b"]


def _asked_beside_a_chain() -> tuple[NodeRegistry, GraphRecord, dict[str, Any], dict[str, ThreadEvent], list[str], str]:
    """A graph with a question beside a chain of three steps, its inputs, and the waiting id the question gets: the
    first step waits on ``first`` and the second on ``second``, having set ``in_second``; ``calls`` lists the steps
    called."""
    registry, gates, calls = NodeRegistry(), {name: ThreadEvent() for name in ("first", "in_second", "second")}, []

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

    graph = make_graph(approve, one, two, three, edges=[("one.result", "two.text"), ("two.result", "three.text")], outputs={"approve": "approve.result", "three": "three.result"})
    inputs = {"question": "y", "text": "x"}
    for gate in gates.values():
        gate.set()
    (waiting,) = Graph.compile(graph, registry).run(inputs).waiting
    for gate in gates.values():
        gate.clear()
    calls.clear()
    return registry, graph, inputs, gates, calls, waiting.id


async def test_a_cancel_ends_the_resume_call_it_reached_and_not_the_one_an_answer_queued_meanwhile_starts() -> None:
    registry, graph, inputs, gates, calls, waiting_id = _asked_beside_a_chain()
    store = MemoryStore()
    worker = _worker(store, registry)

    run_id = await _started(worker, graph, inputs)
    gates["first"].set()
    await asyncio.to_thread(gates["in_second"].wait, 5)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    assert await worker.cancel(await worker.read(run_id))
    gates["second"].set()

    completed = await store.until(run_id, lambda row: row.status not in ("running", "cancelled"))
    ends = [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert (completed.status, ends, calls) == ("complete", ["cancelled", "complete"], ["one", "two", "three"])


async def test_a_cancel_sent_as_a_resume_call_ends_cancels_the_one_an_answer_queued_meanwhile_starts() -> None:
    registry, graph, gate, inputs, waiting_id = asking()

    class CancelAtTheEnd(MemoryStore):
        """Takes a cancel from another worker just before the write that ends the first resume call."""

        cancelled = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if not self.cancelled and "outcome" in changes:
                self.cancelled = True
                assert await other.cancel(RunAsRead.of(self.rows[run_id]))
            return await super().update(run_id, rev, changes)

    store = CancelAtTheEnd()
    worker, other = _worker(store, registry), _worker(store, registry)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    gate.set()

    ended = await store.ended(run_id)
    ends = [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert store.cancelled and (ended.status, ends, ended.outputs) == ("cancelled", ["waiting", "cancelled"], None)
    assert not (await worker.read(run_id)).state.cancel_asked


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
    run_id = await _started(worker, make_graph(many, approve), {"n": 3})
    failed = await store.ended(run_id)
    assert (failed.status, len(failed.waiting)) == ("failed", 2)

    assert await worker.answer(await worker.read(run_id), {failed.waiting[0].id: True}) == "started"
    after = await store.ended(run_id)
    assert (after.status, [waiting_call.id for waiting_call in after.waiting]) == ("failed", [failed.waiting[1].id])


# -- a resume call that raised --------------------------------------------------------------------


@pytest.mark.parametrize("fails", ["the write of an event", "the write of a checkpoint"])
async def test_a_resume_call_the_store_raises_in_is_given_up_and_the_run_is_resumed_once_its_lease_runs_out(fails: str) -> None:
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
            if fails == "the write of a checkpoint" and changes.keys() == {"run", "outcome"} and self.down():
                raise ConnectionError("the store is down for one statement")
            return await super().update(run_id, rev, changes)

    store = RaisesOnce()
    worker = _worker(store, registry, lease=0.3, checkpoint_every=0.01)
    async with worker.serving():
        run_id = await _started(worker, graph, {"n": 40})
        async with asyncio.timeout(5):
            await store.raised.wait()
        at_the_raise = len(calls)
        await asyncio.sleep(0.1)  # a resume call still running would have called more by now, and no lease has run out
        assert len(calls) <= at_the_raise + 1 and store.rows[run_id].status == "running"
        ended = await store.ended(run_id)  # once the lease ran out and a sweep took the run

    ends = [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert (ended.status, ended.cause, ends, ended.outputs) == ("complete", None, ["complete"], {"result": list(range(40))})
    assert await _resume_calls(store, run_id) == 2


def _shout() -> tuple[NodeRegistry, GraphRecord]:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    return registry, make_graph(shout)


def _context_raises(caller: str) -> None:
    raise ConnectionError("the secrets service is down")


async def test_a_resume_call_that_raised_outside_a_node_fails_the_run_with_a_cause_its_run_end_has_too() -> None:
    registry, graph = _shout()
    store = MemoryStore()

    run_id = await _started(_worker(store, registry, context=_context_raises), graph, {"text": "a"})
    failed = await store.ended(run_id)

    assert failed.cause is not None and (failed.status, failed.cause.type, failed.cause.loc) == ("failed", "resume_failed", ())
    assert await store.stored_events(run_id) == [{"type": "run_end", "status": "failed", "cause": failed.cause.model_dump(mode="json")}]


class Tenant:
    """The run's context, which the worker's ``context`` does not give."""


async def test_a_run_conductor_refuses_to_resume_fails_with_its_first_problem_and_logs_them_all(caplog: LogCaptureFixture) -> None:
    registry, store = NodeRegistry(context_type=Tenant), MemoryStore()

    @registry.node("stamp", title="Stamp", description="For a test.")
    def stamp(tenant: Context[Tenant], text: str) -> str:
        return text

    @registry.node("seal", title="Seal", description="For a test.")
    def seal(tenant: Context[Tenant], text: str) -> str:
        return text

    run_id = await _started(_worker(store, registry), make_graph(stamp, seal), {"text": "a"})
    failed = await store.ended(run_id)

    assert failed.cause is not None and (failed.status, failed.cause.type, failed.cause.loc) == ("failed", "context_missing", ("stamp", "tenant"))
    assert failed.cause.ctx["needs"] == "Tenant"
    assert await store.stored_events(run_id) == [{"type": "run_end", "status": "failed", "cause": failed.cause.model_dump(mode="json")}]
    (logged,) = [record.getMessage() for record in caplog.records if run_id in record.getMessage()]
    assert "stamp.tenant" in logged and "seal.tenant" in logged and "\n" not in logged


async def test_a_run_whose_resume_call_raised_after_a_checkpoint_lists_the_calls_waiting_in_that_checkpoint() -> None:
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

    graph = make_graph(approve, busy, opaque, edges=[("busy.result", "opaque.text")], outputs={"approve": "approve.result", "opaque": "opaque.result"})
    run_id = await _started(_worker(store, registry, checkpoint_every=0.01), graph, {"question": "y", "text": "x"})
    await asyncio.sleep(0.05)  # a checkpoint is due by the time the busy step ends
    gate.set()
    failed = await store.ended(run_id)

    assert failed.cause is not None and (failed.status, failed.cause.type) == ("failed", "run_unwritable")
    stored = Run.from_json(store.rows[run_id].run, Graph.compile(graph, registry))
    assert [waiting_call.id for waiting_call in failed.waiting] == [waiting_call.id for waiting_call in stored.waiting] and len(failed.waiting) == 1


async def test_a_run_whose_resume_call_raised_lists_the_calls_waiting_as_stored_and_takes_answers_without_a_resume() -> None:
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
    run_id = await _started(worker, make_graph(many, approve), {"n": 3})
    first, second, third = (waiting_call.id for waiting_call in (await store.ended(run_id)).waiting)

    assert await worker.answer(await worker.read(run_id), {first: True}) == "started"
    failed = await store.ended(run_id)
    assert failed.cause is not None and (failed.status, failed.cause.type, [waiting_call.id for waiting_call in failed.waiting]) == ("failed", "resume_failed", [second, third])

    assert await worker.answer(await worker.read(run_id), {second: True}) == "started"
    assert [waiting_call.id for waiting_call in (await store.ended(run_id)).waiting] == [third]


async def test_a_saved_run_that_cant_be_read_fails_the_run_and_lists_nothing_as_waiting() -> None:
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()
    worker = _worker(store, registry)
    await store.create(dataclasses.replace(left_running("left", graph, Graph.compile(graph, registry).run(inputs), owner="a-dead-worker"), run="{"))

    await worker.sweep()
    failed = await store.ended("left")
    assert failed.cause is not None and (failed.status, failed.cause.type, failed.waiting) == ("failed", "saved_run_unreadable", [])


async def test_a_run_whose_end_the_store_fails_to_write_stays_running_and_is_resumed_once_its_lease_runs_out() -> None:
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

    ends = [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert store.raised and (ended.status, ended.cause, ends) == ("complete", None, ["complete", "complete"])
    assert await _resume_calls(store, run_id) == 2


@host_type("opaque", title="Opaque")
class Opaque:
    """A value pydantic carries but has no JSON form for."""


    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls)


@pytest.mark.parametrize(("graph_of", "at"), [(("opaque",), "opaque.result"), (("opaque", "shown"), "opaque.result")], ids=["an output of the graph", "a value inside the run"])
async def test_a_run_that_cant_be_written_fails_with_run_unwritable_the_first_time_and_is_not_resumed_again(graph_of: tuple[str, ...], at: str) -> None:
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

    ends = [(event["status"], event["cause"]["type"]) for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert ended.cause is not None
    assert (ended.status, ended.cause.type, ended.cause.ctx["where"], ended.waiting, ends) == ("failed", "run_unwritable", at, [], [("failed", "run_unwritable")])
    assert await _resume_calls(store, run_id) == 1 and (await worker.read(run_id)).state.run_state.status == "failed"


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
    assert [json.loads(body) for _, body in events] == await store.stored_events(run.row.id)
    assert await followed(len(events) - 1) == []  # a follower that has every event of a run that waits is told so at once
    assert await followed(len(events) - 2) == events[-1:]


async def test_a_follower_that_got_a_run_end_while_the_row_still_reads_running_keeps_asking_until_it_changes() -> None:
    registry, graph = _shout()

    class RowLags(MemoryStore):
        """Reads a run that has ended as still running a few times, as a reader between the ``run_end`` and the row's write does."""

        lagged = 0

        async def get(self, run_id: str) -> StoredRun | None:
            row = self.rows[run_id]
            if row.status == "complete" and self.lagged < 5:
                self.lagged += 1
                return dataclasses.replace(row, **StoredState.started().columns())
            return row

    store = RowLags()
    worker = _worker(store, registry, poll=0.01)
    run_id = await _started(worker, graph, {"text": "a"})
    await store.ended(run_id)

    async with asyncio.timeout(5):
        events = [json.loads(body) async for _, body in worker.events(await worker.read(run_id))]

    assert [event["type"] for event in events].count("run_end") == 1 and events[-1]["type"] == "run_end"
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


async def test_a_checkpoint_writes_the_runs_progress_with_the_outcome_as_the_run_last_stopped() -> None:
    registry, graph, _, _ = _slow_items()
    written: list[dict[str, Any]] = []

    class Recording(MemoryStore):
        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.keys() == {"run", "outcome"}:
                written.append(json.loads(changes["outcome"]))
            return await super().update(run_id, rev, changes)

    store = Recording()
    run_id = await _started(_worker(store, registry, checkpoint_every=0.01), graph, {"n": 20})
    await store.ended(run_id)

    assert written and {outcome["waiting"] == [] and outcome["outputs"] is None for outcome in written} == {True}
    finished = [outcome["progress"]["nodes"]["slow"]["finished"] for outcome in written]
    assert finished == sorted(finished) and finished[0] < 20 and written[0]["progress"]["nodes"]["slow"]["total"] == 20


async def test_a_run_handed_back_by_a_worker_shutting_down_reads_as_running_with_no_cause_and_so_does_the_run_saved_for_it() -> None:
    registry, graph, called, _ = _slow_items()
    store = MemoryStore()
    worker = _worker(store, registry)

    async with worker.serving():
        run_id = await _started(worker, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)

    run = await worker.read(run_id)
    view = RunView.of(run)
    assert (view.status, view.cause, run.row.owner, run.row.lease_until) == ("running", None, None, 0.0)
    assert Run.from_json(run.row.run, Graph.compile(run.row.graph, registry)).status == "running"


async def test_a_run_handed_back_at_shutdown_stores_no_run_end_and_the_worker_that_takes_it_repeats_no_call() -> None:
    registry, graph, called, calls = _slow_items()
    store = MemoryStore()
    first, second = _worker(store, registry), _worker(store, registry)

    async with first.serving():
        run_id = await _started(first, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)
    assert "run_end" not in [event["type"] for event in await store.stored_events(run_id)]

    await second.sweep()
    assert (await store.ended(run_id)).outputs == {"result": list(range(40))}
    events = [event["type"] for event in await store.stored_events(run_id)]
    assert (events.count("run_start"), events.count("run_end"), events[-1]) == (2, 1, "run_end")
    assert sorted(calls) == list(range(40))


async def test_a_cancel_from_another_worker_that_lands_as_a_worker_hands_the_run_back_is_kept_for_the_worker_that_takes_it() -> None:
    registry, graph, gate, inputs, _ = asking()

    class CancelAtHandBack(MemoryStore):
        """Takes a cancel from another worker just before the write handing the run back."""

        cancelled = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if not self.cancelled and changes.get("lease_until") == 0.0:
                self.cancelled = True
                assert await other.cancel(RunAsRead.of(self.rows[run_id]))
            return await super().update(run_id, rev, changes)

    store = CancelAtHandBack()
    worker, other = _worker(store, registry), _worker(store, registry)

    async with worker.serving():
        run_id = await _started(worker, graph, inputs)
        asyncio.get_running_loop().call_later(0.05, gate.set)

    handed_back = await worker.read(run_id)
    assert store.cancelled and handed_back.state.cancel_asked
    assert (handed_back.row.status, handed_back.row.owner, handed_back.row.lease_until) == ("running", None, 0.0)
    assert "run_end" not in [event["type"] for event in await store.stored_events(run_id)]

    await other.sweep()
    row = await store.until(run_id, lambda row: row.status != "running")
    assert (row.status, row.owner, row.lease_until) == ("cancelled", None, None)
    assert [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"] == ["cancelled"]


async def test_one_renewal_the_store_never_answers_is_followed_at_once_by_the_next_so_the_worker_keeps_the_run() -> None:
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


async def test_a_worker_whose_renewal_never_returns_stops_once_another_took_the_run_and_can_shut_down() -> None:
    registry, graph, called, calls = _slow_items()

    class RenewalHangs(MemoryStore):
        """Never answers the renewals of one worker, as a connection that went quiet."""

        of: str | None = None

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.keys() == {"lease_until"} and self.rows[run_id].owner == self.of:
                await AsyncioEvent().wait()
            return await super().update(run_id, rev, changes)

    store = RenewalHangs()
    hung, other = _worker(store, registry, lease=0.15, checkpoint_every=0.02), _worker(store, registry, lease=0.15, checkpoint_every=0.02)
    store.of = hung.id

    async with asyncio.timeout(8):
        async with hung.serving():
            run_id = await _started(hung, graph, {"n": 40})
            async with other.serving():
                ended = await store.ended(run_id)
        # leaving ``hung.serving()`` waits for its resume call, which ends only once it learns it lost the run

    ends = [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert (ended.status, ends, ended.outputs) == ("complete", ["complete"], {"result": list(range(40))})
    assert len(calls) < 2 * 40  # the resume call that lost the run stopped before it had called everything again


async def test_a_deploy_whose_last_checkpoint_the_store_fails_to_take_leaves_the_run_running_for_the_worker_that_takes_it() -> None:
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
        await asyncio.to_thread(called.wait, 5)
        store.armed = True
    row = store.rows[run_id]
    assert store.raised and (row.status, row.owner) == ("running", deployed.id)

    async with other.serving():
        ended = await store.ended(run_id)  # once the lease the deployed worker left ran out

    ends = [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert (ended.status, ended.cause, ends, ended.outputs) == ("complete", None, ["complete"], {"result": list(range(40))})


async def test_a_run_begun_as_a_worker_shuts_down_is_handed_back_before_the_worker_has_shut_down() -> None:
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


@pytest.mark.parametrize("fails", ["the read of the expired runs", "the claim of a run", "a row that can't be read"])
async def test_a_sweep_that_raises_is_logged_and_the_next_one_takes_the_run(fails: str, caplog: LogCaptureFixture) -> None:
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
    left = left_running("left", graph, Graph.compile(graph, registry).with_inputs({"text": "a"}), owner=None)
    if fails == "a row that can't be read":
        await store.create(dataclasses.replace(left, id="broken", state="{"))
    await store.create(left)

    async with worker.serving():
        ended = await store.ended("left")

    assert ended.outputs == {"result": "A"}
    logged = [record.exc_info is not None for record in caplog.records if record.name == "conductor_providers.fastapi"]
    assert logged and all(logged)


async def test_a_renewal_that_raises_is_logged_and_the_next_one_keeps_the_lease_so_no_other_worker_takes_the_run(caplog: LogCaptureFixture) -> None:
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
    async with other.serving():
        run_id = await _started(holder, graph, {"n": 40})
        ended = await store.ended(run_id)

    assert store.raised and ended.status == "complete"
    assert sorted(calls) == list(range(40)) and await _resume_calls(store, run_id) == 1
    logged = [record.exc_info is not None for record in caplog.records if record.name == "conductor_providers.fastapi"]
    assert logged and all(logged)


@pytest.mark.parametrize("ends", ["complete", "waiting"])
async def test_a_run_that_stopped_running_has_no_lease_so_no_sweep_takes_it_again(ends: str) -> None:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    store = MemoryStore()
    worker = _worker(store, registry)
    run_id = await _started(worker, make_graph(shout if ends == "complete" else approve), {"text": "a"})
    assert (await store.ended(run_id)).status == ends

    row = (await worker.read(run_id)).row
    assert (row.owner, row.lease_until) == (None, None) and await store.expired(time.time()) == []
    await worker.sweep()
    assert (await worker.read(run_id)).row == row and await _resume_calls(store, run_id) == 1


async def test_a_run_whose_worker_died_is_taken_by_one_of_two_workers_and_finished_from_its_checkpoint() -> None:
    registry, graph, gates, calls = gated()
    for gate in gates.values():
        gate.set()
    run = Graph.compile(graph, registry).run({"csv": "a,b", "text": "x", "question": "y"})
    checkpoint = run.answer({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    one, other = _worker(store, registry), _worker(store, registry)
    await store.create(left_running("left", graph, checkpoint, owner="a-dead-worker"))

    await asyncio.gather(one.sweep(), other.sweep())

    assert (await store.ended("left")).outputs == {"busy": "x", "per_item": ["a", "b"]}
    assert calls == ["a", "b"] and await _resume_calls(store, "left") == 1


async def test_a_worker_whose_run_another_took_stops_resuming_it_and_writes_nothing_more() -> None:
    registry, graph, gates, calls = gated()
    inputs = {"csv": "a,b", "text": "x", "question": "y"}
    store = MemoryStore()
    slow, taker = _worker(store, registry, lease=0.15), _worker(store, registry)

    async with slow.serving():  # leaving it waits for the resume call the slow worker still runs
        run_id = await _started(slow, graph, inputs)
        row = (await slow.read(run_id)).row
        assert await store.update(run_id, row.rev, {"lease_until": 0.0})  # as when its lease ran out unrenewed
        await taker.sweep()
        taken = await taker.read(run_id)
        await asyncio.sleep(0.2)  # the slow worker's renewals would have landed by now
        assert taken.row.owner == taker.id and await taker.read(run_id) == taken
        gates["first"].set()
        ended = await store.ended(run_id)

    assert ended.status == "waiting" and RunView.of(await taker.read(run_id)) == ended
    assert [event["type"] for event in await store.stored_events(run_id)].count("run_end") == 1
    for gate in gates.values():
        gate.set()
