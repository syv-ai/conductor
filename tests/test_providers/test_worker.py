"""The ``Worker`` behind ``run_router``, called as the routes call it, on a store in memory: no HTTP, and each wait is
on the store's own writes."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from collections.abc import Mapping
from typing import Any, ClassVar, TypedDict

import polars as pl
import pytest
from conductor import (
    Ask,
    Going,
    GraphRecord,
    NodeError,
    NodeRecord,
    NodeRegistry,
    Run,
    Series,
    Table,
    make_graph,
)
from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.worker import (
    Outcome,
    RefusedError,
    RunAsRead,
    RunView,
    StoredState,
    Worker,
)
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from tests.test_providers.in_memory import MemoryStore, asking, gated


def _worker(store: MemoryStore, registry: NodeRegistry, **settings: Any) -> Worker[str]:
    """A worker on ``store`` whose callers are names, with the router's own defaults where ``settings`` name none."""
    defaults: dict[str, Any] = {"from_run": lambda who: {}, "max_items": None, "lease": 30.0, "checkpoint_every": 5.0, "leg_timeout": 300.0, "poll": 0.1}
    return Worker(store=store, nodes_for=lambda who: registry, caller_type=str, **{**defaults, **settings})


async def _started(worker: Worker[str], graph: GraphRecord, inputs: Mapping[str, Any]) -> str:
    return await worker.begin(graph, inputs, "anna")


async def _starts(store: MemoryStore, run_id: str) -> int:
    """How many legs of the run have started."""
    return [event["type"] for event in await store.stored_events(run_id)].count("run_start")


# -- what a run reads as --------------------------------------------------------------------------


async def test_an_output_that_isnt_finite_reads_as_its_text() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("edges", title="Edges", description="")
    def edges(x: int) -> list[float]:
        return [float("inf"), float("-inf"), float("nan"), x]

    run_id = await _started(_worker(store, registry), make_graph(edges), {"x": 1})

    assert (await store.ended(run_id)).outputs == {"edges.result": ["Infinity", "-Infinity", "NaN", 1.0]}


class Cents:
    """A host's type that only its own schema writes as JSON."""

    type_id: ClassVar[str] = "cents"
    type_title: ClassVar[str] = "Cents"

    def __init__(self, cents: int) -> None:
        self.cents = cents

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, cls) else cls(int(value)),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda held: held.cents),
        )


async def test_an_output_only_its_own_type_can_write_completes_as_it_does_in_the_engine() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("price", title="Price", description="")
    def price(cents: int) -> Cents:
        return Cents(cents)

    @registry.node("prices", title="Prices", description="")
    def prices(cents: int) -> Series[Cents]:
        return [Cents(cents), Cents(cents + 1)]

    graph = GraphRecord(nodes=[NodeRecord(id="price", node="price"), NodeRecord(id="prices", node="prices")])
    run_id = await _started(_worker(store, registry), graph, {"price.cents": 5, "prices.cents": 7})

    ended = await store.ended(run_id)
    assert (ended.status, ended.outputs, ended.cause) == ("complete", {"price.result": 5, "prices.result": [7, 8]}, None)


class Part(TypedDict):
    text: str


async def test_a_table_made_for_each_item_reads_as_its_columns_per_item() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("as-table", title="As table", description="")
    def as_table(text: str) -> Table[Part]:
        return pl.DataFrame({"text": [text]})

    run_id = await _started(_worker(store, registry), make_graph(split, as_table), {"text": "a,b"})

    ended = await store.ended(run_id)
    columns = [{"columns": [{"name": "text", "type": "str", "values": [text]}]} for text in ("a", "b")]
    assert (ended.status, ended.outputs) == ("complete", {"as-table.result": columns})


# -- a ceiling, a resume --------------------------------------------------------------------------


async def test_a_run_stopped_at_its_ceiling_says_why_and_is_resumed_at_the_same_or_a_higher_ceiling() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("up", title="Up", description="")
    def up(text: str) -> str:
        return text.upper()

    worker = _worker(store, registry, max_items=1)

    async def last_end() -> dict[str, Any]:
        return [event for event in await store.stored_events(run_id) if event["type"] == "run_end"][-1]

    run_id = await _started(worker, make_graph(split, up), {"text": "a,b"})
    stopped = await store.ended(run_id)
    assert stopped.status == "stopped" and stopped.cause is not None
    assert stopped.cause.type == (await last_end())["cause"]["type"] == "item_ceiling"
    assert stopped.cause.ctx == {"series": "split.result", "items": 2, "max_items": 1}

    await worker.resume(await worker.read(run_id))
    assert (await store.ended(run_id)).status == "stopped"
    await worker.resume(await worker.read(run_id), max_items=2)
    completed = await store.ended(run_id)
    assert completed.outputs == {"up.result": ["A", "B"]}
    assert completed.cause is None and (await last_end())["cause"] is None


async def test_a_ceiling_raised_by_a_resume_holds_for_the_legs_after_it() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("split", title="Split", description="")
    def split(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("ask", title="Ask", description="")
    def ask(x: int) -> int:
        raise Ask("ok?")

    worker = _worker(store, registry, max_items=3)
    run_id = await _started(worker, make_graph(split, ask, edges=[("split.result", "ask.x")], values={"split.n": 5}), {})
    assert (await store.ended(run_id)).status == "stopped"

    await worker.resume(await worker.read(run_id), max_items=10)
    waiting = (await store.ended(run_id)).waiting
    assert len(waiting) == 5
    assert await worker.answer(await worker.read(run_id), {waiting[0].id: 7}) == "started"
    after = await store.ended(run_id)
    assert (after.status, len(after.waiting)) == ("waiting", 4)


@pytest.mark.parametrize("status", ["running", "waiting", "complete"])
async def test_only_a_run_whose_last_leg_ended_early_is_resumed(status: str) -> None:
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


async def test_a_cancel_reaches_the_leg_an_answer_queued_during_the_last_one_started() -> None:
    registry, graph, gates, calls = gated()
    inputs = {"items.text": "a,b", "busy.text": "x", "approve.text": "y"}
    for gate in ("first", "second"):
        gates[gate].set()
    (waiting,) = registry.load(graph).run(inputs).waiting
    gates["first"].clear()
    gates["second"].clear()
    calls.clear()
    store = MemoryStore()
    worker = _worker(store, registry)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting.id: True}) == "queued"
    gates["first"].set()
    await asyncio.to_thread(gates["held"].wait, 5)
    assert await worker.cancel(await worker.read(run_id))
    gates["second"].set()

    assert (await store.ended(run_id)).status == "cancelled" and calls == ["a"]


async def test_a_cancel_sent_while_a_dead_worker_held_the_run_is_kept_when_another_takes_it_over() -> None:
    registry, graph, gates, calls = gated()
    for gate in gates.values():
        gate.set()
    run = registry.load(graph).run({"items.text": "a,b", "busy.text": "x", "approve.text": "y"})
    answered = run.answer({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    worker = _worker(store, registry)
    await store.create(StoredRun(
        id="taken", graph=graph.model_dump_json(), caller='"anna"', run=answered.to_json(), **StoredState().columns(), outcome=Outcome().model_dump_json(), owner="a-dead-worker", lease_until=0.0,
    ))

    assert await worker.cancel(await worker.read("taken"))
    await worker.take_over()
    assert (await store.ended("taken")).status == "cancelled" and calls == []

    await worker.resume(await worker.read("taken"))
    assert (await store.ended("taken")).status == "complete" and calls == ["a", "b"]


async def test_a_run_that_isnt_running_has_no_leg_to_cancel() -> None:
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
    assert RunView.of(completed).outputs == {"busy.result": "x", "after.result": True}


# -- answers --------------------------------------------------------------------------------------


async def test_two_workers_answering_one_run_at_once_start_one_leg() -> None:
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
    assert (await store.ended(run_id)).status == "complete" and await _starts(store, run_id) == 2


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
    assert (await store.ended(run_id)).outputs == {"busy.result": "x", "after.result": True}
    with pytest.raises(RefusedError) as complete:
        await worker.answer(await worker.read(run_id), {})
    assert [problem.type for problem in complete.value.problems] == ["run_complete"]


async def test_an_answer_queued_during_a_leg_starts_the_next_leg_however_long_its_end_takes_to_write() -> None:
    registry, graph, gate, inputs, waiting_id = asking()

    class SlowEnd(MemoryStore):
        """Writes the end of a leg, the write that takes the queued answers, after three of the lease's renewals."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if {"run", "state"} <= changes.keys():
                await asyncio.sleep(0.3)
            return await super().update(run_id, rev, changes)

    store = SlowEnd()
    worker = _worker(store, registry, lease=0.3)
    async with worker.running():
        run_id = await _started(worker, graph, inputs)
        assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
        gate.set()
        ended = await store.ended(run_id)

    assert (ended.status, ended.refused, ended.outputs) == ("complete", {}, {"busy.result": "x", "after.result": True})
    assert await _starts(store, run_id) == 2


async def test_a_legs_end_costs_the_store_a_read_an_event_and_a_write_also_when_an_answer_was_queued() -> None:
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
    assert (store.reads, store.writes) == (2, 2)  # one of each for the end of each of the two legs
    assert [event["type"] for event in await store.stored_events(run_id)].count("run_end") == 2


async def test_an_answer_that_lands_as_a_legs_end_is_written_is_taken_and_the_run_end_is_stored_once() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()

    class AnswerAtTheEnd(MemoryStore):
        """Takes an answer from another worker, queued for the running leg, just before the write that ends the leg."""

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

    assert store.answered and (ended.status, ended.refused, ended.outputs) == ("complete", {}, {"busy.result": "x", "after.result": True})
    assert [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"] == ["waiting", "complete"]


async def test_a_leg_whose_run_takes_longer_than_the_lease_to_write_keeps_it_and_writes_its_end_once(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    made = []
    to_json = Run.to_json

    def slow_to_json(self: Run) -> str:
        if self.status == "waiting":  # the run as the leg's end writes it
            made.append(self.status)
            time.sleep(0.5)  # a large run, longer than the lease
        return to_json(self)

    monkeypatch.setattr(Run, "to_json", slow_to_json)
    store = MemoryStore()
    one, other = _worker(store, registry, lease=0.3), _worker(store, registry, lease=0.3)
    async with one.running(), other.running():
        run_id = await _started(one, graph, inputs)
        ended = await store.ended(run_id)
        await asyncio.sleep(0.4)  # a second worker that took the run would be running it again by now

    assert [unit.id for unit in ended.waiting] == [waiting_id] and RunView.of(await one.read(run_id)) == ended
    assert await _starts(store, run_id) == 1
    assert made == ["waiting"]


async def test_an_answer_queued_during_a_leg_that_stops_at_its_ceiling_is_taken_and_starts_the_next_leg() -> None:
    registry, graph, gate, inputs, waiting_id = asking(items=True)
    store = MemoryStore()
    worker = _worker(store, registry, max_items=1)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    gate.set()
    ended = await store.ended(run_id)  # the next leg meets the same ceiling

    assert (ended.status, ended.waiting, ended.refused) == ("stopped", [], {})
    assert await _starts(store, run_id) == 2


async def test_an_answer_queued_for_a_question_a_timed_out_leg_never_asked_is_refused_by_the_runs_own_code() -> None:
    registry, graph, gate, inputs, waiting_id = asking(items=True)
    store = MemoryStore()
    worker = _worker(store, registry, leg_timeout=0.2)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    await asyncio.sleep(0.3)  # past the leg's timeout; the call it waits for then finishes
    gate.set()
    ended = await store.ended(run_id)

    assert ended.status == "timed_out"
    assert [(problem.type, problem.ctx["status"]) for problem in ended.refused[waiting_id]] == [("not_waiting", "timed_out")]
    assert await _starts(store, run_id) == 1


async def test_an_answer_queued_during_a_leg_that_ends_cancelled_is_refused_as_run_cancelled_and_starts_nothing() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    worker = _worker(store, registry)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    assert await worker.cancel(await worker.read(run_id))
    gate.set()
    ended = await store.ended(run_id)

    assert ended.status == "cancelled"
    assert [problem.type for problem in ended.refused[waiting_id]] == ["run_cancelled"]
    assert await _starts(store, run_id) == 1


class _FailingEvent(MemoryStore):
    """Fails to add the first event after ``armed`` is set, so the leg raises there."""

    armed = failed = False

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        if self.armed and not self.failed:
            self.failed = True
            raise ConnectionError("the store went away")
        return await super().add_event(run_id, owner, body)


async def test_an_answer_queued_during_a_leg_that_raised_stays_queued_and_is_taken_after_a_resume() -> None:
    registry, graph, gate, inputs, waiting_id = asking(items=True)
    store = _FailingEvent()
    worker = _worker(store, registry)

    run_id = await _started(worker, graph, inputs)
    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "queued"
    store.armed = True
    gate.set()
    failed = await store.ended(run_id)
    assert (failed.status, failed.refused) == ("failed", {})

    await worker.resume(await worker.read(run_id))
    completed = await store.ended(run_id)  # the leg after the resume ended waiting, and took the answer queued two legs before
    assert (completed.status, completed.refused, completed.outputs) == ("complete", {}, {"busy.result": "x", "after.result": True, "each.result": [0, 1, 2]})


async def test_a_question_that_waits_in_a_failed_run_is_answered_without_a_resume() -> None:
    registry, store = NodeRegistry(), MemoryStore()

    @registry.node("many", title="Many", description="")
    def many(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("approve", title="Approve", description="")
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
    assert (after.status, [unit.id for unit in after.waiting]) == ("failed", [failed.waiting[1].id])


# -- a leg that raised ----------------------------------------------------------------------------


async def test_a_leg_whose_event_the_store_fails_to_take_calls_no_node_once_the_run_reads_failed() -> None:
    registry, started = NodeRegistry(), []

    @registry.node("first", title="First", description="")
    def first(n: int) -> Series[str]:
        return [str(i) for i in range(n)]

    @registry.node("slow", title="Slow", description="")
    def slow(text: str) -> str:
        started.append(text)
        time.sleep(0.01)
        return text

    class Flaky(MemoryStore):
        failed_once = False

        async def add_event(self, run_id: str, owner: str, body: str) -> bool:
            if '"node_start"' in body and '"slow"' in body and not self.failed_once:
                self.failed_once = True
                raise OSError("the store is down for a moment")
            return await super().add_event(run_id, owner, body)

    store = Flaky()
    run_id = await _started(_worker(store, registry), make_graph(first, slow), {"n": 100})

    assert (await store.ended(run_id)).status == "failed"
    at_the_end = len(started)
    await asyncio.sleep(0.3)  # a leg still running would have called more by now
    assert len(started) == at_the_end < 100


def _shout() -> tuple[NodeRegistry, GraphRecord]:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    return registry, make_graph(shout)


def _no_values(caller: str) -> dict[type, Any]:
    raise ConnectionError("the secrets service is down")


async def test_a_leg_that_raised_outside_a_node_reads_failed_with_a_cause_that_its_run_end_carries_too() -> None:
    registry, graph = _shout()
    store = MemoryStore()

    run_id = await _started(_worker(store, registry, from_run=_no_values), graph, {"text": "a"})
    failed = await store.ended(run_id)

    assert failed.cause is not None and (failed.status, failed.cause.type, failed.cause.loc) == ("failed", "leg_failed", ())
    assert await store.stored_events(run_id) == [{"type": "run_end", "status": "failed", "cause": failed.cause.model_dump(mode="json")}]


async def test_a_leg_that_raised_ended_with_no_run_so_its_run_lists_nothing_as_waiting() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    legs = []

    def fails_on_the_second_leg(caller: str) -> dict[type, Any]:
        legs.append(caller)
        if len(legs) == 2:
            raise ConnectionError("the secrets service is down")
        return {}

    store = MemoryStore()
    worker = _worker(store, registry, from_run=fails_on_the_second_leg)
    run_id = await _started(worker, graph, inputs)
    assert [unit.id for unit in (await store.ended(run_id)).waiting] == [waiting_id]

    assert await worker.answer(await worker.read(run_id), {waiting_id: True}) == "started"
    failed = await store.ended(run_id)

    assert failed.cause is not None and (failed.status, failed.cause.type, failed.waiting) == ("failed", "leg_failed", [])
    await worker.resume(await worker.read(run_id))
    assert (await store.ended(run_id)).outputs == {"busy.result": "x", "after.result": True}


@pytest.mark.parametrize(("fails", "run_ends"), [("the write of the row", ["complete", "complete"]), ("the run as JSON", ["complete"])])
async def test_a_leg_whose_end_fails_stays_running_and_is_gone_on_with_and_no_run_end_is_stored_twice_for_one_leg(
    fails: str, run_ends: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, graph = _shout()

    class EndFailsOnce(MemoryStore):
        """Raises once in the write that ends a leg, after its ``run_end`` is stored."""

        raised = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if fails == "the write of the row" and changes.get("status") == "complete" and not self.raised:
                self.raised = True
                raise ConnectionError("the store went away for one statement")
            return await super().update(run_id, rev, changes)

    store = EndFailsOnce()
    to_json = Run.to_json

    def failing_to_json(self: Run) -> str:
        if fails == "the run as JSON" and self.status == "complete" and not store.raised:  # before the run_end is stored
            store.raised = True
            raise RuntimeError("a value the run can't write")
        return to_json(self)

    monkeypatch.setattr(Run, "to_json", failing_to_json)
    worker = _worker(store, registry, lease=0.15)
    async with worker.running():
        run_id = await _started(worker, graph, {"text": "a"})
        ended = await store.ended(run_id)  # once the lease ran out and a sweep took the run

    ends = [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"]
    assert store.raised and (ended.status, ended.cause, ends) == ("complete", None, run_ends)
    assert await _starts(store, run_id) == 2


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
    await asyncio.sleep(0.1)  # the follower has the leg's first events by now
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
                return dataclasses.replace(row, **StoredState(run_state=Going("running")).columns())
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


def _slow_items() -> tuple[NodeRegistry, GraphRecord, threading.Event, list[int]]:
    """A graph that works through a Series one slow item at a time; ``called`` is set once the first item is, and
    ``calls`` lists every item called."""
    registry, called, calls = NodeRegistry(), threading.Event(), []

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


async def test_a_run_handed_back_by_a_worker_shutting_down_reads_as_running_with_no_cause() -> None:
    registry, graph, called, _ = _slow_items()
    store = MemoryStore()
    worker = _worker(store, registry)

    async with worker.running():
        run_id = await _started(worker, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)

    run = await worker.read(run_id)
    view = RunView.of(run)
    assert (view.status, view.cause, run.row.owner, run.row.lease_until) == ("running", None, None, 0.0)


async def test_a_run_handed_back_at_shutdown_stores_no_run_end_and_the_worker_that_takes_it_repeats_no_call() -> None:
    registry, graph, called, calls = _slow_items()
    store = MemoryStore()
    first, second = _worker(store, registry), _worker(store, registry)

    async with first.running():
        run_id = await _started(first, graph, {"n": 40})
        await asyncio.to_thread(called.wait, 5)
    assert "run_end" not in [event["type"] for event in await store.stored_events(run_id)]

    await second.take_over()
    assert (await store.ended(run_id)).outputs == {"slow.result": list(range(40))}
    events = [event["type"] for event in await store.stored_events(run_id)]
    assert (events.count("run_start"), events.count("run_end"), events[-1]) == (2, 1, "run_end")
    assert sorted(calls) == list(range(40))


async def test_a_cancel_from_another_worker_that_lands_as_a_worker_shuts_down_keeps_the_run_cancelled() -> None:
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

    async with worker.running():
        run_id = await _started(worker, graph, inputs)
        asyncio.get_running_loop().call_later(0.05, gate.set)

    row = (await worker.read(run_id)).row
    assert store.cancelled and (row.status, row.owner, row.lease_until) == ("cancelled", None, None)
    assert [event["status"] for event in await store.stored_events(run_id) if event["type"] == "run_end"] == ["cancelled"]


@pytest.mark.parametrize("fails", ["the read of the expired runs", "the claim of a run", "a row that can't be read"])
async def test_a_sweep_that_raises_is_logged_and_the_next_one_takes_the_run(fails: str, caplog: pytest.LogCaptureFixture) -> None:
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
    left = StoredRun(
        id="left", graph=graph.model_dump_json(), caller='"anna"', run=registry.load(graph).with_inputs({"text": "a"}).to_json(),
        **StoredState().columns(), outcome=Outcome().model_dump_json(), owner=None, lease_until=0.0,
    )
    if fails == "a row that can't be read":
        await store.create(dataclasses.replace(left, id="broken", state="{"))
    await store.create(left)

    async with worker.running():
        ended = await store.ended("left")

    assert ended.outputs == {"shout.result": "A"}
    logged = [record.exc_info is not None for record in caplog.records if record.name == "conductor_providers.fastapi"]
    assert logged and all(logged)


async def test_a_renewal_that_raises_is_logged_and_the_next_one_keeps_the_lease_so_no_other_worker_takes_the_run(caplog: pytest.LogCaptureFixture) -> None:
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
    async with other.running():
        run_id = await _started(holder, graph, {"n": 40})
        ended = await store.ended(run_id)

    assert store.raised and ended.status == "complete"
    assert sorted(calls) == list(range(40)) and await _starts(store, run_id) == 1
    logged = [record.exc_info is not None for record in caplog.records if record.name == "conductor_providers.fastapi"]
    assert logged and all(logged)


@pytest.mark.parametrize("ends", ["complete", "waiting"])
async def test_a_run_that_stopped_running_holds_no_lease_so_no_sweep_takes_it_again(ends: str) -> None:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    @registry.node("approve", title="Approve", description="")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    store = MemoryStore()
    worker = _worker(store, registry)
    run_id = await _started(worker, make_graph(shout if ends == "complete" else approve), {"text": "a"})
    assert (await store.ended(run_id)).status == ends

    row = (await worker.read(run_id)).row
    assert (row.owner, row.lease_until) == (None, None) and await store.expired(time.time()) == []
    await worker.take_over()
    assert (await worker.read(run_id)).row == row and await _starts(store, run_id) == 1


async def test_a_run_whose_worker_died_is_taken_by_one_of_two_workers_and_finished_from_its_checkpoint() -> None:
    registry, graph, gates, calls = gated()
    for gate in gates.values():
        gate.set()
    run = registry.load(graph).run({"items.text": "a,b", "busy.text": "x", "approve.text": "y"})
    checkpoint = run.answer({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    one, other = _worker(store, registry), _worker(store, registry)
    await store.create(StoredRun(
        id="left", graph=graph.model_dump_json(), caller='"anna"', run=checkpoint.to_json(), **StoredState().columns(), outcome=Outcome().model_dump_json(), owner="a-dead-worker", lease_until=0.0,
    ))

    await asyncio.gather(one.take_over(), other.take_over())

    assert (await store.ended("left")).outputs == {"busy.result": "x", "hold.result": ["a", "b"]}
    assert calls == ["a", "b"] and await _starts(store, "left") == 1


async def test_a_worker_whose_run_another_took_stops_its_leg_and_writes_nothing_more() -> None:
    registry, graph, gates, calls = gated()
    inputs = {"items.text": "a,b", "busy.text": "x", "approve.text": "y"}
    store = MemoryStore()
    slow, taker = _worker(store, registry, lease=0.15), _worker(store, registry)

    async with slow.running():  # leaving it waits for the leg the slow worker still runs
        run_id = await _started(slow, graph, inputs)
        row = (await slow.read(run_id)).row
        assert await store.update(run_id, row.rev, {"lease_until": 0.0})  # as when its lease ran out unrenewed
        await taker.take_over()
        taken = await taker.read(run_id)
        await asyncio.sleep(0.2)  # the slow worker's renewals would have landed by now
        assert taken.row.owner == taker.id and await taker.read(run_id) == taken
        gates["first"].set()
        ended = await store.ended(run_id)

    assert ended.status == "waiting" and RunView.of(await taker.read(run_id)) == ended
    assert [event["type"] for event in await store.stored_events(run_id)].count("run_end") == 1
    for gate in gates.values():
        gate.set()
