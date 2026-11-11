"""``run_router`` served in this process over a store in memory, for what the page's own app doesn't set up: a host's
own ``may_see`` and a registry per caller."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from typing import Annotated, Any, ClassVar, TypedDict

import pytest

httpx = pytest.importorskip("httpx")

import polars as pl  # noqa: E402
from conductor import (  # noqa: E402
    Ask,
    FromRun,
    GraphRecord,
    NodeError,
    NodeRecord,
    NodeRegistry,
    Series,
    Table,
    make_graph,
)
from conductor_providers.fastapi import RunChanges, StoredRun, run_router  # noqa: E402
from fastapi import FastAPI, Header  # noqa: E402
from pydantic import GetCoreSchemaHandler  # noqa: E402
from pydantic_core import core_schema  # noqa: E402


class MemoryStore:
    """A ``RunStore`` in a dict."""

    def __init__(self) -> None:
        self.rows: dict[str, StoredRun] = {}
        self.written: dict[str, list[str]] = {}

    async def create(self, run: StoredRun) -> None:
        self.rows[run.id], self.written[run.id] = run, []

    async def get(self, run_id: str) -> StoredRun | None:
        return self.rows.get(run_id)

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        row = self.rows.get(run_id)
        if row is None or row.rev != rev:
            return False
        self.rows[run_id] = dataclasses.replace(row, **changes, rev=rev + 1)  # type: ignore[arg-type]
        return True

    async def expired(self, now: float) -> list[StoredRun]:
        return [row for row in self.rows.values() if row.lease_until is not None and row.lease_until < now]

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        if self.rows[run_id].owner != owner:
            return False
        self.written[run_id].append(body)
        return True

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        return [(seq, body) for seq, body in enumerate(self.written[run_id]) if seq > after]


def _caller(x_user: Annotated[str, Header()] = "anna") -> str:
    return x_user


async def _until_ended(client: Any, run_id: str, who: str) -> dict[str, Any]:
    for _ in range(500):
        row: dict[str, Any] = (await client.get(f"/runs/{run_id}", headers={"x-user": who})).json()
        if row["status"] != "running":
            return row
        await asyncio.sleep(0.01)
    raise AssertionError("the run never ended")


def test_again_by_another_caller_who_may_see_the_run_is_checked_on_that_callers_own_nodes() -> None:
    full, bare = NodeRegistry(), NodeRegistry()

    @full.node("double", title="Double", description="")
    def double(x: int) -> int:
        return 2 * x

    graph = GraphRecord(nodes=[NodeRecord(id="d", node="double")]).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(
        store=MemoryStore(), registry=lambda who: full if who == "anna" else bare, caller=_caller, may_see=lambda caller, started_by: True,
    ))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["status"] == "complete"
            again = await client.post(f"/runs/{run_id}/again", json={"x": 2}, headers={"x-user": "bo"})
            assert again.status_code == 422
            assert [problem["type"] for problem in again.json()["detail"]] == ["unknown_node"]
            mine = await client.post(f"/runs/{run_id}/again", json={"x": 2})
            assert mine.status_code == 202
            assert (await _until_ended(client, mine.json()["id"], "anna"))["outputs"] == {"d.result": 4}

    asyncio.run(scenario())


def _maybe_caller(x_user: Annotated[str | None, Header()] = None) -> str | None:
    return x_user


def test_again_by_an_anonymous_caller_is_checked_on_the_nodes_of_no_caller_not_the_starters() -> None:
    full, bare = NodeRegistry(), NodeRegistry()

    @full.node("double", title="Double", description="")
    def double(x: int) -> int:
        return 2 * x

    graph = GraphRecord(nodes=[NodeRecord(id="d", node="double")]).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(
        store=MemoryStore(), registry=lambda who: full if who == "anna" else bare, caller=_maybe_caller, may_see=lambda caller, started_by: True,
    ))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}}, headers={"x-user": "anna"})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["status"] == "complete"
            anonymous = await client.post(f"/runs/{run_id}/again", json={"x": 2})
            assert anonymous.status_code == 422
            assert [problem["type"] for problem in anonymous.json()["detail"]] == ["unknown_node"]

    asyncio.run(scenario())


def test_an_output_that_isnt_finite_reads_as_its_text() -> None:
    registry = NodeRegistry()

    @registry.node("edges", title="Edges", description="")
    def edges(x: int) -> list[float]:
        return [float("inf"), float("-inf"), float("nan"), x]

    graph = GraphRecord(nodes=[NodeRecord(id="e", node="edges")]).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["outputs"] == {"e.result": ["Infinity", "-Infinity", "NaN", 1.0]}

    asyncio.run(scenario())


def test_a_run_stopped_at_its_ceiling_says_why_and_is_resumed_with_no_body_or_a_higher_ceiling() -> None:
    registry = NodeRegistry()

    @registry.node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("up", title="Up", description="")
    def up(text: str) -> str:
        return text.upper()

    graph = GraphRecord(nodes=[NodeRecord(id="s", node="split"), NodeRecord(id="u", node="up")], edges=[("s.result", "u.text")]).model_dump(mode="json")
    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, max_items=1))

    def last_end(run_id: str) -> dict[str, Any]:
        ends: list[dict[str, Any]] = [json.loads(body) for body in store.written[run_id] if json.loads(body)["type"] == "run_end"]
        return ends[-1]

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": "a,b"}})).json()["id"]
            stopped = await _until_ended(client, run_id, "anna")
            assert stopped["status"] == "stopped"
            assert stopped["cause"]["type"] == last_end(run_id)["cause"]["type"] == "item_ceiling"
            assert stopped["cause"]["ctx"] == {"series": "s.result", "items": 2, "max_items": 1}
            below_one = await client.post(f"/runs/{run_id}/resume", json={"max_items": 0})
            assert below_one.status_code == 422
            assert [(e["type"], e["loc"]) for e in below_one.json()["detail"]] == [("greater_than", ["body", "max_items"])]
            assert (await client.get(f"/runs/{run_id}")).json()["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume")).status_code == 202
            assert (await _until_ended(client, run_id, "anna"))["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume", json={"max_items": 2})).status_code == 202
            completed = await _until_ended(client, run_id, "anna")
            assert completed["outputs"] == {"u.result": ["A", "B"]}
            assert completed["cause"] is None and last_end(run_id)["cause"] is None

    asyncio.run(scenario())



def test_a_ceiling_raised_by_a_resume_holds_for_the_legs_after_it() -> None:
    registry = NodeRegistry()

    @registry.node("split", title="Split", description="")
    def split(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("ask", title="Ask", description="")
    def ask(x: int) -> int:
        raise Ask("ok?")

    graph = make_graph(split, ask, edges=[("split.result", "ask.x")], values={"split.n": 5}).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller, max_items=3))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume", json={"max_items": 10})).status_code == 202
            waiting = (await _until_ended(client, run_id, "anna"))["waiting"]
            assert len(waiting) == 5
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting[0]["id"]: 7})).status_code == 202
            after = await _until_ended(client, run_id, "anna")
            assert (after["status"], len(after["waiting"])) == ("waiting", 4)

    asyncio.run(scenario())


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


def test_an_output_only_its_own_type_can_write_completes_over_http_as_it_does_in_the_engine() -> None:
    registry = NodeRegistry()

    @registry.node("price", title="Price", description="")
    def price(cents: int) -> Cents:
        return Cents(cents)

    @registry.node("prices", title="Prices", description="")
    def prices(cents: int) -> Series[Cents]:
        return [Cents(cents), Cents(cents + 1)]

    graph = GraphRecord(nodes=[NodeRecord(id="price", node="price"), NodeRecord(id="prices", node="prices")]).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"price.cents": 5, "prices.cents": 7}})).json()["id"]
            ended = await _until_ended(client, run_id, "anna")
            assert (ended["status"], ended["outputs"], ended["cause"]) == ("complete", {"price.result": 5, "prices.result": [7, 8]}, None)

    asyncio.run(scenario())


class Part(TypedDict):
    text: str


def test_a_table_made_for_each_item_reads_as_its_columns_per_item() -> None:
    registry = NodeRegistry()

    @registry.node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("as-table", title="As table", description="")
    def as_table(text: str) -> Table[Part]:
        return pl.DataFrame({"text": [text]})

    graph = make_graph(split, as_table).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": "a,b"}})).json()["id"]
            ended = await _until_ended(client, run_id, "anna")
            columns = [{"columns": [{"name": "text", "type": "str", "values": [text]}]} for text in ("a", "b")]
            assert (ended["status"], ended["outputs"]) == ("complete", {"as-table.result": columns})

    asyncio.run(scenario())


def _gated() -> tuple[NodeRegistry, GraphRecord, dict[str, threading.Event], list[str]]:
    """A graph whose first leg waits on ``first`` beside a question, and whose leg after the answer waits on ``second``
    on its first item of two; ``held`` is set once that item is called."""
    registry, gates, calls = NodeRegistry(), {name: threading.Event() for name in ("first", "second", "held")}, []

    @registry.node("items", title="Items", description="")
    def items(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("busy", title="Busy", description="")
    def busy(text: str) -> str:
        gates["first"].wait(5)
        return text

    @registry.node("approve", title="Approve", description="")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    @registry.node("hold", title="Hold", description="")
    def hold(item: str, ok: bool) -> str:
        calls.append(item)
        gates["held"].set()
        gates["second"].wait(5)
        return item

    graph = make_graph(items, busy, approve, hold, edges=[("items.result", "hold.item"), ("approve.result", "hold.ok")])
    return registry, graph, gates, calls


def test_a_cancel_reaches_the_leg_an_answer_queued_during_the_last_one_started() -> None:
    registry, graph, gates, calls = _gated()
    for gate in ("first", "second"):
        gates[gate].set()
    (waiting,) = registry.load(graph).run({"items.text": "a,b", "busy.text": "x", "approve.text": "y"}).waiting
    gates["first"].clear()
    gates["second"].clear()
    calls.clear()
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            inputs = {"items.text": "a,b", "busy.text": "x", "approve.text": "y"}
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting.id: True})).json() == {"answers": "queued"}
            gates["first"].set()
            await asyncio.to_thread(gates["held"].wait, 5)
            assert (await client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": True}
            gates["second"].set()
            assert (await _until_ended(client, run_id, "anna"))["status"] == "cancelled" and calls == ["a"]

    asyncio.run(scenario())


def test_a_cancel_sent_while_a_dead_worker_held_the_run_is_kept_when_another_takes_it_over() -> None:
    registry, graph, gates, calls = _gated()
    for gate in gates.values():
        gate.set()
    run = registry.load(graph).run({"items.text": "a,b", "busy.text": "x", "approve.text": "y"})
    answered = run.answer({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, lease=0.6))

    async def scenario() -> None:
        await store.create(StoredRun(
            id="taken", graph=graph.model_dump_json(), caller='"anna"', run=answered.to_json(), status="running", owner="a-dead-worker", lease_until=time.time() + 0.3,
        ))
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.post("/runs/taken/cancel")).json() == {"cancelled": True}
            assert (await _until_ended(client, "taken", "anna"))["status"] == "cancelled" and calls == []
            assert (await client.post("/runs/taken/resume", json={})).status_code == 202
            assert (await _until_ended(client, "taken", "anna"))["status"] == "complete" and calls == ["a", "b"]

    asyncio.run(scenario())


def _asking(items: bool = False) -> tuple[NodeRegistry, GraphRecord, threading.Event, dict[str, Any], str]:
    """A graph whose leg waits on ``gate`` beside a question, its inputs, and the waiting id the question gets; with
    ``items``, beside a Series of three items read one at a time too, for a ceiling of 1 to stop."""
    registry, gate = NodeRegistry(), threading.Event()

    @registry.node("busy", title="Busy", description="")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    @registry.node("approve", title="Approve", description="")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    @registry.node("after", title="After", description="")
    def after(ok: bool) -> bool:
        return ok

    @registry.node("three", title="Three", description="")
    def three(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("each", title="Each", description="")
    def each(i: int) -> int:
        return i

    nodes: list[Any] = [busy, approve, after, three, each] if items else [busy, approve, after]
    graph = make_graph(*nodes, edges=[("approve.result", "after.ok"), *([("three.result", "each.i")] if items else [])])
    inputs: dict[str, Any] = {"busy.text": "x", "approve.text": "y", **({"three.n": 3} if items else {})}
    gate.set()
    (waiting,) = registry.load(graph).run(inputs).waiting
    gate.clear()
    return registry, graph, gate, inputs, waiting.id


async def _until_status(store: MemoryStore, run_id: str, *statuses: str) -> StoredRun:
    for _ in range(500):
        if store.rows[run_id].status in statuses:
            return store.rows[run_id]
        await asyncio.sleep(0.01)
    raise AssertionError(f"the run is {store.rows[run_id].status}")


def test_an_answer_queued_during_a_leg_starts_the_next_leg_however_long_its_end_takes_to_write() -> None:
    registry, graph, gate, inputs, waiting_id = _asking()

    class SlowEnd(MemoryStore):
        """Writes the end of a leg, the write that clears the queued answers, after three of the lease's renewals."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.get("pending") == "{}":
                await asyncio.sleep(0.3)
            return await super().update(run_id, rev, changes)

    store = SlowEnd()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, lease=0.3))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
            gate.set()
            ended = await _until_status(store, run_id, "complete", "waiting")
            assert (ended.status, ended.pending, json.loads(ended.outputs or "null")) == ("complete", "{}", {"busy.result": "x", "after.result": True})

    asyncio.run(scenario())


def test_a_leg_whose_run_takes_longer_than_the_lease_to_write_keeps_it_and_writes_its_end_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from conductor import Run

    registry, graph, gate, inputs, waiting_id = _asking()
    gate.set()
    made, slow = [], threading.Event()
    to_json = Run.to_json

    def slow_to_json(self: Run) -> str:
        if slow.is_set() and threading.current_thread() is not threading.main_thread():
            made.append(self.status)
            time.sleep(0.5)  # a large run, longer than the lease
        return to_json(self)

    monkeypatch.setattr(Run, "to_json", slow_to_json)

    class EndIsSlow(MemoryStore):
        async def add_event(self, run_id: str, owner: str, body: str) -> bool:
            if '"run_end"' in body:
                slow.set()
            return await super().add_event(run_id, owner, body)

    store = EndIsSlow()
    apps = [FastAPI(), FastAPI()]
    for app in apps:
        app.include_router(run_router(store=store, registry=registry, caller=_caller, lease=0.3))

    async def scenario() -> None:
        async with apps[0].router.lifespan_context(apps[0]), apps[1].router.lifespan_context(apps[1]):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[0]), base_url="http://t") as client:
                run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
                ended = await _until_status(store, run_id, "waiting")
                await asyncio.sleep(0.4)  # a second worker that took the run would be running it again by now
        assert ended.waiting == store.rows[run_id].waiting and json.loads(ended.waiting)[0]["id"] == waiting_id
        assert [json.loads(body)["type"] for body in store.written[run_id]].count("run_start") == 1
        assert made == ["waiting"]

    asyncio.run(scenario())


class _FailingEvent(MemoryStore):
    """Fails to add the first event after ``armed`` is set, so the leg raises there."""

    armed = failed = False

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        if self.armed and not self.failed:
            self.failed = True
            raise ConnectionError("the store went away")
        return await super().add_event(run_id, owner, body)


def test_an_answer_queued_during_a_leg_that_stops_at_its_ceiling_is_taken_and_starts_the_next_leg() -> None:
    registry, graph, gate, inputs, waiting_id = _asking(items=True)
    runs = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=runs, registry=registry, caller=_caller, max_items=1))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
            gate.set()
            for _ in range(500):
                row = runs.rows[run_id]
                if (row.status, row.pending, row.waiting) == ("stopped", "{}", "[]"):  # the next leg meets the same ceiling
                    break
                await asyncio.sleep(0.01)
            assert (row.status, row.pending, row.waiting, row.refused) == ("stopped", "{}", "[]", "{}")
            assert [json.loads(body)["type"] for body in runs.written[run_id]].count("run_start") == 2

    asyncio.run(scenario())


def test_an_answer_queued_for_a_question_a_timed_out_leg_never_asked_is_refused_by_the_runs_own_code() -> None:
    registry, graph, gate, inputs, waiting_id = _asking(items=True)
    runs = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=runs, registry=registry, caller=_caller, leg_timeout=0.2))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
            await asyncio.sleep(0.3)  # past the leg's timeout; the call it waits for then finishes
            gate.set()
            await _until_status(runs, run_id, "timed_out")
            row = (await client.get(f"/runs/{run_id}")).json()
            assert [(problem["type"], problem["ctx"]["status"]) for problem in row["refused"][waiting_id]] == [("not_waiting", "timed_out")]
            await asyncio.sleep(0.1)  # a leg started on the answer would be running by now
            assert (runs.rows[run_id].status, runs.rows[run_id].pending) == ("timed_out", "{}")
            assert [json.loads(body)["type"] for body in runs.written[run_id]].count("run_start") == 1

    asyncio.run(scenario())


def test_an_answer_queued_during_a_leg_that_ends_cancelled_is_refused_as_run_cancelled_and_starts_nothing() -> None:
    registry, graph, gate, inputs, waiting_id = _asking()
    runs = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=runs, registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
            assert (await client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": True}
            gate.set()
            await _until_status(runs, run_id, "cancelled")
            row = (await client.get(f"/runs/{run_id}")).json()
            assert [problem["type"] for problem in row["refused"][waiting_id]] == ["run_cancelled"]
            await asyncio.sleep(0.1)  # a leg started on the answer would be running by now
            assert (runs.rows[run_id].status, runs.rows[run_id].pending) == ("cancelled", "{}")
            assert [json.loads(body)["type"] for body in runs.written[run_id]].count("run_start") == 1

    asyncio.run(scenario())


def test_an_answer_queued_during_a_leg_that_raised_stays_queued_and_is_taken_after_a_resume() -> None:
    registry, graph, gate, inputs, waiting_id = _asking(items=True)
    runs = _FailingEvent()
    app = FastAPI()
    app.include_router(run_router(store=runs, registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
            runs.armed = True
            gate.set()
            await _until_status(runs, run_id, "failed")
            assert json.loads(runs.rows[run_id].pending) == {waiting_id: True}
            assert (await client.post(f"/runs/{run_id}/resume")).status_code == 202
            await _until_status(runs, run_id, "complete")
            assert (runs.rows[run_id].pending, runs.rows[run_id].refused) == ("{}", "{}")

    asyncio.run(scenario())


def test_a_question_that_waits_in_a_failed_run_is_answered_without_a_resume() -> None:
    registry = NodeRegistry()

    @registry.node("many", title="Many", description="")
    def many(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("approve", title="Approve", description="")
    def approve(x: int) -> bool:
        if x == 2:
            raise NodeError("item 2 fails")
        raise Ask(f"approve {x}?")

    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": make_graph(many, approve).model_dump(mode="json"), "inputs": {"n": 3}})).json()["id"]
            failed = await _until_ended(client, run_id, "anna")
            assert (failed["status"], len(failed["waiting"])) == ("failed", 2)
            answered = await client.post(f"/runs/{run_id}/answers", json={failed["waiting"][0]["id"]: True})
            assert (answered.status_code, answered.json()) == (202, {"answers": "started"})
            after = await _until_ended(client, run_id, "anna")
            assert (after["status"], [unit["id"] for unit in after["waiting"]]) == ("failed", [failed["waiting"][1]["id"]])

    asyncio.run(scenario())


def test_a_cancel_from_another_worker_that_lands_as_a_worker_shuts_down_keeps_the_run_cancelled() -> None:
    registry, graph, gate, inputs, _ = _asking()

    class CancelAtHandBack(MemoryStore):
        """Takes a cancel, as another worker's ``POST /cancel`` writes it, just before the write handing the run back."""

        cancelled = False

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if not self.cancelled and changes.get("status") == "running" and "waiting" in changes:
                self.cancelled = True
                assert await super().update(run_id, self.rows[run_id].rev, {"cancel": True})
            return await super().update(run_id, rev, changes)

    store = CancelAtHandBack()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller))

    async def scenario() -> None:
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            shutting_down = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(0.05)
            gate.set()
            await shutting_down
        row = store.rows[run_id]
        assert store.cancelled and (row.status, row.owner) == ("cancelled", None)

    asyncio.run(scenario())


def test_a_leg_whose_event_the_store_fails_to_take_calls_no_node_once_the_run_reads_failed() -> None:
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
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry))
    graph = make_graph(first, slow).model_dump(mode="json")

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"n": 100}})).json()["id"]
            await _until_status(store, run_id, "failed")
            at_the_end = len(started)
            await asyncio.sleep(0.3)
            assert len(started) == at_the_end < 100

    asyncio.run(scenario())


@dataclasses.dataclass(frozen=True)
class _Member:
    """A caller as FastAPI's class dependency: ``Depends(_Member)`` reads ``x_user`` from a header."""

    x_user: Annotated[str, Header()] = "anna"


def test_a_class_as_the_caller_is_kept_as_that_class_so_its_starter_sees_the_run() -> None:
    registry, seen = NodeRegistry(), []

    @registry.node("whoami", title="Whoami", description="")
    def whoami(text: str, who: FromRun[_Member]) -> str:
        seen.append(who)
        return who.x_user

    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_Member, from_run=lambda caller: {_Member: caller}))
    graph = make_graph(whoami).model_dump(mode="json")

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": "hi"}}, headers={"x-user": "bo"})).json()["id"]
            row = await _until_ended(client, run_id, "bo")
            assert (row["status"], row["outputs"]) == ("complete", {"whoami.result": "bo"})
            assert seen == [_Member("bo")]

    asyncio.run(scenario())


def test_a_caller_with_no_return_annotation_is_refused_when_the_router_is_made() -> None:
    with pytest.raises(TypeError, match="return annotation"):
        run_router(store=MemoryStore(), registry=NodeRegistry(), caller=lambda: "anna")


def test_a_run_handed_back_by_a_worker_shutting_down_reads_as_running_with_no_cause() -> None:
    registry = NodeRegistry()

    @registry.node("many", title="Many", description="Gives several items.")
    def many(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("slow", title="Slow", description="Waits before it answers.")
    def slow(x: int) -> int:
        time.sleep(0.05)
        return x

    graph = make_graph(registry["many"], registry["slow"], edges=[("many.result", "slow.x")]).model_dump(mode="json")
    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"n": 40}})).json()["id"]
            await asyncio.sleep(0.2)
        row = store.rows[run_id]
        assert (row.status, row.owner, row.cause) == ("running", None, None)

    asyncio.run(scenario())


def test_a_run_handed_back_at_shutdown_stores_no_run_end_until_the_worker_that_takes_it_ends_it() -> None:
    registry = NodeRegistry()

    @registry.node("many", title="Many", description="Gives several items.")
    def many(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("slow", title="Slow", description="Waits before it answers.")
    def slow(x: int) -> int:
        time.sleep(0.02)
        return x

    graph = make_graph(registry["many"], registry["slow"], edges=[("many.result", "slow.x")]).model_dump(mode="json")
    store = MemoryStore()
    first, second = FastAPI(), FastAPI()
    for app in (first, second):
        app.include_router(run_router(store=store, registry=registry, caller=_caller, lease=0.3))

    def stored_events(run_id: str) -> list[str]:
        return [json.loads(body)["type"] for body in store.written[run_id]]

    async def scenario() -> None:
        async with first.router.lifespan_context(first), httpx.AsyncClient(transport=httpx.ASGITransport(app=first), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"n": 40}})).json()["id"]
            await asyncio.sleep(0.2)
        assert store.rows[run_id].status == "running" and "run_end" not in stored_events(run_id)
        async with second.router.lifespan_context(second):
            await _until_status(store, run_id, "complete")
        events = stored_events(run_id)
        assert (events.count("run_start"), events.count("run_end"), events[-1]) == (2, 1, "run_end")

    asyncio.run(scenario())


def test_a_leg_that_raised_outside_a_node_reads_failed_with_a_cause_that_its_run_end_carries_too() -> None:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    def no_values(caller: str) -> dict[type, Any]:
        raise ConnectionError("the secrets service is down")

    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, from_run=no_values))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": make_graph(shout).model_dump(mode="json"), "inputs": {"text": "a"}})).json()["id"]
            failed = await _until_ended(client, run_id, "anna")
            assert (failed["status"], failed["cause"]["type"], failed["cause"]["loc"]) == ("failed", "leg_failed", [])
            assert [json.loads(body) for body in store.written[run_id]] == [{"type": "run_end", "status": "failed", "cause": failed["cause"]}]

    asyncio.run(scenario())


def test_every_refusal_of_the_router_is_a_list_of_problems_with_a_code_of_its_own() -> None:
    from conductor import Problem
    from conductor_providers.fastapi import ROUTER_PROBLEM_CODES

    registry, graph, gate, inputs, waiting_id = _asking()

    class ClaimLost(MemoryStore):
        """Loses every claim of a run that waits, as when another worker took it first."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            return not ("owner" in changes and self.rows[run_id].status == "waiting") and await super().update(run_id, rev, changes)

    store = ClaimLost()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller))

    def refusal(response: Any) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
        problems = [Problem.model_validate(problem) for problem in response.json()["detail"]]
        assert all(problem.type in ROUTER_PROBLEM_CODES and problem.msg for problem in problems)
        return response.status_code, [(problem.type, dict(problem.ctx)) for problem in problems]

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            assert refusal(await client.get("/runs/nonesuch")) == (404, [("no_such_run", {})])
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert refusal(await client.get(f"/runs/{run_id}", headers={"x-user": "bo"})) == (404, [("no_such_run", {})])
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
            assert refusal(await client.post(f"/runs/{run_id}/answers", json={waiting_id: False})) == (409, [("already_answered", {"waiting_id": waiting_id})])
            assert refusal(await client.post(f"/runs/{run_id}/resume")) == (409, [("run_not_resumable", {"status": "running"})])
            gate.set()
            await _until_status(store, run_id, "complete")
            assert refusal(await client.post(f"/runs/{run_id}/answers", json={})) == (409, [("run_complete", {})])
            assert refusal(await client.post(f"/runs/{run_id}/resume")) == (409, [("run_not_resumable", {"status": "complete"})])
            waits = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            await _until_status(store, waits, "waiting")
            assert refusal(await client.post(f"/runs/{waits}/answers", json={waiting_id: True})) == (409, [("run_changed", {})])

    asyncio.run(scenario())
    paths = app.openapi()["paths"]
    refused = {"application/json": {"schema": {"$ref": "#/components/schemas/Refused"}}}
    assert [paths["/runs/{run_id}/answers"]["post"]["responses"][code]["content"] for code in ("404", "409")] == [refused, refused]
    assert paths["/runs/{run_id}"]["get"]["responses"]["404"]["content"] == refused


@pytest.mark.parametrize("ends", ["complete", "waiting"])
def test_a_run_that_stopped_running_holds_no_lease_so_no_sweep_takes_it_again(ends: str) -> None:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    @registry.node("approve", title="Approve", description="")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, lease=0.15))
    graph = make_graph(shout if ends == "complete" else approve).model_dump(mode="json")

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": "a"}})).json()["id"]
            ended = await _until_status(store, run_id, ends)
            await asyncio.sleep(0.3)  # several sweeps
            assert [json.loads(body)["type"] for body in store.written[run_id]].count("run_start") == 1
            assert (ended.owner, ended.lease_until) == (None, None) and await store.expired(time.time()) == []

    asyncio.run(scenario())


def test_a_follower_reads_the_row_only_to_learn_whether_a_run_end_was_the_last() -> None:
    registry, graph, gate, inputs, _ = _asking()

    class CountsReads(MemoryStore):
        reads = 0

        async def get(self, run_id: str) -> StoredRun | None:
            self.reads += 1
            return await super().get(run_id)

    store = CountsReads()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, poll=0.01))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            following = asyncio.create_task(client.get(f"/runs/{run_id}/events"))
            await asyncio.sleep(0.1)  # the follower has the leg's first events by now
            before = store.reads
            await asyncio.sleep(0.3)  # thirty polls that find no event
            during = store.reads - before
            gate.set()
            frames = [json.loads(line.removeprefix("data: ")) for line in (await following).text.splitlines() if line.startswith("data: ")]
            assert (during, frames[-1]["type"], frames[-1]["status"]) == (0, "run_end", "waiting")
            again = (await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(frames) - 1)})).text
            assert "data: " not in again  # a follower that has every event of a run that waits is told so at once

    asyncio.run(scenario())


def test_the_schema_names_each_operation_and_body_as_a_generated_client_reads_them() -> None:
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=NodeRegistry(), caller=_caller, published=lambda name: GraphRecord()))
    schema = app.openapi()
    operations = {(verb, path): operation for path, verbs in schema["paths"].items() for verb, operation in verbs.items()}

    assert {key: operation["summary"] for key, operation in operations.items()} == {
        ("get", "/nodes"): "List Nodes",
        ("post", "/load"): "Load Graph",
        ("post", "/runs"): "Start Run",
        ("post", "/graphs/{name}/runs"): "Start Published",
        ("get", "/runs/{run_id}"): "Read Run",
        ("get", "/runs/{run_id}/events"): "Follow Run",
        ("post", "/runs/{run_id}/answers"): "Answer Run",
        ("post", "/runs/{run_id}/resume"): "Resume Run",
        ("post", "/runs/{run_id}/again"): "Run Again",
        ("post", "/runs/{run_id}/cancel"): "Cancel Run",
    }
    assert all(operation.get("description") for operation in operations.values())
    assert "Resume" in schema["components"]["schemas"] and not [name for name in schema["components"]["schemas"] if name.startswith("Body_")]
    assert operations["post", "/runs/{run_id}/resume"]["requestBody"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/Resume", "default": {}}
    for key in [("post", "/graphs/{name}/runs"), ("post", "/runs/{run_id}/answers"), ("post", "/runs/{run_id}/again")]:
        assert operations[key]["requestBody"]["content"]["application/json"]["schema"]["description"], key


def test_the_package_exports_what_a_host_writes_and_calls() -> None:
    import conductor_providers.fastapi as provider

    assert sorted(provider.__all__) == ["ROUTER_PROBLEM_CODES", "RunChanges", "RunStore", "StoredRun", "run_router"]


def test_a_follower_that_sent_a_run_end_while_the_row_still_reads_running_keeps_asking_until_it_changes() -> None:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    class RowLags(MemoryStore):
        """Reads a run that has ended as still running a few times, as a reader between the ``run_end`` and the row's write does."""

        lagged = 0

        async def get(self, run_id: str) -> StoredRun | None:
            row = await super().get(run_id)
            if row is not None and row.status == "complete" and self.lagged < 5:
                self.lagged += 1
                return dataclasses.replace(row, status="running")
            return row

    store = RowLags()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, poll=0.01))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": make_graph(shout).model_dump(mode="json"), "inputs": {"text": "a"}})).json()["id"]
            await _until_status(store, run_id, "complete")
            followed = await asyncio.wait_for(client.get(f"/runs/{run_id}/events"), 5)
            frames = [json.loads(line.removeprefix("data: ")) for line in followed.text.splitlines() if line.startswith("data: ")]
            assert [frame["type"] for frame in frames].count("run_end") == 1 and frames[-1]["type"] == "run_end"
            assert store.lagged == 5

    asyncio.run(scenario())


@pytest.mark.parametrize("fails", ["the write of the row", "the run as JSON"])
def test_a_leg_whose_end_fails_after_its_run_end_is_stored_stays_running_and_is_gone_on_with(fails: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from conductor import Run

    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    class EndFailsOnce(MemoryStore):
        """Raises once after a leg's ``run_end`` is stored: in the write that ends the leg, or as the run is written as JSON."""

        end_stored = raised = False

        async def add_event(self, run_id: str, owner: str, body: str) -> bool:
            self.end_stored = self.end_stored or '"run_end"' in body
            return await super().add_event(run_id, owner, body)

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if fails == "the write of the row" and changes.get("status") == "complete" and not self.raised:
                self.raised = True
                raise ConnectionError("the store went away for one statement")
            return await super().update(run_id, rev, changes)

    store = EndFailsOnce()
    to_json = Run.to_json

    def failing_to_json(self: Run) -> str:
        if fails == "the run as JSON" and store.end_stored and not store.raised:
            store.raised = True
            raise RuntimeError("a value the run can't write")
        return to_json(self)

    monkeypatch.setattr(Run, "to_json", failing_to_json)
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, lease=0.15))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": make_graph(shout).model_dump(mode="json"), "inputs": {"text": "a"}})).json()["id"]
            ended = await _until_status(store, run_id, "complete", "failed")
            ends = [json.loads(body)["status"] for body in store.written[run_id] if '"run_end"' in body]
            assert store.raised and (ended.status, ended.cause, ends) == ("complete", None, ["complete", "complete"])  # one for each leg

    asyncio.run(scenario())


def test_the_problems_page_quotes_the_sentence_of_each_of_the_routers_codes() -> None:
    import pathlib
    import re

    from conductor_providers.fastapi.router import _ROUTER_PROBLEMS

    page = (pathlib.Path(__file__).parents[2] / "docs" / "problems.md").read_text()
    section = page[page.index("## The runs router"):]
    parts = re.split(r"^### `([a-z_]+)`$", section, flags=re.M)
    quoted = {code: body.split("\n\n")[1] for code, body in zip(parts[1::2], parts[2::2])}
    assert quoted == {code: f"> {sentence}" for code, sentence in _ROUTER_PROBLEMS.items()}
