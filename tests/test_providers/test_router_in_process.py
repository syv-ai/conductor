"""``run_router`` served in this process over a store in memory, for what the page's own app doesn't set up: a host's
own ``may_see`` and a registry per caller."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from typing import Annotated, Any, TypedDict

import pytest

httpx = pytest.importorskip("httpx")

import polars as pl  # noqa: E402
from conductor import Ask, FromRun, GraphRecord, NodeRecord, NodeRegistry, Series, Table, make_graph  # noqa: E402
from conductor._internal.errors import PROBLEM_CATALOGUE  # noqa: E402
from conductor_providers.fastapi import RunChanges, StoredRun, run_router  # noqa: E402
from fastapi import FastAPI, Header  # noqa: E402


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
        return [row for row in self.rows.values() if row.status == "running" and row.lease_until < now]

    async def add_event(self, run_id: str, owner: str, body: str) -> int | None:
        if self.rows[run_id].owner != owner:
            return None
        self.written[run_id].append(body)
        return len(self.written[run_id]) - 1

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

    @registry.node("up", title="Up", description="")
    def up(text: str) -> str:
        return text.upper()

    graph = GraphRecord(nodes=[NodeRecord(id="u", node="up")]).model_dump(mode="json")
    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, max_rows=1))

    def last_end(run_id: str) -> dict[str, Any]:
        ends: list[dict[str, Any]] = [json.loads(body) for body in store.written[run_id] if json.loads(body)["type"] == "run_end"]
        return ends[-1]

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": ["a", "b"]}})).json()["id"]
            stopped = await _until_ended(client, run_id, "anna")
            assert stopped["status"] == "stopped"
            assert stopped["cause"]["type"] == last_end(run_id)["cause"]["type"] == "row_ceiling"
            assert stopped["cause"]["ctx"]["max_rows"] == 1 and "rows" in stopped["cause"]["ctx"]
            below_one = await client.post(f"/runs/{run_id}/resume", json={"max_rows": 0})
            assert below_one.status_code == 422
            assert [(e["type"], e["loc"]) for e in below_one.json()["detail"]] == [("greater_than", ["body", "max_rows"])]
            assert (await client.get(f"/runs/{run_id}")).json()["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume")).status_code == 202
            assert (await _until_ended(client, run_id, "anna"))["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume", json={"max_rows": 2})).status_code == 202
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
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller, max_rows=3))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume", json={"max_rows": 10})).status_code == 202
            waiting = (await _until_ended(client, run_id, "anna"))["waiting"]
            assert len(waiting) == 5
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting[0]["id"]: 7})).status_code == 202
            after = await _until_ended(client, run_id, "anna")
            assert (after["status"], len(after["waiting"])) == ("waiting", 4)

    asyncio.run(scenario())


class Part(TypedDict):
    text: str


def test_a_table_made_on_each_row_reads_as_a_list_of_records_per_row() -> None:
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
            assert (ended["status"], ended["outputs"]) == ("complete", {"as-table.result": [[{"text": "a"}], [{"text": "b"}]]})

    asyncio.run(scenario())


def _gated() -> tuple[NodeRegistry, GraphRecord, dict[str, threading.Event], list[str]]:
    """A graph whose first leg waits on ``first`` beside a question, and whose leg after the answer waits on ``second``
    on its first row of two; ``held`` is set once that row is called."""
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


def _asking(rows: bool = False) -> tuple[NodeRegistry, GraphRecord, threading.Event, dict[str, Any], str]:
    """A graph whose leg waits on ``gate`` beside a question, its inputs, and the waiting id the question gets; with
    ``rows``, beside a table of three rows too, for a ceiling of 1 to stop."""
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

    @registry.node("items", title="Items", description="")
    def items(n: int) -> Series[int]:
        return list(range(n))

    nodes: list[Any] = [busy, approve, after, items] if rows else [busy, approve, after]
    graph = make_graph(*nodes, edges=[("approve.result", "after.ok")])
    inputs: dict[str, Any] = {"busy.text": "x", "approve.text": "y", **({"items.n": 3} if rows else {})}
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
        async def add_event(self, run_id: str, owner: str, body: str) -> int | None:
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


class _FailingEnd(MemoryStore):
    """Fails to add a leg's first ``run_end``, so the leg raises there."""

    failed = False

    async def add_event(self, run_id: str, owner: str, body: str) -> int | None:
        if not self.failed and '"run_end"' in body:
            self.failed = True
            raise ConnectionError("the store went away")
        return await super().add_event(run_id, owner, body)


@pytest.mark.parametrize(("ended", "store", "settings"), [
    ("stopped", MemoryStore, {"max_rows": 1}),
    ("timed_out", MemoryStore, {"leg_timeout": 0.2}),
    ("failed", _FailingEnd, {}),
])
def test_an_answer_queued_during_a_leg_that_ends_needing_a_resume_is_refused_and_starts_nothing(
    ended: str, store: type[MemoryStore], settings: dict[str, Any],
) -> None:
    registry, graph, gate, inputs, waiting_id = _asking(rows=True)
    runs = store()
    app = FastAPI()
    app.include_router(run_router(store=runs, registry=registry, caller=_caller, **settings))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
            if ended == "timed_out":
                await asyncio.sleep(0.3)  # past the leg's timeout; the call it waits for then finishes
            gate.set()
            await _until_status(runs, run_id, ended)
            await asyncio.sleep(0.1)  # a leg started on the answer would be running by now
            row = (await client.get(f"/runs/{run_id}")).json()
            assert (row["status"], runs.rows[run_id].pending) == (ended, "{}")
            assert [(problem["type"], problem["ctx"]) for problem in row["refused"][waiting_id]] == [("run_not_waiting", {"status": ended})]
            assert row["refused"][waiting_id][0]["msg"] == PROBLEM_CATALOGUE["run_not_waiting"].format(status=ended)
            direct = await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})
            assert (direct.status_code, direct.json()["detail"]) == (409, row["refused"][waiting_id][0]["msg"])

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
    def first(text: str) -> str:
        return text

    @registry.node("slow", title="Slow", description="")
    def slow(text: str) -> str:
        started.append(text)
        time.sleep(0.01)
        return text

    class Flaky(MemoryStore):
        failed_once = False

        async def add_event(self, run_id: str, owner: str, body: str) -> int | None:
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
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": [str(i) for i in range(100)]}})).json()["id"]
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

    @registry.node("many", title="Many", description="Gives several rows.")
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
