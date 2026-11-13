# Serving runs over HTTP

A person who answers days later means a run has to outlive a request. `syv-conductor-providers` has a FastAPI router for that: it starts a run, streams its events, takes answers, resumes it and cancels it, and it does so on any number of workers sharing one database. You give it three things: where runs are kept, who is calling, and which nodes they may use.

```bash
uv add syv-conductor-providers
```

The test suite runs this page as written: two uvicorn workers on one database, a worker killed mid-run, a worker shut down mid-run, and two answers sent to one run at once.

## A store

The router keeps runs in a `RunStore`, six async methods you write against your database. A run is one row of ten columns, text and numbers. Your store acts on four of them: `id`, `rev`, `owner` and `lease_until`. `status` and `caller` are columns for your own queries, such as a page of the runs that wait for one person. `graph`, `run`, `state` and `outcome` are JSON the router writes and reads, which you only keep. Keep them, `caller` and each event's body as text or as `jsonb`: any form that reads back as the same JSON value, since the router parses them and never compares the text. `state` is where the run is (its status and why it ended early, the answers queued and refused, a cancel that was asked); it is small. `status` is there for your queries only: the router writes it in the same `update` as `state`, from it, and itself reads only `state`. `outcome` is what a reader gets without reading the whole run, written each time the run stops running: what waits, how far the run is (its progress, written with each checkpoint too) and, once complete, the outputs. Every change is `update(run_id, rev, changes)`, written only while the row is still at the `rev` the writer read. That one rule is what lets several workers act on one run. The protocol has no delete: purge old runs with your own statement, and never a run whose `status` is `running`, which its worker and its followers read again and take to be there.

This one is SQLite, for one machine with several workers. Write yours the same way for Postgres:

```python
import asyncio
import dataclasses
import sqlite3
from contextlib import closing
from typing import Any

from conductor_providers.fastapi import RunChanges, StoredRun


class SqliteRunStore:
    def __init__(self, path: str) -> None:
        self.path = path
        with closing(self.connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, graph TEXT, caller TEXT, run TEXT, status TEXT,"
                " state TEXT, outcome TEXT, rev INTEGER, owner TEXT, lease_until REAL);"  # a StoredRun's fields, in their order
                "CREATE TABLE IF NOT EXISTS events (run_id TEXT, seq INTEGER, body TEXT, PRIMARY KEY (run_id, seq));"
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)  # each statement commits

    async def sql(self, statement: str, *args: Any) -> list[Any]:
        """One statement, off the event loop."""

        def execute() -> list[Any]:
            with closing(self.connect()) as connection:
                return connection.execute(statement, args).fetchall()

        return await asyncio.to_thread(execute)

    async def create(self, run: StoredRun) -> None:
        await self.sql("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", *dataclasses.astuple(run))

    async def get(self, run_id: str) -> StoredRun | None:
        found = await self.sql("SELECT * FROM runs WHERE id = ?", run_id)
        return StoredRun(*found[0]) if found else None

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        assignments = "".join(f"{name} = ?, " for name in changes)
        changed = await self.sql(
            f"UPDATE runs SET {assignments}rev = rev + 1 WHERE id = ? AND rev = ? RETURNING id", *changes.values(), run_id, rev
        )
        return bool(changed)

    async def expired(self, now: float) -> list[StoredRun]:
        return [StoredRun(*values) for values in await self.sql("SELECT * FROM runs WHERE lease_until < ?", now)]

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        added = await self.sql(  # numbered after what is there, and only while `owner` holds the run
            "INSERT INTO events SELECT ?, COALESCE((SELECT MAX(seq) + 1 FROM events WHERE run_id = ?), 0), ?"
            " WHERE EXISTS (SELECT 1 FROM runs WHERE id = ? AND owner = ?) RETURNING seq",
            run_id, run_id, body, run_id, owner,
        )
        return bool(added)

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        found = await self.sql("SELECT seq, body FROM events WHERE run_id = ? AND seq > ? ORDER BY seq", run_id, after)
        return [(seq, body) for seq, body in found]
```

Test your store with one call, on an empty database of the test's own. `check_run_store` writes runs and events through the six methods and reads them back: that a run is kept as written, that `get` gives `None` for an id no run has, that of two updates at one `rev` exactly one is written, also when they are made at once, that `expired` gives only the runs whose lease ran out, and that `add_event` numbers a run's events from `0` and appends only for the run's owner. It returns a list of problems, each `{type, loc, msg, ctx}` with the method at fault as `loc`; [what it reports](#what-check_run_store-reports) has an example of each.

```python
from pathlib import Path

from conductor_providers.fastapi.testing import check_run_store


async def test_run_store(tmp_path: Path) -> None:
    assert await check_run_store(SqliteRunStore(str(tmp_path / "check.db"))) == []
```

## The app

Mount the router in one line. `caller` is a FastAPI dependency, your login: a function, whose return annotation it must have, or a class used as the dependency. The type it gives is kept on each run as JSON, so whichever worker goes on with a run knows who started it; a function with no return annotation is a `TypeError` when the router is made:

```python
from dataclasses import dataclass

from conductor import GraphRecord, NodeRegistry
from conductor_providers.fastapi import run_router
from fastapi import FastAPI, HTTPException


@dataclass(frozen=True)
class Database:
    """How a node reaches the database: given this, never a connection, it connects on its own thread."""

    path: str


@dataclass(frozen=True)
class Caller:
    name: str


@dataclass(frozen=True)
class AppContext:
    """What a run's nodes get from the host: who started the run, and how to reach the database."""

    caller: Caller
    database: Database


def current_caller() -> Caller:
    return Caller("anna")  # yours: your login


def context_of(caller: Caller) -> AppContext:
    return AppContext(caller=caller, database=database)


registry = NodeRegistry(context_type=AppContext)  # yours: @registry.node(...) or registry.add(...)
database = Database("runs.db")  # yours
offered: dict[str, GraphRecord] = {}  # yours: the graphs an end user may run


def published(name: str) -> GraphRecord:
    if name not in offered:
        raise HTTPException(404, "no such graph")
    return offered[name]


app = FastAPI()
app.include_router(run_router(
    store=SqliteRunStore(database.path),
    registry=lambda caller: registry,
    caller=current_caller,
    may_see=lambda caller, started_by: caller.name == started_by.name,
    context=context_of,
    published=published,
))
```

That gives these routes:

| Route | What it does |
|---|---|
| `GET /nodes` | The caller's palette: `describe()` of the caller's registry, each node with its fields. |
| `POST /compile` | A graph as an editor draws it: `graph.describe()`. Each node has its problems, how often it runs, and its inputs, each a `field` beside its `source`, which says where the input gets its value on each call. The graph's own inputs are each a `field`, by name, and its outputs a list of names. |
| `POST /runs` | Start a run of `{"graph": ..., "inputs": {...}}`; 202 with its `id`, 422 listing every problem. |
| `POST /graphs/{name}/runs` | Start a graph you offer, on its inputs alone. Only with `published=`. |
| `POST /estimate` | How many calls a run of `{"graph": ..., "inputs": {...}}` will make, before any call that costs something: `run.progress` after `run.estimate()`, with the caller's context and `resume_timeout`. Nothing is stored. `POST /graphs/{name}/estimate` does the same for a graph you offer. |
| `GET /runs/{id}` | Its `status`, what is `waiting` for a person (as of when the run last stopped running, so while it is running it still lists what was just answered; a client hides the ones it answered), `outputs` once complete, by name (each as its type writes it: a table as `{"columns": [{"name", "type", "values"}]}`, a float that isn't finite as its text, a host's type by its own schema), `refused` answers, its `progress` (as `run.progress` says it, as of the last checkpoint while it runs), and the `cause` of a run that failed, stopped, was cancelled or timed out (an `item_limit` cause says the `items` and `max_items`, for choosing a higher item limit to resume with). |
| `GET /runs/{id}/events` | Its events as server-sent events, from any worker. |
| `POST /runs/{id}/answers` | `{waiting_id: value}`; 202 `started` or `queued`, 422 listing every answer the run refuses, 409 for a complete run. A run that `failed`, `stopped`, `timed_out` or was `cancelled` takes answers to what waits in it. |
| `POST /runs/{id}/resume` | Go on with a run that `timed_out`, `stopped`, was `cancelled` or `failed`. |
| `POST /runs/{id}/again` | A new run with some inputs changed; what didn't change isn't called again. |
| `POST /runs/{id}/cancel` | Stop a running run before its next node call. |

Every refusal the router makes has one body, `{"detail": [problem, ...]}`, each problem a `{type, loc, msg, ctx}` to translate by its `type`. A 422 lists what the graph or the run refused, with conductor's codes. `/compile` hands the request's own text to `Graph.compile`, and `/runs` and `/estimate` read theirs once with `read_json`, so a key written twice anywhere in the body, such as two `"graph"` keys or two nodes with one id, is a 422 of `invalid_record` (`ctx.error` `repeated_key`), and nothing is stored. Every fault of a graph record is refused as conductor refuses it, `invalid_record` problems, never in FastAPI's own validation shape. A 404 is `no_such_run`, and a 409 is one of the router's own: `already_answered`, `run_complete`, `run_not_resumable` or `run_changed`. A 404 your own `published` dependency raises is yours, as you wrote it. [The router's own problems](#the-routers-own-problems) has each of the router's codes, and `ROUTER_PROBLEMS` is their catalogue: `ROUTER_PROBLEMS.codes` for a test that pins your translations, and each kind's `sentence` and `ctx_type` to write them from. The OpenAPI schema declares the 404 and the 409 as `Refused`.

## What each part is for

**Who can reach `POST /runs`.** That route takes any graph, so whoever reaches it runs any node on any value. It is an author's route: guard it, and `/nodes` and `/compile`, with your own check, `app.include_router(router, dependencies=[Depends(...)])`. For an end user, offer saved graphs with `published=`: the body fills a graph's inputs and nothing else.

**Who may see a run.** You say: `may_see(caller, started_by)` is asked on every route under `/runs/{id}`, and a caller it turns away gets 404, `no_such_run`, as for an id that names no run. Above, only the caller who started a run may read it or act on it; `may_see=lambda caller, started_by: caller.team == started_by.team` lets a team in. There is no default when you name a `caller`, because the obvious one, the same caller, compares everything your `caller` returns: a record with a `last_login` would lock Anna out of her own run the next time she logs in.

**What `caller` returns.** What identifies the caller and nothing that changes: an id, a tenant. It is kept on the run as JSON when the run starts, and the run's context (`context_of`) and registry are made from it as it was then each time the run is resumed, whichever worker does it.

**A registry per tenant.** `registry=` is a function of the caller, so above it gives every caller the one registry. `registry.filter` gives each tenant the nodes it may use instead. The function is called for every request and every time a run is resumed, so it builds each tenant's registry once:

```python
from functools import cache


@dataclass(frozen=True)
class Member:
    name: str
    tenant: str


@cache
def tenant_nodes(tenant: str) -> NodeRegistry[AppContext]:
    return registry.filter(lambda definition: "public" in definition.tags or f"tenant:{tenant}" in definition.tags)


def nodes_of(member: Member) -> NodeRegistry[AppContext]:
    return tenant_nodes(member.tenant)
```

Pass `registry=nodes_of` and a `caller` that returns a `Member`. A graph that names a node its tenant can't use compiles with `unknown_node` at that node, so `/compile` shows the author where.

**What a node receives.** A node that takes the run's context takes it as its first parameter, `Context[AppContext]`, and gets the `AppContext` that `context_of` returns. A run's nodes are called on a thread of the run's own, and a node with `Policy(concurrency=4)` on four more. So the context gives a node a way to connect, and the node opens its own connection:

```python
from conductor import Context


@registry.node("count_runs", title="Count runs", description="Counts how many runs have a given status.")
def count_runs(ctx: Context[AppContext], status: str) -> int:
    with closing(sqlite3.connect(ctx.database.path)) as connection:
        count: int = connection.execute("SELECT COUNT(*) FROM runs WHERE status = ?", (status,)).fetchone()[0]
        return count
```

A connection or session opened on the event loop and passed in either fails or is shared between threads without a word. `sqlite3` refuses it, so the node fails as `execution_failed`, and the real error is only in the `conductor` log. A pool that is safe across threads (SQLAlchemy's engine, not its session) can go in the context as it is. The context is never part of a run's JSON; the router keeps the caller on the run and calls `context_of` again on any worker. A graph with a node whose `Context[...]` your `AppContext` does not satisfy has [`context_does_not_fit`](problems.md#context_does_not_fit) at that node: `/compile` shows it, and `POST /runs` answers 422.

**Two workers answering one run.** Answering reads the run, checks the answers, then claims the run at the `rev` it read. Two workers that both read `waiting` both send an `update` at that `rev`, and only one of them changes the row. The other answers 409 (`run_changed`) and calls no node. An answer to a waiting call whose question has changed since the client read it (after `again`, say) is refused as `answer_stale`, by its waiting id.

**Answers while a run is running.** A worker resumes a run with one call of `resume()`: it calls every node that is ready, until the run is complete, waits, fails or is stopped, and the run is `running` meanwhile. Answers sent then wait in the run's `state`, and the route says `queued`. A second answer to a question already queued is a 409 (`already_answered`, with the `waiting_id` in its `ctx`). The write that records how the call ended reads them, so none is dropped however the two cross. Each is then given to the run as `run.answer` takes one, on its own, whatever status the run stopped with: a wrong one, or one for something that isn't waiting (`not_waiting`, with the run's `status` in its `ctx`, as when the run completed), stays in `refused` with its problems, for the client to show the person who gave it, and the others make the worker resume the run again. So a question answered while a run was running that then failed on another item, stopped at its item limit, timed out or was cancelled is answered all the same. A cancel ends the call of `resume()` that is going on and no other: an answer queued meanwhile resumes the run again, and it runs. A cancel that came after the last node call is kept when a queued answer resumes the run, and ends that call before it calls anything. An answer that is taken clears what `refused` had for its question. When the worker raised while resuming, there is no run to give the answers to, so they stay queued until the run is resumed and stops again; the run meanwhile lists their questions as waiting again, and an answer sent then is taken at once and replaces the queued one. An answer whose type's own check raises what is no refusal is in `refused` as [`answer_check_failed`](#answer_check_failed). Checking an answer reads the whole run and writes it again; at 10,000 items that waited that is about 100 ms and a 1 MB write, so a client with many single clicks does better to send them together.

**A kill.** A running run is saved every `checkpoint_every` seconds (5 by default), and its lease runs out `lease` seconds (30) after its worker stops renewing it. Then any worker takes the run and goes on from the checkpoint. Only the calls the checkpoint has no record of are made again: those in flight, and those finished since. A cancel sent while the dead worker held the run stays asked: the worker that takes it over ends it `cancelled` before calling anything; a resume or an answer goes on from there. A worker that lost its lease writes nothing more, and stops calling the run's nodes at its next renewal, checkpoint or event, whichever the store answers first: a renewal the store hasn't answered when the next is due is given up, so one that never returns blocks nothing else.

**A deploy.** On shutdown, the router hands back every run its worker is resuming (`run.stream(hand_back=)`): no more nodes are started, the calls already made are waited for and kept, and the run is stored as it is, still `running`. The row is then handed back, its lease already run out, and any worker goes on with it. It repeats no call if the process is given the time to hand its runs back: a call on a thread is waited for as far as its `attempt_timeout`, and a process killed before that is a worker that died, whose run goes on from its last checkpoint. A run that stopped running for any other reason (it waits, is complete, or ended early) has no `lease_until` at all, so `expired` is the one comparison and never returns it. A handed-back run writes no `run_end`: the run did not end, so a follower's stream goes on into the next worker's events. A cancel asked of the run before its worker shuts down ends the run `cancelled`; one that lands as the run is handed back is kept in its `state`, and the worker that takes the run ends it `cancelled` before calling anything.

**Resuming.** A run can end `timed_out` (one call of `resume()` took longer than `resume_timeout`, 300 seconds by default), `stopped` (a Series outgrew `max_items`), `cancelled` or `failed`. `POST /runs/{id}/resume` goes on from where it stopped and calls nothing that finished. It takes no body; `{"max_items": ...}`, a whole number of 1 or more (else 422: `true`, `2.5` or `"5"` as the body's own `int_type`, `0` as conductor's [`invalid_max_items`](problems.md#invalid_max_items)), raises the item limit for a stopped run: the run is stored with it (`run.with_max_items`) as the resume claims it, so it applies from then on, after a kill too (a stored run that can't be read is then a 422 from the resume itself); the router's `max_items` is the item limit a run starts with. Only `complete` is final. A run that ended early, a cancelled one included, still takes answers to what waits in it, as `run.answer` does, and they resume it: one item that keeps failing doesn't hold back the questions beside it, and an approver's answer carries a cancelled run on from where it stopped. A host that wants a cancelled run to stay ended marks its own row. When the worker raises outside any node while resuming a run, in a way that would raise again, the run is written `failed`, in its state and on its `run_end`, so it doesn't stay `running` with nobody on it. Where conductor itself refused to resume the run, the cause is conductor's own problem, which says what to fix: [`context_missing`](problems.md#context_missing) or [`invalid_context`](problems.md#invalid_context) when your `context` gives no context or one of another class, [`saved_run_unreadable`](problems.md#saved_run_unreadable) for a stored run that can't be read. A cause is one problem, so of several it is the first, and the log has them all. For anything else (`context` or your registry function raised) the cause is [`resume_failed`](#resume_failed) and the exception is in the log. The run then lists as `waiting` what waits in it as stored, its last checkpoint, which is the run an answer is given to: those questions are answered without a resume. A run that could not be read lists nothing. The same goes for a run that can't be written as JSON (a node gave a value with no JSON form): `failed` the first time, with [`run_unwritable`](problems.md#run_unwritable) as the cause, which names the output; the fix is the node's type. A `run_end` is stored once, before the row says the run stopped running. If another worker takes the run between the two (its lease ran out just then), that worker ends the run too, so a follower sees two `run_end`s and no call is made twice.

**When your store raises.** A raise from your store never fails a run, wherever it comes: when it writes an event, a checkpoint, a renewal or how a run stopped. It is your outage. The worker logs it (`conductor_providers.fastapi`, with the run's id), stops the run before its next node call and gives the run up as the store has it, `running`; once the lease runs out, any worker goes on from the last checkpoint. A renewal that raises is only tried again. Going on from the checkpoint makes again the node calls made after it, a call that is still on a thread included (a node call longer than the lease can so run twice at once, as after any take-over), and a raise when the worker writes how the run stopped comes after its `run_end` is stored, so a follower then sees the run end twice. That is once per lease for as long as the store fails: a store that refuses a write every time (a size limit, not an outage) costs those calls every lease, so alert on that log.

**Following from any worker.** The worker that resumes a run writes each event to the store, numbered in order. `GET /runs/{id}/events` streams them on whichever worker the client reached, until the run stops running, and goes on from `Last-Event-ID` after a reconnect. An open stream costs the store one `events` read per `poll`. It reads the run's row too only on a poll that found no event while the last event it sent was a `run_end`, or while it has sent none yet (a stream opened before the run's first event, or after its last): that is when the run may have stopped running. A client that connects after the run ended gets every event, and `GET /runs/{id}` has the outputs. Each frame is the event's JSON as `conductor.events` defines it. `node_complete` carries counts only, and `run_end` its `status` and `cause` and leaves the run out, so a frame stays small whatever the size of the table.

**The answer's shape.** `POST /runs/{id}/answers` takes `{waiting.id: value}`, each value what the node returns: its value for one output, a mapping by name for several. A client draws each one's `questions` and checks against their `schema` and `choices` before it sends, and the server checks again. A wrong answer is a 422 listing every fault, each naming what it answered in `ctx.waiting_id` and, for a form, the question in `ctx.question` ([problem codes](problems.md#a-refused-call)). One left out goes on waiting.

**Running again.** `POST /runs/{id}/again` with `{"csv": edited}` starts a new run with that input changed. A call that would get the same arguments as a call in the previous run reuses that call's output or answer, so after one line of a CSV of 5,000 is edited, a node that runs per item is called for that one. The new run is the caller's, on the caller's own nodes: a caller who may see another's run (`may_see`) but whose registry can't run its graph gets the 422 at `again`, as starting the graph would.

**Graphs that use saved graphs as subgraphs.** They are compiled with a registry of the request's own ([a graph inside a graph](subgraphs.md#graphs-saved-per-request)): pass `registry=` a function that builds it, so the shared registry is left as it was.

## The router's own problems

What `run_router` reports itself, beside conductor's problems of the runs it serves. `loc` is `()`: each is about the run as a whole. They are not in `PROBLEMS.codes`; `ROUTER_PROBLEMS` is their catalogue, and each kind in it has its `code`, its `sentence` and the type of its `ctx`, as the kinds of `PROBLEMS` have. The docs build writes each kind here from the function that declares it, in `conductor_providers/fastapi/router_problems.py`.

<!-- problems: conductor_providers.fastapi.ROUTER_PROBLEMS -->

## What `check_run_store` reports

What `check_run_store` finds wrong with a store. `loc` is the method at fault, as `("update",)`. Each kind's example is a store from [a store](#a-store) with one method written wrong. They are not in `PROBLEMS.codes` or `ROUTER_PROBLEMS`: a developer reads them in a test run, and a host never meets them. `RUN_STORE_CHECK_PROBLEMS`, in `conductor_providers.fastapi.testing.problems`, is their catalogue.

<!-- problems: conductor_providers.fastapi.testing.problems.RUN_STORE_CHECK_PROBLEMS -->
