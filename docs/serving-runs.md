# Serving runs over HTTP

A person who answers days later means a run has to outlive a request. `syv-conductor-providers` has a FastAPI router for that: it starts a run, streams its events, takes answers, resumes it and cancels it, and it does so on any number of workers sharing one database. You give it three things: where runs are kept, who is calling, and which nodes they may use.

```bash
uv add syv-conductor-providers
```

The test suite runs this page as written: two uvicorn workers on one database, a worker killed mid-run, a worker shut down mid-run, and two answers sent to one run at once.

## A store

The router keeps runs in a `RunStore`, six async methods you write against your database. A run is one row of plain values: text, numbers and a flag. Every change is `update(run_id, rev, changes)`, written only while the row is still at the `rev` the writer read. That one rule is what lets several workers act on one run.

This one is SQLite, for one machine with several workers. Write yours the same way for Postgres:

```python
import asyncio
import dataclasses
import sqlite3
from contextlib import closing
from typing import Any

from conductor_providers.fastapi import RunChanges, StoredRun

COLUMNS = [f.name for f in dataclasses.fields(StoredRun)]


class SqliteRunStore:
    def __init__(self, path: str) -> None:
        self.path = path
        with closing(self.connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, graph TEXT, caller TEXT, run TEXT, status TEXT,"
                " rev INTEGER, waiting TEXT, outputs TEXT, owner TEXT, lease_until REAL, cancel INTEGER, pending TEXT, refused TEXT, cause TEXT);"
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

    def row(self, values: tuple[Any, ...]) -> StoredRun:
        found = dict(zip(COLUMNS, values))
        return StoredRun(**{**found, "cancel": bool(found["cancel"])})

    async def create(self, run: StoredRun) -> None:
        await self.sql(f"INSERT INTO runs VALUES ({', '.join('?' * len(COLUMNS))})", *dataclasses.astuple(run))

    async def get(self, run_id: str) -> StoredRun | None:
        found = await self.sql("SELECT * FROM runs WHERE id = ?", run_id)
        return self.row(found[0]) if found else None

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        assignments = "".join(f"{name} = ?, " for name in changes)
        changed = await self.sql(
            f"UPDATE runs SET {assignments}rev = rev + 1 WHERE id = ? AND rev = ? RETURNING id", *changes.values(), run_id, rev
        )
        return bool(changed)

    async def expired(self, now: float) -> list[StoredRun]:
        return [self.row(r) for r in await self.sql("SELECT * FROM runs WHERE status = 'running' AND lease_until < ?", now)]

    async def add_event(self, run_id: str, owner: str, body: str) -> int | None:
        added = await self.sql(  # numbered after what is there, and only while `owner` holds the run
            "INSERT INTO events SELECT ?, COALESCE((SELECT MAX(seq) + 1 FROM events WHERE run_id = ?), 0), ?"
            " WHERE EXISTS (SELECT 1 FROM runs WHERE id = ? AND owner = ?) RETURNING seq",
            run_id, run_id, body, run_id, owner,
        )
        return int(added[0][0]) if added else None

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        found = await self.sql("SELECT seq, body FROM events WHERE run_id = ? AND seq > ? ORDER BY seq", run_id, after)
        return [(seq, body) for seq, body in found]
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


def current_caller() -> Caller:
    return Caller("anna")  # yours: your login


registry = NodeRegistry()  # yours: @registry.node(...) or registry.add(...)
database = Database("runs.db")  # yours
offered: dict[str, GraphRecord] = {}  # yours: the graphs an end user may run


def published(name: str) -> GraphRecord:
    if name not in offered:
        raise HTTPException(404, "no such graph")
    return offered[name]


app = FastAPI()
app.include_router(run_router(
    store=SqliteRunStore(database.path),
    registry=registry,
    caller=current_caller,
    from_run=lambda caller: {Caller: caller, Database: database},
    published=published,
))
```

That gives these routes:

| Route | What it does |
|---|---|
| `GET /nodes` | The caller's palette: `registry.describe()`, each node with its fields. |
| `POST /load` | A graph as an editor draws it: `graph.describe()`. Each node has its problems, how often it runs, and its inputs, each a `field` beside its `feed`, which says how the input gets its value on each call. The graph's own inputs are each an `address`, a `node_id` and a `field`. |
| `POST /runs` | Start a run of `{"graph": ..., "inputs": {...}}`; 202 with its `id`, 422 listing every problem. |
| `POST /graphs/{name}/runs` | Start a graph you offer, on its inputs alone. Only with `published=`. |
| `GET /runs/{id}` | Its `status`, what is `waiting` for a person (as of the last leg's end, so while a leg runs it still lists what was just answered; a client hides the ones it answered), `outputs` once complete (each as its type writes it: a table as `{"columns": [{"name", "type", "values"}]}`, a float that isn't finite as its text, a host's type by its own schema), `refused` answers, and the `cause` of a run that failed, stopped, was cancelled or timed out (an `item_ceiling` cause says the `items` and `max_items`, for choosing a resume's ceiling). |
| `GET /runs/{id}/events` | Its events as server-sent events, from any worker. |
| `POST /runs/{id}/answers` | `{waiting_id: value}`; 202 `started` or `queued`, 409, or 422 listing every fault. |
| `POST /runs/{id}/resume` | Go on with a run that `timed_out`, `stopped`, was `cancelled` or `failed`. |
| `POST /runs/{id}/again` | A new run with some inputs changed; what didn't change isn't called again. |
| `POST /runs/{id}/cancel` | Stop a running leg before its next unit. |

## What each part is for

**Who can reach `POST /runs`.** That route takes any graph, so whoever reaches it runs any node on any value. It is an author's route: guard it, and `/nodes` and `/load`, with your own check, `app.include_router(router, dependencies=[Depends(...)])`. For an end user, offer saved graphs with `published=`: the body fills a graph's inputs and nothing else.

**Who may see a run.** By default only the caller who started a run may read it or act on it. Anyone else gets 404. `may_see=lambda caller, started_by: caller.team == started_by.team` widens that.

**A registry per tenant.** `registry=` takes a function of the caller instead of a registry. `registry.filter` gives each tenant the nodes it may use. The function is called for every request and every leg, so it builds each tenant's registry once:

```python
from functools import cache


@dataclass(frozen=True)
class Member:
    name: str
    tenant: str


@cache
def tenant_nodes(tenant: str) -> NodeRegistry:
    return registry.filter(lambda definition: "public" in definition.tags or f"tenant:{tenant}" in definition.tags)


def nodes_of(member: Member) -> NodeRegistry:
    return tenant_nodes(member.tenant)
```

Pass `registry=nodes_of` and a `caller` that returns a `Member`. A graph that names a node its tenant can't use loads with `unknown_node` at that node, so `/load` shows the author where.

**What a node receives.** A leg runs on a thread of its own, and a node with `Policy(concurrency=4)` on four more. So `from_run` gives a node a way to connect, and the node opens its own connection:

```python
from conductor import FromRun


@registry.node("count_runs", title="Count runs", description="Counts how many runs have a given status.")
def count_runs(status: str, db: FromRun[Database]) -> int:
    with closing(sqlite3.connect(db.path)) as connection:
        count: int = connection.execute("SELECT COUNT(*) FROM runs WHERE status = ?", (status,)).fetchone()[0]
        return count
```

A connection or session opened on the event loop and passed in either fails or is shared between threads without a word. `sqlite3` refuses it, so the node fails as `execution_failed`, and the real error is only in the `conductor` log. A pool that is safe across threads (SQLAlchemy's engine, not its session) can go in `from_run` as it is. `from_run` is never part of a run's JSON; the router keeps the caller on the run and builds `from_run` from it again on any worker.

**Two workers answering one run.** Answering reads the run, checks the answers, then claims the run at the `rev` it read. Two workers that both read `waiting` both send an `update` at that `rev`, and only one of them changes the row. The other answers 409 and calls no node. An answer whose place has come to hold another question since the client read it is refused as `answer_stale` by its waiting id.

**Answers during a leg.** While a leg runs, answers wait on the run as `pending`, and the route says `queued`. A second answer to a question already queued is a 409. The write that ends the leg reads them, so none is dropped however the two cross. A leg that ends `waiting` takes each on its own: a wrong one, or one for something that isn't waiting (`not_waiting`, as when the leg completed the run), stays in `refused` with its problems, for the client to show the person who gave it, and the others start the next leg. A leg that ends `stopped`, `timed_out`, `cancelled` or `failed` takes none: each is in `refused` as [`run_not_waiting`](problems.md#run_not_waiting), with the run's `status` in its `ctx`, and the client sends it again after a resume, as `/answers` says of such a run with a 409. Checking an answer loads the run and writes it again; at 10,000 items that is about 60 ms and a 3 MB write, so a client with many single clicks does better to send them together.

**A kill.** A leg saves its run every `checkpoint_every` seconds (5 by default), and its lease runs out `lease` seconds (30) after its worker stops renewing it. Then any worker takes the run and goes on from the checkpoint. Only the calls the checkpoint has no record of are made again: those in flight, and those finished since. A cancel sent while the dead worker held the run stays asked: the worker that takes it over ends it `cancelled` before calling anything, and only `resume` goes on. A worker that lost its lease writes nothing more, and its leg stops at the next renewal.

**A deploy.** On shutdown, the router cancels every leg its worker runs. A cancel waits for the calls already made and keeps them. The run is then handed back, its lease already run out, and any worker goes on with it, repeating no call. A cancel asked of the run as its worker shuts down is kept: the run ends `cancelled`, not handed back.

**Resuming.** A leg can end `timed_out` (past `leg_timeout`, 300 seconds by default), `stopped` (a Series outgrew `max_items`), `cancelled` or `failed`. `POST /runs/{id}/resume` goes on from where it stopped and calls nothing that finished. It takes no body; `{"max_items": ...}`, a whole number of 1 or more (else 422), raises the ceiling for a stopped run, and the raised ceiling holds for every later leg of that run; the router's `max_items` is the ceiling a run starts with. Only `complete` is final. A cancelled run takes no answers, so `/answers` says to resume it first. A leg that raised is written `failed`, so a run never stays `running` with nobody on it.

**Following from any worker.** The leg's worker writes each event to the store, numbered in order. `GET /runs/{id}/events` streams them on whichever worker the client reached, until the run stops running, and goes on from `Last-Event-ID` after a reconnect. A client that connects after the run ended gets every event, and `GET /runs/{id}` has the outputs. Each frame is the event's JSON as `conductor.events` defines it. `node_complete` carries counts only, and `run_end` its `status` and `cause` and leaves the run out, so a frame stays small whatever the size of the table.

**The answer's shape.** `POST /runs/{id}/answers` takes `{waiting.id: value}`, each value what the node returns: its value for one output, a mapping by name for several. A client draws each one's `questions` and checks against their `schema` and `choices` before it sends, and the server checks again. A wrong answer is a 422 listing every fault, each naming what it answered in `ctx.waiting_id` and, for a form, the question in `ctx.question` ([problem codes](problems.md#a-refused-call)). One left out goes on waiting.

**Running again.** `POST /runs/{id}/again` with `{"csv": edited}` starts a new run with that input changed. A call that receives what it received before takes its earlier result or answer, so after one line of a CSV of 5,000 is edited, a node that runs per item is called for that one. The new run is the caller's, on the caller's own nodes: a caller who may see another's run (`may_see`) but whose registry can't run its graph gets the 422 at `again`, as starting the graph would.

**Graphs that embed saved graphs.** They load through a registry of the request's own ([a graph inside a graph](embedded-graphs.md#graphs-saved-per-request)): pass `registry=` a function that builds it, so the shared registry is left as it was.
