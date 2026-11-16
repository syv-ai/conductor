# Serving runs over HTTP

Some runs outlive a request. A node might ask a person to approve something, and the answer comes the next day. Or the server restarts halfway through a long run. For that, `syv-conductor-providers` gives you a FastAPI router that keeps runs in your database. You can start a run, follow its events, answer its questions, resume it and cancel it, with one server process or ten.

```bash
uv add syv-conductor-providers
```

The test suite runs this page as written: two uvicorn workers on one SQLite file, one of them killed mid-run, one shut down mid-run, and two answers sent to one run at once.

## Writing a store

You need three things: a store, a worker and the router. The store comes first. It is a small class you write over your database.

The worker keeps each run as one row of ten columns, a `StoredRun`. Your store does six things with rows: `create` one, `get` one, `update` one, list the `expired` ones, `add_event` and read the `events`.

There is one rule that matters. `update(run_id, rev, changes)` writes only while the row is still at the `rev` it was given, and then moves `rev` on by one. That is what lets several workers share one run safely: of two writes made from the same read, the second loses.

Your store only looks at four columns: `id`, `rev`, `owner` and `lease_until`. `status` and `metadata` are there for your own queries, such as a page of the runs that wait for one person. The rest is JSON the worker writes and reads. Keep it as text, or as `jsonb`: anything that reads back as the same JSON value.

Here is a store on SQLite, good for one machine with several workers. One on Postgres looks the same.

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
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, graph TEXT, metadata TEXT, run TEXT, status TEXT,"
                " state TEXT, outcome TEXT, rev INTEGER, owner TEXT, lease_until REAL);"  # a StoredRun's fields, in order
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

A store raises only when the database fails, since the worker takes an exception for an outage. It never deletes a run whose `status` is `running`. Deleting old runs is up to you.

Test your store with one line, on an empty database of the test's own. `check_run_store` gives one plain message per fault it finds, such as `"update: wrote at rev 1 while the run was at rev 2; write only while the run is at the rev given"`.

```python
from pathlib import Path

from conductor_providers.fastapi.testing import check_run_store


async def test_run_store(tmp_path: Path) -> None:
    assert await check_run_store(SqliteRunStore(str(tmp_path / "check.db"))) == []
```

## The smallest app

With a store in hand, you need two more things: a worker, which does the real work, and the router, which puts routes in front of it.

```python
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Annotated

from conductor import NodeRegistry
from conductor_providers.fastapi import Worker, run_router
from fastapi import Depends, FastAPI


@dataclass(frozen=True)
class AppContext:
    """What a run's nodes get from you: who started the run, and whether it was asked to stop."""

    user: str
    stop: Event


def current_user() -> str:
    return "anna"  # yours: your login


def run_metadata(user: Annotated[str, Depends(current_user)]) -> dict[str, str]:
    return {"user": user}  # kept on every run this user starts


def own_runs(user: Annotated[str, Depends(current_user)]) -> Callable[[dict[str, str]], bool]:
    return lambda metadata: metadata["user"] == user  # a user sees only their own runs


registry = NodeRegistry(context_type=AppContext)  # yours: your nodes
worker = Worker(
    SqliteRunStore("runs.db"),
    nodes_for=lambda metadata: registry,
    context=lambda metadata, stop: AppContext(metadata["user"], stop),
)
app = FastAPI()
app.include_router(run_router(worker, metadata=run_metadata, run_filter=own_runs))
```

Start it with `uvicorn app:app`, and drive a run with `curl`:

```bash
curl -X POST localhost:8000/runs -d '{"graph": {...}, "inputs": {"invoice": "A-17"}}'
# {"id": "8f1c...", "status": "running", "waiting": [], ...}
curl localhost:8000/runs/8f1c...
# {"status": "waiting", "waiting": [{"id": "approve~3fa2c0d1", "prompt": "Approve A-17?", ...}], ...}
curl -X POST localhost:8000/runs/8f1c.../answers -d '{"approve~3fa2c0d1": true}'
# {"status": "running", ...}; a moment later GET says "complete", with the run's outputs
```

## Who sees which run

Every run keeps some metadata: whatever you want to know about it later, such as who started it. The router asks your `metadata` dependency for it when a run starts, and writes it to the run once. Above, it is `{"user": "anna"}`.

Your `run_filter` dependency decides which runs a request may open. It gives back a test of a run's metadata, and the router asks it on every route under `/runs/{id}`, on the row it reads to answer anyway. A run it turns away answers 404, just like an id that doesn't exist. So nobody learns about runs they may not see.

The metadata can be a pydantic model of yours instead of a dict. Here a run is seen by everyone in the team it was started in:

```python
from pydantic import BaseModel


class Member(BaseModel):
    user: str
    team: str


def current_member() -> Member:
    return Member(user="anna", team="sales")  # yours: your login


def team_runs(member: Annotated[Member, Depends(current_member)]) -> Callable[[Member], bool]:
    return lambda started: started.team == member.team
```

Make the worker with `metadata_type=Member`, so it reads the metadata back as a `Member`, and pass `metadata=current_member, run_filter=team_runs` to the router. `nodes_for` and `context` then get a `Member` too, which is how you give each team its own nodes.

A host with no logins gives every run empty metadata and lets everyone in:

```python
def no_metadata() -> dict[str, str]:
    return {}


def every_run() -> Callable[[dict[str, str]], bool]:
    return lambda metadata: True
```

Neither dependency has a default, since no rule is safe for every host. The worker checks the metadata against its `metadata_type` before it stores a run. Since the metadata comes from your own code, a value of another type is a bug there, and pydantic's error is raised as a 500.

The routes that take any graph (`POST /runs`, `/estimate`, `/compile` and `/nodes`) run whatever nodes the graph names. Guard them the usual FastAPI way: `app.include_router(router, dependencies=[Depends(is_author)])`. On those routes, the metadata picks the nodes the request may use, through `nodes_for`.

## The routes

Every route under `/runs` but the events route answers with the run as it is after the call: its `id`, `metadata`, `status`, `cause`, `waiting`, `outputs`, `progress` and `refused`. Its `metadata` is the JSON the run keeps, whatever type you gave the worker.

| Route | What it does |
|---|---|
| `GET /nodes` | The nodes this request may use, as an editor's palette shows them. |
| `POST /compile` | Compiles a graph record and describes it, with every problem. Nothing runs. |
| `POST /runs` | Starts a run of `{"graph", "inputs"}`. With `"previous": run_id`, the run reuses that run's work. |
| `POST /estimate` | Says how many calls that run would make, calling only the nodes that cost nothing. Nothing is stored. |
| `GET /runs/{id}` | The run, as of when it last stopped running. |
| `GET /runs/{id}/events` | The run's events, as server-sent events, from whichever worker runs it. |
| `POST /runs/{id}/answers` | Answers what waits, as `{waiting_id: value}`, and goes on with the run. |
| `POST /runs/{id}/declines` | Declines one question, as `{"waiting_id": ..., "reason": ...}`. The run ends `failed` with `answer_declined`, and isn't resumed; a resume asks the question again. A waiting run takes it at once, so its events get a `run_end` with no `run_start` before it. A running run is given it when its resume call stops, and that call's `run_end` says `failed`. It can still refuse it then, as `not_waiting` after an answer to the same question or as `answer_stale`: the decline is in the run's `refused`, and the run goes on as it would have. |
| `POST /runs/{id}/resume` | Goes on with a run that ended early. `{"max_items": n}` raises its item limit. |
| `POST /runs/{id}/again` | A new run of the same graph with some inputs changed. What didn't change isn't called again. |
| `POST /runs/{id}/cancel` | Asks the run to stop before its next node call. A run that answers `running` here ends `cancelled`. A cancel that comes after its last node call started is dropped if the run completes or ends early. If the run ends `waiting`, it still ends `cancelled`. |

Answers and declines sent while a run is running are queued and given to the run when it stops, in the order they came. The route still answers 202, with the run `running`. One the run turns down then shows up in the run's `refused`, by its waiting id. Of an answer and a decline to one question, the first that came is taken, and the second is refused as `not_waiting`.

## When a call is refused

Every refusal has the shape of FastAPI's own 422: `{"detail": [{"type", "loc", "msg", ...}, ...]}`. Each route's OpenAPI entry declares the refusals it can answer with. A missing run, or one the filter hides, is a 404. Resuming a run that didn't end early is a 409. Everything else is a 422: a graph that doesn't compile, inputs it refuses, a wrong answer. Those problems are conductor's own, listed on the [problems page](problems.md).

The worker has four codes of its own, in `SERVING_PROBLEMS`:

<!-- problems: conductor_providers.fastapi.SERVING_PROBLEMS -->

If you show people your own text for each code, one test keeps your translations complete when a code is added:

```python
from conductor import PROBLEMS
from conductor_providers.fastapi import SERVING_PROBLEMS


def test_every_code_is_translated(translations: dict[str, str]) -> None:
    assert PROBLEMS.codes | SERVING_PROBLEMS.codes <= translations.keys()
```

## What a node gets

A node that takes the run's context as its first parameter gets what your `context` function returns. The worker calls it with the run's metadata and a stop event each time it resumes the run, on whichever worker that is.

Each run's nodes are called on threads of their own. So give a node a way to connect, never an open connection, and let the node open its own:

```python
from contextlib import closing

from conductor import Context


@registry.node("count_runs", title="Count runs", description="Counts how many runs have a given status.")
def count_runs(ctx: Context[AppContext], status: str) -> int:
    with closing(sqlite3.connect("runs.db")) as connection:
        count: int = connection.execute("SELECT COUNT(*) FROM runs WHERE status = ?", (status,)).fetchone()[0]
        return count
```

A cancel stops a run before its next node call, so a call that is already going on runs to its end. For a slow call that costs money, such as a language model streaming its reply, read the stop event. The worker sets it when the run is cancelled, or when another worker takes the run over.

```python
from collections.abc import Iterator

from conductor import NodeError


def reply_chunks(prompt: str) -> Iterator[str]:
    yield from prompt.split()  # yours: a model's reply as it streams


@registry.node("reply", title="Reply", description="Writes a reply chunk by chunk, and stops when the run is cancelled.")
def reply(ctx: Context[AppContext], prompt: str) -> str:
    chunks = []
    for chunk in reply_chunks(prompt):
        if ctx.stop.is_set():
            raise NodeError("The reply was stopped because the run was cancelled.", code="reply_stopped")
        chunks.append(chunk)
    return " ".join(chunks)
```

Raise when you stop, and never return what you have so far. A value a node returns is kept, so a resume wouldn't call the node again.

## Many workers

Run as many workers on one store as you like. Each of them serves every run.

**A kill.** A running run is saved every few seconds (`checkpoint_every`). Its worker holds a lease on it and keeps renewing it. When a worker dies, its lease runs out (`lease`, 30 seconds), and another worker goes on from the last checkpoint. Only the calls made since that checkpoint are made again.

**A deploy.** When a worker shuts down, it stops starting nodes, waits for the calls already going on, saves the run and hands it back. Another worker picks it up within seconds, and no call is made twice.

**A store outage.** If your store raises, the worker never fails the run for it. It stops the run and lets the lease run out. Once the store is back, a worker goes on from the last checkpoint.

**Two answers at once.** Of two workers that answer one run at the same moment, one goes on with the run. The other reads the run again. If the run is still running, the second answer is queued, and the run turns it down as `not_waiting` in its `refused` when it stops. If the run has already finished, the second answer is a 422 with the same code. Either way, the question was answered once.

The worker's docstrings have the finer points, such as a cancel during a take-over, a run that can't be saved, or an answer that lands while a run's end is being written.

## Your own routes

The router is a thin layer over the `Worker`, and you can call the worker from routes of your own. Here is a route that offers a saved graph by name, so an end user fills in its inputs and never sends a graph:

```python
from typing import Any

from conductor import GraphRecord, InvalidCallError
from conductor_providers.fastapi import RunView
from fastapi import APIRouter, Body, HTTPException

saved_graphs: dict[str, GraphRecord] = {}  # yours: the graphs you offer
own_routes = APIRouter(prefix="/saved")


@own_routes.post("/{name}/runs", status_code=202)
async def start_saved_graph(
    name: str,
    inputs: Annotated[dict[str, Any], Body()],
    metadata: Annotated[dict[str, str], Depends(run_metadata)],
) -> RunView:
    if name not in saved_graphs:
        raise HTTPException(404, "There is no saved graph by that name.")
    try:
        return await worker.begin(saved_graphs[name], inputs, metadata)
    except InvalidCallError as refused:
        raise HTTPException(422, [problem.model_dump(mode="json") for problem in refused.problems]) from None


app.include_router(own_routes)
```

The worker refuses a call with conductor's `InvalidCallError`. The router turns that into a 404, 409 or 422 for you, so in a route of your own you do it yourself, as above. The run this route starts is read, answered and followed through the router's routes.

Every worker method that acts on a run takes the run's id, and returns the run as it is after the call. Each of them takes your run filter as `allows=`, as the router passes it, and turns away a run the request may not see; so does `begin` when you give it a `previous` run. With no logins, pass `allows=lambda metadata: True`. Following a run's events takes two steps: awaiting `worker.events(...)` checks the run, so a run that isn't there is refused before you start, and then you iterate what it gives:

```python
async def print_events(run_id: str) -> None:
    async for number, body in await worker.events(run_id, allows=lambda metadata: True):
        print(number, body)
```

Enter `worker.serving()` in your app's lifespan, as the router does for you. A worker that never served refuses to start a run, since nothing of its own would ever pick the run up.
