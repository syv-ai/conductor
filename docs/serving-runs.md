# Serving runs over HTTP

Conductor has no web server in it. A person who answers days later means a run has to outlive a request, and how a host stores that, who may touch it and what a client sees are the host's. This page is one file to copy, with FastAPI routes to start a run, follow it, read it, answer it, resume it and cancel it. It needs `fastapi>=0.135`, and the test suite runs it as written, as two workers sharing one database.

Three things in conductor hold it up:

- `run.stream()` runs one leg (the stretch between one pause and the next) and says what happens as it goes.
- `run.answer` writes a person's answers into a run that is `waiting`.
- A `Run` is JSON: `run.to_json()`, and `graph.load_run` to read it back.

What a serving host has to handle, and where this file does it:

1. **Nodes run on other threads.** A leg runs on a thread of its own, and a node with `Policy(concurrency=4)` on four more. So a node is given a way to connect, never an open connection: see `Database`.
2. **Two workers can act on one run at once.** Every change of a run is one `UPDATE … WHERE rev = ?`, and only the worker whose update changed the row goes on: see `claim`.
3. **A process can die mid-leg.** The worker running a leg holds a lease and renews it. Another worker takes over a run whose lease ran out, from the run's last checkpoint: see `hold` and `sweep`.
4. **A deploy stops legs.** On shutdown, each leg is cancelled and handed back for another worker to go on with: see `lifespan`.
5. **A leg can end `timed_out`, `stopped`, `cancelled` or `failed`.** Conductor can resume each of these, so the file has a route for it: see `/resume`.
6. **A follower may be on another worker.** Events go to a table, and any worker streams them from there.
7. **People answer while a leg runs.** Those answers wait in a table and are applied when the leg ends.

Everything else (login, who may see which run, how long to keep events) is yours, and the file marks each place with `# yours`.

```python
import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager, closing
from dataclasses import asdict, dataclass
from typing import Annotated, Any

import polars as pl
from conductor import GraphDescription, GraphRecord, InvalidCallError, NodeRegistry, RegistryDescription, Run, Waiting
from conductor.events import RunCheckpoint, RunEnd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from pydantic_core import to_json

registry = NodeRegistry()  # yours: @registry.node(...) or registry.add(...)
WORKER = uuid.uuid4().hex  # this process
LEASE = 30.0  # seconds a worker holds a run between renewals
CHECKPOINT_EVERY = 5.0  # seconds between saves of a running leg
LEG_TIMEOUT = 300.0  # seconds one leg may run before it ends timed_out
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Database:
    """How to reach the database. A node is given this, never a connection, and connects on its own thread."""

    path: str

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)  # each statement commits
        connection.row_factory = sqlite3.Row
        return connection


database = Database("runs.db")  # yours

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, graph TEXT, caller TEXT, status TEXT, waiting TEXT, outputs TEXT, run TEXT,
    rev INTEGER, owner TEXT, lease_until REAL, cancel INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (run_id TEXT, seq INTEGER, body TEXT, PRIMARY KEY (run_id, seq));
CREATE TABLE IF NOT EXISTS pending_answers (run_id TEXT, unit_id TEXT, value TEXT, PRIMARY KEY (run_id, unit_id));
"""


def execute(statement: str, *args: Any) -> list[sqlite3.Row]:
    with closing(database.connect()) as connection:
        return connection.execute(statement, args).fetchall()


async def sql(statement: str, *args: Any) -> list[sqlite3.Row]:
    """One statement, off the event loop, which serves every run."""
    return await asyncio.to_thread(execute, statement, *args)


@dataclass(frozen=True)
class Caller:
    name: str


def current_caller() -> Caller:
    return Caller("anna")  # yours: your login


CallerDep = Annotated[Caller, Depends(current_caller)]


class RunView(BaseModel):
    """What a client reads about a run."""

    status: str
    rev: int
    waiting: list[Waiting] = []
    outputs: dict[str, Any] | None = None


# -- legs ------------------------------------------------------------------------------------------

legs: dict[str, asyncio.Event] = {}  # the legs this worker runs, by run, each with the event that cancels it
tasks: set[asyncio.Task[None]] = set()  # the event loop only holds a task weakly
stopping = False


async def claim(run_id: str, *, status: str, rev: int, run: str | None = None) -> bool:
    """Take the run for this worker if it is still `status` at `rev` and nobody holds it. Of two workers, one wins."""
    now = time.time()
    taken = await sql(
        "UPDATE runs SET status = 'running', owner = ?, lease_until = ?, rev = rev + 1, cancel = 0, run = COALESCE(?, run)"
        " WHERE id = ? AND status = ? AND rev = ? AND (owner IS NULL OR lease_until < ?) RETURNING id",
        WORKER, now + LEASE, run, run_id, status, rev, now,
    )
    return bool(taken)


def launch(run_id: str, max_rows: int | None = None) -> None:
    """Run the next leg of a run this worker claimed, in the background, so a closed connection doesn't end it."""
    legs[run_id] = asyncio.Event()
    task = asyncio.get_running_loop().create_task(drive(run_id, legs[run_id], max_rows))
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def drive(run_id: str, cancel: asyncio.Event, max_rows: int | None) -> None:
    (row,) = await sql("SELECT * FROM runs WHERE id = ?", run_id)
    keep_lease = asyncio.create_task(hold(run_id, cancel))
    try:
        run = registry.load(row["graph"]).load_run(row["run"])
        from_run: dict[type, Any] = {Caller: Caller(**json.loads(row["caller"])), Database: database}
        async for event in run.stream(from_run=from_run, max_rows=max_rows, timeout=LEG_TIMEOUT, cancel=cancel, checkpoint_every=CHECKPOINT_EVERY):
            if isinstance(event, RunCheckpoint):
                await sql("UPDATE runs SET run = ? WHERE id = ? AND owner = ?", await asyncio.to_thread(event.run.to_json), run_id, WORKER)
                continue
            await sql(  # written only while this worker holds the run, numbered after what is there
                "INSERT INTO events SELECT ?, COALESCE((SELECT MAX(seq) + 1 FROM events WHERE run_id = ?), 0), ?"
                " WHERE EXISTS (SELECT 1 FROM runs WHERE id = ? AND owner = ?)",
                run_id, run_id, event.model_dump_json(), run_id, WORKER,
            )
            if isinstance(event, RunEnd):
                await finish(run_id, event.run)
    except Exception:
        log.exception("the leg of run %s failed", run_id)
        await sql("UPDATE runs SET status = 'failed', owner = NULL, rev = rev + 1 WHERE id = ? AND owner = ?", run_id, WORKER)
    finally:
        keep_lease.cancel()
        legs.pop(run_id, None)


async def finish(run_id: str, run: Run) -> None:
    """Write how the leg ended, unless another worker took the run over meanwhile."""
    status, lease_until = run.status, None
    if stopping and run.status == "cancelled":  # stopped by a deploy: hand it back for any worker to go on with
        status, lease_until = "running", 0.0
    waiting = json.dumps([unit.model_dump(mode="json") for unit in run.waiting])
    outputs = None
    if run.status == "complete":  # a table as its records
        outputs = to_json({address: v.to_dicts() if isinstance(v, pl.DataFrame) else v for address, v in run.outputs.items()}).decode()
    await sql(
        "UPDATE runs SET status = ?, waiting = ?, outputs = ?, run = ?, rev = rev + 1, owner = NULL, lease_until = ?"
        " WHERE id = ? AND owner = ?",
        status, waiting, outputs, await asyncio.to_thread(run.to_json), lease_until, run_id, WORKER,
    )
    await apply_pending(run_id)


async def hold(run_id: str, cancel: asyncio.Event) -> None:
    """Renew the lease; stop the leg when the run is cancelled, or when the lease is lost to another worker."""
    while True:
        await asyncio.sleep(LEASE / 3)
        renewed = await sql("UPDATE runs SET lease_until = ? WHERE id = ? AND owner = ? RETURNING cancel", time.time() + LEASE, run_id, WORKER)
        if not renewed or renewed[0]["cancel"]:
            cancel.set()
        if not renewed:
            return


async def sweep() -> None:
    """Go on with the runs whose worker died or handed them back."""
    while True:
        for row in await sql("SELECT id, rev FROM runs WHERE status = 'running' AND lease_until < ?", time.time()):
            if await claim(row["id"], status="running", rev=row["rev"]):
                launch(row["id"])
        await asyncio.sleep(LEASE / 3)


async def apply_pending(run_id: str) -> None:
    """Answers that came while a leg ran: those still waiting are answered and the next leg starts; the rest are dropped."""
    (row,) = await sql("SELECT * FROM runs WHERE id = ?", run_id)
    pending = {r["unit_id"]: json.loads(r["value"]) for r in await sql("SELECT * FROM pending_answers WHERE run_id = ?", run_id)}
    if row["status"] != "waiting" or not pending:
        return
    await sql("DELETE FROM pending_answers WHERE run_id = ?", run_id)
    run = registry.load(row["graph"]).load_run(row["run"])
    still = {unit_id: value for unit_id, value in pending.items() if unit_id in {unit.id for unit in run.waiting}}
    try:
        answered = run.answer(still) if still else None
    except InvalidCallError:
        log.warning("dropped answers to run %s that no longer fit", run_id)
        return
    if answered is not None and await claim(run_id, status="waiting", rev=row["rev"], run=answered.to_json()):
        launch(run_id)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global stopping
    with closing(database.connect()) as connection:
        connection.executescript(SCHEMA)
    sweeping = asyncio.create_task(sweep())
    yield
    stopping = True
    sweeping.cancel()
    for cancel in legs.values():
        cancel.set()  # a cancel keeps the calls already made, so the next worker repeats none
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


# -- routes ----------------------------------------------------------------------------------------


@app.exception_handler(InvalidCallError)
def refused(request: Request, error: InvalidCallError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [problem.model_dump(mode="json") for problem in error.errors()]})


@app.get("/nodes")
def nodes() -> RegistryDescription:
    return registry.describe()


@app.post("/load")
def load(graph: GraphRecord) -> GraphDescription:
    return registry.load(graph).describe()


async def start_run(graph: GraphRecord, run: Run, caller: Caller) -> str:
    run_id = uuid.uuid4().hex
    await sql(
        "INSERT INTO runs (id, graph, caller, status, waiting, run, rev, owner, lease_until) VALUES (?, ?, ?, 'running', '[]', ?, 0, ?, ?)",
        run_id, graph.model_dump_json(), json.dumps(asdict(caller)), run.to_json(), WORKER, time.time() + LEASE,
    )
    launch(run_id)
    return run_id


@app.post("/runs", status_code=202)
async def start(graph: GraphRecord, caller: CallerDep) -> dict[str, str]:
    return {"id": await start_run(graph, registry.load(graph).with_inputs(), caller)}


async def run_row(run_id: str) -> sqlite3.Row:
    rows = await sql("SELECT * FROM runs WHERE id = ?", run_id)  # yours: may this caller see it?
    if not rows:
        raise HTTPException(404, "no such run")
    return rows[0]


RowDep = Annotated[sqlite3.Row, Depends(run_row)]


@app.get("/runs/{run_id}")
def read(row: RowDep) -> RunView:
    outputs = json.loads(row["outputs"]) if row["outputs"] else None
    return RunView(status=row["status"], rev=row["rev"], waiting=json.loads(row["waiting"]), outputs=outputs)


@app.get("/runs/{run_id}/events", response_class=EventSourceResponse)
async def follow(row: RowDep, last_event_id: Annotated[int | None, Header()] = None) -> AsyncIterable[ServerSentEvent]:
    """The run's events from the table, on any worker, until the run stops running."""
    seen = -1 if last_event_id is None else last_event_id
    while True:
        ((status,),) = await sql("SELECT status FROM runs WHERE id = ?", row["id"])
        new = await sql("SELECT seq, body FROM events WHERE run_id = ? AND seq > ? ORDER BY seq", row["id"], seen)
        for event in new:
            seen = event["seq"]
            yield ServerSentEvent(raw_data=event["body"], id=str(seen))
        if not new and status != "running":
            return
        await asyncio.sleep(0.1)


@app.post("/runs/{run_id}/answers", status_code=202)
async def answer(row: RowDep, answers: dict[str, Any], if_match: Annotated[int | None, Header()] = None) -> dict[str, str]:
    if row["status"] == "running":
        for unit_id, value in answers.items():
            await sql("INSERT OR REPLACE INTO pending_answers VALUES (?, ?, ?)", row["id"], unit_id, json.dumps(value))
        await apply_pending(row["id"])  # the leg may have ended meanwhile
        return {"answers": "queued"}
    if row["status"] != "waiting":
        raise HTTPException(409, f"the run is {row['status']}; resume it first")
    answered = registry.load(row["graph"]).load_run(row["run"]).answer(answers)  # every wrong answer is refused at once
    if not await claim(row["id"], status="waiting", rev=row["rev"] if if_match is None else if_match, run=answered.to_json()):
        raise HTTPException(409, "the run changed since it was read")
    launch(row["id"])
    return {"answers": "started"}


class Resume(BaseModel):
    max_rows: int | None = None


@app.post("/runs/{run_id}/resume", status_code=202)
async def resume(row: RowDep, body: Resume) -> dict[str, str]:
    if row["status"] not in ("timed_out", "stopped", "cancelled", "failed"):
        raise HTTPException(409, f"the run is {row['status']}")
    if not await claim(row["id"], status=row["status"], rev=row["rev"]):
        raise HTTPException(409, "the run changed since it was read")
    launch(row["id"], body.max_rows)
    return {"id": row["id"]}


@app.post("/runs/{run_id}/cancel")
async def cancel(row: RowDep) -> dict[str, bool]:
    cancelled = await sql("UPDATE runs SET cancel = 1 WHERE id = ? AND status = 'running' RETURNING id", row["id"])
    if cancelled and row["id"] in legs:
        legs[row["id"]].set()  # on this worker, at once; on another, at its next renewal
    return {"cancelled": bool(cancelled)}
```

## What each part is for

**Who can reach `POST /runs`.** That route takes any graph, so whoever reaches it runs any node on any value. It is an author's endpoint: put your author check on it with `dependencies=[Depends(...)]`, and the same on `/nodes` and `/load`. For an end user, offer one saved graph and let the body fill its inputs and nothing else:

```python
published: dict[str, GraphRecord] = {}  # yours: the graphs you offer


@app.post("/graphs/{name}/runs", status_code=202)
async def run_published(name: str, inputs: dict[str, Any], caller: CallerDep) -> dict[str, str]:
    return {"id": await start_run(published[name], registry.load(published[name]).with_inputs(inputs), caller)}
```

**What a node receives.** Every value in `from_run` is used from the leg's thread, and from several threads at once by a node with `concurrency` above 1. So `from_run` carries `Database`, which knows how to connect, and a node opens its own connection:

```python
from conductor import FromRun


@registry.node("count_runs")
def count_runs(status: str, db: FromRun[Database]) -> int:
    with closing(db.connect()) as connection:
        count: int = connection.execute("SELECT COUNT(*) FROM runs WHERE status = ?", (status,)).fetchone()[0]
        return count
```

A connection or session opened on the event loop and passed in either fails or is shared between threads without a word. `sqlite3` refuses it, so the node fails as `execution_failed`, and the real error is only in the `conductor` log. A pool that is safe across threads (SQLAlchemy's engine, not its session) can go in `from_run` as it is. `from_run` is never part of the run's JSON, so the row keeps what it was built from (`caller`), and any worker can build it again.

**Two workers answering one run.** Answering reads the row, checks the answers, then claims the run at the `rev` it read. Two workers that both read `waiting` both send an `UPDATE … WHERE rev = ?`, and only one of them changes the row. The other answers 409 and calls no node. A client that sends `If-Match: <rev>` gets the same 409 when the run moved since it read it.

**Answers during a leg.** While a leg runs, answers wait in `pending_answers`. When the leg ends, `apply_pending` answers those still in `run.waiting`, drops the rest, and starts the next leg. Checking an answer loads the run and writes it again. At 10,000 rows that is about 60 ms and a 3 MB write, so a client with many single clicks does better to send them together.

**A kill.** A leg writes a checkpoint of its run every `CHECKPOINT_EVERY` seconds, and its lease runs out `LEASE` seconds after its worker stops renewing it. Then any worker's `sweep` claims the run and resumes it from the checkpoint. Only the calls the checkpoint has no record of are made again: those in flight, and those finished since. Every write a leg makes is fenced by `owner = ?`, so a worker that lost its lease and runs on writes nothing, and its leg stops at the next renewal.

**A deploy.** On shutdown, `lifespan` cancels every leg this worker runs. A cancel waits for the calls already made and keeps them. `finish` then hands the run back as `running`, its lease already run out, and the next `sweep` on any worker goes on with it, repeating no call.

**Resuming.** A leg can end `timed_out` (it ran past `timeout`), `stopped` (a table outgrew `max_rows`), `cancelled` or `failed`. Conductor can resume each of these: `resume()` goes on from where the leg stopped and calls nothing that finished. `POST /runs/{id}/resume` does that, with a higher `max_rows` for a stopped run. Which states are final is the host's choice; this file treats only `complete` as final. A cancelled run takes no answers (`run_cancelled`), so `/answers` says to resume it first. A leg that raised out of `stream` is written `failed`, so a run never stays `running` with nobody on it.

**Following from any worker.** The leg's worker writes each event to `events`, numbered in order. `GET /runs/{id}/events` streams them from the table on whichever worker the client reached, and goes on from `Last-Event-ID` after a reconnect. A client that connects after the run ended gets every event, and `GET /runs/{id}` has the outputs. Each frame is the event's JSON as `conductor.events` defines it. `node_complete` carries counts only, and `run_end` leaves the run out, so a frame stays small whatever the size of the table. The values stay in the run, and a client asks for the ones it shows.

**A registry per request.** A graph that embeds other saved graphs loads through a registry of its own, so the shared `registry` is left as it was ([a graph inside a graph](embedded-graphs.md#graphs-saved-per-request)).

**The answer's shape.** `POST /runs/{id}/answers` takes `{unit.id: value}`, each value what the node returns: its value for one output, a mapping by name for several. A client draws each unit's `questions` and checks against their `schema` and `choices` before it sends, and the server checks again. A wrong answer is a 422 listing every fault, each naming the unit in `ctx.waiting_id` and, for a form, the question in `ctx.question` ([problem codes](problems.md#a-refused-call)). A unit left out goes on waiting.

**Running again.** To edit a row of a finished run, load the run from its row and start a new run of `run.with_inputs(**changed)` with `start_run`. What didn't change is taken from the old run, not called again.
