# Serving runs over HTTP

Conductor has no web server in it. A person who answers days later means a run has to outlive a request, and how a host stores that, who may touch it and what a client sees are the host's. This page is the whole thing as one file to copy: start a run, follow it, read it, answer it, cancel it. It needs `fastapi>=0.135`, and it is what the test suite runs.

Three things hold it up: `run.stream()` runs one leg (the stretch between one pause and the next) and says what happens, `run.answer` writes a person's answer into a stopped run, and a `Run` saves as JSON with `run.to_json()` and reads back with `graph.load_run`.

```python
import asyncio
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any

from conductor import GraphRecord, InvalidCallError, NodeRegistry, Run, Waiting
from conductor.events import Event, RunEnd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel

registry = NodeRegistry()  # your nodes: @registry.node(...) or registry.add(...)
app = FastAPI()


class RunView(BaseModel):
    """What a client reads. `waiting` is its own field so that reading a paused run doesn't load the run."""

    status: str
    waiting: list[Waiting] = []


class SavedRun(RunView):
    """One row of your table: the graph, and the run as JSON."""

    graph: GraphRecord
    run: str


saved: dict[str, SavedRun] = {}  # your table


@dataclass(frozen=True)
class Caller:
    name: str


def current_caller() -> Caller:
    return Caller("anna")  # your login


CallerDep = Annotated[Caller, Depends(current_caller)]


@dataclass
class Leg:
    """A leg running now: what it has emitted, and how to stop it."""

    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    events: list[Event] = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    over: bool = False


# A leg lives on the process that started it, so following and cancelling work
# only on that process: running more than one worker needs an answer of your own.
legs: dict[str, Leg] = {}
tasks: set[asyncio.Task[None]] = set()  # the event loop only holds a task weakly


async def run_leg(run_id: str, events: AsyncIterator[Event], leg: Leg) -> None:
    try:
        async for event in events:
            if isinstance(event, RunEnd):
                saved[run_id] = saved[run_id].model_copy(update={"status": event.status, "waiting": event.run.waiting, "run": event.run.to_json()})
            leg.events.append(event)
            async with leg.changed:
                leg.changed.notify_all()
    finally:
        leg.over = True
        del legs[run_id]
        async with leg.changed:
            leg.changed.notify_all()


def launch(run_id: str, graph: GraphRecord, run: Run, from_run: Mapping[type, Any]) -> None:
    """Start a leg in the background, so a closed connection doesn't end it. A leg that can't start raises here."""
    leg = Leg()
    events = run.stream(from_run=from_run, timeout=300, cancel=leg.cancel)
    legs[run_id] = leg
    saved[run_id] = SavedRun(status="running", graph=graph, run=run.to_json())
    task = asyncio.get_running_loop().create_task(run_leg(run_id, events, leg))
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def saved_run(run_id: str) -> SavedRun:
    if run_id not in saved:
        raise HTTPException(404, "no such run")
    return saved[run_id]


@app.exception_handler(InvalidCallError)
def refused(request: Request, error: InvalidCallError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [problem.model_dump(mode="json") for problem in error.errors()]})


@app.get("/nodes")
def nodes() -> dict[str, Any]:
    return registry.describe()


@app.post("/load")
def load(graph: GraphRecord) -> dict[str, Any]:
    return registry.load(graph).describe()


@app.post("/runs", status_code=202)
async def start(graph: GraphRecord, caller: CallerDep) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    launch(run_id, graph, registry.load(graph).with_inputs(), {Caller: caller})
    return {"id": run_id}


@app.get("/runs/{run_id}")
def read(run: Annotated[SavedRun, Depends(saved_run)]) -> RunView:
    return run


@app.get("/runs/{run_id}/events", response_class=EventSourceResponse)
async def follow(run_id: str, _: Annotated[SavedRun, Depends(saved_run)]) -> AsyncIterable[Event]:
    leg, seen = legs.get(run_id), 0
    while leg is not None:
        async with leg.changed:
            await leg.changed.wait_for(lambda: seen < len(leg.events) or leg.over)
        while seen < len(leg.events):
            seen += 1
            yield leg.events[seen - 1]
        if leg.over:
            return


@app.post("/runs/{run_id}/answers", status_code=202)
async def answer(run_id: str, answers: dict[str, Any], caller: CallerDep) -> dict[str, str]:
    row = await saved_run(run_id)
    if row.status != "waiting":
        raise HTTPException(409, "the run is not waiting")
    run, problems = registry.load(row.graph).load_run(row.run), []
    for unit_id, value in answers.items():
        try:
            run = run.answer(unit_id, value)
        except InvalidCallError as wrong:
            problems.extend(wrong.errors())
    if problems:
        raise InvalidCallError(problems)
    launch(run_id, row.graph, run, {Caller: caller})
    return {"id": run_id}


@app.post("/runs/{run_id}/cancel")
async def cancel(run_id: str) -> dict[str, bool]:
    leg = legs.get(run_id)
    if leg is not None:
        leg.cancel.set()
    return {"cancelled": leg is not None}
```

## What each part is for

**Who can reach `POST /runs`.** That route takes any graph, so whoever reaches it runs any node on any value. It is an author's endpoint: put your author check on it with `dependencies=[Depends(...)]`, and the same on `/nodes` and `/load`. For an end user, offer one saved graph and let the body fill its inputs and nothing else:

```python
published: dict[str, GraphRecord] = {}  # the graphs you offer


@app.post("/graphs/{name}/runs", status_code=202)
async def run_published(name: str, inputs: dict[str, Any], caller: CallerDep) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    launch(run_id, published[name], registry.load(published[name]).with_inputs(inputs), {Caller: caller})
    return {"id": run_id}
```

**What a browser sees.** `run_end` writes only `type` and `status`: every value went out on its own `node_complete`, and the run stays on your row. A host that does want the run on the wire writes `event.run.to_json()`. For frames of your own (fewer fields, translated messages), turn each event into your frame where `run_leg` appends it, and the replay holds your frames.

**What a node receives.** A `FromRun[Caller]` parameter gets the `Caller` from `from_run`. A resource that has to live as long as the leg, such as a database session, opens in a `with` around the `async for` in `run_leg`.

**A registry per request.** The graph is loaded by your route, so a graph that embeds other saved graphs loads through a registry of its own: `request_registry = NodeRegistry()`, then `request_registry.add_subregistry(registry)` and `request_registry.add(*definitions)`. The shared `registry` is left as it was.

**Cancelling.** A cancel ends the leg before its next unit. The calls already made finish first and are kept, so resuming the run doesn't make them again (no mail goes out twice). Give a slow node an `attempt_timeout` to bound how long a cancel waits for it.

**One process.** The comment on `legs` is the limit that matters. A leg is an asyncio task, so `GET /runs/{id}/events` and `POST /runs/{id}/cancel` reach it only on the process running it. `GET /runs/{id}` and answering work anywhere, because they read the saved row. With more than one worker you route those two yourself, or keep a cancel flag on the row and stream from the request that started the leg.

**A restart.** The row outlives the process, so a paused run answers as usual. A run that was `running` when the process died stays `running` until your own cleanup ends it. The row holds the run the leg started from, so a host holding a lease can resume it, calling again what that leg had called.

**Writing the row.** `run_leg` runs on the event loop, so a slow synchronous write stalls every other run. Use an async driver, or `asyncio.to_thread` for a large run.

**The answer's shape.** `POST /runs/{id}/answers` takes `{unit.id: value}`, each value what the node returns. A wrong answer is a 422 listing every fault, and a unit left out goes on waiting.
