# create-graph reference

The examples below run top to bottom in one module, on the standard nodes.

```python
import asyncio
from typing import Annotated

import conductor_nodes
from conductor import Asks, CompiledGraph, Edges, Graph, GraphNode, Input, NodeDefinition, Param, Ref, Result, Series, Static
from conductor.execution.engine import collect, execute, execute_sync
from conductor.widgets import Textarea
from conductor_nodes.types import Text

registry = conductor_nodes.registry(categories=["text"])
```

## Compile

`CompiledGraph.from_graph(graph, registry)` is pure and never raises for a fault in the graph. Everything wrong is a `Problem` (`code`, `message`, `fatal`, `node_id`, `field`, `details`), and `execute` refuses a graph with a fatal one (`CompilationError`).

```python
broken = CompiledGraph.from_graph(
    Graph(nodes=[GraphNode(id="loud", type="text-uppercase", version=1, bindings={"text": Edges(refs=(Ref("ghost", "result"),))})]),
    registry,
)
assert not broken.is_runnable
assert [p.code for p in broken.problems] == ["unknown_ref_node"]
```

Key on `code`, never on the message. `conductor.graph.problem.CATALOGUE` lists every code compile emits.

The compiled graph is asked at three scales:

- **The graph:** `problems`, `is_runnable`, `interface` (what the graph takes and returns, named by address), `execution_order()`, `decisions()`.
- **One node:** `compiled.node(node_id)` gives `interface` (with every type the edges gave it), `iterates_on` (the `Index` it runs once per row of, or `None`), `statics`, `dependencies`, `version`, `embedded_in`, `problems`.
- **One field:** `compiled.field(Ref(node_id, name))` gives `type`, `index`, `binding`, `condition`, `problems`.

A graph may name a definition the registry does not hold, such as a stored graph embedded as a node: build its `NodeDefinition` and compile against `registry.extended_with({"that-id": ThatClass})`.

## Rows

A series arriving on an input declared for one value runs the node once per row; its outputs are series on the same index. An input declared `Series[X]` receives the whole series in one call.

```python
rows = CompiledGraph.from_graph(
    Graph(nodes=[
        GraphNode(id="words", type="text-split", version=1, bindings={"text": Static(value="red,green,blue")}),
        GraphNode(id="loud", type="text-uppercase", version=1, bindings={"text": Edges(refs=(Ref("words", "result"),))}),
        GraphNode(id="joined", type="text-join", version=1, bindings={"parts": Edges(refs=(Ref("loud", "result"),))}),
    ]),
    registry,
)
assert rows.node("loud").iterates_on is not None and rows.node("joined").iterates_on is None

results = execute_sync(rows)
assert list(results["loud"]["result"]) == ["RED", "GREEN", "BLUE"]
assert results["loud"]["result"].rows == ((0,), (1,), (2,))
```

- Several refs on different indexes into a `Series[X]` input gather into one series.
- A row a node skipped is absent from its series downstream; a node skipped above its rows is absent from `results`.
- A node's rows run at most `Policy.concurrency` at a time; the version sets it.

## Events

`execute(compiled, *, record=None, cache=None, from_run=None, timeout=None, cancel=None)` is one **leg**, an async generator of `TypedDict` events:

| Event | Carries |
|---|---|
| `node_start` | `node_id` |
| `node_progress` | `node_id`, `done`, `total` (`None` until every row exists) |
| `node_complete` | `node_id`, `result`; `cached` only when `cache` supplied the node |
| `node_skipped` | `node_id` |
| `node_retry` | `node_id`, `row`, `attempt`, `retries`, `error`, `delay` |
| `node_error` | `node_id`, `error`, `cause` |
| `graph_complete` | `results`, `record` |
| `graph_pending` | `pending`, `results`, `record` |
| `graph_error` | `node_id`, `error`, `cause`, `results`, `record` |
| `graph_cancelled` | `results`, `record` |
| `graph_timeout` | `results`, `record`, `elapsed_seconds`, `timeout_seconds` |

```python
async def watch(compiled):
    async for event in execute(compiled):
        if event["type"] == "node_progress":
            print(event["node_id"], event["done"], "of", event["total"])
        elif event["type"].startswith("graph_"):
            return event


ending = asyncio.run(watch(rows))
assert ending["type"] == "graph_complete"
```

A frame carries records (a `Series`, an `ErrorCause`, an `Input`), not JSON; serialise at the host's edge (`conductor_providers.fastapi.sse.sse_frame` does it for server-sent events). `collect(events)` drains a stream and returns the results, or raises `GraphPendingError` or `GraphExecutionError` (with `cause`) for the ending that stopped it; `execute_sync` is `collect` under `asyncio.run`.

## Legs

A node that returns `Asks` waits, and so does everything that reads it; the rest runs on. The leg ends `graph_pending` with every waiting unit: `node_id`, `row`, `prompt`, and `questions` as `Input` records named by address (`node.field`). Answering is the next leg: `cells` from the ending, and the answers in `cache` as the asking node's outputs.

```python
class Approve(NodeDefinition):
    id = "approve"
    title = "Approve"
    description = "Asks a person to approve a proposal."
    category = "review"

    def run(self, proposal: Annotated[Text, Param(title="Proposal", widget=Textarea())]) -> Annotated[Text, Result(title="Decision")] | Asks:
        return Asks(questions=(Input(name="result", dtype=Text, title="Decision", widget=Textarea(), default=proposal),))


registry.register(Approve)

asking = CompiledGraph.from_graph(
    Graph(nodes=[
        GraphNode(id="drafts", type="text-split", version=1, bindings={"text": Static(value="first,second")}),
        GraphNode(id="approve", type="approve", version=1, bindings={"proposal": Edges(refs=(Ref("drafts", "result"),))}),
    ]),
    registry,
)


async def legs():
    first = [event async for event in execute(asking)][-1]
    assert first["type"] == "graph_pending"
    assert [unit["row"] for unit in first["pending"]] == [[0], [1]]

    # The node runs per row, so its answer is a Series on its index, naming the rows it answers.
    index = asking.node("approve").iterates_on
    only_second = Series(index, [Text("Second, approved")], rows=[(1,)])
    second = [event async for event in execute(asking, record=first["record"], cache={"approve": {"result": only_second}})][-1]
    assert [unit["row"] for unit in second["pending"]] == [[0]]

    only_first = Series(index, [Text("First, approved")], rows=[(0,)])
    third = [event async for event in execute(asking, record=second["record"], cache={"approve": {"result": only_first}})][-1]
    return third


done = asyncio.run(legs())
assert list(done["results"]["approve"]["result"]) == ["First, approved", "Second, approved"]
```

- A pending unit's `row` is a list (`[1]`), as it would be in JSON; `Series(rows=...)` takes tuples: `rows=[tuple(unit["row"]) for unit in units]`.
- For a node that runs once, the answer is the value itself: `cache={"approve": {"result": Text("Yes")}}`.
- A row the answer does not name keeps what it has: done stays done, and a waiting row asks again.
- Answering a unit already done, or a row the run has not produced, raises `ValueError`.
- `record` is a `RunRecord`, JSON through `model_dump()`: a host stores it with the run and hands it back; a node the graph has changed since, or that reads one, runs again. Nothing is checkpointed or resumed.
- Every ending carries `record`, so a new run can also start from a failed or stopped one.
- From a script: `except GraphPendingError as pending:` and then `execute_sync(compiled, record=pending.record, cache=...)`.

**Values the run supplies.** A node parameter `Annotated[T, FromRun()]` receives `from_run[T]`: `execute(compiled, from_run={Caller: caller})`. A leg not given a type some node needs raises `TypeError` before anything runs.

## Saving

A `Graph` is a frozen pydantic model and reads back what it wrote:

```python
graph = Graph(nodes=[GraphNode(id="words", type="text-split", version=1, bindings={"text": Static(value="a,b")})])
text = graph.to_yaml()
assert Graph.from_yaml(text) == graph
```

- `graph.to_path("g.yaml")` / `Graph.from_path("g.yaml")`; the suffix picks JSON or YAML, and any other suffix is refused.
- A ref is stored as its address, `"node.field"`.
- A node's description (`NodeDescription`, `Input`, the widgets) is written for an editor and not read back: call `describe()` again.

## Providers

```python
from conductor_providers import react

palette = react.palette_from_registry(registry)   # [cls.describe() for cls in registry.nodes]
wire = react.graph_to_react(graph)                # the node record under each node's data; edges derived
assert react.react_to_graph(wire).nodes[0].id == "words"
```

`conductor_providers.fastapi.conductor_router(registry, from_run=..., entity_resolver=...)` mounts:

- `GET /nodes`: the palette.
- `POST /compile`: every problem.
- `POST /execute`: the frame the leg ended on, `graph_complete` or `graph_pending`; a failed leg fails the request.
- `POST /execute-stream`: server-sent events.
- `GET /entities/{kind}`: `EntityDropdown` choices.

The body is `{graph, record, cache}`. Over HTTP a per-row answer is `{"rows": [[1]], "values": [...]}`.

## Checklist before running a graph

- [ ] Every `type` is in the registry passed to compile, at the pinned `version`; otherwise `problems` says so.
- [ ] Every `Ref` names a node in the graph and one of its outputs; every bindings key names an input.
- [ ] `compiled.is_runnable` is checked, and `problems` is shown when it is not.
- [ ] The host keeps `record` from a pending ending, and every `from_run` type a node needs is passed.
- [ ] A long run has `timeout=` or a `cancel` event the caller owns; closing the stream (`aclose()`, a cancelled task) stops every unit.
