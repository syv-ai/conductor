# create-graph reference

One module, on the standard nodes. Events, errors and providers: `python -m conductor.about events` (or `errors`, `conductor-providers`).

```python
import asyncio
from typing import Annotated

import conductor_nodes
from conductor import Asks, CompiledGraph, execute, From, Graph, GraphNode, Input, NodeDefinition, Param, Result, Series, Static
from conductor.execution.state import RunState
from conductor.widgets import Textarea
from conductor_nodes.types import Text

registry = conductor_nodes.registry(categories=["text"])

# compile never raises for a bad graph; key on each Problem's code
broken = CompiledGraph.from_graph(
    Graph(nodes=[GraphNode(id="loud", type="text-uppercase", version=1, bindings={"text": From("ghost.result")})]),
    registry,
)
assert [p.code for p in broken.problems] == ["unknown_ref_node"]


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
        GraphNode(id="drafts", type="text-split", version=1, bindings={"text": Static("first,second")}),
        GraphNode(id="approve", type="approve", version=1, bindings={"proposal": From("drafts.result")}),
    ]),
    registry,
)


async def legs():
    first = [event async for event in execute(asking)][-1]
    assert first.type == "graph_pending" and [unit.row for unit in first.pending] == [(0,), (1,)]
    stored = RunState.model_validate(first.state.model_dump())      # how a host keeps the state between legs
    answer = Series(asking.node("approve").iterates_on, [Text("Second, approved")], rows=[(1,)])
    second = [event async for event in execute(asking, state=stored, cache={"approve": {"result": answer}})][-1]
    assert [unit.row for unit in second.pending] == [(0,)]


asyncio.run(legs())
```

- An asking node waits with its readers; the rest runs on. Each pending unit carries `node_id`, `row`, `prompt`, `questions`.
- For a node that runs once, the answer is the value: `cache={"approve": {"result": Text("Yes")}}`. A row left out asks again.
- Answering a done unit or an unproduced row raises `ValueError`; a missing `FromRun` value raises `TypeError`. A node changed since the state was taken reruns with its readers.
- Events are records, not JSON; `conductor_providers.fastapi.sse.sse_frame` serialises one. Wrap `execute` in `contextlib.aclosing` so a `break` stops every unit.
