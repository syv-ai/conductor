# create-graph reference

One module, on the standard nodes. Events, errors and providers: `python -m conductor.about events` (or `errors`, `conductor-providers`).

```python
import asyncio
from typing import Annotated

import conductor_nodes
from conductor import AskPerson, CompiledGraph, From, Graph, GraphNode, Param, Result, RunState, Static, node
from conductor.widgets import Textarea


registry = conductor_nodes.registry(categories=["text"])

# compile never raises for a bad graph; key on each Problem's type
broken = CompiledGraph.from_graph(
    Graph(nodes=[GraphNode(id="loud", type="text-uppercase", version=1, bindings={"text": From("ghost.result")})]),
    registry,
)
assert [p.type for p in broken.problems] == ["unknown_ref_node"]


@node("approve", title="Approve", description="Asks a person to approve a proposal.", category="review")
def approve(proposal: Annotated[str, Param(title="Proposal", widget=Textarea())]) -> Annotated[str, Result(title="Decision")] | AskPerson:
    return AskPerson(defaults={"result": proposal}, widgets={"result": Textarea()})


registry = registry.with_nodes([approve])
asking = CompiledGraph.from_graph(
    Graph(nodes=[
        GraphNode(id="drafts", type="text-split", version=1, bindings={"text": Static("first,second")}),
        GraphNode(id="approve", type="approve", version=1, bindings={"proposal": From("drafts.result")}),
    ]),
    registry,
)


async def legs():
    first = await asking.run()
    assert first.status == "waiting" and [unit.row for unit in first.waiting] == [(0,), (1,)]
    stored = RunState.model_validate_json(first.model_dump_json())  # how a host keeps the state between legs
    answered = asking.answer(stored, {stored.waiting[1].id: {"result": "Second, approved"}})
    second = await asking.run(answered)
    assert [unit.row for unit in second.waiting] == [(0,)]


asyncio.run(legs())
```

- An asking node waits with its readers; the rest runs on. Each entry of `state.waiting` carries `id`, `node_id`, `row`, `prompt`, `questions`.
- An answer is `{unit.id: {output: value}}`, the id taken from `state.waiting`. A unit left out goes on waiting and is not called again.
- A wrong answer raises `InvalidCallError`, every fault listed; a missing `FromRun` value raises `InvalidCallError` too. A node changed since the state was taken reuses the rows whose inputs match, and runs the rest again.
- An event writes itself as JSON (`event.model_dump(mode="json")`, each value by its own type), and `conductor.to_jsonable(compiled.results(state))` writes the results. Leaving `async with compiled.stream() as events:` stops every unit.
