# create-graph reference

One module, on three text nodes declared at the top. Events, errors and serving runs: `python -m conductor.about events` (or `errors`, `serving-runs`).

```python
from typing import Annotated

from conductor import Ask, GraphRecord, NodeRecord, NodeRegistry, Param, Series, node
from conductor.widgets import Textarea

registry = NodeRegistry(category="text")


@registry.node("text-split")
def split(text: str, separator: str = ",") -> Series[str]:
    return text.split(separator)


@registry.node("text-uppercase")
def uppercase(text: str) -> str:
    return text.upper()


# loading never raises for a bad graph; key on each Problem's type
broken = registry.load(GraphRecord(nodes=[NodeRecord(id="loud", node="text-uppercase")], edges=[("ghost.result", "loud.text")]))
assert [p.type for p in broken.problems] == ["edge_names_missing_node"]


@node("approve", title="Approve", description="Asks a person to approve a proposal.", category="review")
def approve(proposal: Annotated[str, Param(title="Proposal", widget=Textarea())]) -> Annotated[str, Param(title="Decision")]:
    raise Ask("Approve this?", default=proposal, widget=Textarea())


registry.add(approve)
asking = registry.load(GraphRecord(
    nodes=[NodeRecord(id="drafts", node="text-split", values={"text": "first,second"}), NodeRecord(id="approve", node="approve")],
    edges=[("drafts.result", "approve.proposal")],
))

first = asking.run()
assert first.status == "waiting" and [unit.row for unit in first.waiting] == [(0,), (1,)]
stored = asking.load_run(first.to_json())               # how a host keeps the run between legs
answered = stored.answer(stored.waiting[1].id, "Second, approved")
second = answered.resume()
assert [unit.row for unit in second.waiting] == [(0,)]
```

- An asking node waits with its readers; the rest runs on. Each entry of `run.waiting` carries `id`, `node_id`, `row`, `prompt`, `choices`, `default`, `widget`, `answer_schema` and `inputs`.
- `run.answer(unit, value)` takes the unit or its id, and a value of what the node returns. A unit not answered goes on waiting and is not called again.
- A wrong answer raises `InvalidCallError`; a missing `FromRun` value raises `InvalidCallError` too. `run.again(**changed)` reuses the rows whose inputs match, and runs the rest.
- An event writes itself as JSON (`event.model_dump(mode="json")`), its outputs already JSON. Leaving `async for event in run.stream():` ends the leg before its next unit.
