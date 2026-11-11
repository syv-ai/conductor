# create-graph reference

One module, on three text nodes declared at the top. Events, errors and serving runs: `python -m conductor.about events` (or `errors`, `serving-runs`).

```python
from typing import Annotated

from conductor import Ask, GraphRecord, NodeRecord, NodeRegistry, Param, Series, node
from conductor.widgets import Textarea

registry = NodeRegistry(category="text")


@registry.node("text-split", title="Split", description="Splits a text at a separator.")
def split(text: str, separator: str = ",") -> Series[str]:
    return text.split(separator)


@registry.node("text-uppercase", title="Uppercase", description="Turns a text into capital letters.")
def uppercase(text: str) -> str:
    return text.upper()


# loading never raises for a bad graph; key on each Problem's type
broken = registry.load(GraphRecord(nodes=[NodeRecord(id="loud", node="text-uppercase")], edges=[("ghost.result", "loud.text")]))
assert [p.type for p in broken.problems] == ["edge_names_missing_node"]


@node("approve", title="Approve", description="Asks a person to approve a proposal.", category="review")
def approve(proposal: Annotated[str, Param(title="Proposal", widget=Textarea())]) -> Annotated[str, Param(title="Decision", widget=Textarea())]:
    raise Ask("Approve this?", default=proposal)


registry.add(approve)
asking = registry.load(GraphRecord(
    nodes=[NodeRecord(id="drafts", node="text-split", values={"text": "first,second"}), NodeRecord(id="approve", node="approve")],
    edges=[("drafts.result", "approve.proposal")],
))

first = asking.run()
assert first.status == "waiting" and [unit.inputs["proposal"] for unit in first.waiting] == ["first", "second"]
stored = asking.load_run(first.to_json())               # how a host keeps the run between legs
answered = stored.answer(stored.waiting[1].id, "Second, approved")
second = answered.resume()
assert [unit.inputs["proposal"] for unit in second.waiting] == ["first"]
```

- An asking node waits with its readers; the rest runs on. Each entry of `run.waiting` carries `id`, `loc` (the node, then its item), `prompt`, `inputs` and `questions`: one per output, the field record `describe()` writes, with the `choices`, `default` and `widget` the node asked with.
- `run.answer(unit, value)` takes the unit or its id, and a value of what the node returns (a mapping by output name for several outputs); `run.answer({unit_id: value, ...})` answers several at once. A unit not answered goes on waiting and is not called again.
- A wrong answer raises `InvalidCallError`; a missing `FromRun` value raises `InvalidCallError` too. `run.again(**changed)` reuses the calls whose inputs match, and runs the rest.
- An event writes itself as JSON (`event.model_dump(mode="json")`), its outputs already JSON. Leaving `async for event in run.stream():` ends the leg before its next unit.
