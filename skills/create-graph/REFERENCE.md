# create-graph reference

One module, on three text nodes declared at the top. Events, errors and serving runs: `python -m conductor.about events` (or `errors`, `serving-runs`).

```python
from typing import Annotated

from conductor import Address, Ask, Graph, GraphRecord, NodeRecord, NodeRegistry, Param, RecordConstant, RecordEdges, Run, Series, node
from conductor.widgets import Textarea

registry = NodeRegistry(category="text")


@registry.node("text-split", title="Split", description="Splits a text at a separator.")
def split(text: str, separator: str = ",") -> Series[str]:
    return text.split(separator)


@registry.node("text-uppercase", title="Uppercase", description="Turns a text into capital letters.")
def uppercase(text: str) -> str:
    return text.upper()


# compiling never raises for a bad graph; key on each Problem's type
broken = Graph.compile(GraphRecord(nodes={"loud": NodeRecord(node="text-uppercase", inputs={"text": RecordEdges(sources=(Address("ghost", "result"),))})}), registry)
assert [p.type for p in broken.problems] == ["edge_names_missing_node"]


@node("approve", title="Approve", description="Asks a person to approve a proposal.", category="review", asks=True)
def approve(proposal: Annotated[str, Param(title="Proposal", widget=Textarea())]) -> Annotated[str, Param(title="Decision", widget=Textarea())]:
    raise Ask("Approve this?", default=proposal)


registry.add(approve)
asking = Graph.compile(GraphRecord(nodes={
    "drafts": NodeRecord(node="text-split", inputs={"text": RecordConstant(constant="first,second")}),
    "approve": NodeRecord(node="approve", inputs={"proposal": RecordEdges(sources=(Address("drafts", "result"),))}),
}), registry)

first = asking.run()
assert first.status == "waiting" and [waiting.inputs["proposal"] for waiting in first.waiting] == ["first", "second"]
stored = Run.from_json(first.to_json(), asking)               # how a host keeps the run until it is answered
answered = stored.answer(stored.waiting[1].id, "Second, approved")
second = answered.resume()
assert [waiting.inputs["proposal"] for waiting in second.waiting] == ["first"]
```

- An asking node's call waits, with the calls downstream of it; the rest runs on. Each entry of `run.waiting` carries `id`, `loc` (the node, then its item), `prompt`, `inputs` and `questions`: one per output, the field record `describe()` writes, with the `choices`, `default` and `widget` the node asked with.
- `run.answer(waiting, value)` takes an entry of `run.waiting` or its id, and a value of what the node returns (a mapping by output name for several outputs); `run.answer({waiting_id: value, ...})` answers several at once. A call not answered goes on waiting and is not called again.
- A wrong answer raises `InvalidCallError` (`invalid_answer`); so does a run that needs a context and is given none (`context_missing`). `run.again(**changed)` reuses the calls whose inputs match, and runs the rest.
- An event writes itself as JSON (`event.model_dump(mode="json")`). Events carry counts, not values: read a node's values from `run.results`. Setting `cancel=` stops the stream before its next call, as does leaving the loop once the stream is closed (`async with contextlib.aclosing(run.stream()) as events:`).
