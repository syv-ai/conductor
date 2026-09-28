# add-node reference

One module, one example per shape; the rules are in `python -m conductor.about`.

```python
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from conductor import Asks, deprecated, FromRun, Input, NodeDefinition, Output, Param, Policy, Result, Series, SKIPPED, upgrade, version
from conductor.errors import ExternalFailure, Refuses
from conductor.widgets import Switch, Textarea, TextWidget
from conductor_nodes.types import Flag, Text


@dataclass(frozen=True)
class Branches:                       # one output per field; a shared choice makes them a branch
    yes: Annotated[Text, Result(title="If yes", choice="answer")]
    no: Annotated[Text, Result(title="If no", choice="answer")]


class Route(NodeDefinition):
    id = "route"
    title = "Route"
    description = "Sends the text one way or the other."
    category = "logic"

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())], go: Annotated[Flag, Param(title="Yes?", widget=Switch())]) -> Branches:
        return Branches(yes=text, no=SKIPPED) if go else Branches(yes=SKIPPED, no=text)


class Words(NodeDefinition):
    id = "words"
    title = "Words"
    description = "One row per word."
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Series[Text], Result(title="Words")]:
        return [Text(word) for word in text.split()]      # a list on a Series output: one row per item


class Caller:
    def __init__(self, name: str) -> None:
        self.name = name


class Approve(NodeDefinition):
    id = "approve"
    title = "Approve"
    description = "Asks a person to approve or rewrite the proposal."
    category = "review"

    def run(self, proposal: Annotated[Text, Param(title="Proposal", widget=Textarea())], who: Annotated[Caller, FromRun()]) -> Annotated[Text, Result(title="Decision")] | Asks:
        # a question is an Input named after the output it fills
        return Asks(questions=(Input(name="result", dtype=Text, title="Decision", widget=Textarea(), default=proposal),), prompt=f"{who.name}, approve?")


class Fetch(NodeDefinition):
    id = "fetch"
    title = "Fetch"
    description = "Fetches a page."
    category = "http"

    @version(1)
    @deprecated(header="Use version 2")
    def run_v1(self, url: Annotated[Text, Param(title="URL", widget=TextWidget())]) -> Annotated[Text, Result(title="Body")]: ...

    @version(2, policy=Policy(retries=3, delay=0.5, timeout=10.0, concurrency=4, retry_on=(TimeoutError,)))
    def run(self, address: Annotated[Text, Param(title="Address", widget=TextWidget())]) -> Annotated[Text, Result(title="Body")]:
        if not address:
            raise ExternalFailure("no address")          # retried, like any retry_on class
        return Text(f"<html>{address}</html>")

    @upgrade(1, 2, inputs={"url": "address"})               # the registry moves bindings, locks and edges
    def _rename(values: dict) -> dict:
        return values


class Columns(NodeDefinition):
    id = "columns"
    title = "Columns"
    description = "One output per column name typed in."
    category = "table"

    def run(self, names: Annotated[Text, Param(title="Column names", widget=TextWidget())]) -> Mapping[str, Any]:
        return {name: Text(name.upper()) for name in names.split(",")}

    def compute_outputs(self, declared, values: Mapping[str, Any], arriving) -> tuple[Output, ...]:
        if not values.get("names"):
            raise Refuses("no_columns", "Type at least one column name.")   # a compile Problem on the node
        return tuple(Output(name=name, dtype=Text, title=name) for name in values["names"].split(","))


assert Approve.versions[1].interface.needs == {"who": Caller}
```

- A pydantic model returned from `run` is one value, not one output per field.
- Only `ExternalFailure` and `retry_on` classes retry; attempt `n` waits `delay * 2 ** (n - 1)`. A timed-out attempt is final.
