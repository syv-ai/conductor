# add-node reference

The examples below run top to bottom in one module.

## Types

Every value on an edge has a `DType`: a real class, usually on a builtin, registered by `id` when it is defined. Conductor declares none but `Series`. A host declares its vocabulary once, and every node imports it:

```python
from typing import Annotated, Any

from conductor import Asks, deprecated, DType, FromRun, Input, NodeDefinition, Output, Param, Policy, Result, Series, Single, SKIPPED, upgrade, version


class Text(DType, str):
    id = "text"
    title = "Text"


class Number(DType, float):
    id = "number"
    title = "Number"


class Flag(DType, int):
    id = "flag"
    title = "Yes or no"
```

- `target.accepts(source)` decides whether an edge may land. The default is `issubclass`; override it on a type that takes another (a `Document` that accepts `Text`, say).
- A `DType` does not convert and does not pick a widget.
- In a notebook or a test, `conductor_nodes.types` (`Text`, `Number`, `Flag`, `Json`) will do. A host declares its own.

## Outputs

One output is `Annotated[X, Result(title=...)]`, named `result`. Several outputs are a frozen dataclass whose fields are the outputs, named by the fields:

```python
from dataclasses import dataclass

from conductor.widgets import TextWidget, Textarea


@dataclass(frozen=True)
class Parts:
    head: Annotated[Text, Result(title="Head")]
    tail: Annotated[Text, Result(title="Tail")]


class Split(NodeDefinition):
    id = "split-once"
    title = "Split once"
    description = "Splits a text at the first separator."
    category = "text"

    def run(
        self,
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        separator: Annotated[Text, Param(title="Separator", widget=TextWidget())] = Text(","),
    ) -> Parts:
        head, _, tail = text.partition(separator)
        return Parts(head=Text(head), tail=Text(tail))
```

A pydantic model returned from `run` is one value, not one output per field.

## Series

`Series[X]` is the one collection. A parameter declared `Series[X]` receives the whole series in one call (a reduction); an output declared `Series[X]` is returned as a plain list, and each item becomes a row.

```python


class Words(NodeDefinition):
    id = "words"
    title = "Words"
    description = "One row per word."
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Series[Text], Result(title="Words")]:
        return [Text(word) for word in text.split()]


class Count(NodeDefinition):
    id = "count"
    title = "Count"
    description = "How many values arrived."
    category = "text"

    def run(self, values: Annotated[Series[Text], Param(title="Values")]) -> Annotated[Number, Result(title="Count")]:
        return Number(len(values))
```

A node written for one value never loops: feed it a series and the engine runs it once per row.

## Branching

A node that takes one of two branches returns `SKIPPED` on the other; what reads that output does not run. Outputs that are exclusive alternatives share a `choice`:

```python
from conductor.widgets import Switch


@dataclass(frozen=True)
class Branches:
    yes: Annotated[Text, Result(title="If yes", choice="answer")]
    no: Annotated[Text, Result(title="If no", choice="answer")]


class Route(NodeDefinition):
    id = "route"
    title = "Route"
    description = "Sends the text one way or the other."
    category = "logic"

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())], go: Annotated[Flag, Param(title="Yes?", widget=Switch())]) -> Branches:
        return Branches(yes=text, no=SKIPPED) if go else Branches(yes=SKIPPED, no=text)
```

The standard library's `decision`, `logic-if-empty` and `logic-if-equals` often do this already.

## Asking

A node that needs a person returns `Asks` in place of a result, and says so in its return annotation. Each question is an `Input` named after the output it fills:

```python


class Approve(NodeDefinition):
    id = "approve"
    title = "Approve"
    description = "Asks a person to approve or rewrite the proposal."
    category = "review"

    def run(self, proposal: Annotated[Text, Param(title="Proposal", widget=Textarea())]) -> Annotated[Text, Result(title="Decision")] | Asks:
        return Asks(
            questions=(Input(name="result", dtype=Text, title="Decision", widget=Textarea(), default=proposal),),
            prompt="Approve or rewrite the proposal.",
        )
```

The answer arrives on the next leg as this node's output, and `run` is not called again for it. How a host answers is in the **create-graph** skill.

## FromRun

What only the caller of `execute` has — a clock, a client, who is running the graph — is not an input. Mark the parameter `FromRun()`; the host hands the value in by type:

```python


class Caller:
    def __init__(self, name: str) -> None:
        self.name = name


class Signed(NodeDefinition):
    id = "signed"
    title = "Signed"
    description = "Signs a text with the caller's name."
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())], who: Annotated[Caller, FromRun()]) -> Annotated[Text, Result(title="Signed")]:
        return Text(f"{text} — {who.name}")
```

`Signed.versions[1].interface.needs` is `{"who": Caller}`, and a leg not given a `Caller` refuses to start.

## Versions

An undecorated `run` is version 1. Once there is a second, every version is marked `@version(n)`, `run` included; the current one is the highest number. Retries, delay, timeout and concurrency belong to the version's `Policy`:

```python
from conductor.errors import ExternalFailure


class Fetch(NodeDefinition):
    id = "fetch"
    title = "Fetch"
    description = "Fetches a page."
    category = "http"

    @version(1)
    @deprecated(header="Use version 2", migration="`url` is now `address`.")
    def run_v1(self, url: Annotated[Text, Param(title="URL", widget=TextWidget())]) -> Annotated[Text, Result(title="Body")]:
        return self.run(address=url)

    @version(2, policy=Policy(retries=3, delay=0.5, timeout=10.0, concurrency=4))
    def run(self, address: Annotated[Text, Param(title="Address", widget=TextWidget())]) -> Annotated[Text, Result(title="Body")]:
        try:
            return Text(f"<html>{address}</html>")
        except TimeoutError as e:
            raise ExternalFailure(str(e)) from e

    @upgrade(1, 2)
    def _rename(values: dict) -> dict:
        return {"address": values["url"]}
```

- Retried: an `ExternalFailure` the node raises, or a foreign exception whose class `Policy(retry_on=(...))` names. Never: any other exception from `run` (wrapped as `NodeExecutionError`), `NodeValidationError`, a timeout.
- The delay before attempt `n` is `delay * 2 ** (n - 1)`. `timeout` is how long the leg waits on one attempt; it never interrupts the thread, and a timed-out attempt is final. Set the timeout worth retrying on the client inside `run`.
- `concurrency` bounds how many rows of this node run at once.
- `NodeRegistry.register` wants versions numbered from 1 with no holes.

## Field hooks

When what one placed node has depends on how the author configured it, override a hook and return the fields that placement has. `declared` is the pinned version's declaration, `values` the typed-in values, `arriving` the type on each connected input:

```python
from collections.abc import Mapping

from conductor import Refuses


class Columns(NodeDefinition):
    id = "columns"
    title = "Columns"
    description = "One output per column name typed in."
    category = "table"

    def run(self, names: Annotated[Text, Param(title="Column names", widget=TextWidget())]) -> Mapping[str, Any]:
        return {name: Text(name.upper()) for name in names.split(",")}

    def compute_outputs(self, declared, values: Mapping[str, Any], arriving) -> tuple[Output, ...]:
        names = values.get("names")
        if not names:
            raise Refuses("no_columns", "Type at least one column name.")
        return tuple(Output(name=name, dtype=Text, title=name) for name in names.split(","))
```

- `-> Mapping[str, Any]` says the placed node's computed interface names the outputs; `run` returns a dict naming exactly them.
- `Refuses(code, message)` is compile's `Problem` on the node, not an exception at run time.
- An `Any` output needs `compute_outputs` to type it from what arrives.

## Open interfaces

`**inputs: Single` makes every name connected into the node an input, each received as one value; `**inputs: Series` receives each as a whole series:

```python


class Template(NodeDefinition):
    id = "template"
    title = "Template"
    description = "Fills {name} placeholders from whatever is connected."
    category = "text"

    def run(self, template: Annotated[Text, Param(title="Template", widget=Textarea())], **inputs: Single) -> Annotated[Text, Result(title="Text")]:
        return Text(template.format(**inputs))
```

## Checklist before shipping a node

- [ ] `id`, `title`, `description`, `category`; the id is unique in the registry.
- [ ] Every handle parameter has a `DType` (or `Any`) and a widget; every return a `Result`, or a record of them.
- [ ] `run` is a plain function and returns the declared types.
- [ ] External calls raise `ExternalFailure` or their client's classes are in `retry_on`, and the version's `Policy` sets `retries`.
- [ ] Tests: the happy path, a wrong input, an external failure that retries, each output of a record, each branch.
