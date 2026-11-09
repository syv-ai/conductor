# add-node reference

One module, one example per shape; the rules are in `python -m conductor.about`.

```python
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from conductor import AskPerson, Arriving, Deprecation, ExternalFailure, FromRun, Input, OlderVersion, Upgrade, node, Output, Param, Policy, CompileProblem, Result, Series, SKIPPED
from conductor.widgets import Switch, Textarea, TextWidget



@dataclass(frozen=True)
class Branches:                       # one output per field; a shared choice makes them a branch
    yes: Annotated[str, Result(title="If yes", choice="answer")]
    no: Annotated[str, Result(title="If no", choice="answer")]


@node("route", title="Route", description="Sends the text one way or the other.", category="logic")
def route(text: Annotated[str, Param(title="Text", widget=Textarea())], go: Annotated[bool, Param(title="Yes?", widget=Switch())]) -> Branches:
    return Branches(yes=text, no=SKIPPED) if go else Branches(yes=SKIPPED, no=text)


@node("words", title="Words", description="One row per word.", category="text")
def words(text: Annotated[str, Param(title="Text", widget=Textarea())]) -> Annotated[Series[str], Result(title="Words")]:
    return [word for word in text.split()]      # a list on a Series output: one row per item


class Caller:
    def __init__(self, name: str) -> None:
        self.name = name


@node("approve", title="Approve", description="Asks a person to approve or rewrite the proposal.", category="review")
def approve(proposal: Annotated[str, Param(title="Proposal", widget=Textarea())], who: FromRun[Caller]) -> Annotated[str, Result(title="Decision")] | AskPerson:
    # what is asked is the declared output; a default proposes an answer
    return AskPerson(prompt=f"{who.name}, approve?", defaults={"result": proposal}, widgets={"result": Textarea()})


def fetch_v1(url: Annotated[str, Param(title="URL", widget=TextWidget())]) -> Annotated[str, Result(title="Body")]: ...

@node("fetch", title="Fetch", description="Fetches a page.", category="http", policy=Policy(retries=3, delay=0.5, timeout=10.0, concurrency=4, retry_on=(TimeoutError,)), older=[OlderVersion(fetch_v1, upgrade=Upgrade(renamed_inputs={"url": "address"}), deprecation=Deprecation(header="Use version 2"))])
def fetch(address: Annotated[str, Param(title="Address", widget=TextWidget())]) -> Annotated[str, Result(title="Body")]:
    if not address:
        raise ExternalFailure("no address")          # retried, like any retry_on class
    return f"<html>{address}</html>"


def columns_outputs(names):
    if not names:
        raise CompileProblem("no_columns", "Type at least one column name.")   # a compile Problem on the node
    return tuple(Output(name, str) for name in names.split(","))

@node("columns", title="Columns", description="One output per column name typed in.", category="table", compute_outputs=columns_outputs)
def columns(names: Annotated[str, Param(title="Column names", widget=TextWidget())]) -> Mapping[str, Any]:
    return {name: name.upper() for name in names.split(",")}


assert approve.versions[1].interface.needs == {"who": Caller}
```

- A pydantic model returned from `run` is one value, not one output per field.
- Only `ExternalFailure` and `retry_on` classes retry; attempt `n` waits `delay * 2 ** (n - 1)`. A timed-out attempt is final.
