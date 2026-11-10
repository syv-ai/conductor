# add-node reference

One module, one example per shape; the rules are in `python -m conductor.about`.

```python
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from conductor import SKIPPED, Ask, Choice, Deprecation, ExternalFailure, FromRun, Param, Policy, Series, Upgrade, node
from conductor.widgets import Switch, Textarea, TextWidget


@dataclass(frozen=True)
class Branches:                       # one output per field; a shared choice makes them a branch
    yes: Annotated[str, Param(title="If yes", exclusive_group="answer")]
    no: Annotated[str, Param(title="If no", exclusive_group="answer")]


@node("route", title="Route", description="Sends the text one way or the other.", category="logic")
def route(text: Annotated[str, Param(title="Text", widget=Textarea())], go: Annotated[bool, Param(title="Yes?", widget=Switch())]) -> Branches:
    return Branches(yes=text, no=SKIPPED) if go else Branches(yes=SKIPPED, no=text)


@node("words", title="Words", description="One row per word.", category="text")
def words(text: Annotated[str, Param(title="Text", widget=Textarea())]) -> Annotated[Series[str], Param(title="Words")]:
    return [word for word in text.split()]      # a list on a Series output: one row per item


class Caller:
    def __init__(self, name: str) -> None:
        self.name = name


@node("approve", title="Approve", description="Asks a person to approve or rewrite the proposal.", category="review")
def approve(proposal: Annotated[str, Param(title="Proposal", widget=Textarea())], who: FromRun[Caller]) -> Annotated[str, Param(title="Decision")]:
    # what is asked is what the node returns; a default proposes an answer
    raise Ask(f"{who.name}, approve?", default=proposal, widget=Textarea())


@dataclass(frozen=True)
class Review:                         # a form: one question per field; a field's default is proposed
    approved: bool
    owner: str
    note: str | None = None


@node("review", title="Review", category="review")
def review(proposal: str) -> Review:
    # several outputs: what the run knows goes by output name
    raise Ask(f"Review {proposal}", choices={"owner": [Choice("ida", "Ida B."), Choice("bo", "Bo K.")]}, widgets={"note": Textarea()})


def fetch_v1(url: Annotated[str, Param(title="URL", widget=TextWidget())]) -> Annotated[str, Param(title="Body")]: ...

@node("fetch", title="Fetch", description="Fetches a page.", category="http", policy=Policy(retries=3, retry_delay=0.5, attempt_timeout=10.0, concurrency=4, retry_on=(TimeoutError,)), replaces=node("fetch", deprecation=Deprecation(header="Use version 2"))(fetch_v1), upgrade=Upgrade(renamed_inputs={"url": "address"}))
def fetch(address: Annotated[str, Param(title="Address", widget=TextWidget())]) -> Annotated[str, Param(title="Body")]:
    if not address:
        raise ExternalFailure("no address")          # retried, like any retry_on class
    return f"<html>{address}</html>"


def columns_outputs(names: str) -> dict[str, Any]:
    if not names:
        raise ValueError("Type at least one column name.")   # a problem on the node when a graph loads, not an exception
    return {name: str for name in names.split(",")}

@node("columns", title="Columns", description="One output per column name typed in.", category="table", compute_outputs=columns_outputs)
def columns(names: Annotated[str, Param(title="Column names", widget=TextWidget())]) -> Mapping[str, Any]:
    return {name: name.upper() for name in names.split(",")}


assert approve.from_run == {"who": Caller}
```

- A pydantic model a node returns is one value, not one output per field.
- `Ask` is a `BaseException`: an `except Exception` in the node lets it through.
- Only `ExternalFailure` and `retry_on` classes retry; attempt `n` waits `retry_delay * 2 ** (n - 1)`. A timed-out attempt is final.
