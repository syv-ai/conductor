"""Nodes the programs share: what a node author writes."""
import os
from dataclasses import dataclass
from io import StringIO
from typing import Annotated, Literal, TypedDict

import polars as pl
from conductor import SKIPPED, Ask, NodeRegistry, Param, Series, Table
from conductor._internal.outcomes import all_item_indexes


class Invoice(TypedDict):
    name: str
    email: str | None
    amount: int


def note_call(name: str) -> None:
    if "CALLS" in os.environ:
        with open(os.environ["CALLS"], "a") as calls:
            calls.write(f"{name}\n")




def registry_of(*definitions):
    """A registry holding exactly these nodes."""
    held = NodeRegistry()
    held.add(*definitions)
    return held


invoices = NodeRegistry(category="invoices", tags=("finance",))


@invoices.node("read-csv", title="Read csv", description="For a test.")
def read_csv(text: str) -> Table[Invoice]:
    return pl.read_csv(StringIO(text))


@invoices.node("approve", title="Approve", description="For a test.")
def approve(name: str, amount: int) -> Literal["yes", "no", "auto"]:
    note_call(f"approve {name}")
    if amount < 1000:
        return "auto"
    raise Ask(f"Approve {name} for {amount}?")


@invoices.node("summary", title="Summary", description="For a test.")
def summary(decisions: Series[str]) -> str:
    return " | ".join(decisions)


@invoices.node("mail", title="Mail", description="For a test.")
def mail(email: str) -> str:
    note_call(f"mail {email}")
    return f"sent to {email}"


text = NodeRegistry(category="text")


@text.node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


@text.node("lower", title="Lower", description="For a test.")
def lower(text: str) -> str:
    return text.lower()


@text.node("tail", title="Tail", description="For a test.")
def tail(text: str) -> str:
    return text + "."


@text.node("repeat", title="Repeat", description="For a test.")
def repeat(text: str, times: Annotated[int, Param(title="Times", ge=0)] = 1) -> str:
    return text * times


@text.node("split", title="Split", description="For a test.")
def split(text: str, sep: str = ",") -> Series[str]:
    return text.split(sep)


@text.node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class Length:
    long: str
    short: str


@text.node("route", title="Route", description="For a test.")
def route(text: str) -> Length:
    return Length(text, SKIPPED) if len(text) > 3 else Length(SKIPPED, text)


registry = NodeRegistry()
registry.add_subregistry(invoices)
registry.add_subregistry(text)


def place(waiting):
    """A waiting unit's place, ``node@item``: its id without the fingerprint of its question."""
    return waiting.id.partition("~")[0]


def units_missing(run):
    """Each instance of a node of ``run`` that holds no unit, as its node's path and the instance, with a node
    whose all_item_indexes aren't all known as its path alone. A complete run has none."""
    missing = []
    for node in run.graph._flat_function_nodes:
        made, known = all_item_indexes(run._outcomes, node.loops, ())
        units = run._outcomes[node.name]
        missing += [(node.node_path, key) for key in made if key not in units] + ([] if known else [node.node_path])
    return missing
