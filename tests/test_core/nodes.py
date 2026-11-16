"""Sample nodes that several test files share, written the way a node author would write them."""

import os
from dataclasses import dataclass
from io import StringIO
from typing import Annotated, Literal, TypedDict

import polars as pl
from conductor import SKIPPED, Ask, NodeRegistry, Param, Series, Table


class Invoice(TypedDict):
    name: str
    email: str | None
    amount: int


def note_call(name: str) -> None:
    if "CALLS" in os.environ:
        with open(os.environ["CALLS"], "a") as calls:
            calls.write(f"{name}\n")


def registry_of(*definitions):
    """Returns a new registry with just these node definitions in it."""
    registry = NodeRegistry()
    registry.add(*definitions)
    return registry


invoices = NodeRegistry(category="invoices", tags=("finance",))


@invoices.node("read-csv", title="Read csv", description="For a test.")
def read_csv(text: str) -> Table[Invoice]:
    return pl.read_csv(StringIO(text))


@invoices.node("approve", title="Approve", description="For a test.", asks=True)
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
    """Returns where a waiting call is, like ``node@item``.

    A waiting call's id ends with ``~`` and a fingerprint of its question. This drops that part, so a test can
    compare places without caring about the question.
    """
    return waiting.id.partition("~")[0]


def calls_without_outcome(run):
    """Returns the nodes of ``run`` that still have work left, such as ``["t"]``.

    A node has work left when one of its calls has no outcome yet, or when it isn't known yet how many calls it will
    have. The counts come from ``run.progress``. A call waiting for a person counts as having an outcome. So a
    complete run returns an empty list.
    """
    return [
        name
        for name, count in run.progress.nodes.items()
        if count.total is None or count.finished + count.waiting < count.total
    ]
