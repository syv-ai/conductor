"""Sample nodes that several test files share, written the way a node author would write them."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
from conductor import SKIPPED, Ask, InputEntry, NodeRegistry, Param, Series, Table
from pydantic import TypeAdapter

#: Reads an input's entry from its JSON form, as a graph record reads it: ``input_entry({"constant": "?"})``.
input_entry = TypeAdapter(InputEntry).validate_python


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
    long: str = SKIPPED
    short: str = SKIPPED


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


def jsonb(text: str) -> str:
    """Returns ``text`` the way Postgres ``jsonb`` gives it back, with every object's keys reordered.

    ``jsonb`` puts shorter keys first, then sorts keys of the same length bytewise.
    """

    def order(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: order(value[k])
                for k in sorted(value, key=lambda k: (len(k.encode()), k.encode()))
            }
        return [order(v) for v in value] if isinstance(value, list) else value

    return json.dumps(order(json.loads(text)))


def printed_in_processes(code: str, seeds: tuple[str, ...]) -> set[str]:
    """Runs ``code`` in a new Python process per hash seed in ``seeds``, and returns what the processes printed.

    Each process must exit cleanly. One text in the set means every process printed the same.
    """
    printed = set()
    for seed in seeds:
        process = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stderr
        printed.add(process.stdout)
    return printed
