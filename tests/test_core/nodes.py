"""Nodes the programs share: what a node author writes."""
import io
import os
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import polars as pl
from conductor import SKIPPED, Ask, NodeRegistry, Param, Series, Table
from conductor._internal.units import instances
from pydantic_core import core_schema


class ReadAsItsBuiltin:
    """What a host writes on its subclass of a builtin, ``class Text(ReadAsItsBuiltin, str)``, so pydantic reads a
    value as the builtin and then makes it the subclass."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_after_validator_function(cls, handler(cls.__mro__[-2]))


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


@invoices.node("read-csv", title="Read csv", description="")
def read_csv(text: str) -> Table[Invoice]:
    return pl.read_csv(io.StringIO(text))


@invoices.node("approve", title="Approve", description="")
def approve(name: str, amount: int) -> Literal["yes", "no", "auto"]:
    note_call(f"approve {name}")
    if amount < 1000:
        return "auto"
    raise Ask(f"Approve {name} for {amount}?")


@invoices.node("summary", title="Summary", description="")
def summary(decisions: Series[str]) -> str:
    return " | ".join(decisions)


@invoices.node("mail", title="Mail", description="")
def mail(email: str) -> str:
    note_call(f"mail {email}")
    return f"sent to {email}"


text = NodeRegistry(category="text")


@text.node("upper", title="Upper", description="")
def upper(text: str) -> str:
    return text.upper()


@text.node("lower", title="Lower", description="")
def lower(text: str) -> str:
    return text.lower()


@text.node("tail", title="Tail", description="")
def tail(text: str) -> str:
    return text + "."


@text.node("repeat", title="Repeat", description="")
def repeat(text: str, times: Annotated[int, Param(title="Times", ge=0)] = 1) -> str:
    return text * times


@text.node("split", title="Split", description="")
def split(text: str, sep: str = ",") -> Series[str]:
    return text.split(sep)


@text.node("join", title="Join", description="")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class Length:
    long: str
    short: str


@text.node("route", title="Route", description="")
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
    whose instances aren't all known as its path alone. A complete run has none."""
    missing = []
    for node in run.graph._tree:
        made, known = instances(run._units, node.maps, ())
        units = run._units[node.name]
        missing += [(node.path, key) for key in made if key not in units] + ([] if known else [node.path])
    return missing
