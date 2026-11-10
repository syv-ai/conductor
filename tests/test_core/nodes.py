"""Nodes the programs share: what a node author writes."""
import io
import os
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import polars as pl
from conductor import SKIPPED, Ask, NodeRegistry, Param, Series, Table


class Invoice(TypedDict):
    name: str
    email: str
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


@invoices.node("read-csv")
def read_csv(text: str) -> Table[Invoice]:
    return pl.read_csv(io.StringIO(text))


@invoices.node("approve")
def approve(name: str, amount: int) -> Literal["yes", "no", "auto"]:
    note_call(f"approve {name}")
    if amount < 1000:
        return "auto"
    raise Ask(f"Approve {name} for {amount}?")


@invoices.node("summary")
def summary(decisions: Series[str]) -> str:
    return " | ".join(decisions)


@invoices.node("mail")
def mail(email: str) -> str:
    note_call(f"mail {email}")
    return f"sent to {email}"


text = NodeRegistry(category="text")


@text.node("upper")
def upper(text: str) -> str:
    return text.upper()


@text.node("lower")
def lower(text: str) -> str:
    return text.lower()


@text.node("tail")
def tail(text: str) -> str:
    return text + "."


@text.node("repeat")
def repeat(text: str, times: Annotated[int, Param(title="Times", ge=0)] = 1) -> str:
    return text * times


@text.node("split")
def split(text: str, sep: str = ",") -> Series[str]:
    return text.split(sep)


@text.node("join")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class Length:
    long: str
    short: str


@text.node("route")
def route(text: str) -> Length:
    return Length(text, SKIPPED) if len(text) > 3 else Length(SKIPPED, text)


registry = NodeRegistry()
registry.add_subregistry(invoices)
registry.add_subregistry(text)
