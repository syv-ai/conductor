"""A loaded graph as a host reads it: where each node runs, how each input is fed, the graph's inputs, its problems."""

from __future__ import annotations

from typing import Literal

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.fields import FieldDescription
from conductor._internal.loaded_graph.input_plans import (
    InputGathersItems,
    InputGathersOneValue,
    InputNotFed,
    InputPlan,
    InputReadsOneValue,
    Scope,
)
from conductor._internal.problem import Problem


class RunsOn(ConductorModel):
    """Where a node runs.

    Attributes:
        mode: ``once``, or ``per_item`` of a Series.
        series: The address of the Series it runs per item of; ``None`` for a node that runs once.
        msg: The same, in English.
    """

    mode: Literal["once", "per_item"]
    series: str | None = None
    msg: str


class Feed(ConductorModel):
    """How an input is fed on each call of its node.

    Attributes:
        mode: ``one_value`` for a value typed in, a default or the caller's; ``per_item`` for one item of
            ``series``; ``outer_value`` for a value made outside the items the node runs per; ``gathered`` for
            every item of ``series`` as one list; ``unset`` for an input with no value to give, which a problem
            of the graph says why.
        series: The address of the Series the mode names: for ``outer_value``, the one whose item the value
            belongs to. ``None`` for ``one_value``, for an ``outer_value`` made once in the graph, and for the
            one value of a source that runs once, ``gathered`` as a Series of one.
        msg: The same, in English.
    """

    mode: Literal["one_value", "per_item", "outer_value", "gathered", "unset"]
    series: str | None = None
    msg: str


class PlacedFieldDescription(ConductorModel):
    """An input of a node in a loaded graph.

    Attributes:
        field: The input as its node's hooks and edges made it.
        feed: How it is fed on each call of its node.
    """

    field: FieldDescription
    feed: Feed


class GraphNodeDescription(ConductorModel):
    """One node of a loaded graph as an editor draws it.

    Attributes:
        node: The id of the definition it uses.
        version: The version of that definition.
        loaded: Whether its fields are known: its definition was found at its version and its hooks gave its
            fields. ``problems`` says why when they aren't.
        problems: The problems placed on it. A node whose fields are known may hold some: an input an edge is
            refused for is fed ``unset``.
        runs: Where it runs, when its fields are known.
        inputs: Its inputs as its hooks made them, when its fields are known.
        outputs: Its outputs as its hooks made them, when its fields are known.
    """

    node: str
    version: int
    loaded: bool
    problems: list[Problem]
    runs: RunsOn | None = None
    inputs: list[PlacedFieldDescription] = []
    outputs: list[FieldDescription] = []


class GraphInput(ConductorModel):
    """An open input of a first node, a value of ``graph.inputs``: what a caller fills.

    Attributes:
        address: Where the caller's value goes, ``node.input``.
        node_id: The node whose input it is.
        field: The input. Its ``default`` is what it holds when the caller gives nothing (the record's default,
            else the node's), as JSON, and it is ``required`` when there is none.
    """

    address: str
    node_id: str
    field: FieldDescription


class GraphDescription(ConductorModel):
    """A loaded graph as an editor draws it.

    Attributes:
        nodes: Each node of the record, by id.
        inputs: The graph's inputs, by name.
        outputs: The graph's outputs, as addresses.
        problems: Every problem of the graph.
    """

    nodes: dict[str, GraphNodeDescription]
    inputs: dict[str, GraphInput]
    outputs: list[str]
    problems: list[Problem]


def runs_on(scope: Scope) -> RunsOn:
    """Where a node of ``scope`` runs."""
    return RunsOn(mode="per_item", series=str(scope[-1]), msg=f"per item of {scope[-1]}") if scope else RunsOn(mode="once", msg="once")


def feed_of(scope: Scope, plan: InputPlan) -> Feed:
    """How ``plan`` feeds its input on each call of a node that runs in ``scope``."""
    match plan:
        case InputGathersItems(series=series):
            return Feed(mode="gathered", series=series.address, msg=f"every item of {series.address}")
        case InputGathersOneValue():
            return Feed(mode="gathered", msg="the one value, as a Series of one")
        case InputReadsOneValue(scope=fed_in) if fed_in != scope:
            return Feed(mode="outer_value", series=str(fed_in[-1]) if fed_in else None, msg=f"the value of {f'the item of {fed_in[-1]}' if fed_in else 'the graph'}")
        case InputReadsOneValue(scope=fed_in) if fed_in:
            return Feed(mode="per_item", series=str(fed_in[-1]), msg=f"per item of {fed_in[-1]}")
        case InputNotFed():
            return Feed(mode="unset", msg="no value: a problem of the graph says why")
        case _:
            return Feed(mode="one_value", msg="one value")
