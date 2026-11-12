"""A loaded graph as a host reads it: where each node runs, how each input is fed, the graph's inputs, its problems."""

from __future__ import annotations

from typing import Literal, assert_never

from conductor._internal.compiled.input_plans import (
    InputCollectsLoopResults,
    InputConstant,
    InputFromGraphInput,
    InputMissing,
    InputPlan,
    InputReadsOneValue,
    InputWrapsOneValue,
    OutputAddress,
)
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.field import FieldDescription
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


class InputSource(ConductorModel):
    """Where one input of a node gets its value on each call.

    In the graph ``lines -> shout -> join``, ``shout.line`` is wired to the list ``lines.result`` and takes one
    text, so its source is ``{"mode": "item", "series": "lines.result", "msg": "per item of lines.result"}``.
    ``shout.mark`` has only its default: ``{"mode": "constant", "series": None, "msg": "a constant"}``.

    Attributes:
        mode: One of seven.
            ``constant``: a value written in the graph record, or the input's default; the same on every call in
            every run.
            ``graph_input``: the caller gives it when a run starts.
            ``one_value``: an output of a node that is called once.
            ``item``: one item of the list ``series``; the node is called once per item.
            ``outer_value``: an output of a node outside the innermost loop this node is inside; every call of
            that loop gets the same value.
            ``collected``: every item of ``series``, as one list.
            ``missing``: no value; a problem of the graph says why.
        series: The address of a ``Series`` output. For ``item`` and ``collected``, the list the items are of.
            For ``outer_value``, the list of the loop the value's own node is inside, and ``None`` when that node
            is inside no loop. ``None`` for every other mode, and for ``collected`` when the input is wired to
            one value, which it gets as a list of one.
        msg: The same, in English.
    """

    mode: Literal["constant", "graph_input", "one_value", "item", "outer_value", "collected", "missing"]
    series: str | None = None
    msg: str


class NodeInputDescription(ConductorModel):
    """An input of a node in a loaded graph.

    Attributes:
        field: The input as its node's hooks and edges made it.
        source: Where it gets its value on each call of its node.
    """

    field: FieldDescription
    source: InputSource


class GraphNodeDescription(ConductorModel):
    """One node of a loaded graph as an editor draws it.

    Attributes:
        node: The id of the definition it uses.
        version: The version of that definition.
        found: Whether its fields are known: its definition was found at its version and its hooks gave its
            fields. ``problems`` says why when they aren't.
        problems: The problems placed on it. A node whose fields are known may hold some: an input an edge is
            refused for has the source ``missing``.
        runs: Where it runs, when its fields are known.
        inputs: Its inputs as its hooks made them, when its fields are known.
        outputs: Its outputs as its hooks made them, when its fields are known.
    """

    node: str
    version: int
    found: bool
    problems: list[Problem]
    runs: RunsOn | None = None
    inputs: list[NodeInputDescription] = []
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


def runs_on(loops: tuple[OutputAddress, ...]) -> RunsOn:
    """Where a node of ``loops`` runs."""
    return RunsOn(mode="per_item", series=str(loops[-1]), msg=f"per item of {loops[-1]}") if loops else RunsOn(mode="once", msg="once")


def input_source_of(loops: tuple[OutputAddress, ...], plan: InputPlan) -> InputSource:
    """Returns how an input with ``plan`` gets its value on each call, for a node inside ``loops``.

    In ``lines -> shout -> join``: ``shout.line`` is an ``item`` of ``lines.result``, ``shout.mark`` is a
    ``constant``, ``join.texts`` is ``collected`` from ``lines.result``, and ``lines.text`` is a ``graph_input``.
    """
    match plan:
        case InputConstant():
            return InputSource(mode="constant", msg="a constant")
        case InputFromGraphInput():
            return InputSource(mode="graph_input", msg="the graph input")
        case InputMissing():
            return InputSource(mode="missing", msg="no value: a problem of the graph says why")
        case InputCollectsLoopResults(series=series):
            return InputSource(mode="collected", series=series.address, msg=f"every item of {series.address}")
        case InputWrapsOneValue():
            return InputSource(mode="collected", msg="the one value, as a Series of one")
        case InputReadsOneValue(loops=value_loops):
            if value_loops != loops:  # the value is made outside the node's innermost loop
                made_for = f"the item of {value_loops[-1]}" if value_loops else "the graph"
                return InputSource(mode="outer_value", series=str(value_loops[-1]) if value_loops else None, msg=f"the value of {made_for}")
            if value_loops:
                return InputSource(mode="item", series=str(value_loops[-1]), msg=f"per item of {value_loops[-1]}")
            return InputSource(mode="one_value", msg="one value")
        case _:
            assert_never(plan)
