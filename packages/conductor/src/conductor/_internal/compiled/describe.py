"""What a host is shown of a compiled graph by ``describe()``: every node with its inputs and outputs, whether each
node is called once or once per item, where each input gets its value, the graph inputs and outputs, and the
problems.

In the graph ``lines -> shout -> join``, ``shout`` is called once per item of ``lines.result``, so its ``runs`` is
``{"mode": "per_item", "series": "lines.result", "msg": "per item of lines.result"}``.
"""

from __future__ import annotations

from typing import Literal, assert_never

from conductor._internal.base import ConductorModel
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
from conductor._internal.field import FieldDescription
from conductor._internal.problem import Problem


class RunsOn(ConductorModel):
    """Whether a node is called once or once per item of a list.

    Attributes:
        mode: ``once``, or ``per_item`` of a ``Series`` output.
        series: The address of the ``Series`` output whose items the node is called for: the list of the innermost
            loop the node is inside. ``None`` for a node called once.
        msg: The same in English, for a person to read.
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
            ``one_value``: an output of a node that is called once, read by a node that is called once too.
            ``item``: one item of the list ``series``; the node is called once per item.
            ``outer_value``: an output of a node outside the innermost loop this node is inside; every call of
            that loop gets the same value.
            ``collected``: every item of ``series``, as one list.
            ``missing``: no value; a problem of the graph says why.
        series: The address of a ``Series`` output. For ``item`` and ``collected``, the list the items are of.
            For ``outer_value``, the list of the loop the value's own node is inside, and ``None`` when that node
            is inside no loop. ``None`` for every other mode, and for ``collected`` when the input is wired to
            one value, which it gets as a list of one.
        msg: The same in English, for a person to read.
    """

    mode: Literal["constant", "graph_input", "one_value", "item", "outer_value", "collected", "missing"]
    series: str | None = None
    msg: str


class NodeInputDescription(ConductorModel):
    """An input of a node in a compiled graph.

    Attributes:
        field: The input as compiled: an input a hook added is here too, and a type variable is replaced by the
            type the input's edges read.
        source: Where it gets its value on each call of its node.
    """

    field: FieldDescription
    source: InputSource


class GraphNodeDescription(ConductorModel):
    """One node of a compiled graph, as an editor draws it.

    Attributes:
        node: The id of the node definition it uses (``"shout"``).
        version: The version of that node definition.
        found: Whether the node was compiled: its definition was found at its version, and its hooks returned its
            inputs and outputs. When it was not, ``problems`` says why and the node has no ``runs``, ``inputs``
            or ``outputs``.
        problems: The problems on this node. A compiled node can have some too: an input that gets no value
            because of a problem has the source ``missing``.
        runs: Whether it is called once or once per item; ``None`` when it was not compiled.
        inputs: Its inputs as compiled; empty when it was not compiled.
        outputs: Its outputs as compiled; empty when it was not compiled.
    """

    node: str
    version: int
    found: bool
    problems: list[Problem]
    runs: RunsOn | None = None
    inputs: list[NodeInputDescription] = []
    outputs: list[FieldDescription] = []


class GraphInput(ConductorModel):
    """A graph input: an input of a node that no edge leads to, which the caller gives a value for when a run
    starts. The values of ``graph.inputs`` are these. In the example, ``lines.text``.

    Attributes:
        address: The key the caller's value is kept under, ``node.input``.
        node_id: The node whose input it is.
        field: The input. Its ``default`` is the value used when the caller gives none (the graph record's
            default, else the node's), as JSON, and it is ``required`` when there is none.
    """

    address: str
    node_id: str
    field: FieldDescription


class GraphDescription(ConductorModel):
    """A compiled graph as an editor draws it.

    Attributes:
        nodes: Each node of the graph record, by id.
        inputs: The graph inputs, by name: the input's own name, or its address (``node.input``) when two graph
            inputs share a name.
        outputs: The graph outputs, as addresses (``"join.result"``): every output of every node that no edge
            leaves.
        problems: Every problem of the graph, those on nodes included.
    """

    nodes: dict[str, GraphNodeDescription]
    inputs: dict[str, GraphInput]
    outputs: list[str]
    problems: list[Problem]


def runs_on(loops: tuple[OutputAddress, ...]) -> RunsOn:
    """Returns whether a node inside ``loops`` is called once or once per item: ``per_item`` of the list of the
    innermost loop, or ``once`` when ``loops`` is empty."""
    return RunsOn(mode="per_item", series=str(loops[-1]), msg=f"per item of {loops[-1]}") if loops else RunsOn(mode="once", msg="once")


def input_source_of(loops: tuple[OutputAddress, ...], plan: InputPlan) -> InputSource:
    """Returns how an input with ``plan`` gets its value on each call, for a node inside ``loops``.

    In ``lines -> shout -> join``: ``shout.line`` is an ``item`` of ``lines.result``, ``shout.mark`` is a
    ``constant``, ``join.texts`` is ``collected`` from ``lines.result``, and ``lines.text`` is a ``graph_input``.
    An input of a node inside the loop over ``lines.result``, wired to an output of a node called once, is an
    ``outer_value``.
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
            if value_loops != loops:  # the value's node is inside fewer loops than this node: the same value for every item
                if value_loops:
                    return InputSource(mode="outer_value", series=str(value_loops[-1]), msg=f"the value of the item of {value_loops[-1]}")
                return InputSource(mode="outer_value", msg="one value, the same for every item")
            if value_loops:
                return InputSource(mode="item", series=str(value_loops[-1]), msg=f"per item of {value_loops[-1]}")
            return InputSource(mode="one_value", msg="one value")
        case _:
            assert_never(plan)
