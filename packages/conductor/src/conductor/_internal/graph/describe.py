"""What ``graph.describe()`` shows a host about a compiled graph.

An editor uses it to draw the graph. It lists every node with its inputs and outputs. It says whether each node is
called once or once per item of a list, and where each input gets its value. It also has the graph's inputs,
outputs and problems.

Take the graph ``lines -> shout -> join``, where ``lines`` returns a list of lines. ``shout`` is called once per
item of ``lines.result``, so its ``per_item_of`` is ``"lines.result"``. ``lines`` and ``join`` are each called
once, so theirs is ``None``.
"""

from __future__ import annotations

from typing import Literal, assert_never

from conductor._internal.base import ConductorModel, NodePath, place_text
from conductor._internal.field import FieldDescription
from conductor._internal.problem import Problem

from .input import (
    CollectingInput,
    ConstantInput,
    EdgeInput,
    GraphInput,
    Input,
    MissingInput,
)


class InputSource(ConductorModel):
    """Where one input of a node gets its value on each call.

    In the graph ``lines -> shout -> join``, ``shout.line`` is wired to the list ``lines.result`` but takes one
    text. So each call gets one item, and its source is ``{"mode": "item", "series": "lines.result"}``.
    ``shout.mark`` has only its default, so its source is ``{"mode": "constant", "series": None}``.

    Attributes:
        mode: One of seven:
            ``constant``: a value written in the graph record, or the input's default. It's the same on every call
            of every run.
            ``graph_input``: the caller gives it when a run starts.
            ``one_value``: an output of a node that's called once, going to a node that's called once too.
            ``item``: one item of the list ``series``. The node is called once per item.
            ``outer_value``: an output of a node outside the innermost loop this node is in. Every call in that
            loop gets the same value.
            ``collected``: every item of ``series``, as one list.
            ``missing``: no value. One of the graph's problems says why.
        series: A ``Series`` output, written like ``"lines.result"`` (see ``innermost_series``).
            For ``item`` and ``collected``, it's the list the items come from.
            For ``outer_value``, it's the list of the loop the value's own node is in, or ``None`` when that node
            is in no loop, so the value is the same for every call.
            For every other mode, it's ``None``.
    """

    mode: Literal[
        "constant", "graph_input", "one_value", "item", "outer_value", "collected", "missing"
    ]
    series: str | None = None


class NodeInputDescription(ConductorModel):
    """One input of a node in a compiled graph, and where it gets its value.

    Attributes:
        field: The input as compiled. Inputs a hook added are included, and a type variable is replaced by the
            type the input's edges carry.
        source: Where the input gets its value on each call of its node.
    """

    field: FieldDescription
    source: InputSource


class GraphNodeDescription(ConductorModel):
    """One node of a compiled graph, as an editor draws it.

    Attributes:
        node: The id of the node definition it uses, like ``"shout"``.
        version: The version of that definition.
        found: Whether the node compiled. For that, its definition must be found at its version, and its hooks
            must return its inputs and outputs. When it didn't compile, ``problems`` says why, ``per_item_of`` is
            ``None``, and ``inputs`` and ``outputs`` are empty.
        problems: The problems on this node. A node that compiled can have some too. For example, an input that
            gets no value because of a problem has the source ``missing``.
        per_item_of: The ``Series`` output this node is called once per item of, or ``None`` for a node called
            once. It's the list of the innermost loop the node is in, like ``"lines.result"``. A list returned
            inside the subgraph node ``per-word`` is written ``"per-word/split.result"``.
            It's also ``None`` for a node that didn't compile. Whether an edge from a list puts a node in a loop
            depends on the type of the input it reaches: a ``Series[str]`` input collects the list, and a ``str``
            input loops over it. A node that didn't compile has no input types to go by.
        inputs: Its inputs as compiled. Empty when it didn't compile.
        outputs: Its outputs as compiled. Empty when it didn't compile.
    """

    node: str
    version: int
    found: bool
    problems: list[Problem]
    per_item_of: str | None = None
    inputs: list[NodeInputDescription] = []
    outputs: list[FieldDescription] = []


class GraphDescription(ConductorModel):
    """A compiled graph as an editor draws it.

    Attributes:
        nodes: Every node of the graph record, by id.
        inputs: The graph inputs, by name. Each is the field a caller gives a value for. Its ``default`` is the
            record's default, as JSON, and it's ``required`` when the record gives none.
        outputs: The graph outputs, by name, sorted. Each is the field a complete run gives (see ``graph.outputs``).
        problems: Every problem of the graph, including those on nodes.
    """

    nodes: dict[str, GraphNodeDescription]
    inputs: dict[str, FieldDescription]
    outputs: dict[str, FieldDescription]
    problems: list[Problem]


def innermost_series(loops: tuple[tuple[NodePath, str], ...]) -> str | None:
    """Returns the list of the innermost of ``loops``, written as text, or ``None`` when there are no loops.

    The text is the node path joined by ``/``, then a dot and the output's name. ``"lines.result"`` is a list in
    this graph. ``"per-word/split.result"`` is the output ``result`` of the node ``split``, inside the subgraph
    node ``per-word``. No edge of this graph can name a list inside a subgraph, and it isn't a key of
    ``describe().nodes``.
    """
    return place_text(*loops[-1]) if loops else None


def input_source_of(loops: tuple[tuple[NodePath, str], ...], node_input: Input) -> InputSource:
    """Returns where an input gets its value on each call, for a node inside ``loops``.

    In ``lines -> shout -> join``, ``shout.line`` is an ``item`` of ``lines.result`` and ``shout.mark`` is a
    ``constant``. ``join.texts`` is ``collected`` from ``lines.result``, and ``lines.text`` is a ``graph_input``.
    """
    match node_input:
        case ConstantInput():
            return InputSource(mode="constant")
        case GraphInput():
            return InputSource(mode="graph_input")
        case MissingInput():
            return InputSource(mode="missing")
        case CollectingInput(series=series):
            return InputSource(mode="collected", series=innermost_series((series,)))
        case EdgeInput(loops=value_loops):
            # the value's node is in fewer loops than this node, so every item gets the same value
            if value_loops != loops:
                return InputSource(mode="outer_value", series=innermost_series(value_loops))
            if value_loops:
                return InputSource(mode="item", series=innermost_series(value_loops))
            return InputSource(mode="one_value")
        case _:
            assert_never(node_input)
