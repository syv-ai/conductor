"""What a host is shown of a compiled graph by ``describe()``: every node with its inputs and outputs, whether each
node is called once or once per item, where each input gets its value, the graph inputs and outputs, and the
problems.

In the graph ``lines -> shout -> join``, ``shout`` is called once per item of ``lines.result``, so its ``per_item_of``
is ``"lines.result"``; ``lines`` and ``join`` are called once, and theirs is ``None``.
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

    In the graph ``lines -> shout -> join``, ``shout.line`` is wired to the list ``lines.result`` and takes one
    text, so its source is ``{"mode": "item", "series": "lines.result"}``. ``shout.mark`` has only its default:
    ``{"mode": "constant", "series": None}``. A ``collected`` input with ``series`` ``None`` is wired to one value
    and gets it as a list of one; an ``outer_value`` with ``series`` ``None`` is the output of a node called once,
    the same value for every call.

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
        series: A ``Series`` output, as ``innermost_series`` writes it: ``"lines.result"``. For ``item`` and
            ``collected``, the list the items are of.
            For ``outer_value``, the list of the loop the value's own node is inside, and ``None`` when that node
            is inside no loop. ``None`` for every other mode, and for ``collected`` when the input is wired to
            one value, which it gets as a list of one.
    """

    mode: Literal["constant", "graph_input", "one_value", "item", "outer_value", "collected", "missing"]
    series: str | None = None


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
            inputs and outputs. When it was not, ``problems`` says why, its ``per_item_of`` is ``None`` and
            its ``inputs`` and ``outputs`` are empty.
        problems: The problems on this node. A compiled node can have some too: an input that gets no value
            because of a problem has the source ``missing``.
        per_item_of: The ``Series`` output the node is called once per item of: the list of the innermost loop
            the node is inside, as ``innermost_series`` writes it (``"lines.result"``, or
            ``"per-word/split.result"`` for a list returned inside the subgraph node ``per-word``). ``None`` for a node called once,
            and for a node that was not compiled (``found`` false): its definition or version is not in the
            registry, its id is not a name, a hook failed, or it is a subgraph that contains itself or has problems
            of its own. Whether an edge from a list puts a node inside a loop depends on the type of the input it
            reaches (``Series[str]`` collects it, ``str`` loops over it), and those types are what such a node lacks.
        inputs: Its inputs as compiled; empty when it was not compiled.
        outputs: Its outputs as compiled; empty when it was not compiled.
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
        nodes: Each node of the graph record, by id.
        inputs: The graph inputs, by name, each as the field a caller gives a value for: its ``default`` is the
            record's, as JSON, and it is ``required`` when the record gives none.
        outputs: The names of the graph outputs (``["result"]``).
        problems: Every problem of the graph, those on nodes included.
    """

    nodes: dict[str, GraphNodeDescription]
    inputs: dict[str, FieldDescription]
    outputs: list[str]
    problems: list[Problem]


def innermost_series(loops: tuple[tuple[NodePath, str], ...]) -> str | None:
    """Returns the list of the innermost of ``loops`` as text, or ``None`` when ``loops`` is empty.

    The text is the output's node path joined by ``/``, a dot and the output's name: ``"lines.result"`` for a
    list in this graph, and ``"per-word/split.result"`` for the output ``result`` of the node ``split`` inside
    the subgraph node ``per-word``. A list inside a subgraph is a place inside it: no edge of this graph can
    name it, and it is not a key of ``describe().nodes``.
    """
    return place_text(*loops[-1]) if loops else None


def input_source_of(loops: tuple[tuple[NodePath, str], ...], node_input: Input) -> InputSource:
    """Returns how an input with ``node_input`` gets its value on each call, for a node inside ``loops``.

    In ``lines -> shout -> join``: ``shout.line`` is an ``item`` of ``lines.result``, ``shout.mark`` is a
    ``constant``, ``join.texts`` is ``collected`` from ``lines.result``, and ``lines.text`` is a ``graph_input``.
    An input of a node inside the loop over ``lines.result``, wired to an output of a node called once, is an
    ``outer_value``.
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
        case EdgeInput(as_list=True):
            return InputSource(mode="collected")
        case EdgeInput(loops=value_loops):
            if value_loops != loops:  # the value's node is inside fewer loops than this node: the same value for every item
                return InputSource(mode="outer_value", series=innermost_series(value_loops))
            if value_loops:
                return InputSource(mode="item", series=innermost_series(value_loops))
            return InputSource(mode="one_value")
        case _:
            assert_never(node_input)
