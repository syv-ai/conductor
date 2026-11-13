"""The compiled inputs of a node: for each input, where its value comes from. Made when the graph is compiled.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

Each input of a compiled node is an ``Input``, of one of five kinds:

- ``lines.text`` is a ``GraphInput``: the caller gives the value when a run starts.
- ``shout.mark`` is a ``ConstantInput``: ``"!"`` on every call.
- ``shout.line`` is an ``EdgeInput``: each call reads one value through an edge, here one item of ``lines.result``.
  A ``Series`` input wired to a value that is not made per item of a loop is an ``EdgeInput`` with ``as_list``: a
  call gets that one value as a list of one.
- ``join.texts`` is a ``CollectingInput``: a call gets one value per item of a loop, as a list. Here that is what
  ``shout`` returned for each line.
- A ``MissingInput`` gets no value, because the graph has a problem at this input; such a graph is never run.

Every kind has ``field``, ``sources`` and ``loops``, so code that needs one of them does not ask which kind of
input it has. ``sources`` has one ``Edge`` for each edge into the input: the output the edge comes from,
and whether the edge reads the whole output or one item of it. An input no edge leads to has no sources and no
loops. A source names a node of the compiled graph by its id; the run's flat graph (``run/flat.py``) makes the
arguments a call reads from these, naming function nodes by their names (``"loud/shout"`` for ``shout`` inside
``loud``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, final

from conductor._internal.base import Path, node_path_text, place_text
from conductor._internal.field import Field
from conductor._internal.types import Type


@final
@dataclass(frozen=True, slots=True)
class Edge:
    """What one edge reads from an output, as compiled: the input it leads to holds one per edge.

    Example, in ``lines -> shout -> join``: the edge from ``lines.result`` reads one item of the list per call of
    ``shout``, so it is ``Edge(("lines", "result"), None, (), Type(str), item=True)``. The edge from ``shout.result``
    reads the value one call of ``shout`` returned, ``Edge(("shout", "result"), None, (("lines", "result"),),
    Type(str), item=False)``, with ``depth`` 1: a call of ``shout`` is found by one item index. An edge from the
    field ``email`` of the record ``split.result`` is ``Edge(("split", "result"), "email", ...)``.

    Attributes:
        source: The path of the output the edge reads. In a compiled graph it is the node's id and the output; in
            the flat graph, the function node's path and the output there (``("loud", "shout", "result")`` for
            ``shout`` inside the subgraph node ``loud``).
        field: The field of the record the output returns that the edge reads; ``None`` for the whole output.
        loops: The loops that node is inside, outermost first, each as the path of the list looped over.
        type: The type of the value the edge reads.
        item: Whether the output is a ``Series`` and the edge reads one item of it. The node the edge leads to
            is then inside the loop over this output, and an item is found by one more item index than
            ``depth``.
        depth: How many loops the node is inside: the number of item indexes that say which call of it.
        node_name: The path of the node the edge comes from, as text (``"loud/shout"``): the key a run keeps the
            node's outcomes under.
    """

    source: Path
    field: str | None
    loops: tuple[Path, ...]
    type: Type
    item: bool
    depth: int = dataclasses.field(init=False, compare=False, repr=False)
    node_name: str = dataclasses.field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", len(self.loops))
        object.__setattr__(self, "node_name", node_path_text(self.source[:-1]))

    def __str__(self) -> str:
        """The edge's source as a person reads it: ``"lines.result"``, ``"read.result.email"``,
        ``"loud/shout.result"``."""
        return place_text(self.source) if self.field is None else f"{place_text(self.source)}.{self.field}"

    @property
    def loops_of_value(self) -> tuple[Path, ...]:
        """The loops a node must be inside to read one value from this edge, outermost first: the loops of the
        node it comes from, and for an item of a ``Series`` also the loop over that ``Series``."""
        return (*self.loops, self.source) if self.item else self.loops


# -- how an input gets its value ----------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class ConstantInput:
    """An input with the same value on every call of every run: the constant written in the graph record, or
    else the input's default. In the example, ``shout.mark`` is ``ConstantInput(mark, "!")``.

    Attributes:
        field: The input, as its node declares it.
        value: The value, already read by the input's type.
        sources: Empty: no edge leads to the input.
        loops: Empty: the value is made in no loop.
    """

    field: Field
    value: Any
    sources: ClassVar[tuple[Edge, ...]] = ()
    loops: ClassVar[tuple[Path, ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class MissingInput:
    """An input that gets no value, in a graph that has a problem saying why. Example: ``join.texts`` wired to
    an output ``shout`` does not have. The graph is never run. The node is kept in the compiled graph so an
    editor can still draw it, with this input shown as unset.

    Attributes:
        field: The input, as its node declares it.
        sources: Empty: no edge the input can read leads to it.
        loops: Empty.
    """

    field: Field
    sources: ClassVar[tuple[Edge, ...]] = ()
    loops: ClassVar[tuple[Path, ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class GraphInput:
    """An input whose value the caller gives when a run starts: a graph input. In the example, ``lines.text``
    is ``GraphInput(text, "text")``.

    Attributes:
        field: The input, as its node declares it.
        name: The graph input's name, which the caller gives the value by and a run keeps it under.
        sources: Empty: no edge leads to the input.
        loops: Empty: the value is made in no loop.
    """

    field: Field
    name: str
    sources: ClassVar[tuple[Edge, ...]] = ()
    loops: ClassVar[tuple[Path, ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class EdgeInput:
    """An input that reads one value through its edges on each call. In the example, ``shout.line`` reads one item
    of ``lines.result`` per call, so its ``loops`` is the loop over ``lines.result``.

    With ``as_list``, it is a ``Series`` input (or a ``Table`` input wired to records) whose edges read one value per
    call of its node, so there is no loop to collect, and a call gets that one value as a list of one. Example:
    ``join.texts`` wired straight to an output ``title.result`` that is a ``str``: ``join`` is called with
    ``["Title"]``.

    Attributes:
        field: The input, as its node declares it.
        sources: What each edge into the input reads. With several edges, each comes from a different branch of
            one decision, and a call gets the value from the branch that was chosen.
        loops: The loops the input's node must be inside to read the value, outermost first: those of the
            source with the most loops.
        as_list: Whether a call gets the value as a list of one.
    """

    field: Field
    sources: tuple[Edge, ...]
    loops: tuple[Path, ...]
    as_list: bool = False


@final
@dataclass(frozen=True, slots=True)
class CollectingInput:
    """A ``Series`` input (or a ``Table`` input wired to records) that gets one value per item of a loop, as a
    list. In the example, ``join.texts`` collects the loop over ``lines.result``: ``series`` is ``lines.result``
    and the one source is ``shout.result``, so ``join`` is called once, with ``["A!", "B!"]``.

    Attributes:
        field: The input, as its node declares it.
        series: The list whose loop the input collects: ``lines.result``.
        loops: The loops the input's node must be inside to read the list, outermost first: the loops the node
            that returns ``series`` is inside. The loop over ``series`` itself is not one of them.
        sources: What each edge into the input reads for one item. Several edges come from different branches
            of one decision, as for any input.
    """

    field: Field
    series: Path
    loops: tuple[Path, ...]
    sources: tuple[Edge, ...]


#: How one input of a node gets its value.
Input = ConstantInput | MissingInput | GraphInput | EdgeInput | CollectingInput
