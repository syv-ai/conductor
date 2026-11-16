"""The compiled inputs of a node: for each input, where its value comes from.

The compiler works this out once, when the graph is compiled, and every run uses it.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

Each input of a compiled node is an ``Input``, which is one of five kinds:

- ``lines.text`` is a ``GraphInput``. The caller gives its value when a run starts.
- ``shout.mark`` is a ``ConstantInput``. It's ``"!"`` on every call.
- ``shout.line`` is an ``EdgeInput``. Each call gets one value through an edge, here one item of ``lines.result``.
- ``join.texts`` is a ``CollectingInput``. A call gets one value per item of a loop, as a list. Here, that's what
  ``shout`` returned for each line.
- A ``MissingInput`` gets no value, because the graph has a problem there. Such a graph never runs.

Every kind has ``field``, ``sources`` and ``loops``, so code that only needs those doesn't have to check the kind.
``sources`` has one ``Edge`` per edge into the input. An input with no edge into it has no sources and no loops.

A loop means a node is called once per item of a list. A loop is named by that list: the path of the function
node that returns it, and the output's name. That's so even when the list comes out of a subgraph node, as in
``(("loud", "split"), "result")``.

Paths are written from the graph the node is in, and a path always names a function node, never a subgraph node.
A subgraph is never called itself, only the functions inside it. One compiled subgraph may be used as a node in
many places, and the run blueprint (``run/blueprint.py``) works out its inputs in each place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, final

from conductor._internal.base import NodePath, Source
from conductor._internal.field import Field
from conductor._internal.types import Type


@final
@dataclass(frozen=True, slots=True)
class Edge:
    """What one edge into an input carries, as compiled. An input has one ``Edge`` per edge.

    In ``lines -> shout -> join``, the edge from ``lines.result`` gives ``shout`` one item of the list per call::

        Edge(Source("lines", "result"), ("lines",), "result", (), Type(str), per_item=True)

    The edge from ``shout.result`` gives ``join`` what one call of ``shout`` returned. That call was inside the
    loop over ``lines.result``::

        Edge(Source("shout", "result"), ("shout",), "result", ((("lines",), "result"),), Type(str), per_item=False)

    Say ``shout`` is inside the subgraph node ``loud``, and the edge is wired to ``loud.result``. Then the edge is
    ``Edge(Source("loud", "result"), ("loud", "shout"), "result", ...)``. An edge from the field ``email`` of the
    record ``split.result`` has ``Source("split", "result", "email")``.

    Attributes:
        wired_to: What the graph record wires the edge to: an output, and the field of the record it returns, or
            ``None`` for the whole output. Problems name the edge by it.
        node: The path of the function node that returns the output. That's ``("shout",)``, or
            ``("loud", "shout")`` for an output of the subgraph node ``loud``.
        output: The output's name on that function node.
        loops: The loops that function node is in, outermost first. Each is the node path and output of the list
            it loops over.
        type: The type of the value the edge carries.
        per_item: Whether the output is a ``Series`` that the edge gives one item of per call. If so, the node the
            edge leads to is inside the loop over it, ``(node, output)``.
    """

    wired_to: Source
    node: NodePath
    output: str
    loops: tuple[tuple[NodePath, str], ...]
    type: Type
    per_item: bool

    @property
    def loops_of_value(self) -> tuple[tuple[NodePath, str], ...]:
        """The loops a node must be in to get one value from this edge, outermost first.

        These are the loops of the function node the edge comes from. For an edge that gives one item of a
        ``Series``, the loop over that ``Series`` is added at the end.
        """
        return (*self.loops, (self.node, self.output)) if self.per_item else self.loops

    def from_outside(
        self, subgraph_path: NodePath, subgraph_loops: tuple[tuple[NodePath, str], ...]
    ) -> Edge:
        """Returns this edge of a subgraph's graph, named from the graph the subgraph node is in.

        The subgraph node's path goes in front of ``node``, and the loops follow the subgraph node's own. Say
        ``loud``'s graph has the edge from ``shout.result``, and ``loud`` is inside the loop over ``lines.result``.
        From outside, the edge's ``node`` is ``("loud", "shout")`` and its ``loops`` are
        ``((("lines",), "result"),)``.

        ``wired_to`` stays as it is, since it names the edge in the graph it was drawn in.

        Args:
            subgraph_path: The subgraph node's path in the graph it is in.
            subgraph_loops: The loops the subgraph node is in, named from that graph.
        """
        return replace(
            self,
            node=(*subgraph_path, *self.node),
            loops=loops_from_outside(subgraph_path, subgraph_loops, self.loops),
        )


def loops_from_outside(
    subgraph_path: NodePath,
    subgraph_loops: tuple[tuple[NodePath, str], ...],
    loops: tuple[tuple[NodePath, str], ...],
) -> tuple[tuple[NodePath, str], ...]:
    """Returns loops of a subgraph's graph, named from the graph the subgraph node is in.

    They come after the subgraph node's own loops, and each list's node gets the subgraph node's path in front. Say
    ``loud`` is inside the loop over ``lines.result``, and its graph has the loop over ``split.result``. From
    outside, that's ``((("lines",), "result"), (("loud", "split"), "result"))``.

    Args:
        subgraph_path: The subgraph node's path in the graph it is in.
        subgraph_loops: The loops the subgraph node is in, named from that graph.
        loops: The loops as the subgraph's graph names them.
    """
    return (*subgraph_loops, *[_list_from_outside(subgraph_path, loop) for loop in loops])


def _list_from_outside(
    subgraph_path: NodePath, series: tuple[NodePath, str]
) -> tuple[NodePath, str]:
    """Returns a list of a subgraph's graph, named from the graph the subgraph node is in.

    ``(("split",), "result")`` inside ``loud`` is ``(("loud", "split"), "result")`` from outside.

    Args:
        subgraph_path: The subgraph node's path in the graph it is in.
        series: The list as the subgraph's graph names it: the path of the node that returns it, and the output.
    """
    node, output = series
    return (*subgraph_path, *node), output


# -- how an input gets its value ----------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class ConstantInput:
    """An input that gets the same value on every call of every run.

    The value is the constant written in the graph record, or else the input's default. In the example,
    ``shout.mark`` is ``ConstantInput(mark, "!")``.

    Attributes:
        field: The input, as its node declares it.
        value: The value, already checked and converted by the input's type.
        sources: Always empty, since no edge leads to the input.
        loops: Always empty, since the value isn't made in a loop.
    """

    field: Field
    value: Any
    sources: ClassVar[tuple[Edge, ...]] = ()
    loops: ClassVar[tuple[tuple[NodePath, str], ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class MissingInput:
    """An input that gets no value, because the graph has a problem there.

    For example, ``join.texts`` wired to an output that ``shout`` doesn't have. A graph like that never runs. The
    node is still kept in the compiled graph, so an editor can draw it with this input shown as unset.

    Attributes:
        field: The input, as its node declares it.
        sources: Always empty, since no usable edge leads to the input.
        loops: Always empty.
    """

    field: Field
    sources: ClassVar[tuple[Edge, ...]] = ()
    loops: ClassVar[tuple[tuple[NodePath, str], ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class GraphInput:
    """An input whose value the caller gives when a run starts.

    In the example, ``lines.text`` is ``GraphInput(text, "text")``.

    Attributes:
        field: The input, as its node declares it.
        name: The graph input's name. The caller gives the value by this name, and the run stores it under it.
        sources: Always empty, since no edge leads to the input.
        loops: Always empty, since the value isn't made in a loop.
    """

    field: Field
    name: str
    sources: ClassVar[tuple[Edge, ...]] = ()
    loops: ClassVar[tuple[tuple[NodePath, str], ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class EdgeInput:
    """An input that gets one value through its edges on each call.

    In the example, ``shout.line`` gets one item of ``lines.result`` per call, so its ``loops`` is the loop over
    ``lines.result``.

    An ``EdgeInput`` with ``as_list`` is a ``Series`` input whose edges give one value per call, so there's no loop
    to collect. The call gets that one value as a list of one. For example, if ``join.texts`` is wired straight to
    a ``str`` output ``title.result``, ``join`` is called with ``["Title"]``. A ``Table`` input wired to records
    works the same way.

    Attributes:
        field: The input, as its node declares it.
        sources: What each edge into the input carries. With several edges, each comes from a different branch of
            one decision, and a call gets the value from the branch that was chosen.
        loops: The loops the node must be in to get the value, outermost first. These come from the edges, which
            all have the same ones.
        as_list: Whether a call gets the value as a list of one.
    """

    field: Field
    sources: tuple[Edge, ...]
    loops: tuple[tuple[NodePath, str], ...]
    as_list: bool = False


@final
@dataclass(frozen=True, slots=True)
class CollectingInput:
    """A ``Series`` input that gets one value per item of a loop, collected into a list.

    In the example, ``join.texts`` collects the loop over ``lines.result``. Its ``series`` is ``lines.result`` and
    its one source is ``shout.result``, so ``join`` is called once, with ``["A!", "B!"]``. A ``Table`` input wired
    to records collects them the same way.

    Attributes:
        field: The input, as its node declares it.
        series: The list whose loop the input collects, as the node path and the output:
            ``(("lines",), "result")``.
        loops: The loops the node must be in to get the list, outermost first. These are the loops of the node
            that returns ``series``. The loop over ``series`` itself isn't one of them.
        sources: What each edge into the input carries for one item. Several edges come from different branches
            of one decision, as for any input.
    """

    field: Field
    series: tuple[NodePath, str]
    loops: tuple[tuple[NodePath, str], ...]
    sources: tuple[Edge, ...]


#: One input of a compiled node: one of the five kinds above.
Input = ConstantInput | MissingInput | GraphInput | EdgeInput | CollectingInput
