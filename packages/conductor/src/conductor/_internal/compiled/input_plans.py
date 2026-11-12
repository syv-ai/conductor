"""How each input of a node gets its value: one input plan per input, made when the graph is compiled.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

Its four inputs have four kinds of plan:

- ``lines.text`` is an ``InputFromGraphInput``: the caller gives the value when a run starts.
- ``shout.mark`` is an ``InputConstant``: ``"!"`` on every call.
- ``shout.line`` is an ``InputReadsOneValue``: each call reads one value through an edge, here one item of
  ``lines.result``.
- ``join.texts`` is an ``InputCollectsLoopResults``: a call gets one value per item of a loop, as a list. Here
  that is what ``shout`` returned for each line.

There are two more kinds. An ``InputWrapsOneValue`` is a ``Series`` input wired to a value that is not made per
item of a loop: a call gets that one value as a list of one. An ``InputMissing`` gets no value, because the
graph has a problem at this input; such a graph is never run.

Every kind has ``field``, ``sources`` and ``loops``, so code that needs one of them does not ask which kind of
plan it has. ``sources`` has one ``EdgeSource`` for each edge into the input: the output the edge comes from,
and whether the edge reads the whole output or one item of it. A plan no edge leads to has no sources and no
loops. The three kinds an edge leads to also have ``value(reader, item_indexes)``, which reads the input's
value for one call from the outcomes of a run.

The same classes are used twice. The compiler makes the plans of a compiled graph, and there a source names a
node of that graph by its id. ``flat_nodes.py`` makes the plans a run reads from those: there a source names the
function node that returns the output, by its name (``"loud/shout"`` for ``shout`` inside the subgraph
``loud``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, final

from conductor._internal.field import Field
from conductor._internal.markers import SKIPPED
from conductor._internal.outcomes import ItemIndexes, NotReady
from conductor._internal.value_type import Type

if TYPE_CHECKING:
    from conductor._internal.compiled.flat_nodes import FlatFunctionNode
    from conductor._internal.run.arguments import ResultReader


class OutputAddress(NamedTuple):
    """One output of one node, named in an address as ``node.output``.

    Attributes:
        node_id: The node.
        output: The output's name.
    """

    node_id: str
    output: str

    def __str__(self) -> str:
        return f"{self.node_id}.{self.output}"


# -- what an edge reads ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutputInLoops:
    """One output of one node, with the loops that node is inside.

    Example, in ``lines -> shout -> join``: ``shout`` is called once per item of ``lines.result``, so its output is
    ``OutputInLoops("shout", "result", loops=(OutputAddress("lines", "result"),))``, with ``depth`` 1. The output
    has one value per call of ``shout``, and a call is found by ``depth`` item indexes.

    Attributes:
        node_id: The node. In a compiled graph's own plan it is the node's id. In the flat graph it is the
            name of the function node (``"loud/shout"`` inside a subgraph).
        output: The output's name.
        loops: The loops the node is inside, outermost first, each as the output that is the list looped over.
        depth: How many loops the node is inside: the number of item indexes that say which call of it.
    """

    node_id: str
    output: str
    loops: tuple[OutputAddress, ...]
    depth: int = dataclasses.field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", len(self.loops))

    @property
    def address(self) -> str:
        """The address an edge names it by, ``node.output``."""
        return f"{self.node_id}.{self.output}"


@final
@dataclass(frozen=True, slots=True)
class EdgeSource(OutputInLoops):
    """What one edge reads from an output.

    Example, in ``lines -> shout -> join``: the edge from ``lines.result`` reads one item of the list per call of
    ``shout``, so it is ``EdgeSource("lines", "result", (), None, Type(str), item=True)``. The edge from
    ``shout.result`` reads the value one call of ``shout`` returned, so it has ``item=False``.

    Attributes:
        record_field: The field of a record that the edge reads (``email`` in ``split.result.email``); ``None``
            when it reads the whole value.
        type: The type of the value the edge reads.
        item: Whether the output is a ``Series`` and the edge reads one item of it. The node the edge leads to
            is then inside the loop over this output, and an item is found by one more item index than
            ``depth``.
    """

    record_field: str | None
    type: Type
    item: bool

    @property
    def address(self) -> str:
        """The address an edge names it by, ``node.output`` or ``node.output.field``."""
        return f"{self.node_id}.{self.output}.{self.record_field}" if self.record_field else f"{self.node_id}.{self.output}"

    @property
    def loops_of_value(self) -> tuple[OutputAddress, ...]:
        """The loops a node must be inside to read one value from this source, outermost first: the loops of
        the source's node, and for an item of a ``Series`` also the loop over that ``Series``."""
        return (*self.loops, OutputAddress(self.node_id, self.output)) if self.item else self.loops


# -- how an input gets its value ----------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class InputConstant:
    """An input with the same value on every call of every run: the constant written in the graph record, or
    else the input's default. In the example, ``shout.mark`` is ``InputConstant(mark, "!")``.

    Attributes:
        field: The input, as its node declares it.
        value: The value, already read by the input's type.
        sources: Empty: no edge leads to the input.
        loops: Empty: the value is made in no loop.
    """

    field: Field
    value: Any
    sources: ClassVar[tuple[EdgeSource, ...]] = ()
    loops: ClassVar[tuple[OutputAddress, ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class InputMissing:
    """An input that gets no value, in a graph that has a problem saying why. Example: ``join.texts`` wired to
    an output ``shout`` does not have. The graph is never run. The node is kept in the compiled graph so an
    editor can still draw it, with this input shown as unset.

    Attributes:
        field: The input, as its node declares it.
        sources: Empty: no edge the input can read leads to it.
        loops: Empty.
    """

    field: Field
    sources: ClassVar[tuple[EdgeSource, ...]] = ()
    loops: ClassVar[tuple[OutputAddress, ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class InputFromGraphInput:
    """An input whose value the caller gives when a run starts: a graph input. In the example, ``lines.text``
    is ``InputFromGraphInput(text, "lines.text")``.

    Attributes:
        field: The input, as its node declares it.
        address: The key a run keeps the caller's value under, ``node.input``.
        sources: Empty: no edge leads to the input.
        loops: Empty: the value is made in no loop.
    """

    field: Field
    address: str
    sources: ClassVar[tuple[EdgeSource, ...]] = ()
    loops: ClassVar[tuple[OutputAddress, ...]] = ()


@final
@dataclass(frozen=True, slots=True)
class InputReadsOneValue:
    """An input that reads one value through an edge on each call. In the example, ``shout.line`` reads one item
    of ``lines.result`` per call, so its ``loops`` is the loop over ``lines.result``.

    Attributes:
        field: The input, as its node declares it.
        sources: What each edge into the input reads. With several edges, each comes from a different branch of
            one decision, and a call gets the value from the branch that was chosen.
        loops: The loops the input's node must be inside to read the value, outermost first: those of the
            source with the most loops.
    """

    field: Field
    sources: tuple[EdgeSource, ...]
    loops: tuple[OutputAddress, ...]

    def value(self, reader: ResultReader, item_indexes: ItemIndexes) -> Any:
        """Returns the input's value for the call at ``item_indexes``, as ``reader.read_output`` returns one."""
        sources = self.sources
        return reader.read_output(sources[0], item_indexes) if len(sources) == 1 else reader.read_alternatives(sources, item_indexes)


@final
@dataclass(frozen=True, slots=True)
class InputCollectsLoopResults:
    """A ``Series`` input (or a ``Table`` input wired to records) that gets one value per item of a loop, as a
    list. In the example, ``join.texts`` collects the loop over ``lines.result``: ``series`` is ``lines.result``
    and the one source is ``shout.result``, so ``join`` is called once, with ``["A!", "B!"]``.

    Attributes:
        field: The input, as its node declares it.
        series: The list whose loop the input collects.
        sources: What each edge into the input reads for one item. Several edges come from different branches
            of one decision, as for any input.
        loop_body: In the flat graph, the function nodes inside the loop that the input's value depends on,
            each after the nodes upstream of it: in the example, ``shout``. Empty in a compiled graph's own
            plan.
    """

    field: Field
    series: OutputInLoops
    sources: tuple[EdgeSource, ...]
    loop_body: tuple[FlatFunctionNode, ...] = ()

    @property
    def loops(self) -> tuple[OutputAddress, ...]:
        """The loops the input's node must be inside to read the list, outermost first: the loops the node
        that returns ``series`` is inside. The loop over ``series`` itself is not one of them."""
        return self.series.loops

    def value(self, reader: ResultReader, item_indexes: ItemIndexes) -> Any:
        """Returns the input's value for the call at ``item_indexes``, as ``reader.collect`` returns one."""
        return reader.collect(self, item_indexes)


@final
@dataclass(frozen=True, slots=True)
class InputWrapsOneValue:
    """A ``Series`` input (or a ``Table`` input wired to records) whose edge reads one value per call of its
    node, so there is no loop to collect. A call gets that one value as a list of one. Example: ``join.texts``
    wired straight to an output ``title.result`` that is a ``str``: ``join`` is called with ``["Title"]``.

    Attributes:
        field: The input, as its node declares it.
        sources: What each edge into the input reads. Several edges come from different branches of one
            decision, as for any input.
    """

    field: Field
    sources: tuple[EdgeSource, ...]

    @property
    def loops(self) -> tuple[OutputAddress, ...]:
        """The loops the input's node must be inside to read the value, outermost first: those of its source."""
        return self.sources[0].loops

    def value(self, reader: ResultReader, item_indexes: ItemIndexes) -> Any:
        """Returns the input's value for the call at ``item_indexes``: what its edge reads, in a list of one.
        ``SKIPPED`` and ``NOT_READY`` are returned as they are, and so is a ``None`` unless the input accepts
        ``None`` items (``Series[str | None]``), which gets ``[None]``."""
        sources = self.sources
        value = reader.read_output(sources[0], item_indexes) if len(sources) == 1 else reader.read_alternatives(sources, item_indexes)
        if value is SKIPPED or value is NotReady.NOT_READY or (value is None and not self.field.accepts_none_items):
            return value
        return [value]


#: An input that reads its value through edges: the kinds that have ``value``.
InputFromEdges = InputReadsOneValue | InputCollectsLoopResults | InputWrapsOneValue

#: How one input of a node gets its value.
InputPlan = InputConstant | InputMissing | InputFromGraphInput | InputFromEdges
