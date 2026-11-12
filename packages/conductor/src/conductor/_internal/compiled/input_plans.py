"""How each input of a node gets its value: the plan kinds, and the outputs a plan reads.

An ``OutputInLoops`` is one output of one node with the loops that node is inside. An ``EdgeSource`` is what
one edge reads from such an output: its value, or one item of it when the output is a ``Series``. A
``SeriesAsList`` is how ``run.results`` reads a ``Series`` output: as one plain list.

A plan is one of six kinds. ``InputConstant`` holds a value, ``InputFromGraphInput`` is the
caller's, ``InputMissing`` has none. The three that edges feed (``InputReadsOneValue``, ``InputCollectsLoopResults``,
``InputWrapsOneValue``) each hold ``reads``: every output the input is made from. It is
the one declaration of what the input reads. A run reads those outputs and no
other (``value``), and the units a node waits for are the units of those
outputs, so the two can't differ.

A compiled graph holds the plans of its own nodes, each output naming a node of
that graph by id. Where a node stands among the graphs a run is of, its plans
are written again with every output naming a function by the name a run keeps
its units by (``e/ask``): see ``tree.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, final

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


@final
@dataclass(frozen=True, slots=True)
class SeriesAsList(EdgeSource):
    """A ``Series`` output read as the plain list of its items. No edge reads an output this way: only
    ``run.results`` and ``run.outputs`` do, through the plans of ``display_plans_of``.

    Example: ``lines`` is called once per document and returns that document's lines. ``run.results`` shows
    ``lines.result`` as one list per document, each the list one call returned.
    """


# -- how an input gets its value ----------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class InputConstant:
    """An input that holds one value on every call: typed into the node, or its default.

    Attributes:
        field: The input, as its node declares it.
        value: The value, as the field's type read it.
    """

    field: Field
    value: Any


@final
@dataclass(frozen=True, slots=True)
class InputMissing:
    """An input with no value to give: its typed-in value was refused by its type, it must be fed and nothing
    feeds it, or its edges bring nothing it can take (an edge refused, or one from a node whose fields aren't
    known). The graph holds the problem, so it never runs; the node is still drawn, with this input to fill.

    Attributes:
        field: The input, as its node declares it.
    """

    field: Field


@final
@dataclass(frozen=True, slots=True)
class InputFromGraphInput:
    """An input of a first node that nothing fills: the graph's input, which the caller gives.

    Attributes:
        field: The input, as its node declares it.
        address: Where a run keeps the caller's value, ``node.input``.
    """

    field: Field
    address: str


@final
@dataclass(frozen=True, slots=True)
class InputReadsOneValue:
    """An input that edges bring one value on each call.

    Attributes:
        field: The input, as its node declares it.
        reads: Every output the input is made from: what each edge brings. Several are alternatives, on opposite
            branches of one decision: a call receives the one whose branch was taken.
        loops: The scope the value sits in.
    """

    field: Field
    reads: tuple[EdgeSource, ...]
    loops: tuple[OutputAddress, ...]

    @property
    def sources(self) -> tuple[EdgeSource, ...]:
        """What each edge brings."""
        return self.reads

    def value(self, reader: ResultReader, item_indexes: ItemIndexes) -> Any:
        """What the input holds for ``item_indexes``: what its edge brings, or the best of alternative edges."""
        reads = self.reads
        return reader.read_output(reads[0], item_indexes) if len(reads) == 1 else reader.read_alternatives(reads, item_indexes)


@final
@dataclass(frozen=True, slots=True)
class InputCollectsLoopResults:
    """An input of many (a ``Series``, or a ``Table`` fed records) that receives the items of one map as a list.

    Attributes:
        field: The input, as its node declares it.
        reads: Every output the input is made from: the Series whose items are gathered, where it was made, then
            what each edge brings for an item. Several edges are alternatives, as for any input.
        loop_body: Where the input's node stands, the functions made per item of the Series that what it
            gathers rests on, each after the ones it reads: what stands between the input and everything
            outside the items. A compiled graph's own plan names none.
    """

    field: Field
    reads: tuple[OutputInLoops, ...]
    loop_body: tuple[FlatFunctionNode, ...] = ()

    @property
    def series(self) -> OutputInLoops:
        """The Series whose items are gathered, where it was made."""
        return self.reads[0]

    @property
    def sources(self) -> tuple[EdgeSource, ...]:
        """What each edge brings for an item."""
        return self.reads[1:]  # type: ignore[return-value]  # every output after the Series is what an edge brings

    def value(self, reader: ResultReader, item_indexes: ItemIndexes) -> Any:
        """What the input holds for ``item_indexes``: the items of its Series, or what is outside them when that
        decides."""
        return reader.collect(self, item_indexes)

    @property
    def loops(self) -> tuple[OutputAddress, ...]:
        """The scope the list sits in: where the Series was made."""
        return self.reads[0].loops


@final
@dataclass(frozen=True, slots=True)
class InputWrapsOneValue:
    """An input of many fed by a source that makes one value for each call of the node: the source runs where
    the node runs, or outside the items the node runs per. Each call receives that one value as a list of one.

    Attributes:
        field: The input, as its node declares it.
        reads: Every output the input is made from: what each edge brings. Several are alternatives, as for any
            input.
    """

    field: Field
    reads: tuple[EdgeSource, ...]

    @property
    def sources(self) -> tuple[EdgeSource, ...]:
        """What each edge brings."""
        return self.reads

    def value(self, reader: ResultReader, item_indexes: ItemIndexes) -> Any:
        """What the input holds for ``item_indexes``: the one value its edge brings, as a list of one; where the
        items take a blank, a blank is its one item."""
        reads = self.reads
        value = reader.read_output(reads[0], item_indexes) if len(reads) == 1 else reader.read_alternatives(reads, item_indexes)
        if value is SKIPPED or value is NotReady.NOT_READY or (value is None and not self.field.accepts_none_items):
            return value
        return [value]

    @property
    def loops(self) -> tuple[OutputAddress, ...]:
        """The scope the value sits in: the one its source runs in."""
        return self.reads[0].loops


#: An input that edges feed.
InputFromEdges = InputReadsOneValue | InputCollectsLoopResults | InputWrapsOneValue

#: How one input of a placed node gets its value.
InputPlan = InputConstant | InputMissing | InputFromGraphInput | InputFromEdges
