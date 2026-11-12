"""How each input of a node gets its value: the plan kinds, and the outputs a plan reads.

An ``OutputInLoops`` is one output of one node, in the scope its node runs in: what one
unit of the node holds for it. An edge brings the output's value (``WholeOutput``), one
item of a ``Series`` it holds (``ItemOfSeries``), or a ``Series`` it holds as the one
list it is (``SeriesAsList``).

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




# -- what an edge brings ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutputInLoops:
    """An output of a node, in the scope its node runs in: what one unit of the node holds for it.

    Attributes:
        node_id: The node: its id in a compiled graph's plan; where a node stands, the function's name.
        output: The output's name.
        loops: The scope the node runs in.
        depth: How many maps the node runs in, so how many places of a set of item indexes name one unit of it.
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
class _EdgeSourceBase(OutputInLoops):
    """What an edge brings from an output. Never made on its own: it is one of the three kinds of ``EdgeSource``.

    Attributes:
        record_field: The field an edge names after the output (``split.result.email``); ``None`` for all of it.
        type: The type of what it brings.
    """

    record_field: str | None
    type: Type

    @property
    def address(self) -> str:
        """The address an edge names it by, ``node.output`` or ``node.output.field``."""
        return f"{self.node_id}.{self.output}.{self.record_field}" if self.record_field else f"{self.node_id}.{self.output}"

    @property
    def loops_of_value(self) -> tuple[OutputAddress, ...]:
        """The scope its values sit in: the one its node runs in."""
        return self.loops


@final
@dataclass(frozen=True, slots=True)
class WholeOutput(_EdgeSourceBase):
    """An output read as the one value it is, where its node runs or in a scope under it."""


@final
@dataclass(frozen=True, slots=True)
class ItemOfSeries(_EdgeSourceBase):
    """One item of a ``Series`` output, read inside the map that Series opens."""

    @property
    def loops_of_value(self) -> tuple[OutputAddress, ...]:
        """The scope its values sit in: the map of the Series, under the scope its node runs in."""
        return (*self.loops, OutputAddress(self.node_id, self.output))


@final
@dataclass(frozen=True, slots=True)
class SeriesAsList(_EdgeSourceBase):
    """A ``Series`` output read as the one list it is.

    Only the plans ``run.results`` reads an output with hold one (``display_plans_of``). Example: ``lines`` is called
    once per document and returns that document's lines as a ``Series``. ``results`` shows a list per
    document, each the whole list one call returned.
    """


#: What an edge brings its input.
EdgeSource = WholeOutput | ItemOfSeries | SeriesAsList


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
    reads: tuple[WholeOutput, ...]

    @property
    def sources(self) -> tuple[WholeOutput, ...]:
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
