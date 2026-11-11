"""How each input of a node gets its value: the plan kinds, and the cells a plan reads.

A cell (``OutputAtScope``) is one output of one node, in the scope its node runs in: what one
unit of the node holds for it. An edge brings a cell's value (``WholeOutput``), one
item of a ``Series`` it holds (``ItemOfSeries``), or a ``Series`` it holds as the one
list it is (``WholeSeries``).

A plan is one of six kinds. ``InputHoldsValue`` holds a value, ``InputFromCaller`` is the
caller's, ``InputNotFed`` has none. The three that edges feed (``InputReadsOneValue``, ``InputGathersItems``,
``InputGathersOneValue``) each hold ``reads``: every cell the input is made from. It is
the one declaration of what the input reads. A run reads those cells and no
other (``value``), and the units a node waits for are the units of those
cells, so the two can't differ.

A checked graph holds the plans of its own nodes, each cell naming a node of
that graph by id. Where a node stands among the graphs a run is of, its plans
are written again with every cell naming a function by the name a run keeps
its units by (``e/ask``): see ``tree.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, final

from conductor._internal.fields import SKIPPED, Field, Type
from conductor._internal.units import NOT_SETTLED, ItemPath

if TYPE_CHECKING:
    from conductor._internal.loaded_graph.tree import FunctionAtPath


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


#: A path of maps from the graph's own scope, each the ``Series`` output it works through.
Scope = tuple[OutputAddress, ...]


# -- what an edge brings ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutputAtScope:
    """An output of a node, in the scope its node runs in: what one unit of the node holds for it.

    Attributes:
        node_id: The node: its id in a checked graph's plan; where a node stands, the function's name.
        output: The output's name.
        scope: The scope the node runs in.
        depth: How many maps the node runs in, so how many places of an item path name one unit of it.
    """

    node_id: str
    output: str
    scope: Scope
    depth: int = dataclasses.field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", len(self.scope))

    @property
    def address(self) -> str:
        """The address an edge names it by, ``node.output``."""
        return f"{self.node_id}.{self.output}"


@dataclass(frozen=True, slots=True)
class _Brought(OutputAtScope):
    """What an edge brings from a cell. Never made on its own: it is one of the three kinds of ``Source``.

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
    def read_in(self) -> Scope:
        """The scope its values sit in: the one its node runs in."""
        return self.scope


@final
@dataclass(frozen=True, slots=True)
class WholeOutput(_Brought):
    """An output read as the one value it is, where its node runs or in a scope under it."""


@final
@dataclass(frozen=True, slots=True)
class ItemOfSeries(_Brought):
    """One item of a ``Series`` output, read inside the map that Series opens."""

    @property
    def read_in(self) -> Scope:
        """The scope its values sit in: the map of the Series, under the scope its node runs in."""
        return (*self.scope, OutputAddress(self.node_id, self.output))


@final
@dataclass(frozen=True, slots=True)
class WholeSeries(_Brought):
    """A ``Series`` output read as the one list it is.

    Only a plan written where a node stands holds one. An embedded graph offers an output made per item inside
    it as the ``Series`` of every item's value, so where that output is itself a ``Series``, an item of what is
    offered is the whole of what one unit inside holds.
    """


#: What an edge brings its input.
Source = WholeOutput | ItemOfSeries | WholeSeries


# -- how an input gets its value ----------------------------------------------------------------


class ReadsCells(Protocol):
    """What a plan asks of what reads for it: a way to read one cell, for one item or for an item nobody
    knows. Each answer is a value, ``None`` for a blank, ``SKIPPED`` for a branch not taken, or ``NOT_SETTLED``."""

    def cell(self, cell: OutputAtScope, item_path: ItemPath) -> Any:
        """What ``cell`` holds for ``item_path``, as an edge brings it."""

    def best_of(self, reads: tuple[OutputAtScope, ...], item_path: ItemPath) -> Any:
        """What alternative edges bring one input: the best of what each brings."""

    def gathered(self, plan: InputGathersItems, item_path: ItemPath) -> Any:
        """What the gathering input of ``plan`` holds: its items, or what is outside them when that decides."""


@final
@dataclass(frozen=True, slots=True)
class InputHoldsValue:
    """An input that holds one value on every call: typed into the node, or its default.

    Attributes:
        field: The input's field.
        value: The value, as the field's type read it.
    """

    field: Field
    value: Any


@final
@dataclass(frozen=True, slots=True)
class InputNotFed:
    """An input with no value to give: its typed-in value was refused by its type, it must be fed and nothing
    feeds it, or its edges bring nothing it can take (an edge refused, or one from a node whose fields aren't
    known). The graph holds the problem, so it never runs; the node is still drawn, with this input to fill.

    Attributes:
        field: The input's field.
    """

    field: Field


@final
@dataclass(frozen=True, slots=True)
class InputFromCaller:
    """An input of a first node that nothing fills: the graph's input, which the caller gives.

    Attributes:
        field: The input's field.
        address: Where a run keeps the caller's value, ``node.input``.
    """

    field: Field
    address: str


@final
@dataclass(frozen=True, slots=True)
class InputReadsOneValue:
    """An input that edges bring one value on each call.

    Attributes:
        field: The input's field.
        reads: Every cell the input is made from: what each edge brings. Several are alternatives, on opposite
            branches of one decision: a call receives the one whose branch was taken.
        scope: The scope the value sits in.
    """

    field: Field
    reads: tuple[Source, ...]
    scope: Scope

    @property
    def sources(self) -> tuple[Source, ...]:
        """What each edge brings."""
        return self.reads

    def value(self, reader: ReadsCells, item_path: ItemPath) -> Any:
        """What the input holds for ``item_path``: what its edge brings, or the best of alternative edges."""
        reads = self.reads
        return reader.cell(reads[0], item_path) if len(reads) == 1 else reader.best_of(reads, item_path)


@final
@dataclass(frozen=True, slots=True)
class InputGathersItems:
    """An input of many (a ``Series``, or a ``Table`` fed records) that receives the items of one map as a list.

    Attributes:
        field: The input's field.
        reads: Every cell the input is made from: the Series whose items are gathered, where it was made; where
            the items sit more than one map under it, each Series on the way down; then what each edge brings
            for an item. Several edges are alternatives, as for any input.
        levels: How many of ``reads`` are a Series: one, but for what an embedded graph offers of an output
            made more than one map deep inside it, which is gathered through every one of those maps.
        asked_for_any_item: Where the input's node stands, the functions made per item of the Series that what it
            gathers rests on, each after the ones it reads: what stands between the input and everything
            outside the items. A checked graph's own plan names none.
    """

    field: Field
    reads: tuple[OutputAtScope, ...]
    levels: int = 1
    asked_for_any_item: tuple[FunctionAtPath, ...] = ()

    @property
    def series(self) -> OutputAtScope:
        """The Series whose items are gathered, where it was made."""
        return self.reads[0]

    @property
    def sources(self) -> tuple[Source, ...]:
        """What each edge brings for an item."""
        return self.reads[self.levels :]  # type: ignore[return-value]  # every cell after the Series is what an edge brings

    def value(self, reader: ReadsCells, item_path: ItemPath) -> Any:
        """What the input holds for ``item_path``: the items of its Series, or what is outside them when that
        decides."""
        return reader.gathered(self, item_path)

    @property
    def scope(self) -> Scope:
        """The scope the list sits in: where the Series was made."""
        return self.reads[0].scope


@final
@dataclass(frozen=True, slots=True)
class InputGathersOneValue:
    """An input of many fed by a source that makes one value for each call of the node: the source runs where
    the node runs, or outside the items the node runs per. Each call receives that one value as a list of one.

    Attributes:
        field: The input's field.
        reads: Every cell the input is made from: what each edge brings. Several are alternatives, as for any
            input.
    """

    field: Field
    reads: tuple[WholeOutput, ...]

    @property
    def sources(self) -> tuple[WholeOutput, ...]:
        """What each edge brings."""
        return self.reads

    def value(self, reader: ReadsCells, item_path: ItemPath) -> Any:
        """What the input holds for ``item_path``: the one value its edge brings, as a list of one; where the
        items take a blank, a blank is its one item."""
        reads = self.reads
        value = reader.cell(reads[0], item_path) if len(reads) == 1 else reader.best_of(reads, item_path)
        if value is SKIPPED or value is NOT_SETTLED or (value is None and not self.field.keeps_blanks):
            return value
        return [value]

    @property
    def scope(self) -> Scope:
        """The scope the value sits in: the one its source runs in."""
        return self.reads[0].scope


#: An input that edges feed.
InputFedByEdges = InputReadsOneValue | InputGathersItems | InputGathersOneValue

#: How one input of a placed node gets its value.
InputPlan = InputHoldsValue | InputNotFed | InputFromCaller | InputFedByEdges
