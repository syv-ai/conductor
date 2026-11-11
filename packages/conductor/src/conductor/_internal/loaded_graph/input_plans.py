"""What an edge brings and how an input gets its value: what loading decides for each input, and a run reads.

Scopes. A node at the graph's own scope, ``()``, runs once. A ``Series[T]``
output read by an input that takes one ``T`` opens a map: the reader runs once
per item. A map is named by its Series (``split.result``), and a scope is the
path of maps from the graph's own scope, ``(docs.result, extract.lines)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from conductor._internal.fields import Field, Type


#: One output of one node.
class Port(NamedTuple):
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
Scope = tuple[Port, ...]


class Branch(NamedTuple):
    """One way a decision must go for a node to hold a value.

    Attributes:
        node_id: The node that decides.
        group: Its exclusive group.
        output: The output of the group that must hold the value.
    """

    node_id: str
    group: str
    output: str


@dataclass(frozen=True, slots=True)
class Cell:
    """An output of a node, in the scope its node runs in: what one unit of the node holds for it.

    Attributes:
        node_id: The node.
        output: The output's name.
        scope: The scope the node runs in.
        exclusive: Whether the output is one of an exclusive group. Where its node wasn't called for a blank,
            no branch was chosen, so it is a branch not taken there and not a blank.
    """

    node_id: str
    output: str
    scope: Scope
    exclusive: bool

    @property
    def address(self) -> str:
        """The address an edge names it by, ``node.output``."""
        return f"{self.node_id}.{self.output}"


@dataclass(frozen=True, slots=True)
class _Brought(Cell):
    """What an edge brings from a cell. Never made on its own: an edge brings a ``Value`` or an ``Item``.

    Attributes:
        field: The field an edge names after the output (``split.result.email``); ``None`` for all of it.
        type: The type of what it brings.
    """

    field: str | None
    type: Type

    @property
    def address(self) -> str:
        """The address an edge names it by, ``node.output`` or ``node.output.field``."""
        return f"{self.node_id}.{self.output}.{self.field}" if self.field else f"{self.node_id}.{self.output}"


@dataclass(frozen=True, slots=True)
class Value(_Brought):
    """An output read as the one value it is, where its node runs or in a scope under it."""

    @property
    def read_in(self) -> Scope:
        """The scope its values sit in: the one its node runs in."""
        return self.scope


@dataclass(frozen=True, slots=True)
class Item(_Brought):
    """One item of a ``Series`` output, read inside the map that Series opens."""

    @property
    def read_in(self) -> Scope:
        """The scope its values sit in: the map of the Series, under the scope its node runs in."""
        return (*self.scope, Port(self.node_id, self.output))


#: What an edge brings its input.
Source = Value | Item

# -- how an input gets its value ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fixed:
    """An input that holds one value on every call: typed into the node, or its default.

    Attributes:
        spec: The input's field.
        value: The value, as the field's type read it.
    """

    spec: Field
    value: Any


@dataclass(frozen=True, slots=True)
class Unset:
    """An input with no value to give: its typed-in value was refused by its type, it must be fed and nothing
    feeds it, or its edges bring nothing it can take (an edge refused, or one from a node whose fields aren't
    known). The graph holds the problem, so it never runs; the node is still drawn, with this input to fill.

    Attributes:
        spec: The input's field.
    """

    spec: Field


@dataclass(frozen=True, slots=True)
class Open:
    """An input of a first node that nothing fills: the graph's input, which the caller gives.

    Attributes:
        spec: The input's field.
        address: Where a run keeps the caller's value, ``node.input``.
    """

    spec: Field
    address: str


#: One cell an input is made from, and how many of the maps its node runs in say which unit of the cell's node
#: holds it: the unit at those places of an instance of the reading node, and for a gather every unit under them.
CellRead = tuple[Cell, int]


@dataclass(frozen=True, slots=True)
class Read:
    """An input that edges bring one value on each call.

    Attributes:
        spec: The input's field.
        sources: What each edge brings. Several are alternatives, on opposite branches of one decision: a call
            receives the one whose branch was taken.
        scope: The scope the value sits in.
    """

    spec: Field
    sources: tuple[Source, ...]
    scope: Scope

    @property
    def reads(self) -> tuple[tuple[Source, int], ...]:
        """Every cell the input is made from: what each edge brings, read where its node runs. The tree of a
        graph keeps it for each node where it stands, and a run reads those cells and no other."""
        return tuple([(source, len(source.scope)) for source in self.sources])


@dataclass(frozen=True, slots=True)
class Gather:
    """An input of many (a ``Series``, or a ``Table`` fed records) that receives the items of one map as a list.

    Attributes:
        spec: The input's field.
        sources: What each edge brings for an item. Several are alternatives, as for any input.
        series: The Series whose items are gathered, where it was made.
    """

    spec: Field
    sources: tuple[Source, ...]
    series: Cell

    @property
    def reads(self) -> tuple[CellRead, *tuple[tuple[Source, int], ...]]:
        """Every cell the input is made from: the Series first, then what each edge brings for every item under
        one unit of the Series, so each is read as deep as the Series was made. The tree of a graph keeps it
        for each node where it stands, and a run reads those cells and no other."""
        made_in = len(self.series.scope)
        return ((self.series, made_in), *[(source, made_in) for source in self.sources])

    @property
    def scope(self) -> Scope:
        """The scope the list sits in: where the Series was made."""
        return self.series.scope


@dataclass(frozen=True, slots=True)
class SeriesOfOne:
    """An input of many fed by a source that runs once: each call receives its one value as a list of one.

    Attributes:
        spec: The input's field.
        sources: What each edge brings. Several are alternatives, as for any input.
    """

    spec: Field
    sources: tuple[Value, ...]

    @property
    def reads(self) -> tuple[tuple[Value, int], ...]:
        """Every cell the input is made from: what each edge brings, read where the graph itself runs. The tree
        of a graph keeps it for each node where it stands, and a run reads those cells and no other."""
        return tuple([(source, 0) for source in self.sources])

    @property
    def scope(self) -> Scope:
        """The graph's own scope, where a source that runs once runs."""
        return ()


#: An input that edges feed.
Fed = Read | Gather | SeriesOfOne

#: How one input of a placed node gets its value.
InputPlan = Fixed | Unset | Open | Fed


def _read_before_a_call(plan: InputPlan) -> bool:
    """Whether what edges bring an input is read by the input's type before each call: not when every value they
    bring is already one the field takes as it is, with no limit or choice to hold it to."""
    match plan:
        case Read(spec=spec, sources=sources):
            return not all(spec.whole.holds(s.type) for s in sources)
        case Gather(spec=spec, sources=sources) | SeriesOfOne(spec=spec, sources=sources):
            return spec.whole.record is not None or not all(spec.type.holds(s.type) for s in sources)
        case _:
            return False
