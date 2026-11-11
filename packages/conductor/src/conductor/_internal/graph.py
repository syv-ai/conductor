"""``Graph``: a stored graph checked against the nodes it uses, ready to run.

Loading does everything once: every edge and value is checked, and each node
gets the scope it runs in. What is wrong becomes a ``Problem`` with a place,
never an exception.

Scopes. A node at the graph's own scope, ``()``, runs once. A ``Series[T]``
output read by an input that takes one ``T`` opens a map: the reader runs once
per item. A map is named by its Series (``split.result``), and a scope is the
path of maps from the graph's own scope, ``(docs.result, extract.lines)``. Declared types alone decide every
scope, by three rules:

- an input of one ``T`` fed by a ``Series[T]`` reads one item, inside the map
  that Series opens; a ``Table`` input fed a ``Series`` of tables is one;
- an input fed from an enclosing scope captures that scope's one value;
- an input of ``Series[T]``, or of ``Table[R]`` fed records, gathers: it
  receives every value its source made: the items of the innermost map of
  what feeds it, in order, and it sits in the scope that map's Series was
  made in; or, from a source that runs once, that one value as a Series of
  one.

A node runs in the deepest of its inputs' scopes, and every input's scope
lies on the path to it: items of two Series never pair up
(``series_not_nested``). Readers of one Series read one map.

A table is one value. An embedded graph is one node: its inputs are the inner
graph's inputs, its outputs the inner graph's outputs. It loads once, alone,
however many graphs embed it, and nothing of it is copied into the graph
around it. What a run walks is the tree those graphs make: every node of the
graph and of each graph embedded in it, each a ``NodeAtPath``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import heapq
import inspect
import logging
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, TypeVar

import pydantic

from conductor._internal.ask import output_questions
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import (
    CtxValue,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    Problem,
    make_problem,
)
from conductor._internal.fields import (
    REQUIRED,
    Field,
    FieldDescription,
    FieldValueError,
    Param,
    Series,
    Type,
)
from conductor._internal.node_definition import (
    FunctionDefinition,
    GraphDefinition,
    HookName,
    NodeDefinition,
    Policy,
    is_node_id,
    read_by_hooks,
)
from conductor._internal.record import GraphRecord, NodeRecord

if TYPE_CHECKING:
    from conductor._internal.run import Run, SaveForm

_logger = logging.getLogger("conductor")


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


# -- what an edge brings ------------------------------------------------------------------------


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
Reads = tuple[Cell, int]


@dataclass(frozen=True, slots=True)
class Read:
    """An input that edges bring one value on each call.

    Attributes:
        spec: The input's field.
        reads: What each edge brings, each with how many maps deep its node runs. Several are alternatives, on
            opposite branches of one decision: a call receives the one whose branch was taken.
        scope: The scope the value sits in.
    """

    spec: Field
    reads: tuple[tuple[Source, int], ...]
    scope: Scope

    @property
    def sources(self) -> tuple[Source, ...]:
        """What each edge brings."""
        return tuple(source for source, _ in self.reads)


@dataclass(frozen=True, slots=True)
class Gather:
    """An input of many (a ``Series``, or a ``Table`` fed records) that receives the items of one map as a list.

    Attributes:
        spec: The input's field.
        reads: The Series whose items are gathered, where it was made, and then what each edge brings for an
            item; each with how many maps deep the Series was made, as every item under one unit of the Series
            is gathered. Several edges are alternatives, as for any input.
    """

    spec: Field
    reads: tuple[Reads, *tuple[tuple[Source, int], ...]]

    @property
    def series(self) -> Cell:
        """The Series whose items are gathered, where it was made."""
        return self.reads[0][0]

    @property
    def sources(self) -> tuple[Source, ...]:
        """What each edge brings for an item."""
        return tuple(source for source, _ in self.reads[1:])

    @property
    def scope(self) -> Scope:
        """The scope the list sits in: where the Series was made."""
        return self.series.scope


@dataclass(frozen=True, slots=True)
class SeriesOfOne:
    """An input of many fed by a source that runs once: each call receives its one value as a list of one.

    Attributes:
        spec: The input's field.
        reads: What each edge brings, each read where the graph itself runs. Several are alternatives, as for
            any input.
    """

    spec: Field
    reads: tuple[tuple[Value, int], ...]

    @property
    def sources(self) -> tuple[Value, ...]:
        """What each edge brings."""
        return tuple(source for source, _ in self.reads)

    @property
    def scope(self) -> Scope:
        """The graph's own scope, where a source that runs once runs."""
        return ()


#: An input that edges feed.
Fed = Read | Gather | SeriesOfOne

#: How one input of a placed node gets its value.
InputPlan = Fixed | Unset | Open | Fed


# -- a node of a loaded graph -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class _Placed:
    """What every node of a loaded graph has, whether it is a function or an embedded graph. Never made on its own.

    Attributes:
        id: The node's id in the record.
        scope: The scope it runs in.
        inputs: How each input gets its value, by name; the fields are the node's as its hooks made them.
        outputs: Each output's field, by name, as its hooks made them.
        opens: The ``Series`` outputs some reader works through one item at a time, which a run holds to
            ``max_items``. A reader placed after this node adds to it.
    """

    id: str
    scope: Scope
    inputs: dict[str, InputPlan]
    outputs: dict[str, Field]
    opens: set[str] = dataclasses.field(default_factory=set[str])

    @functools.cached_property
    def reads(self) -> frozenset[str]:
        """The nodes its inputs read."""
        return frozenset(source.node_id for plan in self.inputs.values() if isinstance(plan, Fed) for source in plan.sources)

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input can hold a mapping, so what a unit received needs no key sorting to hash."""
        return all(plan.spec.type.hashes_plainly for plan in self.inputs.values())


@typing.final
@dataclass(frozen=True, kw_only=True)
class PlacedFunction(_Placed):
    """A node of a loaded graph that is a function.

    Attributes:
        definition: The node as its author declared it.
    """

    definition: FunctionDefinition[..., Any]

    @functools.cached_property
    def reuse_digest(self) -> str:
        """What a unit's result rests on besides what it received: the node's version, its code, its fields and its scope.

        So a node redeclared under the same id and version, with other code, a field of another type, limit or
        choice, or one that runs in another scope, is called again by ``run.again``. Code is read from the function's own bytecode and
        constants, not from what it calls.
        """
        specs = (*(plan.spec for plan in self.inputs.values()), *self.outputs.values())
        fields = [(spec.name, spec.whole.reads_as) for spec in specs]
        definition = self.definition
        written = repr((definition.id, definition.version, _code_text(inspect.unwrap(definition.function).__code__), fields, self.scope))
        return hashlib.blake2b(written.encode(), digest_size=8).hexdigest()

    @functools.cached_property
    def questions(self) -> dict[str, FieldDescription]:
        """What a person is asked for each output when the node raises ``Ask``, before the ``Ask`` adds to it."""
        return output_questions(self.outputs)


@typing.final
@dataclass(frozen=True, kw_only=True)
class PlacedGraph(_Placed):
    """A node of a loaded graph that is an embedded graph: the nodes of ``inner`` run where it stands.

    Its inputs are the inner graph's (``inner.inputs``) by their names there, and its outputs the inner graph's
    (``inner.outputs``) by their addresses there. It has no policy and no unit of it is reused whole: the nodes
    inside are called by their own policies, and a later run takes their earlier work node by node, as it would
    with the same nodes drawn in the graph itself.

    Attributes:
        definition: The node as its author declared it.
        inner: The embedded graph, loaded.
    """

    definition: GraphDefinition
    inner: Graph


#: One node of a loaded graph: a function, or an embedded graph.
PlacedNode = PlacedFunction | PlacedGraph

#: A node whatever item it runs for: its id, after the id of each embedded graph's node it is inside, outermost
#: first, ``("e", "ask")``.
NodePath = tuple[str, ...]


@dataclass(eq=False, slots=True, kw_only=True)
class _AtPath:
    """A node where it stands in the tree of graphs a run walks: a node of the graph itself, or of a graph
    embedded in it, at any depth. Never made on its own: it is a ``FunctionAtPath`` or a ``GraphAtPath``.

    A graph loads once and may be embedded in many places; this is a node of it in one of them. A node inside an
    embedded graph runs once for each item its embedding node runs for, and then once per item of its own scope,
    so the maps it runs in are the embedding node's and then its own.

    Attributes:
        embedding: The embedded graph's node it is directly inside; ``None`` for a node of the graph itself.
        inside: The ids of the embedded graphs' nodes it is inside, outermost first; ``()`` for a node of the
            graph itself.
        path: ``inside``, then its own id.
        maps: Every map it runs in, outermost first, each as the node and the output of its ``Series``: those of
            the embedded graph's node it is inside, then those of its own scope.
        base: How many of ``maps`` are the embedded graph's node's.
        inputs: How each input gets its value where the node stands, by name: its plan, and the embedded
            graph's node in whose graph that plan reads, ``None`` for the node's own graph. An open input of a
            node inside an embedded graph has the plan that graph's node has for it.
        reads: Every node whose units its own are made from, each with how many places of an instance of this
            node say which of that node's units: the unit at those places, and where that node runs deeper every
            unit under them. They are the cells its inputs' plans read (``reads`` of each); for a gathering
            input also what the nodes it gathers through read outside the items, which can settle it for
            every item at once; for an embedded graph's node, the cells it offers and the Series they are made
            per item of; and, for a node that reads anything from outside the innermost map it runs in, or
            reads nothing, the Series of that map, since what it reads may be there before its instances are.
        readers: Each node whose ``reads`` holds this one, with the number held there.
        position: Its place in the order a leg tries the nodes of the tree in.
        waits_on: Every node it waits on, however far up; filled when first asked for.
    """

    embedding: GraphAtPath | None
    inside: NodePath
    path: NodePath
    maps: Maps
    base: int
    inputs: dict[str, tuple[InputPlan, GraphAtPath | None]]
    reads: dict[NodeAtPath, int] = dataclasses.field(default_factory=dict["NodeAtPath", int])
    readers: list[tuple[NodeAtPath, int]] = dataclasses.field(default_factory=list[tuple["NodeAtPath", int]])
    position: int = 0
    waits_on: frozenset[NodeAtPath] | None = None


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FunctionAtPath(_AtPath):
    """A function node where it stands in the tree of graphs a run walks.

    Attributes:
        placed: The node, as its graph loaded it.
        policy: The node's policy, as its definition declares it.
        checked: The inputs whose value is read by the input's type before a call, as the types the edges bring
            don't settle it: the node's own, and the open ones an embedded graph's node is handed by such edges.
        gathered_through: For each gathering input, by name, the nodes made per item of its Series that what it
            gathers rests on, in the tree's order: what stands between the input and everything outside the
            items. A run asks them, in this order, whether any item could bring the input a value.
    """

    placed: PlacedFunction
    policy: Policy
    checked: tuple[str, ...]
    gathered_through: dict[str, tuple[NodeAtPath, ...]] = dataclasses.field(default_factory=dict[str, tuple["NodeAtPath", ...]])


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class GraphAtPath(_AtPath):
    """An embedded graph's node where it stands in the tree of graphs a run walks. It is never called: its unit
    is what the outputs of the graph inside it came to.

    Attributes:
        placed: The node, as its graph loaded it.
        offers: Each output it offers: its address in the inner graph, the cell of the inner graph that holds
            it, the maps that cell's node runs in, and whether it is a ``Series``.
        ends_silently: The nodes inside that nothing inside reads and that have no output. Its unit waits for
            theirs as for the outputs it offers, so it is done when the graph inside is.
    """

    placed: PlacedGraph
    offers: tuple[tuple[str, Cell, Maps, bool], ...] = ()
    ends_silently: tuple[NodeAtPath, ...] = ()


#: A node where it stands in the tree of graphs a run walks: a function, or an embedded graph.
NodeAtPath = FunctionAtPath | GraphAtPath

#: The maps a node runs in, outermost first, each as the node and the output of its ``Series``.
Maps = tuple[tuple["NodeAtPath", str], ...]


def waits_on(node: NodeAtPath) -> frozenset[NodeAtPath]:
    """Every node ``node`` waits on, however far up; worked out when first asked, as most nodes never are."""
    if node.waits_on is None:
        seen: set[NodeAtPath] = set()
        stack = [node]
        while stack:
            for read in stack.pop().reads:
                if read not in seen:
                    seen.add(read)
                    stack.append(read)
        node.waits_on = frozenset(seen)
    return node.waits_on


# -- what a host reads of a loaded graph --------------------------------------------------------


class RunsOn(ConductorModel):
    """Where a node runs.

    Attributes:
        mode: ``once``, or ``per_item`` of a Series.
        series: The address of the Series it runs per item of; ``None`` for a node that runs once.
        msg: The same, in English.
    """

    mode: Literal["once", "per_item"]
    series: str | None = None
    msg: str


class Feed(ConductorModel):
    """How an input is fed on each call of its node.

    Attributes:
        mode: ``one_value`` for a value typed in, a default or the caller's; ``per_item`` for one item of
            ``series``; ``outer_value`` for a value made outside the items the node runs per; ``gathered`` for
            every item of ``series`` as one list; ``unset`` for an input with no value to give, which a problem
            of the graph says why.
        series: The address of the Series the mode names: for ``outer_value``, the one whose item the value
            belongs to. ``None`` for ``one_value``, for an ``outer_value`` made once in the graph, and for the
            one value of a source that runs once, ``gathered`` as a Series of one.
        msg: The same, in English.
    """

    mode: Literal["one_value", "per_item", "outer_value", "gathered", "unset"]
    series: str | None = None
    msg: str


class PlacedFieldDescription(ConductorModel):
    """An input of a node in a loaded graph.

    Attributes:
        field: The input as its node's hooks and edges made it.
        feed: How it is fed on each call of its node.
    """

    field: FieldDescription
    feed: Feed


class GraphNodeDescription(ConductorModel):
    """One node of a loaded graph as an editor draws it.

    Attributes:
        node: The id of the definition it uses.
        version: The version of that definition.
        loaded: Whether its fields are known: its definition was found at its version and its hooks gave its
            fields. ``problems`` says why when they aren't.
        problems: The problems placed on it. A node whose fields are known may hold some: an input an edge is
            refused for is fed ``unset``.
        runs: Where it runs, when its fields are known.
        inputs: Its inputs as its hooks made them, when its fields are known.
        outputs: Its outputs as its hooks made them, when its fields are known.
    """

    node: str
    version: int
    loaded: bool
    problems: list[Problem]
    runs: RunsOn | None = None
    inputs: list[PlacedFieldDescription] = []
    outputs: list[FieldDescription] = []


class GraphInput(ConductorModel):
    """An open input of a first node, a value of ``graph.inputs``: what a caller fills.

    Attributes:
        address: Where the caller's value goes, ``node.input``.
        node_id: The node whose input it is.
        field: The input. Its ``default`` is what it holds when the caller gives nothing (the record's default,
            else the node's), as JSON, and it is ``required`` when there is none.
    """

    address: str
    node_id: str
    field: FieldDescription


class GraphDescription(ConductorModel):
    """A loaded graph as an editor draws it.

    Attributes:
        nodes: Each node of the record, by id.
        inputs: The graph's inputs, by name.
        outputs: The graph's outputs, as addresses.
        problems: Every problem of the graph.
    """

    nodes: dict[str, GraphNodeDescription]
    inputs: dict[str, GraphInput]
    outputs: list[str]
    problems: list[Problem]


@dataclass(frozen=True, slots=True)
class InputField:
    """What a run keeps of a graph input.

    Attributes:
        name: Its name in ``graph.inputs``: the input's own, or its address where two inputs share a name.
        address: Where the caller's value is kept, ``node.input``.
        spec: Its field, whose default is what it holds when the caller gives nothing: the record's default, else
            the node's, and ``REQUIRED`` when the caller must give one.
    """

    name: str
    address: str
    spec: Field


@dataclass(frozen=True, eq=False)
class Graph:
    """A graph loaded against a registry: ``registry.load(record)`` makes one.

    ``problems`` is everything wrong with it, and a graph with none runs: ``graph.run(**inputs)``.

    Attributes:
        record: The stored graph it was loaded from.
        problems: Everything wrong with it; empty for a graph that runs.
        inputs: The open inputs of its first nodes, by name (``node.input`` where two share a name): what a
            caller fills.
    """

    record: GraphRecord
    problems: list[Problem]
    inputs: dict[str, GraphInput]
    #: Each node whose fields are known, by id, each after the nodes its inputs read. A graph with problems holds
    #: the nodes they are on too, with ``Unset`` for an input that gets nothing, and never runs.
    _nodes: dict[str, PlacedNode]
    #: Each graph input as a run keeps it, by address.
    _input_fields: dict[str, InputField]
    #: The nodes that read each node.
    _readers: dict[str, list[PlacedNode]]
    #: The nodes with a ``FromRun`` parameter, an embedded graph's too, as ``from_run_values`` reads them.
    _needs_from_run: list[tuple[str, FunctionDefinition[..., Any]]]

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}"
        return f"<Graph of {len(self.record.nodes)} node{'s' if len(self.record.nodes) != 1 else ''}, {state}>"

    # -- what a host reads ------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """Whether the graph has no problems, so it runs."""
        return not self.problems

    @functools.cached_property
    def outputs(self) -> list[str]:
        """The outputs of the nodes nothing reads, as addresses."""
        return [f"{p.id}.{o}" for p in self._nodes.values() if not self._readers[p.id] for o in p.outputs]

    @property
    def plan(self) -> dict[str, RunsOn]:
        """Where each node runs: ``once``, or ``per_item`` of the innermost Series of its scope."""
        return {placed.id: _runs_on(placed.scope) for placed in self._nodes.values()}

    @property
    def exclusive_groups(self) -> dict[str, dict[str, list[str]]]:
        """Each node's exclusive groups and their outputs, for a host to say which way a run went."""
        found: dict[str, dict[str, list[str]]] = {}
        for placed in self._nodes.values():
            for name, spec in placed.outputs.items():
                if spec.param.exclusive_group:
                    found.setdefault(placed.id, {}).setdefault(spec.param.exclusive_group, []).append(name)
        return found

    def describe(self) -> GraphDescription:
        """The loaded graph as an editor draws it: every node of the record, with its fields and where it runs.

        Each node lists its own ``problems``, and ``problems`` beside the nodes lists every one. A node with a
        problem is drawn with its fields all the same, an input whose edge is refused fed ``unset``, and so is a
        node that reads it. A ``type_mismatch`` names the input in its ``loc``; a ``cycle`` and
        ``series_not_nested`` name the node. Only a node whose fields aren't known is there with ``loaded``
        false and no fields, its problems saying why: an unknown node or version, an id no address can say, a
        graph that embeds itself, a hook that failed or wasn't called for a value it reads, an embedded graph
        with problems of its own. An embedded graph is one node, with the fields it offers; one with problems
        isn't placed, and its problems are this graph's, each ``loc`` the embedding node's id and then the
        place inside it.
        """
        nodes: dict[str, GraphNodeDescription] = {}
        for record in self.record.nodes:
            own = [p for p in self.problems if p.loc[:1] == (record.id,)]
            placed = self._nodes.get(record.id)
            if placed is None:
                nodes[record.id] = GraphNodeDescription(node=record.node, version=record.version, loaded=False, problems=own)
                continue
            nodes[record.id] = GraphNodeDescription(
                node=record.node, version=record.version, loaded=True, problems=own, runs=_runs_on(placed.scope),
                inputs=[PlacedFieldDescription(field=plan.spec.description, feed=_feed(placed.scope, plan)) for plan in placed.inputs.values()],
                outputs=[spec.description for spec in placed.outputs.values()],
            )
        return GraphDescription(nodes=nodes, inputs=dict(self.inputs), outputs=self.outputs, problems=list(self.problems))

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of what a run of this graph rests on: the record, but for each node's ``display``, which
        conductor never reads, and the fingerprint of every graph embedded in it.

        Keys are sorted first, so a record a JSON store gives back in another key order has the same fingerprint.
        A saved run names it, and loading a save into a graph whose fingerprint differs is refused.
        """
        from conductor._internal.run import canonical_json

        own = self.record.model_dump(mode="json", exclude={"nodes": {"__all__": {"display"}}})
        embedded = {p.id: p.inner.fingerprint for p in self._nodes.values() if isinstance(p, PlacedGraph)}
        return hashlib.blake2b(canonical_json([own, embedded]), digest_size=8).hexdigest()

    # -- running ----------------------------------------------------------------

    def with_inputs(
        self, inputs: Mapping[str, Any] = types.MappingProxyType({}), /, *, max_items: int | None = None, earlier: Run | None = None, **by_name: Any,
    ) -> Run:
        """A run with the caller's inputs checked, not started; ``run.resume()`` starts it.

        Args:
            inputs: The inputs by name (``graph.inputs``) or by address (``node.input``), for names Python can't
                take as keywords (``{"a.text": ...}``).
            max_items: The most items any one Series a node works through may hold before the run stops.
            earlier: A run of this graph, or of an earlier version of it, that lends its work as ``run.again``
                does: a node with the same id, version, code, fields and scope is not called where it receives
                what it received there, so after an edit only the nodes it changes, and what they feed, run again.
            **by_name: The inputs, as keywords.

        Returns:
            The run, ``ready``.

        Raises:
            InvalidCallError: The graph has problems, or an input is missing, unknown or not of its type.
        """
        from conductor._internal.run import checked_leg, no_units, start_run

        if self.problems:
            raise InvalidCallError(self.problems)
        checked_leg(max_items=max_items)
        checked = self._checked_inputs(*self._given_by_address({**inputs, **by_name}))
        return start_run(self, checked, max_items, earlier._carried(into=self) if earlier is not None else {}, no_units(self))

    def run(
        self, inputs: Mapping[str, Any] = types.MappingProxyType({}), /, *, max_items: int | None = None,
        from_run: Mapping[type, Any] = types.MappingProxyType({}), timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops."""
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(from_run=from_run, timeout=timeout)

    def load_run(self, saved: str | bytes | Mapping[str, Any]) -> Run:
        """A run of this graph from what ``run.to_json()`` wrote; a graph with problems refuses it with them.

        A save is read from its JSON text, a mapping written as JSON first, so each value is read as its type
        reads JSON.
        """
        from conductor._internal.run import run_from_json

        return run_from_json(self, saved)

    @functools.cached_property
    def _save_form(self) -> SaveForm:
        """How this graph's runs are read from JSON and written as it, made when the first one is."""
        from conductor._internal.run import save_form

        return save_form(self)

    @functools.cached_property
    def _tree(self) -> list[NodeAtPath]:
        """Every node of the graph and of each graph embedded in it, at any depth, each after the nodes it reads:
        the order a leg tries them in. An embedded graph's node comes after the nodes inside it."""
        tree: list[NodeAtPath] = []
        _add_to_tree(self, None, tree, {})
        for node in tree:
            for read, places in node.reads.items():
                read.readers.append((node, places))
        return tree

    @functools.cached_property
    def _ids_inside(self) -> dict[NodePath, list[str]]:
        """The ids of the nodes of each graph of ``_tree``, by the path of the node that embeds it, ``()`` for
        the graph itself: what a run keeps units by."""
        ids: dict[NodePath, list[str]] = {}
        for node in self._tree:
            ids.setdefault(node.inside, []).append(node.placed.id)
        return ids

    @functools.cached_property
    def _node_at(self) -> dict[NodePath, NodeAtPath]:
        """Each node of ``_tree``, by its path."""
        return {node.path: node for node in self._tree}

    @functools.cached_property
    def _functions(self) -> dict[NodePath, PlacedFunction]:
        """Every function node of ``_tree``, by its path."""
        return {node.path: node.placed for node in self._tree if isinstance(node, FunctionAtPath)}

    @functools.cached_property
    def _threads(self) -> int:
        """How many calls a leg over the graph can have on threads at once: the ``concurrency`` of every node
        called on a thread, those of each embedded graph with them."""
        return sum(policy.concurrency for placed in self._functions.values() if (policy := placed.definition.policy).threaded)

    def _given_by_address(self, given: Mapping[str, Any]) -> tuple[dict[str, Any], list[Problem]]:
        """What a caller gave by name or by address, by address; with a problem for each name the graph doesn't
        offer and each input given by both its name and its address."""
        fields = {**{f.address: f for f in self._input_fields.values()}, **{f.name: f for f in self._input_fields.values()}}
        by_address: dict[str, Any] = {}
        problems: list[Problem] = []
        for name, value in given.items():
            offered = fields.get(name)
            if offered is None:
                problems.append(make_problem("not_offered", "inputs", name, name=name, inputs=list(self.inputs)))
            elif offered.address in by_address:
                problems.append(make_problem("input_given_twice", "inputs", offered.name, name=offered.name, address=offered.address))
            else:
                by_address[offered.address] = value
        return by_address, problems

    def _checked_inputs(self, given: Mapping[str, Any], problems: list[Problem]) -> dict[str, Any]:
        """The value of every graph input, by address: the caller's, read by the input's field, else its default.

        Raises:
            InvalidCallError: With ``problems`` and then every input that is missing or not of its type.
        """
        problems, values = list(problems), dict[str, Any]()
        for address, offered in self._input_fields.items():
            if address not in given:
                if offered.spec.default is REQUIRED:
                    problems.append(make_problem("missing_input", "inputs", offered.name, name=offered.name))
                values[address] = offered.spec.default
                continue
            value, spec = given[address], offered.spec
            try:
                values[address] = spec.whole.read(value)
            except FieldValueError as refused:
                whole = all(not error["loc"] for error in refused.ctx.get("errors", ()))  # the list itself was refused, not an item of it
                if not spec.many and type(value) is list and whole:  # the one mistake whose fix is in the graph
                    problems.append(make_problem(
                        "invalid_input", "inputs", offered.name, name=offered.name, expected=spec.whole.id, length=len(value), hint="make_a_series",
                    ))
                else:
                    problems.append(make_problem("invalid_input", "inputs", offered.name, name=offered.name, reason=refused.reason, **refused.ctx))
        if problems:
            raise InvalidCallError(problems)
        return values


# -- loading ------------------------------------------------------------------------------------


def _add_to_tree(graph: Graph, embedding: GraphAtPath | None, tree: list[NodeAtPath], beside: dict[NodePath, dict[str, NodeAtPath]]) -> None:
    """Add the nodes of ``graph`` to ``tree``, each after the nodes it reads, and those of each graph embedded
    in it before that graph's node.

    Args:
        graph: The graph: the one a run is of, or one embedded in it.
        embedding: The embedded graph's node that ``graph`` is the inner graph of; ``None`` for the run's own.
        tree: The nodes so far, added to.
        beside: The nodes of each graph placed so far by id, by the path of the node that embeds it, added to.
    """
    inside = embedding.path if embedding is not None else ()
    above = embedding.maps if embedding is not None else ()
    here = beside[inside] = {}
    for placed in graph._nodes.values():
        inputs: dict[str, tuple[InputPlan, GraphAtPath | None]] = {}
        for name, plan in placed.inputs.items():
            if embedding is not None and isinstance(plan, Open):  # the embedded graph's node is given it, as a caller would give it
                handed, giver = embedding.inputs[graph._input_fields[plan.address].name]
                inputs[name] = (handed, giver or embedding)
            else:
                inputs[name] = (plan, None)
        maps = (*above, *((here[port.node_id], port.output) for port in placed.scope))
        if isinstance(placed, PlacedGraph):
            node: NodeAtPath = (embedded := GraphAtPath(
                placed=placed, embedding=embedding, inside=inside, path=(*inside, placed.id), maps=maps, base=len(above), inputs=inputs,
            ))
            _add_to_tree(placed.inner, embedded, tree, beside)
            within = beside[node.path]
            embedded.offers = tuple(
                (address, Cell(last.placed.id, output, last.placed.scope, spec.param.exclusive_group is not None), last.maps, spec.many)
                for address in placed.inner.outputs
                for last, output in [(within[address.partition(".")[0]], address.partition(".")[2])] for spec in [last.placed.outputs[output]]
            )
            embedded.ends_silently = tuple(within[last.id] for last in placed.inner._nodes.values() if not last.outputs and not placed.inner._readers[last.id])
            made_from = [(within[cell.node_id], made_in) for _, cell, made_in, _ in embedded.offers] + [(last, last.maps) for last in embedded.ends_silently]
            for last, made_in in made_from:  # each unit of it is made from every unit inside, under its instance
                for read in (last, *(series for series, _ in made_in[len(maps):])):
                    node.reads[read] = len(maps)
        else:
            node = FunctionAtPath(
                placed=placed, embedding=embedding, inside=inside, path=(*inside, placed.id), maps=maps, base=len(above), inputs=inputs,
                policy=placed.definition.policy, checked=tuple(name for name, (plan, _) in inputs.items() if _read_before_a_call(plan)),
            )
            for name, (plan, giver) in inputs.items():
                if isinstance(plan, Fed):
                    within, base = (here, len(above)) if giver is None else (beside[giver.inside], giver.base)
                    for cell, depth in plan.reads:
                        read = within[cell.node_id]
                        node.reads[read] = min(base + depth, node.reads.get(read, base + depth))
                    if isinstance(plan, Gather):
                        node.gathered_through[name] = _gathered_through(node, (within[plan.series.node_id], plan.series.output), [within[s.node_id] for s in plan.sources])
        if maps and min(node.reads.values(), default=0) < len(maps):  # what it reads may be there before its instances are
            node.reads.setdefault(maps[-1][0], len(maps) - 1)
        here[placed.id] = node
        node.position = len(tree)
        tree.append(node)


def _gathered_through(node: FunctionAtPath, series: tuple[NodeAtPath, str], sources: list[NodeAtPath]) -> tuple[NodeAtPath, ...]:
    """Finds the nodes made per item of ``series`` that the ``sources`` of a gathering input of ``node`` rest on,
    in the tree's order, by each one's own ``reads``; and adds to ``node.reads`` what they read outside the
    items. So the unit that settles the input for every item at once makes the gathering node ready, and what
    a run asks of the items it asks of these nodes and no other."""
    through: set[NodeAtPath] = set()
    stack = sources
    while stack:
        inner = stack.pop()
        if series in inner.maps and inner not in through:
            through.add(inner)
            for read, places in inner.reads.items():
                if series in read.maps:
                    stack.append(read)
                else:
                    node.reads[read] = min(places, node.reads.get(read, places))
    return tuple(sorted(through, key=lambda inner: inner.position))


def load_graph(record: GraphRecord, definitions: Mapping[str, NodeDefinition]) -> Graph:
    """``record`` loaded against ``definitions``, the newest version of each node by id: the one way a host's
    ``Graph`` is made."""
    return _Loader(definitions, (), {}).load(record)


#: An embedded graph loaded alone, by its definition's id and version and the graphs it sits in.
_AloneKey = tuple[str, int, tuple[str, ...]]

#: What stands for a typed-in value its field's type refused, while a node is placed.
_REFUSED: Any = object()

#: Reads what a hook returns: fields by name, a name holding none of the marks an address is written with.
_HOOK_FIELDS = pydantic.TypeAdapter(dict[Annotated[str, pydantic.StringConstraints(min_length=1, pattern=r"^[^./@]+$")], Any])


@dataclass(frozen=True, slots=True)
class _Planned:
    """A node's inputs, planned.

    Attributes:
        inputs: How each input gets its value, by name.
        scope: The scope the node runs in.
        rests_on: The branches the node rests on.
        bound: What the node's type variables are bound to by what arrives on its edges; ``None`` for a node whose
            inputs hold no type variable, which is nearly every node, so nothing is bound or replaced for it.
    """

    inputs: dict[str, InputPlan]
    scope: Scope
    rests_on: frozenset[Branch]
    bound: Mapping[TypeVar, Any] | None


class _Loader:
    """Loads one record: the working state of placing its nodes, each after the nodes it reads.

    Attributes:
        definitions: The newest version of each node, by id.
        embedded_in: The ids of the graph nodes this record sits in, outermost first, so one that embeds itself
            is found.
        alone: Each embedded graph already loaded, shared by every loader one load makes, so each loads once.
        problems: What is wrong with the record, as found.
        nodes: Each node placed so far, by id.
        branches: The ways decisions must go for each placed node to hold a value, as far as load can tell: what
            every edge into one of its inputs rests on. It says which edges into one input are alternatives.
        opened: Each first node's open input, with its default, in the order placed.
        position: Each node's place in the record.
    """

    def __init__(self, definitions: Mapping[str, NodeDefinition], embedded_in: tuple[str, ...], alone: dict[_AloneKey, Graph]) -> None:
        self.definitions, self.embedded_in, self.alone = definitions, embedded_in, alone
        self.problems: list[Problem] = []
        self.nodes: dict[str, PlacedNode] = {}
        self.branches: dict[str, frozenset[Branch]] = {}
        self.opened: list[tuple[GraphInput, Field]] = []
        self.position: dict[str, int] = {}

    def load(self, record: GraphRecord) -> Graph:
        """The graph of ``record``: every node whose fields are known placed, and every problem noted."""
        refused = {n.id for n in record.nodes if not is_node_id(n.id)}
        self.problems += [make_problem("invalid_node_id", node_id) for node_id in dict.fromkeys(n.id for n in record.nodes) if node_id in refused]
        known: dict[str, NodeRecord] = {}
        twice: set[str] = set()
        for node_record in record.nodes:
            if node_record.id in known:
                twice.add(node_record.id)
            elif node_record.id not in refused:
                known[node_record.id] = node_record
        self.problems += [make_problem("duplicate_node_id", node_id) for node_id in sorted(twice)]
        self.position = {node_id: i for i, node_id in enumerate(known)}
        into: dict[str, dict[str, list[str]]] = {node_id: {} for node_id in known}
        seen: set[tuple[str, str]] = set()
        for source, target in record.edges:
            node_id, _, name = target.partition(".")
            if source.partition(".")[0] in refused or node_id in refused:
                continue  # its own invalid_node_id says why
            if (source, target) in seen:
                self.problems.append(make_problem("repeated_edge", node_id, name, source=source))
                continue
            seen.add((source, target))
            missing = [n for n in (source.partition(".")[0], node_id) if n not in known]
            if missing:
                self.problems.append(make_problem("edge_names_missing_node", node_id, name, node_id=missing[0], source=source, target=target))
            if node_id in known:  # the input is drawn as fed, whether or not its source exists
                into[node_id].setdefault(name, []).append(source)
        ordered = self._order(list(known), into)
        for node_id in ordered:
            match self._resolve(known[node_id]):
                case FunctionDefinition() as function:
                    self._place_function(known[node_id], function, into[node_id])
                case GraphDefinition() as graph:
                    self._place_graph(known[node_id], graph, into[node_id])
                case None:
                    pass
        names = [spec.name for _, spec in self.opened]
        inputs: dict[str, GraphInput] = {}
        input_fields: dict[str, InputField] = {}
        for graph_input, spec in self.opened:
            name = spec.name if names.count(spec.name) == 1 else graph_input.address
            inputs[name] = graph_input
            input_fields[graph_input.address] = InputField(name, graph_input.address, spec)
        readers: dict[str, list[PlacedNode]] = {node_id: [] for node_id in self.nodes}
        needs_from_run: list[tuple[str, FunctionDefinition[..., Any]]] = []
        for placed in self.nodes.values():
            match placed:
                case PlacedFunction(definition=function):
                    if function.from_run:
                        needs_from_run.append((placed.id, function))
                case PlacedGraph(inner=inner):
                    needs_from_run += inner._needs_from_run
            for source_id in placed.reads:
                readers[source_id].append(placed)
        return Graph(record, self.problems, inputs, self.nodes, input_fields, readers, needs_from_run)

    def _resolve(self, record: NodeRecord) -> NodeDefinition | None:
        """The definition a node of the record names, at its version; ``None`` with the problem when there is none."""
        newest = self.definitions.get(record.node)
        if newest is None:
            self.problems.append(make_problem("unknown_node", record.id, node=record.node, choices=sorted(self.definitions)))
            return None
        if record.version not in newest.versions:
            self.problems.append(make_problem("unknown_version", record.id, node=record.node, version=record.version, newest=newest.version))
            return None
        return newest.version_of(record.version)

    def _order(self, order: list[str], into: dict[str, dict[str, list[str]]]) -> list[str]:
        """The nodes with each after what its edges read, ties in the record's order.

        A node on a cycle is a problem. The first of a cycle's nodes in the record is placed before the node
        whose edge closes the cycle, so that edge brings it nothing, and the rest follow in order.
        """
        position = self.position
        reads = {n: frozenset({s.partition(".")[0] for sources in into[n].values() for s in sources} & position.keys()) for n in order}
        readers: dict[str, list[str]] = {n: [] for n in order}
        for n, sources in reads.items():
            for source in sources:
                readers[source].append(n)
        waiting = {n: len(sources) for n, sources in reads.items()}
        ready = [position[n] for n in order if not waiting[n]]
        heapq.heapify(ready)
        done: list[str] = []
        on_cycles: list[str] | None = None
        while len(done) < len(order):
            if not ready:  # every node left waits on a cycle
                if on_cycles is None:
                    stuck = {n for n in order if waiting[n]}
                    on_cycles = [n for n in order if n in stuck and _reaches(n, n, readers, stuck)]  # on a cycle, not only below one
                    self.problems += [make_problem("cycle", n) for n in on_cycles]
                first = next(n for n in on_cycles if waiting[n] > 0)
                waiting[first] = 0
                ready.append(position[first])
            n = order[heapq.heappop(ready)]
            done.append(n)
            for reader in readers[n]:
                waiting[reader] -= 1
                if not waiting[reader]:
                    heapq.heappush(ready, position[reader])
        return done

    # -- placing a node ---------------------------------------------------------

    def _place_function(self, record: NodeRecord, definition: FunctionDefinition[..., Any], edges: dict[str, list[str]]) -> None:
        """Place a function node: its declared inputs and the ones its hook adds planned, then its outputs, as
        declared or as its hook makes them of the types arriving. A node a hook gives no fields for is left out;
        any other is kept, whatever problems its inputs have."""
        fields = dict(definition.inputs)
        typed = self._typed(record, fields)
        added: dict[str, Field] = {}
        if definition.compute_added_inputs is not None:
            made = self._hook(record, definition, "compute_added_inputs", definition.compute_added_inputs, typed, {}, ())
            if made is None:
                return
            added = made
            taken = sorted(added.keys() & (fields.keys() | definition.from_run.keys()))
            if taken:
                self.problems.append(make_problem("compute_hook_failed", record.id, reason=f"it adds {taken[0]!r}, which the node already declares", error="added_input_taken"))
                return
            fields.update(added)
            typed.update(self._typed(record, added))
        planned = self._planned(record, fields, typed, read_by_hooks(definition), edges, generic=definition.generic)
        bound = planned.bound
        outputs: dict[str, Field] | None = dict(definition.outputs) if bound is None else {name: spec.bind(bound) for name, spec in definition.outputs.items()}
        if definition.compute_outputs is not None:
            arriving = {name: plan.sources[0].type.bare for name, plan in planned.inputs.items() if isinstance(plan, Fed)}
            outputs = self._hook(record, definition, "compute_outputs", definition.compute_outputs, typed, arriving, tuple(added))
        if outputs is not None:
            self._keep(PlacedFunction(
                id=record.id, scope=planned.scope, inputs=planned.inputs, outputs=outputs, definition=definition,
            ), planned.rests_on)

    def _place_graph(self, record: NodeRecord, definition: GraphDefinition, edges: dict[str, list[str]]) -> None:
        """Place an embedded graph as one node: the inner graph loaded alone, its inputs and outputs this node's fields.

        An input is offered under the name the inner graph gives it (``graph.inputs``), and an output under its
        address in the inner graph (``graph.outputs``), so an edge reads ``e.upper.result``. An output made once
        inside keeps its field. One made per item inside, at any depth, is a ``Series`` of every item's value in
        order, as ``run.outputs`` gives it: a ``Series`` made per item is a ``Series`` of lists, and a table made
        per item a ``Series`` of tables.

        A graph that embeds itself is a problem here. The inner graph's own problems are this graph's, each with
        this node's id in front of its place. Either way the node is left out: what a graph with problems offers
        isn't known.
        """
        if definition.id in self.embedded_in:
            self.problems.append(make_problem("embeds_itself", record.id, node=definition.id))
            return
        stack = (*self.embedded_in, definition.id)
        key = (definition.id, definition.version, stack)
        if key not in self.alone:
            self.alone[key] = _Loader(self.definitions, stack, self.alone).load(definition.graph)
        inner = self.alone[key]
        if inner.problems:
            self.problems += [problem.model_copy(update={"loc": (record.id, *problem.loc)}) for problem in inner.problems]
            return
        fields = {offered.name: Field(offered.name, offered.spec.hint, offered.spec.default) for offered in inner._input_fields.values()}
        outputs: dict[str, Field] = {}
        for address in inner.outputs:
            node_id, _, output = address.partition(".")
            at = inner._nodes[node_id]
            spec = at.outputs[output]
            group = spec.param.exclusive_group
            if group:  # a decision inside is this node's, named by where it is made
                spec = Field(spec.name, Annotated[spec.hint, Param(exclusive_group=f"{node_id}.{group}")], spec.default)
            outputs[address] = _of_every_item(address, spec) if at.scope else Field(address, spec.hint, spec.default)
        planned = self._planned(record, fields, self._typed(record, fields), frozenset(), edges)
        self._keep(PlacedGraph(
            id=record.id, scope=planned.scope, inputs=planned.inputs, outputs=outputs, definition=definition, inner=inner,
        ), planned.rests_on)

    def _keep(self, placed: PlacedNode, rests_on: frozenset[Branch]) -> None:
        """Keep a node that placed, and note on each node whose ``Series`` it reads one item at a time that the
        Series is worked through, so a run holds it to ``max_items``."""
        for plan in placed.inputs.values():
            for source in plan.sources if isinstance(plan, Read) else ():
                if isinstance(source, Item):
                    self.nodes[source.node_id].opens.add(source.output)
        self.nodes[placed.id] = placed
        self.branches[placed.id] = rests_on

    def _planned(
        self, record: NodeRecord, fields: Mapping[str, Field], typed: Mapping[str, Any], fixed_at_load: frozenset[str],
        edges: dict[str, list[str]], *, generic: bool = False,
    ) -> _Planned:
        """Plan every input of a node, and settle the scope it runs in.

        An input whose edges bring nothing it takes, or can't meet the node's other inputs in one scope, is
        ``Unset``, with the problem: the node is kept, for an editor to draw, in a graph that never runs.

        Args:
            record: The node in the record.
            fields: The node's inputs, by name.
            typed: Each value and default typed into the record, as its field read it, by name.
            fixed_at_load: The inputs a hook reads when the graph loads, which a caller therefore never fills.
            edges: The addresses wired into each input, by name.
            generic: Whether a field holds a type variable for the edges to bind.
        """
        for name in sorted({*record.values, *record.defaults, *edges} - fields.keys()):
            self.problems.append(make_problem("unknown_input", record.id, name, inputs=list(fields)))
        sources = {name: self._sources(record.id, name, edges[name]) for name in fields if name in edges}
        bound = None
        if generic:
            bound = _bound(fields, sources)
            fields = {name: spec.bind(bound) for name, spec in fields.items()}  # a placed node's fields hold no type variable
        first = not any(name in fields for name in edges)  # a first node: nothing is wired into it, so the caller fills it
        inputs: dict[str, InputPlan] = {}
        for name, spec in fields.items():
            if name in record.values and name in record.defaults:
                self.problems.append(make_problem("value_and_default", record.id, name))
            if name in edges:
                if name in record.values or name in record.defaults:
                    self.problems.append(make_problem("value_and_edge", record.id, name))
                brought = sources[name]
                inputs[name] = Unset(spec) if brought is None else self._fed(record.id, spec, brought)
            elif name in record.values:
                inputs[name] = _fixed(spec, typed[name])
            elif first and name not in fixed_at_load:
                default = typed.get(name, spec.default)
                if default is _REFUSED:
                    inputs[name] = Unset(spec)
                    continue
                address = f"{record.id}.{name}"
                opened = spec if default is spec.default else Field(name, spec.hint, default)  # the record's default is the caller's
                self.opened.append((GraphInput(address=address, node_id=record.id, field=opened.description), opened))
                inputs[name] = Open(spec, address)
            elif name in record.defaults:
                self.problems.append(make_problem("default_on_later_node", record.id, name))
                inputs[name] = _fixed(spec, typed[name])
            elif spec.default is REQUIRED and name not in fixed_at_load:  # a hook that reads it says so itself, as hook_needs_value
                self.problems.append(make_problem("unbound_on_later_node", record.id, name))
                inputs[name] = Unset(spec)
            else:
                inputs[name] = Fixed(spec, spec.default)
        for name in self._never_meet(record.id, inputs):
            inputs[name] = Unset(fields[name])
        scopes = [plan.scope for plan in inputs.values() if isinstance(plan, Fed)]
        scope = max(scopes, key=len, default=())
        for name, plan in inputs.items():
            if isinstance(plan, Fed) and scope[: len(plan.scope)] != plan.scope:  # items of two Series never pair up
                series = [scope[-1], plan.scope[-1]]
                outer, inner = sorted(series, key=lambda port: list(self.nodes).index(port.node_id))  # the one placed later can read the other with no cycle
                reading = next(n for n, p in inputs.items() if isinstance(p, Fed) and p.scope == scope)
                self.problems.append(make_problem(
                    "series_not_nested", record.id, series=[str(s) for s in series], inputs=[reading, name],
                    wire_from=str(outer), wire_into=inner.node_id,
                ))
                inputs[name] = Unset(plan.spec)
        return _Planned(inputs, scope, self._rests_on(inputs), bound)

    def _typed(self, record: NodeRecord, fields: Mapping[str, Field]) -> dict[str, Any]:
        """Each value typed into the record for one of ``fields``, and else each default, as the field reads it;
        ``_REFUSED``, with the problem, where it can't."""
        typed: dict[str, Any] = {}
        for name, spec in fields.items():
            if name not in record.values and name not in record.defaults:
                continue
            try:
                typed[name] = spec.whole.read(record.values[name] if name in record.values else record.defaults[name])
            except FieldValueError as refused:
                self.problems.append(make_problem("invalid_value", record.id, name, reason=refused.reason, **refused.ctx))
                typed[name] = _REFUSED
        return typed

    def _hook(
        self, record: NodeRecord, definition: FunctionDefinition[..., Any], kind: HookName, hook: Callable[..., Mapping[Any, Any]],
        typed: Mapping[str, Any], arriving: Mapping[str, Any], added: tuple[str, ...],
    ) -> dict[str, Field] | None:
        """Call a hook with what its parameters read: its answer read as fields, or ``None`` with the problem.

        Args:
            record: The node in the record.
            definition: The node's definition.
            kind: Which hook it is.
            hook: The hook.
            typed: Each value and default typed into the record, as its field read it, by input.
            arriving: The type arriving on each input edges feed, by input; ``Any`` arrives on any other.
            added: The inputs ``compute_added_inputs`` added, which ``**added: type`` reads the arriving types of.
        """
        kwargs: dict[str, Any] = {}
        for name, receives in definition.hook_parameters[kind].items():
            match receives:
                case "arriving_type":
                    kwargs[name] = arriving.get(name, Any)
                case "added_arriving_types":
                    kwargs.update({input_name: arriving.get(input_name, Any) for input_name in added})
                case "value":
                    value = typed.get(name, definition.inputs[name].default)  # @node checked the parameter names a declared input
                    if value is REQUIRED:
                        self.problems.append(make_problem("hook_needs_value", record.id, name))
                        return None
                    if value is _REFUSED:
                        return None  # its own invalid_value says why
                    kwargs[name] = value
        try:
            returned = hook(**kwargs)
        except NodeError as error:  # the hook's own account, with its own code and its ctx kept whole
            own = {"error_ctx": dict(error.ctx)} if error.ctx else {}
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=error.message, error=error.code, **own))
            return None
        except Exception as error:  # noqa: BLE001  the hook's failure is this node's problem
            _logger.error("the %s hook of node %r failed", kind, record.id, exc_info=error)
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=str(error), error=type(error).__name__))
            return None
        try:
            hints = _HOOK_FIELDS.validate_python(returned)
        except pydantic.ValidationError as invalid:
            for fault in invalid.errors():
                if fault["loc"]:  # a name refused: the place is the name
                    self.problems.append(make_problem("parameter_name_invalid", record.id, str(fault["loc"][0])))
                else:  # the value itself refused: it is no mapping
                    reason = f"its {kind} hook returned a {type(returned).__name__}; a hook returns its fields by name, as a mapping of each name to its type"
                    self.problems.append(make_problem("compute_hook_failed", record.id, reason=reason, error="not_a_mapping"))
            return None
        try:
            return {name: Field(name, *_hint_and_default(hint)) for name, hint in hints.items()}
        except NodeDeclarationError as error:  # a hint conductor can't carry is the hook's failure
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=error.message, error=error.code))
            return None
        except AttributeError as error:
            if error.name not in ("type_id", "type_title"):  # any other is a bug in the host's own code, as at @node
                raise
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=str(error), error="AttributeError"))
            return None

    # -- edges ------------------------------------------------------------------

    def _sources(self, node_id: str, name: str, addresses: list[str]) -> list[Source] | None:
        """What the edges from ``addresses`` bring one input, in the order their nodes and outputs have in the record.

        Returns:
            ``None`` when an edge names nothing a node gives, with the problem, and ``None`` when its node's
            fields aren't known or it is placed later, on a cycle.
        """
        sources: list[Source] = []
        for address in addresses:
            source = self._source(address, node_id, name)
            if source is None:
                return None
            sources.append(source)
        if len(sources) > 1:
            sources.sort(key=self._edge_order)
        return sources

    def _fed(self, node_id: str, spec: Field, sources: list[Source]) -> Fed | Unset:
        """How edges bringing ``sources`` feed one input: one value on each call, or every value its source made.

        An input of one value reads, whatever its type: a ``Series`` reaches it one item at a time. A ``Series``
        input gathers. A ``Table`` input reads when every edge brings a table, a ``Series`` of tables one table at
        a time, and else gathers records.

        Returns:
            The input's plan; ``Unset``, with the problem, when an edge brings a type the input doesn't take.
        """
        gathers, wanted = _taken(spec, sources)
        for source in sources:
            if wanted is not source.type and wanted.fits(source.type) is None:  # the same type, which most edges join, fits
                self.problems.append(_mismatch(node_id, spec, source))
                return Unset(spec)
        deepest = max((s.read_in for s in sources), key=len)
        if not gathers:
            return Read(spec, tuple((s, len(s.scope)) for s in sources), deepest)
        if not deepest:
            return SeriesOfOne(spec, tuple((s, 0) for s in sources if isinstance(s, Value)))
        made_by, output = deepest[-1]
        series = Cell(made_by, output, deepest[:-1], self.nodes[made_by].outputs[output].param.exclusive_group is not None)
        made_in = len(series.scope)
        return Gather(spec, ((series, made_in), *((s, made_in) for s in sources)))

    def _source(self, address: str, node_id: str, name: str) -> Source | None:
        """What an edge from ``address`` brings: ``node.output``, or ``node.output.field`` for one field of a record.

        The output is the one of the node's whose name the address goes on with after the node's id: an embedded
        graph's is itself an address, ``e.upper.result``.

        Returns:
            An item of a ``Series`` output, or an output's one value; ``None`` with the problem when the address
            names nothing its node gives, and ``None`` when its node isn't placed.
        """
        source_id, _, rest = address.partition(".")
        source = self.nodes.get(source_id)
        if source is None:  # a node whose fields aren't known, one placed later on a cycle, or none: its own problem says why
            return None
        output, _, wanted = rest.partition(".")
        if output not in source.outputs:  # an embedded graph's output: the longest part of the address that names one
            output = next((declared for declared in source.outputs if rest == declared or rest.startswith(f"{declared}.")), "")
            wanted = rest[len(output) + 1 :]
        spec = source.outputs.get(output)
        if spec is None:
            self.problems.append(make_problem("unknown_output", node_id, name, source=address, outputs=[f"{source_id}.{o}" for o in source.outputs]))
            return None
        if wanted and (split := _split_gives(spec.type)):  # a table or a list is one value: only an item of a Series has a field to wire
            self.problems.append(make_problem(
                "unknown_output", node_id, name, source=address, outputs=[f"{source_id}.{o}" for o in source.outputs], field=wanted,
                hint="split_first", whole=split[0], fields=split[1],
            ))
            return None
        brings = _field_type(spec.type, wanted) if wanted else spec.type
        if brings is None:
            fields = [f"{source_id}.{output}.{f}" for f in spec.type.fields]
            self.problems.append(make_problem("unknown_output", node_id, name, source=address, outputs=fields, field=wanted))
            return None
        brought = Item if spec.many else Value
        return brought(source_id, output, source.scope, spec.param.exclusive_group is not None, wanted or None, brings)

    def _edge_order(self, source: Source) -> tuple[int, int, str]:
        """Where an edge into an input falls among several: its node's place in the record, then the output's in its node."""
        return self.position[source.node_id], list(self.nodes[source.node_id].outputs).index(source.output), source.address

    # -- branches ---------------------------------------------------------------

    def _branch(self, source: Source) -> frozenset[Branch]:
        """The ways decisions must go for ``source`` to hold a value."""
        group = self.nodes[source.node_id].outputs[source.output].param.exclusive_group
        own = {Branch(source.node_id, group, source.output)} if group else set[Branch]()
        return self.branches[source.node_id] | own

    def _never_meet(self, node_id: str, inputs: Mapping[str, InputPlan]) -> set[str]:
        """Refuse edges that can never bring a node a value together.

        Several edges into one input are alternatives: each on its own branch of one decision, so a call never has
        two values for it, and all in one scope, so each call has a place for the one it gets. Two ``Series``
        inputs that keep no blanks, fed only from opposite branches, would lose every item.

        Returns:
            The inputs whose edges don't meet these, each with its problem.
        """
        refused: set[str] = set()
        for name, plan in inputs.items():
            if not isinstance(plan, Fed):
                continue
            behind = [self._branch(s) for s in plan.sources]
            scopes = list(dict.fromkeys(s.read_in for s in plan.sources))
            if not all(_opposed(a, b) for i, a in enumerate(behind) for b in behind[i + 1:]):
                self.problems.append(make_problem("several_edges_into_one_input", node_id, name, sources=[s.address for s in plan.sources]))
                refused.add(name)
            elif len(scopes) > 1:
                series = [str(scope[-1]) if scope else None for scope in scopes[:2]]
                self.problems.append(make_problem("alternatives_in_different_series", node_id, name, series=series))
                refused.add(name)
        aligned = [
            (name, [self._branch(s) for s in plan.sources]) for name, plan in inputs.items()
            if isinstance(plan, Gather) and not plan.spec.keeps_blanks and name not in refused
        ]
        for i, (a, behind_a) in enumerate(aligned):
            for b, behind_b in aligned[i + 1:]:
                if all(_opposed(x, y) for x in behind_a for y in behind_b):
                    self.problems.append(make_problem("series_inputs_never_share_an_item", node_id, inputs=[a, b]))
                    refused.update((a, b))
        return refused

    def _rests_on(self, inputs: Mapping[str, InputPlan]) -> frozenset[Branch]:
        """The branches a node rests on: for each input edges feed, what every edge into it rests on. A gather
        rests only on what is decided where it runs or above, as a decision made per item skips items, never it."""
        rests_on: set[Branch] = set()
        for plan in inputs.values():
            if not isinstance(plan, Fed):
                continue
            behind = [self._branch(s) for s in plan.sources]
            shared = behind[0].intersection(*behind[1:])
            if isinstance(plan, Gather):
                shared = frozenset(b for b in shared if len(self.nodes[b.node_id].scope) <= len(plan.scope))
            rests_on |= shared
        return frozenset(rests_on)


def _taken(spec: Field, sources: list[Source]) -> tuple[bool, Type]:
    """Whether the input ``spec`` gathers what ``sources`` bring, and the type each must fit: the input's whole
    type when it reads, one item of a ``Series`` or one record of a table when it gathers.

    A table input reads when every edge brings a table, as one value or as each item of a ``Series`` of tables.
    A value declared ``Any`` says nothing of itself: it is read as a table only where there is nothing to gather,
    made once in the graph, and is taken as a record wherever it is made per item.
    """
    record = spec.whole.record
    gathers = spec.many or (record is not None and not all(
        spec.whole.fits(s.type) is not None and (s.type.bare is not Any or not s.read_in) for s in sources
    ))
    return gathers, spec.whole if not gathers else spec.type if record is None else record


def _bound(fields: Mapping[str, Field], sources: Mapping[str, list[Source] | None]) -> dict[TypeVar, Any]:
    """What a node's type variables are bound to by what arrives on its edges; two places sharing a variable join.

    Only an edge binds: a typed-in value, a default and an open input bind nothing, and a variable nothing binds
    reads as its bound, else ``Any``.
    """
    bound: dict[TypeVar, Any] = {}
    for name, spec in fields.items():
        brought = sources.get(name)
        if not spec.whole.variables or not brought:
            continue
        wanted = _taken(spec, brought)[1]
        for source in brought:
            for variable, hint in (wanted.fits(source.type) or {}).items():
                bound[variable] = bound[variable] | hint if variable in bound else hint
    return bound


def _split_gives(whole: Type) -> tuple[Literal["table", "list"], list[str]] | None:
    """What a Split would make of a value of the type ``whole``.

    Returns:
        Which whole value it takes apart, and the fields of each record or item it gives (none for one of unknown
        columns, and for an item that is no record); ``None`` for a type that is neither a table nor a list.
    """
    if whole.record is not None:
        return "table", list(whole.fields)
    one = whole.options[0] if len(whole.options) == 1 else None
    if one is list or typing.get_origin(one) is list:
        return "list", list(Type(next(iter(typing.get_args(one)), Any)).fields)
    return None


def _mismatch(node_id: str, spec: Field, source: Source) -> Problem:
    """The ``type_mismatch`` of an edge from ``source`` into the input ``spec``.

    It says the fix where one mistake explains it: a whole table or list wired to an input of one value wants a
    Split first, and a table input fed records without a column it names wants that column.
    """
    said: dict[str, CtxValue] = {
        "source": source.address,
        "got": f"Series[{source.type.text}]" if isinstance(source, Item) else source.type.text,
        "expected": spec.whole.id,
    }
    split = _split_gives(source.type)
    if spec.whole.record is not None:
        has = (source.type.record or source.type).fields
        missing = [column for column in spec.whole.fields if column not in has]
        if has and missing:
            said.update(hint="add_column", column=missing[0], column_type=Type(spec.whole.fields[missing[0]]).id, has=list(has))
    elif split and not spec.many:
        said.update(hint="split_first", whole=split[0], fields=split[1])
    return make_problem("type_mismatch", node_id, spec.name, **said)


def _fixed(spec: Field, typed: Any) -> Fixed | Unset:
    """The plan of an input holding a typed-in value, as its field read it."""
    return Unset(spec) if typed is _REFUSED else Fixed(spec, typed)


def _of_every_item(name: str, spec: Field) -> Field:
    """The field ``name`` of an output made once per item, as the ``Series`` of every item's value: of each list, for
    an output that is itself a ``Series``."""
    series: Any = Series
    return Field(name, series[types.GenericAlias(list, (spec.type.hint,)) if spec.many else spec.hint])


def _runs_on(scope: Scope) -> RunsOn:
    """Where a node of ``scope`` runs."""
    return RunsOn(mode="per_item", series=str(scope[-1]), msg=f"per item of {scope[-1]}") if scope else RunsOn(mode="once", msg="once")


def _feed(scope: Scope, plan: InputPlan) -> Feed:
    """How ``plan`` feeds its input on each call of a node that runs in ``scope``."""
    match plan:
        case Gather(series=series):
            return Feed(mode="gathered", series=series.address, msg=f"every item of {series.address}")
        case SeriesOfOne():
            return Feed(mode="gathered", msg="the one value, as a Series of one")
        case Read(scope=fed_in) if fed_in != scope:
            return Feed(mode="outer_value", series=str(fed_in[-1]) if fed_in else None, msg=f"the value of {f'the item of {fed_in[-1]}' if fed_in else 'the graph'}")
        case Read(scope=fed_in) if fed_in:
            return Feed(mode="per_item", series=str(fed_in[-1]), msg=f"per item of {fed_in[-1]}")
        case Unset():
            return Feed(mode="unset", msg="no value: a problem of the graph says why")
        case _:
            return Feed(mode="one_value", msg="one value")


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


def _field_type(record: Type, name: str) -> Type | None:
    """The type of the field ``name`` of a record of the type ``record``.

    Returns:
        A ``TypedDict``'s, dataclass's or pydantic model's declared type for it; a mapping's value type, ``Any``
        for a bare ``dict`` and for a value of any type; ``None`` for a record without the field, and for a type
        that is no record.
    """
    one = record.options[0] if len(record.options) == 1 else None
    if one is Any or one is dict:
        return Type(Any)
    if typing.get_origin(one) in (dict, Mapping):
        return Type(typing.get_args(one)[1])
    return Type(record.fields[name]) if name in record.fields else None


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """What a hook gives for one field: a hint, or ``(hint, default)`` as pydantic's ``create_model`` takes one."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED


def _code_text(code: Any) -> str:
    """A function's bytecode and constants as text that is the same in every process: a nested function by its own code."""
    return f"{code.co_code.hex()}:{[_constant_text(c) for c in code.co_consts]}:{code.co_names}"


def _constant_text(constant: Any) -> str:
    """A constant as text that is the same in every process: a set's members sorted, as its order follows the hash seed."""
    if hasattr(constant, "co_code"):
        return _code_text(constant)
    if isinstance(constant, frozenset):
        return f"frozenset({sorted(_constant_text(c) for c in constant)})"
    if isinstance(constant, tuple):
        return f"({', '.join(_constant_text(c) for c in constant)},)"
    return repr(constant)


def _opposed(a: frozenset[Branch], b: frozenset[Branch]) -> bool:
    """Whether two sets of branches never hold together: one decision goes one way for ``a`` and another for ``b``."""
    return any(x[:2] == y[:2] and x.output != y.output for x in a for y in b)


def _reaches(start: str, goal: str, readers: Mapping[str, list[str]], within: set[str]) -> bool:
    """Whether ``goal`` is read, however far down, by what ``start`` feeds, staying ``within`` those nodes."""
    seen: set[str] = set()
    stack = list(readers[start])
    while stack:
        n = stack.pop()
        if n == goal:
            return True
        if n in within and n not in seen:
            seen.add(n)
            stack.extend(readers[n])
    return False
