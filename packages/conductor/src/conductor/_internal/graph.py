"""``Graph``: a stored graph checked against the nodes it uses, ready to run.

Loading does everything once: every edge and value is checked, and each node
gets the scope it runs in. What is wrong becomes a ``Problem`` with a place,
never an exception.

Scopes. A node at the graph's own scope, ``()``, runs once. A ``Series[T]``
output read by an input that takes one ``T`` opens a map: the reader runs once
per item. A map is named by the address of its Series (``split.result``), and
a scope is the path of maps from the graph's own scope,
``("docs.result", "extract.lines")``. Declared types alone decide every
scope, by three rules:

- an input of one ``T`` fed by a ``Series[T]`` reads one item, inside the map
  that Series opens;
- an input fed from an enclosing scope captures that scope's one value;
- an input of ``Series[T]``, or of ``Table[R]`` fed records, gathers: it
  receives every value its source made: the items of the innermost map of
  what feeds it, in order, and it sits in the scope that map's Series was
  made in; or, from a source that runs once, that one value as a Series of
  one.

A node runs in the deepest of its inputs' scopes, and every input's scope
lies on the path to it: items of two Series never pair up
(``lists_not_collected``). Readers of one Series read one map.

A table is one value. An embedded graph is one node: its inputs are the inner
graph's inputs, its outputs the inner graph's outputs, and a call of it runs
the inner graph.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import heapq
import inspect
import itertools
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from conductor._internal.ask import output_questions
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import InvalidCallError, NodeError, Problem, make_problem
from conductor._internal.fields import (
    REQUIRED,
    FieldDescription,
    FieldSpec,
    FieldValueError,
    accepts,
    already,
    read_field,
    read_given,
    record_fields,
    takes_a_blank,
    type_id,
    without_none,
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
    from conductor._internal.run import Run

#: A path of maps from the graph's own scope, each named by the address of the Series it works through.
Scope = tuple[str, ...]


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
class Value:
    """An output read as the one value it is, where its node runs or in a scope under it.

    Attributes:
        node_id: The node whose output it is.
        output: The output's name.
        field: The field of the value an edge names (``read.result.email``); ``None`` for the value itself.
        scope: The scope the node runs in.
        type: The type of what it brings.
    """

    node_id: str
    output: str
    field: str | None
    scope: Scope
    type: Any

    @property
    def address(self) -> str:
        """The address an edge names it by."""
        return f"{self.node_id}.{self.output}.{self.field}" if self.field else f"{self.node_id}.{self.output}"


@dataclass(frozen=True, slots=True)
class Item:
    """One item of a ``Series`` output, read inside the map the Series opens.

    Attributes:
        node_id: The node whose output the Series is.
        output: The output's name.
        field: The field of each item an edge names (``split.result.email``); ``None`` for the item itself.
        scope: The map's scope: the scope the node runs in, then the Series.
        type: The type of what it brings.
    """

    node_id: str
    output: str
    field: str | None
    scope: Scope
    type: Any

    @property
    def address(self) -> str:
        """The address an edge names it by."""
        return f"{self.node_id}.{self.output}.{self.field}" if self.field else f"{self.node_id}.{self.output}"


#: What an edge brings its input.
Source = Value | Item

#: What can skip one edge of a gather for every item at once: sets of outputs made where the gathering node runs
#: or above it. The edge brings nothing when every output of any one set is a branch not taken.
Gate = tuple[frozenset[Value], ...]


# -- how an input gets its value ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fixed:
    """An input that holds one value on every call: typed into the node, or its default.

    Attributes:
        spec: The input's field.
        value: The value, as the field's type read it.
    """

    spec: FieldSpec
    value: Any


@dataclass(frozen=True, slots=True)
class Open:
    """An input of a first node that nothing fills: the graph's input, which the caller gives.

    Attributes:
        spec: The input's field.
        address: Where a run keeps the caller's value, ``node.input``.
    """

    spec: FieldSpec
    address: str


@dataclass(frozen=True, slots=True)
class Read:
    """An input that edges bring one value on each call.

    Attributes:
        spec: The input's field.
        sources: What each edge brings. Several are alternatives, on opposite branches of one decision: a call
            receives the one whose branch was taken.
        scope: The scope the value sits in.
    """

    spec: FieldSpec
    sources: tuple[Source, ...]
    scope: Scope


@dataclass(frozen=True, slots=True)
class Gather:
    """An input of many (a ``Series``, or a ``Table`` fed records) that receives the items of one map as a list.

    Attributes:
        spec: The input's field.
        sources: What each edge brings for an item. Several are alternatives, as for any input.
        gates: For each of ``sources``, what skips it for every item at once.
        series: The Series whose items are gathered, read whole where it was made.
        scope: The scope the list sits in: where ``series`` was made.
    """

    spec: FieldSpec
    sources: tuple[Source, ...]
    gates: tuple[Gate, ...]
    series: Value
    scope: Scope


@dataclass(frozen=True, slots=True)
class SeriesOfOne:
    """An input of many fed by a source that runs once: each call receives its one value as a list of one.

    Attributes:
        spec: The input's field.
        sources: What each edge brings. Several are alternatives, as for any input.
        scope: The graph's own scope, where such a source runs.
    """

    spec: FieldSpec
    sources: tuple[Value, ...]
    scope: Scope


#: An input that edges feed.
Fed = Read | Gather | SeriesOfOne

#: How one input of a placed node gets its value.
InputPlan = Fixed | Open | Fed


# -- a node of a loaded graph -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class _Placed:
    """What every node of a loaded graph has, whether it is a function or an embedded graph. Never made on its own.

    Attributes:
        id: The node's id in the record.
        scope: The scope it runs in.
        inputs: How each input gets its value, by name; the fields are the node's as its hooks made them.
        outputs: Each output's field, by name, as its hooks made them.
        reads: The nodes its edges read.
        checked: The inputs whose value is read by the input's type before a call, as the types its edges bring
            don't settle it.
        opens: The ``Series`` outputs some reader works through one item at a time, which a run holds to
            ``max_items``. A reader placed after this node adds to it.
    """

    id: str
    scope: Scope
    inputs: dict[str, InputPlan]
    outputs: dict[str, FieldSpec]
    reads: frozenset[str]
    checked: tuple[str, ...]
    opens: set[str] = dataclasses.field(default_factory=set[str])

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input can hold a mapping, so what a unit received needs no key sorting to hash."""
        return all(plan.spec.shape == "table" or plan.spec.plain for plan in self.inputs.values())


@dataclass(frozen=True, kw_only=True)
class PlacedFunction(_Placed):
    """A node of a loaded graph that is a function.

    Attributes:
        definition: The node as its author declared it.
    """

    definition: FunctionDefinition[..., Any]

    @property
    def policy(self) -> Policy:
        """How the engine calls it."""
        return self.definition.policy

    @functools.cached_property
    def reuse_digest(self) -> str:
        """What a unit's result rests on besides what it received: the node's version, its code, its fields and its scope.

        So a node redeclared under the same id and version, with other code or a signature that takes a blank or
        runs in another scope, is called again by ``run.again``. Code is read from the function's own bytecode and
        constants, not from what it calls.
        """
        specs = (*(plan.spec for plan in self.inputs.values()), *self.outputs.values())
        fields = [(spec.name, spec.shape, spec.optional, _type_text(spec.type)) for spec in specs]
        definition = self.definition
        written = repr((definition.id, definition.version, _code_text(inspect.unwrap(definition.function).__code__), fields, self.scope))
        return hashlib.blake2b(written.encode(), digest_size=8).hexdigest()

    @functools.cached_property
    def questions(self) -> dict[str, FieldDescription]:
        """What a person is asked for each output when the node raises ``Ask``, before the ``Ask`` adds to it."""
        return output_questions(self.outputs)


@dataclass(frozen=True, kw_only=True)
class PlacedGraph(_Placed):
    """A node of a loaded graph that is an embedded graph: a call of it runs ``inner``.

    Attributes:
        definition: The node as its author declared it.
        inner: The embedded graph, loaded.
        inner_inputs: The address in ``inner`` of each input this node offers, by the name it offers it under.
        inner_outputs: The address in ``inner`` of each output this node offers, by the name it offers it under.
    """

    definition: GraphDefinition
    inner: Graph
    inner_inputs: dict[str, str]
    inner_outputs: dict[str, str]

    @property
    def policy(self) -> Policy:
        """How the engine calls it: on the leg's own thread, as the nodes inside carry their own policies."""
        return _INLINE

    @functools.cached_property
    def reuse_digest(self) -> str:
        """What a unit's result rests on besides what it received: the node's version, the inner graph and its scope."""
        written = repr((self.definition.id, self.definition.version, self.inner.fingerprint, self.scope))
        return hashlib.blake2b(written.encode(), digest_size=8).hexdigest()


#: One node of a loaded graph: a function, or an embedded graph.
PlacedNode = PlacedFunction | PlacedGraph

#: The policy of a node called on the leg's own thread.
_INLINE = Policy()


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
        mode: ``single`` for one value; ``per_item`` for one item of ``series``; ``captured`` for the value of a
            scope that encloses the node's; ``gathered`` for every item of ``series`` as one list.
        series: The address of the Series the mode names; ``None`` for ``single``, for a value ``captured`` from
            the graph's own scope, and for the one value of a source that runs once, ``gathered`` as a Series of one.
        msg: The same, in English.
    """

    mode: Literal["single", "per_item", "captured", "gathered"]
    series: str | None = None
    msg: str


class PlacedFieldDescription(FieldDescription):
    """An input of a node in a loaded graph.

    Attributes:
        feed: How it is fed on each call of its node.
    """

    feed: Feed


class GraphNodeDescription(ConductorModel):
    """One node of a loaded graph as an editor draws it.

    Attributes:
        node: The id of the definition it uses.
        version: The version of that definition.
        loaded: Whether it loaded; ``problems`` says why when it didn't.
        problems: The problems placed on it.
        runs: Where it runs, once loaded.
        inputs: Its inputs as its hooks made them, once loaded.
        outputs: Its outputs as its hooks made them, once loaded.
    """

    node: str
    version: int
    loaded: bool
    problems: list[Problem]
    runs: RunsOn | None = None
    inputs: list[PlacedFieldDescription] = []
    outputs: list[FieldDescription] = []


class GraphInput(FieldDescription):
    """An open input of a first node, a value of ``graph.inputs``: what a caller fills.

    It is its node's field record; ``default`` is what it holds when the caller gives nothing (the record's
    default, else the node's), as JSON, and ``None`` when it is ``required``.

    Attributes:
        address: Where the caller's value goes, ``node.input``.
        node_id: The node whose input it is.
    """

    address: str
    node_id: str


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
        spec: Its field.
        default: What it holds when the caller gives nothing; ``REQUIRED`` when the caller must.
    """

    name: str
    address: str
    spec: FieldSpec
    default: Any


class Graph:
    """A graph loaded against a registry, with its own copy of the definitions it resolved.

    ``problems`` is everything wrong with it, and a graph with none runs:
    ``graph.run(**inputs)``. ``inputs`` are the open inputs of its first
    nodes, by name (``node.input`` where two share a name), and ``outputs`` the
    outputs of the nodes nothing reads.
    """

    record: GraphRecord
    problems: list[Problem]
    inputs: dict[str, GraphInput]
    #: Each node that loaded, by id, each after the nodes it reads.
    _nodes: dict[str, PlacedNode]
    #: Each graph input as a run keeps it, by address.
    _input_fields: dict[str, InputField]
    #: The ways decisions must go for each node to hold a value, as far as load can tell: what every edge into one
    #: of its inputs rests on. It only says which edges into one input are alternatives.
    _branches: dict[str, frozenset[Branch]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("a graph is loaded with registry.load(record)")

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
        read = {node_id for placed in self._nodes.values() for node_id in placed.reads}
        return [f"{p.id}.{o}" for p in self._nodes.values() if p.id not in read for o in p.outputs]

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

        Each node lists its own ``problems``, and ``problems`` beside the nodes lists every one. A node that didn't
        load (an unknown node, a broken edge, or one below it) is there with ``loaded`` false, its problems saying
        why. An embedded graph is one node, with the fields it offers.
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
                inputs=[plan.spec.describe(PlacedFieldDescription, feed=_feed(placed.scope, plan)) for plan in placed.inputs.values()],
                outputs=[spec.describe() for spec in placed.outputs.values()],
            )
        return GraphDescription(nodes=nodes, inputs=dict(self.inputs), outputs=self.outputs, problems=list(self.problems))

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of what a run of this graph rests on: each node's id, node, version, values and defaults,
        the edges, and the fingerprint of every graph embedded in it.

        Keys are sorted first, so a record a JSON store gives back in another key order has the same fingerprint.
        ``display`` is left out, as conductor never reads it. A saved run names it, and loading a save into a graph
        whose fingerprint differs is refused.
        """
        from conductor._internal.run import canonical_json

        own = self.record.model_dump(mode="json", include={"nodes": {"__all__": {"id", "node", "version", "values", "defaults"}}, "edges": True})
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
        from conductor._internal.run import checked_leg, start_run

        if self.problems:
            raise InvalidCallError(self.problems)
        checked_leg(max_items=max_items)
        run = start_run(self, self._checked_inputs(*self._given_by_address({**inputs, **by_name})), max_items)
        if earlier is not None:
            run._earlier = earlier._carried(into=self)
        return run

    def calls_for(self, lengths: Mapping[str, int] = types.MappingProxyType({}), /) -> dict[str, int | None]:
        """How many times each node is called at most, before any run, given how many items some Series will hold.

        ``graph.calls_for({"upload.files": 11})`` counts 11 calls for a node that runs per item of
        ``upload.files``. A branch not taken calls nothing, so each count is a most.

        Args:
            lengths: The number of items of a Series, by its address.

        Returns:
            Each node's count, by id; ``None`` for a node that runs per item of a Series ``lengths`` doesn't name,
            as how many it holds is known only once its node has run. An embedded graph counts the calls of every
            node in it.

        Raises:
            InvalidCallError: The graph has problems, a length is below 0, or one names a Series no node works
                through.
        """
        if self.problems:
            raise InvalidCallError(self.problems)
        series = {name for placed in self._nodes.values() for name in placed.scope}
        bad = {name: "a length is 0 or more" for name, n in lengths.items() if n < 0}
        bad.update({name: "no node of the graph runs per item of it" for name in lengths if name not in series})
        if bad:
            raise InvalidCallError([make_problem("invalid_input", "lengths", name, name=name, reason=reason) for name, reason in bad.items()])
        counts: dict[str, int | None] = {}
        for placed in self._nodes.values():
            factors = [lengths.get(name) for name in placed.scope]
            if isinstance(placed, PlacedGraph):
                inside = list(placed.inner.calls_for().values())
                factors.append(None if None in inside else sum(n for n in inside if n is not None))
            counts[placed.id] = None if None in factors else functools.reduce(lambda a, b: a * b, (n for n in factors if n is not None), 1)
        return counts

    def run(
        self, inputs: Mapping[str, Any] = types.MappingProxyType({}), /, *, max_items: int | None = None,
        from_run: Mapping[type, Any] = types.MappingProxyType({}), timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops."""
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(from_run=from_run, timeout=timeout)

    def load_run(self, saved: str | bytes | Mapping[str, Any]) -> Run:
        """A run of this graph from what ``run.to_json()`` wrote; a graph with problems refuses it with them.

        A save is read from its JSON text, a mapping written as JSON first, so a float marked as not finite is in
        place before each value's type reads it.
        """
        from conductor._internal.run import run_from_json

        return run_from_json(self, saved)

    def _above_of(self, node_id: str) -> frozenset[str]:
        """Every node ``node_id`` waits on, however far up; worked out when first asked, as most nodes never are."""
        found = self._above.get(node_id)
        if found is None:
            seen: set[str] = set()
            stack = [node_id]
            while stack:
                for read in self._nodes[stack.pop()].reads - seen:
                    seen.add(read)
                    stack.append(read)
            found = self._above[node_id] = frozenset(seen)
        return found

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
                if offered.default is REQUIRED:
                    problems.append(make_problem("missing_input", "inputs", offered.name, name=offered.name))
                values[address] = offered.default
                continue
            try:
                values[address] = read_given(offered.spec, given[address])
            except FieldValueError as refused:
                problems.append(make_problem("invalid_input", "inputs", offered.name, name=offered.name, reason=refused.reason, **refused.ctx))
        if problems:
            raise InvalidCallError(problems)
        return values

    # -- loading ----------------------------------------------------------------

    def _load(self, record: GraphRecord, definitions: Mapping[str, NodeDefinition], embedded_in: tuple[str, ...], alone: dict[_AloneKey, Graph]) -> None:
        """Check ``record`` against ``definitions`` and place each of its nodes, each after the nodes it reads."""
        self.record = record
        self.problems = []
        self.inputs = {}
        self._definitions = dict(definitions)
        self._embedded_in = embedded_in
        self._alone = alone
        self._nodes = {}
        self._input_fields = {}
        self._branches = {}
        self._above: dict[str, frozenset[str]] = {}
        #: Each first node's open input, with its default, in the order placed; made the graph's inputs below.
        self._opened: list[tuple[GraphInput, FieldSpec, Any]] = []
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
        self._position = {node_id: i for i, node_id in enumerate(known)}
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
        ordered, reads = self._order(list(known), into)
        for node_id in ordered:
            definition = self._resolve(known[node_id])
            if definition is not None:
                self._place(known[node_id], definition, into[node_id], reads[node_id])
        names = [graph_input.name for graph_input, _, _ in self._opened]
        for graph_input, spec, default in self._opened:
            name = graph_input.name if names.count(graph_input.name) == 1 else graph_input.address
            self.inputs[name] = graph_input
            self._input_fields[graph_input.address] = InputField(name, graph_input.address, spec, default)
        del self._opened
        #: The nodes with a ``FromRun`` parameter, an embedded graph's too, as ``from_run_values`` reads them.
        self._needs_from_run: list[tuple[str, FunctionDefinition[..., Any]]] = []
        #: The nodes that read each node.
        self._readers: dict[str, list[PlacedNode]] = {node_id: [] for node_id in self._nodes}
        for placed in self._nodes.values():
            match placed:
                case PlacedFunction(definition=function):
                    if function.from_run:
                        self._needs_from_run.append((placed.id, function))
                case PlacedGraph(inner=inner):
                    self._needs_from_run += inner._needs_from_run
            for source_id in placed.reads:
                self._readers[source_id].append(placed)

    def _resolve(self, record: NodeRecord) -> NodeDefinition | None:
        """The definition a node of the record names, at its version; ``None`` with the problem when there is none."""
        newest = self._definitions.get(record.node)
        if newest is None:
            self.problems.append(make_problem("unknown_node", record.id, node=record.node, choices=sorted(self._definitions)))
            return None
        if record.version not in newest.versions:
            self.problems.append(make_problem("unknown_version", record.id, node=record.node, version=record.version, newest=newest.version))
            return None
        return newest.version_of(record.version)

    def _order(self, order: list[str], into: dict[str, dict[str, list[str]]]) -> tuple[list[str], dict[str, frozenset[str]]]:
        """The nodes with each after what it reads, ties in the record's order, and the nodes each one's edges read;
        a node on a cycle is a problem."""
        position = self._position
        reads = {n: frozenset({s.partition(".")[0] for sources in into[n].values() for s in sources} & position.keys()) for n in order}
        readers: dict[str, list[str]] = {n: [] for n in order}
        for n, sources in reads.items():
            for source in sources:
                readers[source].append(n)
        waiting = {n: len(sources) for n, sources in reads.items()}
        ready = [position[n] for n in order if not waiting[n]]
        heapq.heapify(ready)
        done: list[str] = []
        while ready:
            n = order[heapq.heappop(ready)]
            done.append(n)
            for reader in readers[n]:
                waiting[reader] -= 1
                if not waiting[reader]:
                    heapq.heappush(ready, position[reader])
        stuck = {n for n in order if waiting[n]}
        for n in order:
            if n in stuck and _reaches(n, n, readers, stuck):  # on the cycle, not only below it
                self.problems.append(make_problem("cycle", n))
        return done, reads

    def _place(self, record: NodeRecord, definition: NodeDefinition, edges: dict[str, list[str]], reads: frozenset[str]) -> None:
        """Place one node: plan each input, settle its scope, and keep it in ``_nodes``. A node with a problem of
        its own is left out, and so is every node that reads it."""
        match definition:
            case FunctionDefinition():
                embedded = None
                fields = self._declared_inputs(record, definition)
                fixed_at_load = read_by_hooks(definition)
            case GraphDefinition():
                embedded = self._embedded(record, definition)
                fields = embedded.inputs if embedded is not None else None
                fixed_at_load = frozenset[str]()
        if fields is None:
            return
        for name in sorted({*record.values, *record.defaults, *edges} - fields.keys()):
            self.problems.append(make_problem("unknown_input", record.id, name, inputs=list(fields)))
        first = not any(name in fields for name in edges)  # a first node: nothing is wired into it, so the caller fills it
        inputs: dict[str, InputPlan] = {}
        for name, spec in fields.items():
            if name in record.values and name in record.defaults:
                self.problems.append(make_problem("value_and_default", record.id, name))
            if name in edges:
                if name in record.values or name in record.defaults:
                    self.problems.append(make_problem("value_and_edge", record.id, name))
                fed = self._fed(record.id, name, spec, edges[name])
                if fed is None:
                    return
                inputs[name] = fed
            elif name in record.values:
                inputs[name] = Fixed(spec, self._typed(record.id, name, spec, record.values[name]))
            elif first and name not in fixed_at_load:
                default = self._typed(record.id, name, spec, record.defaults[name]) if name in record.defaults else spec.default
                required = default is REQUIRED
                address = f"{record.id}.{name}"
                described = spec.describe(GraphInput, address=address, node_id=record.id, required=required, default=None if required else spec.json_of(default))
                self._opened.append((described, spec, default))
                inputs[name] = Open(spec, address)
            elif name in record.defaults:
                self.problems.append(make_problem("default_on_later_node", record.id, name))
                inputs[name] = Fixed(spec, self._typed(record.id, name, spec, record.defaults[name]))
            elif spec.required and name not in fixed_at_load:  # a hook that reads it says so itself, as hook_needs_value
                self.problems.append(make_problem("unbound_on_later_node", record.id, name))
                return
            else:
                inputs[name] = Fixed(spec, spec.default)
        scope = self._scope(record.id, inputs)
        if scope is None or not self._branches_meet(record.id, inputs):
            return
        checked = tuple(name for name, plan in inputs.items() if _read_before_a_call(plan))
        match definition:
            case FunctionDefinition():
                outputs = self._declared_outputs(record, definition, inputs)
                if outputs is None:
                    return
                placed: PlacedNode = PlacedFunction(
                    id=record.id, scope=scope, inputs=inputs, outputs=outputs, reads=reads, checked=checked, definition=definition,
                )
            case GraphDefinition():
                assert embedded is not None  # the fields above are its inputs
                placed = PlacedGraph(
                    id=record.id, scope=scope, inputs=inputs, outputs=embedded.outputs, reads=reads, checked=checked, definition=definition,
                    inner=embedded.inner, inner_inputs=embedded.inner_inputs, inner_outputs=embedded.inner_outputs,
                )
        for plan in inputs.values():  # a Series read one item at a time is held to max_items by the node that makes it
            for source in plan.sources if isinstance(plan, Read) else ():
                if isinstance(source, Item):
                    self._nodes[source.node_id].opens.add(source.output)
        self._nodes[record.id] = placed

    # -- a function's fields ----------------------------------------------------

    def _declared_inputs(self, record: NodeRecord, definition: FunctionDefinition[..., Any]) -> dict[str, FieldSpec] | None:
        """A function node's inputs: the ones it declares, then the ones its ``compute_added_inputs`` hook adds;
        ``None`` with the problem when the hook fails."""
        if definition.compute_added_inputs is None:
            return dict(definition.inputs)
        added = self._hook(record, definition, "compute_added_inputs", definition.compute_added_inputs, {})
        if added is None:
            return None
        taken = sorted(added.keys() & (definition.inputs.keys() | definition.from_run.keys()))
        if taken:
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=f"it adds {taken[0]!r}, which the node already declares", error="added_input_taken"))
            return None
        return {**definition.inputs, **added}

    def _declared_outputs(self, record: NodeRecord, definition: FunctionDefinition[..., Any], inputs: Mapping[str, InputPlan]) -> dict[str, FieldSpec] | None:
        """A function node's outputs: the ones it declares, or the ones its ``compute_outputs`` hook makes of the
        type arriving on each input an edge feeds; ``None`` with the problem when the hook fails."""
        if definition.compute_outputs is None:
            return dict(definition.outputs)
        arriving = {name: plan.sources[0].type for name, plan in inputs.items() if isinstance(plan, Fed)}
        return self._hook(record, definition, "compute_outputs", definition.compute_outputs, arriving)

    def _hook(
        self, record: NodeRecord, definition: FunctionDefinition[..., Any], kind: HookName, hook: Callable[..., Mapping[Any, Any]],
        arriving: Mapping[str, Any],
    ) -> dict[str, FieldSpec] | None:
        """Call a hook with the node's typed values and the types arriving on its inputs (``Any`` where nothing
        arrives): its answer read as fields, or ``None`` with the problem."""
        kwargs: dict[str, Any] = {}
        for name, reads_arriving in definition.hook_parameters[kind].items():
            if reads_arriving:  # only compute_outputs reads one
                kwargs[name] = arriving.get(name, Any)
                continue
            spec = definition.inputs[name]  # @node checked the parameter names a declared input
            raw = record.values.get(name, record.defaults.get(name, REQUIRED))
            if raw is REQUIRED and spec.default is REQUIRED:
                self.problems.append(make_problem("hook_needs_value", record.id, name))
                return None
            value = spec.default if raw is REQUIRED else self._typed(record.id, name, spec, raw)
            if value is REQUIRED:
                return None  # a value with its own problem
            kwargs[name] = value
        try:
            fields = dict(hook(**kwargs))
        except NodeError as error:  # the hook's own account, with its own code and its ctx kept whole
            own = {"error_ctx": dict(error.ctx)} if error.ctx else {}
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=error.message, error=error.code, **own))
            return None
        except Exception as error:  # noqa: BLE001  the hook's failure is this node's problem
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=str(error), error=type(error).__name__))
            return None
        bad = [name for name in fields if not (isinstance(name, str) and name and not set(name) & set("./@"))]
        if bad:
            self.problems.extend(make_problem("parameter_name_invalid", record.id, str(name)) for name in bad)
            return None
        try:
            return {name: read_field(name, *_hint_and_default(hint), where=f"node {definition.id!r}, ") for name, hint in fields.items()}
        except Exception as error:  # noqa: BLE001  a hint conductor can't carry is the hook's failure
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=str(error), error=type(error).__name__))
            return None

    def _typed(self, node_id: str, name: str, spec: FieldSpec, value: Any) -> Any:
        """A value typed into the record, as ``spec`` reads it; ``REQUIRED`` with the problem when it can't."""
        try:
            return read_given(spec, value)
        except FieldValueError as refused:
            problem = make_problem("invalid_value", node_id, name, reason=refused.reason, **refused.ctx)
            if problem not in self.problems:  # a hook reads the value its node's input reads, and is told once
                self.problems.append(problem)
            return REQUIRED

    # -- an embedded graph's fields ---------------------------------------------

    def _embedded(self, record: NodeRecord, definition: GraphDefinition) -> _Embedded | None:
        """An embedded graph as a node: the inner graph loaded, and its inputs and outputs as this node's fields.

        An inner input or output is offered under its own name where no other shares it, else as ``node/field``,
        as a name holds no dot. An output made per item inside is offered as a ``Series`` of them.

        Returns:
            ``None`` with the problem when the graph embeds itself, has problems of its own, or makes a ``Series``
            or a table per item, which no one output can carry.
        """
        if definition.id in self._embedded_in:
            self.problems.append(make_problem("embeds_itself", record.id, node=definition.id))
            return None
        stack = (*self._embedded_in, definition.id)
        key = (definition.id, definition.version, stack)
        inner = self._alone.get(key)
        if inner is None:
            inner = self._alone[key] = _loaded(definition.graph, self._definitions, stack, self._alone)
        if inner.problems:
            problems = [p.model_dump(mode="json") for p in inner.problems]
            self.problems.append(make_problem("embedded_graph_broken", record.id, graph=definition.id, count=len(problems), problems=problems))
            return None
        inputs: dict[str, FieldSpec] = {}
        inner_inputs: dict[str, str] = {}
        for offered in inner._input_fields.values():
            name = offered.name.replace(".", "/")
            inputs[name] = dataclasses.replace(offered.spec, name=name, default=offered.default, default_json=inner.inputs[offered.name].default)
            inner_inputs[name] = offered.address
        last = [(address, inner._nodes[address.partition(".")[0]]) for address in inner.outputs]
        names = [address.partition(".")[2] for address, _ in last]
        deciding = {(at.id, spec.param.exclusive_group) for _, at in last for spec in at.outputs.values()}
        outputs: dict[str, FieldSpec] = {}
        inner_outputs: dict[str, str] = {}
        for address, at in last:
            output = address.partition(".")[2]
            spec = at.outputs[output]
            if at.scope and spec.shape != "one":
                self.problems.append(make_problem("lists_not_collected", record.id, series=[at.scope[-1], address]))
                return None
            name = output if names.count(output) == 1 else address.replace(".", "/")
            group = spec.param.exclusive_group
            if group and sum(held == group for _, held in deciding) > 1:  # two decisions inside, each its own here
                spec = dataclasses.replace(spec, param=dataclasses.replace(spec.param, exclusive_group=f"{at.id}/{group}"))
            outputs[name] = dataclasses.replace(spec, name=name, shape="many", optional=False, item=spec.type) if at.scope else dataclasses.replace(spec, name=name)
            inner_outputs[name] = address
        return _Embedded(inner, inputs, outputs, inner_inputs, inner_outputs)

    # -- edges ------------------------------------------------------------------

    def _fed(self, node_id: str, name: str, spec: FieldSpec, addresses: list[str]) -> Fed | None:
        """How edges from ``addresses`` feed one input: one value on each call, or every value its source made.

        An input of one value reads, whatever its type: a ``Series`` reaches it one item at a time. A ``Series``
        input gathers. A ``Table`` input reads when every edge brings a table, and else gathers records.

        Returns:
            ``None`` when an edge names nothing a node gives or brings a type the input doesn't take, with the
            problem, and ``None`` when its node didn't load.
        """
        sources: list[Source] = []
        for address in addresses:
            source = self._source(address, node_id, name)
            if source is None:
                return None
            sources.append(source)
        sources.sort(key=self._edge_order)
        gathers = spec.shape == "many" or (spec.shape == "table" and not all(isinstance(s, Value) and accepts(spec.type, s.type) for s in sources))
        wanted = spec.item if gathers else spec.type
        for source in sources:
            if not accepts(wanted, source.type):
                made, taken = type_id(source.type), type_id(wanted)
                got = f"Series[{made}]" if isinstance(source, Item) else made
                expected = f"Series[{taken}]" if spec.shape == "many" else spec.type_id
                self.problems.append(make_problem("type_mismatch", node_id, name, source=source.address, got=got, expected=expected))
                return None
        deepest = max((s.scope for s in sources), key=len)
        if not gathers:
            return Read(spec, tuple(sources), deepest)
        if not deepest:
            return SeriesOfOne(spec, tuple(s for s in sources if isinstance(s, Value)), ())
        made_by, _, made_as = deepest[-1].partition(".")
        series = Value(made_by, made_as, None, deepest[:-1], list)
        return Gather(spec, tuple(sources), tuple(self._gate(s, len(series.scope)) for s in sources), series, series.scope)

    def _source(self, address: str, node_id: str, name: str) -> Source | None:
        """What an edge from ``address`` brings: ``node.output``, or ``node.output.field`` for one field of a record.

        Returns:
            An item of a ``Series`` output, or an output's one value; ``None`` with the problem when the address
            names nothing its node gives, and ``None`` when its node didn't load.
        """
        source_id, _, rest = address.partition(".")
        output, _, wanted = rest.partition(".")
        source = self._nodes.get(source_id)
        if source is None:  # a node that failed to load, or none: its own problem says why
            return None
        spec = source.outputs.get(output)
        hint = _NO_FIELD if spec is None else spec.type if not wanted else _field_type(spec.type, wanted)
        if spec is None or hint is _NO_FIELD:
            if spec is None:
                self.problems.append(make_problem("unknown_output", node_id, name, source=address, outputs=[f"{source_id}.{o}" for o in source.outputs]))
            else:  # the output is there, and has no such field
                fields = [f"{source_id}.{output}.{f}" for f in record_fields(without_none(spec.type))]
                self.problems.append(make_problem("unknown_output", node_id, name, source=address, outputs=fields, field=wanted))
            return None
        if spec.shape == "many":
            return Item(source_id, output, wanted or None, (*source.scope, f"{source_id}.{output}"), hint)
        return Value(source_id, output, wanted or None, source.scope, hint)

    def _edge_order(self, source: Source) -> tuple[int, int, str]:
        """Where an edge into an input falls among several: its node's place in the record, then the output's in its node."""
        return self._position[source.node_id], list(self._nodes[source.node_id].outputs).index(source.output), source.address

    def _gate(self, source: Source, depth: int) -> Gate:
        """What skips ``source`` for every item of a gather at once: the outputs ``depth`` maps deep or shallower
        that it rests on, found by following it up through the nodes inside the map.

        A node is skipped when any one of its inputs is, and an input when every edge into it is. So each set is
        one choice of an edge for some input on the way up, and a set that holds a smaller one adds nothing.
        """
        made_in = source.scope[:-1] if isinstance(source, Item) else source.scope
        if len(made_in) <= depth:
            return (frozenset({Value(source.node_id, source.output, None, made_in, Any)}),)
        found = {
            frozenset[Value]().union(*choice)
            for plan in self._nodes[source.node_id].inputs.values() if isinstance(plan, Fed)
            for choice in itertools.product(*(self._gate(s, depth) for s in plan.sources))
        }
        return tuple(cells for cells in found if not any(fewer < cells for fewer in found))

    def _scope(self, node_id: str, inputs: Mapping[str, InputPlan]) -> Scope | None:
        """The scope a node runs in: the deepest of its inputs', each of which lies on the path to it; ``None``
        with the problem when two inputs sit in maps of different Series."""
        scopes = [plan.scope for plan in inputs.values() if isinstance(plan, Fed)]
        scope = max(scopes, key=len, default=())
        for other in scopes:
            if scope[: len(other)] != other:
                self.problems.append(make_problem("lists_not_collected", node_id, series=[scope[-1], other[-1]]))
                return None
        return scope

    # -- branches ---------------------------------------------------------------

    def _branch(self, source: Source) -> frozenset[Branch]:
        """The ways decisions must go for ``source`` to hold a value."""
        group = self._nodes[source.node_id].outputs[source.output].param.exclusive_group
        own = {Branch(source.node_id, group, source.output)} if group else set[Branch]()
        return self._branches[source.node_id] | own

    def _branches_meet(self, node_id: str, inputs: Mapping[str, InputPlan]) -> bool:
        """Keep the branches a node rests on, and refuse edges that can never bring it a value together.

        Several edges into one input are alternatives: each on its own branch of one decision, so a call never has
        two values for it, and all in one scope, so each call has a place for the one it gets. Two ``Series``
        inputs that keep no blanks, fed only from opposite branches, would lose every item.

        Returns:
            False, with the problem, when an input's edges don't meet these.
        """
        rests_on: set[Branch] = set()
        for name, plan in inputs.items():
            if not isinstance(plan, Fed):
                continue
            behind = [self._branch(s) for s in plan.sources]
            if not all(_opposed(a, b) for i, a in enumerate(behind) for b in behind[i + 1:]):
                self.problems.append(make_problem("several_edges_into_one_input", node_id, name, sources=[s.address for s in plan.sources]))
                return False
            scopes = list(dict.fromkeys(s.scope for s in plan.sources))
            if len(scopes) > 1:
                self.problems.append(make_problem(
                    "alternatives_on_different_items", node_id, name, series=[scope[-1] if scope else None for scope in scopes[:2]],
                ))
                return False
            shared = behind[0].intersection(*behind[1:])  # what every edge into it rests on, the input does
            if isinstance(plan, Gather):  # a gather is skipped only by what is decided where it runs, or above
                shared = frozenset(b for b in shared if len(self._nodes[b.node_id].scope) <= len(plan.scope))
            rests_on |= shared
        self._branches[node_id] = frozenset(rests_on)
        aligned = [
            (name, [self._branch(s) for s in plan.sources]) for name, plan in inputs.items()
            if isinstance(plan, Gather) and not takes_a_blank(plan.spec.item)
        ]
        for i, (a, behind_a) in enumerate(aligned):
            for b, behind_b in aligned[i + 1:]:
                if all(_opposed(x, y) for x in behind_a for y in behind_b):
                    self.problems.append(make_problem("series_inputs_never_share_an_item", node_id, inputs=[a, b]))
                    return False
        return True


#: An embedded graph loaded alone, by its definition's id and version and the graphs it sits in.
_AloneKey = tuple[str, int, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _Embedded:
    """An embedded graph as the node that embeds it reads it.

    Attributes:
        inner: The embedded graph, loaded.
        inputs: The fields of the inputs it offers, by the name it offers each under.
        outputs: The fields of the outputs it offers, by the name it offers each under.
        inner_inputs: The address in ``inner`` of each input, by offered name.
        inner_outputs: The address in ``inner`` of each output, by offered name.
    """

    inner: Graph
    inputs: dict[str, FieldSpec]
    outputs: dict[str, FieldSpec]
    inner_inputs: dict[str, str]
    inner_outputs: dict[str, str]


def load_graph(record: GraphRecord, definitions: Mapping[str, NodeDefinition]) -> Graph:
    """``record`` loaded against ``definitions``: the one way a host's ``Graph`` is made."""
    return _loaded(record, definitions, (), {})


def _loaded(record: GraphRecord, definitions: Mapping[str, NodeDefinition], embedded_in: tuple[str, ...], alone: dict[_AloneKey, Graph]) -> Graph:
    """A graph of ``record``.

    Args:
        record: The stored graph.
        definitions: The newest version of each node, by id.
        embedded_in: The ids of the graph nodes this one sits in, outermost first, so one that embeds itself is found.
        alone: Each embedded graph already loaded, shared by every graph one load makes, so each loads once.
    """
    graph = Graph.__new__(Graph)
    graph._load(record, definitions, embedded_in, alone)
    return graph


def _runs_on(scope: Scope) -> RunsOn:
    """Where a node of ``scope`` runs."""
    return RunsOn(mode="per_item", series=scope[-1], msg=f"per item of {scope[-1]}") if scope else RunsOn(mode="once", msg="once")


def _feed(scope: Scope, plan: InputPlan) -> Feed:
    """How ``plan`` feeds its input on each call of a node that runs in ``scope``."""
    match plan:
        case Gather(series=series):
            return Feed(mode="gathered", series=series.address, msg=f"every item of {series.address}")
        case SeriesOfOne():
            return Feed(mode="gathered", msg="the one value, as a Series of one")
        case Read(scope=fed_in) if fed_in != scope:
            return Feed(mode="captured", series=fed_in[-1] if fed_in else None, msg=f"captured from {f'the item of {fed_in[-1]}' if fed_in else 'the graph'}")
        case Read(scope=fed_in) if fed_in:
            return Feed(mode="per_item", series=fed_in[-1], msg=f"per item of {fed_in[-1]}")
        case _:
            return Feed(mode="single", msg="one value")


def _read_before_a_call(plan: InputPlan) -> bool:
    """Whether what edges bring an input is read by the input's type before each call: not when every value they
    bring is already one the field takes as it is, with no limit or choice to hold it to."""
    match plan:
        case Read(spec=spec, sources=sources):
            return spec.param.holds_a_value or not all(already(spec.type, s.type) for s in sources)
        case Gather(spec=spec, sources=sources) | SeriesOfOne(spec=spec, sources=sources):
            return spec.shape == "table" or spec.param.holds_a_value or not all(already(spec.item, s.type) for s in sources)
        case _:
            return False


#: What ``_field_type`` gives for a field a record doesn't have.
_NO_FIELD: Any = object()


def _field_type(hint: Any, name: str) -> Any:
    """The type of the field ``name`` of a record of type ``hint``.

    Returns:
        A ``TypedDict``'s, dataclass's or pydantic model's declared type for it; a mapping's value type; ``Any``
        for a value of any type; ``_NO_FIELD`` for a record without the field, and for a type that is no record.
    """
    hint = without_none(hint)
    if hint is Any:
        return Any
    if typing.get_origin(hint) in (dict, Mapping):
        return typing.get_args(hint)[1]
    return record_fields(hint).get(name, _NO_FIELD)


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """What a hook gives for one field: a hint, or ``(hint, default)`` as pydantic's ``create_model`` takes one."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED


def _type_text(hint: Any) -> str:
    """A type as text that is the same in every process and tells apart every two types a value is read by differently:
    a class by its module and name, a generic by its arguments (``list[int]``, ``list[str]``), a ``Literal`` by its
    values, a type alias by its name; the metadata of an ``Annotated`` left out."""
    if typing.get_origin(hint) is typing.Annotated:
        return _type_text(typing.get_args(hint)[0])
    if typing.get_origin(hint) is Literal:
        return f"Literal{[_constant_text(value) for value in typing.get_args(hint)]}"
    if isinstance(hint, typing.TypeAliasType):
        return f"{hint.__module__}.{hint.__name__}"
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if args:
        name = "Union" if origin in (typing.Union, types.UnionType) else _type_text(origin)
        return f"{name}[{', '.join(map(_type_text, args))}]"
    if isinstance(hint, type):
        return f"{hint.__module__}.{hint.__qualname__}"
    return repr(hint)


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


def shared_depth(a: Scope, b: Scope) -> int:
    """How many maps two scopes share from the graph's own scope down."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n
