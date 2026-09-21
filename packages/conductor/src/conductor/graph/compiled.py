"""``CompiledGraph`` — everything the compiler learned about a graph, as one immutable value.

``CompiledGraph.from_graph`` builds it from a ``Graph`` and a ``NodeRegistry``.
Everyone else asks it questions and never reads the nodes' bindings
themselves: which version each node uses, which inputs and outputs it
actually has, where each input's value comes from and how each unit
receives it, what type travels on every field, which nodes run once per
row and in what order, under which condition each output appears, and
what is wrong. The engine is one more caller.

The questions come at three scales. The graph itself answers what it
takes and returns (``interface``), what is wrong with it (``problems``,
``is_runnable``), in what order its nodes run (``execution_order``) and
which decisions a run makes (``decisions``). ``node(node_id)`` returns a
``CompiledNode``, everything the compiler knows about one node; ``field(ref)``
returns a ``CompiledField``, everything it knows about one input or
output. Both are windows onto this record, built on each call, never
stored. The graph stores a ``GraphNode`` (a node as the author saved it)
and addresses fields by ``Ref``; what compile learned about that node or
field lives here, and a ``CompiledNode`` hands back the stored node as
``graph_node``.

The words this module uses throughout, each defined once here.

A node's interface is the list of inputs and outputs it actually has.
Usually that is what its version declares, but a node may add or drop
fields depending on the values it holds and the types connected to it.

A series is a value with many rows, and an index names where those rows
come from: ten uploaded documents are a series of ten rows on the index
of the node that uploaded them. A scalar input is an input that takes one
value. A node that receives a series on a scalar input runs once per row
of the series; we say the node iterates on that index.

Any node placed in a graph is a placement of its definition (see
``GraphNode``). Some nodes have a version that is itself a graph; that
graph is an embedded graph, and below "the placement" means the node
that embeds it. The compiler inlines the embedded graph's nodes in place
of that node, so the graph the author drew (the authored graph) differs
from the graph that runs (the expanded graph). In the authored graph the
embedded graph is one node, ``approve``, and its fields are addressed
through it, ``Ref("approve", "check.amount")``. In the expanded graph
its inner nodes are nodes of the run, named ``approve/check``.

``execution_order`` and ``decisions`` speak of the expanded graph;
``problems`` and ``interface`` speak of the authored one. ``node`` answers
for both graphs: an inner node by its expanded id (``approve/check``) and
the placement by its own (``approve``). A question about a field may use either address:
``field(Ref("approve", "check.amount"))`` and
``field(Ref("approve/check", "amount"))`` are the same field.

It is a plain value: immutable, no I/O, no session. The same graph and
the same registry always give the same ``CompiledGraph``, so it can be
cached, and "compile this and assert what it says" is a complete test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from conductor.codec import to_wire
from conductor.graph.binding import Binding, Static
from conductor.graph.expand import expanded_ref
from conductor.graph.problem import Problem
from conductor.graph.receive import Iterate
from conductor.ref import Ref
from conductor.series import Series

if TYPE_CHECKING:
    from conductor.graph.conditions import Condition
    from conductor.graph.model import Graph, GraphNode
    from conductor.graph.receive import Receive
    from conductor.interface import Interface
    from conductor.node import GraphVersion, NodeVersion
    from conductor.registry import NodeRegistry
    from conductor.series import Index


def _written(value: Any, dtype: Any, listed: bool) -> Any:
    """A static as its declared type writes it, for ``CompiledNode.fingerprint``:
    each of many typed-in values (``listed``) through the scalar type, a
    series through its element type, one value through its own."""
    element = getattr(dtype, "element", None)
    if element is not None:
        return [to_wire(item, element) for item in (value.values if isinstance(value, Series) else value)]
    if listed:
        return [to_wire(item, dtype) for item in value]
    return to_wire(value, dtype)


@dataclass(frozen=True)
class CompiledGraph:
    """The result of compiling one graph. Ask it; do not read through it.

    Built by ``from_graph`` and read by the engine (``execution_order``,
    then ``node`` and ``field`` for each node it runs), an editor's
    compile endpoint (``problems``, ``interface``, and ``node`` and
    ``field`` for everything it draws), and anything deciding whether a
    run may start (``is_runnable``).

    Two attributes are public, ``interface`` and ``problems``; every other
    attribute is private and read through the methods here and through
    ``node`` and ``field``. A node compile could not resolve — unknown type
    or version — or could not order — on a cycle — has a fatal ``Problem``
    and no interface, version, index or types; asking ``node`` for it
    raises, and an editor paints it from ``problems`` alone.
    """

    #: The graph as the author saved it, and the registry it was compiled
    #: against: kept so a compiled graph can produce another from the same
    #: two (``bound``), never read back for what compile already learned.
    _graph: Graph
    _registry: NodeRegistry
    _nodes: Mapping[str, GraphNode]
    _versions: Mapping[str, NodeVersion | GraphVersion]
    _interfaces: Mapping[str, Interface]
    _statics: Mapping[str, Mapping[str, Any]]
    #: Per node the walk derived, the pydantic model that validates a call
    #: against its interface — built once here, not once per unit.
    _call_models: Mapping[str, type[BaseModel]]
    _order: tuple[str, ...]
    _iterated: Mapping[str, Index | None]
    _indexes: Mapping[Ref, Index | None]
    _types: Mapping[Ref, Any]
    #: How every input receives its value, as the walk over the edges decided
    #: it once (``conductor.graph.receive``); the engine reads it and decides
    #: nothing of the kind again.
    _receives: Mapping[Ref, Receive]
    _conditions: Mapping[Ref, Condition]
    #: Every node whose version is a graph, by expanded id — the ones the
    #: author placed and the ones nested inside them alike.
    _placements: frozenset[str]
    #: For every node of the expanded graph and every placement, the innermost
    #: embedded graph it came from, or ``None`` for a node the author placed.
    _placement_of: Mapping[str, str | None]
    #: What this graph takes and returns, in the same record a node version
    #: declares: ``inputs`` are the unlocked, connectable fields of the nodes
    #: nothing feeds into (each declaration whole, titled as the author
    #: titled it on that node), ``outputs`` every field of the nodes nothing
    #: reads from, both named by address (``"node.field"``); ``returns`` is
    #: ``Mapping``.
    interface: Interface
    #: Everything wrong with the graph, fatal or not, each on the node and
    #: field it is about. Anchored on the authored graph: a problem found
    #: inside an embedded graph sits on the node the author placed, with the
    #: inner address as the field.
    problems: tuple[Problem, ...]

    @classmethod
    def from_graph(cls, graph: Graph, registry: NodeRegistry) -> CompiledGraph:
        """Compile ``graph`` against ``registry``: the one way to get a ``CompiledGraph``.

        Pure — the same graph and registry always give the same result.
        Every definition the graph names must already be in the registry.
        Nothing raises for a fault in the graph; read ``problems`` and
        ``is_runnable``. The passes are ``compiler._Compilation.build``'s.
        """
        from conductor.graph.compiler import _Compilation

        return _Compilation(graph, registry).build()

    # -- one node, one field ---------------------------------------------------

    def node(self, node_id: str) -> CompiledNode:
        """Everything the compiler knows about one node, by expanded id —
        or, for a node whose version is a graph, by its own id. Raises for
        an id compile has nothing on: not in the graph, or a node it could
        not resolve or order, whose ``Problem`` in ``problems`` says why."""
        if node_id not in self._interfaces:
            raise KeyError(f"no resolved node {node_id!r}: not in the graph, or its problem says why")
        return CompiledNode(self, node_id)

    def field(self, ref: Ref) -> CompiledField:
        """Everything the compiler knows about one input or output, by
        either address: an address on the authored graph reads through to
        the field that runs. Raises for a field the node does not have —
        a programming error, not a state of the graph."""
        at = self.expanded(ref)
        if at.node_id not in self._interfaces:
            if at.node_id != ref.node_id:
                raise KeyError(f"{ref.node_id!r} has no field {ref.field!r} on this node")
            raise KeyError(f"no resolved node {at.node_id!r}: not in the graph, or its problem says why")
        interface = self._interfaces[at.node_id]
        if at.field not in {f.name for f in (*interface.inputs, *interface.outputs)}:
            raise KeyError(f"{at.node_id!r} has no field {at.field!r} on this node")
        return CompiledField(self, at)

    # -- the two graphs ------------------------------------------------------

    def expanded(self, ref: Ref) -> Ref:
        """An address on the authored graph, read through to the node that
        runs: ``Ref("emb", "all.result")`` becomes ``Ref("emb/all", "result")``.
        A host reads engine results by the expanded address, since the
        engine knows only the expanded graph."""
        return expanded_ref(ref, self._placements)

    # -- the run ------------------------------------------------------------------

    def execution_order(self) -> tuple[str, ...]:
        """Expanded node ids in an order where every edge's source precedes
        its target. A node in a cycle is not in it; it has a fatal
        ``Problem`` instead."""
        return self._order

    def decisions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every decision a caller could observe: for each node that runs
        once and declares a ``choice`` group, the group's alternatives in
        field order, keyed by expanded node id. A node that runs per row is
        left out: its decision picks rows and gates nothing downstream."""
        found: dict[str, dict[str, tuple[str, ...]]] = {}
        for node_id in self._order:
            if self._iterated.get(node_id, None) is not None:
                continue
            groups: dict[str, list[str]] = {}
            for out in self._interfaces[node_id].outputs:
                if out.choice is not None:
                    groups.setdefault(out.choice, []).append(out.name)
            if groups:
                found[node_id] = {choice: tuple(names) for choice, names in groups.items()}
        return found

    # -- what is wrong -------------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run refuses otherwise."""
        return not any(p.fatal for p in self.problems)


@dataclass(frozen=True)
class CompiledNode:
    """One node as the compiler left it: what it has, what it holds, how it runs.

    ``CompiledGraph.node(node_id)`` builds one on each call; nothing stores
    it. The engine reads ``interface``, ``call_model``, ``statics``,
    ``runner`` and ``iterates_on`` for each node it runs, and ``version``
    and ``graph_node`` for the policy and the type it reports; an editor
    reads ``interface`` and ``iterates_on`` for each node it draws, and
    filters ``CompiledGraph.problems`` by node id for its marks. Its
    sibling is ``CompiledField``, the same window onto one input or output.

    An attribute a node cannot answer raises: a node the edge walk could
    not derive — its own edges wrong, or a fault upstream of it — has an
    interface but no ``iterates_on`` or ``call_model``, and a node whose
    version is a graph has an interface, a version and ``embedded_in`` but
    no ``graph_node`` or ``statics`` — its inner nodes run in its place.
    The ``Problem`` on it in ``CompiledGraph.problems`` says why.
    """

    _graph: CompiledGraph = field(repr=False)
    #: The expanded id — ``"approve/check"`` for an inner node — or, for a
    #: node whose version is a graph, its own.
    id: str

    @property
    def graph_node(self) -> GraphNode:
        """The node as the author stored it: its type, version number, title
        and bindings. Only for a node the engine runs; the placement of an
        embedded graph is not one, its inner nodes are."""
        return self._graph._nodes[self.id]

    @property
    def version(self) -> NodeVersion | GraphVersion:
        """The version this node uses: its ``run``, interface and policy — or,
        for a node whose version is a graph, that version's interface and
        its graph."""
        return self._graph._versions[self.id]

    @property
    def interface(self) -> Interface:
        """The inputs and outputs this node actually has, with every type
        the edges gave it — not merely what its version declared.

        The one place anything asks what a node has: the engine validates
        a call against it and an editor draws the fields from it. For a
        node whose version is a graph, the interface that version declares."""
        return self._graph._interfaces[self.id]

    @property
    def statics(self) -> Mapping[str, Any]:
        """The values the author typed into this node, by field.

        Each read through the field's declared type — the value, never its
        JSON form. Only what the author typed: an input left to its
        declared default is absent. Where the author typed many values for
        a scalar input the value is a list of them, and the field's
        ``receives`` is ``Iterate`` on the input's own index.
        """
        return self._graph._statics[self.id]

    @property
    def runner(self) -> Callable[..., Any]:
        """The callable that runs this node, on a fresh instance per call."""
        node = self.graph_node
        return self._graph._registry.runner_for(node.type, node.version)

    @property
    def fingerprint(self) -> str:
        """A hash of how the graph places this node: its type, version and
        bindings. A run record stores one per node, and a leg restored into
        a graph whose fingerprint differs runs the node again — a static
        edited, a version bumped, an edge moved all change it; a title or a
        position does not."""
        node = self.graph_node
        declared = {inp.name: inp.dtype for inp in self.interface.inputs}
        bindings: dict[str, Any] = {}
        for name, binding in node.bindings.items():
            if isinstance(binding, Static) and name in self.statics:
                # The value as its type writes it, never as the author spelled
                # it: ``2`` and ``2.0`` on a number are one value, and a graph
                # built in Python with the typed value hashes like the stored
                # graph with its JSON. The author typed many values exactly
                # when the input is received one per row of its own index.
                # A static for an input the node no longer has is hashed as
                # spelled, since no type reads it.
                received = self._graph._receives[Ref(self.id, name)]
                listed = isinstance(received, Iterate)
                bindings[name] = {"static": _written(self.statics[name], declared[name], listed)}
            else:
                bindings[name] = binding.model_dump()
        placed = {"type": node.type, "version": node.version, "bindings": bindings}
        return hashlib.sha256(json.dumps(to_jsonable_python(placed), sort_keys=True).encode("utf-8")).hexdigest()

    @property
    def call_model(self) -> type[BaseModel]:
        """The pydantic model that validates a call against this node's
        interface: one field per input, its declared type and default.
        Built once when the graph is compiled; the engine validates every
        unit's inputs through it. Only for a node the walk over the edges
        derived — a node with a fault upstream has none, and asking raises."""
        return self._graph._call_models[self.id]

    @property
    def iterates_on(self) -> Index | None:
        """The index this node runs once per row of, or ``None`` when it runs
        once. Read off its edges — a series arriving on a scalar input is
        what makes a node run per row — and never stored on the node. For a
        node whose version is a graph, the index its inner nodes run per row
        of, where a series entered it."""
        return self._graph._iterated[self.id]

    @property
    def embedded_in(self) -> str | None:
        """The id of the node whose embedded graph this node belongs to —
        ``"approve"`` for ``"approve/check"`` — or ``None`` for a node the
        author placed."""
        return self._graph._placement_of[self.id]


@dataclass(frozen=True)
class CompiledField:
    """One input or output as the compiler left it: its type, its rows, where its value comes from, how it is received.

    ``CompiledGraph.field(ref)`` builds one on each call; nothing stores
    it. The engine reads ``index``, ``binding`` and ``receives`` to lay
    values out per row, to find them and to hand each unit what it takes;
    an editor reads ``type`` and ``index`` to draw the edge and ``receives``
    to label it, and filters ``CompiledGraph.problems`` by node and field
    for its marks. Its sibling is ``CompiledNode``, the same window onto
    one node.

    An attribute a field cannot answer raises: an input has no
    ``condition`` and an output has no ``binding`` or ``receives``; and no
    field of a node the edge walk could not derive has a ``type``,
    ``index``, ``receives`` or ``condition``, since nothing downstream of
    a fault is guessed at. The ``Problem`` on it says why.
    """

    _graph: CompiledGraph = field(repr=False)
    #: The expanded address of the field — ``Ref("approve/check", "amount")``
    #: however it was asked for.
    ref: Ref

    @property
    def type(self) -> Any:
        """The type that travels on this field: what an edge from it or into
        it carries. An output of a node that runs once per row carries
        ``Series[X]`` even where its declaration says ``X``; an input fed a
        series carries that series, before the engine slices it per row."""
        return self._graph._types[self.ref]

    @property
    def index(self) -> Index | None:
        """Where the rows of the series on this field come from, or ``None``
        for a field that carries one value. For an input fed several series
        gathered together, the fresh index they were gathered onto. Read
        by the engine to lay values out per row."""
        return self._graph._indexes[self.ref]

    @property
    def binding(self) -> Binding | None:
        """Where this input's value comes from: ``Edges`` from other nodes'
        outputs, ``Static`` for a value the author typed, or ``None`` when
        nothing binds it and its declared default applies. Only an input
        has one; asking on an output raises."""
        node = self._graph._nodes[self.ref.node_id]
        if self.ref.field not in {i.name for i in self._graph._interfaces[self.ref.node_id].inputs}:
            raise KeyError(f"{node.type!r} has no input {self.ref.field!r} on this node")
        return node.bindings.get(self.ref.field)

    @property
    def receives(self) -> Receive:
        """How a unit of this input's node receives the value: the cell at
        its own row (``Iterate``), the one cell there is (``Broadcast``),
        everything on the field (``Whole``), the rows under its row
        (``Group``) or its unrelated sources collected (``Gather``) — see
        ``conductor.graph.receive``. Decided once by the
        walk over the edges; the engine's ledger reads it to tell when a
        unit is ready and what to hand it, and an editor may label the
        edge from it. Only an input has one; asking on an output raises."""
        return self._graph._receives[self.ref]

    @property
    def condition(self) -> Condition:
        """Under which condition this output appears: a boolean formula over
        the decisions upstream (see ``conductor.graph.conditions``),
        ``ALWAYS`` when nothing gates it. Derived from ``choice`` groups
        and edges; the engine never reads it. Only an output has one."""
        return self._graph._conditions[self.ref]
