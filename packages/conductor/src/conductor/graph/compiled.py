"""``CompiledGraph`` — everything the compiler learned about a graph, as one immutable value.

``CompiledGraph.from_graph`` builds it from a ``Graph`` and a ``NodeRegistry``.
Everyone else asks it questions and never reads the nodes' bindings
themselves: which version each node uses, which inputs and outputs it
actually has, where each input's value comes from, what type travels on
every field, which nodes run once per row and in what order, under which
condition each output appears, and what is wrong. The engine is one more
caller.

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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from conductor.graph.binding import Binding
from conductor.graph.expand import authored_ref, expanded_ref
from conductor.graph.problem import Problem
from conductor.ref import Ref
from conductor.registry import runner_for

if TYPE_CHECKING:
    from conductor.graph.conditions import Condition
    from conductor.graph.model import Graph, GraphNode
    from conductor.interface import Interface
    from conductor.node import GraphVersion, NodeVersion
    from conductor.registry import NodeRegistry
    from conductor.series import Index


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

    _nodes: Mapping[str, GraphNode]
    _registry: NodeRegistry
    _versions: Mapping[str, NodeVersion | GraphVersion]
    _interfaces: Mapping[str, Interface]
    _statics: Mapping[str, Mapping[str, Any]]
    _dependencies: Mapping[str, frozenset[str]]
    _order: tuple[str, ...]
    _iterated: Mapping[str, Index | None]
    _indexes: Mapping[Ref, Index | None]
    _types: Mapping[Ref, Any]
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
    it. The engine reads ``interface``, ``statics``, ``runner``,
    ``dependencies`` and ``iterates_on`` for each node it runs, and
    ``version`` and ``graph_node`` for the policy and the type it reports;
    an editor reads ``interface``, ``iterates_on`` and ``problems`` for each
    node it draws. Its sibling is ``CompiledField``, the same window onto
    one input or output.

    An attribute a node cannot answer raises: a node the edge walk could
    not derive — its own edges wrong, or a fault upstream of it — has an
    interface but no ``iterates_on``, and a node whose version is a graph
    has an interface, a version and ``embedded_in`` but no ``graph_node``,
    ``statics`` or ``dependencies`` — its inner nodes run in its place. The
    ``Problem`` on it says why.
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
        JSON form. A value is a ``list`` exactly when the author typed
        many values.
        """
        return self._graph._statics[self.id]

    @property
    def runner(self) -> Callable[..., Any]:
        """The callable that runs this node, on a fresh instance per call."""
        node = self.graph_node
        return runner_for(self._graph._registry, node.type, node.version)

    @property
    def dependencies(self) -> frozenset[str]:
        """The nodes this one waits for, read off its edges."""
        return self._graph._dependencies[self.id]

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

    @property
    def problems(self) -> tuple[Problem, ...]:
        """Every problem about this node, on any of its fields or on the
        node itself. Anchored on the authored graph, so an inner node of an
        embedded graph has none of its own: they sit on the node the author
        placed, under the inner address."""
        return tuple(p for p in self._graph.problems if p.node_id == self.id)


@dataclass(frozen=True)
class CompiledField:
    """One input or output as the compiler left it: its type, its rows, where its value comes from.

    ``CompiledGraph.field(ref)`` builds one on each call; nothing stores
    it. The engine reads ``index`` and ``binding`` to lay values out per
    row and to find them; an editor reads ``type`` and ``index`` to draw
    the edge, and ``problems`` to mark the field. Its sibling is
    ``CompiledNode``, the same window onto one node.

    An attribute a field cannot answer raises: an input has no
    ``condition`` and an output has no ``binding``; and no field of a node
    the edge walk could not derive has a ``type``, ``index`` or
    ``condition``, since nothing downstream of a fault is guessed at. The
    ``Problem`` on it says why.
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
    def condition(self) -> Condition:
        """Under which condition this output appears: a boolean formula over
        the decisions upstream (see ``conductor.graph.conditions``),
        ``ALWAYS`` when nothing gates it. Derived from ``choice`` groups
        and edges; the engine never reads it. Only an output has one."""
        return self._graph._conditions[self.ref]

    @property
    def problems(self) -> tuple[Problem, ...]:
        """Every problem about this field, anchored where the author sees
        it: on the node they placed, under the inner address for a field
        inside an embedded graph."""
        anchor = authored_ref(self.ref)
        return tuple(p for p in self._graph.problems if p.node_id == anchor.node_id and p.field == anchor.field)
