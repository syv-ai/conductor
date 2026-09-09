"""``CompiledGraph`` — everything compile learned about a graph, as one immutable value.

``CompiledGraph.from_graph`` builds it from a ``Graph`` and a ``NodeRegistry``.
Everyone else asks it questions and never reads the nodes' bindings
themselves: which version each node uses, which inputs and outputs it
actually has, where each input's value comes from, what type travels on
every field, which nodes run once per row and in what order, under which
condition each output appears, and what is wrong. The engine is one more
caller.

Three words this module uses throughout:

* A **roster** is the list of inputs and outputs one node actually has —
  usually what its version declares, but a node may add or drop fields
  depending on the values it holds and the types connected to it.
* A **series** is a value with many rows, and an **index** names where
  those rows come from. A node that receives a series on a scalar input
  runs once per row; we say it is **lifted** on that index.
* A **placement** is a node whose version is itself a graph — an embedded
  flow. Compile inlines it, so the graph the author drew (the *authored*
  graph) and the graph that runs (the *expanded* graph) differ: in the
  authored graph the embedded flow is one node, ``approve``, whose fields
  are addressed through it, ``Ref("approve", "check.amount")``; in the
  expanded graph its inner nodes are nodes of the run, named
  ``approve/check``.

``execution_order``, ``node``, ``runner``, ``dependencies`` and
``lifted_on`` answer over the expanded graph; problems, rosters and the
interface are about the authored one. A question about a field may use
either address: ``type_of(Ref("approve", "check.amount"))`` and
``type_of(Ref("approve/check", "amount"))`` are the same question.

It is a plain value: immutable, no I/O, no session. The same graph and
the same registry always give the same ``CompiledGraph``, so it can be
cached, and "compile this and assert what it says" is a complete test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor.graph.binding import Binding
from conductor.graph.expand import expanded_ref
from conductor.graph.problem import Problem
from conductor.ref import Ref
from conductor.registry import runner_for

if TYPE_CHECKING:
    from conductor.graph.conditions import Condition
    from conductor.graph.model import Graph, GraphNode
    from conductor.interface import Interface
    from conductor.metadata import Roster
    from conductor.node import GraphVersion, NodeVersion
    from conductor.registry import NodeRegistry
    from conductor.series import Index


@dataclass(frozen=True)
class CompiledGraph:
    """The result of compiling one graph. Ask it; do not read through it.

    Built by ``from_graph`` and read by the engine
    (``execution_order``, ``runner``, ``roster``, ``lifted_on``), an
    editor's compile endpoint (``problems_for``, ``type_of``,
    ``interface``), and anything deciding whether a run may start
    (``is_runnable``).

    Every field but ``interface`` is private, and every public name is a
    question. A node compile could not resolve — unknown type or version —
    has a fatal ``Problem`` and no roster, version, index or types;
    asking about one raises, because the answer is in
    ``problems_for``.
    """

    _nodes: Mapping[str, GraphNode]
    _registry: NodeRegistry
    _versions: Mapping[str, NodeVersion | GraphVersion]
    _rosters: Mapping[str, Roster]
    _statics: Mapping[str, Mapping[str, Any]]
    _dependencies: Mapping[str, frozenset[str]]
    _order: tuple[str, ...]
    _lifted: Mapping[str, Index | None]
    _indexes: Mapping[Ref, Index | None]
    _types: Mapping[Ref, Any]
    _conditions: Mapping[Ref, Condition]
    #: Every node whose version is a graph, by expanded id — the ones the
    #: author placed and the ones nested inside them alike.
    _placements: frozenset[str]
    #: For every node of the expanded graph, the innermost embedded flow it
    #: came from, or ``None`` for a node the author placed.
    _placement_of: Mapping[str, str | None]
    #: What this graph takes and returns, in the same record a node version
    #: declares: ``inputs`` are the unlocked, connectable fields of the nodes
    #: nothing feeds into (each declaration whole, titled as the author
    #: titled it on that node), ``outputs`` every field of the nodes nothing
    #: reads from, both named by address (``"node.field"``); ``returns`` is
    #: ``Mapping``.
    interface: Interface
    _problems: tuple[Problem, ...]

    @classmethod
    def from_graph(cls, graph: Graph, registry: NodeRegistry) -> CompiledGraph:
        """Compile ``graph`` against ``registry``: the one way to get a ``CompiledGraph``.

        Pure — the same graph and registry always give the same result.
        Every definition the graph names must already be in the registry.
        Nothing raises for a fault in the graph; ask ``problems_for`` and
        ``is_runnable``. The passes are ``compiler._Compilation.build``'s.
        """
        from conductor.graph.compiler import _Compilation

        return _Compilation(graph, registry).build()

    # -- the two graphs ------------------------------------------------------

    def expanded(self, ref: Ref) -> Ref:
        """An address on the authored graph, read through to the node that
        runs: ``Ref("emb", "all.result")`` becomes ``Ref("emb/all", "result")``.
        A host reads engine results by the expanded address, since the
        engine knows only the expanded graph."""
        return expanded_ref(ref, self._placements)

    def placement_of(self, node_id: str) -> str | None:
        """The embedded flow a node of the expanded graph came from —
        ``"approve"`` for ``"approve/check"`` — or ``None`` for a node the
        author placed."""
        return self._placement_of[node_id]

    # -- edges -----------------------------------------------------------

    def value_source(self, node_id: str, input_name: str) -> Binding | None:
        """Where this input's value comes from: a ``Binding``, or ``None``.

        ``None`` means nothing binds the input and its declared default
        applies. Asking about an input the node does not have raises: that
        is a programming error, not a state of the graph.
        """
        ref = self.expanded(Ref(node_id, input_name))
        node = self.node(ref.node_id)
        if ref.field not in {i.name for i in self.roster(ref.node_id).inputs}:
            raise KeyError(f"{node.type!r} has no input {ref.field!r} on this node")
        return node.bindings.get(ref.field)

    def dependencies(self, node_id: str) -> frozenset[str]:
        """The nodes this one waits for, read off its edges."""
        return self._dependencies[node_id]

    # -- one node ------------------------------------------------------------

    def node(self, node_id: str) -> GraphNode:
        """One node the engine runs, by expanded id. A node whose version is
        a graph is not among them; its inner nodes are."""
        return self._nodes[node_id]

    def version(self, node_id: str) -> NodeVersion | GraphVersion:
        """The version this node uses: its ``run``, interface and policy — or,
        for an embedded flow, its interface and its graph."""
        return self._versions[node_id]

    def roster(self, node_id: str) -> Roster:
        """The inputs and outputs this node actually has, with every type
        the edges gave it — not merely what its version declared.

        The one place anything asks what a node has: the engine validates
        a call against it and an editor draws the rows from it. For an
        embedded flow, its version's interface."""
        return self._rosters[node_id]

    def statics(self, node_id: str) -> Mapping[str, Any]:
        """The values the author typed into this node, by field.

        Each read through the field's declared type — the value, never its
        JSON form. A value is a ``list`` exactly when the author typed
        many values.
        """
        return self._statics[node_id]

    def runner(self, node_id: str) -> Callable[..., Any]:
        """The callable that runs this node, on a fresh instance per call."""
        node = self.node(node_id)
        return runner_for(self._registry, node.type, node.version)

    def lifted_on(self, node_id: str) -> Index | None:
        """The index this node runs once per row of, or ``None`` when it runs
        once. Read off its edges — a series arriving on a scalar input is
        what makes a node run per row — and stored nowhere. For an embedded
        flow, the index its inner nodes run per row of, where a series
        entered it."""
        return self._lifted[node_id]

    # -- one field -------------------------------------------------------------

    def type_of(self, ref: Ref) -> Any:
        """The type that travels on this field: what an edge from it or into
        it carries. An output of a node that runs once per row carries
        ``Series[X]`` even where its declaration says ``X``; an input fed a
        series carries that series, before the engine slices it per row.
        An address on an embedded flow reads through to the inner field it
        names."""
        return self._types[self.expanded(ref)]

    def index_of(self, ref: Ref) -> Index | None:
        """Where the rows of the series on this field come from, or ``None``
        for a field that carries one value. For an input fed several series
        gathered together, the fresh index they were gathered onto. Read
        by the engine to lay values out per row."""
        return self._indexes[self.expanded(ref)]

    def condition(self, ref: Ref) -> Condition:
        """Under which condition this output appears: a boolean formula over
        the decisions upstream (see ``conductor.graph.conditions``),
        ``ALWAYS`` when nothing gates it. Derived from ``choice`` groups
        and edges; the engine never reads it."""
        return self._conditions[self.expanded(ref)]

    def decisions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every decision a caller could observe: for each node that runs
        once and declares a ``choice`` group, the group's alternatives in
        roster order, keyed by expanded node id. A node that runs per row is
        left out: its decision picks rows and gates nothing downstream."""
        found: dict[str, dict[str, tuple[str, ...]]] = {}
        for node_id in self._order:
            if self._lifted.get(node_id, None) is not None:
                continue
            groups: dict[str, list[str]] = {}
            for out in self._rosters[node_id].outputs:
                if out.choice is not None:
                    groups.setdefault(out.choice, []).append(out.name)
            if groups:
                found[node_id] = {choice: tuple(names) for choice, names in groups.items()}
        return found

    # -- the run ------------------------------------------------------------------

    def execution_order(self) -> tuple[str, ...]:
        """Expanded node ids in an order where every edge's source precedes
        its target. A node in a cycle is not in it; it has a fatal
        ``Problem`` instead."""
        return self._order

    # -- what is wrong -------------------------------------------------------------

    def problems_for(
        self, *, node_id: str | None = None, field: str | None = None
    ) -> tuple[Problem, ...]:
        """Every problem, or those about one node or one of its fields.

        Anchored on the authored graph: a problem found inside an embedded
        flow sits on the node the author placed, with the inner address as
        the field."""
        found = self._problems
        if node_id is not None:
            found = tuple(p for p in found if p.node_id == node_id)
        if field is not None:
            found = tuple(p for p in found if p.field == field)
        return found

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run refuses otherwise."""
        return not any(p.fatal for p in self._problems)
