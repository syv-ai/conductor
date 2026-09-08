"""``CompiledGraph`` — everything known about a flow before it runs, as one value.

``compile_graph`` builds it from a ``Flow`` and a ``NodeRegistry``; everyone
else asks it questions: which version each node pins, which inputs and
outputs it actually has, where each input's value comes from, what type
travels on every field and on which index, which nodes run once per row,
in what order, under which condition each output appears, and what is
wrong. Callers never read the binding table themselves — the engine is
one more caller.

Two views of the graph meet here. The **authored** graph is what the
editor drew; an embedded flow is one node there, and its fields are
addressed through it: ``Ref("approve", "check.amount")``. The
**expanded** graph is what runs: the embedded flow's inner nodes, named
``approve/check``. ``execution_order``, ``node``, ``runner``,
``dependencies`` and ``lifted_on`` answer over the expanded graph;
problems, rosters and the interface are about the authored one. A
question about a field may use either address — ``carried(Ref("approve",
"check.amount"))`` and ``carried(Ref("approve/check", "amount"))`` are
the same question.

It is a plain value: immutable, no I/O, no session. The same flow and the
same registry always give the same ``CompiledGraph``, so it can be cached,
and "compile this and assert what it says" is a complete test.
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
    from conductor.graph.model import GraphNode
    from conductor.interface import Interface
    from conductor.metadata import Roster
    from conductor.node import GraphVersion, NodeVersion
    from conductor.registry import NodeRegistry
    from conductor.series import Index


@dataclass(frozen=True, slots=True)
class Carried:
    """What travels on one field: its type, and its index if it is a series.

    Produced by the lifting pass for every field of every node that has a
    shape, and read through ``CompiledGraph.carried`` — by the engine to
    know which index a value sits on, and by an editor to colour a handle
    or refuse an edge. ``Roster`` says *which* fields a node has; this says
    what arrives at one of them.

    At an output: what the node produces there once lifting is applied — a
    lifted node's scalar output carries ``Series[X]`` on the lift index. At
    an input: what arrives before the engine slices it per row — a lifted
    scalar input carries the series that lifted it; a ``Series[X]`` input
    carries the series it receives whole, or the fresh index a gather
    lands on. ``index`` is ``None`` for a scalar.
    """

    dtype: Any
    index: Index | None


@dataclass(frozen=True)
class CompiledGraph:
    """The result of compiling one flow. Ask it; do not read through it.

    ``compile_graph`` is the only writer. Readers: the engine
    (``execution_order``, ``runner``, ``roster``, ``lifted_on``), an
    editor's compile endpoint (``problems_for``, ``carried``,
    ``interface``), and anything deciding whether a run may start
    (``is_runnable``).

    Every field but ``interface`` is private, and every public name is a
    question. A node that did not resolve — unknown type or version — has a
    fatal ``Problem`` and no roster, version, lift index or carried fields;
    asking about one raises, because the answer is in ``problems_for``.
    """

    _nodes: Mapping[str, GraphNode]
    _registry: NodeRegistry
    _versions: Mapping[str, NodeVersion | GraphVersion]
    _rosters: Mapping[str, Roster]
    _statics: Mapping[str, Mapping[str, Any]]
    _dependencies: Mapping[str, frozenset[str]]
    _order: tuple[str, ...]
    _lifted: Mapping[str, Index | None]
    _carried: Mapping[Ref, Carried]
    _conditions: Mapping[Ref, Condition]
    #: Every placement whose version is a graph, by expanded id — authored
    #: ones and nested ones alike.
    _placements: frozenset[str]
    #: For every expanded node, the innermost placement it came from.
    _placement_of: Mapping[str, str | None]
    #: What this flow takes and returns, in the record a node version
    #: declares: ``inputs`` are the handle-bearing, unlocked fields of the
    #: input nodes (each declaration whole, titled by its placement),
    #: ``outputs`` every field of the output nodes' rosters, both named by
    #: address (``"node.field"``); ``returns`` is ``Mapping``.
    interface: Interface
    _problems: tuple[Problem, ...]

    # -- the two graphs ------------------------------------------------------

    def expanded(self, ref: Ref) -> Ref:
        """A placement-side address read through to the expanded node:
        ``Ref("emb", "all.result")`` → ``Ref("emb/all", "result")``. A host
        reads engine results by expanded address, since the engine knows
        only the expanded graph."""
        return expanded_ref(ref, self._placements)

    def placement_of(self, node_id: str) -> str | None:
        """The placement an expanded node came from — ``"approve"`` for
        ``"approve/check"`` — or ``None`` for a node the author placed."""
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
            raise KeyError(f"{node.type!r} has no input {ref.field!r} on this placement")
        return node.bindings.get(ref.field)

    def dependencies(self, node_id: str) -> frozenset[str]:
        """The nodes this one waits for, read off its edges."""
        return self._dependencies[node_id]

    # -- one node ------------------------------------------------------------

    def node(self, node_id: str) -> GraphNode:
        """One node the engine runs, by expanded id. A placement whose version
        is a graph is not among them; its inner nodes are."""
        return self._nodes[node_id]

    def version(self, node_id: str) -> NodeVersion | GraphVersion:
        """The version this node pins: its ``run``, interface and policy — or,
        for an embedded flow, its interface and its graph."""
        return self._versions[node_id]

    def roster(self, node_id: str) -> Roster:
        """This node's effective inputs and outputs: what its hooks answered,
        with every type bound by the edges — not merely what the version
        declared.

        The one place anything asks what a node has: the engine validates
        a call against it and an editor draws the rows from it. For an
        embedded placement, its version's interface."""
        return self._rosters[node_id]

    def statics(self, node_id: str) -> Mapping[str, Any]:
        """The author's typed values for this node, by field.

        Each ``Static`` validated through the field's declared type — the
        value, never its JSON form. A value is a ``list`` exactly when the
        static holds many values.
        """
        return self._statics[node_id]

    def runner(self, node_id: str) -> Callable[..., Any]:
        """The callable that runs this node, on a fresh instance per call."""
        node = self.node(node_id)
        return runner_for(self._registry, node.type, node.version)

    def lifted_on(self, node_id: str) -> Index | None:
        """The index this node runs once per row of, or ``None`` when it runs
        once. Derived from its edges; nothing stores it. For an embedded
        placement, the index its inner nodes lift on where a series
        entered."""
        return self._lifted[node_id]

    # -- one field -------------------------------------------------------------

    def carried(self, ref: Ref) -> Carried:
        """What travels on this field — see ``Carried``. A placement-side
        address reads through to the inner field it names."""
        return self._carried[self.expanded(ref)]

    def condition(self, ref: Ref) -> Condition:
        """The condition under which this output appears: a boolean formula
        over the upstream decisions (see ``conductor.graph.conditions``),
        ``ALWAYS`` when nothing gates it. Derived from ``choice`` groups
        and edges; the engine never reads it."""
        return self._conditions[self.expanded(ref)]

    def decisions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every decision a caller could observe: for each node that is not
        lifted and declares a ``choice`` group, the group's alternatives in
        roster order, keyed by expanded node id."""
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
        flow sits on its placement, with the inner address as the field."""
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
