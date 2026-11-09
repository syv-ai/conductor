"""``CompiledGraph`` — everything the compiler learned about a graph, as one immutable value, and every verb that runs it.

``CompiledGraph.from_graph`` builds it from a ``Graph`` and a ``NodeRegistry``.
Everyone else asks it questions and never reads the nodes' bindings
themselves: which version each node uses, which inputs and outputs it
actually has, where each input's value comes from and how each unit
receives it, what type travels on every field, which nodes run once per
row and in what order, under which condition each output appears, and
what is wrong. The engine is one more caller.

The questions come at three scales. The graph itself answers what it
takes and returns (``interface``), what is wrong with it (``problems``,
``is_runnable``), in what order its nodes run (``execution_order``), where
each runs and what its inputs receive (``run_plan``) and which decisions a run
makes (``decisions``). ``describe()`` is all of that as one JSON-ready record,
for an editor. ``node(node_id)`` returns a ``CompiledNode``, everything
compile knows about one node, and ``field(ref)`` a ``CompiledField``, the same
for one input or output of a node compile fully worked out. Both hand back the
same value on every call. A node compile couldn't fully work out says so by
its ``state``, and its ``cause`` says why. The graph stores a ``GraphNode`` (a node as the author
saved it) and addresses fields by ``Ref``; what compile learned about that
node or field lives on its ``CompiledNode`` and ``CompiledField``.

The terms this module uses throughout, each defined once here.

A node's interface is the list of inputs and outputs it actually has.
Usually that is what its version declares, but a node may add or drop
fields depending on the values it holds and the types connected to it.

A field that holds many values has a row for each, and an index names where those
rows come from: ten uploaded documents are ten rows on the index
of the node that uploaded them. A scalar input is an input that takes one
value. A node that receives many values on an input for one runs once per row
of them; we say the node runs per row of that index.

Some nodes have a version that is itself a graph, which the host compiled
on its own; that graph is an embedded graph, the node's ``embedded_graph``. Compile embeds the graph's nodes in place of that
node (``conductor._internal.graph.embedding``), so the graph the author drew (the
authored graph) differs from the graph that runs. In the authored graph the
embedded graph is one node, ``approve``; in the graph that runs its inner
nodes are nodes of the run, at their path, ``approve/check``. An embedded
graph is reached from outside only by the names it offers (``embedding``);
``interface`` names its fields ``approve.amount``, as any node's, and ``field``
reads that address through to the field inside.
``execution_order`` and ``decisions`` speak of the graph that runs;
``problems`` speak of the authored one, on the node and field the author
drew. ``node`` answers for both graphs: an inner node by its path
(``approve/check``) and the node that embeds it by its own id (``approve``).

It is a plain value: immutable, no I/O, no session. The same graph and
the same registry always give the same ``CompiledGraph``, so it can be
cached, and "compile this and assert what it says" is a complete test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from conductor._internal.errors import InvalidCallError, Problem, make_problem
from conductor._internal.graph.binding import Static
from conductor._internal.graph.compile_graph import compile_graph
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
from conductor._internal.graph.embedding import (
    NODE_PATH_SEPARATOR,
    EmbeddableGraph,
    embeddable_view_of,
    offered_names,
)
from conductor._internal.graph.graph_description import GraphDescription, describe_graph
from conductor._internal.graph.run_plan import RunPlan
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.graph.graph_records import Graph
    from conductor._internal.node_signature import Interface
    from conductor._internal.registry import NodeRegistry


def _place_of_offered_field(handle: Ref, graph: CompiledGraph) -> Ref:
    """Where the field ``handle`` names on a node that is ``graph`` is in the run: inside, at the node's path."""
    inside = graph._embeddable.inputs.get(handle.field) or graph._embeddable.outputs.get(handle.field)
    if inside is None:
        raise KeyError(f"{handle.node_id!r} offers no field {handle.field!r}")
    return Ref(f"{handle.node_id}{NODE_PATH_SEPARATOR}{inside.node_id}", inside.field)


@dataclass(frozen=True, repr=False)
class CompiledGraph:
    """The result of compiling one graph. Ask it; do not read through it.

    Built by ``from_graph`` and read by the engine (``_engine_lookups``, on
    every step of a run), an editor's compile endpoint (``describe``), and
    anything deciding whether a run may start (``is_runnable``).

    Three attributes are public, ``graph``, ``interface`` and ``problems``; every other
    attribute is private and read through the methods here and through
    ``node`` and ``field``. ``node`` answers for every node compile met and
    raises ``KeyError`` only for an id the graph does not have.
    """

    #: The registry the graph was compiled against, kept so a compiled
    #: graph can produce another (``with_inputs``).
    _registry: NodeRegistry
    #: What ``execution_order`` answers.
    _execution_order: tuple[str, ...]
    #: What this graph takes and returns, in the same record a node version
    #: declares: ``inputs`` are the unlocked, connectable fields of the nodes
    #: nothing feeds into (each declaration whole, titled as the author
    #: titled it on that node), ``outputs`` every field of the nodes nothing
    #: reads from, both named by address (``"node.field"``).
    interface: Interface
    #: The graph as the author saved it — with the values ``with_inputs``
    #: filled, on a copy it made — which a host stores as what ran. Not read
    #: back for what compile learned: ask ``node`` and ``field`` for that.
    graph: Graph
    #: Everything wrong with the graph, fatal or not, each on the node and
    #: field it is about, all on nodes of the authored graph. An embedded
    #: graph that cannot be embedded is one ``embedded_graph_broken`` on the
    #: graph node; its own problems are on it (``node(id).version.problems``).
    problems: tuple[Problem, ...]
    #: Every node compile met, by id: what ``node`` hands back.
    _nodes: Mapping[str, CompiledNode] = field(repr=False)
    #: The tables a merge matches against what they were built from.
    _row_ids_must_survive: frozenset[Ref] = field(repr=False)

    @classmethod
    def from_graph(cls, graph: Graph, registry: NodeRegistry) -> CompiledGraph:
        """Compile ``graph`` against ``registry``: the one way to get a ``CompiledGraph``.

        Pure: the same graph and registry always give the same result. Every
        definition the graph names must already be in the registry. Nothing
        raises for a problem in the graph; read ``problems`` and ``is_runnable``.
        A node's hook that raises anything but ``ComputeHookError`` is a bug in
        the node, and that exception leaves here as it is.
        """
        records = compile_graph(graph, registry)
        return cls(
            _registry=registry, _execution_order=records.order, interface=records.interface, graph=graph, problems=records.problems,
            _nodes=records.nodes, _row_ids_must_survive=records.row_ids_must_survive,
        )

    # -- one node, one field ---------------------------------------------------


    @cached_property
    def _embeddable(self) -> EmbeddableGraph:
        """What a graph that embeds this one asks of it (``conductor._internal.graph.embedding``)."""
        return embeddable_view_of(self)


    @cached_property
    def run_plan(self) -> RunPlan:
        """Where every ready node runs and what each of its inputs receives,
        in execution order: the shape compile decided, for a host to show and
        compare (``conductor._internal.graph.run_plan``). ``str(compiled.run_plan)`` is
        that as text. A node that isn't ready is absent."""
        return RunPlan.model_construct(nodes={
            node_id: node._run_plan for node_id in self._execution_order if (node := self._nodes[node_id])._wiring is not None
        })

    def node(self, node_id: str) -> CompiledNode:
        """One compiled node by its id: its path inside an embedded graph
        (``approve/check``). An id the graph does not have raises ``KeyError``."""
        if node_id not in self._nodes:
            raise KeyError(f"{node_id!r} is not a node of this graph")
        return self._nodes[node_id]

    def field(self, ref: Ref | str) -> CompiledField:
        """One compiled input or output by its address, a ``Ref`` or the string.

        An address on a node that is a graph reads the field that name offers,
        ``Ref("emb", "result")``, the same field as its place in the run,
        ``Ref("emb/all", "result")``. Anything the graph has no such field for
        raises ``KeyError``, naming the problem behind it when a node compile
        couldn't work out is why."""
        ref = ref if isinstance(ref, Ref) else Ref(ref)
        node = self.node(ref.node_id)
        if node._wiring is None:
            raise KeyError(f"{ref.node_id!r} has no fields compile could work out: {node.cause.msg if node.cause else ''}")
        if node.embedded_graph is not None:
            return self.field(_place_of_offered_field(ref, node.embedded_graph))
        compiled_field = node._wiring.fields.get(ref.field)
        if compiled_field is None:
            raise KeyError(f"{ref.node_id!r} has no field {ref.field!r} on this node")
        return compiled_field

    # -- the interface, from each side --------------------------------------------

    def with_inputs(self, inputs: Mapping[str, Any] | None = None, /, **by_name: Any) -> CompiledGraph:
        """This graph with some of its inputs filled, compiled again, to run with those values::

            compiled.with_inputs({"a.text": "hello"}, count=3)

        ``inputs`` is keyed by address (``a.text``, ``emb.text`` on a node that
        is a graph); a keyword is a field name when no other input has that name. A name the interface does not offer raises
        ``InvalidCallError``, code ``not_offered``, every such name at once.
        Each value becomes a ``Static`` on its input in a copy of the graph,
        which compiles against the same registry: a list on an input for one
        value runs the graph once per item. Only what the interface offers: a
        caller's body can go here, and ``Graph.with_values`` is an author's. A value its input's type can't
        read is an ``invalid_static``, raised with any names it doesn't offer.
        The copy offers every input as this one does. Not a run.
        """
        values = {**(inputs or {}), **by_name}
        offered = [input_.name for input_ in self.interface.inputs]
        problems: list[Problem] = []
        statics_to_set: dict[str, dict[str, Static]] = {}
        written_as: dict[tuple[str, str], str] = {}
        for name, value in values.items():
            ref = self._offered_input_named(name, offered, problems)
            if ref is not None:
                statics_to_set.setdefault(ref.node_id, {})[ref.field] = Static(value)
                written_as[ref.node_id, ref.field] = name
        refilled = CompiledGraph.from_graph(self.graph._with_statics(statics_to_set), self._registry)
        problems += [
            problem.model_copy(update={"loc": ("inputs", written_as[problem.loc])})  # type: ignore[index]
            for problem in refilled.problems if problem.type in ("invalid_static", "table_column_missing") and problem.loc in written_as
        ]
        if problems:
            raise InvalidCallError(problems)
        return refilled

    def _offered_input_named(self, name: str, offered: list[Ref], problems: list[Problem]) -> Ref | None:
        """The input a keyword to ``with_inputs`` names; or ``None``, and the problem that lists what is offered."""
        names = offered_names(offered)
        matches = [ref for ref in offered if ref == name or names[ref] == name]
        if matches:
            return matches[0]
        clashing = [ref for ref in offered if ref.field == name]
        if clashing:
            both = ", ".join(str(ref) for ref in clashing)
            reason = f"{name!r} is an input of several nodes ({both}); name one by its address"
        elif offered:
            reason = f"{name!r} is not an input this graph offers; it offers {', '.join(str(ref) for ref in offered)}"
        else:
            reason = f"{name!r}: this graph takes no inputs"
        problems.append(make_problem("not_offered", "inputs", name, reason=reason, offered=[str(ref) for ref in offered]))
        return None

    # -- the run ------------------------------------------------------------------

    def __repr__(self) -> str:
        """One line: the nodes the author drew, whether it runs, how many problems."""
        drawn = tuple(node.id for node in self.graph.nodes)
        return f"CompiledGraph(nodes={drawn!r}, is_runnable={self.is_runnable}, problems={len(self.problems)})"

    def describe(self) -> GraphDescription:
        """This compiled graph as an editor reads it: whether it runs, what is
        wrong, what it takes and returns, where each node runs, and per node
        the node's fields with the type on each (``GraphDescription``)."""
        return describe_graph(self)

    @property
    def execution_order(self) -> tuple[str, ...]:
        """The ids of the nodes that run, in an order where every edge's source precedes
        its target. A node in a cycle is not in it; it has a fatal
        ``Problem`` instead."""
        return self._execution_order

    @property
    def decisions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every decision a caller could observe: for each node that runs
        once and declares an ``exclusive_group`` group, the group's alternatives in
        field order, keyed by node id. A node that runs per row is
        left out: its decision picks rows and holds back nothing downstream. So
        is a node that is not ready: whether it runs per row is not
        known."""
        groups_by_node: dict[str, dict[str, tuple[str, ...]]] = {}
        for node_id in self._execution_order:
            node = self._nodes[node_id]
            if node._wiring is None or node._wiring._runs_per_row_of is not None:
                continue
            groups: dict[str, list[str]] = {}
            for out in node.interface.outputs:
                if out.exclusive_group is not None:
                    groups.setdefault(out.exclusive_group, []).append(out.name)
            if groups:
                groups_by_node[node_id] = {choice: tuple(names) for choice, names in groups.items()}
        return groups_by_node

    # -- running it ----------------------------------------------------------------


    # -- what is wrong -------------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run raises otherwise."""
        return not any(problem.fatal for problem in self.problems)
