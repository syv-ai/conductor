"""Inline an embedded graph as its inner nodes, under the name of the node that embeds it.

A node's version may be a graph rather than a ``run`` (``GraphVersion``).
The node then embeds a graph: one node in the editor and in the graph's
interface, many nodes when it runs. Below, a graph node is such a node
(a ``CompiledNode`` of ``kind`` ``graph``). Compile inlines its graph
here: every inner node becomes ``outer/inner``, inner edges are re-pointed
accordingly, the graph node's own bindings move onto the inner fields
they name, and edges from outside into the graph node are re-pointed at
the inner field they reach. The engine then runs one flat graph.

``/`` separates namespace levels inside a node id and ``.`` stays the
address separator, so an expanded address reads ``approve/check.amount``.
Only compile writes a ``/``: an id holding one, authored or inside an
embedded graph, is refused (``invalid_node_id``), or it could collide
with an inner node's expanded id and be read in its place. So an
expanded id is the path to its node, and the graph node it sits in is
read off it (``embedded_in``). The inverse — ``approve`` plus
``check.amount`` — is how a problem found inside is reported on the node
the author can see, and how ``CompiledGraph`` answers a question asked
with the graph node's own address.

A graph that embeds itself, directly or through another graph, would
expand forever; ``inline`` carries the chain of definitions being entered
and reports the node that closes the ring as a ``cycle``.

The expanded order is the authored order with each graph node replaced
by its inner order, recursively. That is a topological order of the
expanded graph with one extra property ``iteration.derive`` relies on:
every edge that crosses into a graph node's inner nodes is walked before
the first inner node is.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from conductor.graph.binding import Binding, From
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.topology import dependencies_of, order_of
from conductor.node import GraphVersion, NodeDefinition, NodeVersion
from conductor.ref import Ref
from conductor.registry import NodeRegistry

SEPARATOR = "/"


@dataclass(frozen=True)
class Expansion:
    """What ``expand`` returns: the expanded graph and how it was made.

    ``nodes`` holds every node the engine will run, by expanded id — no
    graph node is among them, only their inner nodes. ``order`` is the
    expanded execution order. ``members`` lists, for each graph node, every
    node under it, nested ones included. ``versions`` and ``definitions``
    hold what each node and each graph node resolved to (``resolved``).
    ``problems`` is what went wrong inside: an inner node whose type or
    version the registry lacks, an inner cycle, a binding on the graph node
    naming no inner field. The compiler reads all of it into the
    ``CompiledGraph``. Which graph node a node sits in is not stored: its
    id says (``embedded_in``).
    """

    nodes: dict[str, GraphNode]
    order: tuple[str, ...]
    members: dict[str, tuple[str, ...]]
    #: The version each node uses — a ``NodeVersion`` — and each graph node
    #: uses — a ``GraphVersion``, whose interface is what the graph node
    #: declares it takes and returns. A graph node comes before its inner nodes.
    versions: dict[str, NodeVersion | GraphVersion]
    #: The class each node and each graph node resolved to: what its hooks
    #: and its runner are made from.
    definitions: dict[str, type[NodeDefinition]]
    problems: tuple[Problem, ...]

    @cached_property
    def graphs(self) -> dict[str, Any]:
        """Every graph node, with the version it uses, a graph node before its
        inner ones: the one answer to "is this a graph node?", read by the
        compiler and by ``CompiledGraph``'s fold. A graph node is one whose
        version does not run and that is not itself a node of the run: a
        ``GraphVersion`` from the start, and a compiled graph once
        ``Compilation.place_graphs`` has put its nodes in its place (before
        that, the walk sees it as one node)."""
        return {
            node_id: version for node_id, version in self.versions.items()
            if not isinstance(version, NodeVersion) and node_id not in self.nodes
        }


def resolved(node: GraphNode, registry: NodeRegistry) -> tuple[type[NodeDefinition], NodeVersion | GraphVersion] | Problem:
    """The class a placed node names and the version it pins, or the problem saying which is missing.

    Compile's one lookup of a node, for a node the author placed
    (``Compilation.pin``) and for an inner node of an embedded graph
    (``inline``) alike. A stored graph can name a type the registry has
    since lost or a version the class has since dropped; either is a fatal
    problem on ``node.id``, which is the expanded id for an inner node.
    Everything after compile reads the class and version off the compiled
    node.
    """
    if node.type not in registry:
        return problem("unknown_node_type", node.id, node_type=node.type)
    definition = registry[node.type]
    version = definition.versions.get(node.version)
    if version is None:
        return problem("unknown_node_version", node.id, node_type=node.type, version=node.version)
    return definition, version


def expand(
    authored: Mapping[str, GraphNode],
    order: Sequence[str],
    pinned: Mapping[str, tuple[type[NodeDefinition], NodeVersion | GraphVersion]],
    registry: NodeRegistry,
) -> Expansion:
    """Inline every node in ``authored`` whose version is a ``GraphVersion``.

    ``order`` is the authored execution order; a node in a cycle is not in
    it and already carries a problem, as does a node absent from
    ``pinned`` (the class and version each authored node resolved to). An
    inner node whose type or version the registry lacks is a problem too,
    reported on the expanded id; the compiler records it on the graph node
    the author placed (``Compilation.record_as_the_author_sees_it``).
    """
    expander = _Expander(registry)
    for node_id in order:
        if node_id in pinned:
            definition, version = pinned[node_id]
            expander.inline(authored[node_id], definition, version, chain=())
    expander.reconnect()
    return expander.result()


class _Expander:
    """The expanded graph as it is being built, one node at a time.

    ``inline`` adds one authored node: a plain node goes in as it is; a
    graph node is entered and its inner nodes are added under its name,
    recursively. ``reconnect`` then re-points every edge that names a graph
    node's field at the inner field it reaches, and ``result`` freezes the
    whole into an ``Expansion``.
    """

    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry
        self.problems: list[Problem] = []
        self.nodes: dict[str, GraphNode] = {}
        self.order: list[str] = []
        self.members: dict[str, list[str]] = {}
        self.versions: dict[str, NodeVersion | GraphVersion] = {}
        self.definitions: dict[str, type[NodeDefinition]] = {}

    def inline(
        self,
        node: GraphNode,
        definition: type[NodeDefinition],
        version: NodeVersion | GraphVersion,
        chain: tuple[str, ...],
    ) -> None:
        """Add ``node`` to the expanded graph — itself, or its inner nodes
        under its name when ``version`` is a graph. ``chain`` is the
        definitions (by type id) whose graphs are being entered, so a graph
        that holds a node of a type already in it is a ``cycle`` on that
        node rather than an expansion without end."""
        if not isinstance(version, GraphVersion):
            # A node that runs, or a graph compiled on its own: the walk sees
            # both as one node, and ``Compilation.place_graphs`` lifts the second.
            self.nodes[node.id] = node
            self.order.append(node.id)
            self.versions[node.id] = version
            self.definitions[node.id] = definition
            for outer in _enclosing(node.id):
                self.members.setdefault(outer, []).append(node.id)
            return
        if node.type in chain:
            self.problems.append(problem("cycle", node.id))
            return
        self.versions[node.id] = version
        self.definitions[node.id] = definition
        self.members.setdefault(node.id, [])
        inner_nodes: dict[str, GraphNode] = {}
        for inner in version.graph:
            if SEPARATOR in inner.id:
                self.problems.append(problem("invalid_node_id", f"{node.id}{SEPARATOR}{inner.id}"))
                continue
            namespaced = self._namespaced(node.id, inner)
            inner_nodes[namespaced.id] = namespaced
        moved = self._moved(node, inner_nodes)
        inner_pinned: dict[str, tuple[type[NodeDefinition], NodeVersion | GraphVersion]] = {}
        for inner_id, inner in inner_nodes.items():
            found = resolved(inner, self.registry)
            if isinstance(found, Problem):
                self.problems.append(found)
            else:
                inner_pinned[inner_id] = found
        inner_order, cyclic = order_of(dependencies_of(inner_nodes.values()))
        self.problems.extend(problem("cycle", inner_id) for inner_id in sorted(cyclic))
        for inner_id in inner_order:
            if inner_id in inner_pinned:
                inner_definition, inner_version = inner_pinned[inner_id]
                self.inline(
                    moved.get(inner_id, inner_nodes[inner_id]), inner_definition, inner_version,
                    chain=(*chain, node.type),
                )

    def _moved(self, outer: GraphNode, inner_nodes: Mapping[str, GraphNode]) -> dict[str, GraphNode]:
        """The graph node's bindings, moved onto the inner fields they name.

        A binding on ``check.amount`` replaces whatever the inner ``check``
        held on ``amount`` — a value its author typed or an inner edge — for
        this graph node. A key naming no inner node is a stale binding,
        reported on the graph node with the inner nodes it could have named.
        """
        moved: dict[str, dict[str, Binding]] = {}
        for key, binding in outer.bindings.items():
            first, _, field = key.partition(".")
            inner = f"{outer.id}{SEPARATOR}{first}"
            if not field or inner not in inner_nodes:
                offered = sorted(f"{inner_id.removeprefix(outer.id + SEPARATOR)}.*" for inner_id in inner_nodes)
                self.problems.append(problem("stale_binding", outer.id, key, inputs=", ".join(offered) or "none"))
                continue
            moved.setdefault(inner, {})[field] = binding
        return {
            inner: inner_nodes[inner].model_copy(update={"bindings": {**inner_nodes[inner].bindings, **bindings}})
            for inner, bindings in moved.items()
        }

    def reconnect(self) -> None:
        """Every edge that names a graph node's field now names the inner
        field it reaches — outer edges into a graph node, and inner edges
        into a nested one alike."""
        graphs = self.members.keys()
        for node_id, node in list(self.nodes.items()):
            reconnected = {
                name: From(*(expanded_ref(ref, graphs) for ref in binding.refs))
                if isinstance(binding, From) else binding
                for name, binding in node.bindings.items()
            }
            self.nodes[node_id] = node.model_copy(update={"bindings": reconnected})

    def result(self) -> Expansion:
        return Expansion(
            nodes=self.nodes,
            order=tuple(self.order),
            members={outer: tuple(ids) for outer, ids in self.members.items()},
            versions=self.versions,
            definitions=self.definitions,
            problems=tuple(self.problems),
        )

    @staticmethod
    def _namespaced(outer: str, inner: GraphNode) -> GraphNode:
        """``inner`` renamed under its graph node: id prefixed, inner edges
        re-pointed, locks dropped (a lock hides an input from callers of the
        graph, and the graph node's own locks were already read by the
        interface). Built through the constructor, so the new id is checked."""
        return GraphNode(**{
            **dict(inner),
            "id": f"{outer}{SEPARATOR}{inner.id}",
            "bindings": {
                name: From(*(Ref(f"{outer}{SEPARATOR}{ref.node_id}", ref.field) for ref in binding.refs))
                if isinstance(binding, From) else binding
                for name, binding in inner.bindings.items()
            },
            "locked": (),
        })


def expanded_ref(ref: Ref, graphs: Collection[str]) -> Ref:
    """``Ref("approve", "check.amount")`` → ``Ref("approve/check", "amount")``, as deep as the graph nodes (``graphs``) go."""
    node_id, field = ref.node_id, ref.field
    while node_id in graphs and "." in field:
        inner, field = field.split(".", 1)
        node_id = f"{node_id}{SEPARATOR}{inner}"
    return Ref(node_id, field)


def authored_ref(ref: Ref) -> Ref:
    """``Ref("approve/check", "amount")`` → ``Ref("approve", "check.amount")``:
    the inverse of ``expanded_ref``, the address the author sees. An
    address with no ``/`` in its node id is already the author's."""
    if SEPARATOR not in ref.node_id:
        return ref
    outer, inner = ref.node_id.split(SEPARATOR, 1)
    return Ref(outer, f"{inner.replace(SEPARATOR, '.')}.{ref.field}")


#: The ``details`` keys that hold an address or a node id (``CATALOGUE``'s
#: ``source``, ``source_node``, ``a`` and ``b``); every other value — a
#: type's own sentence, a type id — is left as it is.
ADDRESS_KEYS: frozenset[str] = frozenset({"source", "source_node", "a", "b"})


def authored_address(value: str) -> str:
    """An expanded id or address rewritten to the author's: ``"e/p.a"`` →
    ``"e.p.a"``, ``"e/p"`` → ``"e.p"``; one with no ``/`` as it is."""
    if SEPARATOR not in value:
        return value
    return str(authored_ref(Ref(value))) if "." in value else value.replace(SEPARATOR, ".")


def embedded_in(expanded_id: str) -> str | None:
    """The graph node an expanded id sits in, read off the id — ``approve``
    for ``approve/check``, ``approve/check`` for ``approve/check/amount`` —
    or ``None`` for an id with no ``/``. Exact for every id ``expand``
    wrote, since no other id may hold a ``/``."""
    outer, _, _ = expanded_id.rpartition(SEPARATOR)
    return outer or None


def _enclosing(expanded_id: str) -> list[str]:
    """Every graph node an expanded id sits under, outermost first."""
    parts = expanded_id.split(SEPARATOR)
    return [SEPARATOR.join(parts[:depth]) for depth in range(1, len(parts))]
