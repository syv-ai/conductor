"""Inline an embedded graph as its inner nodes, under the name of the node that embeds it.

A node's version may be a graph rather than a ``run`` (``GraphVersion``).
The node then embeds a graph: one node in the editor and in the graph's
interface, many nodes when it runs. Any placed node is a placement of its
definition (``GraphNode``); in this module "the placement" means the
node that embeds a graph. Compile inlines its graph here: every inner node becomes
``placement/inner``, inner edges are re-pointed accordingly, the
placement's own bindings move onto the inner fields they name, and edges
from outside into the placement are re-pointed at the inner field they
reach. The engine then runs one flat graph.

``/`` separates namespace levels inside a node id and ``.`` stays the
address separator, so an expanded address reads ``approve/check.amount``.
Only compile writes a ``/``: an authored id holding one is refused
(``invalid_node_id``), or it could collide with an inner node's expanded
id and be read in its place. The inverse — ``approve`` plus
``check.amount`` — is how a problem found inside is reported on the node
the author can see, and how ``CompiledGraph`` answers a question asked
with the placement's own address.

A graph that embeds itself, directly or through another graph, would
expand forever; ``inline`` carries the chain of definitions being entered
and reports the node that closes the ring as a ``cycle``.

The expanded order is the authored order with each placement replaced by
its inner order, recursively. That is a topological order of the expanded
graph with one extra property ``iteration.derive`` relies on: every edge
that crosses into a placement's inner nodes is walked before the first
inner node is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from conductor.graph.binding import Binding, Edges
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.topology import dependencies_of, order_of
from conductor.node import GraphVersion, NodeVersion
from conductor.ref import Ref
from conductor.registry import NodeRegistry

SEPARATOR = "/"


@dataclass(frozen=True)
class Expansion:
    """What ``expand`` returns: the expanded graph and how it was made.

    ``nodes`` holds every node the engine will run, by expanded id — no
    placement is among them, only their inner nodes. ``order`` is the
    expanded execution order. ``placement_of`` names, for each node and
    each placement, the innermost placement it came from (``None`` for one
    the author placed). ``members`` lists, for each placement, every node under it,
    nested ones included. ``versions`` is the ``NodeVersion`` each node
    uses. ``problems`` is what went wrong inside: an inner node whose
    type or version the registry lacks, an inner cycle, a binding on the
    placement naming no inner field. The compiler reads all of it into the
    ``CompiledGraph``.
    """

    nodes: dict[str, GraphNode]
    order: tuple[str, ...]
    placement_of: dict[str, str | None]
    members: dict[str, tuple[str, ...]]
    versions: dict[str, NodeVersion]
    #: The ``GraphVersion`` of each placement, authored and nested alike; its
    #: interface is what the placement shows as inputs and outputs.
    placement_versions: dict[str, GraphVersion]
    problems: tuple[Problem, ...]


def expand(
    authored: Mapping[str, GraphNode],
    order: Sequence[str],
    versions: Mapping[str, NodeVersion | GraphVersion],
    registry: NodeRegistry,
) -> Expansion:
    """Inline every node in ``authored`` whose version is a ``GraphVersion``.

    ``order`` is the authored execution order; a node in a cycle is not in
    it and already carries a problem, as does a node absent from
    ``versions``. An inner node whose type or version the registry lacks
    is a problem too, reported on the expanded id and moved onto the
    placement by ``surfaced``.
    """
    expander = _Expander(registry)
    for node_id in order:
        if node_id in versions:
            expander.inline(authored[node_id], versions[node_id], enclosing=None, chain=())
    expander.reconnect()
    return expander.result()


class _Expander:
    """The expanded graph as it is being built, one node at a time.

    ``inline`` adds one authored node: a plain node goes in as it is; a
    node whose version is a graph is entered and its inner nodes are added
    under its name, recursively. ``reconnect`` then re-points every edge
    that names a placement's field at the inner field it reaches, and
    ``result`` freezes the whole into an ``Expansion``.
    """

    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry
        self.problems: list[Problem] = []
        self.nodes: dict[str, GraphNode] = {}
        self.order: list[str] = []
        self.placement_of: dict[str, str | None] = {}
        self.members: dict[str, list[str]] = {}
        self.versions: dict[str, NodeVersion] = {}
        self.placement_versions: dict[str, GraphVersion] = {}
        self.placements: set[str] = set()

    def inline(self, node: GraphNode, version: NodeVersion | GraphVersion, enclosing: str | None, chain: tuple[str, ...]) -> None:
        """Add ``node`` to the expanded graph — itself, or its inner nodes
        under its name when ``version`` is a graph. ``enclosing`` is the
        innermost placement the node sits in, ``None`` at the top; ``chain``
        the definitions (by type id) whose graphs are being entered, so a
        graph that holds a node of a type already in it is a ``cycle`` on
        that node rather than an expansion without end."""
        if isinstance(version, NodeVersion):
            self.nodes[node.id] = node
            self.order.append(node.id)
            self.placement_of[node.id] = enclosing
            self.versions[node.id] = version
            for placement in _enclosing(node.id):
                self.members.setdefault(placement, []).append(node.id)
            return
        if node.type in chain:
            self.problems.append(problem("cycle", node.id))
            return
        self.placement_of[node.id] = enclosing
        self.placements.add(node.id)
        self.placement_versions[node.id] = version
        self.members.setdefault(node.id, [])
        inner_nodes: dict[str, GraphNode] = {}
        for inner in version.graph:
            if SEPARATOR in inner.id:
                self.problems.append(problem("invalid_node_id", f"{node.id}{SEPARATOR}{inner.id}"))
                continue
            namespaced = self._namespaced(node.id, inner)
            inner_nodes[namespaced.id] = namespaced
        moved = self._moved(node, inner_nodes)
        inner_versions: dict[str, NodeVersion | GraphVersion] = {}
        for inner_id, inner in inner_nodes.items():
            if inner.type not in self.registry:
                self.problems.append(problem("unknown_node_type", inner_id, node_type=inner.type))
                continue
            inner_version = self.registry[inner.type].versions.get(inner.version)
            if inner_version is None:
                self.problems.append(
                    problem("unknown_node_version", inner_id, node_type=inner.type, version=inner.version)
                )
                continue
            inner_versions[inner_id] = inner_version
        inner_order, cyclic = order_of(dependencies_of(inner_nodes.values()))
        self.problems.extend(problem("cycle", inner_id) for inner_id in sorted(cyclic))
        for inner_id in inner_order:
            if inner_id in inner_versions:
                self.inline(moved.get(inner_id, inner_nodes[inner_id]), inner_versions[inner_id], enclosing=node.id, chain=(*chain, node.type))

    def _moved(self, placement: GraphNode, inner_nodes: Mapping[str, GraphNode]) -> dict[str, GraphNode]:
        """The placement's bindings, moved onto the inner fields they name.

        A binding on ``check.amount`` replaces whatever the inner ``check``
        held on ``amount`` — a value its author typed or an inner edge — for
        this placement. A key naming no inner node is a stale binding,
        reported on the placement with the inner nodes it could have named.
        """
        moved: dict[str, dict[str, Binding]] = {}
        for key, binding in placement.bindings.items():
            first, _, field = key.partition(".")
            inner = f"{placement.id}{SEPARATOR}{first}"
            if not field or inner not in inner_nodes:
                offered = sorted(f"{inner_id.removeprefix(placement.id + SEPARATOR)}.*" for inner_id in inner_nodes)
                self.problems.append(problem("stale_binding", placement.id, key, inputs=", ".join(offered) or "none"))
                continue
            moved.setdefault(inner, {})[field] = binding
        return {
            inner: inner_nodes[inner].model_copy(update={"bindings": {**inner_nodes[inner].bindings, **bindings}})
            for inner, bindings in moved.items()
        }

    def reconnect(self) -> None:
        """Every edge that names a placement's field now names the inner
        field it reaches — outer edges into a placement, and inner edges
        into a nested one alike."""
        for node_id, node in list(self.nodes.items()):
            reconnected = {
                name: Edges(refs=tuple(expanded_ref(ref, self.placements) for ref in binding.refs))
                if isinstance(binding, Edges) else binding
                for name, binding in node.bindings.items()
            }
            self.nodes[node_id] = node.model_copy(update={"bindings": reconnected})

    def result(self) -> Expansion:
        return Expansion(
            nodes=self.nodes,
            order=tuple(self.order),
            placement_of=self.placement_of,
            members={placement: tuple(ids) for placement, ids in self.members.items()},
            versions=self.versions,
            placement_versions=self.placement_versions,
            problems=tuple(self.problems),
        )

    @staticmethod
    def _namespaced(placement: str, inner: GraphNode) -> GraphNode:
        """``inner`` renamed under the placement: id prefixed, inner edges
        re-pointed, locks dropped (a lock hides an input from callers of the
        graph, and the placement's own locks were already read by the
        interface). Built through the constructor, so the new id is checked."""
        return GraphNode(**{
            **dict(inner),
            "id": f"{placement}{SEPARATOR}{inner.id}",
            "bindings": {
                name: Edges(refs=tuple(Ref(f"{placement}{SEPARATOR}{ref.node_id}", ref.field) for ref in binding.refs))
                if isinstance(binding, Edges) else binding
                for name, binding in inner.bindings.items()
            },
            "locked": (),
        })


def expanded_ref(ref: Ref, placements: frozenset[str] | set[str]) -> Ref:
    """``Ref("approve", "check.amount")`` → ``Ref("approve/check", "amount")``, as deep as the placements go."""
    node_id, field = ref.node_id, ref.field
    while node_id in placements and "." in field:
        inner, field = field.split(".", 1)
        node_id = f"{node_id}{SEPARATOR}{inner}"
    return Ref(node_id, field)


def authored_ref(ref: Ref) -> Ref:
    """``Ref("approve/check", "amount")`` → ``Ref("approve", "check.amount")``:
    the inverse of ``expanded_ref``, the address the author sees. An
    address with no ``/`` in its node id is already the author's."""
    if SEPARATOR not in ref.node_id:
        return ref
    placement, inner = ref.node_id.split(SEPARATOR, 1)
    return Ref(placement, f"{inner.replace(SEPARATOR, '.')}.{ref.field}")


def surfaced(problem: Problem, nodes: Mapping[str, GraphNode], authored: Iterable[str]) -> Problem:
    """Move a problem found inside a placement onto the placement, where the author can see it.

    A problem on a node the author placed (``authored``) is theirs already
    and is left alone — including the one about an authored id that holds
    a ``/``. The author sees ``approve``, not ``approve/check``: the node becomes
    the placement, the field becomes the inner address the authored graph
    already uses (``check.amount``), and the message is prefixed with the
    inner node's title. The code stays the inner problem's, and so do its
    ``details`` under their own keys — every expanded address in them
    rewritten to the author's (``approve/check.amount`` → ``approve.check.amount``)
    — with ``placement`` (the inner node's title) and ``inner_message``
    added beside them, so a host translating by code finds the same keys
    inside a placement as outside one.
    """
    if SEPARATOR not in problem.node_id or problem.node_id in authored:
        return problem
    placement, inner = problem.node_id.split(SEPARATOR, 1)
    inner_address = inner.replace(SEPARATOR, ".")
    title = (nodes[problem.node_id].title if problem.node_id in nodes else None) or inner_address
    return problem.model_copy(update={
        "node_id": placement,
        "field": authored_ref(Ref(problem.node_id, problem.field)).field if problem.field else inner_address,
        "message": f"In '{title}': {problem.message}",
        "details": {
            **{key: authored_address(value) if key in ADDRESS_KEYS else value for key, value in problem.details.items()},
            "placement": title,
            "inner_message": problem.message,
        },
    })


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


def _enclosing(expanded_id: str) -> list[str]:
    """Every placement an expanded id sits under, outermost first."""
    parts = expanded_id.split(SEPARATOR)
    return [SEPARATOR.join(parts[:depth]) for depth in range(1, len(parts))]
