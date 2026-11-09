"""How a graph is compiled: one forward walk, each node from the nodes compiled above it.

``CompiledGraph.from_graph(graph, registry)`` is the door. Pure: no session,
no I/O, no loading. Every definition the graph names must already be in the
registry. Everything wrong with the graph comes back as a ``Problem`` rather
than raising, so an editor can show a half-finished graph with its faults
marked; a hook that raises anything but ``CompileProblem`` is a bug in the
node, and that exception leaves here as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conductor._internal.errors import Problem, make_problem
from conductor._internal.graph.compile_one_node import compile_one_node
from conductor._internal.graph.compiled_node import CompiledNode
from conductor._internal.graph.embedding import SEPARATOR
from conductor._internal.graph.graph_interface import derive_interface
from conductor._internal.graph.graph_records import Graph, GraphNode
from conductor._internal.graph.row_origins import RowOriginsSoFar
from conductor._internal.graph.topology import dependencies_of, order_of
from conductor._internal.interface import Interface
from conductor._internal.ref import Ref
from conductor._internal.registry import NodeRegistry


@dataclass
class WalkSoFar:
    """What compile has finished so far: the nodes above the one it is on, and what they left for it."""

    #: Every node compiled so far, by id; an embedded graph's nodes at their paths, after their graph node.
    compiled: dict[str, CompiledNode] = field(default_factory=dict)
    #: The ids written that can't name a node (``/`` or ``.`` in them), each with its problem.
    invalid_node_ids: dict[str, Problem] = field(default_factory=dict)
    #: The nodes that are a compiled graph.
    graph_nodes: set[str] = field(default_factory=set)
    #: Problems found before a node's turn (its id was used twice).
    found_before: dict[str, list[Problem]] = field(default_factory=dict)
    row_origins: RowOriginsSoFar = field(default_factory=RowOriginsSoFar)
    #: Every problem so far, in the order the nodes were compiled.
    problems: list[Problem] = field(default_factory=list)


@dataclass(frozen=True)
class CompiledRecords:
    """What the walk leaves for ``CompiledGraph``."""

    nodes: dict[str, CompiledNode]
    order: tuple[str, ...]
    interface: Interface
    problems: tuple[Problem, ...]
    row_ids_must_survive: frozenset[Ref]


def _bare_node(node: GraphNode, found: Problem) -> CompiledNode:
    """A node compile can't resolve at all: the id or the wiring around it says why."""
    return CompiledNode(
        id=node.id, graph_node=node, embedded_in=None, problems=(found,), cause=found,
        definition=None, version=None, interface=None, statics={}, wiring=None,
    )


def compile_graph(graph: Graph, registry: NodeRegistry) -> CompiledRecords:
    """Walk ``graph`` once: refused ids and cycles first, then each node in an order where its sources come before it."""
    walk = WalkSoFar()
    authored: dict[str, GraphNode] = {}
    for node in graph.nodes:
        if SEPARATOR in node.id or "." in node.id or "@" in node.id:
            found = make_problem("invalid_node_id", node.id)
            walk.invalid_node_ids[node.id] = found
            walk.compiled[node.id] = _bare_node(node, found)
            walk.problems.append(found)
        elif node.id in authored:
            walk.found_before.setdefault(node.id, []).append(make_problem("duplicate_node_id", node.id))
        else:
            authored[node.id] = node
    dependencies = dependencies_of(authored.values())
    order, cyclic = order_of(dependencies)
    for node_id in sorted(cyclic):
        found = make_problem("cycle", node_id)
        walk.compiled[node_id] = _bare_node(authored[node_id], found)
        walk.problems.append(found)
    for node_id in order:
        compile_one_node(authored[node_id], walk, registry)
    compiled = walk.compiled
    interface = derive_interface(
        graph, {node_id: compiled[node_id].interface for node_id in authored if compiled[node_id].interface is not None}, dependencies,
    )
    execution_order = tuple(
        node_id for node_id, node in compiled.items()
        if node.version is not None and (node.embedded_graph is None or node.wiring is None)
    )
    return CompiledRecords(compiled, execution_order, interface, tuple(walk.problems), frozenset(walk.row_origins.must_survive))


