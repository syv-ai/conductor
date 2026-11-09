"""How a graph is compiled: one forward walk, each node from the nodes compiled above it.

``CompiledGraph.from_graph(graph, registry)`` is the door. Pure: no session,
no I/O, no loading. Every definition the graph names must already be in the
registry. Everything wrong with the graph comes back as a ``Problem`` rather
than raising, so an editor can show a half-finished graph with its faults
marked; a hook that raises anything but ``ComputeHookError`` is a bug in the
node, and that exception leaves here as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conductor._internal.errors import Problem, make_problem
from conductor._internal.graph.compile_one_node import compile_one_node
from conductor._internal.graph.compiled_node import CompiledNode
from conductor._internal.graph.embedding import NODE_PATH_SEPARATOR
from conductor._internal.graph.graph_interface import derive_interface
from conductor._internal.graph.graph_records import Graph, GraphNode
from conductor._internal.graph.row_origins import RowOriginsSoFar
from conductor._internal.graph.topology import dependencies_of, topological_order
from conductor._internal.node_signature import Interface
from conductor._internal.ref import Ref
from conductor._internal.registry import NodeRegistry


@dataclass
class CompileProgress:
    """What compile has finished so far: the nodes above the one it is on, and what they left for it."""

    #: Every node compiled so far, by id; an embedded graph's nodes at their paths, after their graph node.
    compiled: dict[str, CompiledNode] = field(default_factory=dict)
    #: The ids written that can't name a node (``/`` or ``.`` in them), each with its problem.
    invalid_node_ids: dict[str, Problem] = field(default_factory=dict)
    #: The nodes that are a compiled graph.
    embedded_graph_node_ids: set[str] = field(default_factory=set)
    #: Problems found before a node's turn (its id was used twice).
    problems_found_early: dict[str, list[Problem]] = field(default_factory=dict)
    row_origins: RowOriginsSoFar = field(default_factory=RowOriginsSoFar)
    #: Every problem so far, in the order the nodes were compiled.
    problems: list[Problem] = field(default_factory=list)


@dataclass(frozen=True)
class CompileResult:
    """What the walk leaves for ``CompiledGraph``."""

    nodes: dict[str, CompiledNode]
    order: tuple[str, ...]
    interface: Interface
    problems: tuple[Problem, ...]
    row_ids_must_survive: frozenset[Ref]


def _node_compile_could_not_resolve(node: GraphNode, problem: Problem) -> CompiledNode:
    """A node compile can't resolve at all: the id or the wiring around it says why."""
    return CompiledNode(
        id=node.id, graph_node=node, embedded_in=None, problems=(problem,), cause=problem,
        definition=None, version=None, interface=None, statics={}, wiring=None,
    )


def compile_graph(graph: Graph, registry: NodeRegistry) -> CompileResult:
    """Walk ``graph`` once: refused ids and cycles first, then each node in an order where its sources come before it."""
    progress = CompileProgress()
    nodes_by_id: dict[str, GraphNode] = {}
    for node in graph.nodes:
        if NODE_PATH_SEPARATOR in node.id or "." in node.id or "@" in node.id:
            problem = make_problem("invalid_node_id", node.id)
            progress.invalid_node_ids[node.id] = problem
            progress.compiled[node.id] = _node_compile_could_not_resolve(node, problem)
            progress.problems.append(problem)
        elif node.id in nodes_by_id:
            progress.problems_found_early.setdefault(node.id, []).append(make_problem("duplicate_node_id", node.id))
        else:
            nodes_by_id[node.id] = node
    dependencies = dependencies_of(nodes_by_id.values())
    order, cyclic = topological_order(dependencies)
    for node_id in sorted(cyclic):
        problem = make_problem("cycle", node_id)
        progress.compiled[node_id] = _node_compile_could_not_resolve(nodes_by_id[node_id], problem)
        progress.problems.append(problem)
    for node_id in order:
        compile_one_node(nodes_by_id[node_id], progress, registry)
    compiled = progress.compiled
    interface = derive_interface(
        graph, {node_id: compiled[node_id].interface for node_id in nodes_by_id if compiled[node_id].interface is not None}, dependencies,
    )
    execution_order = tuple(
        node_id for node_id, node in compiled.items()
        if node.version is not None and (node.embedded_graph is None or node.wiring is None)
    )
    return CompileResult(compiled, execution_order, interface, tuple(progress.problems), frozenset(progress.row_origins.must_survive))


