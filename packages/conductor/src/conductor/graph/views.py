"""What a graph takes and returns.

A graph declares no inputs and no outputs; they are read off its nodes.
A node with no edge into any input offers its inputs to the caller, and
a node with no edge out of any output returns its outputs. This module
derives that surface. Nothing here is stored and nothing here walks the
graph: the one graph-wide fact it needs, which nodes consume which, comes
in as the dependency map ``topology.dependencies_of`` builds once per
compile. ``is_input_node`` is the same rule for one node, for a caller
holding one; ``lock_problems`` is the one check a lock can fail, kept
apart from the derivation so an editor gets its problems without a
surface being rebuilt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from conductor.graph.binding import Edges
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem
from conductor.interface import Interface
from conductor.ref import Ref

if TYPE_CHECKING:
    from conductor.graph.model import Graph
    from conductor.metadata import Input, Output, Roster
    from conductor.node import GraphVersion, NodeVersion


def is_input_node(node: GraphNode) -> bool:
    """Does no edge lead into any input of this node?

    A typed-in ``Static`` does not count; any ``Edges`` does. Such a node
    is an input node: its inputs are the graph's inputs. The one home of
    the rule, so ``derive_interface``, an editor and a migration agree; in
    a dependency map it reads as an empty set.
    """
    return not any(isinstance(binding, Edges) for binding in node.bindings.values())


def lock_problems(nodes: Mapping[str, GraphNode], rosters: Mapping[str, Roster]) -> tuple[Problem, ...]:
    """Which locks point at a field the node does not have?

    A lock (``GraphNode.locked``) is an input the graph's author closed:
    no caller may fill it, so it leaves the graph's inputs. A lock naming
    a field that is not on the node's roster is stale. It narrows nothing
    and blocks nothing, so the problem is non-fatal, and it is reported on
    every node, connected or not, because it is repairable wherever it
    sits. A node absent from ``rosters`` is one compile could not resolve;
    it carries its own problem.
    """
    return tuple(
        Problem(
            code="unknown_locked_field",
            message=f"The lock on '{node_id}.{name}' points at a field the node does not have.",
            fatal=False,
            node_id=node_id,
            field=name,
        )
        for node_id, roster in rosters.items()
        for name in nodes[node_id].locked
        if name not in {i.name for i in roster.inputs}
    )


def derive_interface(
    graph: Graph,
    rosters: Mapping[str, Roster],
    versions: Mapping[str, NodeVersion | GraphVersion],
    dependencies: Mapping[str, frozenset[str]],
) -> Interface:
    """What this graph takes and returns, read off its nodes.

    Inputs: every input of a node no edge leads into, except inputs with
    no handle and locked ones. Outputs: every output of a node no edge
    leads out of. A node with an edge in, or with any output consumed, is
    an intermediate step and contributes nothing. The rule is per node, so
    ordinary editing does not shift the surface by accident.

    Each ``Input`` / ``Output`` returned is the node's own record, named by
    its address ``Ref(node_id, field)`` and carrying the title the author
    gave that field on that node. ``returns`` is ``Mapping``: a graph
    returns its outputs by address. ``needs`` is the union of what the
    nodes' versions need, by parameter name.

    ``rosters`` holds each node's actual fields (``Roster``) and
    ``versions`` the version record each node pinned; a node in neither is
    one compile could not resolve, and contributes nothing.
    ``dependencies`` is the map compile built once; a node in nobody's set
    is an output node. Fields come in node order, roster order within a
    node.
    """
    consumed = frozenset().union(*dependencies.values())
    inputs: list[Input] = []
    outputs: list[Output] = []
    for node in graph.nodes:
        if node.id not in rosters:
            continue
        roster = rosters[node.id]
        if is_input_node(node):
            inputs.extend(
                _placed(node, declared)
                for declared in roster.inputs
                if declared.show_handle and declared.name not in node.locked
            )
        if node.id not in consumed:
            outputs.extend(_placed(node, declared) for declared in roster.outputs)

    needs: dict[str, type] = {}
    for version in versions.values():
        needs.update(version.interface.needs)
    return Interface(inputs=tuple(inputs), outputs=tuple(outputs), returns=Mapping, needs=needs)


def _placed(node: GraphNode, declared):
    """``declared`` under its address, with the title the author gave it on this node where there is one."""
    content = node.fields.get(declared.name)
    if content is None:
        return replace(declared, name=Ref(node.id, declared.name))
    return replace(declared, name=Ref(node.id, declared.name), title=content.title, description=content.description)
