"""What a flow takes and returns, derived from its nodes.

Nothing here is persisted and nothing here walks the graph: the
graph-wide fact the derivation needs — which nodes something consumes —
comes in as the dependency map ``topology.dependencies_of`` built once
per compile. ``is_input_node`` is the per-placement half of the same
rule, for a caller holding one node; ``lock_problems`` is the one check
a lock can fail, kept apart from the derivation so an editor gets its
problems without a surface being rebuilt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from conductor.graph.binding import Sources
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem
from conductor.interface import Interface
from conductor.ref import Ref

if TYPE_CHECKING:
    from conductor.graph.model import Flow
    from conductor.metadata import Input, Output, Roster
    from conductor.node import GraphVersion, NodeVersion


def is_input_node(node: GraphNode) -> bool:
    """Has this placement no wire into any of its inputs?

    A typed-in ``Static`` does not disqualify it; any ``Sources`` does.
    The one home of the rule, so ``derive_interface``, an editor and a
    migration all agree; in a dependency map it reads as an empty set.
    """
    return not any(isinstance(binding, Sources) for binding in node.bindings.values())


def lock_problems(nodes: Mapping[str, GraphNode], rosters: Mapping[str, Roster]) -> tuple[Problem, ...]:
    """A ``locked`` name the placement's roster does not declare — stale narrowing.

    Non-fatal, and reported on every placement, wired or not: a stale lock
    narrows nothing and blocks nothing, and is repairable wherever it sits.
    A placement absent from ``rosters`` is one compile could not resolve; it
    carries its own problem.
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
    flow: Flow,
    rosters: Mapping[str, Roster],
    versions: Mapping[str, NodeVersion | GraphVersion],
    dependencies: Mapping[str, frozenset[str]],
) -> Interface:
    """What this flow takes and returns, derived from its nodes.

    An **input node** (no wire into any input) offers its unlocked,
    handle-bearing inputs; an **output node** (nothing wired out of any
    output) offers every output. A placement with a wire in, or with any
    output consumed, is an intermediate step and contributes nothing. The
    rule is per node, so ordinary editing does not shift the interface by
    accident.

    Each flow-level ``Input`` / ``Output`` is the placement's own record,
    whole, under its address ``Ref(node_id, field)`` as its name and
    wearing the placement's title. ``returns`` is ``Mapping`` (a flow
    returns its outputs by address); ``needs`` is the union of what the
    placements' versions need, by parameter name.

    ``rosters`` is what each placement's hooks answered and ``versions``
    the version record each placement pinned — a placement in neither is
    one compile could not resolve, and contributes nothing. ``dependencies``
    is the map compile built once; a node in nobody's set is an output
    node. Fields come in node order, roster order within a node.
    """
    consumed = frozenset().union(*dependencies.values())
    inputs: list[Input] = []
    outputs: list[Output] = []
    for node in flow.nodes:
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
    """``declared`` under its address, wearing the placement's title where one was authored."""
    content = node.fields.get(declared.name)
    if content is None:
        return replace(declared, name=Ref(node.id, declared.name))
    return replace(declared, name=Ref(node.id, declared.name), title=content.title, description=content.description)
