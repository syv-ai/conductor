"""Views derived from the stored bindings: dependencies, and a flow's interface.

Nothing here is persisted. A dependency map answers "which nodes must
finish before this one starts" from the one place wiring lives; a flow's
interface answers "what does this flow take and return" from its nodes.

There is deliberately no edge view: an edge is something a canvas draws,
and a canvas derives its own from ``bindings``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from conductor.graph.binding import Sources
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem
from conductor.interface import Interface
from conductor.ref import Ref


def dependencies_of(nodes: Iterable[GraphNode]) -> dict[str, frozenset[str]]:
    """Which nodes each node waits for, by id.

    A set: a node that feeds two inputs of the same target is one
    dependency. Operand order matters only within a ``Sources`` and is
    kept there.
    """
    return {
        node.id: frozenset(
            ref.node_id
            for binding in node.bindings.values()
            if isinstance(binding, Sources)
            for ref in binding.refs
        )
        for node in nodes
    }

if TYPE_CHECKING:
    from conductor.graph.model import Flow
    from conductor.metadata import Input, Output, Roster
    from conductor.node import GraphVersion, NodeVersion


def is_input_node(node: GraphNode) -> bool:
    """Has this placement no wire into any of its inputs?

    A typed-in ``Static`` does not disqualify it; any ``Sources`` does.
    The one home of the rule, so ``derive_interface``, an editor and a
    migration all agree.
    """
    return not any(isinstance(binding, Sources) for binding in node.bindings.values())


def derive_interface(
    flow: "Flow",
    rosters: "Mapping[str, Roster]",
    versions: "Mapping[str, NodeVersion | GraphVersion]",
) -> tuple[Interface, tuple[Problem, ...]]:
    """What this flow takes and returns, derived from its nodes.

    An **input node** (no wire into any input) offers its unlocked,
    handle-bearing inputs; an **output node** (no wire out of any output)
    offers every output. A placement with a wire in, or with any output
    consumed, is an intermediate step and contributes nothing. The rule
    is per node, so ordinary editing does not shift the interface by
    accident.

    Each flow-level ``Input`` / ``Output`` is the placement's own record,
    whole, under its address ``Ref(node_id, field)`` as its name and
    wearing the placement's title. ``returns`` is ``Mapping`` (a flow
    returns its outputs by address); ``needs`` is the union of what the
    placements' versions need, by parameter name — a list for the host,
    since the engine fills each node from its own version's ``needs``.

    ``rosters`` is what each placement's hooks answered (one ``Roster``
    per placement the compiler could resolve; an unresolved placement
    carries its own problem and contributes nothing here) and ``versions``
    the version record each placement pinned. An embedded flow is a node
    here, with its own interface; expansion happens after this view.

    One problem can arise: a ``locked`` name the roster does not declare —
    a stale lock after the roster moved. Non-fatal, reported on every
    placement, wired or not, because it is repairable wherever it sits.

    Fields come in node order, roster order within a node.
    """
    consumed = {
        ref.node_id
        for node in flow.nodes
        for binding in node.bindings.values()
        if isinstance(binding, Sources)
        for ref in binding.refs
    }
    inputs: list["Input"] = []
    outputs: list["Output"] = []
    problems: list[Problem] = []

    for node in flow.nodes:
        if node.id not in rosters:
            continue
        roster = rosters[node.id]
        # A stale lock is reported on every placement, wired or not; a
        # wired placement still contributes no inputs.
        declared_inputs = {i.name for i in roster.inputs}
        for name in node.locked:
            if name not in declared_inputs:
                problems.append(Problem(
                    code="unknown_locked_field",
                    message=f"The lock on '{node.id}.{name}' points at a field the node does not have.",
                    fatal=False,
                    node_id=node.id,
                    field=name,
                ))
        if is_input_node(node):
            for declared in roster.inputs:
                if declared.show_handle and declared.name not in node.locked:
                    inputs.append(_content_of(node, declared))
        if node.id not in consumed:
            for declared in roster.outputs:
                outputs.append(_content_of(node, declared))

    needs: dict[str, type] = {}
    for version in versions.values():
        needs.update(version.interface.needs)

    return (
        Interface(inputs=tuple(inputs), outputs=tuple(outputs), returns=Mapping, needs=needs),
        tuple(problems),
    )


def _content_of(node: GraphNode, declared):
    """``declared`` renamed to its address, with this placement's title where one was authored.

    A placement built by an editor that copies titles carries a
    ``FieldContent`` per field; a hand-built graph may carry none, in which
    case the declaration's own title is the only one there is.
    """
    address = Ref(node.id, declared.name)
    content = node.fields.get(declared.name)
    if content is None:
        return replace(declared, name=address)
    return replace(declared, name=address, title=content.title, description=content.description)
