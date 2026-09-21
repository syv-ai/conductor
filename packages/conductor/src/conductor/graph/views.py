"""What a graph takes and returns.

A graph declares no inputs and no outputs; they are read off its nodes.
A node with no edge into any input offers its inputs to the caller, and
a node with no edge out of any output returns its outputs. This module
derives that surface. Nothing here is stored and nothing here walks the
graph: the one graph-wide fact it needs, which nodes consume which, comes
in as the dependency map ``topology.dependencies_of`` builds once per
compile. ``is_input_node`` is the same rule for one node, for a caller
holding one; ``lock_problems`` is the one check a lock can fail, and
``field_problems`` the two rules every node's fields obey, both kept
apart from the derivation so an editor gets its problems without a
surface being rebuilt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from conductor.dtype import DType
from conductor.graph.binding import Edges
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem, problem
from conductor.interface import Interface
from conductor.ref import Ref

if TYPE_CHECKING:
    from conductor.graph.model import Graph
    from conductor.metadata import Input, Output


def is_input_node(node: GraphNode) -> bool:
    """Does no edge lead into any input of this node?

    A typed-in ``Static`` does not count; any ``Edges`` does. Such a node
    is an input node: its inputs are the graph's inputs. The one home of
    the rule, so ``derive_interface``, an editor and a migration agree; in
    a dependency map it reads as an empty set.
    """
    return not any(isinstance(binding, Edges) for binding in node.bindings.values())


def lock_problems(nodes: Mapping[str, GraphNode], interfaces: Mapping[str, Interface]) -> tuple[Problem, ...]:
    """Which locks point at a field the node does not have?

    A lock (``GraphNode.locked``) is an input the graph's author closed:
    no caller may fill it, so it leaves the graph's inputs. A lock naming
    a field that is not on the node's interface is stale. It narrows nothing
    and blocks nothing, so the problem is non-fatal, and it is reported on
    every node, connected or not, because it is repairable wherever it
    sits. A node absent from ``interfaces`` is one compile could not resolve;
    it carries its own problem.
    """
    return tuple(
        problem("unknown_locked_field", node_id, name)
        for node_id, interface in interfaces.items()
        for name in nodes[node_id].locked
        if name not in {i.name for i in interface.inputs}
    )


def field_problems(
    node_id: str, fields: Sequence[Input | Output], taken: frozenset[str] = frozenset(), *, untyped_ok: bool
) -> list[Problem]:
    """Which of a node's fields break the two rules every field obeys?

    A field name is unique within a node across inputs and outputs, because
    a ``Ref(node, field)`` must name one field: a name repeated in ``fields``,
    or already in ``taken`` (the names on the other side), is
    ``duplicate_field_name``. And every field that can be connected — every
    output, and every input not closed with ``show_handle=False`` — must
    carry a ``DType``, or nothing could connect it: ``handle_needs_dtype``.
    ``Any`` is allowed with ``untyped_ok``: an input typed ``Any`` takes its
    type from its edge and, unconnected, is ``unbound_required`` already; an
    output a hook left ``Any`` once the edges are known can carry nothing,
    so compile asks with ``untyped_ok=False`` there.

    A declaration is checked for both when the class is defined; a hook can
    compute a set of fields that breaks either, so compile asks again on
    what the hooks answered — the inputs before the walk over the edges,
    the outputs as the walk completes them.
    """
    found: list[Problem] = []
    seen = set(taken)
    for declared in fields:
        if declared.name in seen:
            found.append(problem("duplicate_field_name", node_id, declared.name))
        seen.add(declared.name)
        if not getattr(declared, "show_handle", True):
            continue
        if declared.dtype is Any:
            if not untyped_ok:
                found.append(problem("handle_needs_dtype", node_id, declared.name))
        elif not (isinstance(declared.dtype, type) and issubclass(declared.dtype, DType)):
            found.append(problem("handle_needs_dtype", node_id, declared.name))
    return found


def derive_interface(
    graph: Graph,
    interfaces: Mapping[str, Interface],
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
    nodes need, by parameter name.

    ``interfaces`` holds each node's actual fields (its computed
    ``Interface``, which carries the needs of the version it pinned); a
    node not in it is one compile could not resolve, and contributes
    nothing. ``dependencies`` is the map compile built once; a node in
    nobody's set is an output node. Fields come in node order, field order
    within a node.

    Asked twice per compile: of the authored graph, for what the graph
    takes and returns, and of each embedded graph, for what its placement
    takes and returns (``_Compilation.interface``).
    """
    consumed = frozenset().union(*dependencies.values())
    inputs: list[Input] = []
    outputs: list[Output] = []
    needs: dict[str, type] = {}
    for node in graph.nodes:
        if node.id not in interfaces:
            continue
        interface = interfaces[node.id]
        needs.update(interface.needs)
        if is_input_node(node):
            inputs.extend(
                _placed(node, declared)
                for declared in interface.inputs
                if declared.show_handle and declared.name not in node.locked
            )
        if node.id not in consumed:
            outputs.extend(_placed(node, declared) for declared in interface.outputs)
    return Interface(inputs=tuple(inputs), outputs=tuple(outputs), returns=Mapping, needs=needs)


def _placed(node: GraphNode, declared):
    """``declared`` under its address, with the title the author gave it on this node where there is one."""
    content = node.fields.get(declared.name)
    if content is None:
        return declared.model_copy(update={"name": Ref(node.id, declared.name)})
    return declared.model_copy(update={"name": Ref(node.id, declared.name), "title": content.title, "description": content.description})
