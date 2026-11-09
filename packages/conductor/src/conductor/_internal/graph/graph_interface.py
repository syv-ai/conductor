"""What a graph takes and returns.

A graph declares no inputs and no outputs; they are read off its nodes.
A node with no edge into any input offers its inputs to the caller, and
a node with no edge out of any output returns its outputs. This module
derives that surface. Nothing here is stored and nothing here walks the
graph: the one graph-wide fact it needs, which nodes consume which, comes
in as the dependency map ``topology.dependencies_of`` builds once per
compile; a node whose set there is empty is an input node.
``field_problems`` is the two rules every node's fields obey, both kept
apart from the derivation so an editor gets its problems without a
surface being rebuilt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from conductor._internal.errors import Problem, make_problem
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.node_signature import Interface
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.field_records import Input, Output
    from conductor._internal.graph.graph_records import Graph


def field_problems(
    node_id: str, fields: Sequence[Input | Output], taken: frozenset[str] = frozenset(), *, untyped_ok: bool
) -> list[Problem]:
    """Which of a node's fields break the two rules every field obeys?

    A field name is unique within a node across inputs and outputs, because
    a ``Ref(node, field)`` must name one field: a name repeated in ``fields``,
    or already in ``taken`` (the names on the other side), is
    ``duplicate_field_name``. And a field left untyped once its edges are
    known can carry nothing: with ``untyped_ok=False`` it is
    ``output_has_no_type``. An input typed ``Any`` takes its type from its
    edge and, unconnected, is ``unbound_required`` already.

    A declaration is checked for both when ``@node`` runs; a hook can
    compute a set of fields that breaks either, so compile asks again on
    what the hooks answered — the inputs before the walk over the edges,
    the outputs as the walk completes them.
    """
    problems: list[Problem] = []
    seen = set(taken)
    for declared in fields:
        if declared.name in seen:
            problems.append(make_problem("duplicate_field_name", node_id, declared.name))
        seen.add(declared.name)
        if declared.dtype.id == "any" and not untyped_ok:
            problems.append(make_problem("output_has_no_type", node_id, declared.name))
    return problems


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
    gave that field on that node. ``needs`` is the union of what the
    nodes need, by type.

    ``interfaces`` holds each node's actual fields (its computed
    ``Interface``, which carries the needs of the version it pinned); a
    node not in it is one compile could not resolve, and contributes
    nothing. ``dependencies`` is the map compile built once; a node in
    nobody's set is an output node. Fields come in node order, field order
    within a node.

    Asked once per compile, of the authored graph.
    """
    consumed = frozenset().union(*dependencies.values())
    inputs: list[Input] = []
    outputs: list[Output] = []
    needs: set[type] = set()
    for node in graph.nodes:
        if node.id not in interfaces:
            continue
        interface = interfaces[node.id]
        needs |= interface.needs
        if not dependencies[node.id]:
            inputs.extend(
                _as_offered(node, declared)
                for declared in interface.inputs
                if declared.connectable and declared.name not in node.locked
            )
        if node.id not in consumed:
            outputs.extend(_as_offered(node, declared) for declared in interface.outputs)
    return Interface(inputs=tuple(inputs), outputs=tuple(outputs), needs=frozenset(needs))


def _as_offered(node: GraphNode, declared):
    """``declared`` under its address, ``node.field``, an embedded graph's fields under the names it offers; with the title the author gave it on this node where there is one."""
    at = Ref(node.id, declared.name)
    content = node.fields.get(declared.name)
    if content is None:
        return declared.model_copy(update={"name": at})
    return declared.model_copy(update={"name": at, "title": content.title, "description": content.description})
