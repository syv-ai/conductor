"""``make_graph`` writes a graph record from Python: the nodes given, chained or joined by the edges given."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pydantic import JsonValue

from conductor._internal.base import REQUIRED, split_address
from conductor._internal.gate import RefusedError, record_from_parts, record_values
from conductor._internal.node import FunctionDefinition, NodeDefinition
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.record import GraphRecord, NodeRecord


def make_graph(
    *nodes: NodeDefinition,
    edges: Sequence[tuple[str, str]] | None = None,
    constants: Mapping[str, Any] = MappingProxyType({}),
    defaults: Mapping[str, Any] = MappingProxyType({}),
) -> GraphRecord:
    """A graph record written from Python, each node at its version::

        make_graph(lines, shout, join)                                # a chain
        make_graph(lines, shout, edges=[("lines.result", "shout.line")])
        make_graph(lines, shout, join, constants={"shout.mark": "?"})

    A node's id in the graph is its definition's id; a definition used more than once is ``shout-1``, ``shout-2``, and
    so on, skipping an id another node already has. ``constants`` and ``defaults`` are keyed by ``"node.input"``
    addresses of those ids: a constant is the input's value on every call, and a default is the value of a graph
    input when the caller gives none. Each is written in the record as its JSON reads back.

    With no ``edges`` the nodes form a chain: the one required input of each node that no constant or default fills
    gets an edge from the one output of the node before it. With ``edges``, those are every edge of the graph.
    Nothing is matched by name.

    The record is written and not compiled: an input or output a node does not have, a value its input's type
    refuses, an edge between types that do not fit and a graph not finished are in
    ``Graph.compile(record, registry).problems``.

    Raises:
        InvalidCallError: Listing every fault in writing the record itself: a value or default addressed to
            a node not given (``node_not_in_graph``), a value with no JSON form or holding a set
            (``invalid_value``), an edge that is no address (``invalid_record``) and a chain that can't
            be drawn (``not_a_chain``).
    """
    ids = _ids_of(nodes)
    problems: list[Problem] = []
    typed_constants: dict[str, dict[str, JsonValue]] = {node_id: {} for node_id in ids}
    typed_defaults: dict[str, dict[str, JsonValue]] = {node_id: {} for node_id in ids}
    for given, typed in ((constants, typed_constants), (defaults, typed_defaults)):
        by_node: dict[str, dict[str, Any]] = {}
        for address, value in given.items():
            node_id, name = split_address(address)
            if node_id not in typed:
                problems.append(problem.node_not_in_graph(node_id, nodes=ids))
            else:
                by_node.setdefault(node_id, {})[name] = value
        for node_id, named in by_node.items():
            try:
                typed[node_id] = record_values(node_id, named)
            except RefusedError as refused:
                problems += refused.problems
    if edges is None:
        edges, faults = _chain_edges(ids, nodes, typed_constants, typed_defaults)
        problems += faults
    try:
        record = record_from_parts(
            [
                NodeRecord(id=node_id, node=definition.id, version=definition.version, constants=typed_constants[node_id], defaults=typed_defaults[node_id])
                for node_id, definition in zip(ids, nodes, strict=True)
            ],
            edges,
        )
    except RefusedError as refused:  # an edge that is no address
        problems += refused.problems
    if problems:
        raise InvalidCallError(problems)
    return record


def _ids_of(nodes: Sequence[NodeDefinition]) -> list[str]:
    """Each node's id in the graph: its definition's, numbered ``-1``, ``-2`` for one used more than once."""
    used = Counter(definition.id for definition in nodes)
    taken = {definition.id for definition in nodes if used[definition.id] == 1}
    numbered: Counter[str] = Counter()
    ids: list[str] = []
    for definition in nodes:
        node_id = definition.id
        while used[definition.id] > 1 and (node_id == definition.id or node_id in taken):
            numbered[definition.id] += 1
            node_id = f"{definition.id}-{numbered[definition.id]}"
        taken.add(node_id)
        ids.append(node_id)
    return ids


def _chain_edges(
    ids: Sequence[str], nodes: Sequence[NodeDefinition], typed_constants: Mapping[str, Mapping[str, JsonValue]], typed_defaults: Mapping[str, Mapping[str, JsonValue]],
) -> tuple[list[tuple[str, str]], list[Problem]]:
    """The one edge from each node into the next, and ``not_a_chain`` for each pair without exactly one."""
    edges: list[tuple[str, str]] = []
    problems: list[Problem] = []
    for index in range(1, len(nodes)):
        before_id, before, after_id, after = ids[index - 1], nodes[index - 1], ids[index], nodes[index]
        filled = typed_constants[after_id].keys() | typed_defaults[after_id].keys()
        match before, after:
            case FunctionDefinition(), FunctionDefinition():
                outputs = list(before.outputs)
                open_inputs = [name for name, field in after.inputs.items() if field.default is REQUIRED and name not in filled]
            case _:  # a subgraph's fields are known only when it loads, so a chain can't be drawn through it
                outputs, open_inputs = [], []
        if len(outputs) != 1 or len(open_inputs) != 1:
            problems.append(problem.not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs))
        else:
            edges.append((f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"))
    return edges, problems
