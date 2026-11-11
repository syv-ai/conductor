"""``make_graph`` writes a graph record from Python: the nodes given, chained or joined by the edges given."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pydantic import JsonValue, ValidationError

from conductor._internal import problem
from conductor._internal.fields import REQUIRED
from conductor._internal.graph_record import GraphRecord, NodeRecord, not_a_record, typed_in
from conductor._internal.node_definition import FunctionDefinition, NodeDefinition
from conductor._internal.problem import InvalidCallError, Problem, make_problem


def make_graph(
    *nodes: NodeDefinition,
    edges: Sequence[tuple[str, str]] | None = None,
    values: Mapping[str, Any] = MappingProxyType({}),
    defaults: Mapping[str, Any] = MappingProxyType({}),
) -> GraphRecord:
    """A graph record written from Python, each node at its version::

        make_graph(read_file, upper)                                  # a chain
        make_graph(read_file, upper, edges=[("read_file.result", "upper.text")])
        make_graph(read_file, upper, values={"read_file.path": "in.txt"})

    A node's id in the graph is its definition's id; a definition used more
    than once is ``upper-1``, ``upper-2``, and so on, skipping an id another
    node already has. ``values`` and ``defaults`` are keyed by ``"node.input"``
    addresses of those ids, as the record's fields of those names are: a value
    is fixed, and a default stays open to the caller. A list typed into an input
    that takes one value is refused: a node that returns a ``Series`` makes
    the items a graph runs per.

    With no ``edges`` the nodes form a chain: each takes its one required
    input that nothing typed in fills from the one output of the node before
    it. With ``edges``, those are every edge of the graph. Nothing is matched
    by name.

    The record is written and nothing is loaded: an input or output a node
    doesn't have, a value its field refuses, an edge between types that don't
    meet and a graph half drawn are in ``registry.load(record).problems``.

    Raises:
        InvalidCallError: Listing every fault in writing the record itself: a value or default addressed to
            a node not given (``node_not_in_graph``), a value with no JSON form or holding a set
            (``invalid_value``), an edge that is no address (``invalid_record``) and a chain that can't
            be drawn (``not_a_chain``).
    """
    ids = _ids_of(nodes)
    problems: list[Problem] = []
    typed_values: dict[str, dict[str, JsonValue]] = {node_id: {} for node_id in ids}
    typed_defaults: dict[str, dict[str, JsonValue]] = {node_id: {} for node_id in ids}
    for given, typed in ((values, typed_values), (defaults, typed_defaults)):
        by_node: dict[str, dict[str, Any]] = {}
        for address, value in given.items():
            node_id, _, name = address.partition(".")
            if node_id not in typed:
                problems.append(make_problem("node_not_in_graph", node_id, node_id=node_id, nodes=ids))
            else:
                by_node.setdefault(node_id, {})[name] = value
        for node_id, named in by_node.items():
            typed[node_id], faults = typed_in(node_id, named)
            problems += faults
    if edges is None:
        edges, faults = _chain_edges(ids, nodes, typed_values, typed_defaults)
        problems += faults
    try:
        record = GraphRecord(
            nodes=[
                NodeRecord(id=node_id, node=definition.id, version=definition.version, values=typed_values[node_id], defaults=typed_defaults[node_id])
                for node_id, definition in zip(ids, nodes, strict=True)
            ],
            edges=edges,
        )
    except ValidationError as invalid:  # an edge that is no address
        problems += not_a_record(invalid)
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
    ids: Sequence[str], nodes: Sequence[NodeDefinition], typed_values: Mapping[str, Mapping[str, JsonValue]], typed_defaults: Mapping[str, Mapping[str, JsonValue]],
) -> tuple[list[tuple[str, str]], list[Problem]]:
    """The one edge from each node into the next, and ``not_a_chain`` for each pair without exactly one."""
    edges: list[tuple[str, str]] = []
    problems: list[Problem] = []
    for index in range(1, len(nodes)):
        before_id, before, after_id, after = ids[index - 1], nodes[index - 1], ids[index], nodes[index]
        filled = typed_values[after_id].keys() | typed_defaults[after_id].keys()
        match before, after:
            case FunctionDefinition(), FunctionDefinition():
                outputs = list(before.outputs)
                open_inputs = [name for name, spec in after.inputs.items() if spec.default is REQUIRED and name not in filled]
            case _:  # a stored graph's fields are known only when it loads, so a chain can't be drawn through it
                outputs, open_inputs = [], []
        if len(outputs) != 1 or len(open_inputs) != 1:
            problems.append(problem.not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs))
        else:
            edges.append((f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"))
    return edges, problems
