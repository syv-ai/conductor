"""Skip propagation logic."""

from typing import Any

from conductor._sentinel import is_skipped
from conductor.execution.results import extract_output
from conductor.graph.model import GraphNode


def should_skip_node(
    node: GraphNode,
    edge_map: dict[tuple[str, str], list[tuple[str, str, str]]],
    results: dict[str, Any],
    consume_map: dict[tuple[str, str], tuple[str, str]] | None = None,
    skipped_edges: set[str] | None = None,
    incoming_map: dict[str, list[tuple[str, str, str, str]]] | None = None,
) -> bool:
    """Determine if a node should be skipped.

    A node is skipped if ALL of its incoming values (edges + consume
    bindings) are SKIPPED. Edges whose ``id`` appears in ``skipped_edges``
    also count as SKIPPED — this is how decision-node edge guards mark
    branches as "not taken". A node with no incoming sources is never
    skipped.

    ``incoming_map`` is an optional pre-built inverted view of the edges
    (see :func:`conductor.graph.topology.build_incoming_map`). When
    provided, lookup is O(1) per node instead of scanning the whole
    ``edge_map``. The old ``edge_map``-based path is kept for compat
    with callers that pass a ``None`` incoming_map.
    """
    incoming_sources: list[tuple[str, str, str]] = []
    if incoming_map is not None:
        for _target_handle, source_id, source_handle, edge_id in incoming_map.get(node.id, ()):
            incoming_sources.append((source_id, source_handle, edge_id))
    else:
        for (target_id, _handle), sources in edge_map.items():
            if target_id == node.id:
                incoming_sources.extend(sources)

    if consume_map:
        for (target_id, _handle), source in consume_map.items():
            if target_id == node.id:
                incoming_sources.append((source[0], source[1], ""))

    if not incoming_sources:
        return False

    skipped_edges = skipped_edges or set()

    for source_id, source_handle, edge_id in incoming_sources:
        if edge_id and edge_id in skipped_edges:
            continue
        source_result = results.get(source_id)
        if source_result is None:
            continue
        if is_skipped(source_result):
            continue
        value = extract_output(source_result, source_handle)
        if not is_skipped(value):
            return False

    return True
