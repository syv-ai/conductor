"""Skip propagation logic."""

from typing import Any

from conductor._sentinel import SKIPPED, is_skipped
from conductor.execution.results import extract_output
from conductor.graph.model import GraphNode


def should_skip_node(
    node: GraphNode,
    edge_map: dict[tuple[str, str], list[tuple[str, str]]],
    results: dict[str, Any],
    consume_map: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> bool:
    """Determine if a node should be skipped.

    A node is skipped if ALL of its incoming values (edges + consume
    bindings) are SKIPPED. A node with no incoming sources is never skipped.
    """
    incoming_sources: list[tuple[str, str]] = []
    for (target_id, _handle), sources in edge_map.items():
        if target_id == node.id:
            incoming_sources.extend(sources)

    if consume_map:
        for (target_id, _handle), source in consume_map.items():
            if target_id == node.id:
                incoming_sources.append(source)

    if not incoming_sources:
        return False

    for source_id, source_handle in incoming_sources:
        source_result = results.get(source_id)
        if source_result is None:
            continue
        if is_skipped(source_result):
            continue
        value = extract_output(source_result, source_handle)
        if not is_skipped(value):
            return False

    return True
