"""Input resolution — resolve node inputs from edges and static data."""

from collections import defaultdict
from typing import Any

from conductor._sentinel import is_skipped
from conductor.execution.results import extract_output
from conductor.graph.model import GraphNode
from conductor.registry import NodeRegistry
from conductor.types import NodeResult

MULTI_VALUE_SEPARATOR = "\n\n"


class InputResolver:
    """Resolves all inputs for a node from edges and static data."""

    def __init__(self, registry: NodeRegistry) -> None:
        self._registry = registry
        # Per-node-type cache of param metadata — avoids a linear scan on
        # every resolve.
        self._param_cache: dict[str, dict[str, tuple[bool, bool]]] = {}

    def resolve(
        self,
        node: GraphNode,
        edge_map: dict[tuple[str, str], list[tuple[str, str, str]]],
        results: dict[str, NodeResult],
        node_map: dict[str, GraphNode],
        consume_map: dict[tuple[str, str], tuple[str, str]] | None = None,
        skipped_edges: set[str] | None = None,
        incoming_map: dict[str, list[tuple[str, str, str, str]]] | None = None,
    ) -> dict[str, Any]:
        """Resolve all inputs for a node.

        Precedence (first match wins):
            1. Explicit edges targeting this input (edges in ``skipped_edges``
               are treated as absent)
            2. Shared-reference consume bindings (``consume_map``)
            3. Static data on the node
            4. Widget default (not materialized here; handled by Pydantic)
        """
        skipped_edges = skipped_edges or set()
        inputs: dict[str, Any] = dict(node.data or {})

        # (2) Consume bindings overlay static data before edges take over.
        if consume_map:
            for (target_id, target_handle), (source_id, source_handle) in consume_map.items():
                if target_id != node.id:
                    continue
                source_result = results.get(source_id)
                if source_result is None:
                    continue
                if is_skipped(source_result):
                    inputs[target_handle] = source_result
                    continue
                value = extract_output(source_result, source_handle)
                inputs[target_handle] = value

        # (1) Edge-based resolution. Gather all incoming (source, handle, edge_id)
        # per target_handle in one pass.
        by_handle: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        if incoming_map is not None:
            for target_handle, source_id, source_handle, edge_id in incoming_map.get(node.id, ()):
                if not target_handle:
                    continue
                if edge_id and edge_id in skipped_edges:
                    continue
                by_handle[target_handle].append((source_id, source_handle, edge_id))
        else:
            for (target_id, target_handle), sources in edge_map.items():
                if target_id != node.id or not target_handle:
                    continue
                for sid, shandle, eid in sources:
                    if eid and eid in skipped_edges:
                        continue
                    by_handle[target_handle].append((sid, shandle, eid))

        for target_handle, live_sources in by_handle.items():
            if not live_sources:
                continue
            uses_cl, expects_list = self._param_info(node.type, target_handle)
            if uses_cl:
                values, labels = self._collect_values_with_labels(
                    live_sources, results, node_map,
                )
                if values:
                    unique_labels = _make_labels_unique(labels)
                    inputs[target_handle] = dict(
                        zip(unique_labels, values, strict=False),
                    )
            else:
                values = self._collect_values(live_sources, results)
                if not values:
                    continue
                if len(values) == 1:
                    inputs[target_handle] = values[0]
                elif expects_list:
                    inputs[target_handle] = values
                else:
                    inputs[target_handle] = MULTI_VALUE_SEPARATOR.join(
                        str(v) for v in values
                    )

        return inputs

    def _collect_values(
        self,
        sources: list[tuple[str, str, str]],
        results: dict[str, NodeResult],
    ) -> list[Any]:
        values: list[Any] = []
        for source_id, source_handle, _edge_id in sources:
            source_result = results.get(source_id)
            if source_result is None or is_skipped(source_result):
                continue
            value = extract_output(source_result, source_handle)
            if is_skipped(value):
                continue
            values.append(value)
        return values

    def _collect_values_with_labels(
        self,
        sources: list[tuple[str, str, str]],
        results: dict[str, NodeResult],
        node_map: dict[str, GraphNode],
    ) -> tuple[list[Any], list[str]]:
        """Collect values and human-readable labels from sources."""
        values: list[Any] = []
        labels: list[str] = []
        for source_id, source_handle, _edge_id in sources:
            source_result = results.get(source_id)
            if source_result is None or is_skipped(source_result):
                continue
            value = extract_output(source_result, source_handle)
            if is_skipped(value):
                continue
            values.append(value)
            source_node = node_map.get(source_id)
            if source_node:
                node_def = self._registry.get(source_node.type)
                label = node_def.name if node_def else source_id
            else:
                label = source_id
            labels.append(f"{label} ({source_handle})")
        return values, labels

    def _param_info(self, node_type: str, param_name: str) -> tuple[bool, bool]:
        """Return ``(uses_connection_list, expects_list)`` for ``param_name``."""
        cache = self._param_cache.get(node_type)
        if cache is None:
            node_def = self._registry.get(node_type)
            cache = {}
            if node_def is not None:
                for inp in node_def.inputs:
                    cache[inp.name] = (inp.uses_connection_list, inp.expects_list)
            self._param_cache[node_type] = cache
        return cache.get(param_name, (False, False))


def _make_labels_unique(labels: list[str]) -> list[str]:
    """Deduplicate labels by appending _2, _3, etc."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            result.append(f"{label}_{seen[label]}")
        else:
            seen[label] = 1
            result.append(label)
    return result
