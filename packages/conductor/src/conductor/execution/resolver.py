"""Input resolution — resolve node inputs from edges and static data."""

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

    def resolve(
        self,
        node: GraphNode,
        edge_map: dict[tuple[str, str], list[tuple[str, str]]],
        results: dict[str, NodeResult],
        node_map: dict[str, GraphNode],
        consume_map: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Resolve all inputs for a node.

        Precedence (first match wins):
            1. Explicit edges targeting this input
            2. Shared-reference consume bindings (``consume_map``)
            3. Static data on the node
            4. Widget default (not materialized here; handled by Pydantic)
        """
        inputs: dict[str, Any] = dict(node.data or {})

        # (2) Consume bindings overlay static data before edges take over. A
        #     compile-time check guarantees no input handle has both an edge
        #     and a consume binding, so ordering relative to the edge loop is
        #     only about the static-data case.
        if consume_map:
            for (target_id, target_handle), (source_id, source_handle) in consume_map.items():
                if target_id != node.id:
                    continue
                source_result = results.get(source_id)
                if source_result is None:
                    continue
                if is_skipped(source_result):
                    inputs[target_handle] = source_result  # SKIPPED sentinel
                    continue
                value = extract_output(source_result, source_handle)
                inputs[target_handle] = value

        for (target_id, target_handle), sources in edge_map.items():
            if target_id != node.id:
                continue
            if not target_handle:
                continue

            uses_connection_list = self._param_uses_connection_list(
                node.type, target_handle
            )

            if uses_connection_list:
                # ConnectionList: build a labeled dict from all sources
                values, labels = self._collect_values_with_labels(
                    sources, results, node_map
                )
                if values:
                    unique_labels = _make_labels_unique(labels)
                    inputs[target_handle] = dict(
                        zip(unique_labels, values, strict=False)
                    )
            else:
                values = self._collect_values(sources, results)
                if not values:
                    continue

                expects_list = self._param_expects_list(node.type, target_handle)

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
        sources: list[tuple[str, str]],
        results: dict[str, NodeResult],
    ) -> list[Any]:
        values: list[Any] = []
        for source_id, source_handle in sources:
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
        sources: list[tuple[str, str]],
        results: dict[str, NodeResult],
        node_map: dict[str, GraphNode],
    ) -> tuple[list[Any], list[str]]:
        """Collect values and human-readable labels from sources."""
        values: list[Any] = []
        labels: list[str] = []
        for source_id, source_handle in sources:
            source_result = results.get(source_id)
            if source_result is None or is_skipped(source_result):
                continue
            value = extract_output(source_result, source_handle)
            if is_skipped(value):
                continue
            values.append(value)
            # Label: try to get the node's name from the registry, fall back to ID
            source_node = node_map.get(source_id)
            if source_node:
                node_def = self._registry.get(source_node.type)
                label = node_def.name if node_def else source_id
            else:
                label = source_id
            labels.append(f"{label} ({source_handle})")
        return values, labels

    def _param_expects_list(self, node_type: str, param_name: str) -> bool:
        node_def = self._registry.get(node_type)
        if node_def is None:
            return False
        for inp in node_def.inputs:
            if inp.name == param_name:
                return inp.expects_list
        return False

    def _param_uses_connection_list(self, node_type: str, param_name: str) -> bool:
        node_def = self._registry.get(node_type)
        if node_def is None:
            return False
        for inp in node_def.inputs:
            if inp.name == param_name:
                return inp.uses_connection_list
        return False


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
