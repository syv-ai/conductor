"""ForEachNode compound node implementation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from conductor.compound.protocol import CompoundNodeType, Region
from conductor.execution.events import (
    NodeCompleteEvent,
    NodeProgressEvent,
    NodeStartEvent,
)
from conductor.execution.results import extract_output, filter_skipped, normalize_result
from conductor.graph.regions import discover_for_each_regions

MAX_ITERATIONS = 1000


class ForEachNode:
    """Compound node for for-each loop iteration."""

    def __init__(self, region: Region, execution_order: tuple[str, ...]) -> None:
        self.region = region
        self.body_order = [nid for nid in execution_order if nid in region.body_ids]

    def execute(self, req: Any) -> Any:
        from conductor.execution.resolver import InputResolver

        items = _prepare_items(req.inputs.get("items", []))
        items = items[:MAX_ITERATIONS]
        parallel = req.inputs.get("execution_mode", "Sequential") == "Parallel"
        state = req.state

        state.emit(NodeStartEvent(type="node_start", node_id=self.region.end_id))

        def run_one(item: Any, idx: int) -> Any:
            # Create overlay: start node produces (item, index)
            overlay = {self.region.start_id: normalize_result((item, idx + 1))}
            local = _execute_subgraph(state, self.body_order, overlay)
            return _resolve_end_inputs(local, self.region.end_id, state)

        if not items:
            end_result = normalize_result([])
            state.results[self.region.end_id] = end_result
            state.emit(NodeCompleteEvent(
                type="node_complete", node_id=self.region.end_id,
                result=filter_skipped(end_result),
            ))
            return []

        if parallel:
            with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
                collected = list(pool.map(
                    lambda pair: run_one(pair[1], pair[0]),
                    enumerate(items),
                ))
            state.emit(NodeProgressEvent(
                type="node_progress", node_id=self.region.end_id,
                current=len(items), total=len(items),
            ))
        else:
            collected = []
            for idx, item in enumerate(items):
                if state.is_cancelled():
                    break
                collected.append(run_one(item, idx))
                state.emit(NodeProgressEvent(
                    type="node_progress", node_id=self.region.end_id,
                    current=idx + 1, total=len(items),
                ))

        end_result = normalize_result(collected)
        state.results[self.region.end_id] = end_result
        state.emit(NodeCompleteEvent(
            type="node_complete", node_id=self.region.end_id,
            result=filter_skipped(end_result),
        ))

        return items  # Start node's own result


def _prepare_items(raw: Any) -> list[Any]:
    """Normalize input items to a list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [line.strip() for line in raw.split("\n") if line.strip()]
    if isinstance(raw, dict):
        return list(raw.values())
    return [raw]


def _execute_subgraph(
    state: Any,
    node_ids: list[str],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Execute a subset of nodes with result isolation."""
    from conductor.execution.engine import _dispatch_node
    from conductor.execution.results import normalize_result
    from conductor.execution.skip import should_skip_node

    local_results = {**state.results, **overlay}
    compiled = state.compiled

    for node_id in node_ids:
        node = compiled.node_map[node_id]

        if should_skip_node(node, compiled.edge_map, local_results):
            from conductor._sentinel import SKIPPED
            local_results[node_id] = SKIPPED
            continue

        inputs = state.resolver.resolve(
            node, compiled.edge_map, local_results, compiled.node_map
        )

        result = _dispatch_node(
            node.type, node_id, inputs, node.data or {}, state, compiled
        )
        local_results[node_id] = normalize_result(result)

    return local_results


def _resolve_end_inputs(
    local_results: dict[str, Any],
    end_id: str,
    state: Any,
) -> Any:
    """Resolve the end node's inputs and return the collected value."""
    compiled = state.compiled
    end_node = compiled.node_map[end_id]

    for (target_id, target_handle), sources in compiled.edge_map.items():
        if target_id != end_id:
            continue
        for source_id, source_handle in sources:
            source_result = local_results.get(source_id)
            if source_result is not None:
                return extract_output(source_result, source_handle)

    return None


FOR_EACH = CompoundNodeType(
    start_type_prefix="for-each-start",
    end_type_prefix="for-each-end",
    discover=discover_for_each_regions,
    factory=lambda region, order: ForEachNode(region, order),
)
