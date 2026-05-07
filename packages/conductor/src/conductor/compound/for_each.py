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

        # Parallel-zip aggregation: when ``items`` has N sources wired
        # (a dict from the ConnectionList resolver), each iteration
        # yields a tuple ``(elem_0, elem_1, …, elem_{N-1})``. Single
        # source falls back to the legacy 1-tuple shape so per-iteration
        # work that wired ``output_1`` keeps working unchanged.
        raw = req.inputs.get("items", [])
        items = _prepare_items_zip(raw)
        items = items[:MAX_ITERATIONS]
        parallel = req.inputs.get("execution_mode", "Sequential") == "Parallel"
        state = req.state

        state.emit(NodeStartEvent(type="node_start", node_id=self.region.end_id))

        def run_one(item: tuple, idx: int) -> Any:
            # Pad the per-iteration tuple to the start node's full
            # output schema: (output_1=Item, output_2=Index,
            # output_3=Item-2, output_4=Item-3, output_5=Item-4).
            # Sources beyond 4 are dropped — the schema caps Item slots
            # at 4 today; bump ``loop.py`` if you need more.
            padded = list(item) + [None] * (4 - len(item))
            overlay_value = (padded[0], idx + 1, padded[1], padded[2], padded[3])
            overlay = {self.region.start_id: normalize_result(overlay_value)}
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


def _prepare_items_zip(raw: Any) -> list[tuple]:
    """Normalize ``items`` for parallel-zip iteration.

    Returns a list of tuples — each tuple is one iteration's per-source
    elements. Iteration count = ``min(len(s) for s in sources)`` when
    multi-source; longer sources are truncated so every iteration has a
    value at every position. Single-source falls back to 1-tuples.

    Shapes accepted:

    * ``dict`` (the ConnectionList resolver's output): keys are source
      labels, values are the source-side values. >1 key triggers
      parallel-zip; 1 key collapses to single-source.
    * ``list``: single-source fast path — wrap each element in a
      1-tuple.
    * ``str``: split by newline (legacy convenience), 1-tuple per line.
    * scalar: single 1-tuple containing the scalar.
    """
    if isinstance(raw, dict):
        if len(raw) == 0:
            return []
        sources = list(raw.values())
        if len(sources) == 1:
            return [(item,) for item in _prepare_items(sources[0])]
        # All sources should be iterable; coerce non-list scalars to
        # single-element lists so parallel-zip still has something at
        # position 0 (later iterations skip that source).
        coerced: list[list[Any]] = []
        for src in sources:
            if isinstance(src, list):
                coerced.append(src)
            elif isinstance(src, str):
                coerced.append(
                    [line.strip() for line in src.split("\n") if line.strip()]
                )
            else:
                coerced.append([src])
        min_len = min(len(s) for s in coerced)
        return [tuple(s[i] for s in coerced) for i in range(min_len)]
    return [(item,) for item in _prepare_items(raw)]


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

        if should_skip_node(
            node, compiled.edge_map, local_results, compiled.consume_map,
            state.skipped_edges, compiled.incoming_map,
        ):
            from conductor._sentinel import SKIPPED
            local_results[node_id] = SKIPPED
            continue

        inputs = state.resolver.resolve(
            node, compiled.edge_map, local_results, compiled.node_map,
            compiled.consume_map, state.skipped_edges, compiled.incoming_map,
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

    for _target_handle, source_id, source_handle, _edge_id in compiled.incoming_map.get(end_id, ()):
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
