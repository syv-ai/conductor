"""ForEachNode compound node implementation.

Supports unbounded parallel-zip iteration: any number of sources wired
into ``for-each-start.items`` are zipped element-wise into per-iteration
tuples, and any number of body→end edges are transposed into per-slot
``Collected`` lists. Both markers register with ``dynamic_handles=True``,
which lifts the strict-handle requirement so the start can emit
``output_3, output_4, output_5, …`` and the end can accept
``item_2, item_3, item_4, …`` without changing the registered schema.
"""

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

# Stable handle ordering. The start emits ``output_1`` (Item-1) +
# ``output_2`` (Index) first to keep backward compat with single-source
# flows; additional Item slots fan out at ``output_3``, ``output_4``, …
START_PRIMARY_ITEM = "output_1"
START_INDEX = "output_2"
END_PRIMARY_INPUT = "item"


class ForEachNode:
    """Compound node for for-each loop iteration."""

    def __init__(self, region: Region, execution_order: tuple[str, ...]) -> None:
        self.region = region
        self.body_order = [nid for nid in execution_order if nid in region.body_ids]

    def execute(self, req: Any) -> Any:
        # Parallel-zip aggregation. The ``items`` ConnectionList resolver
        # delivers a dict {label: source_value}; each source contributes
        # one position to the per-iteration tuple.
        raw = req.inputs.get("items", [])
        items = _prepare_items_zip(raw)
        items = items[:MAX_ITERATIONS]
        parallel = req.inputs.get("execution_mode", "Sequential") == "Parallel"
        state = req.state

        # Discover the dynamic shape of the end node from the compiled
        # incoming-edge map. ``end_input_handles`` preserves the order
        # in which the user wired body→end edges so the per-slot
        # ``Collected`` outputs come out in a deterministic order.
        end_input_handles = _discover_end_input_handles(
            state.compiled, self.region.end_id
        )

        state.emit(NodeStartEvent(type="node_start", node_id=self.region.end_id))

        def run_one(item: tuple, idx: int) -> tuple:
            # Start node's per-iteration overlay:
            #   output_1 = Item-1  (first wired source's element)
            #   output_2 = Index   (1-based)
            #   output_3 = Item-2  (second wired source's element)
            #   output_4 = Item-3
            #   …
            # Build the dict directly so we can place ``output_2`` (Index)
            # between Item-1 and the rest without juggling tuple positions.
            start_result: dict[str, Any] = {
                START_PRIMARY_ITEM: item[0] if len(item) > 0 else None,
                START_INDEX: idx + 1,
            }
            for slot_idx in range(1, len(item)):
                start_result[f"output_{slot_idx + 2}"] = item[slot_idx]

            overlay = {self.region.start_id: start_result}
            local = _execute_subgraph(state, self.body_order, overlay)
            return _resolve_end_inputs(local, end_input_handles, self.region.end_id, state)

        if not items:
            end_result = _build_end_result([], end_input_handles)
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

        end_result = _build_end_result(collected, end_input_handles)
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


def _discover_end_input_handles(compiled: Any, end_id: str) -> tuple[str, ...]:
    """Extract the wired end-input handle names in stable order.

    The primary ``item`` handle always sits at position 0 even when
    unwired — it's the schema-declared entrypoint. Additional wired
    handles (``item_2, item_3, …`` or any other names the host emits)
    follow in the order they were registered in the compiled graph.
    """
    seen: set[str] = set()
    ordered: list[str] = [END_PRIMARY_INPUT]
    seen.add(END_PRIMARY_INPUT)

    for target_handle, _source_id, _source_handle, _edge_id in compiled.incoming_map.get(end_id, ()):
        if target_handle in seen:
            continue
        seen.add(target_handle)
        ordered.append(target_handle)

    return tuple(ordered)


def _resolve_end_inputs(
    local_results: dict[str, Any],
    end_input_handles: tuple[str, ...],
    end_id: str,
    state: Any,
) -> tuple[Any, ...]:
    """Resolve every wired end-input into a per-slot tuple, in stable order."""
    compiled = state.compiled
    per_slot: dict[str, Any] = {}

    for target_handle, source_id, source_handle, _edge_id in compiled.incoming_map.get(end_id, ()):
        source_result = local_results.get(source_id)
        if source_result is None:
            continue
        per_slot[target_handle] = extract_output(source_result, source_handle)

    return tuple(per_slot.get(h) for h in end_input_handles)


def _build_end_result(
    collected: list[tuple],
    end_input_handles: tuple[str, ...],
) -> dict[str, Any]:
    """Transpose per-iteration tuples into per-slot output lists.

    Maps slot-i (in ``end_input_handles`` order) to ``output_{i+1}`` so
    ``output_1`` is the legacy ``Collected``, ``output_2`` is the
    second-wired-input's collected list, etc. Also writes a
    ``result`` alias pointing at slot 0 so callers reading
    ``results[end_id]["result"]`` keep working.
    """
    n = max(1, len(end_input_handles))
    per_slot: list[list[Any]] = [[] for _ in range(n)]
    for tup in collected:
        for idx in range(n):
            per_slot[idx].append(tup[idx] if idx < len(tup) else None)

    result: dict[str, Any] = {}
    for idx, slot_values in enumerate(per_slot):
        result[f"output_{idx + 1}"] = slot_values
    result["result"] = per_slot[0]
    return result


FOR_EACH = CompoundNodeType(
    start_type_prefix="for-each-start",
    end_type_prefix="for-each-end",
    discover=discover_for_each_regions,
    factory=lambda region, order: ForEachNode(region, order),
)
