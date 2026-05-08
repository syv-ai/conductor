"""ForEachNode compound node implementation.

Supports unbounded parallel-zip iteration: any number of sources wired
into ``for-each-start.items`` are zipped element-wise into per-iteration
tuples, and any number of body→end edges are transposed into per-slot
``Collected`` lists. Both markers register with ``dynamic_handles=True``,
which lifts the strict-handle requirement so the start can emit
``output_3, output_4, output_5, …`` and the end can accept
``item_2, item_3, item_4, …`` without changing the registered schema.

Multi-source truncation
-----------------------

When more than one list is wired into ``items``, iteration count is the
shortest source's length. If any source is longer than ``min_len``, the
compound emits a :class:`~conductor.execution.events.RuntimeWarningEvent`
with ``warning="for_each_zip_truncation"`` before iteration begins. The
``payload`` carries the per-source lengths plus ``min_len`` so callers
can flag data-shape bugs without sifting through node logs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from conductor.compound.protocol import CompoundNodeType, Region
from conductor.execution.events import (
    NodeCompleteEvent,
    NodeProgressEvent,
    NodeStartEvent,
    RuntimeWarningEvent,
)
from conductor.execution.results import extract_output, filter_skipped, normalize_result
from conductor.graph.regions import discover_for_each_regions

MAX_ITERATIONS = 1000

# Hard cap on nested-for-each depth. Mirrors
# ``compound/subprocess.py:_MAX_SUBPROCESS_DEPTH`` so a pathological
# flow with self-referential or deeply-nested loops can't run away with
# the engine. Depth is tracked on the per-flow ``state.context`` so
# concurrent flows don't share the counter.
_MAX_FOR_EACH_DEPTH = 16
_FOR_EACH_DEPTH_KEY = "_for_each_depth"

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
        state = req.state

        # Nested-for-each depth cap. Mirrors the subprocess depth check
        # (compound/subprocess.py) so a pathological flow with self-
        # referential or deeply-nested for-each regions can't run away
        # with the engine. The counter lives on ``state.context`` so
        # concurrent flows don't share it; a try/finally below restores
        # the previous depth even when iteration raises.
        prev_depth = state.context.get(_FOR_EACH_DEPTH_KEY, 0)
        depth = prev_depth + 1
        if depth > _MAX_FOR_EACH_DEPTH:
            from conductor.errors import NodeExecutionError
            raise NodeExecutionError(
                f"For-each recursion depth exceeded "
                f"({depth}>{_MAX_FOR_EACH_DEPTH}) at node "
                f"'{self.region.start_id}'",
                node_id=self.region.start_id,
                node_type="for-each-start",
            )
        state.context[_FOR_EACH_DEPTH_KEY] = depth
        try:
            return self._execute_inner(req)
        finally:
            if prev_depth == 0:
                state.context.pop(_FOR_EACH_DEPTH_KEY, None)
            else:
                state.context[_FOR_EACH_DEPTH_KEY] = prev_depth

    def _execute_inner(self, req: Any) -> Any:
        # Parallel-zip aggregation. The ``items`` ConnectionList resolver
        # delivers a dict {label: source_value}; each source contributes
        # one position to the per-iteration tuple.
        raw = req.inputs.get("items", [])
        items, truncation = _prepare_items_zip(raw)
        items = items[:MAX_ITERATIONS]
        parallel = req.inputs.get("execution_mode", "Sequential") == "Parallel"
        state = req.state

        # Surface multi-source-zip truncation as a non-fatal runtime
        # warning so hosts can flag data-shape bugs (e.g. one source is
        # secretly empty) without scraping node logs.
        if truncation is not None:
            source_lengths, min_len = truncation
            state.emit(RuntimeWarningEvent(
                type="runtime_warning",
                node_id=self.region.start_id,
                warning="for_each_zip_truncation",
                message=(
                    f"for-each '{self.region.start_id}' truncated to "
                    f"min_len={min_len} from sources "
                    f"{source_lengths}"
                ),
                payload={
                    "source_lengths": source_lengths,
                    "min_len": min_len,
                },
            ))

        # Discover the dynamic shape of the end node from the compiled
        # incoming-edge map. ``end_input_handles`` preserves the order
        # in which the user wired body→end edges so the per-slot
        # ``Collected`` outputs come out in a deterministic order.
        end_input_handles = _discover_end_input_handles(
            state.compiled, self.region.end_id
        )
        end_output_names = _discover_end_output_names(
            state.compiled, self.region.end_id, end_input_handles,
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
            end_result = _build_end_result([], end_input_handles, end_output_names)
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

        end_result = _build_end_result(collected, end_input_handles, end_output_names)
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


def _prepare_items_zip(
    raw: Any,
) -> tuple[list[tuple], tuple[dict[str, int], int] | None]:
    """Normalize ``items`` for parallel-zip iteration.

    Returns ``(items, truncation)``. ``items`` is a list of tuples — each
    tuple is one iteration's per-source elements. Iteration count =
    ``min(len(s) for s in sources)`` when multi-source; longer sources
    are truncated so every iteration has a value at every position.
    Single-source falls back to 1-tuples.

    ``truncation`` is ``None`` if no truncation occurred (single-source
    or all sources of equal length). Otherwise it is a tuple
    ``(source_lengths, min_len)`` where ``source_lengths`` maps the
    per-source label to its original length. The caller uses this to
    emit a runtime-warning event before iteration starts.

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
            return [], None
        labels = list(raw.keys())
        sources = list(raw.values())
        if len(sources) == 1:
            return [(item,) for item in _prepare_items(sources[0])], None
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
        lengths = [len(s) for s in coerced]
        min_len = min(lengths)
        truncation: tuple[dict[str, int], int] | None = None
        if any(length > min_len for length in lengths):
            truncation = (
                {label: length for label, length in zip(labels, lengths, strict=False)},
                min_len,
            )
        return (
            [tuple(s[i] for s in coerced) for i in range(min_len)],
            truncation,
        )
    return [(item,) for item in _prepare_items(raw)], None


def _execute_subgraph(
    state: Any,
    node_ids: list[str],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Execute a subset of nodes with result isolation."""
    from conductor.execution.engine import _dispatch_node
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


def _discover_end_output_names(
    compiled: Any,
    end_id: str,
    end_input_handles: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the per-slot output handle names emitted by the for-each-end.

    Compile-time ``compute_outputs`` may rename slots (``"items"``,
    ``"summaries"``, …) instead of the default ``output_{idx+1}``. When
    ``state.compiled.node_outputs`` carries an entry for the end node we
    use those names — skipping the synthetic ``"result"`` alias if
    present — so iteration matches the schema the palette advertised.

    Falls back to ``output_{idx+1}`` for every slot when the node has no
    resolved entry (the legacy default). Resolved-but-shorter sequences
    are padded with the legacy default for the missing tail.
    """
    n = max(1, len(end_input_handles))
    legacy = tuple(f"output_{idx + 1}" for idx in range(n))

    resolved_map = getattr(compiled, "node_outputs", None) or {}
    resolved = resolved_map.get(end_id)
    if not resolved:
        return legacy

    # Drop the synthetic ``"result"`` alias if a hook included it; the
    # alias is always written separately by ``_build_end_result``.
    resolved_names = tuple(o.name for o in resolved if o.name != "result")
    if not resolved_names:
        return legacy
    if len(resolved_names) >= n:
        return resolved_names[:n]
    return resolved_names + legacy[len(resolved_names):]


def _build_end_result(
    collected: list[tuple],
    end_input_handles: tuple[str, ...],
    end_output_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Transpose per-iteration tuples into per-slot output lists.

    Maps slot-i (in ``end_input_handles`` order) to the resolved output
    name from ``end_output_names`` if provided, otherwise to the legacy
    ``output_{idx+1}``. Always writes a ``result`` alias pointing at slot
    0 so callers reading ``results[end_id]["result"]`` keep working.

    Iterations whose body terminated in a skipped branch (e.g. an
    if-else inside the loop) contribute ``SKIPPED`` for the slot — those
    are filtered out so collected lists match the behaviour of multi-edge
    aggregation in :mod:`conductor.execution.resolver` (which already
    drops SKIPPED). This keeps the shape of the collected list aligned
    with what callers expect when conditional logic lives inside a
    for-each body.
    """
    from conductor._sentinel import is_skipped

    n = max(1, len(end_input_handles))
    per_slot: list[list[Any]] = [[] for _ in range(n)]
    for tup in collected:
        for idx in range(n):
            value = tup[idx] if idx < len(tup) else None
            if is_skipped(value):
                continue
            per_slot[idx].append(value)

    if end_output_names is None or len(end_output_names) < n:
        names = tuple(f"output_{idx + 1}" for idx in range(n))
    else:
        names = end_output_names[:n]

    result: dict[str, Any] = {}
    for idx, slot_values in enumerate(per_slot):
        result[names[idx]] = slot_values
    result["result"] = per_slot[0]
    return result


FOR_EACH = CompoundNodeType(
    start_type_prefix="for-each-start",
    end_type_prefix="for-each-end",
    discover=discover_for_each_regions,
    factory=lambda region, order: ForEachNode(region, order),
)
