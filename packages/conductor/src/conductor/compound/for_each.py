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

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from conductor.compound.protocol import CompoundNodeType, Region
from conductor.execution.events import (
    NodeCompleteEvent,
    NodeProgressEvent,
    NodeSkippedEvent,
    NodeStartEvent,
    RuntimeWarningEvent,
)
from conductor.execution.results import extract_output, filter_skipped, normalize_result
from conductor.graph.regions import discover_for_each_regions

if TYPE_CHECKING:
    from conductor.metadata import OutputMetadata
    from conductor.registry.dynamic_outputs import ComputeOutputsContext

MAX_ITERATIONS = 1000

# Global cap on body-node executions in flight at once across an entire
# loop run. Independent body nodes within an iteration run concurrently
# (DAG-aware, like the top-level graph), but a single semaphore shared by
# every iteration bounds the total — so a Parallel loop (8 items) over a
# multi-node body can't multiply into a connection-storm of concurrent
# LLM calls. Matches the item-level worker cap.
_BODY_CONCURRENCY = 8

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
# Primary input target handle on the for-each-end. Edges wired into the
# end node should target this handle; each (source_id, source_handle)
# pair becomes one collected-output slot, in edge order. The handle is
# a ConnectionList so multiple sources can stack onto it cleanly.
END_PRIMARY_INPUT = "items"
# Legacy target handles still accepted for backward compatibility — pre-
# ConnectionList for-each-end schemas exposed one handle per wired
# source (``item``, ``item_2``, ``item_3``, …). Edges that target any
# of these (or any handle starting with ``item``) are remapped onto the
# new model so old saved flows keep loading.
_LEGACY_END_HANDLE_PREFIX = "item"


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
        # incoming-edge map. ``end_slots`` preserves the order
        # in which the user wired body→end edges so the per-slot
        # ``Collected`` outputs come out in a deterministic order.
        end_slots = _discover_end_slots(
            state.compiled, self.region.end_id
        )
        end_output_names = _discover_end_output_names(
            state.compiled, self.region.end_id, end_slots,
        )

        state.emit(NodeStartEvent(type="node_start", node_id=self.region.end_id))

        # One semaphore shared by every iteration caps total body-node
        # concurrency across the whole loop (see _BODY_CONCURRENCY).
        body_semaphore = threading.BoundedSemaphore(_BODY_CONCURRENCY)

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
            local = _execute_subgraph(
                state, self.body_order, overlay, body_semaphore
            )
            return _resolve_end_inputs(local, end_slots, self.region.end_id, state)

        if not items:
            end_result = _build_end_result([], end_slots, end_output_names)
            state.results[self.region.end_id] = end_result
            state.emit(NodeCompleteEvent(
                type="node_complete", node_id=self.region.end_id,
                result=filter_skipped(end_result),
            ))
            return []

        if parallel:
            # Submit + as_completed (instead of pool.map) so a progress
            # event fires each time an item finishes — a live counter in
            # parallel mode too. Results are slotted back by index so the
            # collected order still matches item order regardless of which
            # iteration finishes first.
            collected: list[Any] = [None] * len(items)
            total = len(items)
            with ThreadPoolExecutor(max_workers=min(total, 8)) as pool:
                futures = {
                    pool.submit(run_one, item, idx): idx
                    for idx, item in enumerate(items)
                }
                done = 0
                for fut in as_completed(futures):
                    collected[futures[fut]] = fut.result()
                    done += 1
                    state.emit(NodeProgressEvent(
                        type="node_progress", node_id=self.region.end_id,
                        current=done, total=total,
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

        end_result = _build_end_result(collected, end_slots, end_output_names)
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


def _body_levels(compiled: Any, node_ids: list[str]) -> list[list[str]]:
    """Group body nodes into dependency levels (topological layers).

    Level 0 holds body nodes with no body-internal dependency; level *k*
    holds nodes whose body deps are all in earlier levels. Nodes in the
    same level are mutually independent and safe to run concurrently;
    levels run in order so a node never starts before its dependencies
    have produced results. Only body->body edges constrain ordering —
    deps pointing outside the body (e.g. the for-each-start overlay) are
    already present in ``local_results``. Order within a level follows
    ``node_ids`` (topological) for deterministic event ordering.
    """
    body_set = set(node_ids)
    deps: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for nid in node_ids:
        for _th, src, _sh, _eid in compiled.incoming_map.get(nid, []):
            if src in body_set and src != nid:
                deps[nid].add(src)
    # Shared-reference (consume) edges are dependencies too.
    for (tid, _th), (src, _sh) in compiled.consume_map.items():
        if tid in body_set and src in body_set and src != tid:
            deps[tid].add(src)

    order_index = {nid: i for i, nid in enumerate(node_ids)}
    done: set[str] = set()
    remaining = list(node_ids)
    levels: list[list[str]] = []
    while remaining:
        ready = [nid for nid in remaining if deps[nid] <= done]
        if not ready:
            # Defensive: a cycle shouldn't survive compilation, but never
            # hang — drain the rest sequentially in topological order.
            levels.extend(
                [nid] for nid in sorted(remaining, key=lambda n: order_index[n])
            )
            break
        ready.sort(key=lambda nid: order_index[nid])
        levels.append(ready)
        done.update(ready)
        remaining = [nid for nid in remaining if nid not in done]
    return levels


def _run_body_node(
    node_id: str,
    state: Any,
    compiled: Any,
    local_results: dict[str, Any],
    lock: threading.Lock,
    semaphore: threading.Semaphore | None,
) -> None:
    """Resolve, skip-check, and dispatch a single body node.

    Safe to call from multiple threads for distinct nodes in the same
    level: reads take a snapshot of ``local_results`` under ``lock`` (so a
    sibling's write can't mutate the dict mid-resolve), the actual
    dispatch is gated by the shared ``semaphore``, and the result is
    written back under the lock.
    """
    from conductor._sentinel import SKIPPED
    from conductor.execution.engine import _dispatch_node
    from conductor.execution.results import filter_skipped
    from conductor.execution.skip import should_skip_node

    node = compiled.node_map[node_id]

    with lock:
        snapshot = dict(local_results)

    if should_skip_node(
        node, compiled.edge_map, snapshot, compiled.consume_map,
        state.skipped_edges, compiled.incoming_map,
    ):
        with lock:
            local_results[node_id] = SKIPPED
        state.emit(NodeSkippedEvent(type="node_skipped", node_id=node_id))
        return

    inputs = state.resolver.resolve(
        node, compiled.edge_map, snapshot, compiled.node_map,
        compiled.consume_map, state.skipped_edges, compiled.incoming_map,
    )

    state.emit(NodeStartEvent(type="node_start", node_id=node_id))
    if semaphore is not None:
        with semaphore:
            result = _dispatch_node(
                node.type, node_id, inputs, node.data or {}, state, compiled
            )
    else:
        result = _dispatch_node(
            node.type, node_id, inputs, node.data or {}, state, compiled
        )
    normalized = normalize_result(result)
    with lock:
        local_results[node_id] = normalized
    state.emit(NodeCompleteEvent(
        type="node_complete", node_id=node_id,
        result=filter_skipped(normalized) if isinstance(normalized, dict) else normalized,
    ))


def _execute_subgraph(
    state: Any,
    node_ids: list[str],
    overlay: dict[str, Any],
    semaphore: threading.Semaphore | None = None,
) -> dict[str, Any]:
    """Execute a loop body with result isolation and DAG-aware concurrency.

    Body nodes are grouped into dependency levels (see ``_body_levels``):
    mutually-independent nodes in a level run concurrently — bounded by
    ``semaphore``, a cap shared across the whole loop so parallel
    iterations don't multiply into a connection-storm — while dependent
    nodes still run in order. A linear body collapses to single-node
    levels and runs inline, identical to the old sequential path.

    Emits per-body-node ``node_start`` / ``node_complete`` /
    ``node_skipped`` events so hosts can paint body-node status the same
    way they do for top-level nodes.
    """
    local_results = {**state.results, **overlay}
    compiled = state.compiled
    lock = threading.Lock()

    for level in _body_levels(compiled, node_ids):
        if len(level) == 1:
            _run_body_node(
                level[0], state, compiled, local_results, lock, semaphore
            )
            continue
        with ThreadPoolExecutor(max_workers=len(level)) as pool:
            # list() drains the iterator so the first worker exception is
            # re-raised here — a failing body node fails the iteration,
            # matching the sequential path.
            list(pool.map(
                lambda nid: _run_body_node(
                    nid, state, compiled, local_results, lock, semaphore
                ),
                level,
            ))

    return local_results


EndSlotKey = tuple[str, str]  # (source_id, source_handle)


def _is_end_input_edge(target_handle: str) -> bool:
    """True for edges that should land on the for-each-end's collection.

    Accepts the new ``items`` ConnectionList target as well as the
    legacy per-source handles (``item``, ``item_2``, …) so old saved
    flows compile without a host-side migration step.
    """
    if target_handle == END_PRIMARY_INPUT:
        return True
    return target_handle.startswith(_LEGACY_END_HANDLE_PREFIX)


def compute_for_each_end_outputs(
    ctx: "ComputeOutputsContext",
) -> "list[OutputMetadata]":
    """Default ``compute_outputs`` hook for ``for-each-end``.

    Types each collected slot as ``list[<inner>]`` where ``<inner>`` is the
    wired source's element type — one ``list[...]`` level unwrapped when the
    source already produces a list (the loop body runs once per item, so a
    body output of type ``T`` collects into ``list[T]``). Without this hook
    the end's collected outputs are untyped, which silently weakens
    compile-time type-checking of whatever consumes the loop result.

    Slot names (``output_1``, ``output_2``, …), ordering, and the
    dedup-by-``(source_node, source_handle)`` rule mirror
    :func:`_discover_end_slots` exactly — and the edge filter reuses
    :func:`_is_end_input_edge` — so the compile-time schema lines up slot
    for slot with the runtime's per-slot transposition (including legacy
    ``item``/``item_N`` target handles on flows saved before the
    ConnectionList switch). Labels come from the source output with any
    sub-output prefix stripped so hosts show clean names.

    Returns ``ctx.defaults`` unchanged when nothing is wired into the end.
    """
    from conductor.metadata import OutputMetadata
    from conductor.registry.dynamic_outputs import strip_sub_output_prefix

    seen: set[EndSlotKey] = set()
    bindings = []
    for binding in ctx.incoming:
        if not _is_end_input_edge(binding.target_handle):
            continue
        key: EndSlotKey = (binding.source_node_id, binding.source_handle)
        if key in seen:
            continue
        seen.add(key)
        bindings.append(binding)

    if not bindings:
        return list(ctx.defaults)

    outputs: list[OutputMetadata] = []
    for idx, binding in enumerate(bindings):
        source_type = binding.source_output.type_str or "any"
        if source_type.startswith("list[") and source_type.endswith("]"):
            inner = source_type[5:-1]
        else:
            inner = source_type
        source_label = binding.source_output.label or binding.source_handle
        outputs.append(
            OutputMetadata(
                name=f"output_{idx + 1}",
                type_str=f"list[{inner}]",
                label=strip_sub_output_prefix(source_label),
                description=binding.source_output.description,
                download=binding.source_output.download,
                filename=binding.source_output.filename,
            )
        )
    return outputs


def _discover_end_slots(compiled: Any, end_id: str) -> tuple[EndSlotKey, ...]:
    """Extract one slot per wired ``(source_id, source_handle)``.

    Slot order matches the order edges appear in ``incoming_map`` — i.e.
    the order the host saved them. Each wired source contributes one
    collected-output position regardless of which target handle it used
    (``items`` going forward, legacy ``item``/``item_N`` for old flows).
    Sources targeting a non-end-collection handle (e.g. a future
    secondary control input) are ignored here.
    """
    seen: set[EndSlotKey] = set()
    ordered: list[EndSlotKey] = []
    for target_handle, source_id, source_handle, _edge_id in compiled.incoming_map.get(end_id, ()):
        if not _is_end_input_edge(target_handle):
            continue
        key: EndSlotKey = (source_id, source_handle)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def _resolve_end_inputs(
    local_results: dict[str, Any],
    end_slots: tuple[EndSlotKey, ...],
    end_id: str,
    state: Any,
) -> tuple[Any, ...]:
    """Resolve every wired end-input into a per-slot tuple, in stable order.

    Slot identity is the upstream ``(source_id, source_handle)`` pair —
    so a body node with two outputs both wired into the end produces
    two collected lists, and two different body nodes wired through
    the same ``items`` handle also produce two collected lists.
    """
    per_slot: dict[EndSlotKey, Any] = {}
    for (source_id, source_handle) in end_slots:
        source_result = local_results.get(source_id)
        if source_result is None:
            continue
        per_slot[(source_id, source_handle)] = extract_output(
            source_result, source_handle,
        )
    return tuple(per_slot.get(slot) for slot in end_slots)


def _discover_end_output_names(
    compiled: Any,
    end_id: str,
    end_slots: tuple[EndSlotKey, ...],
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
    n = max(1, len(end_slots))
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
    end_slots: tuple[EndSlotKey, ...],
    end_output_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Transpose per-iteration tuples into per-slot output lists.

    Maps slot-i (in ``end_slots`` order) to the resolved output
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

    n = max(1, len(end_slots))
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
