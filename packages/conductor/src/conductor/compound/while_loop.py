"""While-loop compound node.

Mirrors :mod:`conductor.compound.for_each` but iterates on a CEL
condition instead of over a finite collection. Useful for
retry-with-backoff-until-success, poll-until-ready, paginate-until-empty,
and similar "bounded but not known up front" patterns.

Each iteration, the engine:

1. Evaluates the CEL ``condition`` against ``(iteration_count,
   last_body_result)``.
2. If true, runs the body subgraph.
3. Captures the value edge-connected to the ``while-end`` marker and uses
   it as ``last_body_result`` for the next round.
4. If false, stops.

A ``max_iterations`` cap prevents runaway loops — blowing past it raises
:class:`conductor.errors.LoopRunawayError`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from conductor.compound.protocol import CompoundNodeType, Region
from conductor.errors import CompilationError, LoopRunawayError, NodeExecutionError
from conductor.execution.events import (
    NodeCompleteEvent,
    NodeProgressEvent,
    NodeStartEvent,
)
from conductor.execution.results import (
    extract_output,
    filter_skipped,
    normalize_result,
)
from conductor.expr import ExpressionError
from conductor.expr import parse as parse_expr
from conductor.graph.model import GraphEdge, GraphNode

DEFAULT_MAX_ITERATIONS = 1000


class WhileNode:
    """Compound node for while-loop iteration."""

    def __init__(self, region: Region, execution_order: tuple[str, ...]) -> None:
        self.region = region
        self.body_order = [nid for nid in execution_order if nid in region.body_ids]

    def execute(self, req: Any) -> Any:
        condition_src = req.inputs.get("condition") or req.data.get("condition")
        if not condition_src:
            raise NodeExecutionError(
                f"While-start node '{self.region.start_id}' has no `condition` "
                f"— add a CEL expression to its `condition` input.",
                node_id=req.node_id,
                node_type=req.node_type,
            )
        try:
            expr = parse_expr(condition_src)
        except ExpressionError as e:
            raise NodeExecutionError(
                f"Invalid while-start condition {condition_src!r}: {e}",
                node_id=req.node_id,
                node_type=req.node_type,
                original=e,
            ) from e

        max_iter = int(req.inputs.get("max_iterations", DEFAULT_MAX_ITERATIONS))
        negate = bool(req.inputs.get("negate", False))  # `until` is `while not`

        state = req.state
        state.emit(NodeStartEvent(type="node_start", node_id=self.region.end_id))

        last_body_result: Any = None
        iteration = 0
        final: Any = None

        while iteration < max_iter:
            if state.is_cancelled():
                break

            ctx = {
                "iteration": iteration,
                "last": last_body_result,
                "result": last_body_result,
                "store": state.store.to_dict(),
                "$": {
                    "iteration": iteration,
                    "last": last_body_result,
                    "store": state.store.to_dict(),
                },
            }
            try:
                predicate = bool(expr.evaluate(ctx))
            except ExpressionError as e:
                raise NodeExecutionError(
                    f"While condition {condition_src!r} failed at iteration "
                    f"{iteration}: {e}",
                    node_id=req.node_id,
                    node_type=req.node_type,
                    original=e,
                ) from e

            if negate:
                predicate = not predicate

            if not predicate:
                break

            iteration += 1
            overlay = {
                self.region.start_id: normalize_result((iteration, last_body_result)),
            }
            local = _execute_subgraph(state, self.body_order, overlay)
            last_body_result = _resolve_end_inputs(local, self.region.end_id, state)
            final = last_body_result

            state.emit(NodeProgressEvent(
                type="node_progress",
                node_id=self.region.end_id,
                current=iteration,
                total=max_iter,
            ))
        else:
            # Loop condition stayed True for max_iter iterations — runaway.
            raise LoopRunawayError(
                self.region.start_id, iteration, max_iter,
            )

        end_result = normalize_result(final)
        state.results[self.region.end_id] = end_result
        state.emit(NodeCompleteEvent(
            type="node_complete",
            node_id=self.region.end_id,
            result=filter_skipped(end_result),
        ))

        # Start-node's own result: (iterations_run, final_value)
        return (iteration, final)


def _execute_subgraph(
    state: Any,
    node_ids: list[str],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Run the body subgraph for one iteration (sequentially)."""
    from conductor._sentinel import SKIPPED
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
            local_results[node_id] = SKIPPED
            continue

        inputs = state.resolver.resolve(
            node, compiled.edge_map, local_results, compiled.node_map,
            compiled.consume_map, state.skipped_edges, compiled.incoming_map,
        )

        result = _dispatch_node(
            node.type, node_id, inputs, node.data or {}, state, compiled,
        )
        local_results[node_id] = normalize_result(result)

    return local_results


def _resolve_end_inputs(
    local_results: dict[str, Any],
    end_id: str,
    state: Any,
) -> Any:
    """Extract the value the body connected to the while-end marker."""
    compiled = state.compiled

    for _target_handle, source_id, source_handle, _edge_id in compiled.incoming_map.get(end_id, ()):
        source_result = local_results.get(source_id)
        if source_result is not None:
            return extract_output(source_result, source_handle)

    return None


def discover_while_regions(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> list[Region]:
    """Discover while-loop regions by BFS from start to end nodes."""
    forward: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        forward[edge.source].append(edge.target)

    start_ids = [n.id for n in nodes if n.type.startswith("while-start")]
    end_ids = {n.id for n in nodes if n.type.startswith("while-end")}
    matched_ends: set[str] = set()

    regions: list[Region] = []
    for start_id in start_ids:
        visited: set[str] = set()
        queue = deque(forward.get(start_id, []))
        found_end: str | None = None
        body: set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in end_ids:
                found_end = current
                continue

            body.add(current)
            for neighbor in forward.get(current, []):
                queue.append(neighbor)

        if found_end is None:
            raise CompilationError(
                f"While-start node '{start_id}' has no matching end node."
            )

        matched_ends.add(found_end)
        regions.append(Region(
            start_id=start_id,
            end_id=found_end,
            body_ids=frozenset(body),
        ))

    orphan_ends = end_ids - matched_ends
    if orphan_ends:
        raise CompilationError(
            f"While-end node(s) {orphan_ends} have no matching start node."
        )

    return regions


WHILE = CompoundNodeType(
    start_type_prefix="while-start",
    end_type_prefix="while-end",
    discover=discover_while_regions,
    factory=lambda region, order: WhileNode(region, order),
)
