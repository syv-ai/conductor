"""Execution engine — eager-scheduled, parallel, with retry support."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from conductor._sentinel import SKIPPED
from conductor.errors import (
    FlowExecutionError,
    FlowExecutionException,
    FlowPausedError,
    FlowPausedException,
    HumanInputRequired,
    NodeConnectionError,
    NodeExecutionError,
    NodeExecutionException,
    NodeValidationError,
    NodeValidationException,
)
from conductor.execution.checkpoint import FlowCheckpoint
from conductor.execution.events import (
    EventSink,
    ExecutionEvent,
    FlowCancelledEvent,
    FlowCompleteEvent,
    FlowErrorEvent,
    FlowPausedEvent,
    FlowTimeoutEvent,
    NodeCompleteEvent,
    NodeErrorEvent,
    NodeRetryEvent,
    NodeSkippedEvent,
    NodeStartEvent,
)
from conductor.execution.resolver import InputResolver
from conductor.execution.results import filter_all_skipped, filter_skipped, normalize_result
from conductor.execution.retry import NO_RETRY, RetryConfig
from conductor.execution.skip import should_skip_node
from conductor.execution.state import FlowRunState
from conductor.execution.store import FlowStore
from conductor.graph.compiler import CompiledGraph
from conductor.types import NodeResult


# Internal sentinel pushed into the event queue when all work is done
_DONE = object()
_FATAL = object()


# =========================================================================
# Primary entry point
# =========================================================================


async def execute(
    compiled: CompiledGraph,
    *,
    timeout_seconds: int = 300,
    context: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
    retry: RetryConfig | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Execute a compiled graph with eager parallel scheduling.

    Nodes start as soon as all their dependencies are done — independent
    branches run concurrently. Retry is configurable per-node or globally.
    """
    state = _build_state(compiled, timeout_seconds, context)
    async for event in _run_eager(state, cache=cache or {}, retry=retry or NO_RETRY):
        yield event


# =========================================================================
# Resume from checkpoint
# =========================================================================


async def resume(
    compiled: CompiledGraph,
    checkpoint: FlowCheckpoint | dict[str, Any],
    response: Any,
    *,
    timeout_seconds: int = 300,
    context: dict[str, Any] | None = None,
    retry: RetryConfig | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Resume a paused flow with a human's response."""
    if isinstance(checkpoint, dict):
        checkpoint = FlowCheckpoint.from_dict(checkpoint)

    state = _build_state(
        compiled, timeout_seconds, context or checkpoint.context,
    )
    state.results = dict(checkpoint.results)
    state.store = FlowStore(dict(checkpoint.store_data))

    # Inject human response as the waiting node's result
    state.results[checkpoint.waiting_node_id] = normalize_result(response)

    yield NodeCompleteEvent(
        type="node_complete",
        node_id=checkpoint.waiting_node_id,
        result=normalize_result(response),
    )

    async for event in _run_eager(state, cache={}, retry=retry or NO_RETRY):
        yield event


# =========================================================================
# Eager scheduler
# =========================================================================


async def _run_eager(
    state: FlowRunState,
    *,
    cache: dict[str, Any],
    retry: RetryConfig,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Eager-scheduled execution loop.

    Nodes are dispatched as soon as all dependencies complete.
    Independent branches run concurrently.
    """
    compiled = state.compiled
    event_queue: asyncio.Queue = asyncio.Queue()

    # Build dependency graph from edge_map
    deps, dependents = _build_dep_graph(compiled)

    # Compute schedulable nodes (exclude managed, already completed)
    all_nodes = set(compiled.execution_order)
    schedulable = all_nodes - compiled.managed_ids

    # Track in-degree (number of unfinished deps)
    in_degree: dict[str, int] = {}
    for node_id in schedulable:
        # Only count deps that are also schedulable
        node_deps = deps.get(node_id, set())
        dep_count = len(node_deps & schedulable)
        in_degree[node_id] = dep_count

    # Pre-satisfy nodes that are already in results (resume / cache)
    for node_id in list(state.results.keys()):
        if node_id in schedulable:
            in_degree.pop(node_id, None)
            # Decrement dependents
            for dep_id in dependents.get(node_id, set()):
                if dep_id in in_degree:
                    in_degree[dep_id] = max(0, in_degree[dep_id] - 1)

    # Apply cache
    for node_id, cached_result in cache.items():
        if node_id in in_degree:
            state.results[node_id] = cached_result
            await event_queue.put(NodeCompleteEvent(
                type="node_complete", node_id=node_id,
                result=filter_skipped(cached_result) if isinstance(cached_result, dict) else cached_result,
                cached=True,
            ))
            in_degree.pop(node_id)
            for dep_id in dependents.get(node_id, set()):
                if dep_id in in_degree:
                    in_degree[dep_id] = max(0, in_degree[dep_id] - 1)

    # Track running tasks (strong references prevent GC)
    running: dict[str, asyncio.Task] = {}

    def _find_ready() -> list[str]:
        """Find nodes with all deps satisfied, not yet running or done."""
        return [
            nid for nid, deg in in_degree.items()
            if deg == 0 and nid not in running and nid not in state.results
        ]

    def _dispatch_ready() -> None:
        """Create tasks for all ready nodes."""
        for node_id in _find_ready():
            task = asyncio.create_task(
                _execute_node_async(
                    node_id, state, compiled, event_queue, retry,
                ),
                name=f"node-{node_id}",
            )
            running[node_id] = task

    # Initial dispatch
    _dispatch_ready()

    # If nothing to do (all already completed, e.g. empty graph)
    if not running and not _find_ready():
        yield FlowCompleteEvent(
            type="flow_complete",
            results=filter_all_skipped(state.results),
        )
        return

    # Main event loop
    while running or _find_ready():
        if state.is_cancelled():
            _cancel_all(running)
            yield FlowCancelledEvent(
                type="flow_cancelled",
                completed_nodes=list(state.results.keys()),
            )
            return

        if state.is_timed_out():
            _cancel_all(running)
            yield FlowTimeoutEvent(
                type="flow_timeout",
                completed_nodes=list(state.results.keys()),
                elapsed_seconds=time.monotonic() - state._started_at,
                timeout_seconds=state._timeout_seconds,
            )
            return

        # Dispatch any newly ready nodes
        _dispatch_ready()

        if not running:
            break

        # Wait for next event from any running task
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue  # Re-check cancel/timeout

        # Internal sentinel: a node task finished
        if isinstance(event, _NodeDone):
            running.pop(event.node_id, None)

            if event.error:
                _cancel_all(running)
                if event.error_event:
                    yield event.error_event
                    yield FlowErrorEvent(
                        type="flow_error",
                        error=event.error_event["error"],
                        is_validation=event.error_event.get("is_validation", False),
                    )
                # If error_event is None, the event (e.g. flow_timeout) was already pushed
                return

            if event.paused:
                _cancel_all(running)
                yield event.pause_event
                return

            # Success or skip — unlock dependents
            for dep_id in dependents.get(event.node_id, set()):
                if dep_id in in_degree:
                    in_degree[dep_id] = max(0, in_degree[dep_id] - 1)

            continue  # Don't yield internal sentinel to caller

        # Regular event — yield to caller
        yield event

    # All done
    yield FlowCompleteEvent(
        type="flow_complete",
        results=filter_all_skipped(state.results),
    )


# =========================================================================
# Per-node async execution (runs as a task)
# =========================================================================


@dataclass
class _NodeDone:
    """Internal sentinel pushed when a node task finishes."""
    node_id: str
    error: bool = False
    error_event: dict | None = None
    paused: bool = False
    pause_event: dict | None = None


async def _execute_node_async(
    node_id: str,
    state: FlowRunState,
    compiled: CompiledGraph,
    event_queue: asyncio.Queue,
    retry: RetryConfig,
) -> None:
    """Execute a single node with retry, pushing events to the queue."""
    node = compiled.node_map[node_id]
    sink = state._event_sink

    # Skip propagation
    if should_skip_node(node, compiled.edge_map, state.results, compiled.consume_map):
        state.results[node_id] = SKIPPED
        await event_queue.put(NodeSkippedEvent(type="node_skipped", node_id=node_id))
        await event_queue.put(_NodeDone(node_id=node_id))
        return

    await event_queue.put(NodeStartEvent(type="node_start", node_id=node_id))

    # Resolve inputs
    inputs = state.resolver.resolve(
        node, compiled.edge_map, state.results, compiled.node_map,
        compiled.consume_map,
    )

    # Determine retry config: node-level overrides global
    node_def = compiled.registry.get(node.type)
    if node_def and node_def.max_retries > 0:
        max_retries = node_def.max_retries
        base_delay = node_def.retry_delay
        backoff = 2.0
    else:
        max_retries = retry.max_retries
        base_delay = retry.delay
        backoff = retry.backoff_factor

    attempt = 0
    last_error: Exception | None = None

    while attempt <= max_retries:
        if attempt > 0:
            delay = base_delay * (backoff ** (attempt - 1))
            await event_queue.put(NodeRetryEvent(
                type="node_retry",
                node_id=node_id,
                attempt=attempt,
                max_retries=max_retries,
                error=str(last_error),
                delay=delay,
            ))
            await asyncio.sleep(delay)

            # Re-resolve inputs in case upstream changed (unlikely but safe)
            inputs = state.resolver.resolve(
                node, compiled.edge_map, state.results, compiled.node_map,
                compiled.consume_map,
            )

        try:
            # Run sync node function in a separate thread (non-blocking)
            remaining = state.remaining_seconds()
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _dispatch_node, node.type, node_id, inputs,
                    node.data or {}, state, compiled,
                ),
                timeout=max(0.1, remaining) if remaining is not None else None,
            )

            result = normalize_result(result)
            state.results[node_id] = result
            await event_queue.put(NodeCompleteEvent(
                type="node_complete", node_id=node_id,
                result=filter_skipped(result),
            ))

            # Drain compound node events
            while (evt := sink.pop()) is not None:
                await event_queue.put(evt)

            await event_queue.put(_NodeDone(node_id=node_id))
            return

        except (asyncio.TimeoutError, TimeoutError):
            # Emit as flow_timeout (not node_error) to match the expected contract
            await event_queue.put(FlowTimeoutEvent(
                type="flow_timeout",
                completed_nodes=list(state.results.keys()),
                elapsed_seconds=time.monotonic() - state._started_at,
                timeout_seconds=state._timeout_seconds,
            ))
            await event_queue.put(_NodeDone(node_id=node_id, error=True, error_event=None))
            return

        except HumanInputRequired as e:
            cp = FlowCheckpoint(
                completed_node_ids=list(state.results.keys()),
                waiting_node_id=node_id,
                waiting_node_type=node.type,
                results=dict(state.results),
                store_data=dict(state.store._data),
                context=dict(state.context),
                prompt=e.prompt,
                input_schema=e.schema,
                execution_index=-1,  # Not used in eager mode
            )
            await event_queue.put(_NodeDone(
                node_id=node_id,
                paused=True,
                pause_event=FlowPausedEvent(
                    type="flow_paused",
                    node_id=node_id,
                    prompt=e.prompt,
                    schema=e.schema,
                    checkpoint=cp.to_dict(),
                ),
            ))
            return

        except NodeValidationError as e:
            # Validation errors are never retried
            await event_queue.put(_NodeDone(
                node_id=node_id,
                error=True,
                error_event=NodeErrorEvent(
                    type="node_error", node_id=node_id,
                    error=str(e), is_validation=True,
                ),
            ))
            return

        except (NodeExecutionError, NodeConnectionError) as e:
            last_error = e
            attempt += 1
            if attempt > max_retries:
                await event_queue.put(_NodeDone(
                    node_id=node_id,
                    error=True,
                    error_event=NodeErrorEvent(
                        type="node_error", node_id=node_id,
                        error=str(e), is_validation=False,
                    ),
                ))
                return
            # Loop continues to retry


# =========================================================================
# Node dispatch (unchanged)
# =========================================================================


def _dispatch_node(
    node_type: str,
    node_id: str,
    inputs: dict[str, Any],
    data: dict[str, Any],
    state: FlowRunState,
    compiled: CompiledGraph,
) -> Any:
    """Route execution to the right handler."""
    from conductor.execution.request import NodeExecRequest

    req = NodeExecRequest(
        node_id=node_id,
        node_type=node_type,
        inputs=inputs,
        data=data,
        state=state,
    )

    # 1. Compound node
    if node_id in compiled.compound_nodes:
        return compiled.compound_nodes[node_id].execute(req)

    # 2. Extension node
    ext = compiled.extension_resolver
    if ext and ext.is_known_type(node_type):
        executor = ext.create_executor(node_type)
        return executor.execute(req)

    # 3. Registry node
    node_def = compiled.registry.get(node_type)
    if not node_def:
        raise NodeExecutionError(
            f"Node type '{node_type}' not found in registry",
            node_id=node_id, node_type=node_type,
        )

    # 3a. Class-based node
    if hasattr(node_def, "_node_class") and node_def._node_class is not None:
        try:
            instance = node_def._node_class()
            return instance.execute(req)
        except (NodeValidationError, NodeExecutionError, HumanInputRequired):
            raise
        except Exception as e:
            raise NodeExecutionError(
                f"Execution failed for {node_type}: {type(e).__name__}: {e}",
                node_id=node_id, node_type=node_type, original=e,
            ) from e

    # 3b. Function-based node
    if not node_def.func:
        raise NodeExecutionError(
            f"Node type '{node_type}' has no callable",
            node_id=node_id, node_type=node_type,
        )

    if node_def.validation_model:
        from pydantic import ValidationError
        try:
            validated = node_def.validation_model(**inputs)
            inputs = validated.model_dump()
        except ValidationError as e:
            raise NodeValidationError(
                str(e), node_id=node_id, node_type=node_type, original=e,
            ) from e

    inputs = _inject_store(node_def.func, inputs, state)

    try:
        return node_def.func(**inputs)
    except (NodeValidationError, NodeExecutionError, NodeConnectionError, HumanInputRequired):
        raise
    except Exception as e:
        raise NodeExecutionError(
            f"Execution failed for {node_type}: {type(e).__name__}: {e}",
            node_id=node_id, node_type=node_type, original=e,
        ) from e


def _inject_store(func: Any, inputs: dict[str, Any], state: FlowRunState) -> dict[str, Any]:
    """If the function has a FlowStore parameter, inject it."""
    from typing import get_type_hints

    try:
        hints = get_type_hints(func)
    except Exception:
        return inputs

    for name, hint in hints.items():
        if hint is FlowStore:
            inputs = {**inputs, name: state.store}
            break
    return inputs


# =========================================================================
# Sync wrappers
# =========================================================================


async def collect(events: AsyncGenerator[ExecutionEvent, None]) -> dict[str, Any]:
    """Consume all events, return final results."""
    results: dict[str, Any] = {}
    async for event in events:
        if event["type"] == "flow_complete":
            results = event["results"]
        elif event["type"] == "flow_paused":
            raise FlowPausedException(event["checkpoint"])
        elif event["type"] in ("flow_error", "flow_cancelled", "flow_timeout"):
            raise FlowExecutionException(event.get("error", "Flow did not complete"))
    return results


def execute_sync(compiled: CompiledGraph, **kwargs: Any) -> dict[str, Any]:
    """Blocking entry point."""
    return asyncio.run(collect(execute(compiled, **kwargs)))


def resume_sync(
    compiled: CompiledGraph,
    checkpoint: FlowCheckpoint | dict[str, Any],
    response: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Blocking resume."""
    return asyncio.run(collect(resume(compiled, checkpoint, response, **kwargs)))


# =========================================================================
# Internal helpers
# =========================================================================


def _build_state(
    compiled: CompiledGraph,
    timeout_seconds: int,
    context: dict[str, Any] | None,
) -> FlowRunState:
    return FlowRunState(
        compiled=compiled,
        resolver=InputResolver(compiled.registry),
        results={},
        _event_sink=EventSink(),
        _started_at=time.monotonic(),
        _timeout_seconds=timeout_seconds,
        context=context or {},
    )


def _build_dep_graph(
    compiled: CompiledGraph,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build deps (node -> deps) and dependents (node -> nodes that depend on it).

    Includes both explicit edges and shared-reference consume bindings — the
    scheduler treats them identically.
    """
    deps: dict[str, set[str]] = defaultdict(set)
    dependents: dict[str, set[str]] = defaultdict(set)

    for (target_id, _handle), sources in compiled.edge_map.items():
        for source_id, _source_handle in sources:
            deps[target_id].add(source_id)
            dependents[source_id].add(target_id)

    # Consume dependencies: if the consumer is managed by a compound region,
    # redirect the dependency to the region's start node (the schedulable
    # representative). Otherwise record as-is.
    managed_to_start = compiled.managed_to_region_start
    for (target_id, _target_handle), (source_id, _source_handle) in compiled.consume_map.items():
        effective_target = managed_to_start.get(target_id, target_id)
        deps[effective_target].add(source_id)
        dependents[source_id].add(effective_target)

    return dict(deps), dict(dependents)


def _cancel_all(running: dict[str, asyncio.Task]) -> None:
    """Cancel all running tasks."""
    for task in running.values():
        task.cancel()
    running.clear()
