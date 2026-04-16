"""Execution engine — the single streaming entry point."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flowengine._sentinel import SKIPPED
from flowengine.errors import (
    FlowExecutionException,
    FlowPausedException,
    HumanInputRequired,
    NodeExecutionException,
    NodeValidationException,
)
from flowengine.execution.checkpoint import FlowCheckpoint
from flowengine.execution.events import (
    EventSink,
    ExecutionEvent,
    FlowCancelledEvent,
    FlowCompleteEvent,
    FlowErrorEvent,
    FlowPausedEvent,
    FlowTimeoutEvent,
    NodeCompleteEvent,
    NodeErrorEvent,
    NodeSkippedEvent,
    NodeStartEvent,
)
from flowengine.execution.resolver import InputResolver
from flowengine.execution.results import filter_all_skipped, filter_skipped, normalize_result
from flowengine.execution.skip import should_skip_node
from flowengine.execution.state import FlowRunState
from flowengine.execution.store import FlowStore
from flowengine.graph.compiler import CompiledGraph
from flowengine.types import NodeResult


# =========================================================================
# Primary entry point
# =========================================================================


async def execute(
    compiled: CompiledGraph,
    *,
    timeout_seconds: int = 300,
    context: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Execute a compiled graph, yielding streaming events.

    If a node raises HumanInputRequired, yields a FlowPausedEvent
    containing a serializable checkpoint. Use resume() to continue.
    """
    state = _build_state(compiled, timeout_seconds, context)
    async for event in _run_loop(state, cache=cache or {}, start_index=0):
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
) -> AsyncGenerator[ExecutionEvent, None]:
    """Resume a paused flow with a human's response.

    Args:
        compiled: The same compiled graph (re-compile from original nodes/edges).
        checkpoint: The checkpoint from the FlowPausedEvent (or its dict form).
        response: The human's response — injected as the waiting node's result.
        timeout_seconds: Timeout for the remaining execution.
        context: Optional context override (defaults to checkpoint's context).
    """
    if isinstance(checkpoint, dict):
        checkpoint = FlowCheckpoint.from_dict(checkpoint)

    # Restore state
    state = _build_state(
        compiled,
        timeout_seconds,
        context or checkpoint.context,
    )
    state.results = dict(checkpoint.results)
    state.store = FlowStore(dict(checkpoint.store_data))

    # Inject the human response as the waiting node's result
    state.results[checkpoint.waiting_node_id] = normalize_result(response)

    yield NodeCompleteEvent(
        type="node_complete",
        node_id=checkpoint.waiting_node_id,
        result=normalize_result(response),
    )

    # Continue from the node after the one that paused
    async for event in _run_loop(
        state,
        cache={},
        start_index=checkpoint.execution_index + 1,
    ):
        yield event


# =========================================================================
# Core execution loop (shared by execute and resume)
# =========================================================================


async def _run_loop(
    state: FlowRunState,
    *,
    cache: dict[str, Any],
    start_index: int,
) -> AsyncGenerator[ExecutionEvent, None]:
    """The inner execution loop — iterates through nodes in order."""
    compiled = state.compiled
    sink = state._event_sink
    completed: list[str] = list(state.results.keys())

    for idx in range(start_index, len(compiled.execution_order)):
        node_id = compiled.execution_order[idx]

        if node_id in compiled.managed_ids:
            continue

        if state.is_cancelled():
            yield FlowCancelledEvent(
                type="flow_cancelled", completed_nodes=completed
            )
            return

        if state.is_timed_out():
            yield FlowTimeoutEvent(
                type="flow_timeout",
                completed_nodes=completed,
                elapsed_seconds=time.monotonic() - state._started_at,
                timeout_seconds=state._timeout_seconds,
            )
            return

        node = compiled.node_map[node_id]

        # Skip propagation
        if should_skip_node(node, compiled.edge_map, state.results):
            state.results[node_id] = SKIPPED
            yield NodeSkippedEvent(type="node_skipped", node_id=node_id)
            completed.append(node_id)
            continue

        yield NodeStartEvent(type="node_start", node_id=node_id)
        await asyncio.sleep(0)

        # Cache hit
        if node_id in cache:
            state.results[node_id] = cache[node_id]
            yield NodeCompleteEvent(
                type="node_complete", node_id=node_id,
                result=filter_skipped(cache[node_id]) if isinstance(cache[node_id], dict) else cache[node_id],
                cached=True,
            )
            completed.append(node_id)
            continue

        # Resolve inputs
        inputs = state.resolver.resolve(
            node, compiled.edge_map, state.results, compiled.node_map
        )

        # Execute
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = loop.run_in_executor(
                    pool, _dispatch_node, node.type, node_id, inputs,
                    node.data or {}, state, compiled,
                )
                remaining = state.remaining_seconds()
                result = await asyncio.wait_for(
                    future,
                    timeout=max(0.1, remaining) if remaining is not None else None,
                )

            result = normalize_result(result)
            state.results[node_id] = result
            completed.append(node_id)
            yield NodeCompleteEvent(
                type="node_complete", node_id=node_id,
                result=filter_skipped(result),
            )

        except asyncio.TimeoutError:
            yield FlowTimeoutEvent(
                type="flow_timeout",
                completed_nodes=completed,
                elapsed_seconds=time.monotonic() - state._started_at,
                timeout_seconds=state._timeout_seconds,
            )
            return

        except HumanInputRequired as e:
            # Checkpoint and pause
            cp = FlowCheckpoint(
                completed_node_ids=completed,
                waiting_node_id=node_id,
                waiting_node_type=node.type,
                results=dict(state.results),
                store_data=dict(state.store._data),
                context=dict(state.context),
                prompt=e.prompt,
                input_schema=e.schema,
                execution_index=idx,
            )
            yield FlowPausedEvent(
                type="flow_paused",
                node_id=node_id,
                prompt=e.prompt,
                schema=e.schema,
                checkpoint=cp.to_dict(),
            )
            return

        except NodeValidationException as e:
            yield NodeErrorEvent(
                type="node_error", node_id=node_id,
                error=str(e), is_validation=True,
            )
            yield FlowErrorEvent(
                type="flow_error", error=str(e), is_validation=True,
            )
            return

        except NodeExecutionException as e:
            yield NodeErrorEvent(
                type="node_error", node_id=node_id,
                error=str(e), is_validation=False,
            )
            yield FlowErrorEvent(
                type="flow_error", error=str(e), is_validation=False,
            )
            return

        # Drain compound node events
        while (event := sink.pop()) is not None:
            yield event

    yield FlowCompleteEvent(
        type="flow_complete",
        results=filter_all_skipped(state.results),
    )


# =========================================================================
# Node dispatch
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
    from flowengine.execution.request import NodeExecRequest

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
        raise NodeExecutionException(f"Node type '{node_type}' not found in registry")

    # 3a. Class-based node
    if hasattr(node_def, "_node_class") and node_def._node_class is not None:
        try:
            instance = node_def._node_class()
            return instance.execute(req)
        except (NodeValidationException, NodeExecutionException, HumanInputRequired):
            raise
        except Exception as e:
            raise NodeExecutionException(
                f"Execution failed for {node_type}: {type(e).__name__}: {e}"
            ) from e

    # 3b. Function-based node
    if not node_def.func:
        raise NodeExecutionException(f"Node type '{node_type}' has no callable")

    # Validate inputs
    if node_def.validation_model:
        from pydantic import ValidationError

        try:
            validated = node_def.validation_model(**inputs)
            inputs = validated.model_dump()
        except ValidationError as e:
            raise NodeValidationException(str(e)) from e

    # Inject FlowStore if the function declares it
    inputs = _inject_store(node_def.func, inputs, state)

    # Call the function
    try:
        return node_def.func(**inputs)
    except (NodeValidationException, NodeExecutionException, HumanInputRequired):
        raise
    except Exception as e:
        raise NodeExecutionException(
            f"Execution failed for {node_type}: {type(e).__name__}: {e}"
        ) from e


def _inject_store(
    func: Any,
    inputs: dict[str, Any],
    state: FlowRunState,
) -> dict[str, Any]:
    """If the function has a FlowStore parameter, inject it."""
    import inspect
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
    """Consume all events, return final results.

    Raises FlowExecutionException on error/cancel/timeout.
    Raises FlowPausedException on human-in-the-loop pause (carries checkpoint).
    """
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
    """Blocking entry point. Runs the async engine and collects results.

    Raises FlowPausedException if a node requests human input.
    """
    return asyncio.run(collect(execute(compiled, **kwargs)))


def resume_sync(
    compiled: CompiledGraph,
    checkpoint: FlowCheckpoint | dict[str, Any],
    response: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Blocking resume. Continues a paused flow with a human's response.

    Raises FlowPausedException if another node requests human input.
    """
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
