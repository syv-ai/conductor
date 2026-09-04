"""Execution engine — eager-scheduled, parallel, with retry support."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from conductor._sentinel import SKIPPED, is_skipped
from conductor.errors import (
    FlowExecutionException,
    FlowPausedException,
    HumanInputRequired,
    InputResolutionError,
    NodeConnectionError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
    SignalRequired,
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
    SignalWaitingEvent,
)
from conductor.execution.request import NodeExecRequest
from conductor.execution.resolver import InputResolver
from conductor.execution.results import filter_all_skipped, filter_skipped, normalize_result
from conductor.execution.retry import NO_RETRY, RetryConfig
from conductor.execution.skip import should_skip_node
from conductor.execution.state import FlowRunState
from conductor.execution.store import FlowStore
from conductor.graph.compiler import CompiledGraph
from conductor.interface import model_of
from conductor.metadata import Output
from conductor.registry import runner_for
from conductor.returns import unpack
from conductor.series import Index, Series

# Internal sentinel pushed into the event queue when all work is done
_DONE = object()
_FATAL = object()

# How often a running node's event sink is polled so the events it emits
# mid-execution stream to the host live instead of arriving in a burst
# once the node finishes. Small enough to feel live, large enough not to
# busy-spin the event loop.
_SINK_DRAIN_INTERVAL = 0.05


async def _drain_sink_live(
    sink: Any, event_queue: asyncio.Queue, interval: float = _SINK_DRAIN_INTERVAL
) -> None:
    """Forward events from a node's sink to ``event_queue`` while it runs.

    A node may push events to ``state._event_sink`` from its worker thread
    while it runs. Without a concurrent drainer those events sit in the
    sink until the node completes and the post-dispatch tail-drain flushes
    them — so a host sees nothing until then. This coroutine
    runs alongside the dispatch task and pumps the sink on a short
    interval; the caller cancels it once dispatch returns, and the
    existing tail-drain flushes anything emitted after the final poll.
    """
    try:
        while True:
            while (evt := sink.pop()) is not None:
                await event_queue.put(evt)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # Final sweep so events emitted between the last poll and
        # cancellation aren't left for the tail-drain to reorder after
        # the parent's node_complete.
        while (evt := sink.pop()) is not None:
            await event_queue.put(evt)
        raise


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
    store_data: dict[str, Any] | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Execute a compiled graph with eager parallel scheduling.

    Nodes start as soon as all their dependencies are done — independent
    branches run concurrently. Retry is configurable per-node or globally.

    ``store_data`` pre-seeds the ``FlowStore`` before the first node runs.
    Useful for hosts that inject per-request context (user, session,
    tenant id, …) via the ``store: FlowStore`` parameter on node
    functions. Separate from ``context`` which is kept for checkpoint
    serialization metadata.
    """
    state = _build_state(compiled, timeout_seconds, context)
    if store_data:
        state.store = FlowStore(dict(store_data))
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
    """Resume a paused flow with a human's response or external signal."""
    if isinstance(checkpoint, dict):
        checkpoint = FlowCheckpoint.from_dict(checkpoint)

    state = _build_state(
        compiled, timeout_seconds, context or checkpoint.context,
    )
    state.results = dict(checkpoint.results)
    state.store = FlowStore(dict(checkpoint.store_data))
    state.skipped_edges = set(checkpoint.skipped_edges)
    state.completed_order = list(checkpoint.completed_node_ids)

    # Inject response as the waiting node's result
    state.results[checkpoint.waiting_node_id] = normalize_result(response)
    state.completed_order.append(checkpoint.waiting_node_id)

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

    schedulable = set(compiled.execution_order)

    # Track in-degree (number of unfinished deps). Nodes with in-degree 0
    # feed a ready_queue — O(1) dispatch instead of re-scanning every tick.
    in_degree: dict[str, int] = {}
    for node_id in schedulable:
        node_deps = deps.get(node_id, set())
        in_degree[node_id] = len(node_deps & schedulable)

    ready_queue: deque[str] = deque()

    def _satisfy(node_id: str) -> None:
        """Mark ``node_id`` as completed — decrement dependents' in-degrees and
        enqueue any that reach zero."""
        for dep_id in dependents.get(node_id, ()):
            if dep_id not in in_degree:
                continue
            in_degree[dep_id] -= 1
            if in_degree[dep_id] <= 0:
                ready_queue.append(dep_id)

    # Pre-satisfy nodes already in results (resume/cache). These do not enter
    # the ready queue themselves; they just unlock their dependents.
    for node_id in list(state.results.keys()):
        if node_id in in_degree:
            in_degree.pop(node_id)
            _satisfy(node_id)

    # Apply cache — emit a node_complete event for each cached node, then
    # treat it as satisfied.
    for node_id, cached_result in cache.items():
        if node_id in in_degree:
            state.results[node_id] = cached_result
            state.completed_order.append(node_id)
            await event_queue.put(NodeCompleteEvent(
                type="node_complete", node_id=node_id,
                result=filter_skipped(cached_result) if isinstance(cached_result, dict) else cached_result,
                cached=True,
            ))
            in_degree.pop(node_id)
            _satisfy(node_id)

    # Seed the ready queue with nodes that started at in_degree 0.
    for nid, deg in in_degree.items():
        if deg == 0:
            ready_queue.append(nid)

    running: dict[str, asyncio.Task] = {}

    def _dispatch_ready() -> None:
        while ready_queue:
            node_id = ready_queue.popleft()
            # Guard against the same id being enqueued twice, or against a
            # cached node re-appearing: skip if already running/completed.
            if node_id in running or node_id in state.results:
                continue
            if node_id not in in_degree:
                continue
            task = asyncio.create_task(
                _execute_node_async(
                    node_id, state, compiled, event_queue, retry,
                ),
                name=f"node-{node_id}",
            )
            running[node_id] = task

    _dispatch_ready()

    if not running and not ready_queue:
        yield FlowCompleteEvent(
            type="flow_complete",
            results=filter_all_skipped(state.results),
        )
        return

    while running or ready_queue:
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

        _dispatch_ready()

        if not running:
            break

        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

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
                return

            if event.paused:
                _cancel_all(running)
                yield event.pause_event
                return

            # Success or skip — unlock dependents via the ready queue.
            _satisfy(event.node_id)
            _dispatch_ready()
            continue

        yield event

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
    node_cls = compiled.registry.get(node.type)
    policy = node_cls.versions[node.version].policy if node_cls is not None else None

    # Skip propagation
    if should_skip_node(
        node, compiled.edge_map, state.results,
        state.skipped_edges, compiled.incoming_map,
    ):
        state.results[node_id] = SKIPPED
        state.completed_order.append(node_id)
        await event_queue.put(NodeSkippedEvent(type="node_skipped", node_id=node_id))
        await event_queue.put(_NodeDone(node_id=node_id))
        return

    try:
        inputs = state.resolver.resolve(
            node, compiled.edge_map, state.results, compiled.node_map,
            state.skipped_edges, compiled.incoming_map,
        )
    except InputResolutionError as e:
        # Raised before the try below; without this the task would end
        # with no ``_NodeDone`` and the scheduler would wait forever.
        await event_queue.put(_NodeDone(
            node_id=node_id,
            error=True,
            error_event=NodeErrorEvent(
                type="node_error", node_id=node_id, error=str(e), is_validation=False,
            ),
        ))
        return

    start_event: NodeStartEvent = NodeStartEvent(type="node_start", node_id=node_id)
    await event_queue.put(start_event)

    # Determine retry config: the version's policy overrides the run-level default
    if policy is not None and policy.retries > 0:
        max_retries = policy.retries
        base_delay = policy.delay
        backoff = 2.0
    else:
        max_retries = retry.max_retries
        base_delay = retry.delay
        backoff = retry.backoff_factor

    # Determine node timeout budget — node-level wins over flow-level
    node_timeout = policy.timeout if policy is not None else None

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

            inputs = state.resolver.resolve(
                node, compiled.edge_map, state.results, compiled.node_map,
                state.skipped_edges, compiled.incoming_map,
            )

        try:
            remaining = state.remaining_seconds()
            effective_timeout = _effective_timeout(remaining, node_timeout)
            # Run dispatch and a live sink-drainer concurrently so events
            # pushed into the sink mid-execution stream to the host. The
            # drainer is cancelled as soon as dispatch settles; the
            # tail-drain below flushes any final stragglers.
            dispatch = asyncio.ensure_future(
                asyncio.to_thread(
                    _dispatch_node, node.type, node_id, inputs,
                    node.data, state, compiled,
                )
            )
            drainer = asyncio.ensure_future(
                _drain_sink_live(sink, event_queue)
            )
            try:
                result = await asyncio.wait_for(
                    dispatch, timeout=effective_timeout
                )
            finally:
                drainer.cancel()
                try:
                    await drainer
                except asyncio.CancelledError:
                    pass

            state.results[node_id] = result
            state.completed_order.append(node_id)
            await event_queue.put(NodeCompleteEvent(
                type="node_complete", node_id=node_id,
                result=filter_skipped(result),
            ))

            while (evt := sink.pop()) is not None:
                await event_queue.put(evt)

            await event_queue.put(_NodeDone(node_id=node_id))
            return

        except (asyncio.TimeoutError, TimeoutError):
            # Distinguish per-node timeout from flow-wide timeout by looking
            # at which budget was smaller (and whichever was actually hit).
            flow_remaining = state.remaining_seconds()
            node_was_tighter = (
                node_timeout is not None
                and (flow_remaining is None or node_timeout < flow_remaining + 0.01)
            )

            if node_was_tighter and not state.is_timed_out():
                last_error = NodeTimeoutError(
                    f"Node '{node_id}' exceeded its timeout of {node_timeout}s",
                    node_id=node_id, node_type=node.type,
                )
                attempt += 1
                if attempt > max_retries:
                    await event_queue.put(_NodeDone(
                        node_id=node_id,
                        error=True,
                        error_event=NodeErrorEvent(
                            type="node_error", node_id=node_id,
                            error=str(last_error), is_validation=False,
                            is_timeout=True,
                        ),
                    ))
                    return
                continue

            # Flow-level timeout
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
                completed_node_ids=list(state.completed_order),
                waiting_node_id=node_id,
                waiting_node_type=node.type,
                results=dict(state.results),
                store_data=state.store.to_dict(),
                context=dict(state.context),
                prompt=e.prompt,
                input_schema=e.schema,
                execution_index=-1,
                skipped_edges=list(state.skipped_edges),
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

        except SignalRequired as e:
            cp = FlowCheckpoint(
                completed_node_ids=list(state.completed_order),
                waiting_node_id=node_id,
                waiting_node_type=node.type,
                results=dict(state.results),
                store_data=state.store.to_dict(),
                context=dict(state.context),
                prompt=f"Waiting for signal '{e.signal_name}'",
                input_schema=None,
                execution_index=-1,
                skipped_edges=list(state.skipped_edges),
                signal_name=e.signal_name,
                correlation=e.correlation,
                signal_timeout_seconds=e.timeout_seconds,
            )
            state.pending_signals[node_id] = {
                "name": e.signal_name,
                "correlation": e.correlation,
                "timeout_seconds": e.timeout_seconds,
            }
            await event_queue.put(SignalWaitingEvent(
                type="signal_waiting",
                node_id=node_id,
                signal_name=e.signal_name,
                correlation=e.correlation,
                timeout_seconds=e.timeout_seconds,
                checkpoint=cp.to_dict(),
            ))
            await event_queue.put(_NodeDone(
                node_id=node_id,
                paused=True,
                pause_event=FlowPausedEvent(
                    type="flow_paused",
                    node_id=node_id,
                    prompt=f"Waiting for signal '{e.signal_name}'",
                    schema=None,
                    checkpoint=cp.to_dict(),
                ),
            ))
            return

        except NodeValidationError as e:
            await event_queue.put(_NodeDone(
                node_id=node_id,
                error=True,
                error_event=NodeErrorEvent(
                    type="node_error", node_id=node_id,
                    error=str(e), is_validation=True,
                ),
            ))
            return

        except (NodeExecutionError, NodeConnectionError, NodeTimeoutError) as e:
            # Respect the per-error retryable classification (see
            # ``errors.py``). Subclasses can opt out of retry by setting
            # ``retryable = False`` on the class, and individual
            # instances may pass ``retryable=False`` to override on a
            # case-by-case basis. Anything fatal short-circuits out of
            # the retry loop without consuming an attempt.
            if not getattr(e, "retryable", True):
                await event_queue.put(_NodeDone(
                    node_id=node_id,
                    error=True,
                    error_event=NodeErrorEvent(
                        type="node_error", node_id=node_id,
                        error=str(e),
                        is_validation=isinstance(e, NodeValidationError),
                    ),
                ))
                return
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

        except Exception as e:
            # Any other error (a bug in a node, a runaway loop) surfaces as
            # a node_error and aborts the flow. Retry does not
            # apply to non-recognized exceptions.
            await event_queue.put(_NodeDone(
                node_id=node_id,
                error=True,
                error_event=NodeErrorEvent(
                    type="node_error", node_id=node_id,
                    error=f"{type(e).__name__}: {e}",
                    is_validation=False,
                ),
            ))
            return


def _effective_timeout(
    remaining: float | None,
    node_timeout: float | None,
) -> float | None:
    """Min of flow-remaining and node-specific timeout."""
    candidates = [x for x in (remaining, node_timeout) if x is not None]
    if not candidates:
        return None
    return max(0.05, min(candidates))



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
) -> dict[str, Any]:
    """Route execution to the right handler; the answer is ``{output name: value}``."""
    req = NodeExecRequest(
        node_id=node_id,
        node_type=node_type,
        inputs=inputs,
        data=data,
        state=state,
    )

    # 1. Extension node
    ext = compiled.extension_resolver
    if ext and ext.is_known_type(node_type):
        executor = ext.create_executor(node_type)
        return normalize_result(executor.execute(req))

    # 2. Registry node
    node_def = compiled.registry.get(node_type)
    if node_def is None:
        raise NodeExecutionError(
            f"Node type '{node_type}' not found in registry",
            node_id=node_id, node_type=node_type,
        )

    # Coerce the raw inputs through the placement's roster before anything
    # else touches them.
    from pydantic import ValidationError

    pinned = compiled.node_map[node_id].version
    roster = compiled.node_inputs.get(
        node_id, node_def.versions[pinned].interface.inputs
    )
    try:
        validated = model_of(roster)(**inputs)
    except ValidationError as e:
        raise NodeValidationError(
            _format_validation_error(e, roster),
            node_id=node_id, node_type=node_type, original=e,
        ) from e

    # The validated instances themselves, not a dump: ``run`` receives the
    # declared dtypes and nothing is re-serialised on the way in.
    inputs = {name: getattr(validated, name) for name in type(validated).model_fields}

    runner = runner_for(compiled.registry, node_type, pinned)
    inputs = _inject_store(runner, inputs, state)
    inputs = _filter_to_signature(runner, inputs)

    try:
        value = runner(**inputs)
    except (NodeValidationError, NodeExecutionError, NodeConnectionError,
            HumanInputRequired, SignalRequired):
        raise
    except Exception as e:
        raise NodeExecutionError(
            f"Execution failed for {node_type}: {type(e).__name__}: {e}",
            node_id=node_id, node_type=node_type, original=e,
        ) from e
    return _outputs_of(
        node_def.versions[pinned].interface.returns, compiled.node_outputs[node_id], value
    )


def _outputs_of(returns: Any, outputs: tuple[Output, ...], value: Any) -> dict[str, Any]:
    """What ``run`` returned, split across the placement's outputs by name.

    ``unpack`` reads the return declaration. A series output is returned by
    the node as a plain sequence and lands here on a fresh root index: its
    rows came from nowhere the engine tracks. ``SKIPPED`` on an output
    passes through as the value it is.
    """
    values = unpack(returns, value, outputs)
    series_outputs = {
        out.name for out in outputs
        if isinstance(out.dtype, type) and issubclass(out.dtype, Series)
    }
    return {
        name: (
            Series(Index.fresh(), v)
            if name in series_outputs and not is_skipped(v) and not isinstance(v, Series)
            else v
        )
        for name, v in values.items()
    }


def _format_validation_error(e: Any, roster: Any) -> str:
    """Collapse a pydantic ``ValidationError`` into a one-line-per-field
    summary suitable for end-user surfaces.

    Pydantic's default ``str(e)`` enumerates every union arm × every nested
    field, producing 7+ lines for a single bad input — unreadable in a
    UI toast. This helper:

      * deduplicates by field path (after stripping union-arm segments
        like ``"list[union[float,int]]"`` from ``loc``),
      * picks the most specific message per field (prefers
        ``"Field required"`` and ``"Input should be ..."`` over generic
        ``"Input should be a valid …"`` from union fan-out),
      * resolves field ids to their declared ``title``.

    Hosts that want structured access can still read ``e.original`` (the
    pydantic ``ValidationError``) off the raised ``NodeValidationError``.
    """
    # Discriminate pydantic union-arm tags from real path segments. Arm
    # tags are produced by pydantic's smart union mode and look like
    # ``"int"``, ``"float"``, ``"list[union[...]]"``, ``"dict[...,...]"``.
    def _is_union_arm(seg: Any) -> bool:
        if not isinstance(seg, str):
            return False
        if seg.startswith(("list[", "dict[", "tuple[", "union[")):
            return True
        return seg in {
            "int", "float", "str", "bool", "bytes",
            "nonetype", "none", "any",
        }

    # Build a {field_title: message} map preserving insertion order.
    seen: dict[str, str] = {}
    label_by_name = {inp.name: inp.title for inp in roster}

    for err in e.errors():
        loc = [seg for seg in err.get("loc", ()) if not _is_union_arm(seg)]
        if not loc:
            continue
        root = loc[0]
        label = label_by_name.get(root, str(root))
        sub_path = ".".join(str(s) for s in loc[1:])
        key = f"{label}.{sub_path}" if sub_path else label

        msg = str(err.get("msg", ""))
        # Strip pydantic's "Value error, " prefix from validator-raised errors.
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]

        if key in seen:
            # Prefer "Field required" / non-generic messages when collapsing.
            current = seen[key]
            if current.startswith("Input should be a valid"):
                seen[key] = msg
            continue
        seen[key] = msg

    if not seen:
        return str(e)

    parts = [f"'{k}': {v}" for k, v in seen.items()]
    return "Invalid inputs — " + "; ".join(parts)


def _inject_store(func: Any, inputs: dict[str, Any], state: FlowRunState) -> dict[str, Any]:
    """If the callable declares a ``FlowStore`` parameter, inject it.

    Read off the signature rather than resolved type hints, because a
    class node's runner carries the method's signature and nothing else.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return inputs
    for name, param in params.items():
        if param.annotation is FlowStore or param.annotation == "FlowStore":
            return {**inputs, name: state.store}
    return inputs



def _filter_to_signature(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that the callable can't accept as keyword arguments.

    The input resolver overlays a node's static ``data`` onto the resolved
    inputs, so a saved flow can carry stray keys (host metadata, migration
    breadcrumbs) that aren't parameters of the node function. Forwarding
    those verbatim raises ``TypeError: got an unexpected keyword argument``.

    A function that declares ``**kwargs`` (``VAR_KEYWORD``) accepts any key,
    so nothing is filtered — this preserves the nodes that
    legitimately receive extra wired handles.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# =========================================================================
# Sync wrappers
# =========================================================================


async def collect(events: AsyncGenerator[ExecutionEvent, None]) -> dict[str, Any]:
    """Consume all events, return final results."""
    results: dict[str, Any] = {}
    last_node_error: str | None = None
    async for event in events:
        et = event["type"]
        if et == "flow_complete":
            results = event["results"]
        elif et == "flow_paused":
            raise FlowPausedException(event["checkpoint"])
        elif et == "node_error":
            last_node_error = event.get("error")
        elif et in ("flow_error", "flow_cancelled", "flow_timeout"):
            msg = event.get("error") or last_node_error or "Flow did not complete"
            raise FlowExecutionException(msg)
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
        resolver=InputResolver(compiled.registry, node_inputs=compiled.node_inputs),
        results={},
        _event_sink=EventSink(),
        _started_at=time.monotonic(),
        _timeout_seconds=timeout_seconds,
        context=context or {},
    )


def _build_dep_graph(
    compiled: CompiledGraph,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build deps (node -> deps) and dependents (node -> nodes that depend on it)."""
    deps: dict[str, set[str]] = defaultdict(set)
    dependents: dict[str, set[str]] = defaultdict(set)

    for target_id, entries in compiled.incoming_map.items():
        for _target_handle, source_id, _source_handle, _wire_id in entries:
            deps[target_id].add(source_id)
            dependents[source_id].add(target_id)

    return dict(deps), dict(dependents)


def _cancel_all(running: dict[str, asyncio.Task]) -> None:
    """Cancel all running tasks."""
    for task in running.values():
        task.cancel()
    running.clear()
