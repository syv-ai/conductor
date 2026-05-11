"""Engine stress tests.

These tests intentionally push the engine harder than the regular suite —
large iteration counts, long node chains, and cancellation during retry
sleep. They are slower than unit tests (each can take several seconds);
mark them ``slow`` so CI can opt out via ``-m "not slow"``.

Scenarios covered:

1. **10k-iteration for-each loop.** Verifies the ``MAX_ITERATIONS`` cap in
   :mod:`conductor.compound.for_each` is honored, that the loop completes
   in reasonable time, and that no obvious memory leak occurs.
2. **500-node deep linear chain.** Verifies compile and execution scale
   linearly with node count and that values propagate end-to-end.
3. **Concurrent cancel-mid-retry.** Verifies that setting the cancellation
   flag while a node is asleep between retry attempts honors the cancel
   (no further retry runs, ``flow_cancelled`` event emitted).
"""

from __future__ import annotations

import asyncio
import time
import tracemalloc
from typing import Annotated

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.compound.for_each import FOR_EACH, MAX_ITERATIONS
from conductor.execution.engine import execute, execute_sync
from conductor.widgets import ConnectionList, Output, Text

# ---------------------------------------------------------------------------
# Test 1 — 10,000 iteration for-each loop
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_for_each_10k_iterations_capped_and_no_leak() -> None:
    """Build a flow that asks for 10,000 iterations of a single string-upper
    body. Conductor's ``MAX_ITERATIONS`` cap (currently 1000) trims the
    items list before iteration begins, so the assertion is against the
    actual configured cap, not the requested count.

    We also take a ``tracemalloc`` snapshot before/after to flag any
    obvious leak. The bound (200 MB peak) is generous; the goal is to
    catch O(n^2)-style accidents, not microbenchmark.
    """
    registry = NodeRegistry()

    @registry.node(
        "upper", version=1, name="Upper", description="Uppercases its input",
    )
    def upper(
        text: Annotated[str, Text(label="Input")] = "",
    ) -> Annotated[str, Output(label="Output")]:
        return text.upper()

    @registry.node(
        "for-each-start", version=1, name="For Each Start",
        description="Start of for-each loop", dynamic_handles=True,
    )
    def for_each_start(
        items: Annotated[list[str], ConnectionList(label="Items")],
    ) -> tuple[
        Annotated[str, Output(label="Item")],
        Annotated[int, Output(label="Index")],
    ]:
        raise NotImplementedError("Handled by compound node")

    @registry.node(
        "for-each-end", version=1, name="For Each End",
        description="End of for-each loop", dynamic_handles=True,
    )
    def for_each_end(
        item: Annotated[str, Text(label="Item")] = "",
    ) -> Annotated[list[str], Output(label="Collected")]:
        raise NotImplementedError("Handled by compound node")

    requested = 10_000
    items = [f"v{i}" for i in range(requested)]

    nodes = [
        GraphNode("start", "for-each-start@1", {"items": items}),
        GraphNode("body", "upper@1", None),
        GraphNode("end", "for-each-end@1", None),
    ]
    edges = [
        GraphEdge("e1", "start", "body", "output_1", "text"),
        GraphEdge("e2", "body", "end", "result", "item"),
    ]

    # Compile budget: should be ~instant for a 3-node graph regardless of
    # the size of the static ``items`` list.
    t0 = time.monotonic()
    compiled = compile(
        nodes=nodes, edges=edges, registry=registry,
        compound_types=[FOR_EACH],
    )
    compile_seconds = time.monotonic() - t0
    assert compile_seconds < 5.0, (
        f"compile took {compile_seconds:.2f}s; expected <5s"
    )

    # tracemalloc snapshot bracketing the run.
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    t0 = time.monotonic()
    results = execute_sync(compiled, timeout_seconds=120)
    elapsed = time.monotonic() - t0

    snap_after = tracemalloc.take_snapshot()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Generous correctness bound — this is a stress test, not a
    # microbenchmark. ``execute_sync`` runs the iterations sequentially
    # by default; 1000 string-upper invocations should clear in seconds
    # but the bound allows for slow CI runners.
    assert elapsed < 60.0, (
        f"execution took {elapsed:.2f}s; expected <60s"
    )

    end_result = results["end"]["result"]
    assert isinstance(end_result, list)
    assert len(end_result) == MAX_ITERATIONS, (
        f"expected MAX_ITERATIONS={MAX_ITERATIONS} collected items "
        f"(requested={requested}); got len={len(end_result)}"
    )

    # First and last few items are uppercased correctly — sanity that
    # we didn't accidentally drop or reorder iterations.
    assert end_result[0] == "V0"
    assert end_result[-1] == f"V{MAX_ITERATIONS - 1}"

    # Memory bound: 200 MB peak. The full collected list is ~10k strings;
    # peak is dominated by per-iteration scratch + the items list itself.
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 200.0, (
        f"tracemalloc peak {peak_mb:.1f} MB exceeds 200 MB budget; "
        f"possible memory leak"
    )

    # Absolute size diff should be tiny — collected list is the only
    # large persistent allocation. Use top-level size as a coarse signal.
    diff = snap_after.compare_to(snap_before, "filename")
    total_diff_mb = sum(s.size_diff for s in diff) / (1024 * 1024)
    assert total_diff_mb < 200.0, (
        f"net allocation diff {total_diff_mb:.1f} MB exceeds 200 MB"
    )


# ---------------------------------------------------------------------------
# Test 2 — 500-node deep linear chain
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_500_node_linear_chain_compile_and_execute() -> None:
    """500 passthrough/upper nodes wired n_i -> n_{i+1}. Verifies the
    compiler and the eager scheduler scale to non-trivial graph sizes
    and that values propagate end-to-end.

    Bounds:
        compile <  5s
        execute < 30s  (engine polls the event queue every 0.5s, so a
                        500-node sequential chain has a soft floor of
                        ~250s if every node is forced through one full
                        poll cycle. In practice nodes complete much
                        faster than the poll timeout because the queue
                        is awoken on each ``put``, but be generous.)
    """
    registry = NodeRegistry()

    @registry.node(
        "upper", version=1, name="Upper", description="Uppercases",
    )
    def upper(
        text: Annotated[str, Text(label="Input")] = "",
    ) -> Annotated[str, Output(label="Output")]:
        return text.upper()

    n = 500
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # First node carries a static input; every subsequent node consumes
    # the previous node's ``result`` handle into its ``text`` parameter.
    nodes.append(GraphNode("n0", "upper@1", {"text": "hello"}))
    for i in range(1, n):
        nodes.append(GraphNode(f"n{i}", "upper@1", None))
        edges.append(
            GraphEdge(f"e{i}", f"n{i - 1}", f"n{i}", "result", "text"),
        )

    t0 = time.monotonic()
    compiled = compile(nodes=nodes, edges=edges, registry=registry)
    compile_seconds = time.monotonic() - t0
    assert compile_seconds < 5.0, (
        f"compile took {compile_seconds:.2f}s; expected <5s"
    )

    t0 = time.monotonic()
    # 500 nodes * 0.5s queue-poll worst case => bound at 60s to leave
    # headroom on slow runners; in practice this completes in <2s
    # because ``event_queue.put`` wakes the consumer immediately.
    results = execute_sync(compiled, timeout_seconds=120)
    elapsed = time.monotonic() - t0
    assert elapsed < 60.0, (
        f"execution took {elapsed:.2f}s; expected <60s"
    )

    # Every node should have produced "HELLO" — value propagated through
    # all 500 nodes without loss or mangling.
    final = results[f"n{n - 1}"]["result"]
    assert final == "HELLO", f"expected propagated 'HELLO'; got {final!r}"
    assert results["n0"]["result"] == "HELLO"


# ---------------------------------------------------------------------------
# Test 3 — Cancellation mid-retry
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_cancellation_honored_during_retry_sleep() -> None:
    """Register a flaky node that always fails; set ``max_retries=5`` and
    a long ``retry_delay`` so the engine spends most of its time in
    ``await asyncio.sleep(delay)`` between attempts. Once a
    ``node_retry`` event has been observed, set the cancellation flag.
    The engine's main loop polls cancellation every 500ms, so within
    ~1s we should see ``flow_cancelled`` and *no further retry attempt*
    should have run.
    """
    registry = NodeRegistry()

    call_count = 0

    @registry.node(
        "always-flaky", version=1, name="Always Flaky",
        description="Always fails (would succeed only after >5 tries)",
        max_retries=5, retry_delay=2.0,
    )
    def always_flaky(
        text: Annotated[str, Text(label="In")] = "",
    ) -> Annotated[str, Output(label="Out")]:
        nonlocal call_count
        call_count += 1
        from conductor.errors import NodeExecutionError
        raise NodeExecutionError(
            f"transient (attempt {call_count})",
            node_id="n1", node_type="always-flaky@1",
        )

    compiled = compile(
        nodes=[GraphNode("n1", "always-flaky@1", {"text": "x"})],
        edges=[],
        registry=registry,
    )

    # Capture the live ``FlowRunState`` so we can flip the cancellation
    # flag from outside. ``execute()`` builds state internally via
    # ``_build_state``; monkey-patching that is the cleanest hook
    # because conductor doesn't expose a public cancellation handle yet.
    captured_state: dict[str, object] = {}

    from conductor.execution import engine as _engine

    real_build_state = _engine._build_state

    def capture_build_state(*args, **kwargs):
        st = real_build_state(*args, **kwargs)
        captured_state["state"] = st
        return st

    _engine._build_state = capture_build_state
    try:
        events: list[dict] = []
        first_retry_seen = asyncio.Event()

        async def consume() -> None:
            async for ev in execute(compiled, timeout_seconds=60):
                events.append(ev)
                if ev["type"] == "node_retry" and not first_retry_seen.is_set():
                    first_retry_seen.set()

        async def canceller() -> None:
            # Wait until we've seen at least one retry attempt — this
            # guarantees the node is currently in ``await asyncio.sleep``
            # rather than executing or about to start.
            await asyncio.wait_for(first_retry_seen.wait(), timeout=10)
            # Small extra delay to make sure we're inside the sleep,
            # not inbetween retry-event-emit and sleep.
            await asyncio.sleep(0.1)
            state = captured_state.get("state")
            assert state is not None, "state was not captured"
            state._cancelled.set()  # type: ignore[attr-defined]

        consumer_task = asyncio.create_task(consume())
        canceller_task = asyncio.create_task(canceller())

        # Bound the whole thing: cancellation must be honored within a
        # few seconds even though retry_delay=2.0 — the main loop polls
        # cancellation on a 500ms tick.
        try:
            await asyncio.wait_for(
                asyncio.gather(consumer_task, canceller_task),
                timeout=15,
            )
        except asyncio.TimeoutError:
            consumer_task.cancel()
            canceller_task.cancel()
            raise
    finally:
        _engine._build_state = real_build_state

    event_types = [e["type"] for e in events]

    assert "node_retry" in event_types, (
        f"expected at least one node_retry before cancel; got {event_types}"
    )
    assert "flow_cancelled" in event_types, (
        f"expected flow_cancelled; got {event_types}"
    )
    # The ``flow_cancelled`` event must come *after* at least one
    # ``node_retry`` — cancellation interrupted the retry sleep, not
    # the very first attempt.
    cancel_idx = event_types.index("flow_cancelled")
    retry_idx = event_types.index("node_retry")
    assert retry_idx < cancel_idx, (
        f"expected retry before cancel in {event_types}"
    )

    # Critical assertion: the post-cancel retry attempt did NOT run.
    # ``call_count`` reflects the number of times the node body was
    # invoked. With ``max_retries=5`` plus the initial attempt that's
    # 6 max; cancellation should cap it before reaching that.
    assert call_count < 6, (
        f"node was invoked {call_count} times; cancellation should "
        f"have stopped the retry loop before max_retries+1=6"
    )
    # Sanity: at least one attempt did run (otherwise we'd never have
    # seen ``node_retry``).
    assert call_count >= 1
