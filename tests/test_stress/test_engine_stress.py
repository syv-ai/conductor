"""Engine stress tests.

These tests push the engine harder than the regular suite — a long node
chain, and cancellation during a retry sleep. They are slower than unit
tests (each can take several seconds); they carry the ``slow`` mark so CI
can opt out via ``-m "not slow"``.

Scenarios covered:

1. **500-node deep linear chain.** Compile and execution scale with node
   count and a value propagates end-to-end.
2. **Cancel mid-retry.** Setting the cancellation flag while a node is
   asleep between retry attempts honours the cancel: no further attempt
   runs and ``flow_cancelled`` is emitted.
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.errors import NodeExecutionError
from conductor.execution.engine import execute, execute_sync
from conductor.node import NodeDefinition, Policy, version
from conductor.returns import Result
from conductor.widgets import Textarea
from conductor_nodes.types import Text

Out = Annotated[Text, Result(title="Output")]


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "Uppercases its input"
    category = "test"

    def run(self, text: Annotated[Text, Textarea(title="Input")] = Text("")) -> Out:
        return Text(text.upper())


@pytest.mark.slow
def test_500_node_linear_chain_compile_and_execute() -> None:
    """500 upper nodes wired n_i -> n_{i+1}. The compiler and the eager
    scheduler scale to non-trivial graph sizes and the value propagates
    end-to-end.

    Bounds:
        compile <  5s
        execute < 60s  (the engine polls the event queue every 0.5s, but
                        the queue is woken on each ``put``, so in practice
                        the chain completes in a couple of seconds; the
                        bound leaves headroom for slow runners.)
    """
    registry = NodeRegistry()
    registry.register(Upper)

    n = 500
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # The first node carries a static input; every subsequent node takes
    # the previous node's ``result`` on its ``text`` parameter.
    nodes.append(GraphNode("n0", "upper", 1, {"text": "hello"}))
    for i in range(1, n):
        nodes.append(GraphNode(f"n{i}", "upper", 1, None))
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
    results = execute_sync(compiled, timeout_seconds=120)
    elapsed = time.monotonic() - t0
    assert elapsed < 60.0, (
        f"execution took {elapsed:.2f}s; expected <60s"
    )

    # Every node produced "HELLO" — the value passed through all 500
    # nodes without loss or mangling.
    final = results[f"n{n - 1}"]["result"]
    assert final == "HELLO", f"expected propagated 'HELLO'; got {final!r}"
    assert results["n0"]["result"] == "HELLO"


@pytest.mark.slow
async def test_cancellation_honored_during_retry_sleep() -> None:
    """A node that always fails, with ``Policy(retries=5, delay=2.0)`` so
    the engine spends most of its time in ``await asyncio.sleep(delay)``
    between attempts. Once a ``node_retry`` event has been observed, the
    cancellation flag is set. The engine's main loop polls cancellation
    every 500ms, so within ~1s ``flow_cancelled`` follows and no further
    attempt runs.
    """
    call_count = 0

    class AlwaysFlaky(NodeDefinition):
        id = "always-flaky"
        title = "Always Flaky"
        description = "Always fails (would succeed only after >5 tries)"
        category = "test"

        @version(1, policy=Policy(retries=5, delay=2.0))
        def run(self, text: Annotated[Text, Textarea(title="In")] = Text("")) -> Out:
            nonlocal call_count
            call_count += 1
            raise NodeExecutionError(
                f"transient (attempt {call_count})",
                node_id="n1", node_type="always-flaky",
            )

    registry = NodeRegistry()
    registry.register(AlwaysFlaky)

    compiled = compile(
        nodes=[GraphNode("n1", "always-flaky", 1, {"text": "x"})],
        edges=[],
        registry=registry,
    )

    # Capture the live ``FlowRunState`` so the cancellation flag can be
    # flipped from outside. ``execute()`` builds state internally via
    # ``_build_state``; patching that is the cleanest hook because the
    # engine exposes no public cancellation handle.
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
            # Wait until at least one retry attempt has been seen — this
            # guarantees the node is currently in ``await asyncio.sleep``
            # rather than executing or about to start.
            await asyncio.wait_for(first_retry_seen.wait(), timeout=10)
            # Small extra delay to make sure we're inside the sleep, not
            # between the retry event and the sleep.
            await asyncio.sleep(0.1)
            state = captured_state.get("state")
            assert state is not None, "state was not captured"
            state._cancelled.set()  # type: ignore[attr-defined]

        consumer_task = asyncio.create_task(consume())
        canceller_task = asyncio.create_task(canceller())

        # Bound the whole thing: cancellation must be honoured within a
        # few seconds even though the delay is 2.0s — the main loop polls
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
    # ``flow_cancelled`` comes *after* at least one ``node_retry`` —
    # cancellation interrupted the retry sleep, not the first attempt.
    cancel_idx = event_types.index("flow_cancelled")
    retry_idx = event_types.index("node_retry")
    assert retry_idx < cancel_idx, (
        f"expected retry before cancel in {event_types}"
    )

    # The post-cancel retry attempt did NOT run. ``call_count`` is the
    # number of times the node body was invoked: with 5 retries plus the
    # initial attempt that is 6 at most; cancellation caps it earlier.
    assert call_count < 6, (
        f"node was invoked {call_count} times; cancellation should "
        f"have stopped the retry loop before retries+1=6"
    )
    # At least one attempt did run (otherwise no ``node_retry``).
    assert call_count >= 1
