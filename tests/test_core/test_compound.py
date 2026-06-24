"""Phase 3: Compound nodes — for-each loop execution."""

import contextvars
from typing import Annotated

import pytest
from conductor.compound.for_each import FOR_EACH
from conductor.execution.engine import execute, execute_sync
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.widgets import ConnectionList, Output, Text


@pytest.fixture
def loop_registry(registry):
    """Registry with nodes needed for loop testing."""

    @registry.node("upper", version=1, name="Upper", description="Uppercases")
    def upper(
        text: Annotated[str, Text(label="Input")],
    ) -> Annotated[str, Output(label="Output")]:
        return text.upper()

    @registry.node(
        "for-each-start", version=1, name="For Each Start",
        description="Start of for-each loop",
    )
    def for_each_start(
        items: Annotated[list[str], ConnectionList(label="Items")],
    ) -> tuple[
        Annotated[str, Output(label="Item")],
        Annotated[int, Output(label="Index")],
    ]:
        # This function body is NOT called directly — ForEachNode handles iteration.
        # The signature defines the outputs available inside the loop body.
        raise NotImplementedError("Handled by compound node")

    @registry.node(
        "for-each-end", version=1, name="For Each End",
        description="End of for-each loop",
    )
    def for_each_end(
        item: Annotated[str, Text(label="Item")],
    ) -> Annotated[list[str], Output(label="Collected")]:
        raise NotImplementedError("Handled by compound node")

    return registry


class TestForEachSequential:
    def test_sequential_loop(self, loop_registry):
        """
        for-each-start([a, b, c]) -> upper -> for-each-end
        Should produce ["A", "B", "C"]
        """
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["a", "b", "c"]}),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),  # item -> upper
            GraphEdge("e2", "body", "end", "result", "item"),      # upper result -> end
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        results = execute_sync(compiled)
        # The end node should collect all iteration results
        end_result = results["end"]["result"]
        assert isinstance(end_result, list)
        assert set(end_result) == {"A", "B", "C"}


class TestForEachParallel:
    def test_parallel_loop(self, loop_registry):
        """Same as sequential but with parallel execution mode."""
        nodes = [
            GraphNode("start", "for-each-start@1", {
                "items": ["x", "y", "z"],
                "execution_mode": "Parallel",
            }),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        results = execute_sync(compiled)
        end_result = results["end"]["result"]
        assert isinstance(end_result, list)
        assert set(end_result) == {"X", "Y", "Z"}


class TestForEachEvents:
    async def test_loop_emits_progress_events(self, loop_registry):
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["a", "b"]}),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        events = []
        async for event in execute(compiled):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "node_progress" in event_types
        assert "flow_complete" in event_types

    async def test_parallel_emits_incremental_progress(self, loop_registry):
        """Parallel mode must emit one ``node_progress`` per finished item
        (1/N, 2/N, … N/N), not a single terminal N/N — so the host can
        show a live counter while items complete out of order."""
        nodes = [
            GraphNode("start", "for-each-start@1", {
                "items": ["a", "b", "c"],
                "execution_mode": "Parallel",
            }),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges, registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        currents = [
            e["current"]
            for e in [ev async for ev in execute(compiled)]
            if e["type"] == "node_progress"
        ]
        # One event per item, counting up to the total — not a lone "3".
        assert currents == [1, 2, 3]

    async def test_progress_streams_live_not_buffered(self, registry):
        """A running compound node's events must reach the host *during*
        execution, not in a burst once the loop finishes.

        Construction makes liveness the only way to pass: the second
        iteration's body blocks on a gate that the consumer only opens
        after it has *received* the first iteration's ``node_progress``.
        Under the old buffered behaviour the host sees nothing until the
        whole compound completes, so the gate is never opened, the body
        raises, and ``flow_complete`` never arrives (the wait below would
        otherwise hang — hence the hard timeout).
        """
        import asyncio
        import threading
        from typing import Annotated

        from conductor.widgets import ConnectionList, Output, Text

        gate = threading.Event()

        @registry.node("gated", version=1, name="Gated", description="gates iter 2")
        def gated(
            text: Annotated[str, Text(label="Input")],
            index: Annotated[int, Text(label="Index")],
        ) -> Annotated[str, Output(label="Output")]:
            if index == 2 and not gate.wait(timeout=5.0):
                raise RuntimeError("gate never opened — progress was buffered")
            return text.upper()

        @registry.node(
            "for-each-start", version=1, name="Start", description="loop start",
        )
        def for_each_start(
            items: Annotated[list[str], ConnectionList(label="Items")],
        ) -> tuple[
            Annotated[str, Output(label="Item")],
            Annotated[int, Output(label="Index")],
        ]:
            raise NotImplementedError("Handled by compound node")

        @registry.node(
            "for-each-end", version=1, name="End", description="loop end",
        )
        def for_each_end(
            item: Annotated[str, Text(label="Item")],
        ) -> Annotated[list[str], Output(label="Collected")]:
            raise NotImplementedError("Handled by compound node")

        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["a", "b"]}),
            GraphNode("body", "gated@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e0", "start", "body", "output_1", "text"),   # item
            GraphEdge("e1", "start", "body", "output_2", "index"),  # 1-based index
            GraphEdge("e2", "body", "end", "result", "item"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges, registry=registry, compound_types=[FOR_EACH],
        )

        async def run() -> list:
            collected = []
            async for event in execute(compiled):
                collected.append(event)
                if event["type"] == "node_progress" and event.get("current") == 1:
                    gate.set()  # open iter 2 only after seeing iter-1 progress live
            return collected

        events = await asyncio.wait_for(run(), timeout=10.0)

        types = [e["type"] for e in events]
        assert "flow_complete" in types
        assert "node_error" not in types
        # The gate was opened, which only happens if iter-1 progress arrived
        # live (mid-run) rather than after the loop finished.
        assert gate.is_set()
        assert events[-1]["type"] == "flow_complete"
        assert set(events[-1]["results"]["end"]["result"]) == {"A", "B"}


class TestForEachBodyConcurrency:
    """The loop body is DAG-scheduled: independent nodes run concurrently,
    dependent nodes stay ordered, and total concurrency is globally capped."""

    async def test_independent_body_nodes_run_in_parallel(self, registry):
        """Two body nodes that both depend only on the loop item must run
        concurrently. Construction makes this deterministic: each waits on a
        2-party barrier, so the loop only completes if they overlap — a
        sequential body would block the first node forever (barrier timeout
        -> error, no flow_complete)."""
        import threading
        from typing import Annotated

        from conductor.widgets import ConnectionList, Output, Text

        barrier = threading.Barrier(2, timeout=5.0)

        @registry.node("a", version=1, name="A", description="a")
        def a(item: Annotated[str, Text(label="i")]) -> Annotated[str, Output(label="o")]:
            barrier.wait()
            return item.upper()

        @registry.node("b", version=1, name="B", description="b")
        def b(item: Annotated[str, Text(label="i")]) -> Annotated[str, Output(label="o")]:
            barrier.wait()
            return item

        @registry.node("for-each-start", version=1, name="S", description="s")
        def fes(items: Annotated[list[str], ConnectionList(label="x")]) -> tuple[
            Annotated[str, Output(label="Item")], Annotated[int, Output(label="Index")]
        ]:
            raise NotImplementedError

        @registry.node("for-each-end", version=1, name="E", description="e")
        def fee(
            ia: Annotated[str, Text(label="ia")], ib: Annotated[str, Text(label="ib")]
        ) -> Annotated[list[str], Output(label="c")]:
            raise NotImplementedError

        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["x"]}),
            GraphNode("na", "a@1", None),
            GraphNode("nb", "b@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "na", "output_1", "item"),
            GraphEdge("e2", "start", "nb", "output_1", "item"),  # both on the item
            GraphEdge("e3", "na", "end", "result", "ia"),
            GraphEdge("e4", "nb", "end", "result", "ib"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges, registry=registry, compound_types=[FOR_EACH]
        )
        types = [e["type"] for e in [ev async for ev in execute(compiled)]]
        assert "flow_complete" in types
        assert "node_error" not in types

    def test_dependent_body_nodes_stay_sequential(self, registry):
        """When one body node feeds another, the consumer must run after the
        producer — proven by the consumer receiving the producer's value."""
        from typing import Annotated

        from conductor.widgets import ConnectionList, Output, Text

        @registry.node("up", version=1, name="Up", description="up")
        def up(item: Annotated[str, Text(label="i")]) -> Annotated[str, Output(label="o")]:
            return item.upper()

        @registry.node("bang", version=1, name="Bang", description="bang")
        def bang(x: Annotated[str, Text(label="i")]) -> Annotated[str, Output(label="o")]:
            return f"{x}!"

        @registry.node("for-each-start", version=1, name="S", description="s")
        def fes(items: Annotated[list[str], ConnectionList(label="x")]) -> tuple[
            Annotated[str, Output(label="Item")], Annotated[int, Output(label="Index")]
        ]:
            raise NotImplementedError

        @registry.node("for-each-end", version=1, name="E", description="e")
        def fee(item: Annotated[str, Text(label="item")]) -> Annotated[list[str], Output(label="c")]:
            raise NotImplementedError

        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["x", "y"]}),
            GraphNode("nu", "up@1", None),
            GraphNode("nb", "bang@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "nu", "output_1", "item"),
            GraphEdge("e2", "nu", "nb", "result", "x"),  # bang depends on up
            GraphEdge("e3", "nb", "end", "result", "item"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges, registry=registry, compound_types=[FOR_EACH]
        )
        results = execute_sync(compiled)
        # "X!" / "Y!" prove up ran before bang in every iteration.
        assert set(results["end"]["result"]) == {"X!", "Y!"}

    async def test_body_concurrency_is_globally_capped(self, registry):
        """Across all iterations of a Parallel loop, the number of body
        nodes dispatching at once never exceeds the global cap, even though
        items x body-nodes far exceeds it."""
        import threading
        import time
        from typing import Annotated

        from conductor.compound import for_each as fe
        from conductor.widgets import ConnectionList, Output, Text

        lock = threading.Lock()
        state = {"cur": 0, "max": 0}

        def tick():
            with lock:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            time.sleep(0.05)
            with lock:
                state["cur"] -= 1

        @registry.node("a", version=1, name="A", description="a")
        def a(item: Annotated[str, Text(label="i")]) -> Annotated[str, Output(label="o")]:
            tick()
            return item

        @registry.node("b", version=1, name="B", description="b")
        def b(item: Annotated[str, Text(label="i")]) -> Annotated[str, Output(label="o")]:
            tick()
            return item

        @registry.node("for-each-start", version=1, name="S", description="s")
        def fes(items: Annotated[list[str], ConnectionList(label="x")]) -> tuple[
            Annotated[str, Output(label="Item")], Annotated[int, Output(label="Index")]
        ]:
            raise NotImplementedError

        @registry.node("for-each-end", version=1, name="E", description="e")
        def fee(
            ia: Annotated[str, Text(label="ia")], ib: Annotated[str, Text(label="ib")]
        ) -> Annotated[list[str], Output(label="c")]:
            raise NotImplementedError

        items = [f"i{n}" for n in range(10)]  # 10 items x 2 nodes = 20 potential
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": items, "execution_mode": "Parallel"}),
            GraphNode("na", "a@1", None),
            GraphNode("nb", "b@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "na", "output_1", "item"),
            GraphEdge("e2", "start", "nb", "output_1", "item"),
            GraphEdge("e3", "na", "end", "result", "ia"),
            GraphEdge("e4", "nb", "end", "result", "ib"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges, registry=registry, compound_types=[FOR_EACH]
        )
        import asyncio

        await asyncio.wait_for(
            asyncio.to_thread(execute_sync, compiled), timeout=20.0
        )
        # Hard guarantee from the shared semaphore.
        assert state["max"] <= fe._BODY_CONCURRENCY
        # And concurrency actually happened (independent nodes ran together).
        assert state["max"] >= 2


class TestForEachEmpty:
    def test_downstream_of_for_each_end_depends_on_region_start(
        self, loop_registry,
    ):
        """Regression: a node connected to ``for-each-end`` must list the
        for-each region's *start* as its dependency in the scheduler's
        dep graph, not the end gate itself.

        ``for-each-end`` is a managed (compound-internal) node, so the
        scheduler removes it from the ``schedulable`` set. Without
        managed-source remapping in ``_build_dep_graph``, the
        dependency from the downstream node to ``for-each-end``
        evaporates — the downstream's in-degree falls to 0 and it
        fires immediately at flow-start, before the loop has produced
        anything. (The bug surfaces probabilistically at runtime
        depending on thread-pool ordering; this test pins the dep
        graph shape directly so the fix can't silently regress.)
        """
        from conductor.execution.engine import _build_dep_graph

        @loop_registry.node(
            "consume-list", version=1, name="Consume",
            description="Read the collected list and report its length.",
        )
        def consume_list(
            items: Annotated[list[str], Text(label="Items")],
        ) -> Annotated[int, Output(label="Count")]:
            return len(items)

        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["a", "b", "c"]}),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
            GraphNode("after", "consume-list@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
            GraphEdge("e3", "end", "after", "result", "items"),
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        deps, _dependents = _build_dep_graph(compiled)
        # ``after`` depends on the compound's start (the only schedulable
        # node in the for-each region), not on the end gate which is
        # managed and never enters the ready queue.
        assert deps.get("after") == {"start"}
        # And the dependents map flows back the same way.
        _deps, dependents = _build_dep_graph(compiled)
        assert "after" in dependents.get("start", set())

    def test_empty_items_produces_empty_result(self, loop_registry):
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": []}),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        results = execute_sync(compiled)
        end_result = results["end"]["result"]
        assert isinstance(end_result, list)
        assert len(end_result) == 0


def test_skipped_filtered_from_for_each_end(registry):
    """An if-else branch inside a for-each body returns SKIPPED for some
    iterations. Those skipped values must NOT appear in the end-collected
    list — they are dropped just like the multi-edge aggregation in
    :mod:`conductor.execution.resolver` already does for SKIPPED values.

    Without this, the collected list would contain interleaved SKIPPED
    sentinels and real values, silently diverging from the resolver's
    behaviour everywhere else.

    The body uses a node that returns the ``SKIPPED`` sentinel directly
    when the input doesn't match, simulating an if-else node whose
    inactive branch propagates SKIPPED downstream. (The decision node's
    edge-level skip propagation only happens in the engine's main
    scheduler — not inside compound subgraph iteration — so the
    sentinel-return path is the cleaner signal here.)
    """
    import conductor_nodes
    from conductor._sentinel import SKIPPED
    from conductor.compound.for_each import FOR_EACH
    from conductor.execution.engine import execute_sync
    from conductor.graph.compiler import compile
    from conductor.graph.model import GraphEdge, GraphNode
    from conductor.widgets import Output, Text

    conductor_nodes.register_all(registry)

    # If-else stand-in: pass single-character values through (uppercased),
    # drop everything else by returning the SKIPPED sentinel.
    @registry.node(
        "skip_long", version=1, name="Skip Long",
        description="returns SKIPPED for long inputs",
    )
    def skip_long(
        text: Annotated[str, Text(label="t")] = "",
    ) -> Annotated[str, Output(label="r")]:
        if len(text) <= 1:
            return text.upper()
        return SKIPPED  # type: ignore[return-value]

    nodes = [
        GraphNode(
            "fe_start", "for-each-start@1",
            {"items": ["a", "bb", "c", "dd"]},
        ),
        GraphNode("body", "skip_long@1", None),
        GraphNode("fe_end", "for-each-end@1", None),
    ]
    edges = [
        GraphEdge("e1", "fe_start", "body", "output_1", "text"),
        GraphEdge("e2", "body", "fe_end", "result", "item"),
    ]

    compiled = compile(
        nodes=nodes, edges=edges,
        registry=registry, compound_types=[FOR_EACH],
    )

    results = execute_sync(compiled)
    end_result = results["fe_end"]["result"]
    # SKIPPED iterations dropped — only the single-char items contribute.
    assert end_result == ["A", "C"], (
        f"expected SKIPPED iterations filtered; got {end_result!r}"
    )


def test_multi_source_zip_emits_truncation_warning(loop_registry):
    """Multi-source for-each with mismatched lengths emits a
    runtime_warning event before iteration starts."""
    import asyncio

    from conductor.compound.for_each import FOR_EACH
    from conductor.execution.engine import execute
    from conductor.graph.compiler import compile
    from conductor.graph.model import GraphEdge, GraphNode

    # Three sources of length 4, 3, 5 respectively. min_len=3 — the
    # shorter "src_b" source determines the iteration count.
    nodes = [
        GraphNode("src_a", "upper@1", {"text": ""}),  # body never runs; data static below
        GraphNode("src_b", "upper@1", {"text": ""}),
        GraphNode("src_c", "upper@1", {"text": ""}),
        GraphNode("start", "for-each-start@1", None),
        GraphNode("body", "upper@1", None),
        GraphNode("end", "for-each-end@1", None),
    ]
    # We can't easily get list-typed sources from upper@1 (it returns a
    # str). The simplest approach: use static dict input on the start
    # node directly. The for-each ConnectionList resolver wraps a dict
    # input as {label: value} when multiple sources are wired; but a
    # static dict on ``items`` is also accepted. We use the dict shape
    # directly since static data flows past the resolver.
    nodes = [
        GraphNode("start", "for-each-start@1", {
            "items": {
                "src_a": ["a1", "a2", "a3", "a4"],   # 4
                "src_b": ["b1", "b2", "b3"],         # 3 (min)
                "src_c": ["c1", "c2", "c3", "c4", "c5"],  # 5
            },
        }),
        GraphNode("body", "upper@1", None),
        GraphNode("end", "for-each-end@1", None),
    ]
    edges = [
        GraphEdge("e1", "start", "body", "output_1", "text"),
        GraphEdge("e2", "body", "end", "result", "item"),
    ]

    compiled = compile(
        nodes=nodes, edges=edges,
        registry=loop_registry, compound_types=[FOR_EACH],
    )

    events = []

    async def go():
        async for ev in execute(compiled):
            events.append(ev)

    asyncio.run(go())

    warnings = [e for e in events if e.get("type") == "runtime_warning"]
    assert len(warnings) == 1, (
        f"expected exactly one runtime_warning; got {warnings!r}"
    )
    w = warnings[0]
    assert w["warning"] == "for_each_zip_truncation"
    assert w["payload"]["min_len"] == 3
    assert w["payload"]["source_lengths"] == {
        "src_a": 4, "src_b": 3, "src_c": 5,
    }
    # Sanity: iteration only ran 3 times (min_len) — body collected 3 items.
    end_result = events[-1]
    assert end_result["type"] == "flow_complete"
    assert len(end_result["results"]["end"]["result"]) == 3


def test_for_each_recursion_capped(loop_registry):
    """Nested for-each beyond ``_MAX_FOR_EACH_DEPTH`` raises a clean error.

    True nested for-each regions in a single compiled flow aren't a
    pattern the v1 region-discovery algorithm supports cleanly (BFS
    from each start matches the first end it sees, so two same-type
    regions inside one flow can't be distinguished). The cap is still
    a useful guardrail: the depth counter on ``state.context``
    increments on every for-each entry, so a body node that recursively
    invokes another for-each (e.g. a subprocess that contains a
    for-each, called from inside a for-each body) will trip the cap.

    This test verifies the cap by directly exercising the depth
    counter on a real ``ForEachNode.execute`` call: pre-set the
    counter to ``cap`` and confirm the next entry raises with a clean
    NodeError. Then pre-set to ``cap - 1`` and confirm it executes.
    """
    from conductor.compound.for_each import (
        _FOR_EACH_DEPTH_KEY,
        _MAX_FOR_EACH_DEPTH,
        FOR_EACH,
    )
    from conductor.errors import FlowExecutionError, NodeExecutionError
    from conductor.execution.engine import execute_sync
    from conductor.graph.compiler import compile
    from conductor.graph.model import GraphEdge, GraphNode

    nodes = [
        GraphNode("start", "for-each-start@1", {"items": ["a"]}),
        GraphNode("body", "upper@1", None),
        GraphNode("end", "for-each-end@1", None),
    ]
    edges = [
        GraphEdge("e1", "start", "body", "output_1", "text"),
        GraphEdge("e2", "body", "end", "result", "item"),
    ]
    compiled = compile(
        nodes=nodes, edges=edges,
        registry=loop_registry, compound_types=[FOR_EACH],
    )

    # Depth cap-1 (i.e. one below the cap): the for-each invocation
    # bumps depth to cap and runs cleanly.
    results = execute_sync(
        compiled,
        context={_FOR_EACH_DEPTH_KEY: _MAX_FOR_EACH_DEPTH - 1},
    )
    assert results["end"]["result"] == ["A"]

    # Depth cap (i.e. exactly at the cap): the next for-each entry
    # would push depth to cap+1 and must raise with a useful message.
    with pytest.raises((FlowExecutionError, NodeExecutionError)) as exc_info:
        execute_sync(
            compiled,
            context={_FOR_EACH_DEPTH_KEY: _MAX_FOR_EACH_DEPTH},
        )
    msg = str(exc_info.value).lower()
    assert "depth" in msg, f"expected 'depth' in error; got {exc_info.value!r}"


# Request-scoped state hosts attach via ContextVar (auth/user context, DB
# handles, tracing spans). The engine propagates these into node threads via
# asyncio.to_thread; the for-each pools must do the same.
_probe_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "for_each_test_probe", default=None
)


@pytest.fixture
def ctx_loop_registry(loop_registry):
    """``loop_registry`` plus a body node that echoes ``_probe_var`` — the
    caller-set context. Returns the sentinel ``"MISSING"`` when the loop
    dropped the context across the thread boundary."""

    @loop_registry.node(
        "ctx-reader", version=1, name="Ctx Reader",
        description="Echoes a contextvar set by the caller",
    )
    def ctx_reader(
        text: Annotated[str, Text(label="Input")],
    ) -> Annotated[str, Output(label="Output")]:
        return _probe_var.get() or "MISSING"

    return loop_registry


class TestForEachContextPropagation:
    """A loop body must see the caller's contextvars in every execution
    path. Regression test for context loss in the for-each thread pools:
    body nodes ran in raw ``ThreadPoolExecutor`` workers that don't inherit
    contextvars, so any node reading request-scoped state crashed — but only
    inside a loop, and only in the threaded paths (Parallel iteration, or a
    Sequential body level with >1 independent node)."""

    def test_parallel_iteration_preserves_context(self, ctx_loop_registry):
        """Parallel mode submits each iteration to a thread pool."""
        nodes = [
            GraphNode("start", "for-each-start@1", {
                "items": ["a", "b", "c"], "execution_mode": "Parallel",
            }),
            GraphNode("body", "ctx-reader@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges,
            registry=ctx_loop_registry, compound_types=[FOR_EACH],
        )

        token = _probe_var.set("ctx-value")
        try:
            results = execute_sync(compiled)
        finally:
            _probe_var.reset(token)

        assert results["end"]["result"] == ["ctx-value", "ctx-value", "ctx-value"]

    def test_sequential_parallel_body_level_preserves_context(
        self, ctx_loop_registry
    ):
        """Even in Sequential mode, a body *level* with >1 independent node
        runs in a thread pool (``_execute_subgraph``). Two independent
        readers fed from the start sit in the same level."""
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["x"]}),
            GraphNode("a", "ctx-reader@1", None),
            GraphNode("b", "ctx-reader@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "a", "output_1", "text"),
            GraphEdge("e2", "start", "b", "output_1", "text"),
            GraphEdge("e3", "a", "end", "result", "item"),
            GraphEdge("e4", "b", "end", "result", "item"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges,
            registry=ctx_loop_registry, compound_types=[FOR_EACH],
        )

        token = _probe_var.set("ctx-value")
        try:
            results = execute_sync(compiled)
        finally:
            _probe_var.reset(token)

        # Two wired sources → two collected slots; both must echo the context.
        assert results["end"]["output_1"] == ["ctx-value"]
        assert results["end"]["output_2"] == ["ctx-value"]
