"""Eager scheduling and retry.

Independent branches run concurrently; a failing node is re-run under the
version's ``Policy`` or, when the version declares none, under the run's
``RetryConfig``; a validation failure is never retried.
"""

import time
from typing import Annotated

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.dtype import DType
from conductor.errors import FlowExecutionException
from conductor.execution.engine import execute, execute_sync
from conductor.execution.retry import RetryConfig
from conductor.node import NodeDefinition, Policy, version
from conductor.returns import Result
from conductor.widgets import Number as NumberWidget
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "eager-retry-test-text"
    title = "Text"


class Num(DType, float):
    id = "eager-retry-test-number"
    title = "Number"


In = Annotated[Txt, Textarea(title="In")]
Out = Annotated[Txt, Result(title="Out")]


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns its text"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        return text


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "Upper-cases its text"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        return Txt(text.upper())


class Slow(NodeDefinition):
    id = "slow"
    title = "Slow"
    description = "Sleeps 0.3s, then upper-cases its text"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        time.sleep(0.3)
        return Txt(text.upper())


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "Joins two texts with a plus"
    category = "test"

    def run(
        self,
        a: Annotated[Txt, Textarea(title="A")] = Txt(""),
        b: Annotated[Txt, Textarea(title="B")] = Txt(""),
    ) -> Out:
        return Txt(f"{a}+{b}")


class AlwaysFails(NodeDefinition):
    id = "always-fails"
    title = "Always fails"
    description = "Raises on every attempt"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        raise RuntimeError("boom")


def _registry(*node_classes: type[NodeDefinition]) -> NodeRegistry:
    reg = NodeRegistry()
    for node_cls in node_classes:
        reg.register(node_cls)
    return reg


class TestEagerScheduling:
    def test_independent_branches_run_in_parallel(self):
        """Two independent branches overlap instead of running one after the other."""
        # A(0.3s) -> C(0.3s) --+
        #                      +--> E
        # B(0.3s) -> D(0.3s) --+
        compiled = compile(
            nodes=[
                GraphNode("a", "slow", 1, {"text": "hello"}),
                GraphNode("b", "slow", 1, {"text": "world"}),
                GraphNode("c", "slow", 1, None),
                GraphNode("d", "slow", 1, None),
                GraphNode("e", "join", 1, None),
            ],
            edges=[
                GraphEdge("e1", "a", "c", "result", "text"),
                GraphEdge("e2", "b", "d", "result", "text"),
                GraphEdge("e3", "c", "e", "result", "a"),
                GraphEdge("e4", "d", "e", "result", "b"),
            ],
            registry=_registry(Slow, Join),
        )

        start = time.monotonic()
        results = execute_sync(compiled)
        elapsed = time.monotonic() - start

        # Sequential would be 5 * 0.3 = 1.5s; eager is A+B, C+D, E = ~0.9s.
        assert elapsed < 1.3, f"Took {elapsed:.2f}s — branches should run in parallel"
        assert results["e"]["result"] == "HELLO+WORLD"

    def test_linear_chain_still_works(self):
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo", 1, {"text": "hello"}),
                GraphNode("n2", "upper", 1, None),
                GraphNode("n3", "echo", 1, None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n2", "n3", "result", "text"),
            ],
            registry=_registry(Echo, Upper),
        )

        assert execute_sync(compiled)["n3"]["result"] == "HELLO"

    async def test_events_emitted_for_parallel_nodes(self):
        compiled = compile(
            nodes=[
                GraphNode("a", "echo", 1, {"text": "x"}),
                GraphNode("b", "echo", 1, {"text": "y"}),
            ],
            edges=[],
            registry=_registry(Echo),
        )

        events = [event async for event in execute(compiled)]

        types = [e["type"] for e in events]
        assert types.count("node_start") == 2
        assert types.count("node_complete") == 2
        assert "flow_complete" in types

    def test_single_node_works(self):
        compiled = compile(
            nodes=[GraphNode("n1", "echo", 1, {"text": "hi"})],
            edges=[],
            registry=_registry(Echo),
        )

        assert execute_sync(compiled)["n1"]["result"] == "hi"


class TestRetry:
    def test_global_retry_retries_on_failure(self):
        """The run-level ``RetryConfig`` re-runs a node that declares no policy."""
        calls = 0

        class Flaky(NodeDefinition):
            id = "flaky"
            title = "Flaky"
            description = "Fails twice, then succeeds"
            category = "test"

            def run(self, text: In = Txt("")) -> Out:
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise RuntimeError(f"Attempt {calls} failed")
                return Txt(f"ok:{text}")

        compiled = compile(
            nodes=[GraphNode("n1", "flaky", 1, {"text": "hello"})],
            edges=[],
            registry=_registry(Flaky),
        )

        results = execute_sync(compiled, retry=RetryConfig(max_retries=3, delay=0.05))
        assert results["n1"]["result"] == "ok:hello"
        assert calls == 3

    def test_global_retry_exhausted_raises(self):
        compiled = compile(
            nodes=[GraphNode("n1", "always-fails", 1, {"text": "hello"})],
            edges=[],
            registry=_registry(AlwaysFails),
        )

        with pytest.raises(FlowExecutionException):
            execute_sync(compiled, retry=RetryConfig(max_retries=2, delay=0.01))

    def test_node_level_retry_overrides_global(self):
        """The version's ``Policy`` wins over the run-level ``RetryConfig``."""
        calls = 0

        class Flaky(NodeDefinition):
            id = "flaky"
            title = "Flaky"
            description = "Fails twice, then succeeds"
            category = "test"

            @version(1, policy=Policy(retries=3, delay=0.01))
            def run(self, text: In = Txt("")) -> Out:
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise RuntimeError("not yet")
                return Txt("done")

        compiled = compile(
            nodes=[GraphNode("n1", "flaky", 1, {"text": "x"})],
            edges=[],
            registry=_registry(Flaky),
        )

        # The run says no retry, the version says 3 — the version wins.
        results = execute_sync(compiled, retry=RetryConfig(max_retries=0))
        assert results["n1"]["result"] == "done"
        assert calls == 3

    def test_no_retry_by_default(self):
        compiled = compile(
            nodes=[GraphNode("n1", "always-fails", 1, {"text": "x"})],
            edges=[],
            registry=_registry(AlwaysFails),
        )

        with pytest.raises(FlowExecutionException):
            execute_sync(compiled)

    def test_validation_errors_not_retried(self):
        """A value the declared type refuses fails before ``run`` and is never retried."""
        calls = 0

        class Typed(NodeDefinition):
            id = "typed"
            title = "Typed"
            description = "Needs a number"
            category = "test"

            @version(1, policy=Policy(retries=5, delay=0.01))
            def run(
                self, num: Annotated[Num, NumberWidget(title="Num")]
            ) -> Annotated[Num, Result(title="Out")]:
                nonlocal calls
                calls += 1
                return num

        compiled = compile(
            nodes=[GraphNode("n1", "typed", 1, {"num": "not-a-number"})],
            edges=[],
            registry=_registry(Typed),
        )

        with pytest.raises(FlowExecutionException):
            execute_sync(compiled)

        assert calls == 0

    async def test_retry_emits_events(self):
        calls = 0

        class Flaky(NodeDefinition):
            id = "flaky"
            title = "Flaky"
            description = "Fails once, then succeeds"
            category = "test"

            def run(self, text: In = Txt("")) -> Out:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("first try fails")
                return Txt("ok")

        compiled = compile(
            nodes=[GraphNode("n1", "flaky", 1, {"text": "x"})],
            edges=[],
            registry=_registry(Flaky),
        )

        events = [
            event
            async for event in execute(compiled, retry=RetryConfig(max_retries=2, delay=0.01))
        ]

        types = [e["type"] for e in events]
        assert "node_retry" in types
        assert "node_complete" in types
        assert "flow_complete" in types

        retry_event = next(e for e in events if e["type"] == "node_retry")
        assert retry_event["attempt"] == 1
        assert retry_event["node_id"] == "n1"


class TestRetryWithParallel:
    def test_flaky_node_in_parallel_branch(self):
        """A flaky node in one branch retries while the other branch completes."""
        calls = {"a": 0, "b": 0}

        class FlakyA(NodeDefinition):
            id = "flaky-a"
            title = "Flaky A"
            description = "Fails once, then succeeds"
            category = "test"

            @version(1, policy=Policy(retries=2, delay=0.01))
            def run(self, text: In = Txt("")) -> Out:
                calls["a"] += 1
                if calls["a"] == 1:
                    raise RuntimeError("first try")
                return Txt(f"A:{text}")

        class FastB(NodeDefinition):
            id = "fast-b"
            title = "Fast B"
            description = "Always works"
            category = "test"

            def run(self, text: In = Txt("")) -> Out:
                calls["b"] += 1
                return Txt(f"B:{text}")

        compiled = compile(
            nodes=[
                GraphNode("n1", "flaky-a", 1, {"text": "x"}),
                GraphNode("n2", "fast-b", 1, {"text": "y"}),
                GraphNode("n3", "join", 1, None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n3", "result", "a"),
                GraphEdge("e2", "n2", "n3", "result", "b"),
            ],
            registry=_registry(FlakyA, FastB, Join),
        )

        results = execute_sync(compiled)
        assert results["n3"]["result"] == "A:x+B:y"
        assert calls["a"] == 2  # retried once
        assert calls["b"] == 1  # no retry needed
