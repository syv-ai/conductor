"""Tests a node whose function runs another graph inside its call, as a host runs an agent's tools.

Conductor has no feature for this: the node compiles the inner graph once and runs it with the public API, passing
on the run's context. These tests pin that it works: the outer run completes and saves, the inner run gets the same
context object, a cancel event the host puts in the context stops both runs, a node with ``concurrency`` set may be
used in both graphs, and a retried outer call runs its inner graph again.

Each inner run is given a ``timeout``, so a run that waits forever ends ``timed_out`` and fails the test instead of
hanging it.
"""

import asyncio
from dataclasses import dataclass, field
from threading import Event as ThreadEvent
from typing import Any

import pytest
from conductor import (
    Context,
    ExternalError,
    Graph,
    NodeRegistry,
    Policy,
    Run,
    Series,
    make_graph,
    node,
)
from conductor.events import RunEnd

#: How many seconds an inner run may take. Past this it ends ``timed_out`` instead of waiting forever.
_BOUND = 10.0


@dataclass(frozen=True)
class Agent:
    """The host's context: a cancel event and a list each call adds its cost to."""

    cancel: ThreadEvent = field(default_factory=ThreadEvent)
    costs: list[str] = field(default_factory=list)
    seen: list[object] = field(default_factory=list)


@node("lookup", title="Lookup", description="For a test: the inner graph's tool.")
def lookup(ctx: Context[Agent], word: str) -> str:
    ctx.costs.append(word)
    ctx.seen.append(ctx)
    return word.upper()


def _registry(*definitions: Any) -> NodeRegistry[Agent]:
    registry = NodeRegistry(context_type=Agent)
    registry.add(*definitions)
    return registry


TOOL = Graph.compile(make_graph(lookup), _registry(lookup))


@node("agent", title="Agent", description="For a test: runs the tool once per word.")
def agent(ctx: Context[Agent], words: list[str]) -> list[str]:
    results = []
    for word in words:
        run = TOOL.run({"word": word}, context=ctx, timeout=_BOUND)
        assert run.status == "complete", run.cause
        results.append(run.outputs["result"])
    return results


def test_a_node_runs_another_graph_several_times_and_the_outer_run_saves_and_reads_back():
    graph = Graph.compile(make_graph(agent), _registry(agent))
    context = Agent()

    run = graph.run({"words": ["a", "b", "c"]}, context=context)

    assert run.status == "complete", run.cause
    assert run.outputs == {"result": ["A", "B", "C"]}
    assert Run.from_json(run.to_json(), graph).outputs == {"result": ["A", "B", "C"]}


def test_the_inner_run_gets_the_outer_runs_context_object_so_a_host_adds_its_cost_there():
    graph = Graph.compile(make_graph(agent), _registry(agent))
    context = Agent()

    graph.run({"words": ["a", "b"]}, context=context)

    assert context.costs == ["a", "b"]
    assert all(seen is context for seen in context.seen)


# -- a cancel in the context --------------------------------------------------------------------


@node("wait", title="Wait", description="For a test: says it started, then waits for the cancel.")
def wait(ctx: Context[Agent], word: str) -> str:
    ctx.costs.append("started")
    ctx.cancel.wait(_BOUND)
    return word


#: ``wait``, then ``lookup``. A cancel set while ``wait`` runs stops the run before ``lookup`` is called.
WAITING_TOOL = Graph.compile(make_graph(wait, lookup), _registry(wait, lookup))
INNER_ENDS: list[RunEnd] = []


async def _last(run: Run[Agent], context: Agent) -> RunEnd:
    async for event in run.stream(context=context, cancel=context.cancel, timeout=_BOUND):
        if isinstance(event, RunEnd):
            return event
    raise AssertionError("a stream ends with a RunEnd")


@node("cancellable_agent", title="Cancellable agent", description="For a test.")
def cancellable_agent(ctx: Context[Agent], word: str) -> str:
    end = asyncio.run(_last(WAITING_TOOL.with_inputs({"word": word}), ctx))
    INNER_ENDS.append(end)
    if end.status != "complete":
        raise ExternalError(f"The tool ended {end.status}.")
    return end.run.outputs["result"]


def test_a_cancel_event_in_the_context_stops_the_inner_run_and_the_outer_run_ends_cancelled():
    INNER_ENDS.clear()
    graph = Graph.compile(make_graph(cancellable_agent), _registry(cancellable_agent))
    context = Agent()

    async def go() -> RunEnd:
        events = graph.with_inputs({"word": "a"}).stream(context=context, cancel=context.cancel)
        async for event in events:
            if isinstance(event, RunEnd):
                return event
        raise AssertionError("a stream ends with a RunEnd")

    async def cancel_once_started() -> None:
        while not context.costs:
            await asyncio.sleep(0.01)
        context.cancel.set()

    async def both() -> RunEnd:
        end, _ = await asyncio.wait_for(asyncio.gather(go(), cancel_once_started()), _BOUND)
        return end

    end = asyncio.run(both())

    assert [inner.status for inner in INNER_ENDS] == ["cancelled"]
    assert context.costs == ["started"]  # lookup was never called
    # The outer call raised once the cancel was set, so the run ends cancelled, and the call has no outcome.
    assert end.status == "cancelled"
    assert end.cause is not None and end.cause.type == "run_cancelled"
    assert end.run.progress.nodes["cancellable_agent"].finished == 0


# -- a node with concurrency used in both graphs ------------------------------------------------


def _graphs(limited_policy: Policy, agent_policy: Policy) -> Graph[Agent]:
    """``words -> limited`` and ``words -> tool_agent``, where ``tool_agent`` runs a graph of ``limited``."""

    @node("limited", title="Limited", description="For a test.", policy=limited_policy)
    def limited(text: str) -> str:
        return text + "!"

    limited_tool = Graph.compile(make_graph(limited), _registry(limited))

    @node("tool_agent", title="Tool agent", description="For a test.", policy=agent_policy)
    def tool_agent(text: str) -> str:
        run = limited_tool.run({"text": text}, timeout=_BOUND)
        assert run.status == "complete", run.cause
        return run.outputs["result"]

    record = make_graph(
        words,
        limited,
        tool_agent,
        edges=[("words.result", "limited.text"), ("words.result", "tool_agent.text")],
        outputs={"limited": "limited.result", "tool": "tool_agent.result"},
    )
    return Graph.compile(record, _registry(words, limited, tool_agent))


@node("words", title="Words", description="For a test.")
def words(text: str) -> Series[str]:
    return text.split()


@pytest.mark.parametrize(
    ("limited_policy", "agent_policy"),
    [
        (Policy(concurrency=1), Policy()),
        (Policy(concurrency=1), Policy(concurrency=1)),
        (Policy(concurrency=1), Policy(concurrency=3)),
        (Policy(concurrency=2, attempt_timeout=_BOUND), Policy(concurrency=3)),
    ],
    ids=[
        "the agent on the run's thread",
        "both counted in the process",
        "the agent on threads",
        "both on threads",
    ],
)
def test_a_node_with_concurrency_used_in_the_outer_graph_and_in_the_inner_graph_does_not_deadlock(
    limited_policy: Policy, agent_policy: Policy
):
    graph = _graphs(limited_policy, agent_policy)
    assert not graph.problems, graph.problems

    run = graph.run({"text": "a b c d"}, timeout=_BOUND)

    assert run.status == "complete", run.cause
    assert run.outputs == {"limited": ["a!", "b!", "c!", "d!"], "tool": ["a!", "b!", "c!", "d!"]}


# -- a retried outer call -----------------------------------------------------------------------


@node(
    "flaky_agent",
    title="Flaky agent",
    description="For a test: fails once after running its tool.",
    policy=Policy(retries=1, retry_delay=0),
)
def flaky_agent(ctx: Context[Agent], word: str) -> str:
    run = TOOL.run({"word": word}, context=ctx, timeout=_BOUND)
    if len(ctx.costs) == 1:
        raise ExternalError("The service was busy.")
    return run.outputs["result"]


def test_a_retried_outer_call_runs_its_inner_graph_again():
    graph = Graph.compile(make_graph(flaky_agent), _registry(flaky_agent))
    context = Agent()

    run = graph.run({"word": "a"}, context=context)

    assert run.status == "complete", run.cause
    assert run.outputs == {"result": "A"}
    assert context.costs == ["a", "a"]
