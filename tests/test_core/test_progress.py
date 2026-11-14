"""``run.progress``, and ``run.resume(only_costs_nothing=True)``, which calls only the nodes that cost nothing.

The graph here is ``upload -> read -> approve``: ``upload`` costs nothing and returns the documents it is given,
``read`` finds the invoices in a document (it stands for a call that costs something), and ``approve`` asks a
person about each invoice.
"""

import asyncio
from typing import Any

import pytest
from conductor import (
    Ask,
    CallCount,
    Context,
    Graph,
    NodeDeclarationError,
    NodeError,
    Policy,
    Run,
    RunProgress,
    Series,
    make_graph,
    node,
    subgraph,
)

from tests.test_core.nodes import registry_of

FREE = Policy(costs_nothing=True)
CALLS: list[str] = []


@node("upload", title="Upload", description="For a test.", policy=FREE)
def upload(files: list[str]) -> Series[str]:
    CALLS.append("upload")
    return files


@node("read", title="Read", description="For a test.")
def read(document: str) -> Series[str]:
    CALLS.append("read")
    return [f"{document}#{index}" for index in range(int(document[-1]))]


@node("approve", title="Approve", description="For a test.", asks=True)
def approve(invoice: str) -> bool:
    raise Ask(f"Approve {invoice}?")


def _graph(*definitions: Any, edges: list[tuple[str, str]]) -> Graph:
    graph = Graph.compile(make_graph(*definitions, edges=edges), registry_of(*definitions))
    assert not graph.problems
    return graph


def _invoices() -> Graph:
    return _graph(upload, read, approve, edges=[("upload.result", "read.document"), ("read.result", "approve.invoice")])


def _events(run: Run, **options: Any) -> list[Any]:
    async def go() -> list[Any]:
        return [event async for event in run.stream(**options)]

    return asyncio.run(go())


def test_a_run_nothing_was_called_for_knows_only_its_first_node() -> None:
    progress = _invoices().with_inputs(files=["d2", "d1"]).progress

    assert progress.nodes == {
        "upload": CallCount(finished=0, waiting=0, known=1, total=1),
        "read": CallCount(finished=0, waiting=0, known=0, total=None),
        "approve": CallCount(finished=0, waiting=0, known=0, total=None),
    }
    assert (progress.finished, progress.known, progress.total) == (0, 1, None)


def test_only_costs_nothing_calls_the_free_nodes_and_leaves_the_run_ready() -> None:
    CALLS.clear()
    run = _invoices().with_inputs(files=["d2", "d1"]).resume(only_costs_nothing=True)

    assert CALLS == ["upload"]
    assert run.status == "ready"
    assert run.progress.nodes["read"] == CallCount(finished=0, waiting=0, known=2, total=2)
    assert run.progress.nodes["approve"].total is None


def test_the_total_is_a_floor_then_exact_with_the_waiting_calls_counted() -> None:
    run = _invoices().with_inputs(files=["d2", "d1"]).resume(only_costs_nothing=True).resume()

    assert run.status == "waiting"
    assert run.progress.nodes["approve"] == CallCount(finished=0, waiting=3, known=3, total=3)
    assert (run.progress.finished, run.progress.waiting, run.progress.total) == (3, 3, 6)
    answered = run.answer(run.waiting[0], True).resume()
    assert answered.progress.nodes["approve"] == CallCount(finished=1, waiting=2, known=3, total=3)


def test_known_grows_while_a_series_above_is_missing_somewhere() -> None:
    run = _invoices().with_inputs(files=["d2", "d1"]).resume(only_costs_nothing=True)
    finished = run.resume()
    run._outcomes[("read",)][(0,)] = finished._outcomes[("read",)][(0,)]  # as a checkpoint has it: one document read

    assert run.progress.nodes["approve"] == CallCount(finished=0, waiting=0, known=2, total=None)


def test_a_run_read_back_has_the_same_progress_and_saves_none() -> None:
    graph = _invoices()
    run = graph.with_inputs(files=["d2", "d1"]).resume()

    assert "progress" not in run.to_json()
    assert Run.from_json(run.to_json(), graph).progress == run.progress


def test_a_full_resume_after_it_repeats_no_call_and_no_event_of_the_free_nodes() -> None:
    CALLS.clear()
    graph = _graph(upload, read, edges=[("upload.result", "read.document")])
    estimated = graph.with_inputs(files=["d2"]).resume(only_costs_nothing=True)
    events = _events(estimated)

    assert {e.loc for e in events if hasattr(e, "loc")} == {("read",)}
    assert CALLS == ["upload", "read"]
    assert events[-1].run.status == "complete"


def test_a_stream_calls_only_the_free_nodes_as_a_resume_does() -> None:
    CALLS.clear()
    events = _events(_invoices().with_inputs(files=["d2", "d1"]), only_costs_nothing=True)

    assert CALLS == ["upload"]
    assert {e.loc for e in events if hasattr(e, "loc")} == {("upload",)}
    assert events[-1].run.status == "ready"
    assert events[-1].run.progress.nodes["read"] == CallCount(finished=0, waiting=0, known=2, total=2)


@pytest.mark.parametrize(("raised", "status", "waiting"), [(Ask("Which files?"), "waiting", 1), (NodeError("No files."), "failed", 0)])
def test_a_free_node_that_asks_or_fails_does_so_with_only_costs_nothing_as_in_a_full_resume(raised: Exception, status: str, waiting: int) -> None:
    @node("pick", title="Pick", description="For a test.", policy=FREE, asks=True)
    def pick(files: list[str]) -> Series[str]:
        raise raised

    graph = _graph(pick, read, edges=[("pick.result", "read.document")])
    estimated = graph.with_inputs(files=["d2"]).resume(only_costs_nothing=True)

    assert estimated.status == status
    assert estimated.progress.nodes["pick"] == CallCount(finished=0, waiting=waiting, known=1, total=1)
    assert estimated.resume().status == status
    full, free = (_events(graph.with_inputs(files=["d2"]), **options) for options in ({}, {"only_costs_nothing": True}))
    assert [event for event in free if hasattr(event, "loc")] == [event for event in full if hasattr(event, "loc")]


def test_a_graph_of_free_nodes_only_is_complete() -> None:
    graph = _graph(upload, edges=[])

    assert graph.with_inputs(files=["a"]).resume(only_costs_nothing=True).status == "complete"


def test_the_item_limit_stops_it_as_it_stops_a_resume() -> None:
    run = _invoices().with_inputs(files=["d2", "d1", "d3"], max_items=2).resume(only_costs_nothing=True)

    assert run.status == "stopped"
    assert run.progress.nodes["read"].total == 3


def test_a_subgraph_node_counts_the_calls_of_the_nodes_inside_it() -> None:
    @node("double", title="Double", description="For a test.")
    def double(value: int) -> int:
        return value * 2

    @node("numbers", title="Numbers", description="For a test.", policy=FREE)
    def numbers(count: int) -> Series[int]:
        return list(range(count))

    twice = subgraph("twice", title="Twice", description="For a test.")(make_graph(double, double))
    graph = Graph.compile(make_graph(numbers, twice, edges=[("numbers.result", "twice.value")]), registry_of(numbers, twice, double))
    progress = graph.with_inputs(count=3).resume(only_costs_nothing=True).progress

    assert list(progress.nodes) == ["numbers", "twice/double-1", "twice/double-2", "twice"]
    assert progress.nodes["twice"] == CallCount(finished=0, waiting=0, known=6, total=6)
    assert (progress.known, progress.total) == (7, 7)


def test_progress_is_json_a_host_reads_back() -> None:
    progress = _invoices().with_inputs(files=["d1"]).resume(only_costs_nothing=True).progress

    assert RunProgress.model_validate_json(progress.model_dump_json()) == progress


def test_a_free_node_is_refused_the_run_context() -> None:
    class HostContext:
        pass

    with pytest.raises(NodeDeclarationError, match="costs_nothing") as raised:
        @node("sign", title="Sign", description="For a test.", policy=FREE)
        def sign(ctx: Context[HostContext], text: str) -> str:
            return text

    assert raised.value.problems[0].type == "costs_nothing_takes_context"
