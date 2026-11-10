"""A leg can save itself while it runs: ``stream(checkpoint_every=...)`` and the state it hands over."""

import asyncio
import threading
from typing import Annotated, Any

import pytest
from conductor import (
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    Param,
    Policy,
    Result,
    RunState,
    Series,
    node,
)
from conductor.events import Event, RunCheckpoint, RunEnd, UnitWaiting
from helpers import answer, registry_of


class Txt(DType, str):
    id = "checkpoint-txt"
    display_title = "Tekst"


Out = Annotated[Txt, Result(title="Svar")]
calls: dict[str, int] = {}
#: Set to let a call of ``Work`` on a line starting with ``hold`` return.
release = threading.Event()


def _finished_rows(state: RunState, node_id: str) -> list[tuple[int, ...]]:
    return [row for row, _ in state.nodes[node_id].finished_units or ()]


def called(what: str) -> None:
    calls[what] = calls.get(what, 0) + 1


@node("lines", title="Lines", description="d", category="test")
def lines_node() -> Annotated[Series[Txt], Result(title="Linjer")]:
    called("lines")
    return [Txt(line) for line in LINES]


@node("work", title="Work", description="d", category="test", policy=Policy(concurrency=8))
def work(line: Annotated[Txt, Param(title="L")]) -> Out | AskPerson:
    called(line)
    if line.startswith("hold"):
        release.wait(10)
    if line.startswith("ask"):
        return AskPerson(prompt="?")
    return Txt(line.upper())


@node("count", title="Count", description="d", category="test")
def count(lines: Annotated[Series[Txt], Param(title="L")]) -> Annotated[Txt, Result(title="N")]:
    called("count")
    return Txt(",".join(sorted(lines)))


LINES = ["a", "b", "hold", "c"]
REGISTRY = registry_of(lines_node, work, count)


def compiled() -> CompiledGraph:
    made = CompiledGraph.from_graph(Graph(nodes=(
        GraphNode(id="lines", type="lines", version=1),
        GraphNode(id="work", type="work", version=1, bindings={"line": From("lines.result")}),
        GraphNode(id="count", type="count", version=1, bindings={"lines": From("work.result")}),
    )), REGISTRY)
    assert made.is_runnable, made.problems
    return made


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch):
    calls.clear()
    release.clear()
    monkeypatch.setattr(f"{__name__}.LINES", ["a", "b", "hold", "c"])
    yield
    release.set()


def events_until(graph: CompiledGraph, stop: Any, **kwargs: Any) -> list[Event]:
    """The leg's events up to the first one ``stop`` accepts, or all of them; the leg then ends at once."""

    async def read() -> list[Event]:
        seen: list[Event] = []
        async with graph.stream(**kwargs) as events:
            async for event in events:
                seen.append(event)
                if stop(event):
                    break
        return seen

    return asyncio.run(read())


def all_but_hold_finished(event: Event) -> bool:
    return isinstance(event, RunCheckpoint) and len(_finished_rows(event.state, "work")) == 3


def test_a_stream_gives_no_checkpoint_unless_asked():
    release.set()
    assert not [e for e in events_until(compiled(), lambda e: False) if isinstance(e, RunCheckpoint)]


def test_every_unit_finishing_gives_a_checkpoint_at_zero_and_a_long_interval_gives_none():
    release.set()
    every_unit = events_until(compiled(), lambda e: False, checkpoint_every=0)
    states = [e.state for e in every_unit if isinstance(e, RunCheckpoint)]
    assert states and all(state.status == "running" and state.cause is None for state in states)
    assert [len(_finished_rows(state, "work")) for state in states] == sorted(len(_finished_rows(state, "work")) for state in states)

    rarely = events_until(compiled(), lambda e: False, checkpoint_every=3600)
    assert not [e for e in rarely if isinstance(e, RunCheckpoint)]


def test_no_checkpoint_right_before_the_end_which_says_the_same():
    release.set()
    seen = events_until(compiled(), lambda e: False, checkpoint_every=0)
    assert isinstance(seen[-1], RunEnd) and not isinstance(seen[-2], RunCheckpoint)


def test_a_checkpoint_does_not_change_when_the_run_goes_on():
    release.set()
    seen = events_until(compiled(), lambda e: False, checkpoint_every=0)
    first = next(e.state for e in seen if isinstance(e, RunCheckpoint))
    finished = {node_id: list(work.finished_units or ()) for node_id, work in first.nodes.items()}

    assert isinstance(seen[-1], RunEnd)
    assert {node_id: list(work.finished_units or ()) for node_id, work in first.nodes.items()} == finished and first.status == "running"


def test_a_checkpoint_taken_mid_fan_out_restored_from_json_finishes_the_run():
    graph = compiled()
    seen = events_until(graph, all_but_hold_finished, checkpoint_every=0)
    state = RunState.model_validate_json(seen[-1].state.model_dump_json())  # type: ignore[union-attr]
    assert state.status == "running" and len(_finished_rows(state, "work")) == 3
    assert calls["hold"] == 1

    release.set()
    calls_before = dict(calls)
    ended = compiled().run_sync(state)

    assert ended.status == "complete"
    assert calls["hold"] == calls_before["hold"] + 1, "a unit in flight runs again"
    assert calls["lines"] == 1 and all(calls[line] == 1 for line in ("a", "b", "c")), "a finished unit is not called again"
    assert calls["count"] == 1
    assert graph.outputs(ended)["count.result"] == "A,B,C,HOLD"


def test_outputs_refuses_a_checkpoint_and_results_reads_what_it_holds():
    graph = compiled()
    state = events_until(graph, all_but_hold_finished, checkpoint_every=0)[-1].state  # type: ignore[union-attr]

    with pytest.raises(InvalidCallError) as raised:
        graph.outputs(state)
    assert raised.value.errors()[0].type == "run_not_complete"
    assert raised.value.errors()[0].ctx["status"] == "running"
    assert sorted(graph.results(state)["work"]["result"]["values"]) == ["A", "B", "C"]


def test_a_checkpoint_carries_who_is_waiting_and_an_answer_goes_into_it(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(f"{__name__}.LINES", ["ask", "b", "hold"])
    graph = compiled()

    def asked_and_b_done(event: Event) -> bool:
        return isinstance(event, RunCheckpoint) and bool(event.state.waiting) and len(_finished_rows(event.state, "work")) == 1

    seen = events_until(graph, asked_and_b_done, checkpoint_every=0)
    assert any(isinstance(e, UnitWaiting) for e in seen)
    state = RunState.model_validate_json(seen[-1].state.model_dump_json())  # type: ignore[union-attr]
    (waiting,) = state.waiting

    answered = answer(compiled(), state, {waiting.id: {"result": "SVAR"}})
    assert answered.status == "running" and not answered.waiting

    release.set()
    ended = compiled().run_sync(answered)
    assert ended.status == "complete"
    assert calls["ask"] == 1, "an answered unit is not asked again"
    assert graph.outputs(ended)["count.result"] == "B,HOLD,SVAR"
