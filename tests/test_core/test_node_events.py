"""The events a stream says about each node, in the order it says them.

Each test streams a small graph and reads the events about its nodes. A node on the stream's own thread is called
in a fixed order, so the events come in a fixed order; where a test needs a call still running while others return,
that call runs on a thread of its own and waits for the test to let it return.
"""

import asyncio
from asyncio import Event as AsyncioEvent
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event as ThreadEvent
from typing import Annotated, Any

from conductor import (
    SKIPPED,
    Ask,
    CallCount,
    Graph,
    GraphRecord,
    NodeRegistry,
    Param,
    Policy,
    Run,
    Series,
    make_graph,
    node,
    subgraph,
)
from conductor.events import Event

from tests.test_core.nodes import registry_of


@node("lines", title="Lines", description="For a test.", policy=Policy(costs_nothing=True))
def lines(text: str) -> Series[str]:
    return text.splitlines()


@node("shout", title="Shout", description="For a test.", policy=Policy(costs_nothing=True))
def shout(line: str, mark: str = "!") -> str:
    return line.upper() + mark


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("approve", title="Approve", description="For a test.", asks=True)
def approve(line: str) -> bool:
    raise Ask(f"Approve {line}?")


@node(
    "fetch",
    title="Fetch",
    description="For a test.",
    policy=Policy(retries=2, retry_delay=0.01, retry_on=(TimeoutError,)),
)
def fetch(line: str) -> str:
    if line == "b":
        raise TimeoutError
    return line


@dataclass(frozen=True)
class Way:
    yes: Annotated[str, Param(exclusive_group="way")] = SKIPPED
    no: Annotated[str, Param(exclusive_group="way")] = SKIPPED


@node("decide", title="Decide", description="For a test.")
def decide(text: str) -> Way:
    return Way(no=text)


@node("both", title="Both", description="For a test.")
def both(first: str, second: str) -> str:
    return first + second


loud = subgraph("loud", title="Loud", description="For a test.")(make_graph(shout))


#: Set by the test once it has seen what it waits for; a gated call returns only after it.
GATE = ThreadEvent()
#: Set by a gated call when the call it holds has started.
HELD = ThreadEvent()


def _gated(node_id: str, held: str, *, threaded: bool) -> Any:
    """A node ``node_id`` that returns its text, holding the call given ``held`` until ``GATE`` is set.

    With ``threaded`` the node is called on threads, so the other calls go on while the held one waits.
    """
    policy = Policy(concurrency=4) if threaded else Policy()

    @node(node_id, title="Gated", description="For a test.", policy=policy)
    def gated(line: str) -> str:
        if line == held:
            HELD.set()
            assert GATE.wait(5), "the test never let the held call return"
        return line.upper() + "!"

    return gated


def _one_line(event: Event) -> str:
    """An event as its type, its loc, and every field that is not at its default; a problem as its type and loc, a
    waiting call as its loc, and a count as its four numbers.

    Example: ``NodeProgress(loc=("shout",), count=CallCount(finished=1, waiting=0, known=3, total=3))`` is
    ``"node_progress shout finished=1 waiting=0 known=3 total=3"``.
    """
    fields = event.model_dump(exclude={"type", "loc"}, exclude_defaults=True)
    if "waiting" in fields:
        fields["waiting"] = "/".join(map(str, fields["waiting"]["loc"]))
    fields.update(fields.pop("count", {}))
    if "cause" in fields:
        fields["cause"] = f"{fields['cause']['type']}@{'/'.join(map(str, fields['cause']['loc']))}"
    return " ".join(
        [
            event.type,
            "/".join(map(str, event.loc)),
            *(f"{name}={value}" for name, value in fields.items()),
        ]
    )


def _streamed(
    run: Run[Any],
    on: Callable[[list[str], AsyncioEvent], None] = lambda said, cancel: None,
) -> tuple[list[str], Run[Any]]:
    """The events ``run.stream()`` says about its nodes, one line each (``_one_line``), and the run it ends with.

    ``on`` is handed the lines said so far each time one is said, with the stream's ``cancel``; it may set ``GATE``
    or ``cancel``, and add a line of its own to mark when it did.

    A node's first ``node_progress`` and its last are always said; one in between is said only when a quarter of a
    second has passed since the one before, so it depends on the clock and is left out.
    """

    async def collect() -> tuple[list[str], Run[Any]]:
        cancel = AsyncioEvent()
        said: list[str] = []
        async for event in run.stream(cancel=cancel):
            if event.type == "run_end":
                return said, event.run
            if event.type.startswith("node_"):
                said.append(_one_line(event))
                on(said, cancel)
        raise AssertionError("the stream ended with no run_end")

    GATE.clear()
    HELD.clear()
    try:
        said, ended = asyncio.run(collect())
    finally:
        GATE.set()
    progress = [index for index, line in enumerate(said) if line.startswith("node_progress ")]
    by_node: dict[str, list[int]] = {}
    for index in progress:
        by_node.setdefault(said[index].split()[1], []).append(index)
    between = {index for indexes in by_node.values() for index in indexes[1:-1]}
    return [line for index, line in enumerate(said) if index not in between], ended


def _said(
    run: Run[Any], on: Callable[[list[str], AsyncioEvent], None] = lambda said, cancel: None
) -> list[str]:
    """The lines of ``_streamed``, without the run."""
    return _streamed(run, on)[0]


def _started(record: GraphRecord, registry: NodeRegistry, **inputs: Any) -> Run[Any]:
    return Graph.compile(record, registry).with_inputs(inputs)


def _lines_shout_join(text: str = "a\nb\nc") -> Run[Any]:
    return _started(make_graph(lines, shout, join), registry_of(lines, shout, join), text=text)


def test_the_three_node_graph_says_the_events_the_module_describes():
    assert _said(_lines_shout_join()) == [
        "node_start lines",
        "node_start shout",
        "node_progress shout finished=1 waiting=0 known=3 total=3",
        "node_start join",
        "node_complete lines outcomes=1",
        "node_progress shout finished=3 waiting=0 known=3 total=3",
        "node_complete shout outcomes=3",
        "node_complete join outcomes=1",
    ]


def test_a_node_is_complete_once_every_call_of_it_has_an_outcome():
    gated = _gated("shout", "b", threaded=True)
    run = _started(make_graph(lines, gated, join), registry_of(lines, gated, join), text="a\nb\nc")

    def let_b_return(said: list[str], cancel: AsyncioEvent) -> None:
        if said[-1] == "node_complete lines outcomes=1":
            said.append("b may return")
            GATE.set()

    said = _said(run, let_b_return)
    let_go = said.index("b may return")
    assert "node_complete shout outcomes=3" in said[let_go:]
    assert not [line for line in said[:let_go] if line.startswith("node_complete shout")]
    assert said[-2:] == ["node_complete shout outcomes=3", "node_complete join outcomes=1"]


def test_a_skipped_node_has_no_start_and_is_said_only_when_no_call_upstream_of_it_is_running():
    other = _gated("other", "b", threaded=True)
    record = make_graph(
        decide,
        other,
        both,
        edges=[("decide.yes", "both.first"), ("other.result", "both.second")],
        inputs={"choice": "decide.text", "other": "other.line"},
    )
    run = _started(record, registry_of(decide, other, both), choice="a", other="b")

    def let_other_return(said: list[str], cancel: AsyncioEvent) -> None:
        if said[-1] == "node_complete decide outcomes=1":
            GATE.set()

    said = _said(run, let_other_return)
    assert "node_start both" not in said
    assert said[-2:] == [
        "node_complete other outcomes=1",
        "node_complete both outcomes=1 skipped=1",
    ]


def test_a_node_with_a_call_waiting_for_a_person_says_so_and_is_not_complete():
    run = _started(make_graph(lines, approve), registry_of(lines, approve), text="a\nb")
    assert _said(run) == [
        "node_start lines",
        "node_start approve",
        "node_waiting approve waiting=approve/0",
        "node_progress approve finished=0 waiting=1 known=2 total=2",
        "node_waiting approve waiting=approve/1",
        "node_complete lines outcomes=1",
        "node_progress approve finished=0 waiting=2 known=2 total=2",
    ]


def test_a_node_with_a_call_still_to_make_is_not_complete_when_the_run_ends_and_says_how_far_it_got():
    gated = _gated("shout", "b", threaded=False)
    run = _started(make_graph(lines, gated, join), registry_of(lines, gated, join), text="a\nb\nc")

    def cancel_during_b(said: list[str], cancel: AsyncioEvent) -> None:
        if said[-1].startswith("node_progress shout finished=1 "):
            assert HELD.wait(5), "the call for b never started"
            cancel.set()
            GATE.set()

    said, ended = _streamed(run, cancel_during_b)
    assert ended.status == "cancelled"
    assert not [line for line in said if line.startswith("node_complete shout")]
    assert ended.progress.nodes["shout"] == CallCount(finished=2, waiting=0, known=3, total=3)
    assert said[-1] == "node_progress shout finished=2 waiting=0 known=3 total=3"


def test_an_outcome_reused_from_the_previous_run_starts_its_node_and_is_counted_as_reused():
    previous = _lines_shout_join().resume()
    run = previous.graph.with_inputs({"text": "a\nb\nc\n"}, previous=previous)
    assert _said(run) == [
        "node_start lines",
        "node_start shout",
        "node_progress shout finished=1 waiting=0 known=3 total=3",
        "node_start join",
        "node_complete lines outcomes=1",
        "node_progress shout finished=3 waiting=0 known=3 total=3",
        "node_complete shout outcomes=3 reused=3",
        "node_complete join outcomes=1 reused=1",
    ]


def test_a_node_complete_when_the_run_was_last_resumed_is_not_said_again():
    run = _lines_shout_join().resume(only_costs_nothing=True)
    assert _said(run) == ["node_start join", "node_complete join outcomes=1"]


def test_a_node_inside_a_subgraph_starts_the_subgraph_node_first_and_the_subgraph_node_is_complete_after_it():
    record = make_graph(
        lines, loud, join, edges=[("lines.result", "loud.line"), ("loud.result", "join.texts")]
    )
    run = _started(record, registry_of(lines, shout, loud, join), text="a\nb")
    assert _said(run) == [
        "node_start lines",
        "node_start loud",
        "node_start loud/shout",
        "node_progress loud/shout finished=1 waiting=0 known=2 total=2",
        "node_start join",
        "node_complete lines outcomes=1",
        "node_progress loud/shout finished=2 waiting=0 known=2 total=2",
        "node_complete loud/shout outcomes=2",
        "node_complete loud outcomes=2",
        "node_complete join outcomes=1",
    ]


def test_a_retry_and_a_failure_name_the_node_and_their_cause_names_the_call():
    run = _started(make_graph(lines, fetch), registry_of(lines, fetch), text="a\nb")
    said = _said(run)
    assert [line for line in said if line.startswith(("node_retry", "node_failed"))] == [
        "node_retry fetch attempt=1 retries=2 delay=0.01 cause=external_failed@fetch/1",
        "node_retry fetch attempt=2 retries=2 delay=0.02 cause=external_failed@fetch/1",
        "node_failed fetch cause=external_failed@fetch/1",
    ]
