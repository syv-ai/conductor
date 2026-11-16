"""The events ``run.stream()`` sends about each node, and the order they come in.

Each test streams a small graph and checks the events about its nodes. Nodes called on the stream's own thread
are called in a fixed order, so their events come in a fixed order too.

Some tests need one call to keep running while the others finish. That call runs on a thread of its own, and it
waits until the test lets it return.
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


#: Set by the test when it has seen the events it waits for. A gated call can't return before this is set.
GATE = ThreadEvent()
#: Set by a gated node when its held call has started.
HELD = ThreadEvent()


def _gated(node_id: str, held: str, *, threaded: bool) -> Any:
    """Returns a node with id ``node_id`` that shouts its text, and holds back one call.

    The call whose text is ``held`` waits until ``GATE`` is set before it returns. With ``threaded``, the node is
    called on threads, so its other calls go on while the held one waits.
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
    """Writes an event as one short line, so a test can compare events as text.

    The line has the event's type, its loc, and each field that isn't at its default. A problem is shortened to
    its type and loc, a waiting call to its loc, and a count to its six numbers, which come first. For example,
    ``NodeProgress(loc=("shout",), count=CallCount(finished=1, waiting=0, skipped=0, not_called=0, known=3,
    total=3))`` becomes ``"node_progress shout finished=1 waiting=0 skipped=0 not_called=0 known=3 total=3"``.
    """
    fields = event.model_dump(exclude={"type", "loc"}, exclude_defaults=True)
    if "waiting" in fields:
        fields["waiting"] = "/".join(map(str, fields["waiting"]["loc"]))
    fields = {**fields.pop("count", {}), **fields}
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
    on: Callable[[list[str], AsyncioEvent], None] = lambda events, cancel: None,
) -> tuple[list[str], Run[Any]]:
    """Streams ``run`` and returns its node events, one line each, with the run the stream ends with.

    Each event is written by ``_one_line``. After each one, ``on`` is called with the lines so far and the
    stream's ``cancel``. It may set ``GATE`` or ``cancel``, and it may add a line of its own to mark when it did.

    A node's first and last ``node_progress`` are always sent. One in between is sent only when a quarter of a
    second has passed since the last, so it depends on the clock. Those are left out.
    """

    async def collect() -> tuple[list[str], Run[Any]]:
        cancel = AsyncioEvent()
        events: list[str] = []
        async for event in run.stream(cancel=cancel):
            if event.type == "run_end":
                return events, event.run
            if event.type.startswith("node_"):
                events.append(_one_line(event))
                on(events, cancel)
        raise AssertionError("the stream ended with no run_end")

    GATE.clear()
    HELD.clear()
    try:
        events, ended = asyncio.run(collect())
    finally:
        GATE.set()
    progress = [index for index, line in enumerate(events) if line.startswith("node_progress ")]
    by_node: dict[str, list[int]] = {}
    for index in progress:
        by_node.setdefault(events[index].split()[1], []).append(index)
    between = {index for indexes in by_node.values() for index in indexes[1:-1]}
    return [line for index, line in enumerate(events) if index not in between], ended


def _event_lines(
    run: Run[Any], on: Callable[[list[str], AsyncioEvent], None] = lambda events, cancel: None
) -> list[str]:
    """The lines ``_streamed`` returns, without the run."""
    return _streamed(run, on)[0]


def _started(record: GraphRecord, registry: NodeRegistry, **inputs: Any) -> Run[Any]:
    return Graph.compile(record, registry).with_inputs(inputs)


def _lines_shout_join(text: str = "a\nb\nc") -> Run[Any]:
    return _started(make_graph(lines, shout, join), registry_of(lines, shout, join), text=text)


def test_the_three_node_graph_says_the_events_the_module_describes():
    assert _event_lines(_lines_shout_join()) == [
        "node_start lines",
        "node_start shout",
        "node_progress shout finished=1 waiting=0 skipped=0 not_called=0 known=3 total=3",
        "node_start join",
        "node_complete lines finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
        "node_progress shout finished=3 waiting=0 skipped=0 not_called=0 known=3 total=3",
        "node_complete shout finished=3 waiting=0 skipped=0 not_called=0 known=3 total=3",
        "node_complete join finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
    ]


def test_a_node_is_complete_once_every_call_of_it_has_an_outcome():
    gated = _gated("shout", "b", threaded=True)
    run = _started(make_graph(lines, gated, join), registry_of(lines, gated, join), text="a\nb\nc")

    def let_b_return(events: list[str], cancel: AsyncioEvent) -> None:
        if (
            events[-1]
            == "node_complete lines finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1"
        ):
            events.append("b may return")
            GATE.set()

    events = _event_lines(run, let_b_return)
    let_go = events.index("b may return")
    assert (
        "node_complete shout finished=3 waiting=0 skipped=0 not_called=0 known=3 total=3"
        in events[let_go:]
    )
    assert not [line for line in events[:let_go] if line.startswith("node_complete shout")]
    assert events[-2:] == [
        "node_complete shout finished=3 waiting=0 skipped=0 not_called=0 known=3 total=3",
        "node_complete join finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
    ]


def test_a_skipped_node_has_no_start_and_is_complete_only_when_no_call_upstream_of_it_is_running():
    other = _gated("other", "b", threaded=True)
    record = make_graph(
        decide,
        other,
        both,
        edges=[("decide.yes", "both.first"), ("other.result", "both.second")],
        inputs={"choice": "decide.text", "other": "other.line"},
    )
    run = _started(record, registry_of(decide, other, both), choice="a", other="b")

    def let_other_return(events: list[str], cancel: AsyncioEvent) -> None:
        if (
            events[-1]
            == "node_complete decide finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1"
        ):
            GATE.set()

    events = _event_lines(run, let_other_return)
    assert "node_start both" not in events
    assert events[-2:] == [
        "node_complete other finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
        "node_complete both finished=1 waiting=0 skipped=1 not_called=0 known=1 total=1",
    ]


def test_a_node_with_a_call_waiting_for_a_person_says_so_and_is_not_complete():
    run = _started(make_graph(lines, approve), registry_of(lines, approve), text="a\nb")
    assert _event_lines(run) == [
        "node_start lines",
        "node_start approve",
        "node_waiting approve waiting=approve/0",
        "node_progress approve finished=0 waiting=1 skipped=0 not_called=0 known=2 total=2",
        "node_waiting approve waiting=approve/1",
        "node_complete lines finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
        "node_progress approve finished=0 waiting=2 skipped=0 not_called=0 known=2 total=2",
    ]


def test_a_node_with_a_call_still_to_make_is_not_complete_when_the_run_ends_and_says_how_far_it_got():
    gated = _gated("shout", "b", threaded=False)
    run = _started(make_graph(lines, gated, join), registry_of(lines, gated, join), text="a\nb\nc")

    def cancel_during_b(events: list[str], cancel: AsyncioEvent) -> None:
        if events[-1].startswith("node_progress shout finished=1 "):
            assert HELD.wait(5), "the call for b never started"
            cancel.set()
            GATE.set()

    events, ended = _streamed(run, cancel_during_b)
    assert ended.status == "cancelled"
    assert not [line for line in events if line.startswith("node_complete shout")]
    assert ended.progress.nodes["shout"] == CallCount(
        finished=2, waiting=0, skipped=0, not_called=0, known=3, total=3
    )
    assert (
        events[-1]
        == "node_progress shout finished=2 waiting=0 skipped=0 not_called=0 known=3 total=3"
    )


def test_an_outcome_reused_from_the_previous_run_starts_its_node_and_is_counted_as_reused():
    previous = _lines_shout_join().resume()
    run = previous.graph.with_inputs({"text": "a\nb\nc\n"}, previous=previous)
    assert _event_lines(run) == [
        "node_start lines",
        "node_start shout",
        "node_progress shout finished=1 waiting=0 skipped=0 not_called=0 known=3 total=3",
        "node_start join",
        "node_complete lines finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
        "node_progress shout finished=3 waiting=0 skipped=0 not_called=0 known=3 total=3",
        "node_complete shout finished=3 waiting=0 skipped=0 not_called=0 known=3 total=3 reused=3",
        "node_complete join finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1 reused=1",
    ]


def test_a_node_completed_by_an_earlier_resume_sends_no_events_again():
    run = _lines_shout_join().resume(only_costs_nothing=True)
    assert _event_lines(run) == [
        "node_start join",
        "node_complete join finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
    ]


def test_a_node_inside_a_subgraph_starts_the_subgraph_node_first_and_the_subgraph_node_is_complete_after_it():
    record = make_graph(
        lines, loud, join, edges=[("lines.result", "loud.line"), ("loud.result", "join.texts")]
    )
    run = _started(record, registry_of(lines, shout, loud, join), text="a\nb")
    assert _event_lines(run) == [
        "node_start lines",
        "node_start loud",
        "node_start loud/shout",
        "node_progress loud/shout finished=1 waiting=0 skipped=0 not_called=0 known=2 total=2",
        "node_start join",
        "node_complete lines finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
        "node_progress loud/shout finished=2 waiting=0 skipped=0 not_called=0 known=2 total=2",
        "node_complete loud/shout finished=2 waiting=0 skipped=0 not_called=0 known=2 total=2",
        "node_complete loud finished=2 waiting=0 skipped=0 not_called=0 known=2 total=2",
        "node_complete join finished=1 waiting=0 skipped=0 not_called=0 known=1 total=1",
    ]


def test_a_node_complete_has_the_count_run_progress_has_for_the_node_a_subgraph_node_too():
    """A host that reads ``run.progress`` and the events sees one count per node, never two.

    A subgraph node is never called itself, so in both its count is the sum of the nodes inside it: ``loud`` runs
    ``shout`` once per line, and has its two calls.
    """
    record = make_graph(
        lines, loud, join, edges=[("lines.result", "loud.line"), ("loud.result", "join.texts")]
    )
    run = _started(record, registry_of(lines, shout, loud, join), text="a\nb")

    async def collect() -> list[Any]:
        return [event async for event in run.stream()]

    events = asyncio.run(collect())
    progress = events[-1].run.progress.nodes
    complete = {
        "/".join(event.loc): event.count for event in events if event.type == "node_complete"
    }

    assert complete == {path: progress[path] for path in ("lines", "loud/shout", "loud", "join")}
    assert complete["loud"] == CallCount(
        finished=2, waiting=0, skipped=0, not_called=0, known=2, total=2
    )


def test_a_retry_and_a_failure_name_the_node_and_their_cause_names_the_call():
    run = _started(make_graph(lines, fetch), registry_of(lines, fetch), text="a\nb")
    events = _event_lines(run)
    assert [line for line in events if line.startswith(("node_retry", "node_failed"))] == [
        "node_retry fetch attempt=1 retries=2 delay=0.01 cause=external_failed@fetch/1",
        "node_retry fetch attempt=2 retries=2 delay=0.02 cause=external_failed@fetch/1",
        "node_failed fetch cause=external_failed@fetch/1",
    ]
