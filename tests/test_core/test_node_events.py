"""The events about each node, worked out by ``NodeEvents`` from what a test tells it by hand.

No thread is started here and no clock is read. Each test runs a small graph to its end once, to have real
outcomes, and then tells a new ``NodeEvents`` about those outcomes one at a time, in an order it chooses, with a
list's ``append`` as the listener.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any

from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
    NodeRegistry,
    Param,
    Policy,
    Series,
    make_graph,
    node,
    subgraph,
)
from conductor._internal.base import ItemIndexes
from conductor._internal.problem import Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.run import node_events as node_events_module
from conductor._internal.run.events import EventBase, NodeWaiting
from conductor._internal.run.node_events import NodeEvents
from conductor._internal.run.outcome import Outcome
from pytest import MonkeyPatch

from tests.test_core.nodes import registry_of


@node("lines", title="Lines", description="For a test.")
def lines(text: str) -> Series[str]:
    return text.splitlines()


@node("shout", title="Shout", description="For a test.")
def shout(line: str, mark: str = "!") -> str:
    return line.upper() + mark


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("approve", title="Approve", description="For a test.", asks=True)
def approve(line: str) -> bool:
    raise Ask(f"Approve {line}?")


@node("fetch", title="Fetch", description="For a test.", policy=Policy(retries=2))
def fetch(line: str) -> str:
    return line


@dataclass(frozen=True)
class Way:
    yes: Annotated[str, Param(exclusive_group="way")] = SKIPPED
    no: Annotated[str, Param(exclusive_group="way")] = SKIPPED


@node("decide", title="Decide", description="For a test.")
def decide(text: str) -> Way:
    return Way(no=text)


@node("other", title="Other", description="For a test.")
def other(text: str) -> str:
    return text


@node("both", title="Both", description="For a test.")
def both(first: str, second: str) -> str:
    return first + second


loud = subgraph("loud", title="Loud", description="For a test.")(make_graph(shout))


class Told:
    """A ``NodeEvents`` for one graph, told by node name what the scheduler would tell it, and the events it said.

    The outcomes it is told about are those of a run of the graph run to its end first. ``finished`` names the
    nodes that were complete when the run was last resumed: their outcomes are in the run before the
    ``NodeEvents`` is created.
    """

    def __init__(self, record: GraphRecord, registry: NodeRegistry, finished: tuple[str, ...] = (), **inputs: Any) -> None:
        graph = Graph.compile(record, registry)
        run = graph.run(inputs)
        self.finished_run: dict[str, dict[ItemIndexes, Outcome]] = run._outcomes
        self.outcomes = {name: dict(outcomes) if name in finished else {} for name, outcomes in self.finished_run.items()}
        self.nodes = run._flat.function_nodes
        self.said: list[EventBase] = []
        finished_paths = frozenset(self.nodes[name].node_path for name in finished)
        self.node_events = NodeEvents(self.said.append, list(self.nodes.values()), self.outcomes, finished_paths)

    def start(self, name: str) -> None:
        """A call of the node is about to be made."""
        self.node_events.call_starts(self.nodes[name])

    def record(self, name: str, *item_indexes: int) -> None:
        """The call's outcome was recorded, with no call made."""
        self.outcomes[name][item_indexes] = self.finished_run[name][item_indexes]
        self.node_events.outcome_recorded(self.nodes[name], item_indexes, self.outcomes[name][item_indexes])

    def call(self, name: str, *item_indexes: int) -> None:
        """The call is about to be made, and then its outcome was recorded."""
        self.start(name)
        self.record(name, *item_indexes)

    def reuse(self, name: str, *item_indexes: int) -> None:
        """The call's outcome was reused from the previous run, and recorded."""
        self.node_events.outcome_reused(self.nodes[name])
        self.record(name, *item_indexes)

    def retry(self, name: str, *item_indexes: int, attempt: int, delay: float) -> None:
        """The call failed with a ``TimeoutError`` and is made again after ``delay`` seconds."""
        self.node_events.call_retries(self.nodes[name], item_indexes, attempt, delay, self._timed_out(name, item_indexes))

    def fail(self, name: str, *item_indexes: int) -> None:
        """The call failed with a ``TimeoutError`` and is not retried."""
        self.node_events.call_failed(self.nodes[name], self._timed_out(name, item_indexes))

    def nothing_is_ready(self, *running: tuple[str, ItemIndexes]) -> None:
        """No call is ready to be made, and these calls are running."""
        self.node_events.nothing_is_ready([(self.nodes[name], item_indexes) for name, item_indexes in running])

    def end(self) -> None:
        """The scheduler has ended."""
        self.node_events.scheduler_ended()

    def take(self) -> list[str]:
        """The events said since the last ``take``, one line each (``_one_line``)."""
        taken = [_one_line(event) for event in self.said]
        self.said.clear()
        return taken

    def _timed_out(self, name: str, item_indexes: ItemIndexes) -> Problem:
        return problem.external_failed(name, *item_indexes, error="TimeoutError")  # the loc of a node of the graph itself


def _one_line(event: EventBase) -> str:
    """An event as its type, its loc, and every field that is not at its default; a problem as its type and loc.

    Example: ``NodeProgress(loc=("shout",), finished=1, total=3)`` is ``"node_progress shout finished=1 total=3"``.
    """
    loc = event.waiting.loc if isinstance(event, NodeWaiting) else event.loc
    fields = event.model_dump(exclude={"type", "loc", "waiting"}, exclude_defaults=True)
    if "cause" in fields:
        fields["cause"] = f"{fields['cause']['type']}@{'/'.join(map(str, fields['cause']['loc']))}"
    return " ".join([event.type, "/".join(map(str, loc)), *(f"{name}={value}" for name, value in fields.items())])


def _lines_shout_join(text: str = "a\nb\nc", finished: tuple[str, ...] = ()) -> Told:
    return Told(make_graph(lines, shout, join), registry_of(lines, shout, join), finished, text=text)


def test_the_three_node_graph_says_the_events_the_module_describes():
    told = _lines_shout_join()
    told.call("lines")
    told.call("shout", 0)
    told.call("shout", 1)
    told.call("shout", 2)
    told.call("join")
    told.nothing_is_ready()
    told.end()
    assert told.take() == [
        "node_start lines",
        "node_start shout",
        "node_progress shout finished=1 total=3",
        "node_start join",
        "node_complete lines outcomes=1",
        "node_progress shout finished=3 total=3",
        "node_complete shout outcomes=3",
        "node_complete join outcomes=1",
    ]


def test_a_node_is_complete_once_every_call_of_it_has_an_outcome():
    told = _lines_shout_join()
    told.call("lines")
    told.call("shout", 0)
    told.call("shout", 2)
    told.take()
    told.nothing_is_ready(("shout", (1,)))
    assert told.take() == ["node_complete lines outcomes=1"]
    told.call("shout", 1)
    told.nothing_is_ready()
    assert told.take() == ["node_progress shout finished=3 total=3", "node_complete shout outcomes=3"]
    told.call("join")
    told.nothing_is_ready()
    assert told.take() == ["node_start join", "node_complete join outcomes=1"]


def test_a_skipped_node_has_no_start_and_is_said_only_when_no_call_upstream_of_it_is_running():
    record = make_graph(decide, other, both, edges=[("decide.yes", "both.first"), ("other.result", "both.second")], inputs={"choice": "decide.text", "other": "other.text"})
    told = Told(record, registry_of(decide, other, both), choice="a", other="b")
    told.call("decide")
    told.record("both")  # skipped as soon as decide has not chosen yes, while other is still running
    told.nothing_is_ready(("other", ()))
    assert told.take() == ["node_start decide", "node_complete decide outcomes=1"]
    told.call("other")
    told.nothing_is_ready()
    assert told.take() == ["node_start other", "node_complete other outcomes=1", "node_skipped both"]


def test_a_node_with_a_call_waiting_for_a_person_says_so_and_is_not_complete():
    told = Told(make_graph(lines, approve), registry_of(lines, approve), text="a\nb")
    told.call("lines")
    told.call("approve", 0)
    told.call("approve", 1)
    told.nothing_is_ready()
    told.end()
    assert told.take() == [
        "node_start lines",
        "node_start approve",
        "node_waiting approve/0",
        "node_progress approve finished=0 total=2",
        "node_waiting approve/1",
        "node_complete lines outcomes=1",
    ]


def test_a_node_with_a_call_still_to_make_is_not_complete_when_the_scheduler_ends_and_says_how_far_it_got():
    told = _lines_shout_join()
    told.call("lines")
    told.call("shout", 0)
    told.call("shout", 1)
    told.take()
    told.end()
    assert told.take() == ["node_complete lines outcomes=1", "node_progress shout finished=2 total=3"]


def test_progress_is_said_at_most_once_in_a_quarter_of_a_second_and_the_last_one_has_the_final_count(monkeypatch: MonkeyPatch):
    clock = [100.0]
    monkeypatch.setattr(node_events_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    told = _lines_shout_join("\n".join("abcdefgh"))
    told.call("lines")
    for index in range(8):
        told.call("shout", index)
        clock[0] += 0.1
    told.call("join")
    told.nothing_is_ready()
    progress = [event for event in told.take() if event.startswith("node_progress")]
    assert progress == [
        "node_progress shout finished=1 total=8",
        "node_progress shout finished=4 total=8",
        "node_progress shout finished=7 total=8",
        "node_progress shout finished=8 total=8",
    ]


def test_an_outcome_reused_from_the_previous_run_starts_its_node_and_is_counted_as_reused():
    told = _lines_shout_join()
    told.call("lines")
    for index in range(3):
        told.reuse("shout", index)
    told.nothing_is_ready()
    assert told.take() == [
        "node_start lines",
        "node_start shout",
        "node_progress shout finished=1 total=3",
        "node_complete lines outcomes=1",
        "node_progress shout finished=3 total=3",
        "node_complete shout outcomes=3 reused=3",
    ]


def test_a_node_complete_when_the_run_was_last_resumed_is_not_said_again():
    told = _lines_shout_join(finished=("lines", "shout"))
    told.call("join")
    told.end()
    assert told.take() == ["node_start join", "node_complete join outcomes=1"]


def test_a_node_inside_a_subgraph_starts_the_subgraph_node_first_and_the_subgraph_node_is_complete_after_it():
    record = make_graph(lines, loud, join, edges=[("lines.result", "loud.line"), ("loud.result", "join.texts")])
    told = Told(record, registry_of(lines, shout, loud, join), text="a\nb")
    told.call("lines")
    told.call("loud/shout", 0)
    told.call("loud/shout", 1)
    told.call("join")
    told.nothing_is_ready()
    assert told.take() == [
        "node_start lines",
        "node_start loud",
        "node_start loud/shout",
        "node_progress loud/shout finished=1 total=2",
        "node_start join",
        "node_complete lines outcomes=1",
        "node_progress loud/shout finished=2 total=2",
        "node_complete loud/shout outcomes=2",
        "node_complete loud outcomes=2",
        "node_complete join outcomes=1",
    ]


def test_a_retry_names_the_call_and_a_failure_names_the_node():
    told = Told(make_graph(lines, fetch), registry_of(lines, fetch), text="a\nb")
    told.call("lines")
    told.start("fetch")
    told.retry("fetch", 1, attempt=1, delay=0.5)
    told.fail("fetch", 1)
    assert told.take() == [
        "node_start lines",
        "node_start fetch",
        "node_retry fetch/1 attempt=1 retries=2 delay=0.5 cause=external_failed@fetch/1",
        "node_failed fetch cause=external_failed@fetch/1",
    ]
