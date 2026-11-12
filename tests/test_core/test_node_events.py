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
    Series,
    make_graph,
    node,
)
from conductor._internal.address import ItemIndexes
from conductor._internal.compiled import FlatFunctionNode
from conductor._internal.outcomes import Outcome
from conductor._internal.scheduler import node_events as node_events_module
from conductor._internal.scheduler.node_events import NodeEvents
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


@node("approve", title="Approve", description="For a test.")
def approve(line: str) -> bool:
    raise Ask(f"Approve {line}?")


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


LINES_SHOUT_JOIN = make_graph(lines, shout, join)


class Told:
    """A ``NodeEvents`` over an empty copy of a finished run's outcomes, and the events it has said so far."""

    def __init__(self, record: GraphRecord, registry: NodeRegistry, **inputs: Any) -> None:
        graph = Graph.compile(record, registry)
        self.finished_run: dict[str, dict[ItemIndexes, Outcome]] = graph.run(inputs)._outcomes
        self.outcomes: dict[str, dict[ItemIndexes, Outcome]] = {name: {} for name in self.finished_run}
        self.nodes: dict[str, FlatFunctionNode] = {flat.name: flat for flat in graph._flat_function_nodes}
        self.said: list[Any] = []
        self.node_events = NodeEvents(self.said.append, graph._flat_function_nodes, self.outcomes, frozenset())

    def call(self, name: str, *item_indexes: int) -> None:
        """Tell it that the call is about to be made, and then that its outcome was recorded."""
        self.node_events.call_starts(self.nodes[name])
        self.record(name, *item_indexes)

    def record(self, name: str, *item_indexes: int) -> None:
        """Tell it that the call's outcome was recorded, with no call made."""
        self.outcomes[name][item_indexes] = self.finished_run[name][item_indexes]
        self.node_events.outcome_recorded(self.nodes[name], item_indexes, self.outcomes[name][item_indexes])

    def nothing_is_ready(self, *running: tuple[str, ItemIndexes]) -> None:
        self.node_events.nothing_is_ready([(self.nodes[name], item_indexes) for name, item_indexes in running])

    def take(self) -> list[tuple[Any, ...]]:
        """The events said since the last ``take``, each as its type, node id and the counts it has."""
        taken = [
            (event.type, event.waiting.loc[0] if event.type == "node_waiting" else event.loc[0])
            + tuple(getattr(event, count) for count in ("done", "total", "outcomes") if hasattr(event, count))
            for event in self.said
        ]
        self.said.clear()
        return taken


def _lines_shout_join(text: str = "a\nb\nc") -> Told:
    return Told(LINES_SHOUT_JOIN, registry_of(lines, shout, join), text=text)


def test_the_three_node_graph_says_the_events_the_module_describes():
    told = _lines_shout_join()
    told.call("lines")
    told.call("shout", 0)
    told.call("shout", 1)
    told.call("shout", 2)
    told.call("join")
    told.nothing_is_ready()
    told.node_events.scheduler_ended()
    assert told.take() == [
        ("node_start", "lines"),
        ("node_start", "shout"),
        ("node_progress", "shout", 1, 3),
        ("node_start", "join"),
        ("node_complete", "lines", 1),
        ("node_progress", "shout", 3, 3),
        ("node_complete", "shout", 3),
        ("node_complete", "join", 1),
    ]


def test_a_node_is_complete_once_every_call_of_it_has_an_outcome():
    told = _lines_shout_join()
    told.call("lines")
    told.call("shout", 0)
    told.call("shout", 2)
    told.take()
    told.nothing_is_ready(("shout", (1,)))
    assert told.take() == [("node_complete", "lines", 1)]
    told.call("shout", 1)
    told.nothing_is_ready()
    assert told.take() == [("node_progress", "shout", 3, 3), ("node_complete", "shout", 3)]
    told.call("join")
    told.nothing_is_ready()
    assert told.take() == [("node_start", "join"), ("node_complete", "join", 1)]


def test_a_skipped_node_has_no_start_and_is_said_only_when_no_call_upstream_of_it_is_running():
    record = make_graph(decide, other, both, edges=[("decide.yes", "both.first"), ("other.result", "both.second")])
    told = Told(record, registry_of(decide, other, both), **{"decide.text": "a", "other.text": "b"})
    told.call("decide")
    told.record("both")  # skipped as soon as decide has not chosen yes, while other is still running
    told.nothing_is_ready(("other", ()))
    assert told.take() == [("node_start", "decide"), ("node_complete", "decide", 1)]
    told.call("other")
    told.nothing_is_ready()
    assert told.take() == [("node_start", "other"), ("node_complete", "other", 1), ("node_skipped", "both")]


def test_a_node_with_a_call_waiting_for_a_person_says_so_and_is_not_complete():
    told = Told(make_graph(lines, approve), registry_of(lines, approve), text="a\nb")
    told.call("lines")
    told.call("approve", 0)
    told.call("approve", 1)
    told.nothing_is_ready()
    told.node_events.scheduler_ended()
    assert told.take() == [
        ("node_start", "lines"),
        ("node_start", "approve"),
        ("node_waiting", "approve"),
        ("node_progress", "approve", 0, 2),
        ("node_waiting", "approve"),
        ("node_complete", "lines", 1),
    ]


def test_a_node_with_a_call_still_to_make_is_not_complete_when_the_scheduler_ends_and_says_how_far_it_got():
    told = _lines_shout_join()
    told.call("lines")
    told.call("shout", 0)
    told.call("shout", 1)
    told.take()
    told.node_events.scheduler_ended()
    assert told.take() == [("node_complete", "lines", 1), ("node_progress", "shout", 2, 3)]


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
    progress = [event for event in told.take() if event[0] == "node_progress"]
    assert progress == [("node_progress", "shout", 1, 8), ("node_progress", "shout", 4, 8), ("node_progress", "shout", 7, 8), ("node_progress", "shout", 8, 8)]


def test_an_outcome_taken_from_an_earlier_run_starts_its_node_and_is_counted_as_reused():
    told = _lines_shout_join()
    told.call("lines")
    for index in range(3):
        told.node_events.outcome_reused(told.nodes["shout"])
        told.record("shout", index)
    told.nothing_is_ready()
    assert [(event.type, event.loc) for event in told.said[:2]] == [("node_start", ("lines",)), ("node_start", ("shout",))]
    complete = next(event for event in told.said if event.type == "node_complete" and event.loc == ("shout",))
    assert (complete.outcomes, complete.reused) == (3, 3)


def test_a_node_complete_when_the_run_was_last_resumed_is_not_said_again():
    told = _lines_shout_join()
    for name, outcomes in told.finished_run.items():
        told.outcomes[name].update(outcomes)
    resumed = NodeEvents(told.said.append, list(told.nodes.values()), told.outcomes, frozenset({("lines",), ("shout",)}))
    resumed.scheduler_ended()
    assert told.take() == [("node_complete", "join", 1)]
