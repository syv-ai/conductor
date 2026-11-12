"""A complete run holds a unit for every instance of every node, whichever nodes ran on threads, and going on
from it calls nothing and changes nothing."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
    NodeRegistry,
    Param,
    Policy,
    Series,
    node,
    subgraph,
)

from tests.test_core.nodes import units_missing

CALLS: Counter[str] = Counter()
_LOCK = threading.Lock()


def _called(name: str) -> None:
    with _LOCK:
        CALLS[name] += 1


def lines(text: str) -> Series[str]:
    _called("lines")
    return text.split(";")


def words(text: str) -> Series[str]:
    _called("words")
    return text.split()


def upper(text: str) -> str:
    _called("upper")
    return text.upper()


def join(items: Series[str]) -> str:
    _called("join")
    return "+".join(items)


@dataclass(frozen=True)
class Way:
    yes: Annotated[str, Param(exclusive_group="way")] = SKIPPED
    no: Annotated[str, Param(exclusive_group="way")] = SKIPPED


def decide(text: str) -> Way:
    _called("decide")
    time.sleep(0.02)  # on a thread it comes back after the Series beside it are made
    return Way(no=text) if text.startswith("no") else Way(yes=text)


def tag(text: str, mark: str) -> str:
    _called("tag")
    return f"{mark}:{text}"


def beside(text: str, items: Series[str]) -> str:
    _called("beside")
    return f"{text} of {'+'.join(items)}"


def pieces(text: str) -> Series[str]:
    _called("pieces")
    time.sleep(0.05)  # on a thread it comes back after the decision beside it
    return text.split()


def ask(text: str) -> str:
    _called("ask")
    raise Ask(f"What about {text}?")


FUNCTIONS = (lines, words, upper, join, decide, tag, beside, pieces, ask)

#: Each graph by what it is: its nodes by id, its edges, and the text every first node of it is given.
GRAPHS: dict[str, tuple[dict[str, str], list[tuple[str, str]], str]] = {
    "a chain": ({"a": "upper", "b": "upper"}, [("a.result", "b.text")], "x"),
    "a gather of a Series with no items": ({"w": "words", "u": "upper", "j": "join"}, [("w.result", "u.text"), ("u.result", "j.items")], ""),
    "a gather per line, one line with no words": (
        {"l": "lines", "w": "words", "u": "upper", "j": "join"}, [("l.result", "w.text"), ("w.result", "u.text"), ("u.result", "j.items")], "a b;;c",
    ),
    "a gather per line of one line with no words": (
        {"l": "lines", "w": "words", "u": "upper", "j": "join"}, [("l.result", "w.text"), ("w.result", "u.text"), ("u.result", "j.items")], "",
    ),
    "a decision above a gather per line, one line with no words": (
        {"d": "decide", "l": "lines", "w": "words", "t": "tag", "j": "join"},
        [("l.result", "w.text"), ("w.result", "t.text"), ("d.yes", "t.mark"), ("t.result", "j.items")], "a b;;c",
    ),
    "a branch not taken above a gather per line, one line with no words": (
        {"d": "decide", "l": "lines", "w": "words", "t": "tag", "j": "join"},
        [("l.result", "w.text"), ("w.result", "t.text"), ("d.yes", "t.mark"), ("t.result", "j.items")], "no;;no",
    ),
    "every item's value gathered beside each item": (
        {"w": "words", "u": "upper", "b": "beside"}, [("w.result", "u.text"), ("w.result", "b.text"), ("u.result", "b.items")], "a b c",
    ),
    "a subgraph whose inner Series has no items, per line": ({"l": "lines", "e": "shout-words"}, [("l.result", "e.text")], "a b;;c"),
    "a subgraph per line, a node inside it reading nothing": ({"l": "lines", "e": "shout-and-stamp"}, [("l.result", "e.text")], "a;b"),
    "a subgraph that gathers a Series with no items": ({"e": "join-words", "u": "upper"}, [("e.j.result", "u.text")], ""),
}

#: The nodes called on threads; every other is called on the leg's own thread.
THREADED: list[frozenset[str]] = [
    frozenset(), frozenset({"lines"}), frozenset({"words"}), frozenset({"lines", "words"}), frozenset({"decide"}), frozenset({"upper", "tag"}), frozenset({"words", "upper"}),
    frozenset({"decide", "lines"}), frozenset(function.__name__ for function in FUNCTIONS),
]


def _registry(threaded: frozenset[str]) -> NodeRegistry:
    registry = NodeRegistry()
    for function in FUNCTIONS:
        name = function.__name__
        policy = {"policy": Policy(concurrency=4)} if name in threaded else {}
        registry.add(node(name, title=name.capitalize(), description=f"The {name} node.", **policy)(function))
    chain = [["w.result", "u.text"]]
    inner = {"nodes": [{"id": "w", "node": "words"}, {"id": "u", "node": "upper"}], "edges": chain}
    registry.add(subgraph("shout-words", title="Shout words", description="Each word of a text, in capitals.")(GraphRecord.model_validate(inner)))
    joined = {"nodes": [*inner["nodes"], {"id": "j", "node": "join"}], "edges": [*chain, ["u.result", "j.items"]]}
    registry.add(subgraph("join-words", title="Join words", description="The words of a text in capitals, joined.")(GraphRecord.model_validate(joined)))
    stamped = {"nodes": [{"id": "u", "node": "upper"}, {"id": "k", "node": "upper", "constants": {"text": "k"}}], "edges": []}
    registry.add(subgraph("shout-and-stamp", title="Shout and stamp", description="A text in capitals, and a K.")(GraphRecord.model_validate(stamped)))
    return registry


def _run(name: str, threaded: frozenset[str]) -> Any:
    return _ran(*GRAPHS[name], threaded)


def _ran(nodes: dict[str, str], edges: list[tuple[str, str]], text: str, threaded: frozenset[str]) -> Any:
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": i, "node": n} for i, n in nodes.items()], "edges": [list(edge) for edge in edges]}), _registry(threaded))
    assert not graph.problems, graph.problems
    return graph.run(dict.fromkeys(graph.inputs, text))  # every first node is given the text


CASES = [pytest.param(name, threaded, id=f"{name}, on threads: {', '.join(sorted(threaded)) or 'none'}") for name in GRAPHS for threaded in THREADED]


@pytest.mark.parametrize(("name", "threaded"), CASES)
def test_a_complete_run_holds_a_unit_for_every_instance_and_the_results_of_the_same_run_with_no_threads(name: str, threaded: frozenset[str]) -> None:
    run = _run(name, threaded)
    assert run.status == "complete" and units_missing(run) == []
    assert run.results == _run(name, frozenset()).results


@pytest.mark.parametrize(("name", "threaded"), CASES)
def test_going_on_from_a_complete_run_calls_nothing_and_changes_nothing(name: str, threaded: frozenset[str]) -> None:
    run = _run(name, threaded)
    CALLS.clear()
    further = run.resume()
    assert CALLS == {} and further.status == "complete"
    assert (further.to_json(), further.results) == (run.to_json(), run.results)


def test_a_gather_over_an_items_empty_series_is_called_with_no_items_when_the_series_maker_above_ran_on_a_thread() -> None:
    run = _run("a gather per line of one line with no words", frozenset({"lines"}))
    assert (run.status, run.results["j"]["result"]) == ("complete", [""])


@pytest.mark.parametrize("threaded", [frozenset(), frozenset({"decide", "pieces"})], ids=["no threads", "the decision and the Series on threads"])
def test_a_unit_skipped_by_a_branch_decided_outside_its_items_is_skipped_once_the_items_are_made_while_a_question_beside_it_waits(threaded: frozenset[str]) -> None:
    nodes = {"d": "decide", "w": "pieces", "q": "ask", "t": "tag"}
    run = _ran(nodes, [("w.result", "q.text"), ("d.yes", "t.text"), ("q.result", "t.mark")], "no a b", threaded)
    assert run.status == "waiting" and len(run.waiting) == 3
    assert [missing for missing in units_missing(run) if missing[0] == ("t",)] == []  # skipped, whichever came back first


def test_a_leg_that_ends_with_a_unit_nothing_made_ready_raises_and_hands_on_no_complete_run() -> None:
    nodes, edges, _ = GRAPHS["a gather of a Series with no items"]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": i, "node": n} for i, n in nodes.items()], "edges": [list(edge) for edge in edges]}), _registry(frozenset({"words"})))
    graph._nodes_by_name["w"].downstream.clear()  # the engine's own fault, made by hand: the Series readies nothing
    with pytest.raises(RuntimeError, match=r"no unit for every instance of \[\('j',\)\]"):
        graph.run(dict.fromkeys(graph.inputs, ""))
