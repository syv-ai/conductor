"""Tests that a run that finishes really is finished.

When a run is complete, every call of every node has an outcome. That must hold whichever nodes ran on threads of
their own, so each graph here is run with several choices of threaded nodes. Resuming a complete run must then call
nothing and change nothing.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, NamedTuple

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
    make_graph,
    node,
    subgraph,
)

from tests.test_core.nodes import calls_without_outcome

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
    time.sleep(0.02)  # Slow, so on a thread it returns after the other nodes' Series.
    return Way(no=text) if text.startswith("no") else Way(yes=text)


def tag(text: str, mark: str) -> str:
    _called("tag")
    return f"{mark}:{text}"


def beside(text: str, items: Series[str]) -> str:
    _called("beside")
    return f"{text} of {'+'.join(items)}"


def pieces(text: str) -> Series[str]:
    _called("pieces")
    time.sleep(0.05)  # Slower still, so on a thread it returns after decide.
    return text.split()


def ask(text: str) -> str:
    _called("ask")
    raise Ask(f"What about {text}?")


FUNCTIONS = (lines, words, upper, join, decide, tag, beside, pieces, ask)


class Case(NamedTuple):
    """One graph to run, and the text to run it with.

    Attributes:
        nodes: Each node's definition id, by node id.
        edges: The edges, as ``("node.output", "node.input")`` pairs.
        text: The value given to every graph input.
        inputs: Graph inputs that feed two nodes at once, for when two nodes at the start both have an input called
            ``text``.
    """

    nodes: dict[str, str]
    edges: list[tuple[str, str]]
    text: str
    inputs: Mapping[str, list[str]] = MappingProxyType({})


#: The graphs under test, by a name that says what each one is.
GRAPHS: dict[str, Case] = {
    "a chain": Case({"a": "upper", "b": "upper"}, [("a.result", "b.text")], "x"),
    "collecting an empty series": Case(
        {"w": "words", "u": "upper", "j": "join"},
        [("w.result", "u.text"), ("u.result", "j.items")],
        "",
    ),
    "collecting per line, one line with no words": Case(
        {"l": "lines", "w": "words", "u": "upper", "j": "join"},
        [("l.result", "w.text"), ("w.result", "u.text"), ("u.result", "j.items")],
        "a b;;c",
    ),
    "collecting per line, the only line with no words": Case(
        {"l": "lines", "w": "words", "u": "upper", "j": "join"},
        [("l.result", "w.text"), ("w.result", "u.text"), ("u.result", "j.items")],
        "",
    ),
    "a branch upstream of collecting per line, one line with no words": Case(
        {"d": "decide", "l": "lines", "w": "words", "t": "tag", "j": "join"},
        [
            ("l.result", "w.text"),
            ("w.result", "t.text"),
            ("d.yes", "t.mark"),
            ("t.result", "j.items"),
        ],
        "a b;;c",
        {"text": ["d.text", "l.text"]},
    ),
    "a branch not chosen upstream of collecting per line, one line with no words": Case(
        {"d": "decide", "l": "lines", "w": "words", "t": "tag", "j": "join"},
        [
            ("l.result", "w.text"),
            ("w.result", "t.text"),
            ("d.yes", "t.mark"),
            ("t.result", "j.items"),
        ],
        "no;;no",
        {"text": ["d.text", "l.text"]},
    ),
    "every item's value collected and given to each item": Case(
        {"w": "words", "u": "upper", "b": "beside"},
        [("w.result", "u.text"), ("w.result", "b.text"), ("u.result", "b.items")],
        "a b c",
    ),
    "a subgraph whose inner Series has no items, per line": Case(
        {"l": "lines", "e": "shout-words"}, [("l.result", "e.text")], "a b;;c"
    ),
    "a subgraph per line, a node inside it reading nothing": Case(
        {"l": "lines", "e": "shout-and-stamp"}, [("l.result", "e.text")], "a;b"
    ),
    "a subgraph that collects an empty series": Case(
        {"e": "join-words", "u": "upper"}, [("e.result", "u.text")], ""
    ),
}

#: The sets of nodes to call on threads. A node not in the set is called on the thread that started the run.
THREADED: list[frozenset[str]] = [
    frozenset(),
    frozenset({"lines"}),
    frozenset({"words"}),
    frozenset({"lines", "words"}),
    frozenset({"decide"}),
    frozenset({"upper", "tag"}),
    frozenset({"words", "upper"}),
    frozenset({"decide", "lines"}),
    frozenset(function.__name__ for function in FUNCTIONS),
]


def _registry(threaded: frozenset[str]) -> NodeRegistry:
    registry = NodeRegistry()
    for function in FUNCTIONS:
        name = function.__name__
        policy = {"policy": Policy(concurrency=4)} if name in threaded else {}
        registry.add(
            node(
                name,
                title=name.capitalize(),
                description=f"The {name} node.",
                asks=function is ask,
                **policy,
            )(function)
        )
    shout_words = {
        "nodes": {
            "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
            "u": {
                "node": "upper",
                "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}},
            },
        },
        "outputs": {"result": {"node": "u", "output": "result"}},
    }
    registry.add(
        subgraph(
            "shout-words", title="Shout words", description="Each word of a text, in capitals."
        )(GraphRecord.model_validate(shout_words))
    )
    joined = {
        "nodes": {
            **shout_words["nodes"],
            "j": {
                "node": "join",
                "inputs": {"items": {"sources": [{"node": "u", "output": "result"}]}},
            },
        },
        "outputs": {"result": {"node": "j", "output": "result"}},
    }
    registry.add(
        subgraph(
            "join-words", title="Join words", description="The words of a text in capitals, joined."
        )(GraphRecord.model_validate(joined))
    )
    stamped = {
        "nodes": {
            "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
            "k": {"node": "upper", "inputs": {"text": {"constant": "k"}}},
        },
        "outputs": {"u": {"node": "u", "output": "result"}, "k": {"node": "k", "output": "result"}},
    }
    registry.add(
        subgraph(
            "shout-and-stamp", title="Shout and stamp", description="A text in capitals, and a K."
        )(GraphRecord.model_validate(stamped))
    )
    return registry


def _run(name: str, threaded: frozenset[str]) -> Any:
    case = GRAPHS[name]
    return _ran(case.nodes, case.edges, case.text, threaded, case.inputs)


def _record(
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    registry: NodeRegistry,
    inputs: Mapping[str, list[str]] = MappingProxyType({}),
) -> GraphRecord:
    """Returns a graph record of ``nodes`` and ``edges``, made with ``make_graph``.

    ``nodes`` maps each node id to a definition id in ``registry``. The tests read each node's result from
    ``run.results``.
    """
    return make_graph(
        *((node_id, registry[name]) for node_id, name in nodes.items()), edges=edges, inputs=inputs
    )


def _ran(
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    text: str,
    threaded: frozenset[str],
    inputs: Mapping[str, list[str]] = MappingProxyType({}),
) -> Any:
    registry = _registry(threaded)
    graph = Graph.compile(_record(nodes, edges, registry, inputs), registry)
    assert not graph.problems, graph.problems
    return graph.run(dict.fromkeys(graph.inputs, text))


CASES = [
    pytest.param(name, threaded, id=f"{name}, on threads: {', '.join(sorted(threaded)) or 'none'}")
    for name in GRAPHS
    for threaded in THREADED
]


@pytest.mark.parametrize(("name", "threaded"), CASES)
def test_a_complete_run_has_an_outcome_for_every_call_and_the_results_of_a_run_with_no_threads(
    name: str, threaded: frozenset[str]
) -> None:
    run = _run(name, threaded)
    assert run.status == "complete" and calls_without_outcome(run) == []
    assert run.results == _run(name, frozenset()).results


@pytest.mark.parametrize(("name", "threaded"), CASES)
def test_resuming_a_complete_run_calls_nothing_and_changes_nothing(
    name: str, threaded: frozenset[str]
) -> None:
    run = _run(name, threaded)
    CALLS.clear()
    further = run.resume()
    assert CALLS == {} and further.status == "complete"
    assert (further.to_json(), further.results) == (run.to_json(), run.results)


def test_a_node_collecting_an_empty_series_gets_an_empty_list_when_the_node_returning_it_ran_on_a_thread() -> (
    None
):
    run = _run("collecting per line, the only line with no words", frozenset({"lines"}))
    assert (run.status, run.results["j"]["result"]) == ("complete", [""])


@pytest.mark.parametrize(
    "threaded",
    [frozenset(), frozenset({"decide", "pieces"})],
    ids=["no threads", "the decision and the Series on threads"],
)
def test_a_node_in_a_loop_downstream_of_a_branch_not_chosen_outside_it_is_skipped_per_item_while_other_calls_wait(
    threaded: frozenset[str],
) -> None:
    nodes = {"d": "decide", "w": "pieces", "q": "ask", "t": "tag"}
    run = _ran(
        nodes,
        [("w.result", "q.text"), ("d.yes", "t.text"), ("q.result", "t.mark")],
        "no a b",
        threaded,
        {"text": ["d.text", "w.text"]},
    )
    assert run.status == "waiting" and len(run.waiting) == 3
    assert "t" not in calls_without_outcome(run)  # Skipped, whichever node returned first.
