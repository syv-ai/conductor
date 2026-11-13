"""A complete run has an outcome for every call of every node, whichever nodes ran on threads, and resuming it
calls nothing and changes nothing."""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    Ask,
    FunctionDefinition,
    Graph,
    GraphRecord,
    NodeRegistry,
    Param,
    Policy,
    RecordGraphInput,
    Series,
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
    time.sleep(0.02)  # on a thread it returns after the other nodes' series
    return Way(no=text) if text.startswith("no") else Way(yes=text)


def tag(text: str, mark: str) -> str:
    _called("tag")
    return f"{mark}:{text}"


def beside(text: str, items: Series[str]) -> str:
    _called("beside")
    return f"{text} of {'+'.join(items)}"


def pieces(text: str) -> Series[str]:
    _called("pieces")
    time.sleep(0.05)  # on a thread it returns after decide
    return text.split()


def ask(text: str) -> str:
    _called("ask")
    raise Ask(f"What about {text}?")


FUNCTIONS = (lines, words, upper, join, decide, tag, beside, pieces, ask)

#: Each graph by what it is: its nodes by id, its edges, and the text every first node of it is given.
GRAPHS: dict[str, tuple[dict[str, str], list[tuple[str, str]], str]] = {
    "a chain": ({"a": "upper", "b": "upper"}, [("a.result", "b.text")], "x"),
    "collecting an empty series": ({"w": "words", "u": "upper", "j": "join"}, [("w.result", "u.text"), ("u.result", "j.items")], ""),
    "collecting per line, one line with no words": (
        {"l": "lines", "w": "words", "u": "upper", "j": "join"}, [("l.result", "w.text"), ("w.result", "u.text"), ("u.result", "j.items")], "a b;;c",
    ),
    "collecting per line, the only line with no words": (
        {"l": "lines", "w": "words", "u": "upper", "j": "join"}, [("l.result", "w.text"), ("w.result", "u.text"), ("u.result", "j.items")], "",
    ),
    "a branch upstream of collecting per line, one line with no words": (
        {"d": "decide", "l": "lines", "w": "words", "t": "tag", "j": "join"},
        [("l.result", "w.text"), ("w.result", "t.text"), ("d.yes", "t.mark"), ("t.result", "j.items")], "a b;;c",
    ),
    "a branch not chosen upstream of collecting per line, one line with no words": (
        {"d": "decide", "l": "lines", "w": "words", "t": "tag", "j": "join"},
        [("l.result", "w.text"), ("w.result", "t.text"), ("d.yes", "t.mark"), ("t.result", "j.items")], "no;;no",
    ),
    "every item's value collected and given to each item": (
        {"w": "words", "u": "upper", "b": "beside"}, [("w.result", "u.text"), ("w.result", "b.text"), ("u.result", "b.items")], "a b c",
    ),
    "a subgraph whose inner Series has no items, per line": ({"l": "lines", "e": "shout-words"}, [("l.result", "e.text")], "a b;;c"),
    "a subgraph per line, a node inside it reading nothing": ({"l": "lines", "e": "shout-and-stamp"}, [("l.result", "e.text")], "a;b"),
    "a subgraph that collects an empty series": ({"e": "join-words", "u": "upper"}, [("e.result", "u.text")], ""),
}

#: The nodes called on threads; every other is called on the thread that called the run.
THREADED: list[frozenset[str]] = [
    frozenset(), frozenset({"lines"}), frozenset({"words"}), frozenset({"lines", "words"}), frozenset({"decide"}), frozenset({"upper", "tag"}), frozenset({"words", "upper"}),
    frozenset({"decide", "lines"}), frozenset(function.__name__ for function in FUNCTIONS),
]


def _registry(threaded: frozenset[str]) -> NodeRegistry:
    registry = NodeRegistry()
    for function in FUNCTIONS:
        name = function.__name__
        policy = {"policy": Policy(concurrency=4)} if name in threaded else {}
        registry.add(node(name, title=name.capitalize(), description=f"The {name} node.", asks=function is ask, **policy)(function))
    shout_words = {"nodes": {"w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
                             "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}}}},
                   "outputs": {"result": {"node": "u", "output": "result"}}}
    registry.add(subgraph("shout-words", title="Shout words", description="Each word of a text, in capitals.")(GraphRecord.model_validate(shout_words)))
    joined = {"nodes": {**shout_words["nodes"], "j": {"node": "join", "inputs": {"items": {"sources": [{"node": "u", "output": "result"}]}}}},
              "outputs": {"result": {"node": "j", "output": "result"}}}
    registry.add(subgraph("join-words", title="Join words", description="The words of a text in capitals, joined.")(GraphRecord.model_validate(joined)))
    stamped = {"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "k": {"node": "upper", "inputs": {"text": {"constant": "k"}}}},
               "outputs": {"u": {"node": "u", "output": "result"}, "k": {"node": "k", "output": "result"}}}
    registry.add(subgraph("shout-and-stamp", title="Shout and stamp", description="A text in capitals, and a K.")(GraphRecord.model_validate(stamped)))
    return registry


def _run(name: str, threaded: frozenset[str]) -> Any:
    return _ran(*GRAPHS[name], threaded)


def _record(nodes: dict[str, str], edges: list[tuple[str, str]], registry: NodeRegistry) -> GraphRecord:
    """The graph record of ``nodes`` and ``edges``, each edge ``(output, input)`` written ``"node.name"``. Every input
    of a node no edge leads into is the graph input of the input's name, so every first node is given the text."""
    inputs: dict[str, dict[str, Any]] = {node_id: {} for node_id in nodes}
    for source, target in edges:
        source_node, output = source.split(".")
        target_node, name = target.split(".")
        inputs[target_node].setdefault(name, {"sources": []})["sources"].append({"node": source_node, "output": output})
    for node_id, node_name in nodes.items():
        if not inputs[node_id]:
            definition = registry[node_name]
            names = definition.inputs if isinstance(definition, FunctionDefinition) else {
                entry.graph_input for inner in definition.graph.nodes.values() for entry in inner.inputs.values() if isinstance(entry, RecordGraphInput)
            }
            inputs[node_id] = {name: {"graph_input": name} for name in names}
    return GraphRecord.model_validate({"nodes": {node_id: {"node": node_name, "inputs": inputs[node_id]} for node_id, node_name in nodes.items()}})


def _ran(nodes: dict[str, str], edges: list[tuple[str, str]], text: str, threaded: frozenset[str]) -> Any:
    registry = _registry(threaded)
    graph = Graph.compile(_record(nodes, edges, registry), registry)
    assert not graph.problems, graph.problems
    return graph.run(dict.fromkeys(graph.inputs, text))


CASES = [pytest.param(name, threaded, id=f"{name}, on threads: {', '.join(sorted(threaded)) or 'none'}") for name in GRAPHS for threaded in THREADED]


@pytest.mark.parametrize(("name", "threaded"), CASES)
def test_a_complete_run_has_an_outcome_for_every_call_and_the_results_of_a_run_with_no_threads(name: str, threaded: frozenset[str]) -> None:
    run = _run(name, threaded)
    assert run.status == "complete" and calls_without_outcome(run) == []
    assert run.results == _run(name, frozenset()).results


@pytest.mark.parametrize(("name", "threaded"), CASES)
def test_resuming_a_complete_run_calls_nothing_and_changes_nothing(name: str, threaded: frozenset[str]) -> None:
    run = _run(name, threaded)
    CALLS.clear()
    further = run.resume()
    assert CALLS == {} and further.status == "complete"
    assert (further.to_json(), further.results) == (run.to_json(), run.results)


def test_a_node_collecting_an_empty_series_gets_an_empty_list_when_the_node_returning_it_ran_on_a_thread() -> None:
    run = _run("collecting per line, the only line with no words", frozenset({"lines"}))
    assert (run.status, run.results["j"]["result"]) == ("complete", [""])


@pytest.mark.parametrize("threaded", [frozenset(), frozenset({"decide", "pieces"})], ids=["no threads", "the decision and the Series on threads"])
def test_a_node_in_a_loop_downstream_of_a_branch_not_chosen_outside_it_is_skipped_per_item_while_other_calls_wait(threaded: frozenset[str]) -> None:
    nodes = {"d": "decide", "w": "pieces", "q": "ask", "t": "tag"}
    run = _ran(nodes, [("w.result", "q.text"), ("d.yes", "t.text"), ("q.result", "t.mark")], "no a b", threaded)
    assert run.status == "waiting" and len(run.waiting) == 3
    assert "t" not in calls_without_outcome(run)  # skipped, whichever node returned first


def test_a_run_that_ends_with_a_call_never_made_ready_raises_rather_than_return_a_complete_run() -> None:
    nodes, edges, _ = GRAPHS["collecting an empty series"]
    registry = _registry(frozenset({"words"}))
    graph = Graph.compile(_record(nodes, edges, registry), registry)
    run = graph.with_inputs(dict.fromkeys(graph.inputs, ""))
    # a bug in the engine, made by hand: the series makes no downstream node ready. The run gets a flat graph of its
    # own with a copy of the node, so the graph's flat graph stays whole.
    series = dataclasses.replace(run._flat.function_nodes["w"], downstream=[])
    broken = dataclasses.replace(run._flat, function_nodes={**run._flat.function_nodes, "w": series}, nodes={**run._flat.nodes, "w": series})
    with pytest.raises(RuntimeError, match=r"no outcome for every call of \[\('j',\)\]"):
        dataclasses.replace(run, _flat=broken).resume()
