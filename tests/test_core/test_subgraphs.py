"""A subgraph is one node of the outer graph, and its inner nodes behave as they would in one graph.

Most tests run a graph with a subgraph node ``e`` and assert the same result as the same nodes in one graph: for
``shout-inner`` (``upper -> exclaim``) inside a loop over ``words``, ``e``'s output ``result`` on ``"ab cd"`` is
``["AB!", "CD!"]``, as ``exclaim.result`` is in the graph ``words -> upper -> exclaim``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from asyncio import Event
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeError,
    NodeRegistry,
    Param,
    Policy,
    Run,
    Series,
    Table,
    make_graph,
    node,
    subgraph,
)
from conductor.events import (
    NodeComplete,
    NodeFailed,
    NodeRetry,
    NodeStart,
    NodeWaiting,
    RunCheckpoint,
    RunEnd,
)

CALLS: Counter[str] = Counter()
AT_ONCE = {"now": 0, "peak": 0}
_LOCK = threading.Lock()
_FAILED_ONCE: set[str] = set()


@pytest.fixture(autouse=True)
def _reset_counters():
    CALLS.clear()
    _FAILED_ONCE.clear()
    AT_ONCE.update(now=0, peak=0)


def _node(node_id: str, **options: Any):
    return node(node_id, title=node_id.capitalize(), description=f"The {node_id} node.", **options)


@_node("words")
def words(text: str) -> Series[str]:
    CALLS["words"] += 1
    return text.split()


@_node("sentences")
def sentences(text: str) -> Series[str]:
    return text.split(";")


@_node("chars")
def chars(text: str) -> Series[str]:
    return list(text)


@_node("upper")
def upper(text: str) -> str:
    CALLS["upper"] += 1
    return text.upper()


@_node("exclaim")
def exclaim(text: str) -> str:
    CALLS["exclaim"] += 1
    return text + "!"


@_node("greet")
def greet(text: str, greeting: str = "hello") -> str:
    return f"{greeting} {text}"


@_node("count")
def count(items: Series[str]) -> int:
    return len(items)


@_node("ask", asks=True)
def ask(text: str) -> str:
    CALLS["ask"] += 1
    raise Ask(f"What about {text}?")


@_node("fetch", policy=Policy(concurrency=2))
def fetch(text: str) -> str:
    with _LOCK:
        CALLS["fetch"] += 1
        AT_ONCE["now"] += 1
        AT_ONCE["peak"] = max(AT_ONCE["peak"], AT_ONCE["now"])
    time.sleep(0.15)
    with _LOCK:
        AT_ONCE["now"] -= 1
    return text + "?"


@_node("flaky", policy=Policy(retries=2, retry_delay=0.01))
def flaky(text: str) -> str:
    CALLS["flaky"] += 1
    if text not in _FAILED_ONCE:
        _FAILED_ONCE.add(text)
        raise ExternalError("the service was away")
    return text + "~"


@_node("stuck", policy=Policy(attempt_timeout=0.1))
def stuck(text: str) -> str:
    CALLS["stuck"] += 1
    if text == "slow":
        time.sleep(0.5)
    return text


@_node("boom")
def boom(text: str) -> str:
    CALLS["boom"] += 1
    if text == "BAD":
        raise NodeError("this one is bad")
    return text


@_node("stamp", policy=Policy(reuse_when_inputs_match=False))
def stamp(text: str) -> str:
    CALLS["stamp"] += 1
    return text


@_node("stagger", policy=Policy(concurrency=4, reuse_when_inputs_match=False))
def stagger(text: str) -> str:
    """Returns later the sooner it was called, so the calls of one ``resume()`` finish in reverse order."""
    with _LOCK:
        CALLS["stagger"] += 1
        order = CALLS["stagger"]
    time.sleep(max(0.0, 0.2 - 0.05 * order))
    return text


@_node("maybe")
def maybe(text: str) -> str | None:
    return None if text == "none" else text


@dataclass(frozen=True)
class Way:
    short: Annotated[str, Param(exclusive_group="way")] = SKIPPED
    long: Annotated[str, Param(exclusive_group="way")] = SKIPPED


@_node("decide")
def decide(text: str) -> Way:
    return Way(short=text) if len(text) < 3 else Way(long=text)


def _columns(names: str) -> dict[str, Any]:
    return {name: str for name in names.split(",")}


@_node("columns", compute_outputs=_columns)
def columns(text: str, names: str = "a") -> dict[str, Any]:
    return dict.fromkeys(names.split(","), text)


@_node("table")
def table(text: str) -> Table:
    raise NotImplementedError


@_node("letters")
def letters(text: str) -> Table:
    return pl.DataFrame({"letter": list(text)})


@_node("height")
def height(table: Table) -> int:
    return table.height


@_node("record")
def record(text: str) -> Any:
    return {"letter": text}


@_node("sheets")
def sheets(text: str) -> Series[Table]:
    return [pl.DataFrame({"letter": list(word)}) for word in text.split()]


@_node("note")
def note(text: str) -> None:
    CALLS["note"] += 1


@_node("show")
def show(value: Any) -> str:
    return str(value)


@_node("pair")
def pair(first: Series[str], second: Series[str]) -> int:
    return len(first)


def _sources(node_id: str, output: str) -> dict[str, Any]:
    return {"sources": [{"node": node_id, "output": output}]}


def chain(*nodes: tuple[str, str], inputs: Sequence[str] = ("text",), outputs: Sequence[str] = ("result",)) -> dict[str, Any]:
    """Nodes in a line, each reading the ``result`` of the one before into its ``text``. Each of ``inputs`` is an input
    of the first node and a graph input of its name; each of ``outputs`` is an output of the last node and a graph
    output of its name."""
    (first, _), (last, _) = nodes[0], nodes[-1]
    records = {node_id: {"node": definition_id, "inputs": {"text": _sources(before[0], "result")}} for before, (node_id, definition_id) in zip(nodes, nodes[1:], strict=False)}
    return {
        "nodes": {first: {"node": nodes[0][1], "inputs": {name: {"graph_input": name} for name in inputs}}, **records},
        "outputs": {name: {"node": last, "output": name} for name in outputs},
    }


def side_by_side(*nodes: tuple[str, str]) -> dict[str, Any]:
    """Nodes side by side, each given the graph input ``text``, and each node's ``result`` a graph output named by
    the node."""
    return {
        "nodes": {node_id: {"node": definition_id, "inputs": {"text": {"graph_input": "text"}}} for node_id, definition_id in nodes},
        "outputs": {node_id: {"node": node_id, "output": "result"} for node_id, _ in nodes},
    }


#: Each subgraph's graph record, by the node definition id it is registered under.
INNER = {
    "shout-inner": chain(("up", "upper"), ("ex", "exclaim")),
    "greet-inner": {
        "nodes": {"g": {"node": "greet", "inputs": {"text": {"graph_input": "text"}, "greeting": {"graph_input": "greeting", "default": "hello"}}}},
        "outputs": {"result": {"node": "g", "output": "result"}},
    },
    "asking-inner": chain(("up", "upper"), ("q", "ask")),
    "fetch-inner": chain(("f", "fetch"), ("up", "upper")),
    "flaky-inner": chain(("f", "flaky")),
    "stuck-inner": chain(("s", "stuck"), ("up", "upper")),
    "boom-inner": chain(("up", "upper"), ("b", "boom")),
    "maybe-inner": chain(("m", "maybe"), ("up", "upper")),
    "lines-inner": chain(("w", "words"), ("q", "ask")),  # a question per word of what it is given
    "fetch-words-inner": chain(("w", "words"), ("f", "fetch")),  # a call on a thread per word
    "upper-fetch-inner": chain(("up", "upper"), ("f", "fetch")),  # its last node is the one on threads
    "two-asks-inner": side_by_side(("f", "asking-inner"), ("g", "asking-inner")),
    "three-asks-inner": side_by_side(("f", "asking-inner"), ("g", "asking-inner"), ("h", "asking-inner")),
    "wrapping": chain(("in", "asking-inner")),  # a subgraph inside a subgraph
    "deep-inner": chain(("w", "words"), ("c", "chars"), ("up", "upper")),  # its output is inside two loops
    "series-per-item-inner": chain(("w", "words"), ("c", "chars")),  # a Series returned per word
    "table-per-item-inner": chain(("w", "words"), ("t", "table")),
    "letters-per-word-inner": chain(("w", "words"), ("t", "letters")),
    "record-per-word-inner": chain(("w", "words"), ("r", "record")),  # each word's value is declared Any
    "columns-inner": chain(("c", "columns"), outputs=("a",)),  # its first node has an input a hook reads, and an output a hook adds
    "broken-inner": chain(("up", "upper"), ("x", "no-such-node")),
    "stamped-inner": chain(("s", "stamp"), ("up", "upper")),
    "staggered-inner": chain(("s", "stagger"), ("q", "ask")),
    "count-inner": chain(("n", "count"), inputs=("items",)),  # its one node collects
    "quiet-inner": chain(("n", "note"), outputs=()),  # it has no output
    "self-a": chain(("x", "self-b")),
    "self-b": chain(("y", "self-a")),
    "deciding-inner": {
        "nodes": {"d": {"node": "decide", "inputs": {"text": {"graph_input": "text"}}}, "s": {"node": "exclaim", "inputs": {"text": _sources("d", "short")}}, "l": {"node": "upper", "inputs": {"text": _sources("d", "long")}}},
        "outputs": {"short": {"node": "s", "output": "result"}, "long": {"node": "l", "output": "result"}},
    },
    "words-deciding-inner": {
        "nodes": {
            "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}}, "d": {"node": "decide", "inputs": {"text": _sources("w", "result")}},
            "s": {"node": "exclaim", "inputs": {"text": _sources("d", "short")}}, "l": {"node": "upper", "inputs": {"text": _sources("d", "long")}},
        },
        "outputs": {"short": {"node": "s", "output": "result"}, "long": {"node": "l", "output": "result"}},
    },
    "deciding-twice-inner": {
        "nodes": {"d": {"node": "decide", "inputs": {"text": {"graph_input": "text"}}}, "s": {"node": "exclaim", "inputs": {"text": _sources("d", "short")}}, "s2": {"node": "upper", "inputs": {"text": _sources("d", "short")}}},
        "outputs": {"exclaimed": {"node": "s", "output": "result"}, "shouted": {"node": "s2", "output": "result"}},
    },
}

REGISTRY = NodeRegistry()
for _definition in (words, sentences, chars, upper, exclaim, greet, count, ask, fetch, flaky, stuck, boom, maybe, decide, columns, table, stamp, stagger, letters, height, sheets, record, note, show, pair):
    REGISTRY.add(_definition)
for _id, _record in INNER.items():
    REGISTRY.add(subgraph(_id, title=_id.capitalize(), description=f"The {_id} node.")(GraphRecord.model_validate(_record)))


def compile_graph(
    nodes: Mapping[str, str],
    edges: Sequence[tuple[str, str]] = (),
    *,
    constants: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
    inputs: Mapping[str, str] | None = None,
    outputs: Mapping[str, str] | None = None,
):
    """The compiled ``make_graph`` of ``nodes`` (node id to the id of its definition in ``REGISTRY``), joined by
    exactly ``edges``, with these ``constants``, ``defaults``, graph ``inputs`` and ``outputs``."""
    record = make_graph(
        *((node_id, REGISTRY[definition_id]) for node_id, definition_id in nodes.items()),
        edges=list(edges), constants=constants or {}, defaults=defaults or {}, inputs=inputs or {}, outputs=outputs or {},
    )
    return Graph.compile(record, REGISTRY)


def inner_nodes(inner: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """The nodes of the subgraph ``inner``'s record and the edges between them, to draw into another graph."""
    nodes = {node_id: record["node"] for node_id, record in INNER[inner]["nodes"].items()}
    edges = [
        (f"{source['node']}.{source['output']}", f"{node_id}.{name}")
        for node_id, record in INNER[inner]["nodes"].items() for name, entry in record["inputs"].items() for source in entry.get("sources", ())
    ]
    return nodes, edges


def inside_a_loop(inner: str, splitter: str = "words"):
    """``splitter`` wired into the subgraph ``inner``, so the subgraph node is called once per item."""
    graph = compile_graph({"w": splitter, "e": inner}, [("w.result", "e.text")])
    assert not graph.problems, graph.problems
    return graph


def in_one_graph(inner: str, splitter: str = "words"):
    """The inner nodes of ``inner`` in one graph after ``splitter``, with no subgraph: what ``inside_a_loop``
    must behave as."""
    record = INNER[inner]
    first = next(node_id for node_id, node_record in record["nodes"].items() if "text" in node_record["inputs"] and "graph_input" in node_record["inputs"]["text"])
    nodes, edges = inner_nodes(inner)
    graph = compile_graph({"outer": splitter, **nodes}, [("outer.result", f"{first}.text"), *edges])
    assert not graph.problems, graph.problems
    return graph


def stream(run, cancel_when=None, **options):
    async def go():
        cancel, events = Event(), []
        async for event in run.stream(cancel=cancel, **options):
            events.append(event)
            if cancel_when is not None and cancel_when(event):
                cancel.set()
        return events

    return asyncio.run(go())


def problems(graph) -> list[tuple[str, tuple]]:
    return [(p.type, p.loc) for p in graph.problems]


# -- compiling: a subgraph is one node with its inner nodes' inputs and outputs --------------------


def test_a_subgraph_node_offers_the_inputs_and_outputs_its_record_declares():
    graph = compile_graph({"e": "shout-inner"})
    assert list(graph.inputs) == ["text"]
    described = graph.describe().nodes["e"]
    assert [i.field.name for i in described.inputs] == ["text"] and [o.name for o in described.outputs] == ["result"]
    two = compile_graph({"a": "shout-inner", "b": "shout-inner"}, [("a.result", "b.text")])
    assert two.is_runnable and two._nodes["a"].inner is two._nodes["b"].inner  # compiled once for both


def test_an_edge_reads_a_subgraph_output_by_its_declared_name():
    graph = compile_graph({"e": "columns-inner", "u": "upper"}, [("e.a", "u.text")])
    assert graph.is_runnable and graph.outputs == ["result"]
    assert problems(compile_graph({"e": "shout-inner", "u": "upper"}, [("e.ex", "u.text")])) == [("unknown_output", ("u", "text"))]


def test_an_unwired_required_input_of_a_subgraph_node_is_a_problem_as_on_any_node():
    record = {  # written out: make_graph would make the open e.text a graph input
        "nodes": {
            "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
            "e": {"node": "greet-inner", "inputs": {"greeting": _sources("w", "result")}},
            "x": {"node": "shout-inner", "inputs": {"text": {"graph_input": "text"}}},
        },
        "outputs": {"e": {"node": "e", "output": "result"}, "x": {"node": "x", "output": "result"}},
    }
    assert problems(Graph.compile(record, REGISTRY)) == [("input_has_no_value", ("e", "text"))]


def test_an_inner_input_a_hook_reads_is_set_inside_and_not_offered():
    assert list(compile_graph({"e": "columns-inner"}).inputs) == ["text"]
    with_constant = compile_graph({"e": "columns-inner"}, constants={"e.names": "a,b"})
    assert problems(with_constant) == [("unknown_input", ("e", "names"))]


def test_a_subgraphs_problems_have_the_subgraph_nodes_id_in_front():
    assert problems(inside_a_loop_with_problems("broken-inner")) == [("unknown_node", ("e", "x"))]
    described = inside_a_loop_with_problems("broken-inner").describe().nodes["e"]
    assert not described.found and [p.loc for p in described.problems] == [("e", "x")]


def inside_a_loop_with_problems(inner: str):
    return compile_graph({"w": "words", "e": inner}, [("w.result", "e.text")])


def test_a_subgraph_that_contains_itself_is_refused_at_the_node_that_closes_the_circle():
    assert problems(compile_graph({"e": "self-a"})) == [("subgraph_contains_itself", ("e", "x", "y"))]


def test_an_output_returned_inside_loops_is_offered_as_one_series_per_loop():
    deep = compile_graph({"e": "deep-inner", "n": "count", "x": "exclaim"}, [("e.result", "n.items"), ("e.result", "x.text")], outputs={"n": "n.result", "x": "x.result"})
    assert deep.is_runnable and (deep.runs_on["n"].series, deep.runs_on["x"].series) == ("e.w.result", "e.c.result")  # each loop is named by its series inside
    described = {inner: compile_graph({"e": inner}).describe().nodes["e"].outputs[0] for inner in ("lines-inner", "deep-inner", "series-per-item-inner")}
    assert [(output.is_series, output.type) for output in described.values()] == [(True, "str"), (True, "Series[str]"), (True, "Series[str]")]
    assert [output.loops for output in described.values()] == [["w.result"], ["w.result", "c.result"], ["w.result", "c.result"]]


def test_a_table_returned_per_item_inside_a_subgraph_is_offered_as_a_series_of_tables():
    graph = compile_graph({"e": "table-per-item-inner"})
    tables = graph.describe().nodes["e"].outputs[0]
    assert graph.problems == [] and (tables.is_series, tables.type, tables.loops) == (True, "Table", ["w.result"])


def test_a_branch_inside_a_subgraph_is_a_branch_of_the_subgraph_node_named_by_its_inner_node():
    last = compile_graph({"e": subgraph_of("decide")})
    assert last.exclusive_groups == {"e": {"d.way": ["long", "short"]}}
    joined = compile_graph({"e": subgraph_of("decide"), "j": "exclaim"}, [("e.short", "j.text"), ("e.long", "j.text")])
    assert joined.is_runnable


def test_two_outputs_on_opposite_branches_inside_a_subgraph_may_share_one_input_as_in_one_graph():
    inside = compile_graph({"w": "words", "e": "deciding-inner", "j": "exclaim", "n": "count"}, [("w.result", "e.text"), ("e.short", "j.text"), ("e.long", "j.text"), ("j.result", "n.items")])
    inner, inner_edges = inner_nodes("deciding-inner")
    one_graph = compile_graph({"w": "words", **inner, "j": "exclaim", "n": "count"}, [("w.result", "d.text"), *inner_edges, ("s.result", "j.text"), ("l.result", "j.text"), ("j.result", "n.items")])
    assert problems(inside) == problems(one_graph) == []
    ran, ran_flat = inside.run(text="ab long cd"), one_graph.run(text="ab long cd")
    assert ran.results["j"] == ran_flat.results["j"] == {"result": ["ab!!", "LONG!", "cd!!"]}
    assert ran.results["n"] == ran_flat.results["n"] == {"result": 3}
    same_branch = compile_graph({"e": "deciding-twice-inner", "j": "exclaim"}, [("e.exclaimed", "j.text"), ("e.shouted", "j.text")])
    assert problems(same_branch) == [("several_edges_into_one_input", ("j", "text"))]  # both need d.short: a call could get two values
    nested = compile_graph({"e": subgraph_of("deciding-inner"), "j": "exclaim"}, [("e.short", "j.text"), ("e.long", "j.text")])
    assert problems(nested) == [] and nested.run(text="ab").results["j"] == {"result": "ab!!"}


def test_two_collected_branches_of_a_loop_inside_a_subgraph_cannot_share_one_input_as_in_one_graph():
    collecting = [("n1.result", "g.value"), ("n2.result", "g.value")]
    nodes = {"n1": "count", "n2": "count", "g": "show"}
    inside = compile_graph({"e": "words-deciding-inner", **nodes}, [("e.short", "n1.items"), ("e.long", "n2.items"), *collecting])
    inner, inner_edges = inner_nodes("words-deciding-inner")
    one_graph = compile_graph({**inner, **nodes}, [*inner_edges, ("s.result", "n1.items"), ("l.result", "n2.items"), *collecting])
    # n1 and n2 are each called whichever branch d chose for a word, so both have a value and g.value would get two
    assert problems(inside) == problems(one_graph) == [("several_edges_into_one_input", ("g", "value"))]


def test_two_lists_collected_from_opposite_branches_inside_a_subgraph_are_refused_as_in_one_graph():
    inside = compile_graph({"w": "words", "e": "deciding-inner", "p": "pair"}, [("w.result", "e.text"), ("e.short", "p.first"), ("e.long", "p.second")])
    inner, inner_edges = inner_nodes("deciding-inner")
    one_graph = compile_graph({"w": "words", **inner, "p": "pair"}, [("w.result", "d.text"), *inner_edges, ("s.result", "p.first"), ("l.result", "p.second")])
    # a word is short or long, so it is left out of one list; p takes one item of each, and no word is in both
    assert problems(inside) == problems(one_graph) == [("lists_from_different_loops", ("p",))]


def subgraph_of(node_id: str) -> str:
    """The id of a subgraph of the one node ``node_id``, whose ``text`` it is given and whose ``short`` and ``long``
    it returns, registered when first asked for."""
    subgraph_id = f"only-{node_id}"
    if subgraph_id not in REGISTRY:
        record = chain(("d", node_id), outputs=("short", "long"))
        REGISTRY.add(subgraph(subgraph_id, title=subgraph_id.capitalize(), description=f"The {subgraph_id} node.")(GraphRecord.model_validate(record)))
    return subgraph_id


# -- running: once and per item, and constants, defaults and edges on a subgraph node --------------


def test_a_subgraph_runs_once_or_per_item_as_its_nodes_do_in_one_graph():
    run = compile_graph({"e": "shout-inner"}).run(text="ab")
    assert run.status == "complete" and run.outputs == {"result": "AB!"}
    graph = inside_a_loop("shout-inner")
    assert graph.runs_on["e"].mode == "per_item"
    run = graph.run(text="ab cd")
    assert run.status == "complete" and run.results["e"] == {"result": ["AB!", "CD!"]}
    assert run.results["e"]["result"] == in_one_graph("shout-inner").run(text="ab cd").results["ex"]["result"]
    two = compile_graph({"a": "shout-inner", "b": "shout-inner"}, [("a.result", "b.text")])
    assert two.run(text="x").outputs == {"result": "X!!"}


def test_a_constant_a_default_and_an_edge_on_a_subgraph_node_act_as_on_any_node():
    graph = compile_graph({"w": "words", "e": "greet-inner"}, [("w.result", "e.text")], constants={"e.greeting": "hej"})
    assert graph.run(text="ab cd").results["e"]["result"] == ["hej ab", "hej cd"]
    first = compile_graph({"e": "greet-inner"}, defaults={"e.greeting": "hi"})
    assert first.inputs["greeting"].field.default == "hi"
    assert first.run(text="ab").outputs == {"result": "hi ab"}  # the subgraph node's default, offered to the caller
    assert first.run(text="ab", greeting="yo").outputs == {"result": "yo ab"}
    assert inside_a_loop("greet-inner").run(text="ab").results["e"]["result"] == ["hello ab"]  # the default greet-inner's record gives greeting
    wired = compile_graph({"w": "words", "g": "upper", "e": "greet-inner"}, [("w.result", "e.text"), ("w.result", "g.text"), ("g.result", "e.greeting")])
    assert wired.run(text="ab cd").results["e"]["result"] == ["AB ab", "CD cd"]


def test_an_output_two_loops_deep_is_the_same_nested_list_alone_and_inside_a_subgraph():
    alone = Graph.compile(GraphRecord.model_validate(INNER["deep-inner"]), REGISTRY).run(text="ab cd").outputs["result"]
    assert alone == [["A", "B"], ["C", "D"]]  # two loops deep: a list for each word, in item order
    graph = compile_graph({"e": "deep-inner", "n": "count", "x": "exclaim"}, [("e.result", "n.items"), ("e.result", "x.text")], outputs={"n": "n.result", "x": "x.result"})
    run = graph.run(text="ab cd")
    assert run.results["e"]["result"] == alone and run.results["n"]["result"] == [2, 2]  # count is called once per word
    assert run.results["x"]["result"] == [["A!", "B!"], ["C!", "D!"]]
    assert compile_graph({"e": "series-per-item-inner"}).run(text="ab cd").outputs == {"result": [["a", "b"], ["c", "d"]]}


def test_a_branch_not_chosen_inside_a_subgraph_skips_the_nodes_downstream_as_in_one_graph():
    graph = compile_graph({"w": "words", "e": subgraph_of("decide"), "s": "exclaim", "l": "upper"}, [("w.result", "e.text"), ("e.short", "s.text"), ("e.long", "l.text")], outputs={"s": "s.result", "l": "l.result"})
    run = graph.run(text="ab long")
    assert run.status == "complete" and run.results["s"]["result"] == ["ab!", None] and run.results["l"]["result"] == [None, "LONG"]
    assert run.results["e"] == {"short": ["ab", None], "long": [None, "long"]}
    inside = compile_graph({"w": "words", "e": "deciding-inner", "n": "count"}, [("w.result", "e.text"), ("e.short", "n.items")])
    assert inside.run(text="ab long cd").results["n"]["result"] == 2  # the item whose branch was not chosen is left out of what count collects


def test_an_item_whose_branch_was_not_chosen_outside_skips_the_inner_nodes_for_that_item():
    graph = compile_graph({"w": "words", "d": "decide", "e": "shout-inner", "n": "count"}, [("w.result", "d.text"), ("d.short", "e.text"), ("e.result", "n.items")])
    run = graph.run(text="ab long cd")
    assert run.status == "complete" and run.results["e"] == {"result": ["AB!", None, "CD!"]}
    assert run.results["n"]["result"] == 2 and run.not_called == []  # skipped, not None: the item is left out of what count collects
    assert CALLS["upper"] == 2 and CALLS["exclaim"] == 2


def test_a_series_input_inside_a_subgraph_collects_the_loop_its_subgraph_node_is_wired_to():
    graph = compile_graph({"w": "words", "up": "upper", "e": "count-inner"}, [("w.result", "up.text"), ("up.result", "e.items")])
    assert graph.runs_on["e"].mode == "once"
    assert graph.run(text="ab cd ef").outputs == {"result": 3}
    per_document = compile_graph({"s": "sentences", "w": "words", "e": "count-inner"}, [("s.result", "w.text"), ("w.result", "e.items")])
    assert per_document.run(text="a b;c").outputs == {"result": [2, 1]}


def test_an_inner_node_given_none_is_listed_as_not_called_by_its_node_path():
    run = inside_a_loop("maybe-inner").run(text="a none")
    assert run.status == "complete" and run.results["e"]["result"] == ["A", None]
    assert [(n.loc, n.input) for n in run.not_called] == [(("e", 1, "up"), "text")]
    none_in = compile_graph({"w": "words", "m": "maybe", "e": "shout-inner"}, [("w.result", "m.text"), ("m.result", "e.text")]).run(text="a none")
    assert none_in.results["e"]["result"] == ["A!", None]
    assert [(n.loc, n.input) for n in none_in.not_called] == [(("e", 1, "up"), "text"), (("e", 1, "ex"), "text")]  # the nodes inside, as in one graph
    one_graph = compile_graph({"w": "words", "m": "maybe", "up": "upper", "ex": "exclaim"}, [("w.result", "m.text"), ("m.result", "up.text"), ("up.result", "ex.text")]).run(text="a none")
    assert [(n.loc, n.input) for n in one_graph.not_called] == [(("up", 1), "text"), (("ex", 1), "text")]


def test_the_item_limit_applies_inside_a_subgraph_and_is_raised_on_resume():
    graph = inside_a_loop("deep-inner")
    run = graph.run(text="ab cdef", max_items=3)
    assert run.status == "stopped" and run.state.cause.type == "item_limit" and run.state.cause.loc == ("e", 1, "c")
    resumed = Run.from_json(run.to_json(), graph).with_max_items(10).resume()
    assert resumed.status == "complete" and resumed.results["e"]["result"] == [[["A", "B"]], [["C", "D", "E", "F"]]]
    assert Run.from_json(run.to_json(), graph).resume().status == "stopped"  # the item limit saved with the run still applies inside


# -- waiting inside, saved, and read back in another process ---------------------------------------

_CHILD = """
import importlib.util, json, sys
from conductor import Run
spec = importlib.util.spec_from_file_location("subgraph_nodes", {module!r})
nodes = importlib.util.module_from_spec(spec)
sys.modules["subgraph_nodes"] = nodes
spec.loader.exec_module(nodes)
graph = nodes.inside_a_loop("asking-inner")
run = Run.from_json(open({path!r}).read(), graph)
ids = [w.id for w in run.waiting]
half = run.answer(ids[0], "first").resume()
open({path!r}, "w").write(half.to_json())
print(json.dumps({{"status": half.status, "ids": ids, "left": [w.id for w in half.waiting], "calls": dict(nodes.CALLS)}}))
"""


def test_a_waiting_subgraph_inside_a_loop_is_half_answered_in_another_process_and_finished(tmp_path: Path):
    graph = inside_a_loop("asking-inner")
    run = graph.run(text="ab cd")
    assert run.status == "waiting"
    ids = [w.id for w in run.waiting]
    assert ids[0].startswith("e@0/q~") and ids[1].startswith("e@1/q~")
    assert [w.loc for w in run.waiting] == [("e", 0, "q"), ("e", 1, "q")]
    assert run.results["e"] == {"result": [None, None]}  # the subgraph node's outputs are None until every inner call has an outcome
    path = tmp_path / "run.json"
    path.write_text(run.to_json())
    CALLS.clear()
    child = subprocess.run([sys.executable, "-c", _CHILD.format(module=__file__, path=str(path))], capture_output=True, text=True, timeout=120, check=False)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == {"status": "waiting", "ids": ids, "left": ids[1:], "calls": {}}  # the other process calls nothing again
    done = Run.from_json(path.read_text(), graph).answer(ids[1], "second").resume()
    assert done.status == "complete" and done.results["e"]["result"] == ["first", "second"]
    assert CALLS == {}
    assert Run.from_json(done.to_json(), graph).outputs == done.outputs  # the subgraph node has no outcome of its own: its outputs are read from the inner outcomes
    assert sorted(json.loads(done.to_json())["outcomes"]) == ["e/q", "e/up", "w"]  # an inner node is saved as any node is, under its node path


def test_a_wrong_answer_inside_a_subgraph_is_refused_by_its_node_path():
    graph = inside_a_loop("asking-inner")
    run = graph.run(text="ab cd")
    first, second = (w.id for w in run.waiting)
    with pytest.raises(InvalidCallError) as refused:
        run.answer({"e@5/q": "x", "e@1/q~00000000": "y", "e@0/up": "z", "e@x/q": "w"})
    assert [(p.type, p.loc) for p in refused.value.errors()] == [
        ("not_waiting", ("e", 5, "q")), ("answer_stale", ("e", 1, "q")), ("not_waiting", ("e", 0, "up")), ("not_waiting", ("e",)),
    ]
    assert refused.value.errors()[1].ctx["current"] == second
    with pytest.raises(InvalidCallError) as twice:
        run.answer({first: "a", "e@0/q": "b"})
    assert [(p.type, p.loc) for p in twice.value.errors()] == [("invalid_answer", ("e", 0, "q"))]
    assert run.answer("e@1/q", "by place").resume().results["e"]["result"] == [None, "by place"]


# -- policy: an inner node is called, retried, timed out and cancelled as in one graph -------------


def test_an_inner_nodes_concurrency_limit_counts_its_calls_across_every_item_of_the_loop():
    run = inside_a_loop("fetch-inner").run(text="a b c d e f")
    assert run.status == "complete" and run.results["e"]["result"] == ["A?", "B?", "C?", "D?", "E?", "F?"]
    assert CALLS["fetch"] == 6 and AT_ONCE["peak"] == 2  # Policy(concurrency=2): two at once in the whole run, not two per item
    AT_ONCE.update(now=0, peak=0)
    assert in_one_graph("fetch-inner").run(text="a b c d e f").status == "complete" and AT_ONCE["peak"] == 2


def test_a_cancel_inside_a_subgraph_keeps_the_calls_on_threads_and_resume_repeats_none():
    graph = inside_a_loop("fetch-inner")
    events = stream(graph.with_inputs(text="a b c d"), cancel_when=lambda e: isinstance(e, NodeStart) and e.loc == ("e", "f"))
    cancelled = events[-1].run
    assert cancelled.status == "cancelled" and cancelled.state.cause.type == "run_cancelled"
    assert CALLS["fetch"] in (1, 2)  # the calls that had a thread when the cancel was seen; those waiting for a thread were dropped
    assert cancelled.results["e"] == {"result": [None, None, None, None]}
    resumed = Run.from_json(cancelled.to_json(), graph).resume()
    assert resumed.status == "complete" and resumed.results["e"]["result"] == ["A?", "B?", "C?", "D?"]
    assert CALLS["fetch"] == 4  # the calls with an outcome were not made again


def test_the_run_timeout_stops_the_calls_inside_a_subgraph():
    graph = inside_a_loop("fetch-inner")
    run = graph.run(text="a b c d", timeout=0.05)
    assert run.status == "timed_out" and run.state.cause.type == "run_timed_out"
    assert CALLS["fetch"] == 2
    assert Run.from_json(run.to_json(), graph).resume().status == "complete" and CALLS["fetch"] == 4


def test_an_inner_node_is_retried_by_its_own_policy():
    events = stream(inside_a_loop("flaky-inner").with_inputs(text="a b"))
    assert events[-1].run.status == "complete" and events[-1].run.results["e"]["result"] == ["a~", "b~"]
    assert CALLS["flaky"] == 4
    retries = sorted((e.loc, e.attempt, e.cause.loc) for e in events if isinstance(e, NodeRetry))
    assert retries == [(("e", 0, "f"), 1, ("e", 0, "f")), (("e", 1, "f"), 1, ("e", 1, "f"))]


def test_an_inner_nodes_attempt_timeout_fails_the_run_at_its_node_path_and_keeps_the_other_items_outcomes():
    graph = inside_a_loop("stuck-inner")
    run = graph.run(text="ok slow")
    assert run.status == "failed" and run.state.cause.type == "timeout" and run.state.cause.loc == ("e", 1, "s")
    assert run.results["e"]["result"] == ["OK", None]
    CALLS.clear()
    assert Run.from_json(run.to_json(), graph).resume().status == "failed" and CALLS == {"stuck": 1}  # only the call that failed


def test_a_failure_inside_a_subgraph_keeps_the_other_inner_outcomes_for_resume():
    graph = inside_a_loop("boom-inner")
    events = stream(graph.with_inputs(text="ok bad"))
    run = events[-1].run
    assert run.status == "failed" and run.state.cause.loc == ("e", 1, "b") and run.state.cause.type == "failed"
    assert [(e.loc, e.cause.loc) for e in events if isinstance(e, NodeFailed)] == [(("e", "b"), ("e", 1, "b"))]  # the event names the node, and its cause the call
    again = Run.from_json(run.to_json(), graph).resume()
    assert again.status == "failed" and CALLS["upper"] == 2 and CALLS["boom"] == 3  # only the failed call is made again


def test_a_checkpoint_taken_mid_run_resumes_with_no_inner_call_repeated():
    graph = inside_a_loop("fetch-inner")
    events = stream(graph.with_inputs(text="a b c d"), checkpoint_every=0.01)
    assert events[-1].run.status == "complete" and CALLS["fetch"] == 4
    mid = next(e.run for e in events if isinstance(e, RunCheckpoint) and 0 < _outcome_count(e.run, "e", "f") < 4)
    assert mid.status == "running"
    with_outcome = _outcome_count(mid, "e", "f")
    resumed = Run.from_json(mid.to_json(), graph).resume()  # as a process that died during the run would
    assert resumed.status == "complete" and resumed.results["e"]["result"] == ["A?", "B?", "C?", "D?"]
    assert CALLS["fetch"] == 4 + (4 - with_outcome)  # only the calls the checkpoint has no outcome for


def _outcome_count(run, subgraph_node: str, node_id: str) -> int:
    """How many calls of the inner node ``node_id`` of ``subgraph_node`` have an outcome in ``run``."""
    return len(run._outcomes[f"{subgraph_node}/{node_id}"])


# -- one address: a node path and item indexes -----------------------------------------------------


def test_an_inner_node_sends_its_events_once_by_its_node_path_as_in_one_graph():
    events = stream(inside_a_loop("asking-inner").with_inputs(text="ab cd"))
    starts = [e.loc for e in events if isinstance(e, NodeStart)]
    assert starts == [("w",), ("e",), ("e", "up"), ("e", "q")]  # once per node, however many items: the subgraph node first
    assert [(e.loc, e.outcomes) for e in events if isinstance(e, NodeComplete)] == [(("w",), 1), (("e", "up"), 2)]
    flat_events = stream(in_one_graph("asking-inner").with_inputs(text="ab cd"))
    assert [e.loc for e in flat_events if isinstance(e, NodeStart)] == [("outer",), ("up",), ("q",)]
    waiting = [e.waiting for e in events if isinstance(e, NodeWaiting)]
    assert [w.id for w in waiting] == [w.id for w in events[-1].run.waiting] and waiting[1].loc == ("e", 1, "q")
    assert isinstance(events[-1], RunEnd) and events[-1].status == "waiting"
    answered = events[-1].run.answer({w.id: "yes" for w in waiting})
    later = stream(answered)
    assert not [e for e in later if isinstance(e, NodeStart)]  # every node started in the first resume()
    assert [(e.loc, e.outcomes) for e in later if isinstance(e, NodeComplete)] == [(("e", "q"), 2), (("e",), 2)]


def test_a_subgraph_inside_a_subgraph_is_addressed_by_its_node_path():
    graph = inside_a_loop("wrapping")
    assert graph.outputs == ["result"]
    run = graph.run(text="ab cd")
    ids = [w.id for w in run.waiting]
    assert ids[0].startswith("e@0/in/q~") and ids[1].startswith("e@1/in/q~") and run.waiting[1].loc == ("e", 1, "in", "q")
    with pytest.raises(InvalidCallError) as refused:
        run.answer("e@1/in/q~00000000", "x")
    assert refused.value.errors()[0].loc == ("e", 1, "in", "q")
    done = Run.from_json(run.answer({ids[1]: "two"}).resume().to_json(), graph).answer(ids[0], "one").resume()
    assert done.status == "complete" and done.outputs == {"result": ["one", "two"]} and CALLS["upper"] == 2


# -- reuse: again reuses inside a subgraph what it reuses in one graph -----------------------------


def test_again_calls_inside_a_subgraph_only_what_it_calls_in_one_graph_after_a_change_or_an_insert():
    for graph, at in ((inside_a_loop("shout-inner"), ("e", "result")), (in_one_graph("shout-inner"), ("ex", "result"))):
        run = graph.run(text="ab cd ef")
        CALLS.clear()
        assert run.again(text="ab xx ef").results[at[0]][at[1]] == ["AB!", "XX!", "EF!"]
        assert CALLS == {"words": 1, "upper": 1, "exclaim": 1}
        CALLS.clear()
        assert run.again(text="zz ab cd ef").results[at[0]][at[1]] == ["ZZ!", "AB!", "CD!", "EF!"]
        assert CALLS == {"words": 1, "upper": 1, "exclaim": 1}
        CALLS.clear()
        assert Run.from_json(run.again(text="ab").to_json(), graph).again(text="ab cd").status == "complete"
        assert CALLS == {"words": 2, "upper": 1, "exclaim": 1}  # outcomes of the previous run that were not reused are saved, and reused later


def test_an_inner_node_whose_policy_forbids_reuse_is_called_again_by_again():
    run = inside_a_loop("stamped-inner").run(text="ab cd")
    CALLS.clear()
    assert run.again().results["e"]["result"] == ["AB", "CD"]
    assert CALLS == {"stamp": 2}  # the subgraph node is never reused whole: stamp is called again, upper is reused


def test_two_documents_with_the_same_line_each_keep_their_own_answer_inside_a_subgraph():
    graph = inside_a_loop("lines-inner", "sentences")  # a question per word of each document
    run = graph.run(text="a b;a c")
    run = run.answer({w.id: f"doc{w.loc[1]}:{w.inputs['text']}" for w in run.waiting}).resume()
    assert run.status == "complete" and run.results["e"]["result"] == [["doc0:a", "doc0:b"], ["doc1:a", "doc1:c"]]
    changed = run.again(text="a b d;a c")
    assert [w.loc[:2] for w in changed.waiting] == [("e", 0)] * 3  # document 0 changed, so its lines are asked again, as in one graph
    assert changed.answer({w.id: "new" for w in changed.waiting}).resume().results["e"]["result"] == [["new", "new", "new"], ["doc1:a", "doc1:c"]]
    inserted = run.again(text="x;a b;a c")
    assert [(w.loc, w.inputs["text"]) for w in inserted.waiting] == [(("e", 0, "q", 0), "x")]
    assert inserted.answer(inserted.waiting[0], "x").resume().results["e"]["result"] == [["x"], ["doc0:a", "doc0:b"], ["doc1:a", "doc1:c"]]
    swapped = run.again(text="a c;a b")
    assert swapped.status == "complete" and swapped.results["e"]["result"] == [["doc1:a", "doc1:c"], ["doc0:a", "doc0:b"]]
    flat_run = in_one_graph("lines-inner", "sentences").run(text="a b;a c")
    flat_run = flat_run.answer({w.id: f"doc{w.loc[1]}:{w.inputs['text']}" for w in flat_run.waiting}).resume()
    assert flat_run.again(text="a c;a b").results["q"]["result"] == [["doc1:a", "doc1:c"], ["doc0:a", "doc0:b"]]  # the same, in one graph


def test_identical_items_reuse_their_answers_in_item_order_whichever_thread_finishes_first():
    graph = inside_a_loop("staggered-inner")
    run = graph.run(text="ab ab ab")
    run = run.answer({w.id: f"answer {w.loc[1]}" for w in run.waiting}).resume()
    assert run.results["e"]["result"] == ["answer 0", "answer 1", "answer 2"]
    CALLS.clear()
    again = Run.from_json(run.to_json(), graph).again()  # stagger is called again, and its first call returns last
    assert again.status == "complete" and CALLS["stagger"] == 3 and CALLS["ask"] == 0
    assert again.results["e"]["result"] == ["answer 0", "answer 1", "answer 2"]
    CALLS.clear()
    longer = run.again(text="zz ab ab ab")
    assert [w.loc for w in longer.waiting] == [("e", 0, "q")]
    assert longer.results["e"]["result"] == [None, "answer 0", "answer 1", "answer 2"]


def test_reuse_reaches_a_subgraph_inside_a_subgraph_and_forget_forgets_every_inner_outcome():
    graph = inside_a_loop("wrapping")
    run = graph.run(text="ab cd")
    done = run.answer({w.id: w.inputs["text"].lower() for w in run.waiting}).resume()
    assert done.outputs == {"result": ["ab", "cd"]}
    CALLS.clear()
    longer = done.again(text="ab cd ef")
    assert longer.status == "waiting" and CALLS == {"words": 1, "upper": 1, "ask": 1}
    assert [w.id.partition("~")[0] for w in longer.waiting] == ["e@2/in/q"]
    CALLS.clear()
    forgotten = done.forget("e").resume()
    assert forgotten.status == "waiting" and CALLS == {"upper": 2, "ask": 2}


def test_a_previous_run_of_another_graph_is_reused_inside_a_subgraph_whose_nodes_are_unchanged():
    run = inside_a_loop("shout-inner").run(text="ab cd")
    edited = compile_graph({"w": "words", "e": "shout-inner", "n": "count"}, [("w.result", "e.text"), ("e.result", "n.items")])
    CALLS.clear()
    assert edited.with_inputs(text="ab cd", previous=run).resume().results["n"]["result"] == 2
    assert CALLS == {}


@pytest.mark.parametrize(
    ("nodes", "edges", "series"),
    [
        ({"w": "words", "t": "letters", "h": "height"}, [("w.result", "t.text"), ("t.result", "h.table")], "w.result"),
        ({"e": "letters-per-word-inner", "h": "height"}, [("e.result", "h.table")], "e.w.result"),
        ({"s": "sheets", "h": "height"}, [("s.result", "h.table")], "s.result"),
    ],
    ids=["in one graph", "returned inside a subgraph", "returned as a Series of tables"],
)
def test_tables_returned_per_item_reach_a_table_input_one_per_item_wherever_they_are_returned(nodes: dict[str, str], edges: list[tuple[str, str]], series: str):
    graph = compile_graph(nodes, edges)
    assert not graph.problems, graph.problems
    run = graph.run(text="ab c")
    assert (graph.runs_on["h"].mode, graph.runs_on["h"].series, run.status, run.results["h"]["result"]) == ("per_item", series, "complete", [2, 1])


# -- a subgraph node reads the same on a run and on that run saved and read back -------------------


def what_a_host_reads(run) -> tuple[Any, ...]:
    """Everything a host reads of ``run``."""
    waiting = [waiting_call.model_dump() for waiting_call in run.waiting]
    not_called = [call.model_dump() for call in run.not_called]
    return run.status, run.results, waiting, not_called, run.outputs if run.status == "complete" else None


def reads_the_same_saved_and_read_back(run) -> bool:
    assert what_a_host_reads(run) == what_a_host_reads(Run.from_json(run.to_json(), run.graph))
    return True


@pytest.mark.parametrize("inner", ["asking-inner", "lines-inner", "wrapping", "staggered-inner"])
@pytest.mark.parametrize("per_item", [False, True], ids=["once", "per item"])
def test_a_subgraph_reads_the_same_saved_and_read_back_whenever_it_waits_and_after_each_answer(inner: str, per_item: bool):
    graph = inside_a_loop(inner, "sentences") if per_item else compile_graph({"e": inner})
    run = graph.run(text="ab cd;ef")
    assert run.status == "waiting" and reads_the_same_saved_and_read_back(run)
    while run.status == "waiting":
        first, *rest = run.waiting
        one = run.answer(first, "yes")  # one answer, not resumed: what it completes reads the same saved and read back
        assert reads_the_same_saved_and_read_back(one) and what_a_host_reads(run) == what_a_host_reads(Run.from_json(run.to_json(), run.graph))  # and the run it was given to is unchanged
        answered = one.answer({waiting_call.id: "yes" for waiting_call in rest})
        assert reads_the_same_saved_and_read_back(answered)
        run = answered.resume()
        assert reads_the_same_saved_and_read_back(run)
    assert run.status == "complete"


def test_an_answer_inside_a_subgraph_shows_in_its_results_before_resume_as_in_one_graph():
    in_subgraph, one_graph = compile_graph({"e": "asking-inner"}).run(text="a"), compile_graph({"up": "upper", "q": "ask"}, [("up.result", "q.text")]).run(text="a")
    assert in_subgraph.answer(in_subgraph.waiting[0], "yes").results == {"e": {"result": "yes"}}
    assert one_graph.answer(one_graph.waiting[0], "yes").results == {"up": {"result": "A"}, "q": {"result": "yes"}}
    assert in_subgraph.results == {"e": {"result": None}}  # the run the answer was given to has none


def test_a_run_stopped_at_the_item_limit_keeps_the_calls_on_threads_inside_a_subgraph_in_its_results():
    graph = compile_graph({"e": "fetch-words-inner", "many": "words", "each": "upper"}, [("many.result", "each.text")], inputs={"few": "e.text", "many": "many.text"}, outputs={"e": "e.result", "each": "each.result"})
    stopped = graph.run(few="a b", many="x y z", max_items=2)
    assert stopped.status == "stopped" and stopped.state.cause.loc == ("many",)
    assert stopped.results["e"] == {"result": ["a?", "b?"]} and reads_the_same_saved_and_read_back(stopped)


def test_a_subgraph_failed_at_one_item_reads_the_same_saved_and_read_back():
    failed = inside_a_loop("boom-inner").run(text="ab bad cd")
    assert failed.status == "failed" and failed.results["e"] == {"result": ["AB", None, None]} and reads_the_same_saved_and_read_back(failed)


def test_an_inner_node_not_called_for_none_reads_the_same_saved_and_read_back():
    run = inside_a_loop("maybe-inner").run(text="ab none cd")
    assert run.status == "complete" and [call.loc for call in run.not_called] == [("e", 1, "up")] and reads_the_same_saved_and_read_back(run)


def test_every_checkpoint_inside_a_subgraph_reads_the_same_saved_and_read_back_whichever_calls_returned():
    events = stream(inside_a_loop("upper-fetch-inner").with_inputs(text="a b c d"), checkpoint_every=0.01)
    checkpoints = [event.run for event in events if isinstance(event, RunCheckpoint)]
    assert checkpoints and all(reads_the_same_saved_and_read_back(checkpoint) for checkpoint in checkpoints)


@pytest.mark.parametrize("inner", ["two-asks-inner", "three-asks-inner"])
@pytest.mark.parametrize("per_item", [False, True], ids=["once", "per item"])
def test_answers_given_at_once_to_subgraphs_side_by_side_inside_another_complete_the_outer_subgraph(inner: str, per_item: bool):
    nodes = {"w": "words", "e": inner} if per_item else {"e": inner}
    graph = compile_graph(nodes, [("w.result", f"e.{name}") for name in compile_graph({"e": inner}).inputs] if per_item else [])
    assert not graph.problems, graph.problems
    run = graph.run(dict.fromkeys(graph.inputs, "ab cd"))
    answered = run.answer({waiting_call.id: "yes" for waiting_call in run.waiting})
    assert reads_the_same_saved_and_read_back(answered)
    assert all(value == (["yes", "yes"] if per_item else "yes") for value in answered.results["e"].values())
    some = run.answer({waiting_call.id: "yes" for waiting_call in run.waiting[1:]})  # all but the first: no subgraph around it completes
    assert reads_the_same_saved_and_read_back(some) and reads_the_same_saved_and_read_back(some.answer(run.waiting[0], "yes"))


def test_three_hundred_answers_given_at_once_inside_one_subgraph_complete_it():
    run = compile_graph({"e": "lines-inner"}).run(text=" ".join(f"w{i}" for i in range(300)))
    answered = run.answer({waiting_call.id: "yes" for waiting_call in run.waiting})
    assert answered.results["e"] == {"result": ["yes"] * 300} and reads_the_same_saved_and_read_back(answered)


@pytest.mark.parametrize(
    ("nodes", "edges"),
    [
        ({"w": "words", "r": "record", "h": "height"}, [("w.result", "r.text"), ("r.result", "h.table")]),
        ({"e": "record-per-word-inner", "h": "height"}, [("e.result", "h.table")]),
    ],
    ids=["in one graph", "returned inside a subgraph"],
)
def test_values_of_any_type_returned_per_item_are_collected_as_records_by_a_table_input_wherever_returned(nodes: dict[str, str], edges: list[tuple[str, str]]):
    graph = compile_graph(nodes, edges)
    assert not graph.problems, graph.problems
    run = graph.run(text="ab c d")
    assert (graph.runs_on["h"].mode, run.status, run.results["h"]["result"]) == ("once", "complete", 3)


def _said_of(events, node_id: str) -> list[tuple[str, tuple]]:
    return [(event.type, event.loc) for event in events if getattr(event, "loc", ())[:1] == (node_id,)]


def test_a_subgraph_with_no_output_sends_the_same_events_on_a_fresh_run_and_on_one_read_back_before_any_call():
    graph = compile_graph({"e": "quiet-inner"})
    said = [("node_start", ("e",)), ("node_start", ("e", "n")), ("node_complete", ("e", "n")), ("node_complete", ("e",))]
    assert _said_of(stream(graph.with_inputs(text="x")), "e") == said
    assert _said_of(stream(Run.from_json(graph.with_inputs(text="x").to_json(), graph)), "e") == said


def test_a_subgraph_with_no_output_completes_when_its_inner_nodes_do_and_not_before():
    graph = compile_graph({"q": "ask", "e": "quiet-inner"}, [("q.result", "e.text")])
    CALLS.clear()
    waiting = graph.run(text="x")
    first = stream(graph.with_inputs(text="x"))
    assert (waiting.status, _said_of(first, "e"), CALLS["note"]) == ("waiting", [], 0)
    for run in (waiting, Run.from_json(waiting.to_json(), graph)):
        said = _said_of(stream(run.answer(run.waiting[0].id, "y")), "e")
        assert said == [("node_start", ("e",)), ("node_start", ("e", "n")), ("node_complete", ("e", "n")), ("node_complete", ("e",))]


def test_a_subgraph_of_no_nodes_runs_to_empty_results_as_a_graph_of_no_nodes_does():
    registry = NodeRegistry()
    registry.add(upper)
    registry.add(subgraph("nothing", title="Nothing", description="The nothing node.")(GraphRecord.model_validate({"nodes": {}})))
    alone = Graph.compile(GraphRecord.model_validate({"nodes": {}}), registry).run()
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "e": {"node": "nothing"}}}), registry)
    run = graph.run(text="x")
    assert graph.problems == [] and (alone.status, alone.results) == ("complete", {})
    assert (run.status, run.results) == ("complete", {"u": {"result": "X"}, "e": {}})
    assert Run.from_json(run.to_json(), graph).results == run.results
