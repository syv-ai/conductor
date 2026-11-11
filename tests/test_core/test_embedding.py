"""An embedded graph is one node: its inputs and outputs are the inner graph's, and its problems the inner graph's
by path."""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    FromRun,
    NodeError,
    NodeRegistry,
    Param,
    Policy,
    Series,
    Table,
    node,
)

CALLS: collections.Counter[str] = collections.Counter()
AT_ONCE = {"now": 0, "peak": 0}
_LOCK = threading.Lock()
_FAILED_ONCE: set[str] = set()


@pytest.fixture(autouse=True)
def _fresh():
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


@_node("ask")
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


class Session:
    def __init__(self, name: str) -> None:
        self.name = name


@_node("signed")
def signed(text: str, session: FromRun[Session]) -> str:
    return f"{text} by {session.name}"


def _columns(names: str) -> dict[str, Any]:
    return {name: str for name in names.split(",")}


@_node("columns", compute_outputs=_columns)
def columns(text: str, names: str = "a") -> dict[str, Any]:
    return dict.fromkeys(names.split(","), text)


@_node("table")
def table(text: str) -> Table:
    raise NotImplementedError


def chain(*nodes: tuple[str, str]) -> dict[str, Any]:
    """Nodes in a line, each reading the ``result`` of the one before into its ``text``."""
    return {"nodes": [{"id": i, "node": n} for i, n in nodes], "edges": [[f"{a[0]}.result", f"{b[0]}.text"] for a, b in zip(nodes, nodes[1:], strict=False)]}


#: Each embedded graph, by the id of the node that embeds it.
INNER = {
    "shout-inner": chain(("up", "upper"), ("ex", "exclaim")),
    "greet-inner": chain(("g", "greet")),
    "asking-inner": chain(("up", "upper"), ("q", "ask")),
    "fetch-inner": chain(("f", "fetch"), ("up", "upper")),
    "flaky-inner": chain(("f", "flaky")),
    "stuck-inner": chain(("s", "stuck"), ("up", "upper")),
    "boom-inner": chain(("up", "upper"), ("b", "boom")),
    "maybe-inner": chain(("m", "maybe"), ("up", "upper")),
    "lines-inner": chain(("w", "words"), ("q", "ask")),  # a question per word of what it is given
    "wrapping": chain(("in", "asking-inner")),  # an embedded graph inside an embedded graph
    "deep-inner": chain(("w", "words"), ("c", "chars"), ("up", "upper")),  # its output sits two maps deep
    "series-per-item-inner": chain(("w", "words"), ("c", "chars")),  # a Series made per word
    "table-per-item-inner": chain(("w", "words"), ("t", "table")),
    "columns-inner": chain(("c", "columns")),  # its first node has an input a hook reads
    "broken-inner": chain(("up", "upper"), ("x", "no-such-node")),
    "signed-inner": chain(("s", "signed")),
    "self-a": chain(("x", "self-b")),
    "self-b": chain(("y", "self-a")),
    "deciding-inner": {
        "nodes": [{"id": "d", "node": "decide"}, {"id": "s", "node": "exclaim"}, {"id": "l", "node": "upper"}],
        "edges": [["d.short", "s.text"], ["d.long", "l.text"]],
    },
}

REGISTRY = NodeRegistry()
for _definition in (words, sentences, chars, upper, exclaim, greet, count, ask, fetch, flaky, stuck, boom, maybe, decide, signed, columns, table):
    REGISTRY.add(_definition)
for _id, _record in INNER.items():
    REGISTRY.add(_node(_id)(_record))


def load(nodes: dict[str, str | dict[str, Any]], edges: list[tuple[str, str]] = ()):  # type: ignore[assignment]
    """A graph of ``nodes`` (id to node id, or to the rest of its record) and ``edges``."""
    records = [{"id": i, **({"node": n} if isinstance(n, str) else n)} for i, n in nodes.items()]
    return REGISTRY.load({"nodes": records, "edges": [list(edge) for edge in edges]})


def mapped(inner: str, splitter: str = "words"):
    """``splitter`` into the embedded graph ``inner``, so the embedded graph runs once per item."""
    graph = load({"w": splitter, "e": inner}, [("w.result", "e.text")])
    assert not graph.problems, graph.problems
    return graph


def flat(inner: str, splitter: str = "words"):
    """The nodes of the embedded graph ``inner`` drawn in the graph itself, after ``splitter``: what ``mapped``
    must behave as."""
    record = INNER[inner]
    first = record["nodes"][0]["id"]
    graph = REGISTRY.load({"nodes": [{"id": "w", "node": splitter}, *record["nodes"]], "edges": [["w.result", f"{first}.text"], *record["edges"]]})
    assert not graph.problems, graph.problems
    return graph


def problems(graph) -> list[tuple[str, tuple]]:
    return [(p.type, p.loc) for p in graph.problems]


# -- loading: an embedded graph is one node with the inner graph's inputs and outputs ----------------------


def test_an_embedded_graph_offers_the_inner_graphs_inputs_by_name_and_outputs_by_address():
    graph = load({"e": "shout-inner"})
    assert list(graph.inputs) == ["text"] and graph.inputs["text"].address == "e.text"
    assert graph.outputs == ["e.ex.result"]
    described = graph.describe().nodes["e"]
    assert [i.name for i in described.inputs] == ["text"] and [o.name for o in described.outputs] == ["ex.result"]
    two = load({"a": "shout-inner", "b": "shout-inner"}, [("a.ex.result", "b.text")])
    assert two.is_runnable and two._nodes["a"].inner is two._nodes["b"].inner  # loaded once, alone


def test_an_edge_reads_one_field_of_an_embedded_graphs_output_after_its_address():
    graph = load({"e": "columns-inner", "u": "upper"}, [("e.c.a", "u.text")])
    assert graph.is_runnable and graph.outputs == ["u.result"]
    assert problems(load({"e": "shout-inner", "u": "upper"}, [("e.ex", "u.text")])) == [("unknown_output", ("u", "text"))]


def test_an_unfed_required_input_of_a_later_embedded_graph_is_unbound_like_any_nodes():
    graph = load({"w": "words", "e": "greet-inner", "x": "shout-inner"}, [("w.result", "e.greeting")])
    assert problems(graph) == [("unbound_on_later_node", ("e", "text"))]


def test_an_inner_input_a_hook_reads_is_set_inside_and_not_offered():
    assert list(load({"e": "columns-inner"}).inputs) == ["text"]
    typed_in = load({"e": {"node": "columns-inner", "values": {"names": "a,b"}}})
    assert problems(typed_in) == [("unknown_input", ("e", "names"))]


def test_an_inner_graphs_problems_are_its_own_with_the_embedding_in_front():
    assert problems(mapped_with_problems("broken-inner")) == [("unknown_node", ("e", "x"))]
    described = mapped_with_problems("broken-inner").describe().nodes["e"]
    assert not described.loaded and [p.loc for p in described.problems] == [("e", "x")]


def mapped_with_problems(inner: str):
    return load({"w": "words", "e": inner}, [("w.result", "e.text")])


def test_a_graph_that_embeds_itself_is_refused_at_the_node_that_closes_the_circle():
    assert problems(load({"e": "self-a"})) == [("embeds_itself", ("e", "x", "y"))]


def test_an_output_made_per_item_inside_is_a_series_and_a_series_made_per_item_a_series_of_lists():
    deep = load({"e": "deep-inner", "n": "count", "x": "exclaim"}, [("e.up.result", "n.items"), ("e.up.result", "x.text")])
    assert deep.is_runnable and deep.plan["x"].series == "e.up.result"
    assert deep.describe().nodes["e"].outputs[0].shape == "many"
    lists = load({"e": "series-per-item-inner"})._nodes["e"].outputs["c.result"]
    assert (lists.shape, lists.type, lists.type_id) == ("many", list[str], "list")


def test_a_table_made_per_item_inside_is_refused_on_the_embedding():
    refused = load({"e": "table-per-item-inner"}).problems
    assert [(p.type, p.loc, p.ctx) for p in refused] == [("table_per_item_not_offered", ("e",), {"node": "table-per-item-inner", "output": "t.result", "series": "w.result"})]


def test_a_decision_inside_is_the_embeddings_by_where_it_is_made():
    graph = load({"e": "deciding-inner", "j": "exclaim"}, [("e.s.result", "j.text"), ("e.l.result", "j.text")])
    assert problems(graph) == [("several_edges_into_one_input", ("j", "text"))]  # s and l both rest on d, which the embedding hides
    last = load({"e": chain_node("decide")})
    assert last.exclusive_groups == {"e": {"d.way": ["d.short", "d.long"]}}
    joined = load({"e": chain_node("decide"), "j": "exclaim"}, [("e.d.short", "j.text"), ("e.d.long", "j.text")])
    assert joined.is_runnable


def chain_node(node_id: str) -> str:
    """The id of an embedded graph of the one node ``node_id``, registered when first asked for."""
    embedded = f"only-{node_id}"
    if embedded not in REGISTRY:
        REGISTRY.add(_node(embedded)(chain(("d", node_id))))
    return embedded
