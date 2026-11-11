"""An embedded graph is one node outside, and embedding a graph changes nothing about how its nodes behave: every
test here holds a graph embedded against what the same nodes do drawn in the graph itself."""

from __future__ import annotations

import asyncio
import collections
import json
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    FromRun,
    GraphRecord,
    InvalidCallError,
    NodeError,
    NodeRegistry,
    Param,
    Policy,
    Series,
    Table,
    node,
    stored_graph,
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


@_node("stamp", policy=Policy(reuse_when_inputs_match=False))
def stamp(text: str) -> str:
    CALLS["stamp"] += 1
    return text


@_node("stagger", policy=Policy(concurrency=4, reuse_when_inputs_match=False))
def stagger(text: str) -> str:
    """Comes back later the earlier it was called, so the calls of one leg finish in reverse."""
    with _LOCK:
        CALLS["stagger"] += 1
        place = CALLS["stagger"]
    time.sleep(max(0.0, 0.2 - 0.05 * place))
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
    "fetch-words-inner": chain(("w", "words"), ("f", "fetch")),  # a call on a thread per word
    "upper-fetch-inner": chain(("up", "upper"), ("f", "fetch")),  # its last node is the one on threads
    "two-asks-inner": {"nodes": [{"id": "f", "node": "asking-inner"}, {"id": "g", "node": "asking-inner"}], "edges": []},
    "three-asks-inner": {"nodes": [{"id": n, "node": "asking-inner"} for n in ("f", "g", "h")], "edges": []},
    "wrapping": chain(("in", "asking-inner")),  # an embedded graph inside an embedded graph
    "deep-inner": chain(("w", "words"), ("c", "chars"), ("up", "upper")),  # its output sits two maps deep
    "series-per-item-inner": chain(("w", "words"), ("c", "chars")),  # a Series made per word
    "table-per-item-inner": chain(("w", "words"), ("t", "table")),
    "letters-per-word-inner": chain(("w", "words"), ("t", "letters")),
    "record-per-word-inner": chain(("w", "words"), ("r", "record")),  # each word's value is declared Any
    "columns-inner": chain(("c", "columns")),  # its first node has an input a hook reads
    "broken-inner": chain(("up", "upper"), ("x", "no-such-node")),
    "signed-inner": chain(("s", "signed")),
    "stamped-inner": chain(("s", "stamp"), ("up", "upper")),
    "staggered-inner": chain(("s", "stagger"), ("q", "ask")),
    "count-inner": chain(("n", "count")),  # its one node gathers
    "quiet-inner": chain(("n", "note")),  # it has no output
    "self-a": chain(("x", "self-b")),
    "self-b": chain(("y", "self-a")),
    "deciding-inner": {
        "nodes": [{"id": "d", "node": "decide"}, {"id": "s", "node": "exclaim"}, {"id": "l", "node": "upper"}],
        "edges": [["d.short", "s.text"], ["d.long", "l.text"]],
    },
}

REGISTRY = NodeRegistry()
for _definition in (words, sentences, chars, upper, exclaim, greet, count, ask, fetch, flaky, stuck, boom, maybe, decide, signed, columns, table, stamp, stagger, letters, height, sheets, record, note):
    REGISTRY.add(_definition)
for _id, _record in INNER.items():
    REGISTRY.add(stored_graph(_id, title=_id.capitalize(), description=f"The {_id} node.")(GraphRecord.model_validate(_record)))


def load(nodes: dict[str, str | dict[str, Any]], edges: list[tuple[str, str]] = ()):  # type: ignore[assignment]
    """A graph of ``nodes`` (id to node id, or to the rest of its record) and ``edges``."""
    records = [{"id": i, **({"node": n} if isinstance(n, str) else n)} for i, n in nodes.items()]
    return REGISTRY.load(GraphRecord.model_validate({"nodes": records, "edges": [list(edge) for edge in edges]}))


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
    graph = REGISTRY.load(GraphRecord.model_validate({"nodes": [{"id": "outer", "node": splitter}, *record["nodes"]], "edges": [["outer.result", f"{first}.text"], *record["edges"]]}))
    assert not graph.problems, graph.problems
    return graph


def stream(run, cancel_when=None, **options):
    async def go():
        cancel, events = asyncio.Event(), []
        async for event in run.stream(cancel=cancel, **options):
            events.append(event)
            if cancel_when is not None and cancel_when(event):
                cancel.set()
        return events

    return asyncio.run(go())


def problems(graph) -> list[tuple[str, tuple]]:
    return [(p.type, p.loc) for p in graph.problems]


# -- loading: an embedded graph is one node with the inner graph's inputs and outputs ----------------------


def test_an_embedded_graph_offers_the_inner_graphs_inputs_by_name_and_outputs_by_address():
    graph = load({"e": "shout-inner"})
    assert list(graph.inputs) == ["text"] and graph.inputs["text"].address == "e.text"
    assert graph.outputs == ["e.ex.result"]
    described = graph.describe().nodes["e"]
    assert [i.field.name for i in described.inputs] == ["text"] and [o.name for o in described.outputs] == ["ex.result"]
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


def test_an_output_returned_inside_loops_is_offered_as_one_series_per_loop():
    deep = load({"e": "deep-inner", "n": "count", "x": "exclaim"}, [("e.up.result", "n.items"), ("e.up.result", "x.text")])
    assert deep.is_runnable and (deep.runs_on["n"].series, deep.runs_on["x"].series) == ("e.w.result", "e.up.result")
    described = {inner: load({"e": inner}).describe().nodes["e"].outputs[0] for inner in ("lines-inner", "deep-inner", "series-per-item-inner")}
    assert [(output.many, output.type) for output in described.values()] == [(True, "str"), (True, "Series[str]"), (True, "Series[str]")]


def test_a_table_made_per_item_inside_is_a_series_of_tables():
    graph = load({"e": "table-per-item-inner"})
    tables = graph._nodes["e"].outputs["t.result"]
    assert graph.problems == [] and (tables.many, tables.type.record is not None) == (True, True)


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
        REGISTRY.add(stored_graph(embedded, title=embedded.capitalize(), description=f"The {embedded} node.")(GraphRecord.model_validate(chain(("d", node_id)))))
    return embedded


# -- running: once and per item, and what is typed or wired on the embedding ---------------------------------


def test_an_embedded_graph_runs_once_and_per_item_as_its_nodes_would_drawn_flat():
    run = load({"e": "shout-inner"}).run(text="ab")
    assert run.status == "complete" and run.outputs == {"e.ex.result": "AB!"}
    graph = mapped("shout-inner")
    assert graph.runs_on["e"].mode == "per_item"
    run = graph.run(text="ab cd")
    assert run.status == "complete" and run.results["e"] == {"ex.result": ["AB!", "CD!"]}
    assert run.results["e"]["ex.result"] == flat("shout-inner").run(text="ab cd").results["ex"]["result"]
    two = load({"a": "shout-inner", "b": "shout-inner"}, [("a.ex.result", "b.text")])
    assert two.run(text="x").outputs == {"b.ex.result": "X!!"}


def test_a_value_a_default_and_an_edge_on_the_embedding_are_any_nodes():
    graph = load({"w": "words", "e": {"node": "greet-inner", "values": {"greeting": "hej"}}}, [("w.result", "e.text")])
    assert graph.run(text="ab cd").results["e"]["g.result"] == ["hej ab", "hej cd"]
    first = load({"e": {"node": "greet-inner", "defaults": {"greeting": "hi"}}})
    assert first.inputs["greeting"].field.default == "hi"
    assert first.run(text="ab").outputs == {"e.g.result": "hi ab"}  # the embedding's default, proposed to the caller
    assert first.run(text="ab", greeting="yo").outputs == {"e.g.result": "yo ab"}
    assert mapped("greet-inner").run(text="ab").results["e"]["g.result"] == ["hello ab"]  # the inner node's own default
    wired = load({"w": "words", "g": "upper", "e": "greet-inner"}, [("w.result", "e.text"), ("w.result", "g.text"), ("g.result", "e.greeting")])
    assert wired.run(text="ab cd").results["e"]["g.result"] == ["AB ab", "CD cd"]


def test_deep_outputs_are_what_run_outputs_gives_alone_and_embedded():
    alone = REGISTRY.load(GraphRecord.model_validate(INNER["deep-inner"])).run(text="ab cd").outputs["up.result"]
    assert alone == [["A", "B"], ["C", "D"]]  # two maps deep, a list for each word, in item order
    graph = load({"e": "deep-inner", "n": "count", "x": "exclaim"}, [("e.up.result", "n.items"), ("e.up.result", "x.text")])
    run = graph.run(text="ab cd")
    assert run.results["e"]["up.result"] == alone and run.results["n"]["result"] == [2, 2]  # count is called once per word
    assert run.results["x"]["result"] == [["A!", "B!"], ["C!", "D!"]]
    assert load({"e": "series-per-item-inner"}).run(text="ab cd").outputs == {"e.c.result": [["a", "b"], ["c", "d"]]}


def test_a_branch_not_taken_inside_skips_what_reads_the_embedding_as_it_would_drawn_flat():
    graph = load({"w": "words", "e": chain_node("decide"), "s": "exclaim", "l": "upper"}, [("w.result", "e.text"), ("e.d.short", "s.text"), ("e.d.long", "l.text")])
    run = graph.run(text="ab long")
    assert run.status == "complete" and run.results["s"]["result"] == ["ab!", None] and run.results["l"]["result"] == [None, "LONG"]
    assert run.results["e"] == {"d.short": ["ab", None], "d.long": [None, "long"]}
    inside = load({"w": "words", "e": "deciding-inner", "n": "count"}, [("w.result", "e.text"), ("e.s.result", "n.items")])
    assert inside.run(text="ab long cd").results["n"]["result"] == 2  # the item whose branch was not taken is left out of a gather


def test_an_item_whose_branch_was_not_taken_outside_skips_the_nodes_inside_for_that_item():
    graph = load({"w": "words", "d": "decide", "e": "shout-inner", "n": "count"}, [("w.result", "d.text"), ("d.short", "e.text"), ("e.ex.result", "n.items")])
    run = graph.run(text="ab long cd")
    assert run.status == "complete" and run.results["e"] == {"ex.result": ["AB!", None, "CD!"]}
    assert run.results["n"]["result"] == 2 and run.not_called == []  # skipped, not blank: the item is left out of the gather
    assert CALLS["upper"] == 2 and CALLS["exclaim"] == 2


def test_a_series_input_inside_gathers_what_the_embedding_is_wired_to():
    graph = load({"w": "words", "up": "upper", "e": "count-inner"}, [("w.result", "up.text"), ("up.result", "e.items")])
    assert graph.runs_on["e"].mode == "once"
    assert graph.run(text="ab cd ef").outputs == {"e.n.result": 3}
    per_document = load({"s": "sentences", "w": "words", "e": "count-inner"}, [("s.result", "w.text"), ("w.result", "e.items")])
    assert per_document.run(text="a b;c").outputs == {"e.n.result": [2, 1]}


def test_a_blank_inside_is_listed_as_not_called_by_its_path():
    run = mapped("maybe-inner").run(text="a none")
    assert run.status == "complete" and run.results["e"]["up.result"] == ["A", None]
    assert [(n.loc, n.input) for n in run.not_called] == [(("e", 1, "up"), "text")]
    blank_in = load({"w": "words", "m": "maybe", "e": "shout-inner"}, [("w.result", "m.text"), ("m.result", "e.text")]).run(text="a none")
    assert blank_in.results["e"]["ex.result"] == ["A!", None]
    assert [(n.loc, n.input) for n in blank_in.not_called] == [(("e", 1, "up"), "text"), (("e", 1, "ex"), "text")]  # the nodes inside, as drawn flat
    drawn_flat = load({"w": "words", "m": "maybe", "up": "upper", "ex": "exclaim"}, [("w.result", "m.text"), ("m.result", "up.text"), ("up.result", "ex.text")]).run(text="a none")
    assert [(n.loc, n.input) for n in drawn_flat.not_called] == [(("up", 1), "text"), (("ex", 1), "text")]


def test_a_node_inside_takes_what_the_host_gives_every_run():
    graph = mapped("signed-inner")
    assert graph.run(text="a b", from_run={Session: Session("ida")}).results["e"]["s.result"] == ["a by ida", "b by ida"]
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text="a b")
    assert [p.type for p in refused.value.errors()] == ["from_run_missing"]


def test_the_item_ceiling_holds_inside_and_is_raised_by_resume():
    graph = mapped("deep-inner")
    run = graph.run(text="ab cdef", max_items=3)
    assert run.status == "stopped" and run.state.cause.type == "item_ceiling" and run.state.cause.loc == ("e", 1, "c")
    resumed = graph.load_run(run.to_json()).with_max_items(10).resume()
    assert resumed.status == "complete" and resumed.results["e"]["up.result"] == [[["A", "B"]], [["C", "D", "E", "F"]]]
    assert graph.load_run(run.to_json()).resume().status == "stopped"  # the ceiling it was saved under still holds inside


# -- pausing inside, saved, loaded in a fresh process --------------------------------------------------------

_CHILD = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("embedding_nodes", {module!r})
nodes = importlib.util.module_from_spec(spec)
sys.modules["embedding_nodes"] = nodes
spec.loader.exec_module(nodes)
graph = nodes.mapped("asking-inner")
run = graph.load_run(open({path!r}).read())
ids = [w.id for w in run.waiting]
half = run.answer(ids[0], "first").resume()
open({path!r}, "w").write(half.to_json())
print(json.dumps({{"status": half.status, "ids": ids, "left": [w.id for w in half.waiting], "calls": dict(nodes.CALLS)}}))
"""


def test_pausing_inside_a_mapped_embedding_half_answered_in_a_fresh_process_and_finished(tmp_path: pathlib.Path):
    graph = mapped("asking-inner")
    run = graph.run(text="ab cd")
    assert run.status == "waiting"
    ids = [w.id for w in run.waiting]
    assert ids[0].startswith("e@0/q~") and ids[1].startswith("e@1/q~")
    assert [w.loc for w in run.waiting] == [("e", 0, "q"), ("e", 1, "q")]
    assert run.results["e"] == {"q.result": [None, None]}  # the embedding has no unit until what it offers is settled
    path = tmp_path / "run.json"
    path.write_text(run.to_json())
    CALLS.clear()
    child = subprocess.run([sys.executable, "-c", _CHILD.format(module=__file__, path=str(path))], capture_output=True, text=True, timeout=120, check=False)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == {"status": "waiting", "ids": ids, "left": ids[1:], "calls": {}}  # the fresh process calls nothing again
    done = graph.load_run(path.read_text()).answer(ids[1], "second").resume()
    assert done.status == "complete" and done.results["e"]["q.result"] == ["first", "second"]
    assert CALLS == {}
    assert graph.load_run(done.to_json()).outputs == done.outputs  # the embedding's unit is not saved: it is read back from the units inside
    assert sorted(json.loads(done.to_json())["units"]) == ["e/q", "e/up", "w"]  # a node inside is saved as any node, under its path


def test_an_answer_inside_is_refused_by_its_path():
    graph = mapped("asking-inner")
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
    assert run.answer("e@1/q", "by place").resume().results["e"]["q.result"] == [None, "by place"]


# -- policy: a node inside is called, retried, timed out and cancelled as the same node drawn flat -----------


def test_an_inner_nodes_concurrency_bounds_the_node_across_the_embeddings_items():
    run = mapped("fetch-inner").run(text="a b c d e f")
    assert run.status == "complete" and run.results["e"]["up.result"] == ["A?", "B?", "C?", "D?", "E?", "F?"]
    assert CALLS["fetch"] == 6 and AT_ONCE["peak"] == 2  # Policy(concurrency=2): two at once, not one, and not two per item of the embedding
    AT_ONCE.update(now=0, peak=0)
    assert flat("fetch-inner").run(text="a b c d e f").status == "complete" and AT_ONCE["peak"] == 2


def test_a_cancel_inside_keeps_the_calls_on_threads_and_a_resume_repeats_none():
    graph = mapped("fetch-inner")
    events = stream(graph.with_inputs(text="a b c d"), cancel_when=lambda e: isinstance(e, NodeStart) and e.loc == ("e", "f"))
    cancelled = events[-1].run
    assert cancelled.status == "cancelled" and cancelled.state.cause.type == "run_cancelled"
    assert CALLS["fetch"] in (1, 2)  # the ones that had a thread when the cancel was seen; those waiting for one were dropped
    assert cancelled.results["e"] == {"up.result": [None, None, None, None]}
    resumed = graph.load_run(cancelled.to_json()).resume()
    assert resumed.status == "complete" and resumed.results["e"]["up.result"] == ["A?", "B?", "C?", "D?"]
    assert CALLS["fetch"] == 4  # the calls kept were not made again


def test_the_leg_timeout_crosses_into_the_embedded_graph():
    graph = mapped("fetch-inner")
    run = graph.run(text="a b c d", timeout=0.05)
    assert run.status == "timed_out" and run.state.cause.type == "run_timed_out"
    assert CALLS["fetch"] == 2
    assert graph.load_run(run.to_json()).resume().status == "complete" and CALLS["fetch"] == 4


def test_an_inner_nodes_retry_is_its_own_policy():
    events = stream(mapped("flaky-inner").with_inputs(text="a b"))
    assert events[-1].run.status == "complete" and events[-1].run.results["e"]["f.result"] == ["a~", "b~"]
    assert CALLS["flaky"] == 4
    retries = sorted((e.loc, e.attempt, e.cause.loc) for e in events if isinstance(e, NodeRetry))
    assert retries == [(("e", 0, "f"), 1, ("e", 0, "f")), (("e", 1, "f"), 1, ("e", 1, "f"))]


def test_an_inner_nodes_attempt_timeout_fails_the_run_at_its_path_and_keeps_the_other_items_work():
    graph = mapped("stuck-inner")
    run = graph.run(text="ok slow")
    assert run.status == "failed" and run.state.cause.type == "timeout" and run.state.cause.loc == ("e", 1, "s")
    assert run.results["e"]["up.result"] == ["OK", None]
    CALLS.clear()
    assert graph.load_run(run.to_json()).resume().status == "failed" and CALLS == {"stuck": 1}  # only the unit that failed


def test_a_failure_inside_keeps_the_finished_work_inside_for_a_resume():
    graph = mapped("boom-inner")
    events = stream(graph.with_inputs(text="ok bad"))
    run = events[-1].run
    assert run.status == "failed" and run.state.cause.loc == ("e", 1, "b") and run.state.cause.type == "failed"
    assert [(e.loc, e.cause.loc) for e in events if isinstance(e, NodeFailed)] == [(("e", "b"), ("e", 1, "b"))]  # the node, and in its cause the unit
    again = graph.load_run(run.to_json()).resume()
    assert again.status == "failed" and CALLS["upper"] == 2 and CALLS["boom"] == 3  # only the failed unit is called again


def test_a_checkpoint_mid_leg_loads_and_resumes_with_no_inner_call_repeated():
    graph = mapped("fetch-inner")
    events = stream(graph.with_inputs(text="a b c d"), checkpoint_every=0.01)
    assert events[-1].run.status == "complete" and CALLS["fetch"] == 4
    mid = next(e.run for e in events if isinstance(e, RunCheckpoint) and 0 < _made(e.run, "e", "f") < 4)
    assert mid.status == "running"
    made = _made(mid, "e", "f")
    resumed = graph.load_run(mid.to_json()).resume()  # as a process that died mid-leg would
    assert resumed.status == "complete" and resumed.results["e"]["up.result"] == ["A?", "B?", "C?", "D?"]
    assert CALLS["fetch"] == 4 + (4 - made)  # only the calls the checkpoint holds no record of


def _made(run, embedding: str, node_id: str) -> int:
    """How many units ``node_id`` inside ``embedding`` holds."""
    return len(run._outcomes[f"{embedding}/{node_id}"])


# -- one address: a path of node ids and item positions -----------------------------------------------------


def test_a_node_inside_says_its_events_once_by_its_path_as_drawn_flat():
    events = stream(mapped("asking-inner").with_inputs(text="ab cd"))
    starts = [e.loc for e in events if isinstance(e, NodeStart)]
    assert starts == [("w",), ("e",), ("e", "up"), ("e", "q")]  # once per node, whatever the items: the embedding first
    assert [(e.loc, e.units) for e in events if isinstance(e, NodeComplete)] == [(("w",), 1), (("e", "up"), 2)]
    flat_events = stream(flat("asking-inner").with_inputs(text="ab cd"))
    assert [e.loc for e in flat_events if isinstance(e, NodeStart)] == [("outer",), ("up",), ("q",)]
    waiting = [e.waiting for e in events if isinstance(e, NodeWaiting)]
    assert [w.id for w in waiting] == [w.id for w in events[-1].run.waiting] and waiting[1].loc == ("e", 1, "q")
    assert isinstance(events[-1], RunEnd) and events[-1].status == "waiting"
    answered = events[-1].run.answer({w.id: "yes" for w in waiting})
    later = stream(answered)
    assert not [e for e in later if isinstance(e, NodeStart)]  # every node started on the first leg
    assert [(e.loc, e.units) for e in later if isinstance(e, NodeComplete)] == [(("e", "q"), 2), (("e",), 2)]


def test_an_embedded_graph_inside_an_embedded_graph_goes_by_path():
    graph = mapped("wrapping")
    assert graph.outputs == ["e.in.q.result"]
    run = graph.run(text="ab cd")
    ids = [w.id for w in run.waiting]
    assert ids[0].startswith("e@0/in/q~") and ids[1].startswith("e@1/in/q~") and run.waiting[1].loc == ("e", 1, "in", "q")
    with pytest.raises(InvalidCallError) as refused:
        run.answer("e@1/in/q~00000000", "x")
    assert refused.value.errors()[0].loc == ("e", 1, "in", "q")
    done = graph.load_run(run.answer({ids[1]: "two"}).resume().to_json()).answer(ids[0], "one").resume()
    assert done.status == "complete" and done.outputs == {"e.in.q.result": ["one", "two"]} and CALLS["upper"] == 2


# -- reuse: embedding a graph changes nothing about what again reuses ---------------------------------------


def test_again_calls_inside_only_what_it_would_drawn_flat_after_a_change_and_after_an_insert():
    for graph, at in ((mapped("shout-inner"), ("e", "ex.result")), (flat("shout-inner"), ("ex", "result"))):
        run = graph.run(text="ab cd ef")
        CALLS.clear()
        assert run.again(text="ab xx ef").results[at[0]][at[1]] == ["AB!", "XX!", "EF!"]
        assert CALLS == {"words": 1, "upper": 1, "exclaim": 1}
        CALLS.clear()
        assert run.again(text="zz ab cd ef").results[at[0]][at[1]] == ["ZZ!", "AB!", "CD!", "EF!"]
        assert CALLS == {"words": 1, "upper": 1, "exclaim": 1}
        CALLS.clear()
        assert graph.load_run(run.again(text="ab").to_json()).again(text="ab cd").status == "complete"
        assert CALLS == {"words": 2, "upper": 1, "exclaim": 1}  # what a run carried and didn't take is saved, and taken later


def test_an_inner_node_that_reads_the_world_is_called_on_again():
    run = mapped("stamped-inner").run(text="ab cd")
    CALLS.clear()
    assert run.again().results["e"]["up.result"] == ["AB", "CD"]
    assert CALLS == {"stamp": 2}  # the embedding's unit is never taken whole: stamp is called, upper reused


def test_two_documents_with_an_identical_inner_line_each_keep_their_own_answer():
    graph = mapped("lines-inner", "sentences")  # a question per word of each document
    run = graph.run(text="a b;a c")
    run = run.answer({w.id: f"doc{w.loc[1]}:{w.inputs['text']}" for w in run.waiting}).resume()
    assert run.status == "complete" and run.results["e"]["q.result"] == [["doc0:a", "doc0:b"], ["doc1:a", "doc1:c"]]
    changed = run.again(text="a b d;a c")
    assert [w.loc[:2] for w in changed.waiting] == [("e", 0)] * 3  # document 0 changed, so its lines are asked again, as drawn flat
    assert changed.answer({w.id: "new" for w in changed.waiting}).resume().results["e"]["q.result"] == [["new", "new", "new"], ["doc1:a", "doc1:c"]]
    inserted = run.again(text="x;a b;a c")
    assert [(w.loc, w.inputs["text"]) for w in inserted.waiting] == [(("e", 0, "q", 0), "x")]
    assert inserted.answer(inserted.waiting[0], "x").resume().results["e"]["q.result"] == [["x"], ["doc0:a", "doc0:b"], ["doc1:a", "doc1:c"]]
    swapped = run.again(text="a c;a b")
    assert swapped.status == "complete" and swapped.results["e"]["q.result"] == [["doc1:a", "doc1:c"], ["doc0:a", "doc0:b"]]
    flat_run = flat("lines-inner", "sentences").run(text="a b;a c")
    flat_run = flat_run.answer({w.id: f"doc{w.loc[1]}:{w.inputs['text']}" for w in flat_run.waiting}).resume()
    assert flat_run.again(text="a c;a b").results["q"]["result"] == [["doc1:a", "doc1:c"], ["doc0:a", "doc0:b"]]  # the same, drawn flat


def test_identical_items_take_their_answers_in_item_order_whatever_thread_finishes_first():
    graph = mapped("staggered-inner")
    run = graph.run(text="ab ab ab")
    run = run.answer({w.id: f"answer {w.loc[1]}" for w in run.waiting}).resume()
    assert run.results["e"]["q.result"] == ["answer 0", "answer 1", "answer 2"]
    CALLS.clear()
    again = graph.load_run(run.to_json()).again()  # stagger is called again, and its first call comes back last
    assert again.status == "complete" and CALLS["stagger"] == 3 and CALLS["ask"] == 0
    assert again.results["e"]["q.result"] == ["answer 0", "answer 1", "answer 2"]
    CALLS.clear()
    longer = run.again(text="zz ab ab ab")
    assert [w.loc for w in longer.waiting] == [("e", 0, "q")]
    assert longer.results["e"]["q.result"] == [None, "answer 0", "answer 1", "answer 2"]


def test_reuse_goes_through_an_embedded_graph_inside_an_embedded_graph_and_forget_forgets_all_of_it():
    graph = mapped("wrapping")
    run = graph.run(text="ab cd")
    done = run.answer({w.id: w.inputs["text"].lower() for w in run.waiting}).resume()
    assert done.outputs == {"e.in.q.result": ["ab", "cd"]}
    CALLS.clear()
    longer = done.again(text="ab cd ef")
    assert longer.status == "waiting" and CALLS == {"words": 1, "upper": 1, "ask": 1}
    assert [w.id.partition("~")[0] for w in longer.waiting] == ["e@2/in/q"]
    CALLS.clear()
    forgotten = done.forget("e").resume()
    assert forgotten.status == "waiting" and CALLS == {"upper": 2, "ask": 2}


def test_work_carried_from_a_run_of_an_earlier_graph_is_taken_inside_where_the_node_is_as_it_was():
    run = mapped("shout-inner").run(text="ab cd")
    edited = load({"w": "words", "e": "shout-inner", "n": "count"}, [("w.result", "e.text"), ("e.ex.result", "n.items")])
    CALLS.clear()
    assert edited.with_inputs(text="ab cd", earlier=run).resume().results["n"]["result"] == 2
    assert CALLS == {}


@pytest.mark.parametrize(
    ("nodes", "edges", "series"),
    [
        ({"w": "words", "t": "letters", "h": "height"}, [("w.result", "t.text"), ("t.result", "h.table")], "w.result"),
        ({"e": "letters-per-word-inner", "h": "height"}, [("e.t.result", "h.table")], "e.t.result"),
        ({"s": "sheets", "h": "height"}, [("s.result", "h.table")], "s.result"),
    ],
    ids=["drawn flat", "made inside an embedded graph", "returned as a Series of tables"],
)
def test_tables_made_per_item_reach_a_table_input_one_table_per_item_wherever_they_are_made(nodes: dict[str, str], edges: list[tuple[str, str]], series: str):
    graph = load(nodes, edges)
    assert not graph.problems, graph.problems
    run = graph.run(text="ab c")
    assert (graph.runs_on["h"].mode, graph.runs_on["h"].series, run.status, run.results["h"]["result"]) == ("per_item", series, "complete", [2, 1])


# -- an embedded graph's node reads alike on a run, on its copy and on either saved and loaded ---------------


def read(run) -> tuple[Any, ...]:
    """Everything a host reads of ``run``."""
    waiting = [unit.model_dump() for unit in run.waiting]
    not_called = [call.model_dump() for call in run.not_called]
    return run.status, run.results, waiting, not_called, run.outputs if run.status == "complete" else None


def reads_alike_loaded(run) -> bool:
    assert read(run) == read(run.graph.load_run(run.to_json()))
    return True


@pytest.mark.parametrize("inner", ["asking-inner", "lines-inner", "wrapping", "staggered-inner"])
@pytest.mark.parametrize("per_item", [False, True], ids=["once", "per item"])
def test_an_embedded_graph_reads_alike_live_and_loaded_at_every_pause_and_after_every_answer(inner: str, per_item: bool):
    graph = mapped(inner, "sentences") if per_item else load({"e": inner})
    run = graph.run(text="ab cd;ef")
    assert run.status == "waiting" and reads_alike_loaded(run)
    while run.status == "waiting":
        first, *rest = run.waiting
        one = run.answer(first, "yes")  # one answer, not resumed: what it settles reads as it will loaded
        assert reads_alike_loaded(one) and read(run) == read(run.graph.load_run(run.to_json()))  # and the run it was given to is as it was
        answered = one.answer({unit.id: "yes" for unit in rest})
        assert reads_alike_loaded(answered)
        run = answered.resume()
        assert reads_alike_loaded(run)
    assert run.status == "complete"


def test_an_answer_inside_an_embedded_graph_shows_in_its_results_before_the_run_goes_on_as_a_flat_nodes_does():
    embedded, drawn = load({"e": "asking-inner"}).run(text="a"), load({"up": "upper", "q": "ask"}, [("up.result", "q.text")]).run(text="a")
    assert embedded.answer(embedded.waiting[0], "yes").results == {"e": {"q.result": "yes"}}
    assert drawn.answer(drawn.waiting[0], "yes").results == {"up": {"result": "A"}, "q": {"result": "yes"}}
    assert embedded.results == {"e": {"q.result": None}}  # the run the answer was given to holds none


def test_a_leg_stopped_at_the_ceiling_keeps_the_calls_on_threads_inside_an_embedded_graph_and_the_embedded_graph_reads_them():
    graph = load({"e": "fetch-words-inner", "many": "words", "each": "upper"}, [("many.result", "each.text")])
    stopped = graph.run({"e.text": "a b", "many.text": "x y z"}, max_items=2)
    assert stopped.status == "stopped" and stopped.state.cause.loc == ("many",)
    assert stopped.results["e"] == {"f.result": ["a?", "b?"]} and reads_alike_loaded(stopped)


def test_a_failure_in_one_item_leaves_the_embedded_graph_of_the_items_before_it_read_alike_live_and_loaded():
    failed = mapped("boom-inner").run(text="ab bad cd")
    assert failed.status == "failed" and failed.results["e"] == {"b.result": ["AB", None, None]} and reads_alike_loaded(failed)


def test_a_blank_inside_an_embedded_graph_reads_alike_live_and_loaded():
    run = mapped("maybe-inner").run(text="ab none cd")
    assert run.status == "complete" and [call.loc for call in run.not_called] == [("e", 1, "up")] and reads_alike_loaded(run)


def test_a_checkpoint_mid_leg_reads_alike_loaded_whatever_inside_an_embedded_graph_has_come_back():
    events = stream(mapped("upper-fetch-inner").with_inputs(text="a b c d"), checkpoint_every=0.01)
    checkpoints = [event.run for event in events if isinstance(event, RunCheckpoint)]
    assert checkpoints and all(reads_alike_loaded(checkpoint) for checkpoint in checkpoints)


@pytest.mark.parametrize("inner", ["two-asks-inner", "three-asks-inner"])
@pytest.mark.parametrize("per_item", [False, True], ids=["once", "per item"])
def test_several_answers_in_one_call_under_embedded_graphs_side_by_side_inside_another_settle_the_one_around_them(inner: str, per_item: bool):
    nodes: dict[str, str | dict[str, Any]] = {"w": "words", "e": inner} if per_item else {"e": inner}
    graph = load(nodes, [("w.result", f"e.{name}") for name in load({"e": inner}).inputs] if per_item else [])
    assert not graph.problems, graph.problems
    run = graph.run(dict.fromkeys(graph.inputs, "ab cd"))
    answered = run.answer({unit.id: "yes" for unit in run.waiting})
    assert reads_alike_loaded(answered)
    assert all(value == (["yes", "yes"] if per_item else "yes") for value in answered.results["e"].values())
    some = run.answer({unit.id: "yes" for unit in run.waiting[1:]})  # all but the first: nothing around it is settled
    assert reads_alike_loaded(some) and reads_alike_loaded(some.answer(run.waiting[0], "yes"))


def test_every_word_answered_in_one_call_inside_one_embedded_graph_settles_it_once():
    run = load({"e": "lines-inner"}).run(text=" ".join(f"w{i}" for i in range(300)))
    answered = run.answer({unit.id: "yes" for unit in run.waiting})
    assert answered.results["e"] == {"q.result": ["yes"] * 300} and reads_alike_loaded(answered)


@pytest.mark.parametrize(
    ("nodes", "edges"),
    [
        ({"w": "words", "r": "record", "h": "height"}, [("w.result", "r.text"), ("r.result", "h.table")]),
        ({"e": "record-per-word-inner", "h": "height"}, [("e.r.result", "h.table")]),
    ],
    ids=["drawn flat", "made inside an embedded graph"],
)
def test_values_of_any_type_made_per_item_are_gathered_as_records_by_a_table_input_wherever_they_are_made(nodes: dict[str, str], edges: list[tuple[str, str]]):
    graph = load(nodes, edges)
    assert not graph.problems, graph.problems
    run = graph.run(text="ab c d")
    assert (graph.runs_on["h"].mode, run.status, run.results["h"]["result"]) == ("once", "complete", 3)


def _said_of(events, node_id: str) -> list[tuple[str, tuple]]:
    return [(event.type, event.loc) for event in events if getattr(event, "loc", ())[:1] == (node_id,)]


def test_an_embedded_graph_with_no_output_says_the_same_on_a_fresh_run_and_on_one_loaded_before_anything_inside_ran():
    graph = load({"e": "quiet-inner"})
    said = [("node_start", ("e",)), ("node_start", ("e", "n")), ("node_complete", ("e", "n")), ("node_complete", ("e",))]
    assert _said_of(stream(graph.with_inputs(text="x")), "e") == said
    assert _said_of(stream(graph.load_run(graph.with_inputs(text="x").to_json())), "e") == said


def test_an_embedded_graph_with_no_output_is_done_when_the_nodes_inside_it_are_and_not_before():
    graph = load({"q": "ask", "e": "quiet-inner"}, [("q.result", "e.text")])
    CALLS.clear()
    waiting = graph.run(text="x")
    first = stream(graph.with_inputs(text="x"))
    assert (waiting.status, _said_of(first, "e"), CALLS["note"]) == ("waiting", [], 0)
    for run in (waiting, graph.load_run(waiting.to_json())):
        said = _said_of(stream(run.answer(run.waiting[0].id, "y")), "e")
        assert said == [("node_start", ("e",)), ("node_start", ("e", "n")), ("node_complete", ("e", "n")), ("node_complete", ("e",))]


def test_an_embedded_graph_of_no_nodes_runs_to_nothing_as_a_graph_of_no_nodes_does():
    registry = NodeRegistry()
    registry.add(upper)
    registry.add(stored_graph("nothing", title="Nothing", description="The nothing node.")(GraphRecord.model_validate({"nodes": [], "edges": []})))
    alone = registry.load(GraphRecord.model_validate({"nodes": [], "edges": []})).run()
    graph = registry.load(GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper"}, {"id": "e", "node": "nothing"}], "edges": []}))
    run = graph.run(text="x")
    assert graph.problems == [] and (alone.status, alone.results) == ("complete", {})
    assert (run.status, run.results) == ("complete", {"u": {"result": "X"}, "e": {}})
    assert graph.load_run(run.to_json()).results == run.results
