"""A node that collects a loop, when a value from outside the loop is ``None`` or on a branch not chosen.

The outside value decides the collecting node only when it decides every item alike. In the graph
``items -> pair -> count``, ``items`` returns ``["a", "b"]``, ``pair`` is called per item with its input ``b`` wired
to ``outside-value``, and ``count`` collects ``pair.result``. When ``outside-value`` returns ``None`` (``pair.b`` takes
no ``None``), no call of ``pair`` can return a value, so ``count`` is not called, for two items as for none. When
``outside-value`` is on a branch not chosen, ``count`` is skipped. Otherwise each item decides, and a loop of no items
is collected as an empty list.

``outside-value`` returns its input ``kind``: ``None`` for ``"no-value"``, ``SKIPPED`` for ``"skip"``. ``maybe`` does the
same per item. A node named ``...-or-none`` takes ``None`` for its whole input; one named ``...-taking-none`` takes a
list whose items may be ``None``. A series is "never returned" when the node that would return it was not called or
was skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    Ask,
    FunctionDefinition,
    Graph,
    GraphRecord,
    NodeDefinition,
    NodeRegistry,
    Param,
    RecordGraphInput,
    Run,
    Series,
    SubgraphDefinition,
    make_graph,
)

REGISTRY = NodeRegistry()


def _node(node_id: str, **declared: Any) -> Any:
    return REGISTRY.node(node_id, title=node_id, description=node_id, **declared)


@_node("items")
def items(text: str) -> Series[str]:
    return text.split()


@_node("outside-value")
def outside_value(kind: str) -> str | None:
    return SKIPPED if kind == "skip" else None if kind == "no-value" else kind


@dataclass(frozen=True)
class Way:
    yes: Annotated[str, Param(exclusive_group="way")] = SKIPPED
    no: Annotated[str, Param(exclusive_group="way")] = SKIPPED


@_node("decide")
def decide(go: str) -> Way:
    return Way(yes=go) if go.lower().startswith("y") else Way(no=go)


@_node("decide-with")
def decide_with(go: str, extra: str) -> Way:
    return Way(yes=go) if go.lower().startswith("y") else Way(no=go)


@_node("pair")
def pair(a: str, b: str) -> str:
    return f"{a}+{b}"


@_node("three")
def three(a: str, b: str, c: str) -> str:
    return f"{a}+{b}+{c}"


@_node("up")
def up(a: str) -> str:
    return a.upper()


@_node("up-or-none")
def up_or_none(a: str | None) -> str:
    return "none" if a is None else a.upper()


@_node("maybe")
def maybe(a: str) -> str | None:
    return None if a == "no-value" else SKIPPED if a == "skip" else a.upper()


@_node("chars")
def chars(a: str) -> Series[str]:
    return list(a)


@_node("count")
def count(values: Series[str]) -> int:
    return len(values)


@_node("count-or-none")
def count_or_none(values: Series[str] | None) -> str:
    return "none" if values is None else str(len(values))


@_node("dashes-for-none")
def dashes_for_none(values: Series[str | None]) -> str:
    return "".join("-" if value is None else "x" for value in values)


@_node("join")
def join(values: Series[str]) -> str:
    return "".join(values)


@_node("join-or-none")
def join_or_none(values: Series[str] | None) -> str:
    return "none" if values is None else "".join(values)


@_node("join-taking-none")
def join_taking_none(values: Series[str | None]) -> str:
    return "".join("-" if value is None else value for value in values)


@_node("with-all")
def with_all(a: str, values: Series[str]) -> str:
    return a + "".join(values)


@_node("with-all-taking-none")
def with_all_taking_none(a: str, values: Series[str | None]) -> str:
    return a + "".join("-" if value is None else value for value in values)


@dataclass(frozen=True)
class Listed:
    items: Series[str]
    note: str | None


@_node("say")
def say(text: str) -> str:
    return text


@_node("items-and-note")
def items_and_note(text: str, note: str | None) -> Listed:
    return Listed(text.split(), note)


REGISTRY.subgraph("pair-then-up-or-none", title="Pair then up", description="A pair, in capitals, or none.")(
    make_graph(pair, up_or_none, edges=[("pair.result", "up-or-none.a")])
)


REGISTRY.subgraph("items-then-up", title="Items then up", description="Each word of a text, in capitals.")(
    make_graph(items, up, edges=[("items.result", "up.a")])
)


REGISTRY.subgraph("pair-then-items-then-up", title="Pair then items then up", description="Each word of a pair, in capitals.")(
    make_graph(pair, items, up, edges=[("pair.result", "items.text"), ("items.result", "up.a")])
)


def _inputs_of(definition: NodeDefinition) -> list[str]:
    match definition:
        case FunctionDefinition():
            return list(definition.inputs)
        case SubgraphDefinition():
            return [entry.graph_input for node in definition.graph.nodes.values() for entry in node.inputs.values() if isinstance(entry, RecordGraphInput)]


def _outputs_of(definition: NodeDefinition) -> list[str]:
    match definition:
        case FunctionDefinition():
            return list(definition.outputs)
        case SubgraphDefinition():
            return list(definition.graph.outputs)


def record_of(nodes: Mapping[str, str], edges: Sequence[tuple[str, str]], names: Sequence[str], inputs: Mapping[str, str] | None = None) -> GraphRecord:
    """The record of ``nodes`` (registry id by node id) and ``edges`` (``("node.output", "node.input")``).

    Each name in ``names`` is a graph input fed to every input of that name no edge leads into; ``inputs`` names a
    graph input fed to one input (``{"first": "e.go"}``). Each output of a node no edge leaves is a graph output,
    named ``"node-output"``.
    """
    entries: dict[str, dict[str, Any]] = {node_id: {} for node_id in nodes}
    for source, target in edges:
        source_node, output = source.split(".")
        target_node, name = target.split(".")
        entries[target_node].setdefault(name, {"sources": []})["sources"].append({"node": source_node, "output": output})
    for name, place in (inputs or {}).items():
        node_id, input_name = place.split(".")
        entries[node_id][input_name] = {"graph_input": name}
    for node_id, registry_id in nodes.items():
        for name in _inputs_of(REGISTRY[registry_id]):
            if name in names and name not in entries[node_id]:
                entries[node_id][name] = {"graph_input": name}
    led_out = {source.split(".")[0] for source, _ in edges}
    outputs = {f"{node_id}-{output}": {"node": node_id, "output": output} for node_id, registry_id in nodes.items() if node_id not in led_out for output in _outputs_of(REGISTRY[registry_id])}
    return GraphRecord.model_validate({"nodes": {node_id: {"node": nodes[node_id], "inputs": entries[node_id]} for node_id in nodes}, "outputs": outputs})


def ran(nodes: dict[str, str], edges: list[tuple[str, str]], inputs: Mapping[str, str] | None = None, **values: str) -> Any:
    """A complete run of ``record_of(nodes, edges, values, inputs)`` given ``values``."""
    graph = Graph.compile(record_of(nodes, edges, list(values), inputs), REGISTRY)
    assert not graph.problems, graph.problems
    run = graph.run(**values)
    assert run.status == "complete"
    return run


def outcome_of(run: Any, node_id: str) -> Any:
    """The outcome of a node inside no loop: its result, ``"not called"`` (an argument was ``None`` where its input
    takes no ``None``), or ``"skipped"``."""
    result = run.results[node_id]["result"]
    if result is not None:
        return result
    return "not called" if any(call.loc == (node_id,) for call in run.not_called) else "skipped"


BOTH_WAYS = (
    {"s": "items", "w": "decide", "t": "outside-value", "y": "three", "z": "pair", "p": "up", "n": "count"},
    [("s.result", "y.a"), ("w.yes", "y.b"), ("t.result", "y.c"), ("s.result", "z.a"), ("w.no", "z.b"), ("y.result", "p.a"), ("z.result", "p.a"), ("p.result", "n.values")],
)


@pytest.mark.parametrize(("go", "kind", "text", "n"), [
    ("no", "no-value", "a b", 2), ("no", "no-value", "", 0), ("no", "x", "a b", 2),
    ("yes", "no-value", "a b", "not called"), ("yes", "no-value", "", "not called"), ("yes", "x", "a", 1),
])
def test_a_none_outside_the_loop_on_the_branch_not_chosen_does_not_decide_the_collecting_node(go: str, kind: str, text: str, n: Any) -> None:
    assert outcome_of(ran(*BOTH_WAYS, go=go, kind=kind, text=text), "n") == n


@pytest.mark.parametrize(("kind", "text", "n"), [("no-value", "a b", 2), ("no-value", "", 0), ("x", "a b", 2), ("skip", "a b", "skipped"), ("skip", "", "skipped")])
def test_a_subgraph_in_the_loop_that_takes_none_from_outside_gives_each_item_a_value(kind: str, text: str, n: Any) -> None:
    run = ran({"s": "items", "t": "outside-value", "e": "pair-then-up-or-none", "n": "count"}, [("s.result", "e.a"), ("t.result", "e.b"), ("e.result", "n.values")], kind=kind, text=text)
    assert outcome_of(run, "n") == n


def _nested_loops(joiner: str, kind: str, text: str) -> Any:
    nodes = {"s": "items", "t": "outside-value", "c": "chars", "m": "pair", "j": joiner, "n": "count"}
    return ran(nodes, [("s.result", "c.a"), ("c.result", "m.a"), ("t.result", "m.b"), ("m.result", "j.values"), ("j.result", "n.values")], kind=kind, text=text)


@pytest.mark.parametrize("text", ["ab cd", "ab", ""])
def test_a_none_or_a_branch_not_chosen_outside_nested_loops_decides_the_inner_and_the_outer_collect(text: str) -> None:
    none_outside, skipped = _nested_loops("join", "no-value", text), _nested_loops("join", "skip", text)
    assert (outcome_of(none_outside, "n"), outcome_of(skipped, "n")) == ("not called", "skipped")
    assert none_outside.results["j"]["result"] == [None] * len(text.split())  # each item's inner collect is not called: None in its place
    assert skipped.results["j"]["result"] is None  # no item can get a value when a branch outside is not chosen: one None for all
    assert [call.loc[0] for call in skipped.not_called] == []
    assert _nested_loops("join", "x", "ab cd").results["n"]["result"] == 2


def test_an_inner_collect_that_takes_none_is_called_and_the_outer_count_is_the_items() -> None:
    run = _nested_loops("join-or-none", "no-value", "ab cd")
    assert (run.results["j"]["result"], outcome_of(run, "n")) == (["none", "none"], 2)
    assert outcome_of(_nested_loops("join-or-none", "no-value", ""), "n") == 0


def test_a_series_input_whose_items_take_none_gets_a_none_per_item_for_a_none_outside() -> None:
    run = _nested_loops("join-taking-none", "no-value", "ab cd")
    assert (run.results["j"]["result"], outcome_of(run, "n")) == (["--", "--"], 2)
    assert outcome_of(_nested_loops("join-taking-none", "skip", "ab cd"), "n") == "skipped"
    nodes, edges = {"s": "items", "t": "outside-value", "p": "pair", "n": "dashes-for-none"}, [("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")]
    assert [outcome_of(ran(nodes, edges, kind="no-value", text=text), "n") for text in ("a b", "")] == ["--", ""]


def test_a_series_never_returned_is_none_for_the_whole_input_whatever_its_items_take() -> None:
    nodes = {"t": "outside-value", "s": "items", "u": "up", "n": "dashes-for-none", "o": "count-or-none"}
    run = ran(nodes, [("t.result", "s.text"), ("s.result", "u.a"), ("u.result", "n.values"), ("u.result", "o.values")], kind="no-value")
    assert (outcome_of(run, "n"), outcome_of(run, "o")) == ("not called", "none")


@pytest.mark.parametrize("text", ["a b", ""])
def test_a_branch_node_in_the_loop_not_called_for_a_none_outside_skips_the_collect(text: str) -> None:
    nodes, edges = {"s": "items", "t": "outside-value", "w": "decide-with"}, [("s.result", "w.go"), ("t.result", "w.extra")]
    direct = ran({**nodes, "n": "count"}, [*edges, ("w.yes", "n.values")], kind="no-value", text=text)
    through = ran({**nodes, "u": "up", "n": "count"}, [*edges, ("w.yes", "u.a"), ("u.result", "n.values")], kind="no-value", text=text)
    assert (outcome_of(direct, "n"), outcome_of(through, "n")) == ("skipped", "skipped")


def test_a_branch_node_not_called_for_one_items_none_leaves_that_item_out_of_the_collect() -> None:
    nodes = {"s": "items", "m": "maybe", "w": "decide", "n": "count"}
    run = ran(nodes, [("s.result", "m.a"), ("m.result", "w.go"), ("w.yes", "n.values")], text="y1 no-value y2")
    assert outcome_of(run, "n") == 2


def test_a_none_outside_beside_a_branch_chosen_per_item_leaves_the_collect_not_called() -> None:
    nodes = {"s": "items", "t": "outside-value", "w": "decide", "p": "pair", "n": "count"}
    edges = [("s.result", "w.go"), ("w.yes", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")]
    assert [outcome_of(ran(nodes, edges, kind="no-value", text=text), "n") for text in ("yes no", "")] == ["not called", "not called"]


def test_a_branch_not_chosen_outside_skips_an_inner_collect_also_for_an_item_whose_series_is_none() -> None:
    nodes = {"s": "items", "t": "outside-value", "m": "maybe", "c": "chars", "p": "pair", "j": "join-or-none", "n": "count"}
    edges = [("s.result", "m.a"), ("m.result", "c.a"), ("c.result", "p.a"), ("t.result", "p.b"), ("p.result", "j.values"), ("j.result", "n.values")]
    run = ran(nodes, edges, kind="skip", text="ab no-value")
    assert (run.results["j"]["result"], outcome_of(run, "n"), run.not_called[0].loc) == (None, "skipped", ("c", 1))
    assert [call.loc[0] for call in run.not_called] == ["c"]  # j is skipped for both items, also for the item whose list is None


OTHER_SERIES = (
    {"r": "items", "s2": "chars", "z": "items", "q": "maybe", "j": "with-all", "n": "count"},
    [("r.result", "s2.a"), ("z.result", "q.a"), ("s2.result", "j.a"), ("q.result", "j.values"), ("j.result", "n.values")],
)


@pytest.mark.parametrize("other", ["x y", "x skip", "no-value x", "skip skip", ""])
def test_a_node_in_a_loop_that_collects_another_loop_gets_it_as_one_value_from_outside(other: str) -> None:
    run = ran(*OTHER_SERIES, inputs={"words": "r.text", "other": "z.text"}, words="ab cd", other=other)
    collected_text = "".join(word.upper() for word in other.split() if word not in ("no-value", "skip"))
    assert run.results["j"]["result"] == [[f"{char}{collected_text}" for char in word] for word in ("ab", "cd")]
    assert run.results["n"]["result"] == [2, 2]


def test_a_node_in_a_loop_collecting_a_series_never_returned_is_not_called_nor_is_the_outer_collect() -> None:
    nodes, edges = OTHER_SERIES
    run = ran({**nodes, "t": "outside-value"}, [*edges, ("t.result", "z.text")], kind="no-value", text="ab cd")
    assert run.results["n"]["result"] == [None, None]
    assert [call.loc for call in run.not_called] == [("z",), ("j", 0, 0), ("j", 0, 1), ("j", 1, 0), ("j", 1, 1), ("n", 0), ("n", 1)]


@pytest.mark.parametrize(("first", "second", "text", "n"), [
    ("yes", "yes", "p q", 2), ("no", "no", "p", 1), ("yes", "no", "p q", "skipped"), ("yes", "no", "", "skipped"), ("no", "yes", "", "skipped"), ("yes", "yes", "", 0),
])
def test_two_alternatives_in_the_loop_skip_the_collect_only_when_both_are_on_a_branch_not_chosen(first: str, second: str, text: str, n: Any) -> None:
    # a needs e.yes and f.yes, b needs e.no and f.no; both are wired to p, which is called per item and collected
    nodes = {"s": "items", "e": "decide", "f": "decide", "a": "three", "b": "three", "p": "up", "n": "count"}
    edges = [
        ("s.result", "a.a"), ("e.yes", "a.b"), ("f.yes", "a.c"), ("s.result", "b.a"), ("e.no", "b.b"), ("f.no", "b.c"),
        ("a.result", "p.a"), ("b.result", "p.a"), ("p.result", "n.values"),
    ]
    assert outcome_of(ran(nodes, edges, inputs={"first": "e.go", "second": "f.go"}, first=first, second=second, text=text), "n") == n


@pytest.mark.parametrize(("ways", "n"), [(["yes"] * 40, 2), (["yes", "no"] * 20, 2), ([*["yes"] * 20, "skip", *["no"] * 19], "skipped")])
def test_forty_stages_of_alternatives_in_the_loop_are_decided_node_by_node_not_path_by_path(ways: list[str], n: Any) -> None:
    """Forty stages, each of two nodes on opposite branches of a branch node outside the loop, each wired to both
    nodes of the stage before: about a million million paths lead through, and eighty nodes are decided."""
    def sources(*places: str) -> dict[str, Any]:
        return {"sources": [dict(zip(("node", "output"), place.split("."), strict=True)) for place in places]}

    nodes: dict[str, Any] = {"s": {"node": "items", "inputs": {"text": {"constant": "p q"}}}}
    before = ["s.result"]
    for stage, way in enumerate(ways):
        nodes[f"t{stage}"] = {"node": "outside-value", "inputs": {"kind": {"constant": way}}}
        nodes[f"w{stage}"] = {"node": "decide", "inputs": {"go": sources(f"t{stage}.result")}}
        nodes[f"x{stage}"] = {"node": "pair", "inputs": {"a": sources(*before), "b": sources(f"w{stage}.yes")}}
        nodes[f"y{stage}"] = {"node": "pair", "inputs": {"a": sources(*before), "b": sources(f"w{stage}.no")}}
        before = [f"x{stage}.result", f"y{stage}.result"]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {**nodes, "n": {"node": "count", "inputs": {"values": sources(*before)}}}}), REGISTRY)
    assert not graph.problems, graph.problems
    assert outcome_of(graph.run(), "n") == n


@pytest.mark.parametrize("counter", ["count", "count-or-none", "dashes-for-none"])
def test_a_series_never_returned_into_a_branch_node_skips_the_collect_as_no_branch_was_chosen(counter: str) -> None:
    nodes, edges = {"t": "outside-value", "s": "items", "w": "decide"}, [("t.result", "s.text"), ("s.result", "w.go")]
    direct = ran({**nodes, "n": counter}, [*edges, ("w.no", "n.values")], kind="no-value")
    through = ran({**nodes, "u": "up", "n": counter}, [*edges, ("w.no", "u.a"), ("u.result", "n.values")], kind="no-value")
    assert (outcome_of(direct, "n"), outcome_of(through, "n")) == ("skipped", "skipped")


def test_a_series_never_returned_is_none_for_the_collect_even_where_the_loop_body_takes_none() -> None:
    nodes = {"t": "outside-value", "s": "items", "u": "up-or-none", "n": "count", "o": "count-or-none"}
    run = ran(nodes, [("t.result", "s.text"), ("s.result", "u.a"), ("u.result", "n.values"), ("u.result", "o.values")], kind="no-value")
    assert (outcome_of(run, "n"), outcome_of(run, "o")) == ("not called", "none")


def test_one_none_wired_to_a_series_input_is_a_list_of_one_none_where_its_items_take_none() -> None:
    nodes = {"t": "outside-value", "n": "count", "o": "count-or-none", "k": "dashes-for-none"}
    run = ran(nodes, [("t.result", "n.values"), ("t.result", "o.values"), ("t.result", "k.values")], kind="no-value")
    assert (outcome_of(run, "n"), outcome_of(run, "o"), outcome_of(run, "k")) == ("not called", "none", "-")
    assert outcome_of(ran({"t": "outside-value", "k": "dashes-for-none"}, [("t.result", "k.values")], kind="skip"), "k") == "skipped"


def test_a_list_of_one_none_from_outside_the_loop_is_taken_only_where_its_items_take_none() -> None:
    nodes, edges = {"s": "items", "t": "outside-value", "x": "with-all", "n": "count"}, [("s.result", "x.a"), ("t.result", "x.values"), ("x.result", "n.values")]
    assert outcome_of(ran(nodes, edges, kind="no-value", text="a b"), "n") == "not called"  # with-all's list of one takes no None
    assert outcome_of(ran({**nodes, "x": "with-all-taking-none"}, edges, kind="no-value", text="a b"), "n") == 2


@pytest.mark.parametrize(("kind", "counter", "n"), [
    ("no-value", "count", "not called"), ("no-value", "count-or-none", "none"), ("skip", "count", "skipped"), ("skip", "count-or-none", "skipped"),
    ("a b", "count", 2), ("", "count", 0),
])
def test_a_series_returned_or_not_inside_a_subgraph_is_collected_as_with_the_same_nodes_flat(kind: str, counter: str, n: Any) -> None:
    flat = ran({"t": "outside-value", "w": "items", "u": "up", "n": counter}, [("t.result", "w.text"), ("w.result", "u.a"), ("u.result", "n.values")], kind=kind)
    in_subgraph = ran({"t": "outside-value", "e": "items-then-up", "n": counter}, [("t.result", "e.text"), ("e.result", "n.values")], kind=kind)
    assert (outcome_of(flat, "n"), outcome_of(in_subgraph, "n")) == (n, n)


@pytest.mark.parametrize("text", ["a b", ""])
def test_a_collect_reading_only_the_series_counts_its_items_and_one_reading_a_none_beside_it_is_not_called(text: str) -> None:
    """When the loop body reads nothing from outside the loop but the series, the collecting node's outcome is
    decided by the series alone. A second output of the node that returns the series is another value: when it is
    ``None``, the collecting node that reads it is not called."""
    nodes = {"x": "say", "t": "outside-value", "l": "items-and-note", "p": "pair", "u": "up", "n": "count", "m": "count"}
    edges = [("x.result", "l.text"), ("t.result", "l.note"), ("l.items", "u.a"), ("u.result", "m.values"), ("l.items", "p.a"), ("l.note", "p.b"), ("p.result", "n.values")]
    run = ran(nodes, edges, kind="no-value", text=text)
    assert (outcome_of(run, "m"), outcome_of(run, "n")) == (len(text.split()), "not called")
    assert outcome_of(ran(nodes, edges, kind="x", text=text), "n") == len(text.split())


@pytest.mark.parametrize(("kind", "n"), [("no-value", "not called"), ("skip", "skipped"), ("v", 2)])
def test_a_subgraph_in_the_loop_whose_series_depends_on_outside_decides_the_outer_collect_as_flat(kind: str, n: Any) -> None:
    flat = ran(
        {"s": "items", "t": "outside-value", "p": "pair", "w": "items", "u": "up", "j": "join", "n": "count"},
        [("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "w.text"), ("w.result", "u.a"), ("u.result", "j.values"), ("j.result", "n.values")],
        kind=kind, text="a b",
    )
    in_subgraph = ran(
        {"s": "items", "t": "outside-value", "e": "pair-then-items-then-up", "j": "join", "n": "count"},
        [("s.result", "e.a"), ("t.result", "e.b"), ("e.result", "j.values"), ("j.result", "n.values")],
        kind=kind, text="a b",
    )
    assert (outcome_of(flat, "n"), outcome_of(in_subgraph, "n")) == (n, n)


@_node("both")
def both(a: Series[str], b: Series[str]) -> str:
    return f"{'|'.join(a)}:{''.join(b)}"


REGISTRY.subgraph("summary-and-letters", title="Summary and letters", description="A text in capitals beside its letters in capitals.")(
    make_graph(up, chars, maybe, both, edges=[("chars.result", "maybe.a"), ("up.result", "both.a"), ("maybe.result", "both.b")], inputs={"text": ["up.a", "chars.a"]})
)


def test_a_series_input_wired_to_a_per_item_value_gets_that_items_value_flat_or_in_a_subgraph() -> None:
    """A document's summary beside the document's letters: the node is called once per document, for the letters it
    collects, so its other ``Series`` input, wired to the summary, gets that document's summary, never every one."""
    flat = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "items", "inputs": {"text": {"graph_input": "text"}}}, "u": {"node": "up", "inputs": {"a": {"sources": [{"node": "s", "output": "result"}]}}}, "c": {"node": "chars", "inputs": {"a": {"sources": [{"node": "s", "output": "result"}]}}}, "x": {"node": "maybe", "inputs": {"a": {"sources": [{"node": "c", "output": "result"}]}}}, "j": {"node": "both", "inputs": {"a": {"sources": [{"node": "u", "output": "result"}]}, "b": {"sources": [{"node": "x", "output": "result"}]}}}}}), REGISTRY)
    in_subgraph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "items", "inputs": {"text": {"graph_input": "text"}}}, "e": {"node": "summary-and-letters", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}}}), REGISTRY)
    assert not flat.problems and not in_subgraph.problems
    feed = {field.field.name: (field.source.mode, field.source.series) for field in flat.describe().nodes["j"].inputs}
    assert feed == {"a": ("collected", None), "b": ("collected", "c.result")}
    assert flat.run(text="ab c").results["j"]["result"] == in_subgraph.run(text="ab c").results["e"]["result"] == ["AB:AB", "C:C"]



def shown(nodes: dict[str, str], edges: list[tuple[str, str]], node_id: str, output: str = "result", **inputs: str) -> tuple[Any, Any, Any]:
    """An output in ``run.results``, in ``run.outputs``, and in the results of the run read back with ``Run.from_json``."""
    run = ran(nodes, edges, **inputs)
    return run.results[node_id][output], run.outputs.get(f"{node_id}-{output}", "not a graph output"), Run.from_json(run.to_json(), run.graph).results[node_id][output]


@pytest.mark.parametrize(("kind", "result"), [("no-value", None), ("skip", None), ("", []), ("a b", ["A", "B"])])
def test_a_per_item_output_is_shown_as_a_collect_would_get_it_flat_or_in_a_subgraph(kind: str, result: Any) -> None:
    flat = shown({"t": "outside-value", "w": "items", "u": "up"}, [("t.result", "w.text"), ("w.result", "u.a")], "u", kind=kind)
    in_subgraph = shown({"t": "outside-value", "e": "items-then-up"}, [("t.result", "e.text")], "e", kind=kind)
    assert flat == in_subgraph == (result, result, result)


@pytest.mark.parametrize(("kind", "result"), [("no-value", [None, None]), ("skip", None), ("x", ["a+x", "b+x"])])
def test_a_per_item_output_decided_from_outside_is_none_per_item_or_one_none_when_skipped(kind: str, result: Any) -> None:
    nodes, edges = {"s": "items", "t": "outside-value", "p": "pair", "n": "count"}, [("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")]
    assert shown(nodes, edges, "p", kind=kind, text="a b") == (result, "not a graph output", result)


def test_an_output_of_nested_loops_is_a_list_per_outer_item_or_none_when_never_returned() -> None:
    nodes, edges = {"t": "outside-value", "s": "items", "c": "chars", "u": "up"}, [("t.result", "s.text"), ("s.result", "c.a"), ("c.result", "u.a")]
    assert shown(nodes, edges, "u", kind="ab c")[0] == [["A", "B"], ["C"]]
    assert shown(nodes, edges, "u", kind="no-value") == (None, None, None)


@_node("approve", asks=True)
def approve(a: str) -> str:
    raise Ask("ok?")


def test_a_waiting_runs_results_have_none_for_each_call_still_waiting_also_when_read_back() -> None:
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "items", "inputs": {"text": {"graph_input": "text"}}}, "q": {"node": "approve", "inputs": {"a": {"sources": [{"node": "s", "output": "result"}]}}}}}), REGISTRY)
    waiting = graph.run(text="a b")
    assert waiting.results["q"]["result"] == [None, None]
    one = waiting.answer(waiting.waiting[0].id, "yes").resume()
    assert one.status == "waiting"
    assert one.results["q"]["result"] == ["yes", None] == Run.from_json(one.to_json(), graph).results["q"]["result"]


def test_a_subgraph_in_a_loop_shows_a_list_per_item_and_none_where_its_series_was_never_returned() -> None:
    flat = ran({"s": "items", "m": "maybe", "w": "items", "u": "up"}, [("s.result", "m.a"), ("m.result", "w.text"), ("w.result", "u.a")], text="ab no-value cd")
    in_subgraph = ran({"s": "items", "m": "maybe", "e": "items-then-up"}, [("s.result", "m.a"), ("m.result", "e.text")], text="ab no-value cd")
    assert flat.results["u"]["result"] == [["AB"], None, ["CD"]]  # a list for each item and None for the item whose series was never returned
    assert in_subgraph.results["e"]["result"] == [["AB"], None, ["CD"]] == in_subgraph.outputs["e-result"]
    never = ran({"t": "outside-value", "s": "items", "e": "items-then-up"}, [("t.result", "s.text"), ("s.result", "e.text")], kind="no-value")
    assert never.results["e"]["result"] is None and never.outputs["e-result"] is None and never.outputs_to_json() == '{"e-result":null}'


REGISTRY.subgraph("words-up-and-letters", title="Words and letters", description="Each word in capitals, and each letter of each word.")(
    make_graph(items, up, chars, maybe, edges=[("items.result", "up.a"), ("items.result", "chars.a"), ("chars.result", "maybe.a")], outputs={"words": "up.result", "letters": "maybe.result"})
)


def test_two_series_inputs_wired_to_a_subgraphs_per_word_and_per_letter_outputs_are_called_per_word() -> None:
    """The letters are a list per word, so the node is called once per word with that word's letters, and the
    other input gets that word's one value as a list of one."""
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "words-up-and-letters", "inputs": {"text": {"graph_input": "text"}}}, "n": {"node": "both", "inputs": {"a": {"sources": [{"node": "e", "output": "words"}]}, "b": {"sources": [{"node": "e", "output": "letters"}]}}}}}), REGISTRY)
    assert not graph.problems
    assert graph.run(text="ab cd").results["n"]["result"] == ["AB:AB", "CD:CD"]
