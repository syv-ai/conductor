"""Tests what happens to a node that collects a loop when a value from outside the loop is missing.

A value can be missing in two ways. It can be ``None`` where the input doesn't take ``None``, or it can sit on a
skipped branch.

Take the graph ``items -> pair -> count``. ``items`` returns ``["a", "b"]``. ``pair`` is called once per item, and
its input ``b`` is wired to ``outside-value``, a node outside the loop. ``count`` collects every ``pair.result``.

- If ``outside-value`` returns ``None``, and ``pair.b`` doesn't take ``None``, no call of ``pair`` can return a
  value. So ``count`` is not called. That holds for two items, and for none.
- If ``outside-value`` is on a skipped branch, ``count`` is skipped.
- Otherwise each item has its own say, and a loop with no items is collected as an empty list.

So the outside value decides the collecting node only when it decides every item the same way.

The sample nodes follow a few patterns:

- ``outside-value`` returns its input ``kind``, except ``None`` for ``"no-value"`` and ``SKIPPED`` for ``"skip"``.
  ``maybe`` does the same for each item.
- A node named ``...-or-none`` takes ``None`` for its whole input. A node named ``...-taking-none`` takes a list
  whose items may be ``None``.
- A Series is "never returned" when the node that would return it was not called or was skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
    NodeRegistry,
    Run,
    Series,
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
    if kind == "skip":
        return SKIPPED
    if kind == "no-value":
        return None
    return kind


@dataclass(frozen=True)
class Way:
    yes: str = SKIPPED
    no: str = SKIPPED


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
    if a == "no-value":
        return None
    if a == "skip":
        return SKIPPED
    return a.upper()


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


REGISTRY.subgraph(
    "pair-then-up-or-none", title="Pair then up", description="A pair, in capitals, or none."
)(make_graph(pair, up_or_none, edges=[("pair.result", "up-or-none.a")]))


REGISTRY.subgraph(
    "items-then-up", title="Items then up", description="Each word of a text, in capitals."
)(make_graph(items, up, edges=[("items.result", "up.a")]))


REGISTRY.subgraph(
    "pair-then-items-then-up",
    title="Pair then items then up",
    description="Each word of a pair, in capitals.",
)(make_graph(pair, items, up, edges=[("pair.result", "items.text"), ("items.result", "up.a")]))


def ran(
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    inputs: Mapping[str, str | Sequence[str]] | None = None,
    outputs: Mapping[str, str] | None = None,
    **values: str,
) -> Any:
    """Builds a graph with ``make_graph``, runs it with ``values``, and returns the run, which must be complete.

    Args:
        nodes: The definition id in ``REGISTRY``, by node id.
        edges: Every edge of the graph. Given ``edges``, even an empty list, ``make_graph`` adds no chain.
        inputs: Graph inputs to name, passed on to ``make_graph``.
        outputs: Graph outputs to name, passed on to ``make_graph``.
        values: The value of each graph input.
    """
    record = make_graph(
        *((node_id, REGISTRY[registry_id]) for node_id, registry_id in nodes.items()),
        edges=edges,
        inputs=inputs or {},
        outputs=outputs or {},
    )
    graph = Graph.compile(record, REGISTRY)
    assert not graph.problems, graph.problems
    run = graph.run(values)
    assert run.status == "complete"
    return run


def outcome_of(run: Any, node_id: str) -> Any:
    """Returns what happened to a node that isn't in a loop.

    That is its result if it has one. Otherwise it is ``"none input"``, when an argument was ``None`` and its input
    doesn't take ``None``, or ``"skipped"``.
    """
    result = run.results[node_id]["result"]
    if result is not None:
        return result
    return (
        "none input" if any(call.loc == (node_id,) for call in run.none_input_calls) else "skipped"
    )


BOTH_WAYS = (
    {
        "s": "items",
        "w": "decide",
        "t": "outside-value",
        "y": "three",
        "z": "pair",
        "p": "up",
        "n": "count",
    },
    [
        ("s.result", "y.a"),
        ("w.yes", "y.b"),
        ("t.result", "y.c"),
        ("s.result", "z.a"),
        ("w.no", "z.b"),
        ("y.result", "p.a"),
        ("z.result", "p.a"),
        ("p.result", "n.values"),
    ],
)


@pytest.mark.parametrize(
    ("go", "kind", "text", "n"),
    [
        ("no", "no-value", "a b", 2),
        ("no", "no-value", "", 0),
        ("no", "x", "a b", 2),
        ("yes", "no-value", "a b", "none input"),
        ("yes", "no-value", "", "none input"),
        ("yes", "x", "a", 1),
    ],
)
def test_a_none_outside_the_loop_on_the_skipped_branch_does_not_decide_the_collecting_node(
    go: str, kind: str, text: str, n: Any
) -> None:
    assert outcome_of(ran(*BOTH_WAYS, go=go, kind=kind, text=text), "n") == n


@pytest.mark.parametrize(
    ("kind", "text", "n"),
    [
        ("no-value", "a b", 2),
        ("no-value", "", 0),
        ("x", "a b", 2),
        ("skip", "a b", "skipped"),
        ("skip", "", "skipped"),
    ],
)
def test_a_subgraph_in_the_loop_that_takes_none_from_outside_gives_each_item_a_value(
    kind: str, text: str, n: Any
) -> None:
    run = ran(
        {"s": "items", "t": "outside-value", "e": "pair-then-up-or-none", "n": "count"},
        [("s.result", "e.a"), ("t.result", "e.b"), ("e.result", "n.values")],
        kind=kind,
        text=text,
    )
    assert outcome_of(run, "n") == n


def _nested_loops(joiner: str, kind: str, text: str) -> Any:
    nodes = {
        "s": "items",
        "t": "outside-value",
        "c": "chars",
        "m": "pair",
        "j": joiner,
        "n": "count",
    }
    return ran(
        nodes,
        [
            ("s.result", "c.a"),
            ("c.result", "m.a"),
            ("t.result", "m.b"),
            ("m.result", "j.values"),
            ("j.result", "n.values"),
        ],
        kind=kind,
        text=text,
    )


@pytest.mark.parametrize("text", ["ab cd", "ab", ""])
def test_a_none_or_a_skipped_branch_outside_nested_loops_decides_the_inner_and_the_outer_collect(
    text: str,
) -> None:
    none_outside, skipped = (
        _nested_loops("join", "no-value", text),
        _nested_loops("join", "skip", text),
    )
    assert (outcome_of(none_outside, "n"), outcome_of(skipped, "n")) == ("none input", "skipped")
    assert none_outside.results["j"]["result"] == [None] * len(
        text.split()
    )  # For each item, the inner collect is not called, so there is a None in its place.
    # With the branch outside skipped, no item can get a value, so there is one None for the whole loop.
    assert skipped.results["j"]["result"] is None
    assert [call.loc[0] for call in skipped.none_input_calls] == []
    assert _nested_loops("join", "x", "ab cd").results["n"]["result"] == 2


def test_an_inner_collect_that_takes_none_is_called_and_the_outer_count_is_the_items() -> None:
    run = _nested_loops("join-or-none", "no-value", "ab cd")
    assert (run.results["j"]["result"], outcome_of(run, "n")) == (["none", "none"], 2)
    assert outcome_of(_nested_loops("join-or-none", "no-value", ""), "n") == 0


def test_a_series_input_whose_items_take_none_gets_a_none_per_item_for_a_none_outside() -> None:
    run = _nested_loops("join-taking-none", "no-value", "ab cd")
    assert (run.results["j"]["result"], outcome_of(run, "n")) == (["--", "--"], 2)
    assert outcome_of(_nested_loops("join-taking-none", "skip", "ab cd"), "n") == "skipped"
    nodes, edges = (
        {"s": "items", "t": "outside-value", "p": "pair", "n": "dashes-for-none"},
        [("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")],
    )
    assert [
        outcome_of(ran(nodes, edges, kind="no-value", text=text), "n") for text in ("a b", "")
    ] == ["--", ""]


def test_a_series_never_returned_is_none_for_the_whole_input_whatever_its_items_take() -> None:
    nodes = {
        "t": "outside-value",
        "s": "items",
        "u": "up",
        "n": "dashes-for-none",
        "o": "count-or-none",
    }
    run = ran(
        nodes,
        [
            ("t.result", "s.text"),
            ("s.result", "u.a"),
            ("u.result", "n.values"),
            ("u.result", "o.values"),
        ],
        outputs={"n": "n.result", "o": "o.result"},
        kind="no-value",
    )
    assert (outcome_of(run, "n"), outcome_of(run, "o")) == ("none input", "none")


@pytest.mark.parametrize("text", ["a b", ""])
def test_a_branch_node_in_the_loop_with_a_none_input_from_outside_skips_the_collect(
    text: str,
) -> None:
    nodes, edges = (
        {"s": "items", "t": "outside-value", "w": "decide-with"},
        [("s.result", "w.go"), ("t.result", "w.extra")],
    )
    direct = ran(
        {**nodes, "n": "count"}, [*edges, ("w.yes", "n.values")], kind="no-value", text=text
    )
    through = ran(
        {**nodes, "u": "up", "n": "count"},
        [*edges, ("w.yes", "u.a"), ("u.result", "n.values")],
        kind="no-value",
        text=text,
    )
    assert (outcome_of(direct, "n"), outcome_of(through, "n")) == ("skipped", "skipped")


def test_a_branch_node_with_a_none_input_for_one_item_leaves_that_item_out_of_the_collect() -> None:
    nodes = {"s": "items", "m": "maybe", "w": "decide", "n": "count"}
    run = ran(
        nodes,
        [("s.result", "m.a"), ("m.result", "w.go"), ("w.yes", "n.values")],
        text="y1 no-value y2",
    )
    assert outcome_of(run, "n") == 2


def test_a_none_outside_beside_a_branch_taken_per_item_makes_the_collect_a_none_input() -> None:
    nodes = {"s": "items", "t": "outside-value", "w": "decide", "p": "pair", "n": "count"}
    edges = [("s.result", "w.go"), ("w.yes", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")]
    assert [
        outcome_of(ran(nodes, edges, kind="no-value", text=text), "n") for text in ("yes no", "")
    ] == ["none input", "none input"]


def test_a_skipped_branch_outside_skips_an_inner_collect_also_for_an_item_whose_series_is_none() -> (
    None
):
    nodes = {
        "s": "items",
        "t": "outside-value",
        "m": "maybe",
        "c": "chars",
        "p": "pair",
        "j": "join-or-none",
        "n": "count",
    }
    edges = [
        ("s.result", "m.a"),
        ("m.result", "c.a"),
        ("c.result", "p.a"),
        ("t.result", "p.b"),
        ("p.result", "j.values"),
        ("j.result", "n.values"),
    ]
    run = ran(nodes, edges, kind="skip", text="ab no-value")
    assert (run.results["j"]["result"], outcome_of(run, "n"), run.none_input_calls[0].loc) == (
        None,
        "skipped",
        ("c", 1),
    )
    assert [call.loc[0] for call in run.none_input_calls] == [
        "c"
    ]  # j is skipped for both items, even the one whose list is None.


OTHER_SERIES = (
    {"r": "items", "s2": "chars", "z": "items", "q": "maybe", "j": "with-all", "n": "count"},
    [
        ("r.result", "s2.a"),
        ("z.result", "q.a"),
        ("s2.result", "j.a"),
        ("q.result", "j.values"),
        ("j.result", "n.values"),
    ],
)


@pytest.mark.parametrize("other", ["x y", "x skip", "no-value x", "skip skip", ""])
def test_a_node_in_a_loop_that_collects_another_loop_gets_it_as_one_value_from_outside(
    other: str,
) -> None:
    run = ran(
        *OTHER_SERIES, inputs={"words": "r.text", "other": "z.text"}, words="ab cd", other=other
    )
    collected_text = "".join(
        word.upper() for word in other.split() if word not in ("no-value", "skip")
    )
    assert run.results["j"]["result"] == [
        [f"{char}{collected_text}" for char in word] for word in ("ab", "cd")
    ]
    assert run.results["n"]["result"] == [2, 2]


def test_a_node_in_a_loop_collecting_a_series_never_returned_is_a_none_input_and_so_is_the_outer_collect() -> (
    None
):
    nodes, edges = OTHER_SERIES
    run = ran(
        {**nodes, "t": "outside-value"},
        [*edges, ("t.result", "z.text")],
        kind="no-value",
        text="ab cd",
    )
    assert run.results["n"]["result"] == [None, None]
    assert [call.loc for call in run.none_input_calls] == [
        ("z",),
        ("j", 0, 0),
        ("j", 0, 1),
        ("j", 1, 0),
        ("j", 1, 1),
        ("n", 0),
        ("n", 1),
    ]


@pytest.mark.parametrize(
    ("first", "second", "text", "n"),
    [
        ("yes", "yes", "p q", 2),
        ("no", "no", "p", 1),
        ("yes", "no", "p q", "skipped"),
        ("yes", "no", "", "skipped"),
        ("no", "yes", "", "skipped"),
        ("yes", "yes", "", 0),
    ],
)
def test_two_alternatives_in_the_loop_skip_the_collect_only_when_both_are_on_a_skipped_branch(
    first: str, second: str, text: str, n: Any
) -> None:
    # a needs e.yes and f.yes, and b needs e.no and f.no. Both feed p, which is called per item and then collected.
    nodes = {
        "s": "items",
        "e": "decide",
        "f": "decide",
        "a": "three",
        "b": "three",
        "p": "up",
        "n": "count",
    }
    edges = [
        ("s.result", "a.a"),
        ("e.yes", "a.b"),
        ("f.yes", "a.c"),
        ("s.result", "b.a"),
        ("e.no", "b.b"),
        ("f.no", "b.c"),
        ("a.result", "p.a"),
        ("b.result", "p.a"),
        ("p.result", "n.values"),
    ]
    assert (
        outcome_of(
            ran(
                nodes,
                edges,
                inputs={"first": "e.go", "second": "f.go", "text": "s.text"},
                first=first,
                second=second,
                text=text,
            ),
            "n",
        )
        == n
    )


@pytest.mark.parametrize(
    ("ways", "n"),
    [
        (["yes"] * 40, 2),
        (["yes", "no"] * 20, 2),
        ([*["yes"] * 20, "skip", *["no"] * 19], "skipped"),
    ],
)
def test_forty_stages_of_alternatives_in_the_loop_are_decided_node_by_node_not_path_by_path(
    ways: list[str], n: Any
) -> None:
    """Forty stages of alternatives are worked out one node at a time, not one path at a time.

    Each stage is two nodes, one on each branch of a branch node outside the loop. Each is wired to both nodes of
    the stage before. That gives about a million million paths through the graph, but only eighty nodes to decide.
    """

    def sources(*places: str) -> dict[str, Any]:
        return {
            "sources": [
                dict(zip(("node", "output"), place.split("."), strict=True)) for place in places
            ]
        }

    nodes: dict[str, Any] = {"s": {"node": "items", "inputs": {"text": {"constant": "p q"}}}}
    before = ["s.result"]
    for stage, way in enumerate(ways):
        nodes[f"t{stage}"] = {"node": "outside-value", "inputs": {"kind": {"constant": way}}}
        nodes[f"w{stage}"] = {"node": "decide", "inputs": {"go": sources(f"t{stage}.result")}}
        nodes[f"x{stage}"] = {
            "node": "pair",
            "inputs": {"a": sources(*before), "b": sources(f"w{stage}.yes")},
        }
        nodes[f"y{stage}"] = {
            "node": "pair",
            "inputs": {"a": sources(*before), "b": sources(f"w{stage}.no")},
        }
        before = [f"x{stage}.result", f"y{stage}.result"]
    graph = Graph.compile(
        GraphRecord.model_validate(
            {"nodes": {**nodes, "n": {"node": "count", "inputs": {"values": sources(*before)}}}}
        ),
        REGISTRY,
    )
    assert not graph.problems, graph.problems
    assert outcome_of(graph.run(), "n") == n


@pytest.mark.parametrize("counter", ["count", "count-or-none", "dashes-for-none"])
def test_a_series_never_returned_into_a_branch_node_skips_the_collect_as_no_branch_was_taken(
    counter: str,
) -> None:
    nodes, edges = (
        {"t": "outside-value", "s": "items", "w": "decide"},
        [("t.result", "s.text"), ("s.result", "w.go")],
    )
    direct = ran({**nodes, "n": counter}, [*edges, ("w.no", "n.values")], kind="no-value")
    through = ran(
        {**nodes, "u": "up", "n": counter},
        [*edges, ("w.no", "u.a"), ("u.result", "n.values")],
        kind="no-value",
    )
    assert (outcome_of(direct, "n"), outcome_of(through, "n")) == ("skipped", "skipped")


def test_a_series_never_returned_is_none_for_the_collect_even_where_the_loop_body_takes_none() -> (
    None
):
    nodes = {
        "t": "outside-value",
        "s": "items",
        "u": "up-or-none",
        "n": "count",
        "o": "count-or-none",
    }
    run = ran(
        nodes,
        [
            ("t.result", "s.text"),
            ("s.result", "u.a"),
            ("u.result", "n.values"),
            ("u.result", "o.values"),
        ],
        outputs={"n": "n.result", "o": "o.result"},
        kind="no-value",
    )
    assert (outcome_of(run, "n"), outcome_of(run, "o")) == ("none input", "none")


@pytest.mark.parametrize(
    ("kind", "counter", "n"),
    [
        ("no-value", "count", "none input"),
        ("no-value", "count-or-none", "none"),
        ("skip", "count", "skipped"),
        ("skip", "count-or-none", "skipped"),
        ("a b", "count", 2),
        ("", "count", 0),
    ],
)
def test_a_series_returned_or_not_inside_a_subgraph_is_collected_as_with_the_same_nodes_flat(
    kind: str, counter: str, n: Any
) -> None:
    flat = ran(
        {"t": "outside-value", "w": "items", "u": "up", "n": counter},
        [("t.result", "w.text"), ("w.result", "u.a"), ("u.result", "n.values")],
        kind=kind,
    )
    in_subgraph = ran(
        {"t": "outside-value", "e": "items-then-up", "n": counter},
        [("t.result", "e.text"), ("e.result", "n.values")],
        kind=kind,
    )
    assert (outcome_of(flat, "n"), outcome_of(in_subgraph, "n")) == (n, n)


@pytest.mark.parametrize("text", ["a b", ""])
def test_a_collect_reading_only_the_series_counts_its_items_and_one_reading_a_none_beside_it_is_a_none_input(
    text: str,
) -> None:
    """A collect whose loop reads only the Series counts its items, whatever else is ``None``.

    When the nodes in the loop read nothing from outside it but the Series, the Series alone decides the collecting
    node. The node that returns the Series also returns a second output, ``note``. That is a separate value. When
    it is ``None``, a collecting node whose loop reads it is not called.
    """
    nodes = {
        "x": "say",
        "t": "outside-value",
        "l": "items-and-note",
        "p": "pair",
        "u": "up",
        "n": "count",
        "m": "count",
    }
    edges = [
        ("x.result", "l.text"),
        ("t.result", "l.note"),
        ("l.items", "u.a"),
        ("u.result", "m.values"),
        ("l.items", "p.a"),
        ("l.note", "p.b"),
        ("p.result", "n.values"),
    ]
    outputs = {"m": "m.result", "n": "n.result"}
    run = ran(nodes, edges, outputs=outputs, kind="no-value", text=text)
    assert (outcome_of(run, "m"), outcome_of(run, "n")) == (len(text.split()), "none input")
    assert outcome_of(ran(nodes, edges, outputs=outputs, kind="x", text=text), "n") == len(
        text.split()
    )


@pytest.mark.parametrize(("kind", "n"), [("no-value", "none input"), ("skip", "skipped"), ("v", 2)])
def test_a_subgraph_in_the_loop_whose_series_depends_on_outside_decides_the_outer_collect_as_flat(
    kind: str, n: Any
) -> None:
    flat = ran(
        {
            "s": "items",
            "t": "outside-value",
            "p": "pair",
            "w": "items",
            "u": "up",
            "j": "join",
            "n": "count",
        },
        [
            ("s.result", "p.a"),
            ("t.result", "p.b"),
            ("p.result", "w.text"),
            ("w.result", "u.a"),
            ("u.result", "j.values"),
            ("j.result", "n.values"),
        ],
        kind=kind,
        text="a b",
    )
    in_subgraph = ran(
        {
            "s": "items",
            "t": "outside-value",
            "e": "pair-then-items-then-up",
            "j": "join",
            "n": "count",
        },
        [
            ("s.result", "e.a"),
            ("t.result", "e.b"),
            ("e.result", "j.values"),
            ("j.result", "n.values"),
        ],
        kind=kind,
        text="a b",
    )
    assert (outcome_of(flat, "n"), outcome_of(in_subgraph, "n")) == (n, n)


@_node("both")
def both(a: str, b: Series[str]) -> str:
    return f"{a}:{''.join(b)}"


REGISTRY.subgraph(
    "summary-and-letters",
    title="Summary and letters",
    description="A text in capitals beside its letters in capitals.",
)(
    make_graph(
        up,
        chars,
        maybe,
        both,
        edges=[("chars.result", "maybe.a"), ("up.result", "both.a"), ("maybe.result", "both.b")],
        inputs={"text": ["up.a", "chars.a"]},
    )
)


def test_an_input_of_one_value_beside_a_collecting_input_gets_its_items_value_flat_or_in_a_subgraph() -> (
    None
):
    """A node that collects each document's letters gets that document's summary beside them.

    The node is called once per document, since it collects the letters of one document at a time. So its other
    input, wired to the summary, gets only that document's summary, never all of them.
    """
    flat = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "items", "inputs": {"text": {"graph_input": "text"}}},
                    "u": {
                        "node": "up",
                        "inputs": {"a": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "c": {
                        "node": "chars",
                        "inputs": {"a": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "x": {
                        "node": "maybe",
                        "inputs": {"a": {"sources": [{"node": "c", "output": "result"}]}},
                    },
                    "j": {
                        "node": "both",
                        "inputs": {
                            "a": {"sources": [{"node": "u", "output": "result"}]},
                            "b": {"sources": [{"node": "x", "output": "result"}]},
                        },
                    },
                }
            }
        ),
        REGISTRY,
    )
    in_subgraph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "items", "inputs": {"text": {"graph_input": "text"}}},
                    "e": {
                        "node": "summary-and-letters",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                }
            }
        ),
        REGISTRY,
    )
    assert not flat.problems and not in_subgraph.problems
    feed = {
        field.field.name: (field.source.mode, field.source.series)
        for field in flat.describe().nodes["j"].inputs
    }
    assert feed == {"a": ("item", "s.result"), "b": ("collected", "c.result")}
    assert (
        flat.run({"text": "ab c"}).results["j"]["result"]
        == in_subgraph.run({"text": "ab c"}).results["e"]["result"]
        == ["AB:AB", "C:C"]
    )


def shown(
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    node_id: str,
    output: str = "result",
    **inputs: str,
) -> tuple[Any, Any, Any]:
    """Returns one output three ways.

    They are ``run.results``, ``run.outputs``, and the results of the run after it is saved and loaded again.
    """
    run = ran(nodes, edges, outputs={"shown": f"{node_id}.{output}"}, **inputs)
    return (
        run.results[node_id][output],
        run.outputs["shown"],
        Run.from_json(run.to_json(), run.graph).results[node_id][output],
    )


@pytest.mark.parametrize(
    ("kind", "result"), [("no-value", None), ("skip", None), ("", []), ("a b", ["A", "B"])]
)
def test_a_per_item_output_is_shown_as_a_collect_would_get_it_flat_or_in_a_subgraph(
    kind: str, result: Any
) -> None:
    flat = shown(
        {"t": "outside-value", "w": "items", "u": "up"},
        [("t.result", "w.text"), ("w.result", "u.a")],
        "u",
        kind=kind,
    )
    in_subgraph = shown(
        {"t": "outside-value", "e": "items-then-up"}, [("t.result", "e.text")], "e", kind=kind
    )
    assert flat == in_subgraph == (result, result, result)


@pytest.mark.parametrize(
    ("kind", "result"), [("no-value", [None, None]), ("skip", None), ("x", ["a+x", "b+x"])]
)
def test_a_per_item_output_decided_from_outside_is_none_per_item_or_one_none_when_skipped(
    kind: str, result: Any
) -> None:
    nodes, edges = (
        {"s": "items", "t": "outside-value", "p": "pair", "n": "count"},
        [("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")],
    )
    assert shown(nodes, edges, "p", kind=kind, text="a b") == (result, result, result)


def test_an_output_of_nested_loops_is_a_list_per_outer_item_or_none_when_never_returned() -> None:
    nodes, edges = (
        {"t": "outside-value", "s": "items", "c": "chars", "u": "up"},
        [("t.result", "s.text"), ("s.result", "c.a"), ("c.result", "u.a")],
    )
    assert shown(nodes, edges, "u", kind="ab c")[0] == [["A", "B"], ["C"]]
    assert shown(nodes, edges, "u", kind="no-value") == (None, None, None)


@_node("approve", asks=True)
def approve(a: str) -> str:
    raise Ask("ok?")


def test_a_waiting_runs_results_have_none_for_each_call_still_waiting_also_when_read_back() -> None:
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "items", "inputs": {"text": {"graph_input": "text"}}},
                    "q": {
                        "node": "approve",
                        "inputs": {"a": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                }
            }
        ),
        REGISTRY,
    )
    waiting = graph.run({"text": "a b"})
    assert waiting.results["q"]["result"] == [None, None]
    one = waiting.answer(waiting.waiting[0].id, "yes").resume()
    assert one.status == "waiting"
    assert (
        one.results["q"]["result"]
        == ["yes", None]
        == Run.from_json(one.to_json(), graph).results["q"]["result"]
    )


def test_a_subgraph_in_a_loop_shows_a_list_per_item_and_none_where_its_series_was_never_returned() -> (
    None
):
    flat = ran(
        {"s": "items", "m": "maybe", "w": "items", "u": "up"},
        [("s.result", "m.a"), ("m.result", "w.text"), ("w.result", "u.a")],
        text="ab no-value cd",
    )
    in_subgraph = ran(
        {"s": "items", "m": "maybe", "e": "items-then-up"},
        [("s.result", "m.a"), ("m.result", "e.text")],
        text="ab no-value cd",
    )
    assert flat.results["u"]["result"] == [
        ["AB"],
        None,
        ["CD"],
    ]  # A list for each item, and None for the item whose Series was never returned.
    assert (
        in_subgraph.results["e"]["result"]
        == [["AB"], None, ["CD"]]
        == in_subgraph.outputs["result"]
    )
    never = ran(
        {"t": "outside-value", "s": "items", "e": "items-then-up"},
        [("t.result", "s.text"), ("s.result", "e.text")],
        kind="no-value",
    )
    assert (
        never.results["e"]["result"] is None
        and never.outputs["result"] is None
        and never.outputs_to_json() == '{"result":null}'
    )


REGISTRY.subgraph(
    "words-up-and-letters",
    title="Words and letters",
    description="Each word in capitals, and each letter of each word.",
)(
    make_graph(
        items,
        up,
        chars,
        maybe,
        edges=[("items.result", "up.a"), ("items.result", "chars.a"), ("chars.result", "maybe.a")],
        outputs={"words": "up.result", "letters": "maybe.result"},
    )
)


def test_a_node_wired_to_a_subgraphs_per_word_and_per_letter_outputs_is_called_per_word() -> None:
    """The letters come as one list per word, so the node is called once per word.

    Each call gets that word's letters. Its other input, which takes one value, gets that word's.
    """
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "e": {
                        "node": "words-up-and-letters",
                        "inputs": {"text": {"graph_input": "text"}},
                    },
                    "n": {
                        "node": "both",
                        "inputs": {
                            "a": {"sources": [{"node": "e", "output": "words"}]},
                            "b": {"sources": [{"node": "e", "output": "letters"}]},
                        },
                    },
                }
            }
        ),
        REGISTRY,
    )
    assert not graph.problems
    assert graph.run({"text": "ab cd"}).results["n"]["result"] == ["AB:AB", "CD:CD"]


def test_a_chain_of_a_thousand_nodes_in_a_loop_is_collected() -> None:
    """A long chain inside a loop doesn't hit Python's recursion limit.

    Whether any item of the loop can have a value is worked out one node at a time, in the order a run calls them.
    """
    chain = [(f"u{i}", "up") for i in range(1000)]
    edges = [
        ("s.result", "u0.a"),
        *[(f"u{i}.result", f"u{i + 1}.a") for i in range(999)],
        ("u999.result", "n.values"),
    ]
    assert outcome_of(ran({"s": "items", **dict(chain), "n": "count"}, edges, text="a b"), "n") == 2


@_node("pair-reversed")
def pair_reversed(b: str, a: str) -> str:
    return f"{a}+{b}"


@_node("settings")
def settings() -> dict[str, str]:
    return {}


@pytest.mark.parametrize("text", ["", "a"])
def test_the_order_of_a_nodes_parameters_does_not_change_a_collect_over_a_loop(text: str) -> None:
    """Swapping the order of a node's parameters doesn't change how a loop is collected.

    Inside the loop, ``t`` reads a field that ``settings`` doesn't return. The pair's other input is on a skipped
    branch. The run ends the same way with either order, for an empty list and for one item.
    """
    statuses = []
    for pair_id in ("pair", "pair-reversed"):
        nodes = {
            "s": "items",
            "c": "settings",
            "t": "pair",
            "w": "decide",
            "p": pair_id,
            "n": "count",
        }
        edges = [
            ("s.result", "t.a"),
            ("c.result.k", "t.b"),
            ("t.result", "p.a"),
            ("w.yes", "p.b"),
            ("p.result", "n.values"),
        ]
        graph = Graph.compile(
            make_graph(
                *((node_id, REGISTRY[registry_id]) for node_id, registry_id in nodes.items()),
                edges=edges,
            ),
            REGISTRY,
        )
        assert not graph.problems, graph.problems
        statuses.append(graph.run({"text": text, "go": "no"}).status)
    assert statuses[0] == statuses[1]


def test_two_sibling_loops_joined_through_collecting_nodes() -> None:
    """Two loops side by side, where one feeds the other only through collecting nodes.

    ``x`` is called once per letter of each word of ``first``. It also reads ``wt``, which collects a loop over
    ``second`` in two steps, first per word and then whole. Each collecting node works out only its own loop.
    """
    nodes = {
        "o": "items",
        "op": "items",
        "a": "chars",
        "b": "chars",
        "y": "pair",
        "w": "join",
        "wt": "join",
        "x": "pair",
        "j": "join",
        "jt": "join",
    }
    edges = [
        ("o.result", "a.a"),
        ("op.result", "b.a"),
        ("b.result", "y.a"),
        ("op.result", "y.b"),
        ("y.result", "w.values"),
        ("w.result", "wt.values"),
        ("a.result", "x.a"),
        ("wt.result", "x.b"),
        ("x.result", "j.values"),
        ("j.result", "jt.values"),
    ]
    run = ran(
        nodes, edges, inputs={"first": "o.text", "second": "op.text"}, first="ab c", second="xy"
    )
    assert run.results["j"]["result"] == ["a+x+xyy+xyb+x+xyy+xy", "c+x+xyy+xy"]
