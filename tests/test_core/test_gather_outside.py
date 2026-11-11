"""What is outside the items decides a gathering node only when it is certain: when no item could bring a value
because a branch outside wasn't taken the node is skipped, and when none could for a blank outside it isn't
called, with two items as with none. Otherwise each item decides, and no items is an empty Series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import SKIPPED, NodeRegistry, Param, Series, make_graph

REGISTRY = NodeRegistry()


def _node(node_id: str) -> Any:
    return REGISTRY.node(node_id, title=node_id, description=node_id)


@_node("items")
def items(text: str) -> Series[str]:
    return text.split()


@_node("template")
def template(kind: str) -> str | None:
    return SKIPPED if kind == "skip" else None if kind == "blank" else kind


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
    return None if a == "blank" else SKIPPED if a == "skip" else a.upper()


@_node("chars")
def chars(a: str) -> Series[str]:
    return list(a)


@_node("count")
def count(values: Series[str]) -> int:
    return len(values)


@_node("count-or-none")
def count_or_none(values: Series[str] | None) -> str:
    return "none" if values is None else str(len(values))


@_node("count-with-blanks")
def count_with_blanks(values: Series[str | None]) -> str:
    return "".join("-" if value is None else "x" for value in values)


@_node("join")
def join(values: Series[str]) -> str:
    return "".join(values)


@_node("join-or-none")
def join_or_none(values: Series[str] | None) -> str:
    return "none" if values is None else "".join(values)


@_node("join-with-blanks")
def join_with_blanks(values: Series[str | None]) -> str:
    return "".join("-" if value is None else value for value in values)


@_node("with-all")
def with_all(a: str, values: Series[str]) -> str:
    return a + "".join(values)


@_node("with-all-or-blanks")
def with_all_or_blanks(a: str, values: Series[str | None]) -> str:
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


REGISTRY.node("pair-then-up-or-none", title="Pair then up", description="A pair, in capitals, or none.")(
    make_graph(pair, up_or_none, edges=[("pair.result", "up-or-none.a")])
)


REGISTRY.node("items-then-up", title="Items then up", description="Each word of a text, in capitals.")(
    make_graph(items, up, edges=[("items.result", "up.a")])
)


REGISTRY.node("pair-then-items-then-up", title="Pair then items then up", description="Each word of a pair, in capitals.")(
    make_graph(pair, items, up, edges=[("pair.result", "items.text"), ("items.result", "up.a")])
)


def ran(nodes: dict[str, str], edges: list[tuple[str, str]], **inputs: str) -> Any:
    graph = REGISTRY.load({"nodes": [{"id": i, "node": n} for i, n in nodes.items()], "edges": [list(edge) for edge in edges]})
    assert not graph.problems, graph.problems
    run = graph.run(**inputs)
    assert run.status == "complete"
    return run


def came_to(run: Any, node_id: str) -> Any:
    """What the node that runs once came to: its result, ``"not called"`` for a blank, or ``"skipped"``."""
    result = run.results[node_id]["result"]
    if result is not None:
        return result
    return "not called" if any(call.loc == (node_id,) for call in run.not_called) else "skipped"


BOTH_WAYS = (
    {"s": "items", "w": "decide", "t": "template", "y": "three", "z": "pair", "p": "up", "n": "count"},
    [("s.result", "y.a"), ("w.yes", "y.b"), ("t.result", "y.c"), ("s.result", "z.a"), ("w.no", "z.b"), ("y.result", "p.a"), ("z.result", "p.a"), ("p.result", "n.values")],
)


@pytest.mark.parametrize(("go", "kind", "text", "n"), [
    ("no", "blank", "a b", 2), ("no", "blank", "", 0), ("no", "x", "a b", 2),
    ("yes", "blank", "a b", "not called"), ("yes", "blank", "", "not called"), ("yes", "x", "a", 1),
])
def test_a_blank_outside_read_only_on_the_branch_not_taken_leaves_the_items_to_decide(go: str, kind: str, text: str, n: Any) -> None:
    assert came_to(ran(*BOTH_WAYS, go=go, kind=kind, text=text), "n") == n


@pytest.mark.parametrize(("kind", "text", "n"), [("blank", "a b", 2), ("blank", "", 0), ("x", "a b", 2), ("skip", "a b", "skipped"), ("skip", "", "skipped")])
def test_an_embedded_graph_in_the_items_that_takes_the_blank_gives_each_item_a_value(kind: str, text: str, n: Any) -> None:
    run = ran({"s": "items", "t": "template", "e": "pair-then-up-or-none", "n": "count"}, [("s.result", "e.a"), ("t.result", "e.b"), ("e.up-or-none.result", "n.values")], kind=kind, text=text)
    assert came_to(run, "n") == n


def _nested(joiner: str, kind: str, text: str) -> Any:
    nodes = {"s": "items", "t": "template", "c": "chars", "m": "pair", "j": joiner, "n": "count"}
    return ran(nodes, [("s.result", "c.a"), ("c.result", "m.a"), ("t.result", "m.b"), ("m.result", "j.values"), ("j.result", "n.values")], kind=kind, text=text)


@pytest.mark.parametrize("text", ["ab cd", "ab", ""])
def test_a_gather_in_the_items_below_a_blank_or_a_branch_not_taken_outside_comes_to_the_same_for_every_item_and_so_does_the_gather_above(text: str) -> None:
    blank, skipped = _nested("join", "blank", text), _nested("join", "skip", text)
    assert (came_to(blank, "n"), came_to(skipped, "n")) == ("not called", "skipped")
    assert blank.results["j"]["result"] == skipped.results["j"]["result"] == [None] * len(text.split())
    assert [call.loc[0] for call in skipped.not_called] == []
    assert _nested("join", "x", "ab cd").results["n"]["result"] == 2


def test_an_inner_gather_that_takes_a_blank_series_is_called_so_the_items_decide_above() -> None:
    run = _nested("join-or-none", "blank", "ab cd")
    assert (run.results["j"]["result"], came_to(run, "n")) == (["none", "none"], 2)
    assert came_to(_nested("join-or-none", "blank", ""), "n") == 0


def test_a_series_that_takes_a_blank_per_item_receives_each_items_blank_for_a_blank_outside() -> None:
    run = _nested("join-with-blanks", "blank", "ab cd")
    assert (run.results["j"]["result"], came_to(run, "n")) == (["--", "--"], 2)
    assert came_to(_nested("join-with-blanks", "skip", "ab cd"), "n") == "skipped"
    nodes, edges = {"s": "items", "t": "template", "p": "pair", "n": "count-with-blanks"}, [("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")]
    assert [came_to(ran(nodes, edges, kind="blank", text=text), "n") for text in ("a b", "")] == ["--", ""]


def test_a_series_never_made_for_a_blank_is_the_inputs_own_blank_whatever_its_items_take() -> None:
    nodes = {"t": "template", "s": "items", "u": "up", "n": "count-with-blanks", "o": "count-or-none"}
    run = ran(nodes, [("t.result", "s.text"), ("s.result", "u.a"), ("u.result", "n.values"), ("u.result", "o.values")], kind="blank")
    assert (came_to(run, "n"), came_to(run, "o")) == ("not called", "none")


@pytest.mark.parametrize("text", ["a b", ""])
def test_a_decision_in_the_items_not_called_for_a_blank_outside_chose_no_branch_gathered_directly_or_through_a_reader(text: str) -> None:
    nodes, edges = {"s": "items", "t": "template", "w": "decide-with"}, [("s.result", "w.go"), ("t.result", "w.extra")]
    direct = ran({**nodes, "n": "count"}, [*edges, ("w.yes", "n.values")], kind="blank", text=text)
    through = ran({**nodes, "u": "up", "n": "count"}, [*edges, ("w.yes", "u.a"), ("u.result", "n.values")], kind="blank", text=text)
    assert (came_to(direct, "n"), came_to(through, "n")) == ("skipped", "skipped")


def test_a_decision_in_the_items_not_called_for_a_blank_item_is_that_items_own_and_is_left_out() -> None:
    nodes = {"s": "items", "m": "maybe", "w": "decide", "n": "count"}
    run = ran(nodes, [("s.result", "m.a"), ("m.result", "w.go"), ("w.yes", "n.values")], text="y1 blank y2")
    assert came_to(run, "n") == 2


def test_a_blank_outside_beside_a_branch_decided_per_item_leaves_the_gathering_node_uncalled() -> None:
    nodes = {"s": "items", "t": "template", "w": "decide", "p": "pair", "n": "count"}
    edges = [("s.result", "w.go"), ("w.yes", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")]
    assert [came_to(ran(nodes, edges, kind="blank", text=text), "n") for text in ("yes no", "")] == ["not called", "not called"]


def test_a_branch_not_taken_outside_skips_a_gather_whose_series_is_blank_for_an_item() -> None:
    nodes = {"s": "items", "t": "template", "m": "maybe", "c": "chars", "p": "pair", "j": "join-or-none", "n": "count"}
    edges = [("s.result", "m.a"), ("m.result", "c.a"), ("c.result", "p.a"), ("t.result", "p.b"), ("p.result", "j.values"), ("j.result", "n.values")]
    run = ran(nodes, edges, kind="skip", text="ab blank")
    assert (run.results["j"]["result"], came_to(run, "n"), run.not_called[0].loc) == ([None, None], "skipped", ("c", 1))
    assert [call.loc[0] for call in run.not_called] == ["c"]  # j is skipped for both items, never left uncalled for the blank Series


OTHER_SERIES = (
    {"r": "items", "s2": "chars", "z": "items", "q": "maybe", "j": "with-all", "n": "count"},
    [("r.result", "s2.a"), ("z.result", "q.a"), ("s2.result", "j.a"), ("q.result", "j.values"), ("j.result", "n.values")],
)


@pytest.mark.parametrize("other", ["x y", "x skip", "blank x", "skip skip", ""])
def test_a_node_in_the_items_that_gathers_another_series_reads_it_as_one_value_made_outside(other: str) -> None:
    run = ran(*OTHER_SERIES, **{"r.text": "ab cd", "z.text": other})
    gathered = "".join(word.upper() for word in other.split() if word not in ("blank", "skip"))
    assert run.results["j"]["result"] == [f"{char}{gathered}" for char in "abcd"]
    assert run.results["n"]["result"] == [2, 2]


def test_a_node_in_the_items_that_gathers_a_series_never_made_is_not_called_and_nor_is_the_gather_above() -> None:
    nodes, edges = OTHER_SERIES
    run = ran({**nodes, "t": "template"}, [*edges, ("t.result", "z.text")], kind="blank", text="ab cd")
    assert run.results["n"]["result"] == [None, None]
    assert [call.loc for call in run.not_called] == [("z",), ("j", 0, 0), ("j", 0, 1), ("j", 1, 0), ("j", 1, 1), ("n", 0), ("n", 1)]


@pytest.mark.parametrize(("first", "second", "text", "n"), [
    ("yes", "yes", "p q", 2), ("no", "no", "p", 1), ("yes", "no", "p q", "skipped"), ("yes", "no", "", "skipped"), ("no", "yes", "", "skipped"), ("yes", "yes", "", 0),
])
def test_alternatives_in_the_items_skip_a_gather_only_when_the_branch_of_every_one_was_not_taken_outside(first: str, second: str, text: str, n: Any) -> None:
    # a needs e.yes and f.yes, b needs e.no and f.no; either feeds p, which runs per item and is gathered
    nodes = {"s": "items", "e": "decide", "f": "decide", "a": "three", "b": "three", "p": "up", "n": "count"}
    edges = [
        ("s.result", "a.a"), ("e.yes", "a.b"), ("f.yes", "a.c"), ("s.result", "b.a"), ("e.no", "b.b"), ("f.no", "b.c"),
        ("a.result", "p.a"), ("b.result", "p.a"), ("p.result", "n.values"),
    ]
    assert came_to(ran(nodes, edges, **{"e.go": first, "f.go": second, "text": text}), "n") == n


@pytest.mark.parametrize(("ways", "n"), [(["yes"] * 40, 2), (["yes", "no"] * 20, 2), ([*["yes"] * 20, "skip", *["no"] * 19], "skipped")])
def test_a_long_chain_of_alternatives_in_the_items_is_asked_node_by_node_however_many_ways_lead_through_it(ways: list[str], n: Any) -> None:
    """Forty stages, each two nodes on opposite branches of a decision made outside the items, both fed by both
    nodes of the stage before: a million million ways up, eighty nodes to ask."""
    nodes: list[dict[str, Any]] = [{"id": "s", "node": "items", "values": {"text": "p q"}}]
    edges: list[tuple[str, str]] = []
    before = ["s.result"]
    for stage, way in enumerate(ways):
        nodes += [{"id": f"t{stage}", "node": "template", "values": {"kind": way}}, {"id": f"w{stage}", "node": "decide"}]
        nodes += [{"id": f"x{stage}", "node": "pair"}, {"id": f"y{stage}", "node": "pair"}]
        edges += [(f"t{stage}.result", f"w{stage}.go"), (f"w{stage}.yes", f"x{stage}.b"), (f"w{stage}.no", f"y{stage}.b")]
        edges += [(source, f"{node_id}{stage}.a") for source in before for node_id in "xy"]
        before = [f"x{stage}.result", f"y{stage}.result"]
    graph = REGISTRY.load({"nodes": [*nodes, {"id": "n", "node": "count"}], "edges": [list(edge) for edge in [*edges, *((source, "n.values") for source in before)]]})
    assert not graph.problems, graph.problems
    assert came_to(graph.run(), "n") == n


@pytest.mark.parametrize("counter", ["count", "count-or-none", "count-with-blanks"])
def test_a_series_never_made_below_a_decision_made_per_item_skips_the_gather_as_no_branch_was_chosen(counter: str) -> None:
    nodes, edges = {"t": "template", "s": "items", "w": "decide"}, [("t.result", "s.text"), ("s.result", "w.go")]
    direct = ran({**nodes, "n": counter}, [*edges, ("w.no", "n.values")], kind="blank")
    through = ran({**nodes, "u": "up", "n": counter}, [*edges, ("w.no", "u.a"), ("u.result", "n.values")], kind="blank")
    assert (came_to(direct, "n"), came_to(through, "n")) == ("skipped", "skipped")


def test_a_series_never_made_is_the_gathers_own_blank_where_the_nodes_per_item_would_take_a_blank_item() -> None:
    nodes = {"t": "template", "s": "items", "u": "up-or-none", "n": "count", "o": "count-or-none"}
    run = ran(nodes, [("t.result", "s.text"), ("s.result", "u.a"), ("u.result", "n.values"), ("u.result", "o.values")], kind="blank")
    assert (came_to(run, "n"), came_to(run, "o")) == ("not called", "none")


def test_a_series_of_one_whose_one_value_is_blank_is_received_by_a_series_that_takes_a_blank_per_item() -> None:
    nodes = {"t": "template", "n": "count", "o": "count-or-none", "k": "count-with-blanks"}
    run = ran(nodes, [("t.result", "n.values"), ("t.result", "o.values"), ("t.result", "k.values")], kind="blank")
    assert (came_to(run, "n"), came_to(run, "o"), came_to(run, "k")) == ("not called", "none", "-")
    assert came_to(ran({"t": "template", "k": "count-with-blanks"}, [("t.result", "k.values")], kind="skip"), "k") == "skipped"


def test_a_node_in_the_items_whose_series_of_one_takes_a_blank_made_outside_lets_the_items_decide() -> None:
    nodes, edges = {"s": "items", "t": "template", "x": "with-all", "n": "count"}, [("s.result", "x.a"), ("t.result", "x.values"), ("x.result", "n.values")]
    assert came_to(ran(nodes, edges, kind="blank", text="a b"), "n") == "not called"  # with-all takes no blank in its Series of one
    assert came_to(ran({**nodes, "x": "with-all-or-blanks"}, edges, kind="blank", text="a b"), "n") == 2


@pytest.mark.parametrize(("kind", "counter", "n"), [
    ("blank", "count", "not called"), ("blank", "count-or-none", "none"), ("skip", "count", "skipped"), ("skip", "count-or-none", "skipped"),
    ("a b", "count", 2), ("", "count", 0),
])
def test_a_series_made_inside_an_embedded_graph_and_never_made_there_is_gathered_outside_as_the_same_nodes_drawn_flat(kind: str, counter: str, n: Any) -> None:
    flat = ran({"t": "template", "w": "items", "u": "up", "n": counter}, [("t.result", "w.text"), ("w.result", "u.a"), ("u.result", "n.values")], kind=kind)
    embedded = ran({"t": "template", "e": "items-then-up", "n": counter}, [("t.result", "e.text"), ("e.up.result", "n.values")], kind=kind)
    assert (came_to(flat, "n"), came_to(embedded, "n")) == (n, n)


def test_asking_an_embedded_graph_for_an_output_it_does_not_offer_fails_at_once() -> None:
    """No loaded graph has such an edge; the question reads the embedded graph's offers by address and nothing else."""
    from conductor._internal.graph import Cell
    from conductor._internal.outside_the_items import _Asking, _could_hold

    graph = REGISTRY.load({"nodes": [{"id": "s", "node": "items"}, {"id": "e", "node": "pair-then-up-or-none"}], "edges": [["s.result", "e.a"]]})
    node_at = graph._node_at
    asking = _Asking(node_at[("s",)], Cell("s", "result", (), False), {node_at[("e",)]: None}, node_at, None)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="no-such-output"):
        _could_hold((), 1, Cell("e", "no-such-output", (), False), asking)


@pytest.mark.parametrize("text", ["a b", ""])
def test_a_gather_asks_nothing_when_only_the_series_has_a_say_and_asks_when_a_node_per_item_reads_anything_else(text: str) -> None:
    """Where all the nodes per item read outside the items is the Series, the gather comes to what the Series is.
    A second output of the node that makes the Series is something else: blank, it leaves the gather not called."""
    nodes = {"x": "say", "t": "template", "l": "items-and-note", "p": "pair", "u": "up", "n": "count", "m": "count"}
    edges = [("x.result", "l.text"), ("t.result", "l.note"), ("l.items", "u.a"), ("u.result", "m.values"), ("l.items", "p.a"), ("l.note", "p.b"), ("p.result", "n.values")]
    run = ran(nodes, edges, kind="blank", text=text)
    assert (came_to(run, "m"), came_to(run, "n")) == (len(text.split()), "not called")
    assert came_to(ran(nodes, edges, kind="x", text=text), "n") == len(text.split())


@pytest.mark.parametrize(("kind", "n"), [("blank", "not called"), ("skip", "skipped"), ("v", 2)])
def test_a_series_made_inside_an_embedded_graph_in_the_items_and_never_made_for_what_is_outside_them_decides_the_gather_above(kind: str, n: Any) -> None:
    flat = ran(
        {"s": "items", "t": "template", "p": "pair", "w": "items", "u": "up", "j": "join", "n": "count"},
        [("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "w.text"), ("w.result", "u.a"), ("u.result", "j.values"), ("j.result", "n.values")],
        kind=kind, text="a b",
    )
    embedded = ran(
        {"s": "items", "t": "template", "e": "pair-then-items-then-up", "j": "join", "n": "count"},
        [("s.result", "e.a"), ("t.result", "e.b"), ("e.up.result", "j.values"), ("j.result", "n.values")],
        kind=kind, text="a b",
    )
    assert (came_to(flat, "n"), came_to(embedded, "n")) == (n, n)
