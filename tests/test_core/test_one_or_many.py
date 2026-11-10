"""One value or many: the value a caller gives decides, and a node written for one value never sees the difference.

The scenarios are an editor's: a user uploads contracts and appendices, one or many, to a graph whose author picked
a node that reads one file or one that reads them all.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Annotated, Any, Literal, TypeAliasType

import pytest
from conductor import Ask, GraphRecord, InvalidCallError, NodeRegistry, Param, Series, node
from pydantic import BaseModel, ValidationError

CALLS: Counter[str] = Counter()
registry = NodeRegistry()


@registry.node("read")
def read(file: str) -> str:
    """Reads one file: run once per file when given several."""
    CALLS["read"] += 1
    return file.upper()


@registry.node("review")
def review(text: str) -> str:
    """Reviews one contract."""
    CALLS["review"] += 1
    return f"rev({text})"


@registry.node("overall")
def overall(texts: Series[str]) -> str:
    """Reviews every contract in one call."""
    CALLS["overall"] += 1
    return f"overall({len(texts)})"


@registry.node("compare")
def compare(contract: str, appendix: str) -> str:
    CALLS["compare"] += 1
    return f"{contract}~{appendix}"


@registry.node("with_appendices")
def with_appendices(contract: str, appendices: Series[str]) -> str:
    CALLS["with_appendices"] += 1
    return f"{contract}+{len(appendices)}"


@registry.node("tags")
def tags(tags: list[str]) -> int:
    return len(tags)


@registry.node("anything")
def anything(value: Any) -> str:
    return type(value).__name__


@registry.node("split")
def split(text: str) -> Series[str]:
    return text.split(",")


@registry.node("approve")
def approve(text: str) -> bool:
    raise Ask(f"approve {text}?")


def drawn(*nodes: tuple[str, str] | dict[str, Any], edges: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    """A record: each node ``(id, node)`` or a node record of its own, and edges ``(from, to)``."""
    return {
        "nodes": [n if isinstance(n, dict) else {"id": n[0], "node": n[1]} for n in nodes],
        "edges": [list(e) for e in edges],
    }


def ran(record: dict[str, Any], reg: NodeRegistry = registry, max_rows: int | None = None, **inputs: Any) -> Any:
    """The run of a record that loads clean, its calls counted afresh in ``CALLS``."""
    CALLS.clear()
    graph = reg.load(record)
    assert graph.problems == [], graph.problems
    return graph.run(inputs, max_rows=max_rows)


FILES = [f"k{i}" for i in range(11)]
PER = drawn(("read", "read"), ("review", "review"), edges=(("read.result", "review.text"),))
TOGETHER = drawn(("read", "read"), ("overall", "overall"), edges=(("read.result", "overall.texts"),))
BOTH = drawn(("read", "read"), ("review", "review"), ("overall", "overall"), edges=(("read.result", "review.text"), ("review.result", "overall.texts")))


# -- the scenario weighted highest: 11 contracts, one review each, all in one, or both ----------------------------


def test_eleven_contracts_reviewed_one_by_one_make_eleven_reviews_and_one_given_as_a_value_makes_one():
    run = ran(PER, file=FILES)
    assert CALLS == {"read": 11, "review": 11}
    assert run.outputs == {"review.result": [f"rev(K{i})" for i in range(11)]}
    assert run.graph.plan["review"].table == "read.file"

    run = ran(PER, file=FILES[0])
    assert CALLS == {"read": 1, "review": 1} and run.outputs == {"review.result": "rev(K0)"}
    assert run.graph.plan["review"].mode == "once"
    assert ran(PER, file=FILES[:1]).outputs == {"review.result": ["rev(K0)"]}  # a list of one is one row
    assert ran(PER, file=[]).outputs == {"review.result": []} and CALLS == {}


def test_eleven_contracts_reviewed_all_together_make_one_call_and_one_value_gathers_as_a_list_of_one():
    run = ran(TOGETHER, file=FILES)
    assert CALLS == {"read": 11, "overall": 1} and run.outputs == {"overall.result": "overall(11)"}
    assert ran(TOGETHER, file=FILES[0]).outputs == {"overall.result": "overall(1)"}


def test_both_in_one_graph_review_each_contract_and_then_all_of_them_once():
    run = ran(BOTH, file=FILES)
    assert CALLS == {"read": 11, "review": 11, "overall": 1} and run.outputs == {"overall.result": "overall(11)"}
    assert ran(BOTH, file=FILES[0]).outputs == {"overall.result": "overall(1)"}


def test_an_editor_says_before_any_run_how_many_calls_the_drawing_makes_for_several_files():
    """With one test file the three drawings run alike; ``calls_for`` tells them apart before a user uploads 11."""
    assert registry.load(PER).calls_for(file=11) == {"read": 11, "review": 11}
    assert registry.load(TOGETHER).calls_for(file=11) == {"read": 11, "overall": 1}
    assert registry.load(BOTH).calls_for(file=11) == {"read": 11, "review": 11, "overall": 1}
    assert registry.load(PER).calls_for() == {"read": 1, "review": 1}
    assert registry.load(PER).calls_for({"read.file": 0}) == {"read": 0, "review": 0}


def test_calls_for_counts_none_where_a_node_makes_the_rows_and_refuses_what_a_run_would():
    made = registry.load(drawn(("split", "split"), ("review", "review"), edges=(("split.result", "review.text"),)))
    assert made.calls_for() == {"split": 1, "review": None}
    with pytest.raises(InvalidCallError) as refused:
        made.calls_for(nothing=3)
    assert [p.type for p in refused.value.errors()] == ["not_offered"]
    with pytest.raises(InvalidCallError) as refused:
        made.calls_for(text=-1)
    assert [p.type for p in refused.value.errors()] == ["invalid_input"]


def test_calls_for_a_length_makes_no_list_of_that_length():
    import tracemalloc

    graph = registry.load(drawn(("r", "review")))
    tracemalloc.start()
    try:
        counts = graph.calls_for(text=10**7)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert counts == {"r": 10**7}
    assert peak < 1_000_000


# -- two uploads ---------------------------------------------------------------------------------------------------


TWO = drawn(("c", "read"), ("a", "read"), ("cmp", "compare"), edges=(("c.result", "cmp.contract"), ("a.result", "cmp.appendix")))
GATHERED = drawn(("c", "read"), ("a", "read"), ("w", "with_appendices"), edges=(("c.result", "w.contract"), ("a.result", "w.appendices")))


def test_two_uploads_meeting_at_a_node_are_refused_until_a_node_collects_one_and_one_value_is_copied_down():
    """Ten files are ten rows; which contract goes with which appendix is the author's to draw, never guessed."""
    graph = registry.load(TWO)
    assert graph.problems == []
    with pytest.raises(InvalidCallError) as refused:
        graph.run({"c.file": ["c1", "c2"], "a.file": ["a1", "a2"]})
    assert [(p.type, p.loc, p.ctx["tables"]) for p in refused.value.errors()] == [("lists_not_collected", ("cmp",), ["c.file", "a.file"])]
    assert "collect one of them first" in refused.value.errors()[0].msg
    with pytest.raises(InvalidCallError):
        graph.calls_for({"c.file": 2, "a.file": 2})
    assert ran(TWO, **{"c.file": "c", "a.file": ["a1", "a2"]}).outputs == {"cmp.result": ["C~A1", "C~A2"]}
    assert ran(TWO, **{"c.file": "c", "a.file": "a"}).outputs == {"cmp.result": "C~A"}


def test_a_contract_with_all_its_appendices_collects_the_appendices_and_reads_each_once():
    assert ran(GATHERED, **{"c.file": "c", "a.file": []}).outputs == {"w.result": "C+0"}
    assert ran(GATHERED, **{"c.file": "c", "a.file": "a"}).outputs == {"w.result": "C+1"}
    run = ran(GATHERED, **{"c.file": ["c1", "c2"], "a.file": ["a1", "a2", "a3"]})
    assert run.outputs == {"w.result": ["C1+3", "C2+3"]}
    assert CALLS == {"read": 5, "with_appendices": 2}
    assert registry.load(GATHERED).calls_for({"c.file": 11, "a.file": 5}) == {"c": 11, "a": 5, "w": 11}


def test_a_list_counts_against_max_rows_before_any_call():
    run = ran(PER, max_rows=10, file=FILES)
    assert run.status == "stopped" and run.cause.type == "row_ceiling" and run.cause.ctx["rows"] == 11
    assert CALLS == {}


def test_two_lists_typed_into_one_node_are_refused_as_lists_that_meet():
    typed = drawn({"id": "cmp", "node": "compare", "values": {"contract": ["c1", "c2"], "appendix": ["a1", "a2"]}})
    assert [p.type for p in registry.load(typed).problems] == ["lists_not_collected"]


# -- an embedded graph, per outer row ------------------------------------------------------------------------------


def test_an_embedded_graph_fed_one_file_per_outer_row_runs_per_row_whether_it_reviews_each_or_all():
    reg = NodeRegistry()
    reg.add_subregistry(registry)
    reg.add(node("per")(GraphRecord.model_validate(PER)), node("together")(GraphRecord.model_validate(TOGETHER)))
    for inner, results in (("per", ["rev(X)", "rev(Y)", "rev(Z)"]), ("together", ["overall(1)", "overall(1)", "overall(1)"])):
        outer = drawn(("s", "split"), ("e", inner), edges=(("s.result", "e.file"),))
        assert list(ran(outer, reg, text="x,y,z").outputs.values()) == [results]
    alone = reg.load(drawn(("e", "per")))
    assert alone.inputs["file"].per_item
    assert ran(drawn(("e", "per")), reg, file=["a", "b"]).outputs == {"e.result": ["rev(A)", "rev(B)"]}
    assert ran(drawn(("e", "per")), reg, file="a").outputs == {"e.result": "rev(A)"}


# -- an API caller -------------------------------------------------------------------------------------------------


def test_an_api_caller_sending_one_item_or_a_list_gets_one_value_or_a_list_and_the_graph_is_unchanged():
    graph = registry.load(PER)
    fingerprint = graph.fingerprint
    assert graph.run(json.loads('{"file": "a"}')).outputs == {"review.result": "rev(A)"}
    assert graph.run(json.loads('{"file": ["a", "b"]}')).outputs == {"review.result": ["rev(A)", "rev(B)"]}
    assert graph.fingerprint == fingerprint and graph.describe().inputs["file"].per_item


def test_an_input_whose_type_takes_a_list_takes_it_as_one_value():
    assert registry.load(drawn(("t", "tags"))).inputs["tags"].per_item is False
    assert ran(drawn(("t", "tags")), tags=["x", "y"]).outputs == {"t.result": 2}
    assert ran(drawn(("v", "anything")), value=["x", "y"]).outputs == {"v.result": "list"}
    assert ran(drawn(("o", "overall")), texts="one").outputs == {"o.result": "overall(1)"}


def test_a_run_given_a_list_that_waits_saves_and_loads_back_into_the_same_rows():
    graph = registry.load(drawn(("q", "approve")))
    run = graph.run(text=["a", "b"])
    assert [w.row for w in run.waiting] == [(0,), (1,)]
    saved = graph.load_run(run.to_json())
    assert saved.graph.plan["q"].mode == "per_row"
    done = saved.answer({w.id: w.row == (1,) for w in saved.waiting}).resume()
    assert done.outputs == {"q.result": [False, True]}
    one = graph.run(text="a")
    assert graph.load_run(one.to_json()).answer(one.waiting[0].id, True).resume().outputs == {"q.result": True}


# -- each item of a list is checked as one value is ------------------------------------------------


def _checked_registry() -> NodeRegistry:
    held = NodeRegistry()

    @held.node("count")
    def count(n: Annotated[int, Param(ge=0)]) -> int:
        return n

    @held.node("level")
    def level(n: Literal[1, 2, 3]) -> int:
        return n

    @held.node("pick")
    def pick(s: Annotated[str, Param(choices=["a", "b"])]) -> str:
        return s

    return held


def _refused_as(call: Any) -> list[tuple[str, str]]:
    with pytest.raises(InvalidCallError) as refused:
        call()
    return [(p.type, p.ctx["errors"][0]["type"]) for p in refused.value.errors()]


@pytest.mark.parametrize(("node_id", "items", "error"), [
    ("count", [1, -1], "greater_than_equal"),
    ("count", [1, True], "bool_is_not_a_number"),
    ("level", [1, True], "bool_is_not_a_number"),
    ("pick", ["a", "z"], "not_a_choice"),
], ids=["limit", "bool-on-int", "bool-on-literal", "choice"])
def test_each_item_of_a_list_given_or_typed_in_keeps_the_inputs_limits_and_choices(node_id, items, error):
    held = _checked_registry()
    graph = held.load(GraphRecord(nodes=[{"id": "x", "node": node_id}]))
    name = "s" if node_id == "pick" else "n"

    assert _refused_as(lambda: graph.run({name: items})) == [("invalid_input", error)]
    (problem,) = held.load(GraphRecord(nodes=[{"id": "x", "node": node_id, "values": {name: items}}])).problems
    assert (problem.type, problem.ctx["errors"][0]["type"]) == ("invalid_value", error)


def test_a_yes_or_no_never_lands_on_a_literal_of_numbers():
    graph = _checked_registry().load(GraphRecord(nodes=[{"id": "x", "node": "level"}]))

    assert _refused_as(lambda: graph.run(n=True)) == [("invalid_input", "bool_is_not_a_number")]
    assert graph.run(n=2).outputs == {"x.result": 2}


def test_a_yes_or_no_never_lands_on_a_union_a_number_is_in_and_a_yes_or_no_is_not():
    held = NodeRegistry()

    @held.node("either")
    def either(n: int | float) -> str:
        return repr(n)

    @held.node("or-text")
    def or_text(n: int | str | None = None) -> str:
        return repr(n)

    @held.node("or-flag")
    def or_flag(n: int | bool) -> str:
        return repr(n)

    @held.node("returns-yes")
    def returns_yes(n: int) -> Literal[1, 2] | float:
        return True  # type: ignore[return-value]

    for node_id in ("either", "or-text"):
        graph = held.load(GraphRecord(nodes=[{"id": "x", "node": node_id}]))
        assert _refused_as(lambda: graph.run(n=True)) == [("invalid_input", "bool_is_not_a_number")]
        assert graph.run(n=2).outputs == {"x.result": "2"}
    assert held.load(GraphRecord(nodes=[{"id": "x", "node": "or-flag"}])).run(n=True).outputs == {"x.result": "True"}
    assert held.load(GraphRecord(nodes=[{"id": "x", "node": "returns-yes"}])).run(n=1).status == "failed"


class Order(BaseModel):
    count: Annotated[int, Param(title="Count", ge=0)]


def test_a_param_holds_its_limits_inside_a_hosts_own_model_and_its_schema_says_them():
    with pytest.raises(ValidationError) as refused:
        Order(count=-1)
    assert refused.value.errors()[0]["type"] == "greater_than_equal"
    assert Order.model_json_schema()["properties"]["count"] == {"type": "integer", "minimum": 0, "title": "Count"}


Tree = TypeAliasType("Tree", "dict[str, Tree] | list[Tree] | int")


def test_a_list_given_to_a_type_alias_whose_values_may_be_lists_is_one_value():
    held = NodeRegistry()

    @held.node("depth")
    def depth(tree: Tree) -> str:
        return repr(tree)

    graph = held.load(GraphRecord(nodes=[{"id": "d", "node": "depth"}]))

    assert graph.inputs["tree"].per_item is False
    assert graph.run(tree=[1, [2]]).outputs == {"d.result": "[1, [2]]"}
