"""The graph record: its shape, how it is saved, what loading it reports, and ``make_graph``.

A record is ``{"nodes": [...], "edges": [[from, to], ...]}``. ``registry.load``
reads it into a ``Graph``; what the record says that can't be meant is a
problem of that graph, never an exception.
"""

from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
from conductor import (
    SKIPPED,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    Param,
    Series,
    Table,
    Upgrade,
    make_graph,
    node,
)
from pydantic import ValidationError

from tests.test_core.nodes import registry_of


@node("text", title="Text in", description="")
def text_in(value: str = "") -> str:
    return value


@node("upper", title="Upper", description="")
def upper(text: str) -> str:
    return text.upper()


@node("join", title="Join", description="")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class YesNo:
    yes: Annotated[str, Param(exclusive_group="answer")]
    no: Annotated[str, Param(exclusive_group="answer")]


@node("decide", title="Decide", description="")
def decide(text: str) -> YesNo:
    return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)


@node("both", title="Both", description="")
def both(yes: str, no: str) -> str:
    return yes + no


@dataclass(frozen=True)
class Two:
    zed: str
    alpha: str


@node("two", title="Two", description="")
def two(text: str) -> Two:
    return Two(f"zed {text}", f"alpha {text}")


REGISTRY = registry_of(text_in, upper, join, decide, both, two)


def _problems(record: GraphRecord) -> list[tuple[str, tuple]]:
    return [(p.type, p.loc) for p in REGISTRY.load(record).problems]


# -- the record's shape ---------------------------------------------------------


def test_a_node_is_its_id_its_node_and_what_is_typed_in():
    assert list(NodeRecord.model_fields) == ["id", "node", "version", "values", "defaults", "display"]
    assert list(GraphRecord.model_fields) == ["nodes", "edges"]


def test_a_bare_node_saves_as_its_id_and_node():
    """Empty fields and the version 1 are left out when saved."""
    assert NodeRecord(id="a", node="upper").model_dump() == {"id": "a", "node": "upper"}
    assert NodeRecord(id="a", node="upper", version=2, values={"text": "hi"}).model_dump() == {
        "id": "a", "node": "upper", "version": 2, "values": {"text": "hi"},
    }
    graph = GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper", "version": 1, "defaults": {}, "display": {}}]})
    assert graph.model_dump(mode="json") == {"nodes": [{"id": "a", "node": "upper"}], "edges": []}
    assert graph.model_dump(mode="json", include={"nodes", "edges"}) == {"nodes": [{"id": "a", "node": "upper"}], "edges": []}


def test_the_record_reads_back_what_it_saved():
    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node="t", values={"x": {"refs": ["q.r"]}}, display={"x": 10}),
        NodeRecord(id="b", node="t", version=2, defaults={"y": 1}),
    ], edges=[("a.result", "b.y")])

    assert GraphRecord.model_validate_json(graph.model_dump_json()) == graph
    assert graph.model_dump(mode="json")["edges"] == [["a.result", "b.y"]]


def test_display_is_the_hosts_own_and_takes_any_key():
    assert NodeRecord(id="n", node="t", display={"x": 10, "y": 20, "anything": [1]}).display == {"x": 10, "y": 20, "anything": [1]}


def test_an_edge_is_two_addresses():
    with pytest.raises(ValidationError):
        GraphRecord(edges=[("a", "b.x")])
    with pytest.raises(ValidationError):
        GraphRecord(edges=[("a.b.", "b.x")])
    assert GraphRecord(edges=[("a.result.Amount in USD.", "b.x")]).edges == (("a.result.Amount in USD.", "b.x"),)
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": [{"id": "a", "node": "t", "bindings": {}}]})


def test_edges_are_a_set_so_their_order_never_tells_two_records_apart():
    nodes = [NodeRecord(id="a", node="t"), NodeRecord(id="b", node="t")]
    one_way = GraphRecord(nodes=nodes, edges=[("a.x", "b.x"), ("a.y", "b.y")])
    other_way = GraphRecord(nodes=nodes, edges=[("a.y", "b.y"), ("a.x", "b.x")])

    assert one_way == other_way and one_way.model_dump_json() == other_way.model_dump_json()


# -- loading: what the record says that can't be meant ----------------------------


def test_several_edges_into_one_input_gather_in_node_order_whatever_the_list_order():
    nodes = [NodeRecord(id="first", node="text", values={"value": "a"}), NodeRecord(id="second", node="text", values={"value": "b"}), NodeRecord(id="j", node="join")]
    for edges in ([("second.result", "j.texts"), ("first.result", "j.texts")], [("first.result", "j.texts"), ("second.result", "j.texts")]):
        assert REGISTRY.load(GraphRecord(nodes=nodes, edges=edges)).run().outputs == {"j.result": "a+b"}
    nodes.reverse()
    assert REGISTRY.load(GraphRecord(nodes=nodes, edges=[("first.result", "j.texts"), ("second.result", "j.texts")])).run().outputs == {"j.result": "b+a"}


def test_two_outputs_of_one_node_into_one_input_gather_in_the_order_the_node_declares_them():
    """``two`` declares ``zed`` before ``alpha``; their addresses sort the other way."""
    graph = REGISTRY.load(GraphRecord(
        nodes=[NodeRecord(id="t", node="two"), NodeRecord(id="j", node="join")],
        edges=[("t.alpha", "j.texts"), ("t.zed", "j.texts")],
    ))

    assert graph.problems == []
    assert graph.run(text="x").outputs == {"j.result": "zed x+alpha x"}


def test_a_repeated_edge_is_a_problem():
    record = GraphRecord(nodes=[NodeRecord(id="a", node="text"), NodeRecord(id="b", node="upper")], edges=[("a.result", "b.text"), ("a.result", "b.text")])

    assert _problems(record) == [("repeated_edge", ("b", "text"))]


def test_an_edge_naming_a_node_the_graph_does_not_have_is_one_problem_at_either_end():
    into = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="a", node="text")], edges=[("a.result", "gone.text")]))
    out_of = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="b", node="upper")], edges=[("gone.result", "b.text")]))

    assert [(p.type, p.loc, p.ctx["node_id"]) for p in into.problems] == [("edge_names_missing_node", ("gone", "text"), "gone")]
    assert [(p.type, p.loc, p.ctx["node_id"]) for p in out_of.problems] == [("edge_names_missing_node", ("b", "text"), "gone")]


def test_an_input_set_two_ways_is_a_problem():
    record = GraphRecord(nodes=[
        NodeRecord(id="a", node="text"),
        NodeRecord(id="b", node="both", values={"yes": "1", "no": "2"}, defaults={"no": "3"}),
    ], edges=[("a.result", "b.yes")])

    assert sorted(_problems(record)) == [("value_and_default", ("b", "no")), ("value_and_edge", ("b", "yes"))]


def test_a_value_for_an_input_the_node_does_not_have_is_a_problem():
    assert _problems(GraphRecord(nodes=[NodeRecord(id="u", node="upper", values={"text": "a", "colour": "red"})])) == [("unknown_input", ("u", "colour"))]


@pytest.mark.parametrize("bad", ["a.b", "a/b", "a@b"])
def test_a_node_id_with_a_dot_a_slash_or_an_at_is_a_problem(bad):
    assert ("invalid_node_id", (bad,)) in _problems(GraphRecord(nodes=[NodeRecord(id=bad, node="upper")]))


def test_two_nodes_under_one_id_is_a_problem():
    assert _problems(GraphRecord(nodes=[NodeRecord(id="a", node="upper"), NodeRecord(id="a", node="upper")])) == [("duplicate_node_id", ("a",))]


# -- what a graph takes and gives -------------------------------------------------


def test_a_graph_takes_the_open_inputs_of_its_first_nodes_and_gives_what_nothing_reads():
    graph = REGISTRY.load(GraphRecord(
        nodes=[NodeRecord(id="a", node="text"), NodeRecord(id="b", node="upper"), NodeRecord(id="c", node="both", values={"no": "!"})],
        edges=[("a.result", "b.text")],
    ))

    assert graph.problems == []
    assert {name: (i.address, i.required) for name, i in graph.inputs.items()} == {"value": ("a.value", False), "yes": ("c.yes", True)}
    assert graph.outputs == ["b.result", "c.result"]


def test_a_value_closes_its_input_and_a_default_leaves_it_open_prefilled():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="b", node="both", values={"yes": "y"}, defaults={"no": "n"})]))

    assert list(graph.inputs) == ["no"] and not graph.inputs["no"].required and graph.inputs["no"].default == "n"
    assert graph.inputs["no"].default == "n"
    assert graph.run().outputs == {"b.result": "yn"}
    assert graph.run(no="!").outputs == {"b.result": "y!"}


def test_two_open_inputs_of_one_name_are_called_by_address():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="a", node="upper"), NodeRecord(id="b", node="upper")]))

    assert list(graph.inputs) == ["a.text", "b.text"]
    assert graph.run({"a.text": "x", "b.text": "y"}).outputs == {"a.result": "X", "b.result": "Y"}


def test_a_list_given_to_an_open_input_of_one_value_is_a_row_per_item_and_one_value_is_one():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="j", node="join")], edges=[("u.result", "j.texts")]))

    assert graph.inputs["text"].per_item
    assert graph.run(text=["a", "b"]).outputs == {"j.result": "A+B"}
    assert graph.run(text="a").outputs == {"j.result": "A"}


# -- a graph written from Python: make_graph -----------------------------------------------


@node("hello", title="Hello", description="")
def _hello(name: str) -> str:
    return f"hello {name}"


@node("shout", title="Shout", description="")
def _shout(text: str, times: int = 1) -> str:
    return text.upper() + "!" * times


@node("email", title="Email", description="")
def _email() -> str:
    return "ida@x"


@node("count", title="Count", description="")
def _count() -> int:
    return 3


PLAIN = registry_of(_hello, _shout, _email, _count)


def test_make_graph_with_no_edges_chains_each_node_to_the_one_before():
    graph = make_graph(_hello, _shout, values={"hello.name": "ida"})

    assert graph == GraphRecord(nodes=[
        NodeRecord(id="hello", node="hello", values={"name": "ida"}),
        NodeRecord(id="shout", node="shout"),
    ], edges=[("hello.result", "shout.text")])
    assert PLAIN.load(graph).run().outputs == {"shout.result": "HELLO IDA!"}


def test_make_graph_numbers_a_node_used_twice():
    graph = make_graph(_hello, _shout, _shout, values={"hello.name": "ida", "shout-2.times": 2})

    assert [n.id for n in graph.nodes] == ["hello", "shout-1", "shout-2"]
    assert graph.edges == (("hello.result", "shout-1.text"), ("shout-1.result", "shout-2.text"))
    assert PLAIN.load(graph).run().outputs == {"shout-2.result": "HELLO IDA!!!"}


def test_make_graph_skips_an_id_another_node_already_has():
    @node("hello-1", title="Other", description="")
    def _other(name: str) -> str:
        return name

    assert [n.id for n in make_graph(_hello, _hello, _other, edges=[]).nodes] == ["hello-2", "hello-3", "hello-1"]


def test_make_graph_with_edges_takes_those_as_every_edge_and_matches_nothing_by_name():
    graph = make_graph(_hello, _shout, edges=[])

    assert graph.edges == ()
    assert list(PLAIN.load(graph).inputs) == ["name", "text", "times"]


def test_make_graph_refuses_a_chain_it_cannot_draw_at_the_call():
    with pytest.raises(InvalidCallError, match="pass edges=") as refused:
        make_graph(_email, _count)
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("not_a_chain", ("count",))]
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, _shout, values={"hello.name": "x"})
    assert [(p.type, p.loc, p.ctx["inputs"]) for p in refused.value.errors()] == [("not_a_chain", ("hello",), [])]


def test_make_graph_refuses_a_value_on_a_node_it_does_not_have():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, values={"nobody.name": "x"})
    assert [(p.type, p.ctx["nodes"]) for p in refused.value.errors()] == [("node_not_in_graph", ["hello"])]


def test_make_graph_checks_every_address_and_value_at_the_call_and_lists_every_fault():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(
            _hello, _shout,
            edges=[("hello.reslt", "shout.text"), ("nobody.result", "shout.text"), ("hello.result", "shout.txt")],
            values={"hello.nmae": "x"}, defaults={"hello.name": 3},
        )
    assert sorted((p.type, p.loc) for p in refused.value.errors()) == [
        ("edge_names_missing_node", ("shout", "text")),
        ("invalid_value", ("hello", "name")),
        ("unknown_input", ("hello", "nmae")),
        ("unknown_input", ("shout", "txt")),
        ("unknown_output", ("shout", "text")),
    ]


@pytest.mark.parametrize("value", [{"ida", "bo"}, frozenset({"ida"}), [{"ida"}], {"names": {"ida"}}], ids=["set", "frozenset", "in-a-list", "in-a-dict"])
def test_make_graph_refuses_a_set_typed_in_as_set_has_no_order(value: object):
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, values={"hello.name": value})
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, [e["type"] for e in problem.ctx["errors"]]) == ("invalid_value", ("hello", "name"), ["set_has_no_order"])


def test_make_graph_refuses_a_type_mismatch_and_a_field_a_hook_does_not_give_at_the_call():
    @node("fan", title="Fan", description="", compute_outputs=lambda n: {f"o{i}": int for i in range(n)})
    def fan(n: int = 2) -> dict:
        return {f"o{i}": i for i in range(n)}

    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, edges=[("count.result", "hello.name")])
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("type_mismatch", ("hello", "name"))]
    with pytest.raises(InvalidCallError) as refused:
        make_graph(fan, _shout, edges=[("fan.o5", "shout.text")])
    assert [p.type for p in refused.value.errors()] == ["unknown_output"]


def test_every_code_make_graph_raises_from_the_load_is_a_catalogued_problem():
    from conductor._internal.errors import PROBLEM_CATALOGUE
    from conductor._internal.record import SAID_AT_THE_CALL

    assert SAID_AT_THE_CALL <= PROBLEM_CATALOGUE.keys()


def test_make_graph_writes_defaults_a_list_of_items_among_them():
    record = make_graph(_hello, defaults={"hello.name": ["you"]})
    assert record.nodes[0].defaults == {"name": ["you"]}
    assert PLAIN.load(record).run().outputs == {"hello.result": ["hello you"]}
    assert PLAIN.load(record).run(name=["ida", "bo"]).outputs == {"hello.result": ["hello ida", "hello bo"]}
    assert PLAIN.load(record).run(name="ida").outputs == {"hello.result": "hello ida"}


def test_make_graph_writes_each_node_at_its_version():
    def older(name: str) -> str:
        return name

    @node("v", title="Newer", description="", replaces=node("v", title="Older", description="")(older), upgrade=Upgrade())
    def _newer(name: str) -> str:
        return name

    assert make_graph(_newer).nodes[0].version == 2


def test_a_loaded_graph_keeps_its_own_definitions_whatever_the_registry_does_later():
    registry = registry_of(upper)
    graph = registry.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper")]))

    registry.remove("upper")

    assert graph.run(text="a").outputs == {"u.result": "A"}
    assert [p.type for p in registry.load(graph.record).problems] == ["unknown_node"]


def test_the_value_given_decides_one_or_many_and_each_item_is_checked_against_the_input():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper")]))

    assert graph.run(text="a").outputs == {"u.result": "A"}
    assert graph.run(text=["a"]).outputs == {"u.result": ["A"]}
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a", 3])
    assert [p.type for p in refused.value.errors()] == ["invalid_input"]


def test_a_list_typed_in_or_left_as_a_default_is_a_row_per_item_and_a_caller_may_give_one_value_instead():
    typed = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper", values={"text": ["x", "y"]})]))
    listed = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper", defaults={"text": ["a", "b"]})]))

    assert typed.run().outputs == {"u.result": ["X", "Y"]}
    assert _problems(GraphRecord(nodes=[NodeRecord(id="u", node="upper", values={"text": ["x", 3]})])) == [("invalid_value", ("u", "text"))]
    assert listed.plan["u"].mode == "per_row" and listed.inputs["text"].default == ["a", "b"]
    assert listed.run().outputs == {"u.result": ["A", "B"]}
    assert listed.run(text="c").outputs == {"u.result": "C"}


def test_a_default_on_a_later_node_is_a_problem_since_no_caller_reaches_it():
    record = GraphRecord(nodes=[NodeRecord(id="a", node="text"), NodeRecord(id="b", node="both", defaults={"no": "n"})], edges=[("a.result", "b.yes")])

    assert _problems(record) == [("default_on_later_node", ("b", "no"))]


def test_an_input_given_by_name_and_by_address_and_a_ceiling_below_one_are_refused_at_the_call():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper")]))

    with pytest.raises(InvalidCallError) as twice:
        graph.run({"text": "a", "u.text": "b"})
    with pytest.raises(InvalidCallError) as ceiling:
        graph.run(text="a", max_rows=0)
    with pytest.raises(InvalidCallError) as lengths_twice:
        graph.calls_for({"u.text": 5}, text=3)
    with pytest.raises(InvalidCallError) as both:
        graph.with_inputs({"u.text": ["b"]}, nope=1, text=["a"])

    assert [p.type for p in twice.value.errors()] == ["input_given_twice"]
    assert [p.type for p in lengths_twice.value.errors()] == ["input_given_twice"]
    assert sorted(p.type for p in both.value.errors()) == ["input_given_twice", "not_offered"]
    assert [p.type for p in ceiling.value.errors()] == ["invalid_max_rows"]
    with pytest.raises(InvalidCallError):
        graph.with_inputs(text="a").resume(max_rows=-1)


def test_a_stored_graph_with_problems_of_its_own_is_one_problem_where_it_is_used():
    broken = node("broken", title="Broken", description="")(GraphRecord(nodes=[NodeRecord(id="x", node="gone")]))
    graph = registry_of(upper, broken).load(GraphRecord(nodes=[NodeRecord(id="b", node="broken"), NodeRecord(id="u", node="upper")], edges=[("b.result", "u.text")]))

    (problem,) = graph.problems
    assert (problem.type, problem.loc, problem.ctx["graph"]) == ("embedded_graph_broken", ("b",), "broken")
    assert [p["type"] for p in problem.ctx["problems"]] == ["unknown_node"]


def test_an_edge_naming_a_column_a_table_does_not_declare_is_unknown_output_with_the_column():
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    class Row(TypedDict):
        name: str

    @node("rows", title="Rows", description="")
    def rows() -> Table[Row]:
        return pl.DataFrame({"name": ["a"]})

    graph = registry_of(rows, upper).load(GraphRecord(nodes=[NodeRecord(id="r", node="rows"), NodeRecord(id="u", node="upper")], edges=[("r.result.email", "u.text")]))

    (problem,) = graph.problems
    assert (problem.type, problem.ctx["column"]) == ("unknown_output", "email")


def test_a_series_of_tables_is_refused_where_it_is_declared():
    import polars as pl
    from conductor import NodeDeclarationError, Table

    with pytest.raises(NodeDeclarationError) as raised:

        @node("tables", title="Tables", description="")
        def tables(text: str) -> Series[Table]:
            return [pl.DataFrame()]

    assert raised.value.code == "series_of_tables"


# -- an embedded graph is one node from outside -------------------------------------


@node("tail", title="Tail", description="")
def tail(text: str) -> str:
    return text + "."


@node("split", title="Split", description="")
def split(text: str) -> Series[str]:
    return text.split(",")


ENRICH = node("enrich", title="Enrich", description="")(make_graph(upper, tail))
PAIR = node("pair", title="Pair", description="")(GraphRecord(nodes=[NodeRecord(id="x", node="upper"), NodeRecord(id="y", node="upper"), NodeRecord(id="b", node="both")],
                                edges=[("x.result", "b.yes"), ("y.result", "b.no")]))
GATHER = node("gather", title="Gather", description="")(GraphRecord(nodes=[NodeRecord(id="hold", node="upper"), NodeRecord(id="all", node="join")], edges=[("hold.result", "all.texts")]))
EMBEDDED = registry_of(upper, tail, split, join, both, ENRICH, PAIR, GATHER)


def test_an_embedding_is_filled_wired_and_read_by_its_own_id():
    graph = EMBEDDED.load({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}], "edges": [["e.result", "u.text"]]})

    assert graph.problems == [] and {n: i.address for n, i in graph.inputs.items()} == {"text": "e.text"}
    run = graph.run({"e.text": "hi"})
    assert run.outputs == {"u.result": "HI."} and set(run.results) == {"e", "u"} and run.results["e"] == {"result": "HI."}
    assert set(graph.plan) == {"e", "u"} and set(graph.describe().nodes) == {"e", "u"}
    assert [f.name for f in graph.describe().nodes["e"].inputs] == ["text"]


def test_a_wrong_address_on_an_embedding_is_a_problem_on_it_and_never_names_an_inner_node():
    graph = EMBEDDED.load({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}, {"id": "v", "node": "upper"}],
                           "edges": [["e.nope", "u.text"], ["v.result", "e.tail"], ["e/upper.result", "v.text"]]})

    found = sorted((p.type, p.loc) for p in graph.problems)
    assert found == [("edge_names_missing_node", ("v", "text")), ("unknown_input", ("e", "tail")), ("unknown_output", ("u", "text"))]
    assert not any("/" in str(p.loc) for p in graph.problems)


def test_inner_fields_sharing_a_name_are_offered_by_the_inner_node_as_its_record_says_it():
    graph = EMBEDDED.load({"nodes": [{"id": "s", "node": "upper"}, {"id": "c", "node": "pair"}], "edges": [["s.result", "c.x/text"]]})

    assert graph.problems == [] and {n: i.address for n, i in graph.inputs.items()} == {"text": "s.text", "y/text": "c.y/text"}
    assert graph.run({"s.text": "a", "c.y/text": "b"}).outputs == {"c.result": "AB"}


def test_a_graph_embedded_per_outer_row_folds_its_inner_reductions_per_outer_row():
    graph = EMBEDDED.load({"nodes": [{"id": "s", "node": "split"}, {"id": "g", "node": "gather"}], "edges": [["s.result", "g.text"]]})

    assert {k: v.msg for k, v in graph.plan.items()} == {"s": "once", "g": "per row of s.result"}
    assert graph.run(text="a,b,c").outputs == {"g.result": ["A", "B", "C"]}


def test_a_stored_graph_offers_its_fields_in_the_palette_of_the_registry_that_holds_its_nodes():
    described = next(n for n in EMBEDDED.describe().nodes if n.id == "pair")

    assert [f.name for f in described.inputs] == ["x/text", "y/text"] and [f.name for f in described.outputs] == ["result"]


def test_a_reduction_over_rows_a_branch_never_made_is_skipped_not_called_on_nothing():
    calls: list[list[str]] = []

    @node("gate", title="Gate", description="")
    def gate(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)

    @node("count-all", title="Count all", description="")
    def count_all(texts: Series[str]) -> str:
        calls.append(list(texts))
        return str(len(texts))

    graph = registry_of(gate, split, count_all).load({"nodes": [{"id": "g", "node": "gate"}, {"id": "s", "node": "split"}, {"id": "c", "node": "count-all"}],
                                                      "edges": [["g.no", "s.text"], ["s.result", "c.texts"]]})

    assert graph.run(text="yes").results["c"]["result"] is None and calls == []
    assert graph.run(text="a,b").outputs == {"c.result": "2"}


def test_a_skipped_item_of_a_series_output_is_a_gap_that_reads_back_from_json():
    @node("some", title="Some", description="")
    def some(text: str) -> Series[str]:
        return ["a", SKIPPED, "c"]

    graph = registry_of(some, upper).load(make_graph(some, upper))
    run = graph.run(text="x")

    assert run.results["some"]["result"] == ["a", None, "c"] and run.results["upper"]["result"] == ["A", None, "C"]
    assert graph.load_run(run.to_json()).results == run.results


def test_a_table_input_gets_its_declared_columns_and_a_source_lacking_one_is_refused_at_load():
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    class Names(TypedDict):
        name: str

    class Contacts(TypedDict):
        name: str
        email: str

    @node("contacts", title="Contacts", description="")
    def contacts() -> Table[Contacts]:
        return pl.DataFrame({"name": ["a"], "email": ["a@x"]})

    @node("names", title="Names", description="")
    def names() -> Table[Names]:
        return pl.DataFrame({"name": ["a"]})

    @node("columns", title="Columns", description="")
    def columns(table: Table[Names]) -> str:
        return ",".join(table.columns)

    @node("emails", title="Emails", description="")
    def emails(table: Table[Contacts]) -> str:
        return ",".join(table.columns)

    registry = registry_of(contacts, names, columns, emails)
    narrowed = registry.load({"nodes": [{"id": "c", "node": "contacts"}, {"id": "n", "node": "columns"}], "edges": [["c.result", "n.table"]]})
    short = registry.load({"nodes": [{"id": "n", "node": "names"}, {"id": "e", "node": "emails"}], "edges": [["n.result", "e.table"]]})

    assert narrowed.run().outputs == {"n.result": "name"}
    assert [(p.type, p.loc) for p in short.problems] == [("columns_missing", ("e", "table"))]


def test_a_node_gathering_two_series_of_different_depths_runs_above_the_shallower():
    @node("parts", title="Parts", description="")
    def parts(text: str) -> Series[str]:
        return text.split("|")

    @node("pair-lists", title="Pair lists", description="")
    def pair_lists(a: Series[str], b: Series[str]) -> str:
        return f"{len(a)}/{len(b)}"

    graph = registry_of(parts, split, pair_lists).load({
        "nodes": [{"id": "p", "node": "parts"}, {"id": "s", "node": "split"}, {"id": "x", "node": "pair-lists"}],
        "edges": [["p.result", "s.text"], ["p.result", "x.a"], ["s.result", "x.b"]],
    })

    assert graph.plan["x"].msg == "once" and graph.run(text="a,b|c").outputs == {"x.result": "2/3"}


def test_an_authored_id_with_a_slash_beside_an_embedding_is_its_own_problem_and_an_edge_into_an_embedding_says_its_outer_address():
    beside = EMBEDDED.load({"nodes": [{"id": "e", "node": "enrich"}, {"id": "e/upper", "node": "upper"}]})
    ghost = EMBEDDED.load({"nodes": [{"id": "e", "node": "enrich"}], "edges": [["ghost.result", "e.text"]]})

    assert [(p.type, p.loc) for p in beside.problems] == [("invalid_node_id", ("e/upper",))]
    assert [(p.type, p.loc, p.ctx["target"]) for p in ghost.problems] == [("edge_names_missing_node", ("e", "text"), "e.text")]


def test_a_gather_whose_rows_were_all_skipped_by_a_decision_on_its_own_row_is_skipped():
    """A per-document decision copied down onto the lines skips every line of that document, so the per-document join is skipped too."""

    @node("pair-two", title="Pair two", description="")
    def pair_two(a: str, b: str) -> str:
        return f"{a}:{b}"

    @node("decide-doc", title="Decide doc", description="")
    def decide_doc(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("x") else YesNo(SKIPPED, text)

    @node("docs", title="Docs", description="")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    graph = registry_of(docs, decide_doc, split, pair_two, join).load({
        "nodes": [{"id": "d", "node": "docs"}, {"id": "g", "node": "decide-doc"}, {"id": "l", "node": "split"}, {"id": "p", "node": "pair-two"}, {"id": "j", "node": "join"}],
        "edges": [["d.result", "g.text"], ["d.result", "l.text"], ["l.result", "p.a"], ["g.yes", "p.b"], ["p.result", "j.texts"]],
    })

    assert graph.run(text="x|y,no").results["j"]["result"] == ["x:x", None]


def test_a_gather_below_a_series_output_skipped_whole_is_skipped():
    @dataclass(frozen=True)
    class FullOrEmpty:
        full: Annotated[Series[str], Param(exclusive_group="size")]
        empty: Annotated[Series[str], Param(exclusive_group="size")]

    @node("decide-list", title="Decide list", description="")
    def decide_list(text: str) -> FullOrEmpty:
        return FullOrEmpty(text.split(","), SKIPPED) if text else FullOrEmpty(SKIPPED, [])

    graph = registry_of(decide_list, upper, join).load({
        "nodes": [{"id": "d", "node": "decide-list"}, {"id": "up", "node": "upper"}, {"id": "after", "node": "join"}],
        "edges": [["d.empty", "up.text"], ["up.result", "after.texts"]],
    })

    assert graph.run(text="a,b").results["after"]["result"] is None


def test_a_column_whose_name_has_a_dot_is_wired_by_its_address():
    import polars as pl
    from conductor import Table

    @node("sheet", title="Sheet", description="")
    def sheet() -> Table:
        return pl.DataFrame({"Amount in USD.": [3, 4]})

    @node("double", title="Double", description="")
    def double(amount: int) -> int:
        return amount * 2

    record = make_graph(sheet, double, edges=[("sheet.result.Amount in USD.", "double.amount")])
    assert registry_of(sheet, double).load(record).run().outputs == {"double.result": [6, 8]}


def test_a_node_display_is_the_hosts_own_kept_and_outside_the_fingerprint():
    plain = GraphRecord(nodes=[NodeRecord(id="h", node="hello")])
    shown = GraphRecord(nodes=[NodeRecord(id="h", node="hello", display={"x": 3})])
    assert GraphRecord.model_validate_json(shown.model_dump_json()) == shown
    assert PLAIN.load(plain).fingerprint == PLAIN.load(shown).fingerprint


def test_make_graph_refuses_a_value_beside_a_default_and_what_is_not_a_node():
    @node("count", title="Count", description="")
    def count(items: Series[str]) -> int:
        return len(items)

    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, count, edges=[], values={"hello.name": "a"}, defaults={"hello.name": "b"})
    assert sorted(p.type for p in refused.value.errors()) == ["value_and_default"]


def test_a_value_typed_in_from_python_is_its_json_so_the_record_equals_its_own_reload():
    import datetime
    import enum

    class Named:
        type_id: ClassVar[str] = "color"
        type_title: ClassVar[str] = "Color"

    class Color(Named, enum.Enum):
        RED = "red"

    @node("typed", title="Typed", description="")
    def typed(when: datetime.date, color: Color, pair: tuple[int, int]) -> str:
        return f"{when} {color.value} {pair[0]}"

    record = make_graph(typed, values={"typed.when": datetime.date(2026, 1, 2), "typed.color": Color.RED, "typed.pair": (1, 2)})
    assert record.nodes[0].values == {"when": "2026-01-02", "color": "red", "pair": [1, 2]}
    reloaded = GraphRecord.model_validate_json(record.model_dump_json())
    assert reloaded == record
    loaded, from_json = registry_of(typed).load(record), registry_of(typed).load(reloaded)
    assert loaded.fingerprint == from_json.fingerprint
    assert loaded.run().outputs == from_json.run().outputs == {"typed.result": "2026-01-02 red 1"}


def test_a_table_typed_in_is_refused_at_the_call_and_said_to_be_given_as_rows():
    import polars as pl

    @node("count", title="Count", description="")
    def count(rows: Table) -> int:
        return rows.height

    with pytest.raises(InvalidCallError) as refused:
        make_graph(count, values={"count.rows": pl.DataFrame({"a": [1]})})
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc) == ("invalid_value", ("count", "rows"))
    assert problem.ctx["reason"] == "a value typed into a graph is saved with it as JSON; give a table as a list of rows"


def test_a_record_holds_only_json_in_its_values_and_display():
    with pytest.raises(ValidationError):
        NodeRecord(id="a", node="a", display={"x": object()})
    with pytest.raises(ValidationError):
        NodeRecord(id="a", node="a", values={"x": object()})


def test_a_float_that_isnt_finite_typed_in_is_written_as_its_text_so_a_record_equals_its_json():
    from conductor import Ask

    @node("half", title="Half", description="")
    def half(n: float) -> float:
        return n / 2

    @node("ask", title="Ask", description="")
    def ask(x: float) -> str:
        raise Ask("ok?")

    registry = registry_of(half, ask)
    record = make_graph(half, values={"half.n": float("nan")})
    reloaded = GraphRecord.model_validate_json(record.model_dump_json())
    waiting = registry.load(make_graph(half, ask, values={"half.n": float("inf")}))

    assert record.nodes[0].values == {"n": "NaN"}
    assert reloaded == record and registry.load(reloaded).fingerprint == registry.load(record).fingerprint
    stored = registry.load(waiting.record.model_dump_json())
    assert stored.load_run(waiting.run().to_json()).status == "waiting"
    with pytest.raises(ValidationError):
        NodeRecord(id="x", node="half", display={"z": float("inf")})
    with pytest.raises(ValidationError):
        NodeRecord(id="x", node="half", values={"x": float("nan")})


def test_make_graph_lists_the_faults_it_finds_itself_beside_the_ones_loading_finds():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, edges=[("count.result", "hello.name"), ("count", "hello.name")], values={"nobody.x": 1})
    assert sorted(p.type for p in refused.value.errors()) == ["invalid_record", "node_not_in_graph", "type_mismatch"]


def test_make_graph_checks_each_node_against_the_definition_it_was_given():
    @node("v", title="V1", description="")
    def v1(text: str) -> str:
        return text

    @node("v", title="V2", description="", replaces=v1)
    def v2(text: str) -> str:
        return text

    with pytest.raises(InvalidCallError) as refused:
        make_graph(v2, v1, edges=[], values={"v-1.nope": 1})
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("unknown_input", ("v-1", "nope"))]

    @node("v", title="Other", description="")
    def other(n: int) -> int:
        return n

    with pytest.raises(NodeDeclarationError) as raised:
        make_graph(v1, other, edges=[])
    assert raised.value.code == "duplicate_node_id"
