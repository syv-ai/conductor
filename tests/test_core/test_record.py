"""The graph record: its shape, how it is saved, what loading it reports, and ``make_graph``.

A record is ``{"nodes": [...], "edges": [[from, to], ...]}``. ``registry.load``
reads it into a ``Graph``; what the record says that can't be meant is a
problem of that graph, never an exception.
"""

import json
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Problem,
    Series,
    Table,
    Upgrade,
    host_type,
    make_graph,
    node,
    stored_graph,
)
from pydantic import ValidationError

from tests.test_core.nodes import registry_of


@node("text", title="Text in", description="For a test.")
def text_in(value: str = "") -> str:
    return value


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class YesNo:
    yes: Annotated[str, Param(exclusive_group="answer")]
    no: Annotated[str, Param(exclusive_group="answer")]


@node("decide", title="Decide", description="For a test.")
def decide(text: str) -> YesNo:
    return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)


@node("both", title="Both", description="For a test.")
def both(yes: str, no: str) -> str:
    return yes + no


@dataclass(frozen=True)
class Two:
    zed: str
    alpha: str


@node("two", title="Two", description="For a test.")
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


def test_several_edges_into_one_input_that_can_all_hold_a_value_are_a_problem_naming_each():
    nodes = [NodeRecord(id="first", node="text", values={"value": "a"}), NodeRecord(id="second", node="text", values={"value": "b"}), NodeRecord(id="j", node="join")]
    graph = REGISTRY.load(GraphRecord(nodes=nodes, edges=[("second.result", "j.texts"), ("first.result", "j.texts")]))

    assert [(p.type, p.loc, p.ctx["sources"]) for p in graph.problems] == [("several_edges_into_one_input", ("j", "texts"), ["first.result", "second.result"])]


def test_a_repeated_edge_is_a_problem():
    record = GraphRecord(nodes=[NodeRecord(id="a", node="text"), NodeRecord(id="b", node="upper")], edges=[("a.result", "b.text"), ("a.result", "b.text")])

    assert _problems(record) == [("repeated_edge", ("b", "text"))]


def test_an_edge_naming_a_node_the_graph_does_not_have_is_one_problem_at_either_end():
    into = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="a", node="text")], edges=[("a.result", "gone.text")]))
    out_of = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="b", node="upper")], edges=[("gone.result", "b.text")]))

    assert [(p.type, p.loc, p.ctx["missing"]) for p in into.problems] == [("edge_names_missing_node", ("gone", "text"), "gone")]
    assert [(p.type, p.loc, p.ctx["missing"]) for p in out_of.problems] == [("edge_names_missing_node", ("b", "text"), "gone")]


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
    assert {name: (i.address, i.field.required) for name, i in graph.inputs.items()} == {"value": ("a.value", False), "yes": ("c.yes", True)}
    assert graph.outputs == ["b.result", "c.result"]


def test_a_value_closes_its_input_and_a_default_leaves_it_open_prefilled():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="b", node="both", values={"yes": "y"}, defaults={"no": "n"})]))

    assert list(graph.inputs) == ["no"] and not graph.inputs["no"].field.required and graph.inputs["no"].field.default == "n"
    assert graph.run().outputs == {"b.result": "yn"}
    assert graph.run(no="!").outputs == {"b.result": "y!"}


def test_two_open_inputs_of_one_name_are_called_by_address():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="a", node="upper"), NodeRecord(id="b", node="upper")]))

    assert list(graph.inputs) == ["a.text", "b.text"]
    assert graph.run({"a.text": "x", "b.text": "y"}).outputs == {"a.result": "X", "b.result": "Y"}


# -- a graph written from Python: make_graph -----------------------------------------------


@node("hello", title="Hello", description="For a test.")
def _hello(name: str) -> str:
    return f"hello {name}"


@node("shout", title="Shout", description="For a test.")
def _shout(text: str, times: int = 1) -> str:
    return text.upper() + "!" * times


@node("email", title="Email", description="For a test.")
def _email() -> str:
    return "ida@x"


@node("count", title="Count", description="For a test.")
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
    @node("hello-1", title="Other", description="For a test.")
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
    assert refused.value.errors()[0].ctx == {"node_id": "hello", "before": "count", "inputs": [], "outputs": ["result"]}


def test_make_graph_refuses_a_value_on_a_node_it_does_not_have():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, values={"nobody.name": "x"})
    assert [(p.type, p.ctx["nodes"]) for p in refused.value.errors()] == [("node_not_in_graph", ["hello"])]


def test_make_graph_writes_an_input_an_output_and_a_value_it_was_given_wrong_and_loading_reports_each():
    record = make_graph(
        _hello, _shout,
        edges=[("hello.reslt", "shout.text"), ("nobody.result", "shout.text"), ("hello.result", "shout.txt")],
        values={"hello.nmae": "x"}, defaults={"hello.name": 3},
    )
    assert sorted((p.type, p.loc) for p in PLAIN.load(record).problems) == [
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


def test_make_graph_writes_a_type_mismatch_and_a_field_a_hook_does_not_give_and_loading_reports_them():
    @node("fan", title="Fan", description="For a test.", compute_outputs=lambda n: {f"o{i}": int for i in range(n)})
    def fan(n: int = 2) -> dict:
        return {f"o{i}": i for i in range(n)}

    mismatched = make_graph(_count, _hello, edges=[("count.result", "hello.name")])
    assert [(p.type, p.loc) for p in PLAIN.load(mismatched).problems] == [("type_mismatch", ("hello", "name"))]
    not_given = make_graph(fan, _shout, edges=[("fan.o5", "shout.text")])
    assert [p.type for p in registry_of(fan, _shout).load(not_given).problems] == ["unknown_output"]


def test_make_graph_writes_defaults_and_a_list_for_an_input_of_one_value_is_reported_when_it_loads():
    record = make_graph(_hello, defaults={"hello.name": "you"})
    assert record.nodes[0].defaults == {"name": "you"}
    assert PLAIN.load(record).run().outputs == {"hello.result": "hello you"}
    assert PLAIN.load(record).run(name="ida").outputs == {"hello.result": "hello ida"}
    listed = make_graph(_hello, defaults={"hello.name": ["you"]})
    assert [(p.type, p.loc) for p in PLAIN.load(listed).problems] == [("invalid_value", ("hello", "name"))]


def test_make_graph_writes_each_node_at_its_version():
    def older(name: str) -> str:
        return name

    @node("v", title="Newer", description="For a test.", replaces=node("v", title="Older", description="For a test.")(older), upgrade=Upgrade())
    def _newer(name: str) -> str:
        return name

    assert make_graph(_newer).nodes[0].version == 2


def test_a_loaded_graph_keeps_its_own_definitions_whatever_the_registry_does_later():
    registry = registry_of(upper)
    graph = registry.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper")]))

    registry.remove("upper")

    assert graph.run(text="a").outputs == {"u.result": "A"}
    assert [p.type for p in registry.load(graph.record).problems] == ["unknown_node"]


def test_the_type_decides_one_or_many_so_a_list_given_for_an_input_of_one_value_is_refused():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="j", node="join")], edges=[("u.result", "j.texts")]))

    assert graph.runs_on["u"].mode == "once" and graph.run(text="a").outputs == {"j.result": "A"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a"])
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_input", ("inputs", "text"))]


def test_a_list_typed_in_or_left_as_a_default_for_an_input_of_one_value_is_not_a_value_of_its_type():
    assert _problems(GraphRecord(nodes=[NodeRecord(id="u", node="upper", values={"text": ["x", "y"]})])) == [("invalid_value", ("u", "text"))]
    assert _problems(GraphRecord(nodes=[NodeRecord(id="u", node="upper", defaults={"text": ["a", "b"]})])) == [("invalid_value", ("u", "text"))]


def test_a_default_on_a_later_node_is_a_problem_since_no_caller_reaches_it():
    record = GraphRecord(nodes=[NodeRecord(id="a", node="text"), NodeRecord(id="b", node="both", defaults={"no": "n"})], edges=[("a.result", "b.yes")])

    assert _problems(record) == [("default_on_later_node", ("b", "no"))]


def test_an_input_given_by_name_and_by_address_and_a_ceiling_below_one_are_refused_at_the_call():
    graph = REGISTRY.load(GraphRecord(nodes=[NodeRecord(id="u", node="upper")]))

    with pytest.raises(InvalidCallError) as twice:
        graph.run({"text": "a", "u.text": "b"})
    with pytest.raises(InvalidCallError) as ceiling:
        graph.run(text="a", max_items=0)
    with pytest.raises(InvalidCallError) as both:
        graph.with_inputs({"u.text": "b"}, nope=1, text="a")

    assert [p.type for p in twice.value.errors()] == ["input_given_twice"]
    assert sorted(p.type for p in both.value.errors()) == ["input_given_twice", "not_offered"]
    assert [p.type for p in ceiling.value.errors()] == ["invalid_max_items"]
    with pytest.raises(InvalidCallError):
        graph.with_inputs(text="a").with_max_items(-1)


def test_a_stored_graph_with_problems_of_its_own_is_one_problem_where_it_is_used():
    broken = stored_graph("broken", title="Broken", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="x", node="gone")]))
    graph = registry_of(upper, broken).load(GraphRecord(nodes=[NodeRecord(id="b", node="broken"), NodeRecord(id="u", node="upper")], edges=[("b.result", "u.text")]))

    assert [(problem.type, problem.loc) for problem in graph.problems] == [("unknown_node", ("b", "x"))]


def _a_table_read_by_one_text(source: str) -> Problem:
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    class Row(TypedDict):
        name: str

    @node("rows", title="Rows", description="For a test.")
    def rows() -> Table[Row]:
        return pl.DataFrame({"name": ["a"]})

    graph = registry_of(rows, upper).load(GraphRecord(nodes=[NodeRecord(id="r", node="rows"), NodeRecord(id="u", node="upper")], edges=[(source, "u.text")]))
    (problem,) = graph.problems
    return problem


def test_an_edge_naming_a_column_of_a_whole_table_is_unknown_output_as_a_table_is_one_value():
    problem = _a_table_read_by_one_text("r.result.name")

    assert (problem.type, problem.loc, problem.ctx["record_field"]) == ("unknown_output", ("u", "text"), "name")


@pytest.mark.parametrize("source", ["r.result.name", "r.result"])
def test_a_table_wired_where_one_value_is_read_says_to_split_it_first(source):
    assert _a_table_read_by_one_text(source).ctx["hint"] == "split_first"


def test_a_series_of_tables_is_accepted_where_it_is_declared():
    import polars as pl
    from conductor import Table

    @node("tables", title="Tables", description="For a test.")
    def tables(text: str) -> Series[Table]:
        return [pl.DataFrame()]

    assert tables.outputs["result"].many is True
    assert tables.outputs["result"].type.id.startswith("Table")


# -- a subgraph is one node from outside -------------------------------------


@node("tail", title="Tail", description="For a test.")
def tail(text: str) -> str:
    return text + "."


@node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",")


ENRICH = stored_graph("enrich", title="Enrich", description="For a test.")(make_graph(upper, tail))
PAIR = stored_graph("pair", title="Pair", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="x", node="upper"), NodeRecord(id="y", node="upper"), NodeRecord(id="b", node="both")],
                                edges=[("x.result", "b.yes"), ("y.result", "b.no")]))
GATHER = stored_graph("gather", title="Gather", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="hold", node="upper"), NodeRecord(id="all", node="join")], edges=[("hold.result", "all.texts")]))
EMBEDDED = registry_of(upper, tail, split, join, both, ENRICH, PAIR, GATHER)


def test_an_embedding_is_filled_wired_and_read_by_its_own_id():
    graph = EMBEDDED.load(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}], "edges": [["e.tail.result", "u.text"]]}))

    assert graph.problems == [] and {n: i.address for n, i in graph.inputs.items()} == {"text": "e.text"}
    run = graph.run({"e.text": "hi"})
    assert run.outputs == {"u.result": "HI."} and set(run.results) == {"e", "u"} and run.results["e"] == {"tail.result": "HI."}
    assert set(graph.runs_on) == {"e", "u"} and set(graph.describe().nodes) == {"e", "u"}
    assert [f.field.name for f in graph.describe().nodes["e"].inputs] == ["text"]


def test_a_wrong_address_on_an_embedding_is_a_problem_on_it_and_never_names_an_inner_node():
    graph = EMBEDDED.load(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}, {"id": "v", "node": "upper"}],
                           "edges": [["e.nope", "u.text"], ["v.result", "e.tail"], ["e/upper.result", "v.text"]]}))

    found = sorted((p.type, p.loc) for p in graph.problems)
    assert found == [("edge_names_missing_node", ("v", "text")), ("unknown_input", ("e", "tail")), ("unknown_output", ("u", "text"))]
    assert not any("/" in str(p.loc) for p in graph.problems)


def test_inner_fields_sharing_a_name_are_offered_by_the_inner_node_as_its_record_says_it():
    first = EMBEDDED.load(GraphRecord.model_validate({"nodes": [{"id": "c", "node": "pair"}]}))
    later = EMBEDDED.load(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "upper"}, {"id": "c", "node": "pair"}], "edges": [["s.result", "c.x.text"]]}))

    assert first.problems == [] and {n: i.address for n, i in first.inputs.items()} == {"x.text": "c.x.text", "y.text": "c.y.text"}
    assert [(p.type, p.loc) for p in later.problems] == [("unbound_on_later_node", ("c", "y.text"))]  # an embedding is fed as any later node is


def test_inner_fields_sharing_a_name_are_filled_by_the_names_the_embedding_offers():
    graph = EMBEDDED.load(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "upper"}, {"id": "c", "node": "pair", "values": {"y.text": "b"}}], "edges": [["s.result", "c.x.text"]]}))

    assert graph.problems == [] and {n: i.address for n, i in graph.inputs.items()} == {"text": "s.text"}
    assert graph.run(text="a").outputs == {"c.b.result": "AB"}


def test_a_stored_graph_offers_its_fields_in_the_palette_of_the_registry_that_holds_its_nodes():
    described = next(n for n in EMBEDDED.describe().nodes if n.id == "pair")

    assert [f.name for f in described.inputs] == ["x.text", "y.text"] and [f.name for f in described.outputs] == ["b.result"]


def test_a_gather_of_a_series_a_branch_never_made_is_skipped_not_called_on_nothing():
    calls: list[list[str]] = []

    @node("gate", title="Gate", description="For a test.")
    def gate(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)

    @node("count-all", title="Count all", description="For a test.")
    def count_all(texts: Series[str]) -> str:
        calls.append(list(texts))
        return str(len(texts))

    graph = registry_of(gate, split, count_all).load(GraphRecord.model_validate({"nodes": [{"id": "g", "node": "gate"}, {"id": "s", "node": "split"}, {"id": "c", "node": "count-all"}],
                                                      "edges": [["g.no", "s.text"], ["s.result", "c.texts"]]}))

    assert graph.run(text="yes").results["c"]["result"] is None and calls == []
    assert graph.run(text="a,b").outputs == {"c.result": "2"}


def test_a_skipped_item_of_a_series_output_is_a_gap_that_reads_back_from_json():
    @node("some", title="Some", description="For a test.")
    def some(text: str) -> Series[str]:
        return ["a", SKIPPED, "c"]

    graph = registry_of(some, upper).load(make_graph(some, upper))
    run = graph.run(text="x")

    assert run.results["some"]["result"] == ["a", None, "c"] and run.results["upper"]["result"] == ["A", None, "C"]
    assert graph.load_run(run.to_json()).results == run.results


def test_a_table_input_gets_its_declared_columns():
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    class Names(TypedDict):
        name: str

    class Contacts(TypedDict):
        name: str
        email: str

    @node("contacts", title="Contacts", description="For a test.")
    def contacts() -> Table[Contacts]:
        return pl.DataFrame({"name": ["a"], "email": ["a@x"]})

    @node("columns", title="Columns", description="For a test.")
    def columns(table: Table[Names]) -> str:
        return ",".join(table.columns)

    registry = registry_of(contacts, columns)
    narrowed = registry.load(GraphRecord.model_validate({"nodes": [{"id": "c", "node": "contacts"}, {"id": "n", "node": "columns"}], "edges": [["c.result", "n.table"]]}))

    assert registry.load(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "columns"}]})).run(table=pl.DataFrame({"name": ["a"], "email": ["a@x"]})).outputs == {"n.result": "name"}
    assert narrowed.run().outputs == {"n.result": "name"}


def test_a_node_gathering_two_series_of_different_depths_runs_per_item_of_the_shallower_and_reads_it_whole_there():
    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Series[str]:
        return text.split("|")

    @node("pair-lists", title="Pair lists", description="For a test.")
    def pair_lists(a: Series[str], b: Series[str]) -> str:
        return f"{len(a)}/{len(b)}"

    graph = registry_of(parts, split, pair_lists).load(GraphRecord.model_validate({
        "nodes": [{"id": "p", "node": "parts"}, {"id": "s", "node": "split"}, {"id": "x", "node": "pair-lists"}],
        "edges": [["p.result", "s.text"], ["p.result", "x.a"], ["s.result", "x.b"]],
    }))

    assert (graph.runs_on["x"].mode, graph.runs_on["x"].series) == ("per_item", "p.result")  # a gather goes up one level
    assert graph.run(text="a,b|c").outputs == {"x.result": ["2/2", "2/1"]}


def test_an_authored_id_with_a_slash_beside_an_embedding_is_its_own_problem_and_an_edge_into_an_embedding_says_its_outer_address():
    beside = EMBEDDED.load(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}, {"id": "e/upper", "node": "upper"}]}))
    ghost = EMBEDDED.load(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}], "edges": [["ghost.result", "e.text"]]}))

    assert [(p.type, p.loc) for p in beside.problems] == [("invalid_node_id", ("e/upper",))]
    assert [(p.type, p.loc, p.ctx["target"]) for p in ghost.problems] == [("edge_names_missing_node", ("e", "text"), "e.text")]


def test_a_gather_whose_items_were_all_skipped_by_a_decision_above_them_is_skipped():
    """A per-document decision read by every line of the document skips each of them, so the per-document join is skipped too."""

    @node("pair-two", title="Pair two", description="For a test.")
    def pair_two(a: str, b: str) -> str:
        return f"{a}:{b}"

    @node("decide-doc", title="Decide doc", description="For a test.")
    def decide_doc(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("x") else YesNo(SKIPPED, text)

    @node("docs", title="Docs", description="For a test.")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    graph = registry_of(docs, decide_doc, split, pair_two, join).load(GraphRecord.model_validate({
        "nodes": [{"id": "d", "node": "docs"}, {"id": "g", "node": "decide-doc"}, {"id": "l", "node": "split"}, {"id": "p", "node": "pair-two"}, {"id": "j", "node": "join"}],
        "edges": [["d.result", "g.text"], ["d.result", "l.text"], ["l.result", "p.a"], ["g.yes", "p.b"], ["p.result", "j.texts"]],
    }))

    assert graph.run(text="x|y,no").results["j"]["result"] == ["x:x", None]


def test_a_gather_below_a_series_output_skipped_whole_is_skipped():
    @dataclass(frozen=True)
    class FullOrEmpty:
        full: Annotated[Series[str], Param(exclusive_group="size")]
        empty: Annotated[Series[str], Param(exclusive_group="size")]

    @node("decide-list", title="Decide list", description="For a test.")
    def decide_list(text: str) -> FullOrEmpty:
        return FullOrEmpty(text.split(","), SKIPPED) if text else FullOrEmpty(SKIPPED, [])

    graph = registry_of(decide_list, upper, join).load(GraphRecord.model_validate({
        "nodes": [{"id": "d", "node": "decide-list"}, {"id": "up", "node": "upper"}, {"id": "after", "node": "join"}],
        "edges": [["d.empty", "up.text"], ["up.result", "after.texts"]],
    }))

    assert graph.run(text="a,b").results["after"]["result"] is None


def test_a_records_field_whose_name_has_a_dot_is_wired_by_its_address():
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    Amount = TypedDict("Amount", {"Amount in USD.": int})  # noqa: UP013

    @node("sheet", title="Sheet", description="For a test.")
    def sheet() -> Table[Amount]:  # type: ignore[valid-type]
        return pl.DataFrame({"Amount in USD.": [3, 4]})

    @node("each", title="Each", description="For a test.")
    def each(table: Table[Amount]) -> Series[Amount]:  # type: ignore[valid-type]
        return table.to_dicts()

    @node("double", title="Double", description="For a test.")
    def double(amount: int) -> int:
        return amount * 2

    record = make_graph(sheet, each, double, edges=[("sheet.result", "each.table"), ("each.result.Amount in USD.", "double.amount")])
    assert registry_of(sheet, each, double).load(record).run().outputs == {"double.result": [6, 8]}


def test_a_node_display_is_the_hosts_own_kept_and_outside_the_fingerprint():
    plain = GraphRecord(nodes=[NodeRecord(id="h", node="hello")])
    shown = GraphRecord(nodes=[NodeRecord(id="h", node="hello", display={"x": 3})])
    assert GraphRecord.model_validate_json(shown.model_dump_json()) == shown
    assert PLAIN.load(plain).fingerprint == PLAIN.load(shown).fingerprint


@pytest.mark.parametrize("name", [name for name in NodeRecord.model_fields if name != "display"])
def test_every_field_of_a_node_but_its_display_is_in_the_fingerprint(name: str):
    node_record = NodeRecord(id="h", node="hello")
    held = getattr(node_record, name)
    changed = held + 1 if isinstance(held, int) else f"{held}x" if isinstance(held, str) else {"name": "changed"}
    other = node_record.model_copy(update={name: changed})
    assert PLAIN.load(GraphRecord(nodes=[node_record])).fingerprint != PLAIN.load(GraphRecord(nodes=[other])).fingerprint


def test_a_graphs_fingerprint_is_the_one_its_saved_runs_name():
    record = GraphRecord(nodes=[NodeRecord(id="h", node="hello", values={"name": "Ida"}, display={"x": 3}), NodeRecord(id="s", node="shout")], edges=[("h.result", "s.text")])
    assert PLAIN.load(record).fingerprint == "13cacdb319660389"


def test_make_graph_writes_a_value_beside_a_default_and_loading_reports_it():
    record = make_graph(_hello, values={"hello.name": "a"}, defaults={"hello.name": "b"})

    assert (record.nodes[0].values, record.nodes[0].defaults) == ({"name": "a"}, {"name": "b"})
    assert [(p.type, p.loc) for p in PLAIN.load(record).problems] == [("value_and_default", ("hello", "name"))]


def test_a_value_typed_in_from_python_is_its_json_so_the_record_equals_its_own_reload():
    import datetime
    import enum

    @host_type("color", title="Color")
    class Color(enum.Enum):
        RED = "red"

    @node("typed", title="Typed", description="For a test.")
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

    @node("count", title="Count", description="For a test.")
    def count(rows: Table) -> int:
        return rows.height

    with pytest.raises(InvalidCallError) as refused:
        make_graph(count, values={"count.rows": pl.DataFrame({"a": [1]})})
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc) == ("invalid_value", ("count", "rows"))
    assert problem.ctx["reason"] == "a value typed into a graph is saved with it as JSON; give a table as a list of records"


def test_a_record_holds_only_json_in_its_values_and_display():
    with pytest.raises(ValidationError):
        NodeRecord(id="a", node="a", display={"x": object()})
    with pytest.raises(ValidationError):
        NodeRecord(id="a", node="a", values={"x": object()})


def test_a_float_that_isnt_finite_typed_in_is_written_as_its_text_so_a_record_equals_its_json():
    from conductor import Ask

    @node("half", title="Half", description="For a test.")
    def half(n: float) -> float:
        return n / 2

    @node("ask", title="Ask", description="For a test.")
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


def test_make_graph_raises_every_fault_of_its_own_and_none_that_loading_finds():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, edges=[("count.result", "hello.name"), ("count", "hello.name")], values={"nobody.x": 1})
    assert sorted(p.type for p in refused.value.errors()) == ["invalid_record", "node_not_in_graph"]


def test_make_graph_writes_each_node_by_the_id_and_version_of_the_definition_given_and_loads_none_of_them():
    @node("v", title="V1", description="For a test.")
    def v1(text: str) -> str:
        return text

    @node("v", title="V2", description="For a test.", replaces=v1)
    def v2(text: str) -> str:
        return text

    record = make_graph(v2, v1, edges=[], values={"v-1.nope": 1})
    assert [(n.id, n.node, n.version) for n in record.nodes] == [("v-1", "v", 2), ("v-2", "v", 1)]
    assert [(p.type, p.loc) for p in registry_of(v2).load(record).problems] == [("unknown_input", ("v-1", "nope"))]

    @node("v", title="Other", description="For a test.")
    def other(n: int) -> int:
        return n

    unrelated = make_graph(v1, other, edges=[])
    assert [(n.id, n.node, n.version) for n in unrelated.nodes] == [("v-1", "v", 1), ("v-2", "v", 1)]


def test_an_edge_that_is_no_address_is_told_the_two_forms_an_address_has():
    """A third part names a field of a record, which is what loading takes; a table has no field to wire."""
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, edges=[("count", "hello.name")])
    [problem] = refused.value.errors()
    assert problem.type == "invalid_record" and "'node.field'" in problem.msg and "'node.output.field' for one field of a record" in problem.msg
    assert "column" not in problem.msg


def test_make_graph_says_an_edge_that_is_no_address_as_loading_the_same_record_says_it():
    with pytest.raises(InvalidCallError) as written:
        make_graph(_count, _hello, edges=[("count.result", "hello.name"), ("count", "hello.name")])
    with pytest.raises(InvalidCallError) as loaded:
        PLAIN.load(json.dumps({"nodes": [{"id": "count", "node": "count"}, {"id": "hello", "node": "hello"}], "edges": [["count.result", "hello.name"], ["count", "hello.name"]]}))

    assert written.value.errors() == loaded.value.errors()
    assert [(p.loc, p.ctx["at"]) for p in written.value.errors()] == [(("record", "edges", 1, 0), "edges.1.0")]
