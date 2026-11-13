"""Which loop each node of a compiled graph is inside, decided from the declared types alone.

An output declared ``Series[str]`` wired to an input that takes one ``str`` starts a loop: that node, and every node
downstream of it, is called once per item. An input declared ``Series`` or ``Table`` collects the loop, and its node
is called once. An output from outside a loop that a node inside it reads is an outer value, the same for every item.
In ``documents -> upper -> count``, ``upper`` is called once per document and ``count`` once, with every result.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import pytest
from conductor import (
    SKIPPED,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Param,
    Series,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)


class Contact(TypedDict):
    name: str
    email: str | None


class Named(TypedDict):
    name: str


class Aged(TypedDict):
    name: str
    age: int


def _node(node_id: str):
    return node(node_id, title=node_id.capitalize(), description=f"The {node_id} node.")


@_node("contacts")
def contacts(path: str) -> Table[Contact]: ...


@_node("rows")
def rows(table: Table[Contact]) -> Series[Contact]: ...


@_node("send")
def send(email: str) -> bool: ...


@_node("collect")
def collect(records: Table[Named], sent: Series[bool | None]) -> Table: ...


@_node("names")
def names(table: Table[Named]) -> int: ...


@_node("ages")
def ages(table: Table[Aged]) -> int: ...


@_node("documents")
def documents(folder: str) -> Series[str]: ...


@_node("lines")
def lines(document: str) -> Series[str]: ...


@_node("count")
def count(items: Series[str]) -> int: ...


@_node("total")
def total(counts: Series[int]) -> int: ...


@_node("pair")
def pair(a: str, b: str) -> str: ...


@_node("pair-or-alone")
def pair_or_alone(a: str, b: str | None) -> str: ...


@_node("upper")
def upper(text: str) -> str: ...


@_node("both")
def both(first: Series[str], second: Series[str]) -> int: ...


@dataclass(frozen=True)
class Way:
    yes: Annotated[str, Param(exclusive_group="way")] = SKIPPED
    no: Annotated[str, Param(exclusive_group="way")] = SKIPPED


@_node("decide")
def decide(text: str) -> Way: ...


@dataclass(frozen=True)
class Size:
    small: Annotated[str, Param(exclusive_group="size")] = SKIPPED
    large: Annotated[str, Param(exclusive_group="size")] = SKIPPED


@_node("measure")
def measure(text: str) -> Size: ...


@_node("anything")
def anything(value: Any) -> Any: ...


@_node("height")
def height(table: Table) -> int: ...


@_node("sheets")
def sheets(folder: str) -> Series[Table[Contact]]: ...


@_node("sheet")
def sheet(text: str) -> Table[Contact]: ...


@_node("stacked")
def stacked(tables: Series[Table[Contact]]) -> Table[Contact]: ...


@_node("mappings")
def mappings(folder: str) -> Series[dict[str, int]]: ...


@_node("unknowns")
def unknowns(folder: str) -> Series[Any]: ...


@host_type("person", title="Person")
@dataclass(frozen=True)
class Person:
    name: str


@_node("people")
def people(folder: str) -> Series[Person]: ...


REGISTRY = NodeRegistry()
for _definition in (
    contacts, rows, send, collect, names, ages, documents, lines, count, total, pair, pair_or_alone, upper, both, decide, measure, anything,
    height, sheets, sheet, stacked, mappings, unknowns, people,
):
    REGISTRY.add(_definition)


def record_of(nodes: Mapping[str, str], edges: Sequence[tuple[str, str]], registry: NodeRegistry = REGISTRY, **given: Any) -> GraphRecord:
    """``make_graph`` of ``nodes`` (node id to node definition id in ``registry``) and ``edges``
    (``("node.output", "node.input")``, or ``("node.output.field", "node.input")`` for one field of a record);
    ``given`` are its ``constants``, ``inputs`` and ``outputs``."""
    return make_graph(*((node_id, registry[name]) for node_id, name in nodes.items()), edges=list(edges), **given)


def compile_graph(nodes: Mapping[str, str], edges: Sequence[tuple[str, str]], registry: NodeRegistry = REGISTRY, **given: Any):
    """The compiled graph of ``record_of(nodes, edges, registry, **given)``."""
    return Graph.compile(record_of(nodes, edges, registry, **given), registry)


def input_sources(graph, node_id: str) -> dict[str, tuple[str, str | None]]:
    """Each input of ``node_id`` with where its value comes from, as ``graph.describe()`` says it: for example
    ``{"text": ("item", "d.result")}`` for one item of ``d.result``, ``("collected", "d.result")`` for the whole loop."""
    return {i.field.name: (i.source.mode, i.source.series) for i in graph.describe().nodes[node_id].inputs}


def problems(graph) -> list[tuple[str, tuple]]:
    return [(p.type, p.loc) for p in graph.problems]


# -- starting, reading into and collecting a loop --------------------------------------------------


def test_a_series_wired_to_an_input_of_one_value_starts_a_loop_that_its_downstream_nodes_are_inside():
    graph = compile_graph({"d": "documents", "u": "upper", "v": "upper"}, [("d.result", "u.text"), ("u.result", "v.text")])
    assert graph.is_runnable
    assert {n: (r.mode, r.series) for n, r in graph.runs_on.items()} == {"d": ("once", None), "u": ("per_item", "d.result"), "v": ("per_item", "d.result")}
    assert input_sources(graph, "u") == {"text": ("item", "d.result")}


def test_a_node_inside_a_loop_gets_an_output_from_outside_it_as_an_outer_value():
    graph = compile_graph({"d": "documents", "u": "upper", "p": "pair"}, [("d.result", "p.a"), ("u.result", "p.b")])
    assert graph.runs_on["p"].series == "d.result"
    assert input_sources(graph, "p") == {"a": ("item", "d.result"), "b": ("outer_value", None)}


def test_a_series_input_collects_the_loop_and_its_node_is_called_once_outside_it():
    graph = compile_graph({"d": "documents", "u": "upper", "c": "count"}, [("d.result", "u.text"), ("u.result", "c.items")])
    assert graph.runs_on["c"].mode == "once"
    assert input_sources(graph, "c") == {"items": ("collected", "d.result")}


def test_nested_loops_are_collected_one_level_at_a_time():
    graph = compile_graph(
        {"d": "documents", "l": "lines", "u": "upper", "c": "count", "t": "total"},
        [("d.result", "l.document"), ("l.result", "u.text"), ("u.result", "c.items"), ("c.result", "t.counts")],
    )
    assert graph.is_runnable
    assert {n: r.series for n, r in graph.runs_on.items()} == {"d": None, "l": "d.result", "u": "l.result", "c": "d.result", "t": None}
    assert input_sources(graph, "c") == {"items": ("collected", "l.result")} and input_sources(graph, "t") == {"counts": ("collected", "d.result")}


def test_a_series_wired_to_a_series_input_is_collected_like_the_outputs_of_its_own_loop():
    graph = compile_graph({"d": "documents", "u": "upper", "b": "both"}, [("d.result", "u.text"), ("d.result", "b.first"), ("u.result", "b.second")])
    assert graph.is_runnable and graph.runs_on["b"].mode == "once"
    assert input_sources(graph, "b") == {"first": ("collected", "d.result"), "second": ("collected", "d.result")}


def test_an_input_wired_to_the_outer_series_of_a_nested_loop_gets_its_item_as_an_outer_value():
    graph = compile_graph(
        {"d": "documents", "l": "lines", "c": "count", "p": "pair", "u": "upper"},
        [("d.result", "l.document"), ("l.result", "c.items"), ("l.result", "p.a"), ("d.result", "p.b")],
        outputs={"count": "c.result", "pair": "p.result", "upper": "u.result"},
    )
    assert graph.is_runnable
    assert input_sources(graph, "p") == {"a": ("item", "l.result"), "b": ("outer_value", "d.result")}


# -- what is refused -------------------------------------------------------------------------------


def test_a_node_wired_to_the_items_of_two_series_that_are_not_nested_is_refused():
    graph = compile_graph({"a": "documents", "b": "documents", "p": "pair"}, [("a.result", "p.a"), ("b.result", "p.b")], inputs={"folder": ["a.folder", "b.folder"]})
    assert problems(graph) == [("series_not_nested", ("p",))]
    assert graph.problems[0].ctx["series"] == ["a.result", "b.result"]


def test_two_nodes_wired_to_one_series_are_inside_the_same_loop():
    graph = compile_graph({"d": "documents", "u": "upper", "v": "upper", "p": "pair"}, [("d.result", "u.text"), ("d.result", "v.text"), ("u.result", "p.a"), ("v.result", "p.b")])
    assert graph.is_runnable and graph.runs_on["p"].series == "d.result"


def test_a_series_input_gets_the_one_value_of_a_source_that_runs_once_as_a_series_of_one():
    graph = compile_graph({"u": "upper", "c": "count"}, [("u.result", "c.items")])
    assert graph.is_runnable and graph.runs_on["c"].mode == "once"
    assert input_sources(graph, "c") == {"items": ("collected", None)}


def test_a_list_constant_on_an_input_of_one_value_is_an_invalid_value():
    graph = compile_graph({"u": "upper"}, [], constants={"u.text": ["a", "b"]})
    assert problems(graph) == [("invalid_value", ("u", "text"))]


def test_a_series_of_the_wrong_item_type_is_a_type_mismatch_between_the_two_series_types():
    graph = compile_graph({"d": "documents", "t": "total"}, [("d.result", "t.counts")])
    assert (graph.problems[0].type, graph.problems[0].ctx["got"], graph.problems[0].ctx["expected"]) == ("type_mismatch", "Series[str]", "Series[int]")


# -- a table is one value --------------------------------------------------------------------------


def test_a_table_is_passed_whole_and_starts_no_loop():
    graph = compile_graph({"c": "contacts", "n": "names"}, [("c.result", "n.table")])
    assert graph.is_runnable and graph.runs_on["n"].mode == "once" and input_sources(graph, "n") == {"table": ("one_value", None)}


def test_a_table_fits_a_table_input_by_its_columns_and_the_mismatch_names_both():
    graph = compile_graph({"c": "contacts", "a": "ages"}, [("c.result", "a.table")])
    assert problems(graph) == [("type_mismatch", ("a", "table"))]
    assert graph.problems[0].ctx["got"] == "Table[name: str, email: str | None]"
    assert graph.problems[0].ctx["expected"] == "Table[name: str, age: int]"


def test_a_table_on_an_input_of_one_text_is_a_type_mismatch():
    graph = compile_graph({"c": "contacts", "u": "upper"}, [("c.result", "u.text")])
    assert problems(graph) == [("type_mismatch", ("u", "text"))]


def test_an_input_of_any_type_takes_a_table_as_the_one_value_it_is():
    graph = compile_graph({"c": "contacts", "a": "anything"}, [("c.result", "a.value")])
    assert graph.is_runnable and graph.runs_on["a"].mode == "once"


def test_a_table_input_collects_records_in_step_with_a_series_input_from_the_same_loop():
    graph = compile_graph(
        {"c": "contacts", "r": "rows", "s": "send", "k": "collect"},
        [("c.result", "r.table"), ("r.result.email", "s.email"), ("r.result", "k.records"), ("s.result", "k.sent")],
    )
    assert graph.is_runnable
    assert graph.runs_on["s"].series == "r.result" and graph.runs_on["k"].mode == "once"
    assert input_sources(graph, "k") == {"records": ("collected", "r.result"), "sent": ("collected", "r.result")}


@pytest.mark.parametrize("source", ["rows", "mappings", "unknowns"], ids=["records of declared columns", "mappings", "items of any type"])
def test_a_table_input_of_any_columns_collects_a_series_whose_items_can_be_records(source: str):
    graph = compile_graph({"c": "contacts", "s": source, "h": "height"}, [*([("c.result", "s.table")] if source == "rows" else []), ("s.result", "h.table")],
                          outputs={"height": "h.result"})
    assert graph.is_runnable and graph.runs_on["h"].mode == "once" and input_sources(graph, "h") == {"table": ("collected", "s.result")}


@pytest.mark.parametrize(
    ("nodes", "edges", "got"),
    [
        ({"s": "upper"}, [("s.result", "h.table")], "str"),
        ({"s": "documents"}, [("s.result", "h.table")], "Series[str]"),
        ({"d": "documents", "s": "upper"}, [("d.result", "s.text"), ("s.result", "h.table")], "str"),
        ({"s": "people"}, [("s.result", "h.table")], "Series[person]"),
    ],
    ids=["one text", "a series of texts", "a text returned per item", "a series of a dataclass, which no table input collects"],
)
def test_a_table_input_of_any_columns_refuses_values_that_are_not_records(nodes: dict, edges: list, got: str):
    graph = compile_graph({**nodes, "h": "height"}, edges)
    assert problems(graph) == [("type_mismatch", ("h", "table"))]
    assert (graph.problems[0].ctx["got"], graph.problems[0].ctx["expected"]) == (got, "Table")


def test_a_table_input_wired_to_a_series_of_tables_starts_a_loop_over_the_tables():
    graph = compile_graph({"s": "sheets", "h": "height", "n": "names"}, [("s.result", "h.table"), ("s.result", "n.table")], outputs={"height": "h.result", "names": "n.result"})
    assert graph.is_runnable
    assert {n: (graph.runs_on[n].mode, graph.runs_on[n].series, input_sources(graph, n)["table"]) for n in ("h", "n")} == {
        "h": ("per_item", "s.result", ("item", "s.result")), "n": ("per_item", "s.result", ("item", "s.result")),
    }


def test_a_table_returned_per_item_is_read_per_item_and_a_series_of_tables_input_collects_them():
    graph = compile_graph({"d": "documents", "t": "sheet", "h": "height", "k": "stacked"}, [("d.result", "t.text"), ("t.result", "h.table"), ("t.result", "k.tables")],
                          outputs={"height": "h.result", "stacked": "k.result"})
    assert graph.is_runnable
    assert (graph.runs_on["h"].mode, graph.runs_on["h"].series, graph.runs_on["k"].mode) == ("per_item", "d.result", "once")
    assert input_sources(graph, "k") == {"tables": ("collected", "d.result")}
    whole = compile_graph({"s": "sheets", "k": "stacked"}, [("s.result", "k.tables")])
    assert whole.is_runnable and input_sources(whole, "k") == {"tables": ("collected", "s.result")}


def test_an_output_of_any_type_wired_to_a_table_input_is_one_value_or_collected_from_its_loop():
    once = compile_graph({"a": "anything", "h": "height"}, [("a.result", "h.table")])
    assert once.is_runnable and input_sources(once, "h") == {"table": ("one_value", None)}
    per_item = compile_graph({"d": "documents", "a": "anything", "h": "height"}, [("d.result", "a.value"), ("a.result", "h.table")])
    assert per_item.is_runnable and per_item.runs_on["h"].mode == "once" and input_sources(per_item, "h") == {"table": ("collected", "d.result")}
    items = compile_graph({"u": "unknowns", "h": "height"}, [("u.result", "h.table")])
    assert items.is_runnable and input_sources(items, "h") == {"table": ("collected", "u.result")}


def test_an_edge_may_name_one_field_of_a_record_and_an_unknown_field_lists_the_ones_there_are():
    graph = compile_graph({"c": "contacts", "r": "rows", "s": "send"}, [("c.result", "r.table"), ("r.result.phone", "s.email")])
    assert problems(graph) == [("unknown_output", ("s", "email"))]
    assert graph.problems[0].ctx["outputs"] == ["r.result.name", "r.result.email"] and graph.problems[0].ctx["record_field"] == "phone"


# -- several edges into one input ------------------------------------------------------------------


def test_two_edges_into_one_series_input_from_nodes_that_both_run_are_refused():
    edges = [("d.result", "u.text"), ("d.result", "v.text"), ("u.result", "c.items"), ("v.result", "c.items")]
    graph = compile_graph({"d": "documents", "u": "upper", "v": "upper", "c": "count"}, edges)
    assert problems(graph) == [("several_edges_into_one_input", ("c", "items"))]


def test_edges_from_both_branches_of_one_node_into_a_series_input_collect_one_loop():
    edges = [("d.result", "w.text"), ("w.yes", "u.text"), ("w.no", "v.text"), ("u.result", "c.items"), ("v.result", "c.items")]
    graph = compile_graph({"d": "documents", "w": "decide", "u": "upper", "v": "upper", "c": "count"}, edges)
    assert graph.is_runnable and input_sources(graph, "c") == {"items": ("collected", "d.result")}


def test_edges_from_two_branches_into_one_input_from_different_loops_are_refused():
    edges = [("w.yes", "a.folder"), ("w.no", "b.folder"), ("a.result", "u.text"), ("b.result", "u.text")]
    graph = compile_graph({"w": "decide", "a": "documents", "b": "documents", "u": "upper"}, edges)
    assert problems(graph) == [("alternatives_from_different_loops", ("u", "text"))]
    assert graph.problems[0].ctx["series"] == ["a.result", "b.result"]


# -- the graph's inputs ----------------------------------------------------------------------------


def test_an_input_is_a_graph_input_only_where_its_entry_says_so_and_one_with_no_entry_is_a_problem_on_any_node():
    record = GraphRecord.model_validate({"nodes": {
        "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
        "p": {"node": "pair", "inputs": {"a": {"sources": [{"node": "u", "output": "result"}]}}},
        "v": {"node": "upper"},
    }})
    graph = Graph.compile(record, REGISTRY)
    assert list(graph.inputs) == ["text"] and problems(graph) == [("input_has_no_value", ("p", "b")), ("input_has_no_value", ("v", "text"))]


def test_a_graph_input_of_a_series_is_given_whole_and_starts_no_loop():
    graph = compile_graph({"c": "count"}, [])
    assert graph.is_runnable and graph.inputs["items"].field.is_series and graph.runs_on["c"].mode == "once"
    assert input_sources(graph, "c") == {"items": ("graph_input", None)}


def test_a_list_given_to_a_graph_input_of_one_value_is_refused_saying_how_a_loop_is_made():
    graph = compile_graph({"u": "upper"}, [])
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a", "b", "c"])
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx) == (
        "invalid_input", ("inputs", "text"), {"name": "text", "expected": "str", "length": 3, "variant": "make_a_series"},
    )
    assert "only when a node returns it as a Series" in problem.msg


# -- a subgraph is one node ------------------------------------------------------------------------


def _with_subgraph(inner_nodes: dict[str, str], inner_edges: list[tuple[str, str]]) -> NodeRegistry:
    """A registry of this file's node definitions and the subgraph ``inner`` of ``inner_nodes`` and ``inner_edges``."""
    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    registry.add(subgraph("inner", title="Inner", description="A subgraph.")(record_of(inner_nodes, inner_edges)))
    return registry


def test_a_subgraph_is_one_node_with_the_inputs_and_outputs_its_record_declares():
    registry = _with_subgraph({"u": "upper", "v": "upper"}, [("u.result", "v.text")])
    graph = compile_graph({"d": "documents", "e": "inner", "c": "count"}, [("d.result", "e.text"), ("e.result", "c.items")], registry)
    assert graph.is_runnable and list(graph._nodes) == ["d", "e", "c"]
    described = graph.describe().nodes["e"]
    assert [i.field.name for i in described.inputs] == ["text"] and [o.name for o in described.outputs] == ["result"]
    assert graph.runs_on["e"].series == "d.result" and input_sources(graph, "c") == {"items": ("collected", "d.result")}


def test_an_output_inside_a_loop_of_a_subgraph_is_offered_as_a_series():
    registry = _with_subgraph({"d": "documents", "u": "upper"}, [("d.result", "u.text")])
    graph = compile_graph({"e": "inner", "c": "count"}, [("e.result", "c.items")], registry)
    assert graph.is_runnable and graph.describe().nodes["e"].outputs[0].is_series
    assert input_sources(graph, "c") == {"items": ("collected", "e.d.result")}


def test_the_inputs_a_subgraph_declares_are_given_by_the_outer_graphs_inputs_as_any_nodes_are():
    registry = _with_subgraph({"p": "pair"}, [])
    record = GraphRecord.model_validate({"nodes": {"e": {"node": "inner", "inputs": {"a": {"graph_input": "first"}, "b": {"graph_input": "second"}}}}})
    graph = Graph.compile(record, registry)
    assert list(graph.inputs) == ["first", "second"] and graph.inputs["first"].field.type == "str"


def test_a_subgraphs_problems_are_the_outer_graphs_with_the_subgraph_nodes_id_in_front():
    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    inner = {  # written out: make_graph would make the open p.b a graph input
        "nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "p": {"node": "pair", "inputs": {"a": {"sources": [{"node": "u", "output": "result"}]}}}},
        "outputs": {"result": {"node": "p", "output": "result"}},
    }
    registry.add(subgraph("inner", title="Inner", description="A subgraph.")(GraphRecord.model_validate(inner)))
    assert problems(compile_graph({"e": "inner"}, [], registry)) == [("input_has_no_value", ("e", "p", "b"))]


def test_the_graph_fingerprint_changes_with_its_subgraphs():
    one = compile_graph({"e": "inner"}, [], _with_subgraph({"u": "upper"}, []))
    other = compile_graph({"e": "inner"}, [], _with_subgraph({"u": "upper", "v": "upper"}, [("u.result", "v.text")]))
    assert one.fingerprint != other.fingerprint
