"""Where each node of a loaded graph runs: a Series read one item at a time opens a map, a value from outside it is
captured, a ``Series`` or ``Table`` input gathers, and declared types alone decide all of it."""

import time
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypedDict

import pytest
from conductor import SKIPPED, InvalidCallError, NodeRegistry, Param, Series, Table, node


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


@dataclass(frozen=True)
class Person:
    type_id: ClassVar[str] = "person"
    type_title: ClassVar[str] = "Person"
    name: str


@_node("people")
def people(folder: str) -> Series[Person]: ...


REGISTRY = NodeRegistry()
for _definition in (
    contacts, rows, send, collect, names, ages, documents, lines, count, total, pair, pair_or_alone, upper, both, decide, measure, anything,
    height, sheets, sheet, stacked, mappings, unknowns, people,
):
    REGISTRY.add(_definition)


def load(nodes: dict[str, str | dict], edges: list[tuple[str, str]], registry: NodeRegistry = REGISTRY):
    """A graph of ``nodes`` (id to node id, or to the rest of its record) and ``edges``."""
    records = [{"id": i, **({"node": n} if isinstance(n, str) else n)} for i, n in nodes.items()]
    return registry.load({"nodes": records, "edges": [list(edge) for edge in edges]})


def feeds(graph, node_id: str) -> dict[str, tuple[str, str | None]]:
    return {i.field.name: (i.feed.mode, i.feed.series) for i in graph.describe().nodes[node_id].inputs}


def problems(graph) -> list[tuple[str, tuple]]:
    return [(p.type, p.loc) for p in graph.problems]


# -- the three rules --------------------------------------------------------------------------------


def test_a_series_read_by_an_input_of_one_value_opens_a_map_and_what_reads_the_reader_runs_in_it_too():
    graph = load({"d": "documents", "u": "upper", "v": "upper"}, [("d.result", "u.text"), ("u.result", "v.text")])
    assert graph.is_runnable
    assert {n: (r.mode, r.series) for n, r in graph.plan.items()} == {"d": ("once", None), "u": ("per_item", "d.result"), "v": ("per_item", "d.result")}
    assert feeds(graph, "u") == {"text": ("per_item", "d.result")}


def test_a_value_from_an_enclosing_scope_is_captured():
    graph = load({"d": "documents", "u": "upper", "p": "pair"}, [("d.result", "p.a"), ("u.result", "p.b")])
    assert graph.plan["p"].series == "d.result"
    assert feeds(graph, "p") == {"a": ("per_item", "d.result"), "b": ("outer_value", None)}


def test_a_series_input_gathers_one_map_and_sits_where_its_series_was_made():
    graph = load({"d": "documents", "u": "upper", "c": "count"}, [("d.result", "u.text"), ("u.result", "c.items")])
    assert graph.plan["c"].mode == "once"
    assert feeds(graph, "c") == {"items": ("gathered", "d.result")}


def test_nested_maps_gather_one_level_at_a_time():
    graph = load(
        {"d": "documents", "l": "lines", "u": "upper", "c": "count", "t": "total"},
        [("d.result", "l.document"), ("l.result", "u.text"), ("u.result", "c.items"), ("c.result", "t.counts")],
    )
    assert graph.is_runnable
    assert {n: r.series for n, r in graph.plan.items()} == {"d": None, "l": "d.result", "u": "l.result", "c": "d.result", "t": None}
    assert feeds(graph, "c") == {"items": ("gathered", "l.result")} and feeds(graph, "t") == {"counts": ("gathered", "d.result")}


def test_a_series_read_whole_is_the_gather_of_its_own_map():
    graph = load({"d": "documents", "u": "upper", "b": "both"}, [("d.result", "u.text"), ("d.result", "b.first"), ("u.result", "b.second")])
    assert graph.is_runnable and graph.plan["b"].mode == "once"
    assert feeds(graph, "b") == {"first": ("gathered", "d.result"), "second": ("gathered", "d.result")}


def test_a_gather_inside_a_map_captured_from_deeper_in_the_same_map_is_on_one_path():
    graph = load(
        {"d": "documents", "l": "lines", "c": "count", "p": "pair", "u": "upper"},
        [("d.result", "l.document"), ("l.result", "c.items"), ("l.result", "p.a"), ("d.result", "p.b")],
    )
    assert graph.is_runnable
    assert feeds(graph, "p") == {"a": ("per_item", "l.result"), "b": ("outer_value", "d.result")}


# -- what is refused ----------------------------------------------------------------------------------


def test_items_of_two_series_never_pair_up():
    graph = load({"a": "documents", "b": "documents", "p": "pair"}, [("a.result", "p.a"), ("b.result", "p.b")])
    assert problems(graph) == [("series_not_nested", ("p",))]
    assert graph.problems[0].ctx["series"] == ["a.result", "b.result"]


def test_two_readers_of_one_series_read_one_map():
    graph = load({"d": "documents", "u": "upper", "v": "upper", "p": "pair"}, [("d.result", "u.text"), ("d.result", "v.text"), ("u.result", "p.a"), ("v.result", "p.b")])
    assert graph.is_runnable and graph.plan["p"].series == "d.result"


def test_a_series_input_gets_the_one_value_of_a_source_that_runs_once_as_a_series_of_one():
    graph = load({"u": "upper", "c": "count"}, [("u.result", "c.items")])
    assert graph.is_runnable and graph.plan["c"].mode == "once"
    assert feeds(graph, "c") == {"items": ("gathered", None)}


def test_a_list_typed_into_an_input_of_one_value_is_not_a_value_of_its_type():
    graph = load({"u": {"node": "upper", "values": {"text": ["a", "b"]}}}, [])
    assert problems(graph) == [("invalid_value", ("u", "text"))]


def test_an_item_of_the_wrong_type_is_a_mismatch_said_as_the_series():
    graph = load({"d": "documents", "t": "total"}, [("d.result", "t.counts")])
    assert (graph.problems[0].type, graph.problems[0].ctx["got"], graph.problems[0].ctx["expected"]) == ("type_mismatch", "Series[str]", "Series[int]")


# -- a table is one value --------------------------------------------------------------------------


def test_a_table_passes_whole_and_never_makes_items():
    graph = load({"c": "contacts", "n": "names"}, [("c.result", "n.table")])
    assert graph.is_runnable and graph.plan["n"].mode == "once" and feeds(graph, "n") == {"table": ("one_value", None)}


def test_a_table_fits_a_table_input_by_its_columns_and_the_mismatch_names_both():
    graph = load({"c": "contacts", "a": "ages"}, [("c.result", "a.table")])
    assert problems(graph) == [("type_mismatch", ("a", "table"))]
    assert graph.problems[0].ctx["got"] == "Table[name: str, email: str | None]"
    assert graph.problems[0].ctx["expected"] == "Table[name: str, age: int]"


def test_a_table_on_an_input_of_one_text_is_a_type_mismatch():
    graph = load({"c": "contacts", "u": "upper"}, [("c.result", "u.text")])
    assert problems(graph) == [("type_mismatch", ("u", "text"))]


def test_an_input_of_any_type_takes_a_table_as_the_one_value_it_is():
    graph = load({"c": "contacts", "a": "anything"}, [("c.result", "a.value")])
    assert graph.is_runnable and graph.plan["a"].mode == "once"


def test_a_table_input_gathers_records_and_lines_up_with_a_series_input_on_the_same_map():
    graph = load(
        {"c": "contacts", "r": "rows", "s": "send", "k": "collect"},
        [("c.result", "r.table"), ("r.result.email", "s.email"), ("r.result", "k.records"), ("s.result", "k.sent")],
    )
    assert graph.is_runnable
    assert graph.plan["s"].series == "r.result" and graph.plan["k"].mode == "once"
    assert feeds(graph, "k") == {"records": ("gathered", "r.result"), "sent": ("gathered", "r.result")}


@pytest.mark.parametrize("source", ["rows", "mappings", "unknowns"], ids=["records of declared columns", "mappings", "items of any type"])
def test_a_table_input_of_any_columns_gathers_what_can_be_a_record(source: str):
    graph = load({"c": "contacts", "s": source, "h": "height"}, [*([("c.result", "s.table")] if source == "rows" else []), ("s.result", "h.table")])
    assert graph.is_runnable and graph.plan["h"].mode == "once" and feeds(graph, "h") == {"table": ("gathered", "s.result")}


@pytest.mark.parametrize(
    ("nodes", "edges", "got"),
    [
        ({"s": "upper"}, [("s.result", "h.table")], "str"),
        ({"s": "documents"}, [("s.result", "h.table")], "Series[str]"),
        ({"d": "documents", "s": "upper"}, [("d.result", "s.text"), ("s.result", "h.table")], "str"),
        ({"s": "people"}, [("s.result", "h.table")], "Series[person]"),
    ],
    ids=["one text", "a series of texts", "a text made per item", "a series of a dataclass, which no table input gathers"],
)
def test_a_table_input_of_any_columns_refuses_what_is_no_record_as_one_of_declared_columns_does(nodes: dict, edges: list, got: str):
    graph = load({**nodes, "h": "height"}, edges)
    assert problems(graph) == [("type_mismatch", ("h", "table"))]
    assert (graph.problems[0].ctx["got"], graph.problems[0].ctx["expected"]) == (got, "Table")


def test_a_series_of_tables_is_read_one_table_per_item_by_a_table_input_as_any_series_is_by_an_input_of_one_value():
    graph = load({"s": "sheets", "h": "height", "n": "names"}, [("s.result", "h.table"), ("s.result", "n.table")])
    assert graph.is_runnable
    assert {n: (graph.plan[n].mode, graph.plan[n].series, feeds(graph, n)["table"]) for n in ("h", "n")} == {
        "h": ("per_item", "s.result", ("per_item", "s.result")), "n": ("per_item", "s.result", ("per_item", "s.result")),
    }


def test_a_table_made_per_item_is_read_per_item_and_a_series_of_tables_input_gathers_the_tables():
    graph = load({"d": "documents", "t": "sheet", "h": "height", "k": "stacked"}, [("d.result", "t.text"), ("t.result", "h.table"), ("t.result", "k.tables")])
    assert graph.is_runnable
    assert (graph.plan["h"].mode, graph.plan["h"].series, graph.plan["k"].mode) == ("per_item", "d.result", "once")
    assert feeds(graph, "k") == {"tables": ("gathered", "d.result")}
    whole = load({"s": "sheets", "k": "stacked"}, [("s.result", "k.tables")])
    assert whole.is_runnable and feeds(whole, "k") == {"tables": ("gathered", "s.result")}


def test_a_value_of_any_type_is_read_as_a_table_only_where_there_is_nothing_to_gather():
    once = load({"a": "anything", "h": "height"}, [("a.result", "h.table")])
    assert once.is_runnable and feeds(once, "h") == {"table": ("one_value", None)}
    per_item = load({"d": "documents", "a": "anything", "h": "height"}, [("d.result", "a.value"), ("a.result", "h.table")])
    assert per_item.is_runnable and per_item.plan["h"].mode == "once" and feeds(per_item, "h") == {"table": ("gathered", "d.result")}
    items = load({"u": "unknowns", "h": "height"}, [("u.result", "h.table")])
    assert items.is_runnable and feeds(items, "h") == {"table": ("gathered", "u.result")}


def test_an_edge_may_name_one_field_of_a_record_and_an_unknown_field_lists_the_ones_there_are():
    graph = load({"c": "contacts", "r": "rows", "s": "send"}, [("c.result", "r.table"), ("r.result.phone", "s.email")])
    assert problems(graph) == [("unknown_output", ("s", "email"))]
    assert graph.problems[0].ctx["outputs"] == ["r.result.name", "r.result.email"] and graph.problems[0].ctx["field"] == "phone"


# -- edges into one input are alternatives -----------------------------------------------------------


def test_two_edges_into_a_series_input_are_alternatives_like_any_input():
    edges = [("d.result", "u.text"), ("d.result", "v.text"), ("u.result", "c.items"), ("v.result", "c.items")]
    graph = load({"d": "documents", "u": "upper", "v": "upper", "c": "count"}, edges)
    assert problems(graph) == [("several_edges_into_one_input", ("c", "items"))]


def test_alternatives_per_item_of_one_decision_are_gathered_as_one_map():
    edges = [("d.result", "w.text"), ("w.yes", "u.text"), ("w.no", "v.text"), ("u.result", "c.items"), ("v.result", "c.items")]
    graph = load({"d": "documents", "w": "decide", "u": "upper", "v": "upper", "c": "count"}, edges)
    assert graph.is_runnable and feeds(graph, "c") == {"items": ("gathered", "d.result")}


def test_alternatives_that_run_per_item_of_different_series_are_refused():
    edges = [("w.yes", "a.folder"), ("w.no", "b.folder"), ("a.result", "u.text"), ("b.result", "u.text")]
    graph = load({"w": "decide", "a": "documents", "b": "documents", "u": "upper"}, edges)
    assert problems(graph) == [("alternatives_in_different_series", ("u", "text"))]
    assert graph.problems[0].ctx["series"] == ["a.result", "b.result"]


# -- what skips a gather for every item at once ------------------------------------------------------


def gates(graph, node_id: str, name: str) -> list[list[set[str]]]:
    plan = graph._nodes[node_id].inputs[name]
    return [sorted((sorted(cell.address for cell in cells) for cells in gate.skips)) for gate in plan.gates]


def test_a_gather_is_gated_by_its_series_and_by_each_branch_decided_outside_the_map():
    edges = [("w.yes", "p.a"), ("d.result", "p.b"), ("p.result", "c.items")]
    graph = load({"w": "decide", "d": "documents", "p": "pair", "c": "count"}, edges)
    assert gates(graph, "c", "items") == [[["d.result"], ["w.yes"]]]


def blanks(graph, node_id: str, name: str) -> list[list[str]]:
    plan = graph._nodes[node_id].inputs[name]
    return [sorted(cell.address for cell in gate.blanks) for gate in plan.gates]


def test_a_gather_is_left_uncalled_by_a_blank_outside_the_map_that_reaches_it_through_inputs_taking_none():
    edges = [("w.yes", "p.a"), ("d.result", "p.b"), ("p.result", "c.items")]
    assert blanks(load({"w": "decide", "d": "documents", "p": "pair", "c": "count"}, edges), "c", "items") == [["d.result", "w.yes"]]
    graph = load({"m": "upper", "d": "documents", "p": "pair-or-alone", "c": "count"}, [("d.result", "p.a"), ("m.result", "p.b"), ("p.result", "c.items")])
    assert graph.is_runnable and blanks(graph, "c", "items") == [["d.result"]]  # p takes m.result's blank, so it is called for every item


def test_a_decision_made_per_item_gates_nothing():
    edges = [("d.result", "w.text"), ("w.yes", "u.text"), ("u.result", "c.items")]
    graph = load({"d": "documents", "w": "decide", "u": "upper", "c": "count"}, edges)
    assert gates(graph, "c", "items") == [[["d.result"]]]


def test_alternatives_inside_the_map_gate_a_gather_only_when_every_one_of_them_is_skipped():
    # x needs w.yes and m.small; y needs w.no and m.large; either feeds p, which runs per item and is gathered
    edges = [
        ("w.yes", "x.a"), ("m.small", "x.b"), ("w.no", "y.a"), ("m.large", "y.b"),
        ("x.result", "p.a"), ("y.result", "p.a"), ("d.result", "p.b"), ("p.result", "c.items"),
    ]
    graph = load({"w": "decide", "m": "measure", "d": "documents", "x": "pair", "y": "pair", "p": "pair", "c": "count"}, edges)
    assert graph.is_runnable
    assert gates(graph, "c", "items") == [[["d.result"], ["x.result", "y.result"]]]


# -- the graph's inputs ---------------------------------------------------------------------------------


def test_a_first_nodes_open_inputs_are_the_graphs_and_a_later_nodes_must_be_fed():
    graph = load({"u": "upper", "p": "pair"}, [("u.result", "p.a")])
    assert list(graph.inputs) == ["text"] and problems(graph) == [("unbound_on_later_node", ("p", "b"))]


def test_a_graph_input_of_a_series_is_given_whole_and_opens_no_map():
    graph = load({"c": "count"}, [])
    assert graph.is_runnable and graph.inputs["items"].field.many and graph.plan["c"].mode == "once"
    assert feeds(graph, "c") == {"items": ("one_value", None)}


def test_a_list_given_at_an_input_of_one_value_is_refused_saying_what_makes_a_graph_run_per_item():
    graph = load({"u": "upper"}, [])
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a", "b", "c"])
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx) == (
        "invalid_input", ("inputs", "text"), {"name": "text", "expected": "str", "length": 3, "hint": "make_a_series"},
    )
    assert "a node that makes a Series" in problem.msg


# -- an embedded graph is one node -----------------------------------------------------------------------


def _embedding(inner_nodes: dict[str, str], inner_edges: list[tuple[str, str]]) -> NodeRegistry:
    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    record = {"nodes": [{"id": i, "node": n} for i, n in inner_nodes.items()], "edges": [list(edge) for edge in inner_edges]}
    registry.add(node("inner", title="Inner", description="An embedded graph.")(record))
    return registry


def test_an_embedded_graph_is_one_node_with_the_inner_graphs_inputs_and_outputs():
    registry = _embedding({"u": "upper", "v": "upper"}, [("u.result", "v.text")])
    graph = load({"d": "documents", "e": "inner", "c": "count"}, [("d.result", "e.text"), ("e.v.result", "c.items")], registry)
    assert graph.is_runnable and list(graph._nodes) == ["d", "e", "c"]
    described = graph.describe().nodes["e"]
    assert [i.field.name for i in described.inputs] == ["text"] and [o.name for o in described.outputs] == ["v.result"]
    assert graph.plan["e"].series == "d.result" and feeds(graph, "c") == {"items": ("gathered", "d.result")}


def test_an_output_made_per_item_inside_an_embedded_graph_is_offered_as_a_series():
    registry = _embedding({"d": "documents", "u": "upper"}, [("d.result", "u.text")])
    graph = load({"e": "inner", "c": "count"}, [("e.u.result", "c.items")], registry)
    assert graph.is_runnable and graph.describe().nodes["e"].outputs[0].many
    assert feeds(graph, "c") == {"items": ("gathered", "e.u.result")}


def test_a_first_embedded_graphs_inputs_are_the_outer_graphs():
    registry = _embedding({"p": "pair"}, [])
    graph = load({"e": "inner"}, [], registry)
    assert list(graph.inputs) == ["a", "b"] and graph.inputs["a"].address == "e.a"


def test_an_embedded_graphs_problems_are_the_graphs_with_the_embedding_in_front():
    registry = _embedding({"p": "pair", "u": "upper"}, [("u.result", "p.a")])
    assert problems(load({"e": "inner"}, [], registry)) == [("unbound_on_later_node", ("e", "p", "b"))]


def test_the_fingerprint_follows_the_graphs_embedded_in_it():
    one = load({"e": "inner"}, [], _embedding({"u": "upper"}, []))
    other = load({"e": "inner"}, [], _embedding({"u": "upper", "v": "upper"}, [("u.result", "v.text")]))
    assert one.fingerprint != other.fingerprint


# -- a gate is worked out once per node -----------------------------------------------------------------


def _staged(stages: int) -> Any:
    """A map with ``stages`` stages inside it, each two nodes on opposite branches of a decision made outside
    the map, both fed by both nodes of the stage before; a gather of the last stage."""
    nodes: dict[str, str | dict] = {"d": "documents"}
    edges: list[tuple[str, str]] = []
    before = ["d.result"]
    for stage in range(stages):
        nodes.update({f"w{stage}": "decide", f"x{stage}": "pair", f"y{stage}": "pair"})
        edges += [(source, f"{node_id}{stage}.a") for source in before for node_id in "xy"]
        edges += [(f"w{stage}.yes", f"x{stage}.b"), (f"w{stage}.no", f"y{stage}.b")]
        before = [f"x{stage}.result", f"y{stage}.result"]
    nodes["c"] = "count"
    return load(nodes, [*edges, *((source, "c.items") for source in before)])


def test_a_gate_behind_a_chain_of_alternatives_is_worked_out_once_per_node_not_once_per_way_to_it():
    """Each stage doubles the ways up to the stages before it; loading follows each node's gate once all the same."""
    def seconds(stages: int) -> float:
        started = time.perf_counter()
        graph = _staged(stages)
        took = time.perf_counter() - started
        assert graph.is_runnable and [len(gate.skips) for gate in graph._nodes["c"].inputs["items"].gates] == [stages + 1, stages + 1]
        return took

    seconds(2)  # whatever a first load pays once
    assert min(seconds(16) for _ in range(3)) < 8 * min(seconds(8) for _ in range(3))  # twice the nodes; 2**8 times the ways
