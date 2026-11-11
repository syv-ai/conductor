"""``split``, ``table-collect``, ``text-list`` and ``text-split``: what each gives, how its hook types it, and what
it refuses."""

from typing import Any, TypedDict

import conductor_nodes
import polars as pl
import pytest
from conductor import InvalidCallError, NodeRegistry, Series, Table, make_graph, node
from conductor_nodes import tables, text


class Contact(TypedDict):
    name: str
    email: str | None


def _node(node_id: str):
    return node(node_id, title=node_id.capitalize(), description=f"The {node_id} node.")


@_node("contacts")
def contacts() -> Table[Contact]:
    return pl.DataFrame({"name": ["Ann", "Bo", "Cy"], "email": ["ann@x.dk", None, "cy@x.dk"]})


@_node("csv")
def csv() -> Table:
    return pl.DataFrame({"name": ["Ann", "Bo"], "age": [31, 47]})


@_node("tags")
def tags() -> list[str]:
    return ["red", "blue"]


@_node("words")
def words() -> Series[str]:
    return ["a", "b"]


@_node("nobody")
def nobody() -> Series[Contact]:
    return []


@_node("sentence")
def sentence() -> str:
    return "one two"


SENT: list[str] = []


@_node("send")
def send(email: str) -> str:
    SENT.append(email)
    return f"sent to {email}"


@_node("shout")
def shout(text: str) -> str:
    return text.upper()


@_node("header")
def header(table: Table) -> list[str]:
    return table.columns


REGISTRY = NodeRegistry()
REGISTRY.add_subregistry(conductor_nodes.registry)
for _definition in (contacts, csv, tags, words, nobody, sentence, send, shout, header):
    REGISTRY.add(_definition)


def load(*nodes: Any, edges: list[tuple[str, str]], values: dict[str, Any] | None = None):
    return REGISTRY.load(make_graph(*nodes, edges=edges, values=values or {}))


def output(graph, node_id: str):
    (described,) = graph.describe().nodes[node_id].outputs
    return described


# -- split ---------------------------------------------------------------------------------------------


def test_split_gives_a_tables_records_typed_as_its_row_type_declares_them():
    graph = load(contacts, tables.split, edges=[("contacts.result", "split.value")])
    described = output(graph, "split")
    assert graph.problems == [] and (described.shape, described.type) == ("many", "record")
    assert described.json_schema["properties"] == {
        "name": {"title": "Name", "type": "string"}, "email": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Email"},
    }
    assert graph.run().results["split"]["result"] == [
        {"name": "Ann", "email": "ann@x.dk"}, {"name": "Bo", "email": None}, {"name": "Cy", "email": "cy@x.dk"},
    ]


def test_a_node_reading_a_field_of_a_split_record_is_not_called_where_the_cell_is_blank_and_collect_keeps_the_record():
    SENT.clear()
    graph = load(contacts, tables.split, send, tables.collect, edges=[
        ("contacts.result", "split.value"), ("split.result.email", "send.email"),
        ("split.result", "table-collect.records"), ("send.result", "table-collect.sent"),
    ], values={"table-collect.columns": ["sent"]})
    run = graph.run()
    assert run.status == "complete" and SENT == ["ann@x.dk", "cy@x.dk"]
    assert [(n.node_id, n.item, n.input) for n in run.not_called] == [("send", (1,), "email")]
    assert run.results["table-collect"]["result"].to_dicts() == [
        {"name": "Ann", "email": "ann@x.dk", "sent": "sent to ann@x.dk"},
        {"name": "Bo", "email": None, "sent": None},
        {"name": "Cy", "email": "cy@x.dk", "sent": "sent to cy@x.dk"},
    ]


def test_split_gives_records_of_unknown_columns_for_a_table_of_any_and_a_field_of_one_is_read_when_it_arrives():
    graph = load(csv, tables.split, shout, edges=[("csv.result", "split.value"), ("split.result.name", "shout.text")])
    assert graph.problems == [] and output(graph, "split").type == "dict"
    assert graph.run().results["shout"]["result"] == ["ANN", "BO"]


def test_split_gives_a_lists_items_of_the_lists_own_type():
    graph = load(tags, tables.split, shout, edges=[("tags.result", "split.value"), ("split.result", "shout.text")])
    assert graph.problems == [] and (output(graph, "split").shape, output(graph, "split").type) == ("many", "str")
    assert graph.plan["shout"].series == "split.result" and graph.run().results["shout"]["result"] == ["RED", "BLUE"]


def test_a_list_typed_into_split_is_items_of_any_type():
    graph = load(tables.split, edges=[], values={"split.value": ["a", 1]})
    assert output(graph, "split").type == "Any" and graph.run().results["split"]["result"] == ["a", 1]


def test_split_refuses_what_is_neither_a_table_nor_a_list_where_its_hook_reads_the_arriving_type():
    graph = load(sentence, tables.split, edges=[("sentence.result", "split.value")])
    (problem,) = graph.problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("split",), "type_mismatch")
    assert problem.ctx["reason"] == "'value' takes a table or a list, and what is wired to it is neither; wire a table or a list, or use Split text for a text"


# -- table-collect ---------------------------------------------------------------------------------------


def test_collect_types_its_table_by_the_records_and_what_arrives_on_each_new_column():
    graph = load(contacts, tables.split, send, sentence, tables.collect, edges=[
        ("contacts.result", "split.value"), ("split.result.email", "send.email"), ("split.result", "table-collect.records"),
        ("send.result", "table-collect.sent"), ("sentence.result", "table-collect.team"),
    ], values={"table-collect.columns": ["sent"], "table-collect.constant_columns": ["team"]})
    assert graph.problems == []
    columns = output(graph, "table-collect").columns
    assert list(columns) == ["name", "email", "sent", "team"] and columns["name"] == {"type": "string"}
    assert columns["sent"] == {"anyOf": [{"type": "string"}, {"type": "null"}]} and columns["team"] == {"type": "string"}
    feeds = {i.name: i.feed.mode for i in graph.describe().nodes["table-collect"].inputs}
    assert (feeds["records"], feeds["sent"], feeds["team"]) == ("gathered", "gathered", "one_value")
    assert graph.run().results["table-collect"]["result"]["team"].to_list() == ["one two"] * 3


def test_collect_refuses_a_new_column_the_records_already_have_naming_it():
    graph = load(contacts, tables.split, tables.collect, edges=[
        ("contacts.result", "split.value"), ("split.result", "table-collect.records"), ("split.result.name", "table-collect.email"),
    ], values={"table-collect.columns": ["email"]})
    (problem,) = graph.problems
    assert (problem.type, problem.loc, problem.ctx["error"], problem.ctx["error_ctx"]) == ("compute_hook_failed", ("table-collect",), "column_taken", {"column": "email"})


def test_collect_refuses_a_column_named_twice():
    graph = load(tables.collect, edges=[], values={"table-collect.columns": ["team"], "table-collect.constant_columns": ["team"]})
    (problem,) = graph.problems
    assert (problem.type, problem.ctx["error"], problem.ctx["error_ctx"]) == ("compute_hook_failed", "column_named_twice", {"column": "team"})


def test_collect_of_no_records_is_a_table_with_every_column_and_no_lines():
    graph = load(nobody, sentence, tables.collect, header, edges=[
        ("nobody.result", "table-collect.records"), ("sentence.result", "table-collect.team"), ("table-collect.result", "header.table"),
    ], values={"table-collect.constant_columns": ["team"]})
    run = graph.run()
    assert run.status == "complete", run.state
    assert run.results["table-collect"]["result"].height == 0 and run.results["header"]["result"] == ["name", "email", "team"]


def test_collect_of_records_of_unknown_columns_is_a_table_of_any_columns():
    graph = load(csv, tables.split, shout, tables.collect, edges=[
        ("csv.result", "split.value"), ("split.result.name", "shout.text"), ("split.result", "table-collect.records"),
        ("shout.result", "table-collect.loud"),
    ], values={"table-collect.columns": ["loud"]})
    assert graph.problems == [] and output(graph, "table-collect").columns == {}
    assert graph.run().results["table-collect"]["result"].to_dicts() == [{"name": "Ann", "age": 31, "loud": "ANN"}, {"name": "Bo", "age": 47, "loud": "BO"}]


def test_collect_takes_records_only():
    with pytest.raises(InvalidCallError) as refused:
        load(words, tables.collect, edges=[("words.result", "table-collect.records")])
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx["got"], problem.ctx["expected"]) == ("type_mismatch", ("table-collect", "records"), "Series[str]", "Series[dict]")


# -- text-list and text-split -------------------------------------------------------------------------------


def test_text_list_gives_its_texts_once_and_once_per_item_of_what_is_wired_to_for_each():
    alone = load(text.text_list, shout, edges=[("text-list.result", "shout.text")], values={"text-list.items": ["a", "b"]})
    assert alone.plan["text-list"].mode == "once" and alone.run().results["shout"]["result"] == ["A", "B"]
    inside = load(words, text.text_list, edges=[("words.result", "text-list.for_each")], values={"text-list.items": ["x", "y", "z"]})
    assert inside.plan["text-list"].series == "words.result"
    assert inside.run().results["text-list"]["result"] == [["x", "y", "z"], ["x", "y", "z"]]


def test_text_split_is_titled_split_text_splits_on_a_newline_and_an_empty_text_has_no_parts():
    assert text.split.title == "Split text" and text.split.inputs["separator"].default == "\n"
    assert text.split("a\nb") == ["a", "b"] and text.split("a,b", separator=",") == ["a", "b"] and text.split("") == []
    assert tables.split.title == "Split" and tables.collect.title == "Collect into a table" and text.text_list.title == "Text list"
