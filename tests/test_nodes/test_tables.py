"""``series-split``, ``series-stack``, ``series-gather``, ``table-collect``, ``text-list`` and ``text-split``: what each gives, how it is typed, and
what it refuses."""

from typing import Any, TypedDict

import conductor_nodes
import polars as pl
import pytest
from conductor import Graph, NodeRegistry, Series, Table, make_graph, node
from conductor_nodes import series, tables, text


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


@_node("counts")
def counts() -> Series[int]:
    return [1, 2]


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


@_node("one-row")
def one_row(contact: Contact) -> Table[Contact]:
    return pl.DataFrame([contact], schema={"name": pl.String, "email": pl.String})


@_node("header")
def header(table: Table) -> list[str]:
    return table.columns


REGISTRY = NodeRegistry()
REGISTRY.add_subregistry(conductor_nodes.registry)
for _definition in (contacts, csv, tags, words, counts, nobody, sentence, send, shout, one_row, header):
    REGISTRY.add(_definition)


def load(*nodes: Any, edges: list[tuple[str, str]], values: dict[str, Any] | None = None):
    return Graph.compile(make_graph(*nodes, edges=edges, values=values or {}), REGISTRY)


def output(graph, node_id: str):
    (described,) = graph.describe().nodes[node_id].outputs
    return described


def columns_of(field) -> dict[str, Any]:
    """A table field's columns, each column's name to its cell's JSON schema, read from the field's schema."""
    return {c["properties"]["name"]["const"]: c["properties"]["values"]["items"] for c in field.json_schema["properties"]["columns"]["prefixItems"]}


# -- split ---------------------------------------------------------------------------------------------


def test_split_gives_a_tables_records_typed_as_its_row_type_declares_them():
    graph = load(contacts, series.split, edges=[("contacts.result", "series-split.value")])
    described = output(graph, "series-split")
    assert graph.problems == [] and (described.many, described.type) == (True, "record")
    assert described.json_schema["properties"] == {
        "name": {"title": "Name", "type": "string"}, "email": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Email"},
    }
    assert graph.run().results["series-split"]["result"] == [
        {"name": "Ann", "email": "ann@x.dk"}, {"name": "Bo", "email": None}, {"name": "Cy", "email": "cy@x.dk"},
    ]


def test_a_node_reading_a_field_of_a_split_record_is_not_called_where_the_cell_is_blank_and_collect_keeps_the_record():
    SENT.clear()
    graph = load(contacts, series.split, send, tables.collect, edges=[
        ("contacts.result", "series-split.value"), ("series-split.result.email", "send.email"),
        ("series-split.result", "table-collect.records"), ("send.result", "table-collect.sent"),
    ], values={"table-collect.columns": ["sent"]})
    run = graph.run()
    assert run.status == "complete" and SENT == ["ann@x.dk", "cy@x.dk"]
    assert [(n.loc[0], n.loc[1:], n.input) for n in run.not_called] == [("send", (1,), "email")]
    assert run.results["table-collect"]["result"].to_dicts() == [
        {"name": "Ann", "email": "ann@x.dk", "sent": "sent to ann@x.dk"},
        {"name": "Bo", "email": None, "sent": None},
        {"name": "Cy", "email": "cy@x.dk", "sent": "sent to cy@x.dk"},
    ]


def test_split_gives_records_of_unknown_columns_for_a_table_of_any_and_a_field_of_one_is_read_when_it_arrives():
    graph = load(csv, series.split, shout, edges=[("csv.result", "series-split.value"), ("series-split.result.name", "shout.text")])
    assert graph.problems == [] and output(graph, "series-split").type == "Any"
    assert graph.run().results["shout"]["result"] == ["ANN", "BO"]


def test_split_gives_a_lists_items_of_the_lists_own_type():
    graph = load(tags, series.split, shout, edges=[("tags.result", "series-split.value"), ("series-split.result", "shout.text")])
    assert graph.problems == [] and (output(graph, "series-split").many, output(graph, "series-split").type) == (True, "str")
    assert graph.runs_on["shout"].series == "series-split.result" and graph.run().results["shout"]["result"] == ["RED", "BLUE"]


def test_a_list_typed_into_split_is_items_of_any_type():
    graph = load(series.split, edges=[], values={"series-split.value": ["a", 1]})
    assert output(graph, "series-split").type == "Any" and graph.run().results["series-split"]["result"] == ["a", 1]


def test_what_is_neither_a_table_nor_a_list_is_a_type_mismatch_on_the_edge_into_split():
    (problem,) = load(sentence, series.split, edges=[("sentence.result", "series-split.value")]).problems
    assert (problem.type, problem.loc, problem.ctx["got"], problem.ctx["expected"]) == ("type_mismatch", ("series-split", "value"), "str", "Table | list")


def test_stack_puts_the_tables_made_per_item_into_one_table_of_their_row_type():
    graph = load(contacts, series.split, one_row, series.stack, edges=[
        ("contacts.result", "series-split.value"), ("series-split.result", "one-row.contact"), ("one-row.result", "series-stack.tables"),
    ])
    assert graph.problems == [] and not output(graph, "series-stack").many and output(graph, "series-stack").type.startswith("Table") and set(columns_of(output(graph, "series-stack"))) == {"name", "email"}
    stacked = graph.run().results["series-stack"]["result"]
    assert stacked.to_dicts() == graph.run().results["contacts"]["result"].to_dicts()


def test_gather_gives_the_values_made_per_item_as_one_list():
    graph = load(series.split, text.uppercase, series.gather, edges=[
        ("series-split.result", "text-uppercase.text"), ("text-uppercase.result", "series-gather.values"),
    ], values={"series-split.value": ["a", "b"]})
    assert graph.problems == [] and graph.run().results["series-gather"]["result"] == ["A", "B"]


def test_gather_takes_the_type_of_the_values_it_gathers():
    graph = load(counts, series.gather, edges=[("counts.result", "series-gather.values")])
    assert graph.problems == [] and output(graph, "series-gather").type == "int"


# -- table-collect ---------------------------------------------------------------------------------------


def test_collect_types_its_table_by_the_records_and_what_arrives_on_each_new_column():
    graph = load(contacts, series.split, send, sentence, tables.collect, edges=[
        ("contacts.result", "series-split.value"), ("series-split.result.email", "send.email"), ("series-split.result", "table-collect.records"),
        ("send.result", "table-collect.sent"), ("sentence.result", "table-collect.team"),
    ], values={"table-collect.columns": ["sent"], "table-collect.constant_columns": ["team"]})
    assert graph.problems == []
    columns = columns_of(output(graph, "table-collect"))
    assert list(columns) == ["name", "email", "sent", "team"] and columns["name"] == {"type": "string"}
    assert columns["sent"] == {"anyOf": [{"type": "string"}, {"type": "null"}]} and columns["team"] == {"type": "string"}
    feeds = {i.field.name: i.feed.mode for i in graph.describe().nodes["table-collect"].inputs}
    assert (feeds["records"], feeds["sent"], feeds["team"]) == ("collected", "collected", "one_value")
    assert graph.run().results["table-collect"]["result"]["team"].to_list() == ["one two"] * 3


def test_collect_takes_the_records_as_a_table_and_keeps_its_columns_as_they_are():
    @_node("small-numbers")
    def small_numbers() -> Table:
        return pl.DataFrame({"n": pl.Series([1, 2], dtype=pl.Int32), "name": ["a", "b"]})

    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    registry.add(small_numbers)
    graph = Graph.compile(make_graph(small_numbers, series.split, shout, tables.collect, edges=[
        ("small-numbers.result", "series-split.value"), ("series-split.result.name", "shout.text"), ("series-split.result", "table-collect.records"),
        ("shout.result", "table-collect.loud"),
    ], values={"table-collect.columns": ["loud"]}), registry)
    made = graph.run().results["table-collect"]["result"]

    assert [i.field.type for i in graph.describe().nodes["table-collect"].inputs if i.field.name == "records"] == ["Table"]
    assert made.schema == {"n": pl.Int32, "name": pl.String, "loud": pl.String} and made["loud"].to_list() == ["A", "B"]


def test_collect_given_a_whole_table_types_its_result_by_that_tables_columns():
    graph = load(contacts, sentence, tables.collect, edges=[("contacts.result", "table-collect.records"), ("sentence.result", "table-collect.team")], values={"table-collect.constant_columns": ["team"]})

    assert graph.problems == [] and list(columns_of(output(graph, "table-collect"))) == ["name", "email", "team"]
    assert graph.run().results["table-collect"]["result"]["team"].to_list() == ["one two"] * 3


def test_collect_refuses_a_new_column_the_records_already_have_naming_it():
    graph = load(contacts, series.split, tables.collect, edges=[
        ("contacts.result", "series-split.value"), ("series-split.result", "table-collect.records"), ("series-split.result.name", "table-collect.email"),
    ], values={"table-collect.columns": ["email"]})
    (problem,) = graph.problems
    assert (problem.type, problem.loc, problem.ctx["error"], problem.ctx["error_ctx"]) == ("compute_hook_failed", ("table-collect",), "column_taken", {"column": "email"})


def test_collect_refuses_a_column_named_twice():
    graph = load(tables.collect, edges=[], values={"table-collect.columns": ["team"], "table-collect.constant_columns": ["team"]})
    (problem,) = graph.problems
    assert (problem.type, problem.ctx["error"], problem.ctx["error_ctx"]) == ("compute_hook_failed", "column_named_twice", {"column": "team"})


@pytest.mark.parametrize(("people", "expected"), [(3, {"column": "word", "records": 3, "values": 2}), (1, {"column": "word", "records": 1, "values": 2})])
def test_collect_refuses_a_column_wired_from_another_series_than_its_records_naming_it(people: int, expected: dict[str, Any]):
    """A column holds one value per record only when it is made per record: one from an unrelated Series is refused, longer or shorter."""
    @_node("people")
    def people_table() -> Table[Contact]:
        return pl.DataFrame({"name": [f"p{i}" for i in range(people)], "email": [None] * people}, schema={"name": pl.String, "email": pl.String})

    held = NodeRegistry()
    held.add_subregistry(REGISTRY)
    held.add(people_table)
    graph = Graph.compile(make_graph(people_table, series.split, words, tables.collect, edges=[
        ("people.result", "series-split.value"), ("series-split.result", "table-collect.records"), ("words.result", "table-collect.word"),
    ], values={"table-collect.columns": ["word"]}), held)
    assert graph.problems == []
    cause = graph.run().state.cause
    assert (cause.type, cause.loc, cause.ctx) == ("columns_not_aligned", ("table-collect",), expected)
    assert "'word' holds 2 values" in cause.msg and f"{people} record" in cause.msg and "once per record" in cause.msg


def test_collect_says_where_a_value_made_once_goes_when_it_is_wired_as_a_column_per_record():
    @_node("two-people")
    def two_people() -> Table[Contact]:
        return pl.DataFrame({"name": ["Ida", "Bo"], "email": [None, None]}, schema={"name": pl.String, "email": pl.String})

    held = NodeRegistry()
    held.add_subregistry(REGISTRY)
    held.add(two_people)
    graph = Graph.compile(make_graph(two_people, series.split, sentence, tables.collect, edges=[
        ("two-people.result", "series-split.value"), ("series-split.result", "table-collect.records"), ("sentence.result", "table-collect.team"),
    ], values={"table-collect.columns": ["team"]}), held)
    assert graph.problems == []
    cause = graph.run().state.cause
    assert (cause.type, cause.ctx["values"]) == ("columns_not_aligned", 1)
    assert "holds 1 value for" in cause.msg and "if it is one value for every record, name it in 'constant_columns'" in cause.msg


def test_collect_of_no_records_is_a_table_with_every_column_and_no_lines():
    graph = load(nobody, sentence, tables.collect, header, edges=[
        ("nobody.result", "table-collect.records"), ("sentence.result", "table-collect.team"), ("table-collect.result", "header.table"),
    ], values={"table-collect.constant_columns": ["team"]})
    run = graph.run()
    assert run.status == "complete", run.state
    assert run.results["table-collect"]["result"].height == 0 and run.results["header"]["result"] == ["name", "email", "team"]


def test_collect_of_records_of_unknown_columns_is_a_table_of_any_columns():
    graph = load(csv, series.split, shout, tables.collect, edges=[
        ("csv.result", "series-split.value"), ("series-split.result.name", "shout.text"), ("series-split.result", "table-collect.records"),
        ("shout.result", "table-collect.loud"),
    ], values={"table-collect.columns": ["loud"]})
    collected = output(graph, "table-collect")
    assert graph.problems == [] and collected.type == "Table" and "prefixItems" not in collected.json_schema["properties"]["columns"]
    assert graph.run().results["table-collect"]["result"].to_dicts() == [{"name": "Ann", "age": 31, "loud": "ANN"}, {"name": "Bo", "age": 47, "loud": "BO"}]


def test_collect_takes_records_only():
    (problem,) = load(words, tables.collect, edges=[("words.result", "table-collect.records")]).problems
    assert (problem.type, problem.loc, problem.ctx["got"], problem.ctx["expected"]) == ("type_mismatch", ("table-collect", "records"), "Series[str]", "Table")


# -- text-list and text-split -------------------------------------------------------------------------------


def test_text_list_gives_its_texts_once_and_once_per_item_of_what_is_wired_to_for_each():
    alone = load(text.text_list, shout, edges=[("text-list.result", "shout.text")], values={"text-list.items": ["a", "b"]})
    assert alone.runs_on["text-list"].mode == "once" and alone.run().results["shout"]["result"] == ["A", "B"]
    inside = load(words, text.text_list, edges=[("words.result", "text-list.for_each")], values={"text-list.items": ["x", "y", "z"]})
    assert inside.runs_on["text-list"].series == "words.result"
    assert inside.run().results["text-list"]["result"] == [["x", "y", "z"], ["x", "y", "z"]]


def test_text_split_is_titled_split_text_splits_on_a_newline_and_an_empty_text_has_no_parts():
    assert text.split.title == "Split text" and text.split.inputs["separator"].default == "\n"
    assert text.split("a\nb") == ["a", "b"] and text.split("a,b", separator=",") == ["a", "b"] and text.split("") == []
    assert series.split.title == "Split" and tables.collect.title == "Collect into a table" and text.text_list.title == "Text list"
