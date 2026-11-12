"""What an author is told for the first mistakes with a table, a list or two Series: the problem, where, and the fix."""

import typing
from typing import Any, TypedDict

from conductor import Graph, GraphRecord, NodeRegistry, Series, Table, node


class Contact(TypedDict):
    name: str
    email: str


class Amount(TypedDict):
    name: str
    amount: float


def _node(node_id: str):
    return node(node_id, title=node_id.capitalize(), description=f"The {node_id} node.")


@_node("contacts")
def contacts(path: str) -> Table[Contact]: ...


@_node("csv")
def csv(path: str) -> Table: ...


@_node("tags")
def tags(path: str) -> list[str]: ...


@_node("people")
def people(path: str) -> list[Contact]: ...


@_node("records")
def records(table: Table[Contact]) -> Series[Contact]: ...


@_node("send")
def send(to: str) -> str: ...


@_node("report")
def report(table: Table[Amount]) -> str: ...


@_node("count")
def count(values: Series[str]) -> int: ...


@_node("documents")
def documents(folder: str) -> Series[str]: ...


@_node("questions")
def questions(about: Any = None) -> Series[str]: ...


@_node("ask")
def ask(question: str, document: str) -> str: ...


REGISTRY = NodeRegistry()
for _definition in (contacts, csv, tags, people, records, send, report, count, documents, questions, ask):
    REGISTRY.add(_definition)


def problem(nodes: dict[str, str], edges: list[tuple[str, str]]):
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": i, "node": n} for i, n in nodes.items()], "edges": [list(edge) for edge in edges]}), REGISTRY)
    (found,) = graph.problems
    return found


# -- a Split is missing --------------------------------------------------------------------------------


def test_a_whole_table_into_an_input_of_one_value_says_to_split_first_and_lists_the_fields():
    found = problem({"read": "contacts", "send": "send"}, [("read.result", "send.to")])
    assert (found.type, found.loc) == ("type_mismatch", ("send", "to"))
    assert found.ctx == {
        "node_id": "send", "field": "to", "source": "read.result", "got": "Table[name: str, email: str]", "expected": "str",
        "hint": "split_first", "whole": "table", "fields": ["name", "email"],
    }
    assert found.msg == (
        "'read.result' is a whole table and 'send.to' takes one str. To run 'send' once for each record or item in it, "
        "put a Split between them and wire Split's result, or one field of it."
    )


def test_a_table_of_any_columns_says_to_split_first_and_knows_no_fields():
    found = problem({"read": "csv", "send": "send"}, [("read.result", "send.to")])
    assert (found.ctx["hint"], found.ctx["whole"], found.ctx["fields"]) == ("split_first", "table", [])


def test_a_whole_list_into_an_input_of_one_value_says_to_split_first():
    found = problem({"tags": "tags", "send": "send"}, [("tags.result", "send.to")])
    assert (found.type, found.loc, found.ctx["hint"], found.ctx["whole"], found.ctx["fields"]) == ("type_mismatch", ("send", "to"), "split_first", "list", [])
    assert found.msg.startswith("'tags.result' is a whole list and 'send.to' takes one str.")
    assert problem({"people": "people", "send": "send"}, [("people.result", "send.to")]).ctx["fields"] == ["name", "email"]


def test_a_whole_list_into_a_series_input_is_the_ordinary_mismatch():
    found = problem({"tags": "tags", "count": "count"}, [("tags.result", "count.values")])
    assert found.type == "type_mismatch" and "hint" not in found.ctx


def test_a_field_named_on_a_whole_table_is_no_output_and_says_to_split_first():
    found = problem({"read": "contacts", "send": "send"}, [("read.result.email", "send.to")])
    assert (found.type, found.loc) == ("unknown_output", ("send", "to"))
    assert found.ctx == {"source": "read.result.email", "outputs": ["read.result"], "record_field": "email", "hint": "split_first", "whole": "table", "fields": ["name", "email"]}
    assert found.msg == (
        "'read.result.email' names a field of a whole table, which has no field to wire. Split it first: put a Split after it "
        "and wire the field from Split's result."
    )


def test_a_field_named_on_a_whole_list_says_to_split_first_too():
    found = problem({"people": "people", "send": "send"}, [("people.result.email", "send.to")])
    assert (found.type, found.ctx["hint"], found.ctx["whole"], found.ctx["fields"]) == ("unknown_output", "split_first", "list", ["name", "email"])


def test_a_field_of_a_record_of_unknown_columns_is_any_value_read_when_it_arrives():
    @_node("lines")
    def lines(table: Table) -> Series[dict]: ...

    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    registry.add(lines)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "read", "node": "csv"}, {"id": "lines", "node": "lines"}, {"id": "send", "node": "send"}], "edges": [["read.result", "lines.table"], ["lines.result.email", "send.to"]]}), registry)
    assert graph.is_runnable and graph.runs_on["send"].series == "lines.result"


# -- a column is missing ---------------------------------------------------------------------------------


def test_a_table_without_a_column_the_input_names_says_which_and_what_the_table_has():
    found = problem({"read": "contacts", "report": "report"}, [("read.result", "report.table")])
    assert (found.type, found.loc) == ("type_mismatch", ("report", "table"))
    assert found.ctx == {
        "node_id": "report", "field": "table", "source": "read.result", "got": "Table[name: str, email: str]", "expected": "Table[name: str, amount: float]",
        "hint": "add_column", "column": "amount", "column_type": "float", "has": ["name", "email"],
    }
    assert found.msg == "'report.table' needs the column 'amount' (float); 'read.result' has name, email. Add the column in a node between, or connect a table that has it."


def test_records_gathered_without_a_column_the_table_input_names_say_the_same():
    found = problem({"read": "contacts", "records": "records", "report": "report"}, [("read.result", "records.table"), ("records.result", "report.table")])
    assert (found.ctx["hint"], found.ctx["column"], found.ctx["has"], found.ctx["source"]) == ("add_column", "amount", ["name", "email"], "records.result")


def test_a_column_of_another_type_is_the_ordinary_mismatch_naming_both_tables_columns():
    @_node("amounts")
    def amounts(path: str) -> Table[TypedDict("Texts", {"name": str, "amount": str})]: ...  # type: ignore[valid-type]  # noqa: UP013

    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    registry.add(amounts)
    (found,) = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "amounts"}, {"id": "r", "node": "report"}], "edges": [["a.result", "r.table"]]}), registry).problems
    assert found.type == "type_mismatch" and "hint" not in found.ctx and found.ctx["got"] == "Table[name: str, amount: str]"


# -- two Series that don't nest ---------------------------------------------------------------------------


def test_two_series_that_do_not_nest_name_the_wire_that_would_nest_them():
    found = problem({"upload": "documents", "list": "questions", "ask": "ask"}, [("list.result", "ask.question"), ("upload.result", "ask.document")])
    assert (found.type, found.loc) == ("series_not_nested", ("ask",))
    assert found.ctx == {"node_id": "ask", "series": ["list.result", "upload.result"], "inputs": ["question", "document"], "wire_from": "upload.result", "wire_into": "list"}
    assert found.msg == (
        "'ask' reads items of list.result, upload.result, and neither is made inside the other, so nothing says which item goes with which. "
        "Make one inside the other (wire 'upload.result' into 'list', the node that makes the other), or give one of them whole to a Series input."
    )


def test_the_wire_it_names_nests_them():
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "upload", "node": "documents"}, {"id": "list", "node": "questions"}, {"id": "ask", "node": "ask"}],
        "edges": [["list.result", "ask.question"], ["upload.result", "ask.document"], ["upload.result", "list.about"]],
    }), REGISTRY)
    assert graph.is_runnable and graph.runs_on["ask"].series == "list.result" and graph.runs_on["list"].series == "upload.result"


# -- the row type of a table type ---------------------------------------------------------------------------


def test_a_hook_reads_the_row_type_of_the_table_arriving():
    seen: list[Any] = []

    def outputs(table: type) -> dict[str, Any]:
        seen.append(typing.get_args(table))
        return {"result": int}

    @node("look", title="Look", description="Looks at a table.", compute_outputs=outputs)
    def look(table: Table) -> dict[str, Any]:
        return {"result": 0}

    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    registry.add(look)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "read", "node": "contacts"}, {"id": "look", "node": "look"}], "edges": [["read.result", "look.table"]]}), registry)
    assert graph.is_runnable and seen == [(Contact,)]
