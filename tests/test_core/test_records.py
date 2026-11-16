"""Tests for a node that returns a table where its type says a ``Series`` of records.

A node declared ``-> Series[Contact]`` may return a polars table, one row per record. The run stores the table as it
is. An edge that names a field, such as ``contacts.result.email``, reads that column. A node further on still gets
plain records. And the table is checked column by column when the node returns it.
"""

import json
from typing import Any, NotRequired, TypedDict

import polars as pl
import pytest
from conductor import (
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    Run,
    Series,
    Source,
    Table,
    make_graph,
    node,
    subgraph,
)

from tests.test_core.nodes import registry_of


class Contact(TypedDict):
    name: str
    email: str | None


FRAME = pl.DataFrame({"name": ["ida", "bo"], "email": ["ida@x.dk", None], "age": [41, 52]})
RECEIVED: list[Any] = []


@node("contacts", title="Contacts", description="For a test.")
def contacts(text: str) -> Series[Contact]:
    return FRAME  # type: ignore[return-value]  a table may stand for a Series of its records


@node("listed", title="Listed", description="For a test.")
def listed(text: str) -> Series[Contact]:
    return [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]


@node("greet", title="Greet", description="For a test.")
def greet(name: str) -> str:
    return f"hi {name}"


@node("one", title="One", description="For a test.")
def one(contact: Contact) -> str:
    RECEIVED.append(contact)
    return contact["name"]


@node("every", title="Every", description="For a test.")
def every(found: Series[Contact]) -> int:
    RECEIVED.append(found)
    return len(found)


@node("names", title="Names", description="For a test.")
def names(found: Series[str]) -> str:
    RECEIVED.append(found)
    return "+".join(found)


@node("mails", title="Mails", description="For a test.")
def mails(found: Series[str | None]) -> int:
    RECEIVED.append(found)
    return len(found)


def _graph(source: Any = contacts) -> Any:
    reads = {
        "greet": ("name", "name"),
        "one": ("contact", None),
        "every": ("found", None),
        "names": ("found", "name"),
        "mails": ("found", "email"),
    }
    readers = {
        node_id: NodeRecord(
            node=node_id, inputs={name: InputEntry(sources=[Source("c", "result", field)])}
        )
        for node_id, (name, field) in reads.items()
    }
    return Graph.compile(
        GraphRecord(
            nodes={
                "c": NodeRecord(node=source.id, inputs={"text": InputEntry(graph_input="text")}),
                **readers,
            },
            outputs={node_id: Source(node_id, "result") for node_id in reads},
        ),
        registry_of(source, greet, one, every, names, mails),
    )


@pytest.mark.parametrize(
    "source", [contacts, listed], ids=["returned as a table", "returned as a list"]
)
def test_a_node_receives_plain_records_whichever_way_the_series_was_returned(source: Any) -> None:
    RECEIVED.clear()
    run = _graph(source).run(text="x")

    assert run.outputs == {
        "greet": ["hi ida", "hi bo"],
        "one": ["ida", "bo"],
        "every": 2,
        "names": "ida+bo",
        "mails": 2,
    }
    records = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert RECEIVED == [records, ["ida@x.dk", None], ["ida", "bo"], *records] and all(
        type(record) is dict for record in RECEIVED[3:]
    )  # the nodes are called in the order of their ids
    assert run.results["c"]["result"] == records  # a host reads plain records too


def test_a_series_returned_as_a_table_is_kept_and_saved_as_that_table_with_only_its_declared_columns() -> (
    None
):
    graph = _graph()
    run = graph.run(text="x")

    saved = json.loads(run.to_json())
    assert saved["outcomes"]["c"]["outputs"]["result"] == [
        {
            "columns": [
                {"name": "name", "type": "str", "values": ["ida", "bo"]},
                {"name": "email", "type": "str", "values": ["ida@x.dk", None]},
            ]
        }
    ]
    assert json.loads(_graph(listed).run(text="x").to_json())["outcomes"]["c"]["outputs"][
        "result"
    ] == [[{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]]

    read_back = Run.from_json(run.to_json(), graph)
    assert read_back.to_json() == run.to_json() and read_back.results == run.results
    RECEIVED.clear()
    assert read_back.forget("one").resume().outputs == run.outputs and RECEIVED == [
        {"name": "ida", "email": "ida@x.dk"},
        {"name": "bo", "email": None},
    ]


@pytest.mark.parametrize(
    ("frame", "reason"),
    [
        (
            pl.DataFrame({"name": ["ida", None], "email": ["a", "b"]}),
            "column 'name' has a blank cell, and its type doesn't allow blanks",
        ),
        (
            pl.DataFrame({"email": ["a", "b"]}),
            "the column 'name' is missing, and the row type needs it",
        ),
        (
            pl.DataFrame({"name": [1, 2], "email": ["a", "b"]}),
            "column 'name': Input should be a valid string",
        ),
    ],
)
def test_a_table_returned_for_a_series_of_records_is_checked_column_by_column_and_fails_the_run(
    frame: pl.DataFrame, reason: str
) -> None:
    @node("wrong", title="Wrong", description="For a test.")
    def wrong(text: str) -> Series[Contact]:
        return frame  # type: ignore[return-value]

    run = Graph.compile(make_graph(wrong), registry_of(wrong)).run(text="x")

    assert (run.status, run.cause.type, run.cause.ctx["reason"], run.cause.ctx["column"]) == (
        "failed",
        "invalid_output",
        reason,
        "name",
    )


def test_a_table_returned_for_a_series_of_texts_fails_the_run() -> None:
    @node("texts", title="Texts", description="For a test.")
    def texts(text: str) -> Series[str]:
        return FRAME  # type: ignore[return-value]

    run = Graph.compile(make_graph(texts), registry_of(texts)).run(text="x")

    assert (run.status, run.cause.type) == ("failed", "invalid_output")


def test_a_table_of_any_columns_is_a_series_of_its_records() -> None:
    @node("anything", title="Anything", description="For a test.")
    def anything(text: str) -> Series[Any]:
        return FRAME  # type: ignore[return-value]

    @node("show", title="Show", description="For a test.")
    def show(record: Any) -> str:
        return repr(record)

    graph = Graph.compile(
        make_graph(anything, show, edges=[("anything.result", "show.record")]),
        registry_of(anything, show),
    )
    run = graph.run(text="x")

    assert run.outputs["result"] == [
        "{'name': 'ida', 'email': 'ida@x.dk', 'age': 41}",
        "{'name': 'bo', 'email': None, 'age': 52}",
    ]
    assert Run.from_json(run.to_json(), graph).results == run.results


def test_a_series_of_records_kept_as_a_table_counts_its_rows_against_the_item_limit() -> None:
    run = _graph().run(text="x", max_items=1)

    assert (run.status, run.cause.type, run.cause.ctx["items"]) == ("stopped", "item_limit", 2)


def test_a_series_of_records_returned_per_item_inside_a_subgraph_reaches_a_series_input_outside_as_plain_records() -> (
    None
):
    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Series[str]:
        return text.split(",")

    @node("count", title="Count", description="For a test.")
    def count(found: Series[Contact]) -> int:
        RECEIVED.append(found)
        return len(found)

    registry = registry_of(parts, contacts, count)
    inner = GraphRecord.model_validate(
        {
            "nodes": {
                "p": {"node": "parts", "inputs": {"text": {"graph_input": "text"}}},
                "c": {
                    "node": "contacts",
                    "inputs": {"text": {"sources": [{"node": "p", "output": "result"}]}},
                },
            },
            "outputs": {"result": {"node": "c", "output": "result"}},
        }
    )
    registry.add(subgraph("inner", title="Inner", description="For a test.")(inner))
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "e": {"node": "inner", "inputs": {"text": {"graph_input": "text"}}},
                    "count": {
                        "node": "count",
                        "inputs": {"found": {"sources": [{"node": "e", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "count", "output": "result"}},
            }
        ),
        registry,
    )
    RECEIVED.clear()
    run = graph.run(text="x,y")

    records = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert (run.status, run.outputs["result"]) == ("complete", [2, 2]) and RECEIVED == [
        records,
        records,
    ]  # one call per part
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


# -- how a run stores a Series, and what reads it ---------------------------------------------------------


def test_one_record_of_five_without_the_field_an_edge_reads_fails_only_the_call_for_that_record() -> (
    None
):
    @node("five", title="Five", description="For a test.")
    def five(text: str) -> Series[dict]:
        return [{"name": "a"}, {"name": "b"}, {"other": "c"}, {"name": "d"}, {"name": "e"}]

    graph = Graph.compile(
        make_graph(five, greet, edges=[("five.result.name", "greet.name")]),
        registry_of(five, greet),
    )
    run = graph.run(text="x")

    cause = run.cause
    assert (run.status, cause.type, cause.loc, cause.ctx["record_field"]) == (
        "failed",
        "invalid_edge_value",
        ("greet", 2),
        "name",
    )
    assert run.results["greet"]["result"][:3] == [
        "hi a",
        "hi b",
        None,
    ]  # the calls for the two records before it ran


def test_a_table_input_wired_to_a_series_stored_as_a_table_is_called_with_that_table() -> None:
    got: list[Any] = []

    @node("rows", title="Rows", description="For a test.")
    def rows(table: Table[Contact]) -> int:
        got.append(table)
        return table.height

    graph = Graph.compile(
        make_graph(contacts, rows, edges=[("contacts.result", "rows.table")]),
        registry_of(contacts, rows),
    )
    run = graph.run(text="x")

    assert run.outputs == {"result": 2} and got[0].to_dicts() == [
        {"name": "ida", "email": "ida@x.dk"},
        {"name": "bo", "email": None},
    ]


def test_a_table_input_loses_the_rows_another_input_of_the_same_series_has_no_value_for() -> None:
    got: list[Any] = []

    @node("mail", title="Mail", description="For a test.")
    def mail(email: str) -> str:
        return email.upper()

    @node("both", title="Both", description="For a test.")
    def both(table: Table[Contact], mails: Series[str]) -> int:
        got.append((table, mails))
        return table.height

    graph = Graph.compile(
        make_graph(
            contacts,
            mail,
            both,
            edges=[
                ("contacts.result.email", "mail.email"),
                ("contacts.result", "both.table"),
                ("mail.result", "both.mails"),
            ],
        ),
        registry_of(contacts, mail, both),
    )
    run = graph.run(text="x")

    assert run.outputs["result"] == 1  # bo has no email, so mail isn't called for him
    assert got[0][0].to_dicts() == [{"name": "ida", "email": "ida@x.dk"}] and got[0][1] == [
        "IDA@X.DK"
    ]


def test_a_call_that_takes_a_table_is_reused_when_the_table_has_the_same_cells() -> None:
    calls: list[pl.DataFrame] = []

    @node("height", title="Height", description="For a test.")
    def height(table: Table) -> int:
        calls.append(table)
        return table.height

    graph = Graph.compile(make_graph(height), registry_of(height))
    frame = pl.DataFrame({"name": ["ida", "bo"], "age": [41, 52]})
    run = graph.run(table=frame)

    run = run.again(table=pl.DataFrame({"name": ["ida", "bo"], "age": [41, 52]}))
    assert len(calls) == 1  # the same cells, so the call is reused
    for changed in (
        pl.DataFrame({"name": ["ida", "bo"], "age": [41, 53]}),
        frame.rename({"age": "years"}),
    ):
        run = run.again(table=changed)
    assert len(calls) == 3


# -- a table two outputs share is written under each in a saved run --------------------------------------------


@node("people", title="People", description="For a test.")
def people(text: str) -> Table[Contact]:
    return FRAME.select("name", "email")


@node("records", title="Records", description="For a test.")
def records(table: Table[Contact]) -> Series[Contact]:
    return table  # type: ignore[return-value]  a table may stand for a Series of its records


def _people_as_records() -> Any:
    return Graph.compile(
        make_graph(
            people,
            records,
            greet,
            edges=[("people.result", "records.table"), ("records.result.name", "greet.name")],
        ),
        registry_of(people, records, greet),
    )


def test_a_table_returned_as_the_records_of_another_output_is_written_under_both_and_reads_back_equal() -> (
    None
):
    graph = _people_as_records()
    run = graph.run(text="x")
    saved = json.loads(run.to_json())

    assert (
        saved["outcomes"]["records"]["outputs"]["result"]
        == saved["outcomes"]["people"]["outputs"]["result"]
    )
    read_back = Run.from_json(run.to_json(), graph)
    table, again = run.results["people"]["result"], read_back.results["people"]["result"]
    assert again.equals(table) and again.schema == table.schema
    assert read_back.results["records"]["result"] == run.results["records"]["result"]
    assert read_back.to_json() == run.to_json() and read_back.outputs == run.outputs
    assert read_back.forget("greet").resume().outputs == {"result": ["hi ida", "hi bo"]}


def test_a_saved_run_with_null_for_a_done_call_whose_output_takes_no_none_is_refused() -> None:
    graph = _people_as_records()
    saved = json.loads(graph.run(text="x").to_json())
    saved["outcomes"]["records"]["outputs"]["result"] = [None]  # not listed under "skipped"

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)
    (found,) = refused.value.problems
    assert (found.type, found.ctx["where"]) == ("saved_run_unreadable", "outcomes.records")
    assert "call 0 is done and has null for 'result'" in found.msg


# -- the JSON a host gets for a table ------------------------------------------------------------------------


def test_the_json_a_host_gets_for_a_table_goes_into_polars_and_into_pandas_in_one_line_each() -> (
    None
):
    import pandas as pd

    graph = Graph.compile(make_graph(people), registry_of(people))
    table = json.loads(graph.run(text="x").outputs_to_json())["result"]

    in_polars = pl.DataFrame({c["name"]: c["values"] for c in table["columns"]})
    in_pandas = pd.DataFrame({c["name"]: c["values"] for c in table["columns"]})

    assert in_polars.to_dicts() == [
        {"name": "ida", "email": "ida@x.dk"},
        {"name": "bo", "email": None},
    ]
    assert list(in_pandas.columns) == ["name", "email"] and in_pandas["name"].tolist() == [
        "ida",
        "bo",
    ]
    # pandas shows a null cell as its own missing value
    assert in_pandas["email"][0] == "ida@x.dk" and pd.isna(in_pandas["email"][1])


class _MaybeNamed(TypedDict):
    name: NotRequired[str]
    email: str


class _Partial(TypedDict, total=False):
    name: str


@pytest.mark.parametrize("row", [_MaybeNamed, _Partial])
@pytest.mark.parametrize("hint", ["input table", "output table", "input series", "output series"])
def test_a_row_type_that_may_leave_a_field_out_is_refused_when_the_node_is_declared(
    row: Any, hint: str
) -> None:
    kind = Table if "table" in hint else Series

    if hint.startswith("input"):

        def declared(rows: kind[row]) -> int:  # type: ignore[valid-type]
            return 0
    else:

        def declared() -> kind[row]:  # type: ignore[valid-type]
            return []

    with pytest.raises(NodeDeclarationError) as refused:
        node("declared", title="Declared", description="For a test.")(declared)
    assert refused.value.problems[0].type == "row_field_optional"
    assert "name" in refused.value.problems[0].msg and "| None" in refused.value.problems[0].msg
