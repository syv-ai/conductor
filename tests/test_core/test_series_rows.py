"""Tests for a node that returns a table where its type says a ``Series`` of rows.

A node declared ``-> Series[Contact]`` may return a polars table, one row per item. The run stores the table as it
is. An edge that names a field, such as ``contacts.result.email``, reads that column. A node further on still gets
plain rows. And the table is checked column by column when the node returns it.
"""

import json
from enum import Enum
from typing import Annotated, Any, Literal, NotRequired, TypedDict

import polars as pl
import pytest
from conductor import (
    Graph,
    GraphInputRecord,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    Param,
    Run,
    Series,
    Source,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationInfo, field_validator

from tests.test_core.nodes import input_entry, registry_of


class Contact(TypedDict):
    name: str
    email: str | None


FRAME = pl.DataFrame({"name": ["ida", "bo"], "email": ["ida@x.dk", None], "age": [41, 52]})
RECEIVED: list[Any] = []


@node("contacts", title="Contacts", description="For a test.")
def contacts(text: str) -> Series[Contact]:
    return FRAME  # type: ignore[return-value]  a table may stand for a Series of its rows


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
            node=node_id, inputs={name: input_entry({"sources": [Source("c", "result", field)]})}
        )
        for node_id, (name, field) in reads.items()
    }
    return Graph.compile(
        GraphRecord(
            inputs={"text": GraphInputRecord()},
            nodes={
                "c": NodeRecord(
                    node=source.id, inputs={"text": input_entry({"graph_input": "text"})}
                ),
                **readers,
            },
            outputs={node_id: Source(node_id, "result") for node_id in reads},
        ),
        registry_of(source, greet, one, every, names, mails),
    )


@pytest.mark.parametrize(
    "source", [contacts, listed], ids=["returned as a table", "returned as a list"]
)
def test_a_node_receives_plain_rows_whichever_way_the_series_was_returned(source: Any) -> None:
    RECEIVED.clear()
    run = _graph(source).run({"text": "x"})

    assert run.outputs == {
        "greet": ["hi ida", "hi bo"],
        "one": ["ida", "bo"],
        "every": 2,
        "names": "ida+bo",
        "mails": 2,
    }
    rows = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert RECEIVED == [rows, ["ida@x.dk", None], ["ida", "bo"], *rows] and all(
        type(row) is dict for row in RECEIVED[3:]
    )  # the nodes are called in the order of their ids
    assert run.results["c"]["result"] == rows  # a host reads plain rows too


def test_a_series_returned_as_a_table_is_kept_and_saved_as_that_table_with_only_its_declared_columns() -> (
    None
):
    graph = _graph()
    run = graph.run({"text": "x"})

    saved = json.loads(run.to_json())
    assert saved["outcomes"]["c"]["outputs"]["result"] == [
        {
            "columns": [
                {"name": "name", "type": "str", "values": ["ida", "bo"]},
                {"name": "email", "type": "str", "values": ["ida@x.dk", None]},
            ]
        }
    ]
    assert json.loads(_graph(listed).run({"text": "x"}).to_json())["outcomes"]["c"]["outputs"][
        "result"
    ] == [[{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]]

    read_back = Run.from_json(run.to_json(), graph)
    assert read_back.to_json() == run.to_json() and read_back.results == run.results
    RECEIVED.clear()
    assert read_back.with_forgotten("one").resume().outputs == run.outputs and RECEIVED == [
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
def test_a_table_returned_for_a_series_of_rows_is_checked_column_by_column_and_fails_the_run(
    frame: pl.DataFrame, reason: str
) -> None:
    @node("wrong", title="Wrong", description="For a test.")
    def wrong(text: str) -> Series[Contact]:
        return frame  # type: ignore[return-value]

    run = Graph.compile(make_graph(wrong), registry_of(wrong)).run({"text": "x"})

    assert (run.status, run.cause.type, run.cause.ctx["reason"], run.cause.ctx["column"]) == (
        "failed",
        "invalid_output",
        reason,
        "name",
    )


@pytest.mark.parametrize(
    "returned", [pl.DataFrame({"email": ["a"]}), [{"email": "a"}]], ids=["frame", "rows"]
)
def test_a_missing_column_is_column_missing_whether_the_table_is_a_frame_or_rows(
    returned: Any,
) -> None:
    """A table that lacks a column fails the same way in both forms, so a host checks one code."""

    @node("wrong", title="Wrong", description="For a test.")
    def wrong(text: str) -> Table[Contact]:
        return returned  # type: ignore[no-any-return]

    cause = Graph.compile(make_graph(wrong), registry_of(wrong)).run({"text": "x"}).cause

    assert (cause.ctx["reason"], [error["type"] for error in cause.ctx["errors"]]) == (
        "the column 'name' is missing, and the row type needs it",
        ["column_missing"],
    )


def test_a_field_missing_inside_a_cell_is_not_called_a_missing_column() -> None:
    """A value in a cell that lacks a field names the column and the field, and doesn't say the column is missing.

    The error keeps pydantic's type ``missing``, since the column is there.
    """

    class Price(TypedDict):
        amount: int
        currency: str

    class Priced(TypedDict):
        name: str
        price: Price

    @node("priced", title="Priced", description="For a test.")
    def priced(text: str) -> Table[Priced]:
        return [{"name": "a", "price": {"amount": 1}}]  # type: ignore[return-value]

    cause = Graph.compile(make_graph(priced), registry_of(priced)).run({"text": "x"}).cause

    assert (cause.ctx["reason"], cause.ctx["column"], cause.ctx["errors"][0]["type"]) == (
        "column 'price': the field 'currency' is required",
        "price",
        "missing",
    )


@pytest.mark.parametrize("returns", [Series[str], Series[dict[str, int]]])
def test_a_table_returned_for_texts_or_dicts_of_one_type_fails_the_run(returns: Any) -> None:
    @node("texts", title="Texts", description="For a test.")
    def texts(text: str) -> returns:  # type: ignore[valid-type]
        return FRAME

    run = Graph.compile(make_graph(texts), registry_of(texts)).run({"text": "x"})

    assert (run.status, run.cause.type) == ("failed", "invalid_output")


@pytest.mark.parametrize("returns", [Series[Any], Series[dict], Series[dict[str, Any]]])
def test_a_table_of_any_columns_is_a_series_of_its_rows(returns: Any) -> None:
    @node("anything", title="Anything", description="For a test.")
    def anything(text: str) -> returns:  # type: ignore[valid-type]
        return FRAME

    @node("show", title="Show", description="For a test.")
    def show(row: Any) -> str:
        return repr(row)

    graph = Graph.compile(
        make_graph(anything, show, edges=[("anything.result", "show.row")]),
        registry_of(anything, show),
    )
    run = graph.run({"text": "x"})

    assert run.outputs["result"] == [
        "{'name': 'ida', 'email': 'ida@x.dk', 'age': 41}",
        "{'name': 'bo', 'email': None, 'age': 52}",
    ]
    assert Run.from_json(run.to_json(), graph).results == run.results


def test_a_series_of_rows_kept_as_a_table_counts_its_rows_against_the_item_limit() -> None:
    run = _graph().run({"text": "x"}, max_items=1)

    assert (run.status, run.cause.type, run.cause.ctx["items"]) == ("stopped", "item_limit", 2)


def test_a_series_of_rows_returned_per_item_inside_a_subgraph_reaches_a_series_input_outside_as_plain_rows() -> (
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
            "inputs": {"text": {}},
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
                "inputs": {"text": {}},
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
    run = graph.run({"text": "x,y"})

    rows = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert (run.status, run.outputs["result"]) == ("complete", [2, 2]) and RECEIVED == [
        rows,
        rows,
    ]  # one call per part
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


# -- how a run stores a Series, and what reads it ---------------------------------------------------------


def test_one_row_of_five_without_the_field_an_edge_reads_fails_only_the_call_for_that_row() -> None:
    @node("five", title="Five", description="For a test.")
    def five(text: str) -> Series[dict]:
        return [{"name": "a"}, {"name": "b"}, {"other": "c"}, {"name": "d"}, {"name": "e"}]

    graph = Graph.compile(
        make_graph(five, greet, edges=[("five.result.name", "greet.name")]),
        registry_of(five, greet),
    )
    run = graph.run({"text": "x"})

    cause = run.cause
    assert (run.status, cause.type, cause.loc, cause.ctx["field"]) == (
        "failed",
        "invalid_edge_value",
        ("greet", 2),
        "name",
    )
    assert cause.ctx["errors"] == [
        {
            "type": "no_such_field",
            "loc": [],
            "msg": "it has no field 'name'",
            "ctx": {"field": "name"},
        }
    ]
    assert run.results["greet"]["result"][:3] == [
        "hi a",
        "hi b",
        None,
    ]  # the calls for the two rows before it ran


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
    run = graph.run({"text": "x"})

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
    run = graph.run({"text": "x"})

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
    run = graph.run({"table": frame})

    run = run.again({"table": pl.DataFrame({"name": ["ida", "bo"], "age": [41, 52]})})
    assert len(calls) == 1  # the same cells, so the call is reused
    for changed in (
        pl.DataFrame({"name": ["ida", "bo"], "age": [41, 53]}),
        frame.rename({"age": "years"}),
    ):
        run = run.again({"table": changed})
    assert len(calls) == 3


# -- a table two outputs share is written under each in a saved run --------------------------------------------


@node("people", title="People", description="For a test.")
def people(text: str) -> Table[Contact]:
    return FRAME.select("name", "email")


@node("rows", title="Rows", description="For a test.")
def rows(table: Table[Contact]) -> Series[Contact]:
    return table  # type: ignore[return-value]  a table may stand for a Series of its rows


def _people_as_rows() -> Any:
    return Graph.compile(
        make_graph(
            people,
            rows,
            greet,
            edges=[("people.result", "rows.table"), ("rows.result.name", "greet.name")],
        ),
        registry_of(people, rows, greet),
    )


def test_a_table_returned_as_the_rows_of_another_output_is_written_under_both_and_reads_back_equal() -> (
    None
):
    graph = _people_as_rows()
    run = graph.run({"text": "x"})
    saved = json.loads(run.to_json())

    assert (
        saved["outcomes"]["rows"]["outputs"]["result"]
        == saved["outcomes"]["people"]["outputs"]["result"]
    )
    read_back = Run.from_json(run.to_json(), graph)
    table, again = run.results["people"]["result"], read_back.results["people"]["result"]
    assert again.equals(table) and again.schema == table.schema
    assert read_back.results["rows"]["result"] == run.results["rows"]["result"]
    assert read_back.to_json() == run.to_json() and read_back.outputs == run.outputs
    assert read_back.with_forgotten("greet").resume().outputs == {"result": ["hi ida", "hi bo"]}


def test_a_saved_run_with_null_for_a_done_call_whose_output_takes_no_none_is_refused() -> None:
    graph = _people_as_rows()
    saved = json.loads(graph.run({"text": "x"}).to_json())
    saved["outcomes"]["rows"]["outputs"]["result"] = [None]  # not listed under "skipped"

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)
    (found,) = refused.value.problems
    assert (found.type, found.ctx["where"]) == ("saved_run_unreadable", "outcomes.rows")
    assert "call 0 is done and has null for 'result'" in found.msg


# -- the JSON a host gets for a table ------------------------------------------------------------------------


def test_the_json_a_host_gets_for_a_table_goes_into_polars_and_into_pandas_in_one_line_each() -> (
    None
):
    import pandas as pd

    graph = Graph.compile(make_graph(people), registry_of(people))
    table = json.loads(graph.run({"text": "x"}).outputs_to_json())["result"]

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


# -- rows collected into a table ----------------------------------------------------------------------------------

CHECKED: list[int] = []


def _counted(value: int) -> int:
    CHECKED.append(value)
    return value


class Counted(TypedDict):
    name: str
    amount: Annotated[int, AfterValidator(_counted)]


class Positive(TypedDict):
    name: str
    amount: Annotated[int, Param(ge=0)]


class Tagged(TypedDict):
    name: str
    tags: list[float]


@node("counted-rows", title="Counted rows", description="For a test.")
def counted_rows(k: int) -> Series[Counted]:
    return [{"name": str(i), "amount": i} for i in range(k)]


@node("counted-words", title="Counted words", description="For a test.")
def counted_words(k: int) -> Series[str]:
    return [str(i) for i in range(k)]


@node("counted-row", title="Counted row", description="For a test.")
def counted_row(word: str) -> Counted:
    return {"name": word, "amount": int(word)}


@node("counted-height", title="Counted height", description="For a test.")
def counted_height(table: Table[Counted]) -> int:
    return table.height


@pytest.mark.parametrize(
    "nodes",
    [(counted_rows, counted_height), (counted_words, counted_row, counted_height)],
    ids=["a series of rows", "rows collected from a loop"],
)
def test_rows_whose_source_declares_the_tables_row_type_are_checked_once(nodes) -> None:
    graph = Graph.compile(make_graph(*nodes), registry_of(*nodes))
    CHECKED.clear()
    run = graph.run({"k": 3})

    assert (run.status, run.outputs["result"], CHECKED) == ("complete", 3, [0, 1, 2])


class Amounts(TypedDict):
    name: str
    amount: int


@node("amount-rows", title="Amount rows", description="For a test.")
def amount_rows(k: int) -> Series[Amounts]:
    return [{"name": str(i), "amount": i - 1} for i in range(k)]


@node("positive-height", title="Positive height", description="For a test.")
def positive_height(table: Table[Positive]) -> int:
    return table.height


@node("two-or-more", title="Two or more", description="For a test.")
def two_or_more(table: Annotated[Table[Amounts], Param(min_length=2)]) -> int:
    return table.height


@node("tagged-rows", title="Tagged rows", description="For a test.")
def tagged_rows(k: int) -> Series[Tagged]:
    return [{"name": str(i), "tags": [float("nan")]} for i in range(k)]


@node("tagged-height", title="Tagged height", description="For a test.")
def tagged_height(table: Table[Tagged]) -> int:
    return table.height


@pytest.mark.parametrize(
    ("nodes", "k", "error"),
    [
        ((amount_rows, positive_height), 3, "greater_than_equal"),
        ((amount_rows, two_or_more), 1, "too_short"),
        ((tagged_rows, tagged_height), 1, "column_not_saved"),
    ],
    ids=[
        "a limit only the input's row has",
        "the table's own limit",
        "a cell a saved run can't keep",
    ],
)
def test_rows_collected_into_a_table_are_still_refused_for_what_their_source_does_not_declare(
    nodes, k, error
) -> None:
    graph = Graph.compile(make_graph(*nodes), registry_of(*nodes))
    run = graph.run({"k": k})

    cause = run.cause
    assert (run.status, cause.type, [one["type"] for one in cause.ctx["errors"]]) == (
        "failed",
        "invalid_edge_value",
        [error],
    )


@node("counted-table", title="Counted table", description="For a test.")
def counted_table(k: int) -> Table[Counted]:
    return pl.DataFrame({"name": [str(i) for i in range(k)], "amount": list(range(k))})


def test_a_table_whose_source_declares_the_same_table_is_checked_once() -> None:
    graph = Graph.compile(
        make_graph(counted_table, counted_height), registry_of(counted_table, counted_height)
    )
    CHECKED.clear()
    run = graph.run({"k": 3})

    assert (run.status, run.outputs["result"], CHECKED) == ("complete", 3, [0, 1, 2])


class MaybeAmount(TypedDict):
    amount: int | None


class Amount(TypedDict):
    amount: int


@node("maybe-amounts", title="Maybe amounts", description="For a test.")
def maybe_amounts(k: int) -> Series[MaybeAmount]:
    return [{"amount": None} for _ in range(k)]


@node("maybe-amount", title="Maybe amount", description="For a test.")
def maybe_amount(word: str) -> MaybeAmount:
    return {"amount": None}


@node("amount-height", title="Amount height", description="For a test.")
def amount_height(table: Table[Amount]) -> int:
    return table.height


@pytest.mark.parametrize(
    "nodes",
    [(maybe_amounts, amount_height), (counted_words, maybe_amount, amount_height)],
    ids=["a series of rows", "rows collected from a loop"],
)
def test_rows_whose_field_may_be_none_are_not_collected_into_a_table_whose_field_may_not(
    nodes,
) -> None:
    graph = Graph.compile(make_graph(*nodes), registry_of(*nodes))

    assert [(problem.type, problem.loc) for problem in graph.problems] == [
        ("type_mismatch", ("amount-height", "table"))
    ]


CONTEXTS: list[Any] = []


@host_type("noted-cents", title="Noted cents")
class NotedCents(BaseModel):
    model_config = ConfigDict(frozen=True)
    cents: int

    @field_validator("cents")
    @classmethod
    def _note(cls, cents: int, info: ValidationInfo) -> int:
        CONTEXTS.append(info.context)
        return cents


class NotedPrice(TypedDict):
    price: NotedCents


@node("noted-frame", title="Noted frame", description="For a test.")
def noted_frame(k: int) -> Series[NotedPrice]:
    return pl.DataFrame({"price": [{"cents": i} for i in range(k)]})


@node("noted-height", title="Noted height", description="For a test.")
def noted_height(table: Table[NotedPrice]) -> int:
    return table.height


def test_a_hosts_validator_in_a_frame_of_rows_collected_into_a_table_is_called_with_no_context() -> (
    None
):
    graph = Graph.compile(
        make_graph(noted_frame, noted_height), registry_of(noted_frame, noted_height)
    )
    CONTEXTS.clear()
    run = graph.run({"k": 2})

    assert (run.status, run.outputs["result"]) == ("complete", 2)
    assert CONTEXTS and all(context is None for context in CONTEXTS)


@node("number-keys", title="Number keys", description="For a test.")
def number_keys(k: int) -> Series[dict[int, str]]:
    return [{i: "x"} for i in range(k)]


@node("bare-height", title="Bare height", description="For a test.")
def bare_height(table: Table) -> int:
    return table.height


def test_rows_collected_into_a_bare_table_are_read_as_rows_whose_keys_are_text() -> None:
    graph = Graph.compile(
        make_graph(number_keys, bare_height), registry_of(number_keys, bare_height)
    )
    run = graph.run({"k": 1})

    cause = run.cause
    assert (run.status, cause.type, [one["type"] for one in cause.ctx["errors"]]) == (
        "failed",
        "invalid_edge_value",
        ["string_type"],
    )


TABLE_CONTEXTS: list[Any] = []


def _look(table: pl.DataFrame, info: ValidationInfo) -> pl.DataFrame:
    TABLE_CONTEXTS.append(info.context)
    return table


class Count(TypedDict):
    n: int


class Measure(TypedDict):
    n: float


@pytest.mark.parametrize("returned", ["list", "frame"])
def test_a_hosts_validator_on_a_table_of_collected_rows_is_called_with_no_context(
    returned: str,
) -> None:
    @node("count-rows", title="Count rows", description="For a test.")
    def count_rows(k: int) -> Series[Count]:
        return pl.DataFrame({"n": [1]}) if returned == "frame" else [{"n": 1}]  # type: ignore[return-value]

    @node("looked-height", title="Looked height", description="For a test.")
    def looked_height(table: Annotated[Table[Count], AfterValidator(_look)]) -> int:
        return table.height

    graph = Graph.compile(
        make_graph(count_rows, looked_height), registry_of(count_rows, looked_height)
    )
    TABLE_CONTEXTS.clear()
    run = graph.run({"k": 1})

    assert (run.status, TABLE_CONTEXTS) == ("complete", [None])


@pytest.mark.parametrize("returned", ["list", "frame"])
def test_whole_numbers_collected_into_a_float_column_are_floats(returned: str) -> None:
    @node("count-rows", title="Count rows", description="For a test.")
    def count_rows(k: int) -> Series[Count]:
        return pl.DataFrame({"n": [1]}) if returned == "frame" else [{"n": 1}]  # type: ignore[return-value]

    @node("measure-types", title="Measure types", description="For a test.")
    def measure_types(table: Table[Measure]) -> str:
        return str(table.schema["n"])

    graph = Graph.compile(
        make_graph(count_rows, measure_types), registry_of(count_rows, measure_types)
    )

    assert graph.run({"k": 1}).outputs == {"result": "Float64"}


NAMED_CHECKS: list[Any] = []


def _noted(value: Any) -> Any:
    NAMED_CHECKS.append(value)
    return value


@host_type("shade", title="Shade")
class Shade(Enum):
    DARK = "dark"


class Named(TypedDict):
    size: Annotated[Literal["s", "m"], AfterValidator(_noted)]
    colour: Annotated[Shade, AfterValidator(_noted)]


def test_a_frame_of_named_values_collected_into_a_table_is_not_read_again() -> None:
    """A frame a Series of rows returned, with a ``Literal`` and an enum column, has the polars types its row type's
    own frame has, so the table input wired to it keeps it and checks no cell again."""

    @node("named-rows", title="Named rows", description="For a test.")
    def named_rows(k: int) -> Series[Named]:
        return pl.DataFrame({"size": ["s"] * k, "colour": ["dark"] * k})  # type: ignore[return-value]

    @node("named-height", title="Named height", description="For a test.")
    def named_height(table: Table[Named]) -> int:
        return table.height

    graph = Graph.compile(
        make_graph(named_rows, named_height), registry_of(named_rows, named_height)
    )
    NAMED_CHECKS.clear()
    run = graph.run({"k": 2})

    assert (run.status, run.outputs["result"], len(NAMED_CHECKS)) == ("complete", 2, 4)
