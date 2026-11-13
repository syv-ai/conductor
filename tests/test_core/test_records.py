"""A ``Series`` of records returned as a table is kept as that table: an edge that names a field reads its column,
a node gets plain records, and the table is checked column by column when the node returns it."""

import json
import subprocess
import sys
from typing import Any, NotRequired, TypedDict

import polars as pl
import pytest
from conductor import (
    Address,
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    Run,
    Series,
    Table,
    make_graph,
    node,
    subgraph,
)
from conductor._internal.series import Records
from pytest import MonkeyPatch

from tests.test_core.nodes import registry_of


class Contact(TypedDict):
    name: str
    email: str | None


FRAME = pl.DataFrame({"name": ["ida", "bo"], "email": ["ida@x.dk", None], "age": [41, 52]})
RECEIVED: list[Any] = []


@node("contacts", title="Contacts", description="For a test.")
def contacts(text: str) -> Series[Contact]:
    return FRAME  # type: ignore[return-value]  a table returned for a Series of records is the Series of its records


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
    reads = {"greet": ("name", "name"), "one": ("contact", None), "every": ("found", None), "names": ("found", "name"), "mails": ("found", "email")}
    readers = {
        node_id: NodeRecord(node=node_id, inputs={name: InputEntry(sources=[Address("c", "result", field)])}) for node_id, (name, field) in reads.items()
    }
    return Graph.compile(GraphRecord(
        nodes={"c": NodeRecord(node=source.id, inputs={"text": InputEntry(graph_input="text")}), **readers},
        outputs={node_id: Address(node_id, "result") for node_id in reads},
    ), registry_of(source, greet, one, every, names, mails))


@pytest.mark.parametrize("source", [contacts, listed], ids=["returned as a table", "returned as a list"])
def test_a_node_receives_plain_records_whichever_way_the_series_was_returned(source: Any) -> None:
    RECEIVED.clear()
    run = _graph(source).run(text="x")

    assert run.outputs == {"greet": ["hi ida", "hi bo"], "one": ["ida", "bo"], "every": 2, "names": "ida+bo", "mails": 2}
    records = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert RECEIVED == [records, ["ida@x.dk", None], ["ida", "bo"], *records] and all(type(record) is dict for record in RECEIVED[3:])  # nodes called in id order
    assert run.results["c"]["result"] == records  # a host reads plain records too


def test_a_series_returned_as_a_table_is_kept_and_saved_as_that_table_with_only_its_declared_columns() -> None:
    graph = _graph()
    run = graph.run(text="x")
    kept = run._outcomes[("c",)][()].outputs["result"]
    assert type(kept) is Records and kept.frame.columns == ["name", "email"] and len(kept) == 2

    saved = json.loads(run.to_json())
    assert saved["outcomes"]["c"]["outputs"]["result"] == [{"columns": [
        {"name": "name", "type": "str", "values": ["ida", "bo"]}, {"name": "email", "type": "str", "values": ["ida@x.dk", None]},
    ]}]
    assert json.loads(_graph(listed).run(text="x").to_json())["outcomes"]["c"]["outputs"]["result"] == [[{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]]

    read_back = Run.from_json(run.to_json(), graph)
    assert type(read_back._outcomes[("c",)][()].outputs["result"]) is Records
    assert read_back.to_json() == run.to_json() and read_back.results == run.results
    RECEIVED.clear()
    assert read_back.forget("one").resume().outputs == run.outputs and RECEIVED == [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]


@pytest.mark.parametrize(("frame", "reason"), [
    (pl.DataFrame({"name": ["ida", None], "email": ["a", "b"]}), "column 'name': a cell is blank, and the column's type takes no blank"),
    (pl.DataFrame({"email": ["a", "b"]}), "the column 'name' its row type declares is missing"),
    (pl.DataFrame({"name": [1, 2], "email": ["a", "b"]}), "column 'name': Input should be a valid string"),
])
def test_a_table_returned_for_a_series_of_records_is_checked_column_by_column_and_fails_the_run(frame: pl.DataFrame, reason: str) -> None:
    @node("wrong", title="Wrong", description="For a test.")
    def wrong(text: str) -> Series[Contact]:
        return frame  # type: ignore[return-value]

    run = Graph.compile(make_graph(wrong), registry_of(wrong)).run(text="x")

    assert (run.status, run.cause.type, run.cause.ctx["reason"], run.cause.ctx["column"]) == ("failed", "invalid_output", reason, "name")


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

    graph = Graph.compile(make_graph(anything, show, edges=[("anything.result", "show.record")]), registry_of(anything, show))
    run = graph.run(text="x")

    assert run.outputs["result"] == ["{'name': 'ida', 'email': 'ida@x.dk', 'age': 41}", "{'name': 'bo', 'email': None, 'age': 52}"]
    assert Run.from_json(run.to_json(), graph).results == run.results


def test_a_series_of_records_kept_as_a_table_counts_its_rows_against_the_item_limit() -> None:
    run = _graph().run(text="x", max_items=1)

    assert (run.status, run.cause.type, run.cause.ctx["items"]) == ("stopped", "item_limit", 2)


def test_a_series_of_records_returned_per_item_inside_a_subgraph_reaches_a_series_input_outside_as_plain_records() -> None:
    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Series[str]:
        return text.split(",")

    @node("count", title="Count", description="For a test.")
    def count(found: Series[Contact]) -> int:
        RECEIVED.append(found)
        return len(found)

    registry = registry_of(parts, contacts, count)
    inner = GraphRecord.model_validate({"nodes": {"p": {"node": "parts", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "contacts", "inputs": {"text": {"sources": [{"node": "p", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}})
    registry.add(subgraph("inner", title="Inner", description="For a test.")(inner))
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "inner", "inputs": {"text": {"graph_input": "text"}}}, "count": {"node": "count", "inputs": {"found": {"sources": [{"node": "e", "output": "result"}]}}}}, "outputs": {"result": {"node": "count", "output": "result"}}}), registry)
    RECEIVED.clear()
    run = graph.run(text="x,y")

    records = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert (run.status, run.outputs["result"]) == ("complete", [2, 2]) and RECEIVED == [records, records]  # one call per part
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


def test_split_returns_the_table_it_was_given_and_the_run_keeps_it_as_a_table() -> None:
    from conductor_nodes.series import split

    @node("people", title="People", description="For a test.")
    def people(text: str) -> Table[Contact]:
        return FRAME

    graph = Graph.compile(make_graph(people, split, greet, edges=[("people.result", "series-split.value"), ("series-split.result.name", "greet.name")]), registry_of(people, split, greet))
    run = graph.run(text="x")

    assert run.outputs == {"result": ["hi ida", "hi bo"]}
    assert type(run._outcomes[("series-split",)][()].outputs["result"]) is Records
    assert run.results["series-split"]["result"] == [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]


# -- how a run stores a Series, and what reads it ---------------------------------------------------------


def test_one_record_of_five_without_the_field_an_edge_reads_fails_only_the_call_for_that_record() -> None:
    @node("five", title="Five", description="For a test.")
    def five(text: str) -> Series[dict]:
        return [{"name": "a"}, {"name": "b"}, {"other": "c"}, {"name": "d"}, {"name": "e"}]

    graph = Graph.compile(make_graph(five, greet, edges=[("five.result.name", "greet.name")]), registry_of(five, greet))
    run = graph.run(text="x")

    cause = run.cause
    assert (run.status, cause.type, cause.loc, cause.ctx["record_field"]) == ("failed", "invalid_edge_value", ("greet", 2), "name")
    assert run.results["greet"]["result"][:3] == ["hi a", "hi b", None]  # the two records before it were read and called


@pytest.mark.parametrize("source", [contacts, listed], ids=["returned as a table", "returned as a list"])
def test_the_column_of_a_field_is_made_once_per_stored_series_and_only_for_a_field_an_edge_reads(source: Any) -> None:
    graph = Graph.compile(make_graph(source, greet, edges=[(f"{source.id}.result.name", "greet.name")]), registry_of(source, greet))
    run = graph.run(text="x")

    stored = run._outcomes[(source.id,)][()].outputs["result"]
    assert dict(stored.columns) == {"name": ["ida", "bo"]}  # "email" was never read, so no list was made of it
    assert stored.columns["name"] is stored.columns["name"]
    assert (len(stored), stored[1], stored.as_list()) == (2, {"name": "bo", "email": None}, [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}])


def test_a_table_input_wired_to_a_series_stored_as_a_table_is_called_with_that_table() -> None:
    got: list[Any] = []

    @node("rows", title="Rows", description="For a test.")
    def rows(table: Table[Contact]) -> int:
        got.append(table)
        return table.height

    graph = Graph.compile(make_graph(contacts, rows, edges=[("contacts.result", "rows.table")]), registry_of(contacts, rows))
    run = graph.run(text="x")

    stored = run._outcomes[("contacts",)][()].outputs["result"]
    assert run.outputs == {"result": 2} and got[0] is stored.frame and dict(stored.columns) == {}


def test_a_table_input_loses_the_rows_another_input_of_the_same_series_has_no_value_for() -> None:
    got: list[Any] = []

    @node("mail", title="Mail", description="For a test.")
    def mail(email: str) -> str:
        return email.upper()

    @node("both", title="Both", description="For a test.")
    def both(table: Table[Contact], mails: Series[str]) -> int:
        got.append((table, mails))
        return table.height

    graph = Graph.compile(make_graph(contacts, mail, both, edges=[
        ("contacts.result.email", "mail.email"), ("contacts.result", "both.table"), ("mail.result", "both.mails"),
    ]), registry_of(contacts, mail, both))
    run = graph.run(text="x")

    assert run.outputs["result"] == 1  # bo has no email, so mail was not called for him
    assert got[0][0].to_dicts() == [{"name": "ida", "email": "ida@x.dk"}] and got[0][1] == ["IDA@X.DK"]


def test_a_table_is_written_once_however_often_it_is_hashed_and_hashes_by_its_cells(monkeypatch: MonkeyPatch) -> None:
    from conductor._internal.base import json_hash
    from conductor._internal.table import write
    from conductor._internal.types import untyped_to_hash

    written: list[int] = []
    as_written = write.written_table
    monkeypatch.setattr(write, "written_table", lambda frame: written.append(1) or as_written(frame))
    frame = pl.DataFrame({"name": ["ida", "bo"], "age": [41, 52]})

    def hashed(table: pl.DataFrame) -> str:
        return json_hash(["code", {"table": table, "n": 1}], 12, fallback=untyped_to_hash)

    first = hashed(frame)
    assert hashed(frame) == first and len(written) == 1
    assert hashed(pl.DataFrame({"name": ["ida", "bo"], "age": [41, 52]})) == first  # by its cells
    assert hashed(pl.DataFrame({"name": ["ida", "bo"], "age": [41, 53]})) != first
    assert hashed(frame.rename({"age": "years"})) != first


# -- a table two outputs share is written once in a saved run------------------------------------------------


@node("people", title="People", description="For a test.")
def people(text: str) -> Table[Contact]:
    return FRAME.select("name", "email")


@node("copied", title="Copied", description="For a test.")
def copied(table: Table[Contact]) -> Table[Contact]:
    return pl.DataFrame(table.to_dicts(), schema=table.schema)


def _read_then_split() -> Any:
    from conductor_nodes.series import split

    return Graph.compile(make_graph(people, split, greet, edges=[("people.result", "series-split.value"), ("series-split.result.name", "greet.name")]), registry_of(people, split, greet, copied))


def test_a_table_split_into_records_is_written_once_in_the_saved_run_and_is_one_table_when_read_back() -> None:
    graph = _read_then_split()
    run = graph.run(text="x")
    saved = json.loads(run.to_json())

    assert saved["outcomes"]["series-split"]["outputs"]["result"] == [None]
    assert saved["tables_written_once"] == [{"output": ["series-split", "result", 0], "same_as": ["people", "result", 0]}]
    assert run.to_json().count('"ida@x.dk"') == 1

    read_back = Run.from_json(run.to_json(), graph)
    table, records = read_back._outcomes[("people",)][()].outputs["result"], read_back._outcomes[("series-split",)][()].outputs["result"]
    assert type(records) is Records and records.frame is table
    assert read_back.to_json() == run.to_json() and read_back.outputs == run.outputs
    assert read_back.forget("greet").resume().outputs == {"result": ["hi ida", "hi bo"]}


def test_a_table_with_equal_cells_that_is_another_object_is_written_in_full_and_reads_back() -> None:
    graph = Graph.compile(make_graph(people, copied, edges=[("people.result", "copied.table")]), registry_of(people, copied))
    run = graph.run(text="x")
    saved = json.loads(run.to_json())

    assert "tables_written_once" not in saved and saved["outcomes"]["copied"]["outputs"]["result"] == saved["outcomes"]["people"]["outputs"]["result"]
    assert Run.from_json(run.to_json(), graph).results["copied"]["result"].equals(run.results["copied"]["result"])


def test_the_saved_run_of_a_graph_with_no_table_has_no_tables_written_once_key() -> None:
    graph = Graph.compile(make_graph(listed, greet, edges=[("listed.result.name", "greet.name")]), registry_of(listed, greet))
    assert "tables_written_once" not in json.loads(graph.run(text="x").to_json())


def test_outcomes_reused_from_a_previous_run_share_their_table_in_the_saved_run_as_the_first_run_did() -> None:
    graph = _read_then_split()
    again = graph.run(text="x").again(text="x")
    saved = json.loads(again.to_json())

    assert saved["tables_written_once"] == [{"output": ["series-split", "result", 0], "same_as": ["people", "result", 0]}]
    assert Run.from_json(again.to_json(), graph).to_json() == again.to_json()


def test_a_table_shared_inside_a_subgraph_is_written_once_under_the_inner_node_paths_in_the_saved_run() -> None:
    from conductor_nodes.series import split

    registry = registry_of(people, split, greet)
    inner = GraphRecord.model_validate({"nodes": {"people": {"node": "people", "inputs": {"text": {"graph_input": "text"}}}, "split": {"node": split.id, "inputs": {"value": {"sources": [{"node": "people", "output": "result"}]}}}}, "outputs": {"result": {"node": "split", "output": "result"}}})
    registry.add(subgraph("inner", title="Inner", description="For a test.")(inner))
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "inner", "inputs": {"text": {"graph_input": "text"}}}, "greet": {"node": "greet", "inputs": {"name": {"sources": [{"node": "e", "output": "result", "field": "name"}]}}}}, "outputs": {"result": {"node": "greet", "output": "result"}}}), registry)
    run = graph.run(text="x")

    assert json.loads(run.to_json())["tables_written_once"] == [{"output": ["e/split", "result", 0], "same_as": ["e/people", "result", 0]}]
    read_back = Run.from_json(run.to_json(), graph)
    assert read_back.to_json() == run.to_json() and read_back.outputs == {"result": ["hi ida", "hi bo"]}


@pytest.mark.parametrize("same_as", [
    ["nobody", "result", 0],  # no such node
    ["people", "result", 3],  # no such call number
    ["people", "other", 0],  # no such output
    ["series-split", "result", 0],  # itself, which is written as null
    ["greet", "result", 0],  # an output that is no table
], ids=["no such node", "no such call number", "no such output", "itself", "not a table"])
def test_a_saved_run_whose_tables_written_once_names_an_output_not_written_in_full_as_a_table_is_refused(same_as: list[Any]) -> None:
    graph = _read_then_split()
    saved = json.loads(graph.run(text="x").to_json())
    saved["tables_written_once"][0]["same_as"] = same_as

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)
    (found,) = refused.value.problems
    assert found.type == "saved_run_unreadable" and found.ctx["where"] == "tables_written_once" and "'series-split.result[0]' can't be read as the table at" in found.msg and "not an output written in full" in found.msg


def test_a_saved_run_whose_tables_written_once_names_a_table_of_other_columns_is_refused_by_the_outputs_type() -> None:
    @node("any-table", title="Any table", description="For a test.")
    def any_table(text: str) -> Table:
        return FRAME.select("name", "email")

    @node("as-contacts", title="As contacts", description="For a test.")
    def as_contacts(table: Table) -> Series[Contact]:
        return table  # type: ignore[return-value]

    graph = Graph.compile(make_graph(any_table, as_contacts, edges=[("any-table.result", "as-contacts.table")]), registry_of(any_table, as_contacts))
    saved = json.loads(graph.run(text="x").to_json())
    assert saved["tables_written_once"] == [{"output": ["as-contacts", "result", 0], "same_as": ["any-table", "result", 0]}]
    saved["outcomes"]["any-table"]["outputs"]["result"][0]["columns"][0]["name"] = "nom"  # a table of any columns takes it

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)
    (found,) = refused.value.problems
    assert found.type == "saved_run_unreadable" and "the column 'name' its row type declares is missing" in found.msg


def test_the_json_a_host_gets_for_a_table_goes_into_polars_and_into_pandas_in_one_line_each() -> None:
    import pandas as pd

    graph = Graph.compile(make_graph(people), registry_of(people))
    table = json.loads(graph.run(text="x").outputs_to_json())["result"]

    in_polars = pl.DataFrame({c["name"]: c["values"] for c in table["columns"]})
    in_pandas = pd.DataFrame({c["name"]: c["values"] for c in table["columns"]})

    assert in_polars.to_dicts() == [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert list(in_pandas.columns) == ["name", "email"] and in_pandas["name"].tolist() == ["ida", "bo"]
    assert in_pandas["email"][0] == "ida@x.dk" and pd.isna(in_pandas["email"][1])  # pandas shows a null cell as its own missing value


class _MaybeNamed(TypedDict):
    name: NotRequired[str]
    email: str


class _Partial(TypedDict, total=False):
    name: str


@pytest.mark.parametrize("row", [_MaybeNamed, _Partial])
@pytest.mark.parametrize("hint", ["input table", "output table", "input series", "output series"])
def test_a_row_type_that_may_leave_a_field_out_is_refused_when_the_node_is_declared(row: Any, hint: str) -> None:
    kind = Table if "table" in hint else Series

    if hint.startswith("input"):
        def declared(rows: kind[row]) -> int:  # type: ignore[valid-type]
            return 0
    else:
        def declared() -> kind[row]:  # type: ignore[valid-type]
            return []

    with pytest.raises(NodeDeclarationError) as refused:
        node("declared", title="Declared", description="For a test.")(declared)
    assert refused.value.code == "row_field_optional"
    assert "name" in refused.value.problem.msg and "| None" in refused.value.problem.msg


_SERIES_PACKAGE_ALONE = """
import importlib.util, sys, types
from typing import TypedDict

import polars as pl
from pydantic_core import SchemaValidator

# conductor's __init__ imports every package; an empty module in its place loads the series package with only the
# packages it imports itself.
root = types.ModuleType("conductor")
root.__path__ = importlib.util.find_spec("conductor").submodule_search_locations
sys.modules["conductor"] = root

from conductor._internal.series import Records, Series, without_indexes
from conductor._internal.types import Type

class Row(TypedDict):
    n: int

assert "conductor._internal.run" not in sys.modules and "conductor._internal.graph" not in sys.modules
frame = pl.DataFrame({"n": [1, 2, 3]})
assert type(SchemaValidator(Type(Series[Row]).returned_schema).validate_python(frame)) is Records
assert SchemaValidator(Type(Series[int]).schema).validate_python(pl.Series([1, 2])) == [1, 2]
assert without_indexes(frame, {1}).to_dicts() == [{"n": 1}, {"n": 3}]
"""


def test_the_series_package_stores_a_frame_and_reads_a_column_whatever_else_is_imported() -> None:
    done = subprocess.run([sys.executable, "-c", _SERIES_PACKAGE_ALONE], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
