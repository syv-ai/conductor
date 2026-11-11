"""A ``Series`` of records returned as a table is held as that table: an edge that names a field reads the
column, a node still receives plain records, and the table is read by its columns where it enters."""

import json
from typing import Any, TypedDict

import polars as pl
import pytest
from conductor import (
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Series,
    Table,
    make_graph,
    node,
    stored_graph,
)
from conductor._internal.table import Records

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
    return registry_of(source, greet, one, every, names, mails).load(GraphRecord(
        nodes=[NodeRecord(id="c", node=source.id), *(NodeRecord(id=n, node=n) for n in ("greet", "one", "every", "names", "mails"))],
        edges=[("c.result.name", "greet.name"), ("c.result", "one.contact"), ("c.result", "every.found"), ("c.result.name", "names.found"), ("c.result.email", "mails.found")],
    ))


@pytest.mark.parametrize("source", [contacts, listed], ids=["returned as a table", "returned as a list"])
def test_a_node_receives_plain_records_whichever_way_the_series_was_returned(source: Any) -> None:
    RECEIVED.clear()
    run = _graph(source).run(text="x")

    assert run.outputs == {"greet.result": ["hi ida", "hi bo"], "one.result": ["ida", "bo"], "every.result": 2, "names.result": "ida+bo", "mails.result": 2}
    records = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert RECEIVED == [*records, records, ["ida", "bo"], ["ida@x.dk", None]] and all(type(record) is dict for record in RECEIVED[:2])
    assert run.results["c"]["result"] == records  # a host reads plain records too


def test_a_series_returned_as_a_table_is_held_and_saved_as_that_table_with_only_its_declared_columns() -> None:
    graph = _graph()
    run = graph.run(text="x")
    held = run._units["c"][()].out["result"]
    assert type(held) is Records and held.frame.columns == ["name", "email"] and len(held) == 2

    saved = json.loads(run.to_json())
    assert saved["units"]["c"]["outputs"]["result"] == [{"columns": [
        {"name": "name", "type": "str", "values": ["ida", "bo"]}, {"name": "email", "type": "str", "values": ["ida@x.dk", None]},
    ]}]
    assert json.loads(_graph(listed).run(text="x").to_json())["units"]["c"]["outputs"]["result"] == [[{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]]

    loaded = graph.load_run(run.to_json())
    assert type(loaded._units["c"][()].out["result"]) is Records
    assert loaded.to_json() == run.to_json() and loaded.results == run.results
    RECEIVED.clear()
    assert loaded.forget("one").resume().outputs == run.outputs and RECEIVED == [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]


@pytest.mark.parametrize(("frame", "reason"), [
    (pl.DataFrame({"name": ["ida", None], "email": ["a", "b"]}), "column 'name': a cell is blank, and the column's type takes no blank"),
    (pl.DataFrame({"email": ["a", "b"]}), "the column 'name' its row type declares is missing"),
    (pl.DataFrame({"name": [1, 2], "email": ["a", "b"]}), "column 'name': Input should be a valid string"),
])
def test_a_table_returned_for_a_series_of_records_is_read_by_its_columns_where_it_enters(frame: pl.DataFrame, reason: str) -> None:
    @node("wrong", title="Wrong", description="For a test.")
    def wrong(text: str) -> Series[Contact]:
        return frame  # type: ignore[return-value]

    run = registry_of(wrong).load(make_graph(wrong)).run(text="x")

    assert (run.status, run.state.cause.type, run.state.cause.ctx["reason"], run.state.cause.ctx["column"]) == ("failed", "invalid_output", reason, "name")


def test_a_table_is_no_series_of_anything_but_records() -> None:
    @node("texts", title="Texts", description="For a test.")
    def texts(text: str) -> Series[str]:
        return FRAME  # type: ignore[return-value]

    run = registry_of(texts).load(make_graph(texts)).run(text="x")

    assert (run.status, run.state.cause.type) == ("failed", "invalid_output")


def test_a_table_of_any_columns_is_a_series_of_its_records() -> None:
    @node("anything", title="Anything", description="For a test.")
    def anything(text: str) -> Series[Any]:
        return FRAME  # type: ignore[return-value]

    @node("show", title="Show", description="For a test.")
    def show(record: Any) -> str:
        return repr(record)

    graph = registry_of(anything, show).load(make_graph(anything, show, edges=[("anything.result", "show.record")]))
    run = graph.run(text="x")

    assert run.outputs["show.result"] == ["{'name': 'ida', 'email': 'ida@x.dk', 'age': 41}", "{'name': 'bo', 'email': None, 'age': 52}"]
    assert graph.load_run(run.to_json()).results == run.results


def test_a_series_of_records_held_as_a_table_counts_its_items_against_the_ceiling() -> None:
    run = _graph().run(text="x", max_items=1)

    assert (run.status, run.state.cause.type, run.state.cause.ctx["items"]) == ("stopped", "item_ceiling", 2)


def test_a_series_of_records_returned_per_item_inside_a_stored_graph_reaches_a_series_input_outside_as_plain_records() -> None:
    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Series[str]:
        return text.split(",")

    @node("count", title="Count", description="For a test.")
    def count(found: Series[Contact]) -> int:
        RECEIVED.append(found)
        return len(found)

    registry = registry_of(parts, contacts, count)
    inner = GraphRecord(nodes=[NodeRecord(id="p", node="parts"), NodeRecord(id="c", node="contacts")], edges=[("p.result", "c.text")])
    registry.add(stored_graph("inner", title="Inner", description="For a test.")(inner))
    graph = registry.load(GraphRecord(nodes=[NodeRecord(id="e", node="inner"), NodeRecord(id="count", node="count")], edges=[("e.c.result", "count.found")]))
    RECEIVED.clear()
    run = graph.run(text="x,y")

    records = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert (run.status, run.outputs["count.result"]) == ("complete", [2, 2]) and RECEIVED == [records, records]  # one call per part
    assert graph.load_run(run.to_json()).outputs == run.outputs


def test_split_returns_the_table_it_was_given_and_the_run_holds_it_by_column() -> None:
    from conductor_nodes.series import split

    @node("people", title="People", description="For a test.")
    def people(text: str) -> Table[Contact]:
        return FRAME

    graph = registry_of(people, split, greet).load(make_graph(people, split, greet, edges=[("people.result", "series-split.value"), ("series-split.result.name", "greet.name")]))
    run = graph.run(text="x")

    assert run.outputs == {"greet.result": ["hi ida", "hi bo"]}
    assert type(run._units["series-split"][()].out["result"]) is Records
    assert run.results["series-split"]["result"] == [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]


# -- how a run stores a Series, and what reads it ---------------------------------------------------------


def test_one_record_of_five_without_the_field_an_edge_reads_fails_only_the_call_for_that_record() -> None:
    @node("five", title="Five", description="For a test.")
    def five(text: str) -> Series[dict]:
        return [{"name": "a"}, {"name": "b"}, {"other": "c"}, {"name": "d"}, {"name": "e"}]

    graph = registry_of(five, greet).load(make_graph(five, greet, edges=[("five.result.name", "greet.name")]))
    run = graph.run(text="x")

    cause = run.state.cause
    assert (run.status, cause.type, cause.loc, cause.ctx["record_field"]) == ("failed", "invalid_edge_value", ("greet", 2), "name")
    assert run.results["greet"]["result"][:3] == ["hi a", "hi b", None]  # the two records before it were read and called


@pytest.mark.parametrize("source", [contacts, listed], ids=["returned as a table", "returned as a list"])
def test_the_column_of_a_field_is_made_once_per_stored_series_and_only_for_a_field_an_edge_reads(source: Any) -> None:
    graph = registry_of(source, greet).load(make_graph(source, greet, edges=[(f"{source.id}.result.name", "greet.name")]))
    run = graph.run(text="x")

    stored = run._units[source.id][()].out["result"]
    assert dict(stored.columns) == {"name": ["ida", "bo"]}  # "email" was never read, so no list was made of it
    assert stored.columns["name"] is stored.columns["name"]
    assert (len(stored), stored[1], stored.as_list()) == (2, {"name": "bo", "email": None}, [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}])


def test_a_table_input_wired_to_a_series_stored_as_a_table_is_called_with_that_table() -> None:
    got: list[Any] = []

    @node("rows", title="Rows", description="For a test.")
    def rows(table: Table[Contact]) -> int:
        got.append(table)
        return table.height

    graph = registry_of(contacts, rows).load(make_graph(contacts, rows, edges=[("contacts.result", "rows.table")]))
    run = graph.run(text="x")

    stored = run._units["contacts"][()].out["result"]
    assert run.outputs == {"rows.result": 2} and got[0] is stored.frame and dict(stored.columns) == {}


def test_a_table_input_loses_the_rows_another_input_of_the_same_series_has_no_value_for() -> None:
    got: list[Any] = []

    @node("mail", title="Mail", description="For a test.")
    def mail(email: str) -> str:
        return email.upper()

    @node("both", title="Both", description="For a test.")
    def both(table: Table[Contact], mails: Series[str]) -> int:
        got.append((table, mails))
        return table.height

    graph = registry_of(contacts, mail, both).load(make_graph(contacts, mail, both, edges=[
        ("contacts.result.email", "mail.email"), ("contacts.result", "both.table"), ("mail.result", "both.mails"),
    ]))
    run = graph.run(text="x")

    assert run.outputs["both.result"] == 1  # bo has no email, so mail was not called for him
    assert got[0][0].to_dicts() == [{"name": "ida", "email": "ida@x.dk"}] and got[0][1] == ["IDA@X.DK"]


def test_a_table_a_node_received_is_written_once_to_fingerprint_it_however_many_times_it_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from conductor._internal.earlier_work import fingerprint_of
    from conductor._internal.table import write

    written: list[int] = []
    as_written = write.written_table
    monkeypatch.setattr(write, "written_table", lambda frame: written.append(1) or as_written(frame))
    frame = pl.DataFrame({"name": ["ida", "bo"], "age": [41, 52]})

    first = fingerprint_of("code", {"table": frame, "n": 1})
    assert fingerprint_of("code", {"table": frame, "n": 1}) == first and len(written) == 1
    assert fingerprint_of("code", {"table": pl.DataFrame({"name": ["ida", "bo"], "age": [41, 52]}), "n": 1}) == first  # by its cells
    assert fingerprint_of("code", {"table": pl.DataFrame({"name": ["ida", "bo"], "age": [41, 53]}), "n": 1}) != first
    assert fingerprint_of("code", {"table": frame.rename({"age": "years"}), "n": 1}) != first


# -- a table two outputs share is written once in a save ---------------------------------------------------


@node("people", title="People", description="For a test.")
def people(text: str) -> Table[Contact]:
    return FRAME.select("name", "email")


@node("copied", title="Copied", description="For a test.")
def copied(table: Table[Contact]) -> Table[Contact]:
    return pl.DataFrame(table.to_dicts(), schema=table.schema)


def _read_then_split() -> Any:
    from conductor_nodes.series import split

    return registry_of(people, split, greet, copied).load(make_graph(people, split, greet, edges=[("people.result", "series-split.value"), ("series-split.result.name", "greet.name")]))


def test_a_table_read_and_then_split_is_written_once_in_a_save_and_is_one_table_again_when_loaded() -> None:
    graph = _read_then_split()
    run = graph.run(text="x")
    saved = json.loads(run.to_json())

    assert saved["units"]["series-split"]["outputs"]["result"] == [None]
    assert saved["tables_written_once"] == [{"output": ["series-split", "result", 0], "same_as": ["people", "result", 0]}]
    assert run.to_json().count('"ida@x.dk"') == 1

    loaded = graph.load_run(run.to_json())
    table, records = loaded._units["people"][()].out["result"], loaded._units["series-split"][()].out["result"]
    assert type(records) is Records and records.frame is table
    assert loaded.to_json() == run.to_json() and loaded.outputs == run.outputs
    assert loaded.forget("greet").resume().outputs == {"greet.result": ["hi ida", "hi bo"]}


def test_a_table_with_equal_cells_that_is_another_object_is_written_in_full_and_loads() -> None:
    graph = registry_of(people, copied).load(make_graph(people, copied, edges=[("people.result", "copied.table")]))
    run = graph.run(text="x")
    saved = json.loads(run.to_json())

    assert "tables_written_once" not in saved and saved["units"]["copied"]["outputs"]["result"] == saved["units"]["people"]["outputs"]["result"]
    assert graph.load_run(run.to_json()).results["copied"]["result"].equals(run.results["copied"]["result"])


def test_a_save_of_a_graph_with_no_table_has_no_tables_written_once_key() -> None:
    graph = registry_of(listed, greet).load(make_graph(listed, greet, edges=[("listed.result.name", "greet.name")]))
    assert "tables_written_once" not in json.loads(graph.run(text="x").to_json())


def test_work_carried_from_an_earlier_run_shares_its_table_in_a_save_as_the_first_run_did() -> None:
    graph = _read_then_split()
    again = graph.run(text="x").again(text="x")
    saved = json.loads(again.to_json())

    assert saved["tables_written_once"] == [{"output": ["series-split", "result", 0], "same_as": ["people", "result", 0]}]
    assert graph.load_run(again.to_json()).to_json() == again.to_json()


def test_a_table_shared_inside_an_embedded_graph_is_written_once_under_the_nodes_names_in_the_save() -> None:
    from conductor_nodes.series import split

    registry = registry_of(people, split, greet)
    inner = GraphRecord(nodes=[NodeRecord(id="people", node="people"), NodeRecord(id="split", node=split.id)], edges=[("people.result", "split.value")])
    registry.add(stored_graph("inner", title="Inner", description="For a test.")(inner))
    graph = registry.load(GraphRecord(nodes=[NodeRecord(id="e", node="inner"), NodeRecord(id="greet", node="greet")], edges=[("e.split.result.name", "greet.name")]))
    run = graph.run(text="x")

    assert json.loads(run.to_json())["tables_written_once"] == [{"output": ["e/split", "result", 0], "same_as": ["e/people", "result", 0]}]
    loaded = graph.load_run(run.to_json())
    assert loaded.to_json() == run.to_json() and loaded.outputs == {"greet.result": ["hi ida", "hi bo"]}


@pytest.mark.parametrize("same_as", [
    ["nobody", "result", 0],  # no such node
    ["people", "result", 3],  # no such slot
    ["people", "other", 0],  # no such output
    ["series-split", "result", 0],  # itself, which is written as null
    ["greet", "result", 0],  # an output that is no table
], ids=["no such node", "no such slot", "no such output", "itself", "not a table"])
def test_a_save_whose_tables_written_once_names_an_output_that_is_no_table_written_in_full_is_refused(same_as: list[Any]) -> None:
    graph = _read_then_split()
    saved = json.loads(graph.run(text="x").to_json())
    saved["tables_written_once"][0]["same_as"] = same_as

    with pytest.raises(InvalidCallError) as refused:
        graph.load_run(json.dumps(saved))
    (found,) = refused.value.errors()
    assert found.type == "state_unreadable" and "tables_written_once: 'series-split.result[0]' can't be read as the table at" in found.msg and "not an output written in full" in found.msg


def test_a_save_whose_tables_written_once_names_a_table_of_other_columns_is_refused_by_the_outputs_own_type() -> None:
    @node("any-table", title="Any table", description="For a test.")
    def any_table(text: str) -> Table:
        return FRAME.select("name", "email")

    @node("as-contacts", title="As contacts", description="For a test.")
    def as_contacts(table: Table) -> Series[Contact]:
        return table  # type: ignore[return-value]

    graph = registry_of(any_table, as_contacts).load(make_graph(any_table, as_contacts, edges=[("any-table.result", "as-contacts.table")]))
    saved = json.loads(graph.run(text="x").to_json())
    assert saved["tables_written_once"] == [{"output": ["as-contacts", "result", 0], "same_as": ["any-table", "result", 0]}]
    saved["units"]["any-table"]["outputs"]["result"][0]["columns"][0]["name"] = "nom"  # a table of any columns takes it

    with pytest.raises(InvalidCallError) as refused:
        graph.load_run(json.dumps(saved))
    (found,) = refused.value.errors()
    assert found.type == "state_unreadable" and "the column 'name' its row type declares is missing" in found.msg


def test_the_json_a_host_gets_for_a_table_goes_into_polars_and_into_pandas_in_one_line_each() -> None:
    import pandas as pd

    graph = registry_of(people).load(make_graph(people))
    table = json.loads(graph.run(text="x").outputs_json())["people.result"]

    in_polars = pl.DataFrame({c["name"]: c["values"] for c in table["columns"]})
    in_pandas = pd.DataFrame({c["name"]: c["values"] for c in table["columns"]})

    assert in_polars.to_dicts() == [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert list(in_pandas.columns) == ["name", "email"] and in_pandas["name"].tolist() == ["ida", "bo"]
    assert in_pandas["email"][0] == "ida@x.dk" and pd.isna(in_pandas["email"][1])  # pandas shows a blank cell as its own missing value
