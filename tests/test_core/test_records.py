"""A ``Series`` of records returned as a table is held as that table: an edge that names a field reads the
column, a node still receives plain records, and the table is read by its columns where it enters."""

import json
from typing import Any, TypedDict

import polars as pl
import pytest
from conductor import GraphRecord, NodeRecord, Series, Table, make_graph, node
from conductor._internal.fields import Records

from tests.test_core.nodes import registry_of


class Contact(TypedDict):
    name: str
    email: str | None


FRAME = pl.DataFrame({"name": ["ida", "bo"], "email": ["ida@x.dk", None], "age": [41, 52]})
RECEIVED: list[Any] = []


@node("contacts", title="Contacts", description="")
def contacts(text: str) -> Series[Contact]:
    return FRAME  # type: ignore[return-value]  a table returned for a Series of records is the Series of its records


@node("listed", title="Listed", description="")
def listed(text: str) -> Series[Contact]:
    return [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]


@node("greet", title="Greet", description="")
def greet(name: str) -> str:
    return f"hi {name}"


@node("one", title="One", description="")
def one(contact: Contact) -> str:
    RECEIVED.append(contact)
    return contact["name"]


@node("every", title="Every", description="")
def every(found: Series[Contact]) -> int:
    RECEIVED.append(found)
    return len(found)


@node("names", title="Names", description="")
def names(found: Series[str]) -> str:
    RECEIVED.append(found)
    return "+".join(found)


@node("mails", title="Mails", description="")
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
    @node("wrong", title="Wrong", description="")
    def wrong(text: str) -> Series[Contact]:
        return frame  # type: ignore[return-value]

    run = registry_of(wrong).load(make_graph(wrong)).run(text="x")

    assert (run.status, run.state.cause.type, run.state.cause.ctx["reason"], run.state.cause.ctx["column"]) == ("failed", "invalid_output", reason, "name")


def test_a_table_is_no_series_of_anything_but_records() -> None:
    @node("texts", title="Texts", description="")
    def texts(text: str) -> Series[str]:
        return FRAME  # type: ignore[return-value]

    run = registry_of(texts).load(make_graph(texts)).run(text="x")

    assert (run.status, run.state.cause.type) == ("failed", "invalid_output")


def test_a_table_of_any_columns_is_a_series_of_its_records() -> None:
    @node("anything", title="Anything", description="")
    def anything(text: str) -> Series[Any]:
        return FRAME  # type: ignore[return-value]

    @node("show", title="Show", description="")
    def show(record: Any) -> str:
        return repr(record)

    graph = registry_of(anything, show).load(make_graph(anything, show, edges=[("anything.result", "show.record")]))
    run = graph.run(text="x")

    assert run.outputs["show.result"] == ["{'name': 'ida', 'email': 'ida@x.dk', 'age': 41}", "{'name': 'bo', 'email': None, 'age': 52}"]
    assert graph.load_run(run.to_json()).results == run.results


def test_a_series_of_records_held_as_a_table_counts_its_items_against_the_ceiling() -> None:
    run = _graph().run(text="x", max_items=1)

    assert (run.status, run.state.cause.type, run.state.cause.ctx["items"]) == ("stopped", "item_ceiling", 2)


def test_a_series_of_records_made_per_item_inside_an_embedded_graph_is_offered_as_lists_of_plain_records() -> None:
    @node("parts", title="Parts", description="")
    def parts(text: str) -> Series[str]:
        return text.split(",")

    @node("count", title="Count", description="")
    def count(found: Series[list[Contact]]) -> list[int]:
        RECEIVED.append(found)
        return [len(records) for records in found]

    registry = registry_of(parts, contacts, count)
    inner = GraphRecord(nodes=[NodeRecord(id="p", node="parts"), NodeRecord(id="c", node="contacts")], edges=[("p.result", "c.text")])
    registry.add(node("inner", title="Inner", description="")(inner))
    graph = registry.load(GraphRecord(nodes=[NodeRecord(id="e", node="inner"), NodeRecord(id="count", node="count")], edges=[("e.c.result", "count.found")]))
    RECEIVED.clear()
    run = graph.run(text="x,y")

    records = [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
    assert (run.status, run.outputs["count.result"]) == ("complete", [2, 2]) and RECEIVED == [[records, records]]
    assert graph.load_run(run.to_json()).outputs == run.outputs


def test_split_returns_the_table_it_was_given_and_the_run_holds_it_by_column() -> None:
    from conductor_nodes.tables import split

    @node("people", title="People", description="")
    def people(text: str) -> Table[Contact]:
        return FRAME

    graph = registry_of(people, split, greet).load(make_graph(people, split, greet, edges=[("people.result", "split.value"), ("split.result.name", "greet.name")]))
    run = graph.run(text="x")

    assert run.outputs == {"greet.result": ["hi ida", "hi bo"]}
    assert type(run._units["split"][()].out["result"]) is Records
    assert run.results["split"]["result"] == [{"name": "ida", "email": "ida@x.dk"}, {"name": "bo", "email": None}]
