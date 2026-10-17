"""A table is one value of named, typed columns, holding a Polars frame and the ids of its rows."""

import datetime
import subprocess
import sys

import polars as pl
import pytest
from conductor import Table
from conductor.codec import from_wire, to_wire
from conductor.dtype import DType
from conductor.errors import Refuses
from conductor.series import Index
from conductor.table import ROW_ID, column_type, row_id, with_row_ids


class Txt(DType, str):
    id = "txt"
    title = "Text"


class Num(DType, float):
    id = "num"
    title = "Number"


class Flag(DType, int):
    """Built on int, since bool can't be subclassed: it says its column itself."""

    id = "flag"
    title = "Flag"
    column = "Boolean"


class Day(DType, datetime.date):
    id = "day"
    title = "Date"


class Blob(DType):
    """Built on no builtin a column holds, and stating no column."""

    id = "blob"
    title = "Blob"


Contacts = Table[("name", Txt), ("score", Num)]


def test_a_column_type_is_read_off_what_the_type_is_built_on():
    assert (column_type(Txt), column_type(Num), column_type(Day)) == (pl.String(), pl.Float64(), pl.Date())


def test_a_type_whose_base_would_be_wrong_says_its_column_itself():
    assert column_type(Flag) == pl.Boolean()


def test_a_type_with_no_column_cant_be_a_table_column_and_says_which():
    with pytest.raises(TypeError, match="Blob can't be a table column"):
        Table[("file", Blob)]


def test_a_table_type_is_built_once_per_set_of_columns():
    assert Table[("name", Txt), ("score", Num)] is Contacts
    assert Contacts.id == "table" and Contacts.parameterises is Table
    assert repr(Contacts) == "Table[name: Txt, score: Num]"


def test_declared_columns_are_read_as_their_types_and_extra_columns_dropped():
    table = Contacts([{"name": "Ida", "score": "4.5", "extra": 1}, {"name": "Bo", "score": None, "extra": 2}])

    assert table.frame.columns == ["name", "score"]
    assert table.frame["score"].to_list() == [4.5, None]


def test_a_value_that_wont_read_as_its_column_is_refused_naming_the_column():
    with pytest.raises(ValueError, match="column 'score' has a value that is not a Number"):
        Contacts([{"name": "Ida", "score": "many"}])
    with pytest.raises(ValueError, match="no column 'score'"):
        Contacts([{"name": "Ida"}])


def test_a_date_column_reads_dates_written_as_text():
    Visits = Table[("day", Day)]
    assert Visits([{"day": "2024-01-02"}]).frame["day"].to_list() == [datetime.date(2024, 1, 2)]
    with pytest.raises(ValueError, match="not a Date"):
        Visits([{"day": "someday"}])


def test_the_wire_form_round_trips_with_blanks_and_row_ids():
    table = Contacts(pl.DataFrame({ROW_ID: ["contacts:0", "contacts:1"], "name": ["Ida", None], "score": [4.5, None]}))
    wire = to_wire(table, Contacts)

    assert wire == {
        "columns": [{"name": "name", "dtype": "String"}, {"name": "score", "dtype": "Float64"}],
        "data": {"name": ["Ida", None], "score": [4.5, None]},
        "rows": ["contacts:0", "contacts:1"],
    }
    assert from_wire(wire, Contacts) == table


def test_a_bare_table_keeps_every_column_and_round_trips_a_date():
    table = Table(pl.DataFrame({"day": [datetime.date(2024, 1, 2)], "n": [3]}))
    wire = to_wire(table, Table)

    assert wire["rows"] is None
    assert from_wire(wire, Table) == table


def test_a_row_id_names_the_index_and_the_row():
    assert row_id(Index("contacts"), (0,)) == "contacts:0"
    assert row_id(Index("structured.invoices", parent=Index("docs.result")), (0, 1)) == "structured.invoices:0.1"


def test_a_table_that_kept_its_row_ids_keeps_them():
    """A node that filters or sorts hands its rows back with their ids."""
    contacts = with_row_ids(pl.DataFrame({"name": ["Ida", "Bo", "Nina"]}), Index("contacts"), ())
    kept = with_row_ids(contacts.filter(pl.col("name") != "Bo").sort("name"), Index("narrowed"), ())

    assert kept[ROW_ID].to_list() == ["contacts:0", "contacts:2"]


def test_a_table_without_ids_or_with_one_twice_gets_new_rows():
    contacts = with_row_ids(pl.DataFrame({"name": ["Ida", "Bo"]}), Index("contacts"), ())
    chosen = with_row_ids(contacts.select("name"), Index("chosen"), (3,))
    doubled = with_row_ids(pl.concat([contacts, contacts]), Index("doubled"), ())

    assert chosen[ROW_ID].to_list() == ["chosen:3.0", "chosen:3.1"]
    assert doubled[ROW_ID].to_list() == ["doubled:0", "doubled:1", "doubled:2", "doubled:3"]


def test_a_table_input_takes_a_table_with_at_least_its_columns():
    Names = Table[("name", Txt)]

    assert Names.accepts(Contacts) and Table.accepts(Contacts) and Names.accepts(Table)
    assert not Contacts.accepts(Names)
    assert not Names.accepts(Txt)


def test_a_bare_table_is_refused_where_a_node_reads_it_whole():
    with pytest.raises(Refuses) as refused:
        Table.refuses_whole()
    assert refused.value.code == "table_columns_unknown"
    assert Contacts.refuses_whole() is None


def test_a_table_type_describes_its_columns():
    assert Contacts.describe() == {
        "id": "table",
        "columns": [{"name": "name", "type": {"id": "txt"}}, {"name": "score", "type": {"id": "num"}}],
    }


def test_importing_conductor_does_not_import_polars():
    loaded = subprocess.run(
        [sys.executable, "-c", "import sys, conductor; print('polars' in sys.modules)"], capture_output=True, text=True, check=True,
    )
    assert loaded.stdout.strip() == "False"


def test_a_node_returns_a_frame_and_its_output_holds_the_table_type():
    from typing import Annotated

    from conductor import CompiledGraph, GraphNode, NodeRegistry, Param, Static, run_sync
    from conductor.graph.model import Graph
    from conductor.metadata import Result
    from conductor.node import NodeDefinition
    from conductor.widgets import Textarea

    class Read(NodeDefinition):
        id = "read"
        title = "Read"
        description = "d"
        category = "test"

        def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Contacts, Result(title="Contacts")]:
            return pl.DataFrame({"name": text.split(","), "score": [1.0] * len(text.split(","))})

    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="r", type="read", version=1, bindings={"text": Static("Ida,Bo")})]), NodeRegistry(nodes=(Read,)))
    contacts = run_sync(compiled).state.results(compiled)["r"]["result"]

    assert type(contacts) is Contacts
    assert contacts.frame["name"].to_list() == ["Ida", "Bo"]


def test_a_bare_table_with_nested_and_time_columns_round_trips():
    """Nested cells are a later plan's to open; a table holding them still stores and restores as it was."""
    table = Table(pl.DataFrame({
        "tags": [["a", "b"], []], "who": [{"name": "Ida", "age": 3}, {"name": "Bo", "age": 4}],
        "at": [datetime.time(9, 30), None],
    }))

    assert from_wire(to_wire(table, Table), Table) == table
