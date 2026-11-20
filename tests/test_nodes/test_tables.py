"""Tests of the nodes that turn tables into lists and back again.

The nodes are ``series-split``, ``series-stack``, ``series-collect``, ``table-collect``, ``text-list`` and
``text-split``. The tests check what each one returns, the types it declares, and what it refuses.
"""

import json
from typing import Annotated, Any, TypedDict

import conductor_nodes
import polars as pl
import pytest
from conductor import (
    Graph,
    InvalidCallError,
    NodeRegistry,
    Param,
    Run,
    Series,
    Table,
    make_graph,
    node,
)
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


@_node("members")
def members() -> Table[Contact]:
    return pl.DataFrame({"name": ["ida", "bo"], "email": ["ida@x.dk", None], "age": [41, 52]})


@_node("greet")
def greet(name: str) -> str:
    return f"hi {name}"


REGISTRY = NodeRegistry()
REGISTRY.add_subregistry(conductor_nodes.registry)
for _definition in (
    contacts,
    csv,
    tags,
    words,
    counts,
    nobody,
    sentence,
    send,
    shout,
    one_row,
    header,
    members,
    greet,
):
    REGISTRY.add(_definition)


#: ``table-collect`` works out its output when the graph is compiled, so ``make_graph`` is told it's the graph's.
COLLECTED = {"result": "table-collect.result"}


def compile_graph(
    *nodes: Any,
    edges: list[tuple[str, str]],
    constants: dict[str, Any] | None = None,
    outputs: dict[str, str] | None = None,
):
    return Graph.compile(
        make_graph(*nodes, edges=edges, constants=constants or {}, outputs=outputs or {}), REGISTRY
    )


def output(graph, node_id: str):
    (described,) = graph.describe().nodes[node_id].outputs
    return described


def columns_of(field) -> dict[str, Any]:
    """Returns the columns of a table field, read from the field's JSON schema.

    Each column's name maps to the JSON schema of its cells.
    """
    return {
        c["properties"]["name"]["const"]: c["properties"]["values"]["items"]
        for c in field.json_schema["properties"]["columns"]["prefixItems"]
    }


# -- split ---------------------------------------------------------------------------------------------


def test_split_gives_a_tables_rows_typed_as_its_row_type_declares_them():
    graph = compile_graph(contacts, series.split, edges=[("contacts.result", "series-split.value")])
    described = output(graph, "series-split")
    assert graph.problems == [] and (described.is_series, described.type) == (
        True,
        "{name: str, email: str | None}",
    )
    assert described.json_schema["properties"] == {
        "name": {"title": "Name", "type": "string"},
        "email": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Email"},
    }
    assert graph.run().results["series-split"]["result"] == [
        {"name": "Ann", "email": "ann@x.dk"},
        {"name": "Bo", "email": None},
        {"name": "Cy", "email": "cy@x.dk"},
    ]


def test_a_node_reading_a_field_of_a_split_row_is_a_none_input_where_the_field_is_none_and_collect_keeps_the_row():
    SENT.clear()
    graph = compile_graph(
        contacts,
        series.split,
        send,
        tables.collect,
        outputs=COLLECTED,
        edges=[
            ("contacts.result", "series-split.value"),
            ("series-split.result.email", "send.email"),
            ("series-split.result", "table-collect.rows"),
            ("send.result", "table-collect.sent"),
        ],
        constants={"table-collect.columns": ["sent"]},
    )
    run = graph.run()
    assert run.status == "complete" and SENT == ["ann@x.dk", "cy@x.dk"]
    assert [(n.loc[0], n.loc[1:], n.input) for n in run.none_input_calls] == [
        ("send", (1,), "email")
    ]
    assert run.results["table-collect"]["result"].to_dicts() == [
        {"name": "Ann", "email": "ann@x.dk", "sent": "sent to ann@x.dk"},
        {"name": "Bo", "email": None, "sent": None},
        {"name": "Cy", "email": "cy@x.dk", "sent": "sent to cy@x.dk"},
    ]


def test_split_gives_rows_of_unknown_columns_for_a_table_of_any_and_a_field_of_one_is_read_when_it_arrives():
    graph = compile_graph(
        csv,
        series.split,
        shout,
        edges=[("csv.result", "series-split.value"), ("series-split.result.name", "shout.text")],
    )
    assert graph.problems == [] and output(graph, "series-split").type == "Any"
    assert graph.run().results["shout"]["result"] == ["ANN", "BO"]


def test_split_gives_a_lists_items_of_the_lists_own_type():
    graph = compile_graph(
        tags,
        series.split,
        shout,
        edges=[("tags.result", "series-split.value"), ("series-split.result", "shout.text")],
    )
    assert graph.problems == [] and (
        output(graph, "series-split").is_series,
        output(graph, "series-split").type,
    ) == (True, "str")
    assert graph.describe().nodes[
        "shout"
    ].per_item_of == "series-split.result" and graph.run().results["shout"]["result"] == [
        "RED",
        "BLUE",
    ]


def test_a_list_constant_on_split_gives_items_of_any_type():
    graph = compile_graph(series.split, edges=[], constants={"series-split.value": ["a", 1]})
    assert output(graph, "series-split").type == "Any" and graph.run().results["series-split"][
        "result"
    ] == ["a", 1]


def test_what_is_neither_a_table_nor_a_list_is_a_type_mismatch_on_the_edge_into_split():
    (problem,) = compile_graph(
        sentence, series.split, edges=[("sentence.result", "series-split.value")]
    ).problems
    assert (problem.type, problem.loc, problem.ctx["got"], problem.ctx["expected"]) == (
        "type_mismatch",
        ("series-split", "value"),
        "str",
        "Table | list",
    )


# -- split hands on the table it was given, unchanged ----------------------------------------------------


def _split_members():
    return compile_graph(
        members,
        series.split,
        greet,
        edges=[
            ("members.result", "series-split.value"),
            ("series-split.result.name", "greet.name"),
        ],
    )


def test_split_returns_the_table_it_was_given_and_the_run_keeps_it_as_a_table():
    run = _split_members().run()

    assert run.outputs == {"result": ["hi ida", "hi bo"]}
    saved = json.loads(run.to_json())["outcomes"]["series-split"]["outputs"]["result"]
    assert saved == [
        {
            "columns": [
                {"name": "name", "type": "str", "values": ["ida", "bo"]},
                {"name": "email", "type": "str", "values": ["ida@x.dk", None]},
            ]
        }
    ]  # saved as the table it was given, with only the columns its row type declares
    assert run.results["series-split"]["result"] == [
        {"name": "ida", "email": "ida@x.dk"},
        {"name": "bo", "email": None},
    ]


def test_a_table_split_into_rows_is_written_under_both_outputs_and_reads_back_equal():
    graph = _split_members()
    run = graph.run()
    saved = json.loads(run.to_json())

    assert (
        saved["outcomes"]["series-split"]["outputs"]["result"]
        == saved["outcomes"]["members"]["outputs"]["result"]
    )
    read_back = Run.from_json(run.to_json(), graph)
    table, again = run.results["members"]["result"], read_back.results["members"]["result"]
    assert again.equals(table) and again.schema == table.schema
    assert read_back.results["series-split"]["result"] == run.results["series-split"]["result"]
    assert read_back.to_json() == run.to_json() and read_back.outputs == run.outputs
    assert read_back.with_forgotten("greet").resume().outputs == {"result": ["hi ida", "hi bo"]}


def test_a_saved_run_with_null_for_a_done_call_whose_output_takes_no_none_is_refused():
    graph = _split_members()
    saved = json.loads(graph.run().to_json())
    saved["outcomes"]["series-split"]["outputs"]["result"] = [None]  # null, not listed as skipped

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)
    (found,) = refused.value.problems
    assert (found.type, found.ctx["where"]) == ("saved_run_unreadable", "outcomes.series-split")
    assert "call 0 is done and has null for 'result'" in found.msg


def test_stack_puts_the_tables_returned_per_item_into_one_table_of_their_row_type():
    graph = compile_graph(
        contacts,
        series.split,
        one_row,
        series.stack,
        edges=[
            ("contacts.result", "series-split.value"),
            ("series-split.result", "one-row.contact"),
            ("one-row.result", "series-stack.tables"),
        ],
    )
    assert (
        graph.problems == []
        and not output(graph, "series-stack").is_series
        and output(graph, "series-stack").type.startswith("Table")
        and set(columns_of(output(graph, "series-stack"))) == {"name", "email"}
    )
    stacked = graph.run().results["series-stack"]["result"]
    assert stacked.to_dicts() == graph.run().results["contacts"]["result"].to_dicts()


def test_collect_gives_the_values_returned_per_item_as_one_list():
    graph = compile_graph(
        series.split,
        text.uppercase,
        series.collect,
        edges=[
            ("series-split.result", "text-uppercase.text"),
            ("text-uppercase.result", "series-collect.values"),
        ],
        constants={"series-split.value": ["a", "b"]},
    )
    assert graph.problems == [] and graph.run().results["series-collect"]["result"] == ["A", "B"]


def test_collect_takes_the_type_of_the_values_it_collects():
    graph = compile_graph(
        counts, series.collect, edges=[("counts.result", "series-collect.values")]
    )
    assert graph.problems == [] and output(graph, "series-collect").type == "int"


# -- table-collect ---------------------------------------------------------------------------------------


def test_collect_types_its_table_by_the_rows_and_what_arrives_on_each_new_column():
    graph = compile_graph(
        contacts,
        series.split,
        send,
        sentence,
        tables.collect,
        outputs=COLLECTED,
        edges=[
            ("contacts.result", "series-split.value"),
            ("series-split.result.email", "send.email"),
            ("series-split.result", "table-collect.rows"),
            ("send.result", "table-collect.sent"),
            ("sentence.result", "table-collect.team"),
        ],
        constants={"table-collect.columns": ["sent"], "table-collect.constant_columns": ["team"]},
    )
    assert graph.problems == []
    columns = columns_of(output(graph, "table-collect"))
    assert list(columns) == ["name", "email", "sent", "team"] and columns["name"] == {
        "type": "string"
    }
    assert columns["sent"] == {"anyOf": [{"type": "string"}, {"type": "null"}]} and columns[
        "team"
    ] == {"type": "string"}
    feeds = {i.field.name: i.source.mode for i in graph.describe().nodes["table-collect"].inputs}
    assert (feeds["rows"], feeds["sent"], feeds["team"]) == (
        "collected",
        "collected",
        "one_value",
    )
    assert graph.run().results["table-collect"]["result"]["team"].to_list() == ["one two"] * 3


def test_collect_takes_the_rows_as_a_table_and_keeps_its_columns_as_they_are():
    @_node("small-numbers")
    def small_numbers() -> Table:
        return pl.DataFrame({"n": [1, 2], "name": ["a", "b"]})

    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    registry.add(small_numbers)
    graph = Graph.compile(
        make_graph(
            small_numbers,
            series.split,
            shout,
            tables.collect,
            outputs=COLLECTED,
            edges=[
                ("small-numbers.result", "series-split.value"),
                ("series-split.result.name", "shout.text"),
                ("series-split.result", "table-collect.rows"),
                ("shout.result", "table-collect.loud"),
            ],
            constants={"table-collect.columns": ["loud"]},
        ),
        registry,
    )
    made = graph.run().results["table-collect"]["result"]

    assert [
        i.field.type
        for i in graph.describe().nodes["table-collect"].inputs
        if i.field.name == "rows"
    ] == ["Table"]
    assert made.schema == {"n": pl.Int64, "name": pl.String, "loud": pl.String} and made[
        "loud"
    ].to_list() == ["A", "B"]


def test_collect_given_a_whole_table_types_its_result_by_that_tables_columns():
    graph = compile_graph(
        contacts,
        sentence,
        tables.collect,
        outputs=COLLECTED,
        edges=[
            ("contacts.result", "table-collect.rows"),
            ("sentence.result", "table-collect.team"),
        ],
        constants={"table-collect.constant_columns": ["team"]},
    )

    assert graph.problems == [] and list(columns_of(output(graph, "table-collect"))) == [
        "name",
        "email",
        "team",
    ]
    assert graph.run().results["table-collect"]["result"]["team"].to_list() == ["one two"] * 3


@pytest.mark.parametrize("split", [False, True], ids=["whole", "split"])
def test_collect_types_its_table_by_a_generic_rows_filled_in_columns_with_their_limits(split):
    """A generic row, ``Line[int]``, gives the result its columns with ``T`` filled in, and a column keeps its
    ``Param`` limits, whether the rows arrive as the whole ``Table[Line[int]]`` or as the ``Series[Line[int]]``
    that ``series-split`` gives. The hook reads the row as conductor reads every row, so the result's columns are
    the columns the rows have."""

    class Line[T](TypedDict):
        value: T
        count: Annotated[int, Param(ge=0)]

    @_node("lines")
    def lines() -> Table[Line[int]]:
        return pl.DataFrame({"value": [7], "count": [1]})

    registry = NodeRegistry()
    registry.add_subregistry(REGISTRY)
    registry.add(lines)
    into_rows = (
        [("lines.result", "series-split.value"), ("series-split.result", "table-collect.rows")]
        if split
        else [("lines.result", "table-collect.rows")]
    )
    graph = Graph.compile(
        make_graph(
            lines,
            *([series.split] if split else []),
            sentence,
            tables.collect,
            outputs=COLLECTED,
            edges=[*into_rows, ("sentence.result", "table-collect.team")],
            constants={"table-collect.constant_columns": ["team"]},
        ),
        registry,
    )

    assert graph.problems == []
    columns = columns_of(output(graph, "table-collect"))
    assert columns == {
        "value": {"type": "integer"},
        "count": {"minimum": 0, "type": "integer"},
        "team": {"type": "string"},
    }
    assert graph.run().results["table-collect"]["result"].to_dicts() == [
        {"value": 7, "count": 1, "team": "one two"}
    ]


def test_collect_refuses_a_new_column_the_rows_already_have_naming_it():
    graph = compile_graph(
        contacts,
        series.split,
        tables.collect,
        outputs=COLLECTED,
        edges=[
            ("contacts.result", "series-split.value"),
            ("series-split.result", "table-collect.rows"),
            ("series-split.result.name", "table-collect.email"),
        ],
        constants={"table-collect.columns": ["email"]},
    )
    (problem,) = graph.problems
    assert (problem.type, problem.loc, problem.ctx["error"], problem.ctx["error_ctx"]) == (
        "compute_hook_failed",
        ("table-collect",),
        "column_taken",
        {"column": "email"},
    )


@pytest.mark.parametrize(
    ("nodes", "edges", "constants"),
    [
        (
            (csv, sentence),
            [("csv.result", "table-collect.rows"), ("sentence.result", "table-collect.age")],
            {"table-collect.constant_columns": ["age"]},
        ),
        (
            (csv, series.split),
            [
                ("csv.result", "series-split.value"),
                ("series-split.result", "table-collect.rows"),
                ("series-split.result.name", "table-collect.age"),
            ],
            {"table-collect.columns": ["age"]},
        ),
    ],
)
def test_collect_refuses_a_new_column_that_a_table_of_any_columns_already_has_when_it_runs(
    nodes: tuple[Any, ...],
    edges: list[tuple[str, str]],
    constants: dict[str, Any],
):
    """A new column that reuses a name of a bare ``Table``'s columns fails when the node runs.

    A bare ``Table`` declares no columns, so its column names are only known once the node runs. The graph compiles
    without a problem, and the run fails.
    """
    graph = compile_graph(
        *nodes, tables.collect, outputs=COLLECTED, edges=edges, constants=constants
    )
    assert graph.problems == []
    cause = graph.run().cause
    assert (cause.type, cause.loc, cause.ctx) == (
        "column_taken",
        ("table-collect",),
        {"column": "age"},
    )


def test_collect_refuses_a_column_named_twice():
    graph = compile_graph(
        tables.collect,
        outputs=COLLECTED,
        edges=[],
        constants={"table-collect.columns": ["team"], "table-collect.constant_columns": ["team"]},
    )
    (problem,) = graph.problems
    assert (problem.type, problem.ctx["error"], problem.ctx["error_ctx"]) == (
        "compute_hook_failed",
        "column_named_twice",
        {"column": "team"},
    )


@pytest.mark.parametrize(
    ("people", "expected"),
    [
        (3, {"column": "word", "rows": 3, "values": 2}),
        (1, {"column": "word", "rows": 1, "values": 2}),
    ],
)
def test_collect_refuses_a_column_wired_from_another_series_than_its_rows_naming_it(
    people: int, expected: dict[str, Any]
):
    """A column wired from a different Series than the rows fails, whether it is longer or shorter.

    A column needs exactly one value per row. That only holds when the column comes from a node called once for
    each of those rows.
    """

    @_node("people")
    def people_table() -> Table[Contact]:
        return pl.DataFrame(
            {"name": [f"p{i}" for i in range(people)], "email": [None] * people},
            schema={"name": pl.String, "email": pl.String},
        )

    held = NodeRegistry()
    held.add_subregistry(REGISTRY)
    held.add(people_table)
    graph = Graph.compile(
        make_graph(
            people_table,
            series.split,
            words,
            tables.collect,
            outputs=COLLECTED,
            edges=[
                ("people.result", "series-split.value"),
                ("series-split.result", "table-collect.rows"),
                ("words.result", "table-collect.word"),
            ],
            constants={"table-collect.columns": ["word"]},
        ),
        held,
    )
    assert graph.problems == []
    cause = graph.run().cause
    assert (cause.type, cause.loc, cause.ctx) == (
        "columns_not_aligned",
        ("table-collect",),
        expected,
    )
    assert (
        "'word' has 2 values" in cause.msg
        and f"{people} row" in cause.msg
        and "once for each of the same rows" in cause.msg
    )


def test_collect_refuses_a_column_wired_from_a_value_returned_once_and_takes_it_as_a_constant_column():
    @_node("two-people")
    def two_people() -> Table[Contact]:
        return pl.DataFrame(
            {"name": ["Ida", "Bo"], "email": [None, None]},
            schema={"name": pl.String, "email": pl.String},
        )

    held = NodeRegistry()
    held.add_subregistry(REGISTRY)
    held.add(two_people)

    def team_column(kind: str) -> Graph[Any]:
        return Graph.compile(
            make_graph(
                two_people,
                series.split,
                sentence,
                tables.collect,
                outputs=COLLECTED,
                edges=[
                    ("two-people.result", "series-split.value"),
                    ("series-split.result", "table-collect.rows"),
                    ("sentence.result", "table-collect.team"),
                ],
                constants={f"table-collect.{kind}": ["team"]},
            ),
            held,
        )

    (refused,) = team_column("columns").problems
    assert (refused.type, refused.ctx["source"], refused.loc) == (
        "type_mismatch_series_given_one_value",
        "sentence.result",
        ("table-collect", "team"),
    )
    assert refused.msg == (
        "'sentence.result' gives each call of 'table-collect' one str, but 'table-collect.team' takes a Series "
        "of Any. To give it every item of a loop, collect them first in a node outside the loop that takes a "
        "Series. Else make it an input that takes one Any."
    )
    assert team_column("constant_columns").run().results["table-collect"]["result"].to_dicts() == [
        {"name": "Ida", "email": None, "team": "one two"},
        {"name": "Bo", "email": None, "team": "one two"},
    ]


def test_collect_of_no_rows_is_a_table_with_every_column_and_no_lines():
    graph = compile_graph(
        nobody,
        sentence,
        tables.collect,
        header,
        edges=[
            ("nobody.result", "table-collect.rows"),
            ("sentence.result", "table-collect.team"),
            ("table-collect.result", "header.table"),
        ],
        constants={"table-collect.constant_columns": ["team"]},
    )
    run = graph.run()
    assert run.status == "complete", run.cause
    assert run.results["table-collect"]["result"].height == 0 and run.results["header"][
        "result"
    ] == ["name", "email", "team"]


def test_collect_of_rows_of_unknown_columns_is_a_table_of_any_columns():
    graph = compile_graph(
        csv,
        series.split,
        shout,
        tables.collect,
        outputs=COLLECTED,
        edges=[
            ("csv.result", "series-split.value"),
            ("series-split.result.name", "shout.text"),
            ("series-split.result", "table-collect.rows"),
            ("shout.result", "table-collect.loud"),
        ],
        constants={"table-collect.columns": ["loud"]},
    )
    collected = output(graph, "table-collect")
    assert (
        graph.problems == []
        and collected.type == "Table"
        and "prefixItems" not in collected.json_schema["properties"]["columns"]
    )
    assert graph.run().results["table-collect"]["result"].to_dicts() == [
        {"name": "Ann", "age": 31, "loud": "ANN"},
        {"name": "Bo", "age": 47, "loud": "BO"},
    ]


def test_collect_takes_rows_only():
    (problem,) = compile_graph(
        words, tables.collect, outputs=COLLECTED, edges=[("words.result", "table-collect.rows")]
    ).problems
    assert (problem.type, problem.loc, problem.ctx["got"], problem.ctx["expected"]) == (
        "type_mismatch",
        ("table-collect", "rows"),
        "Series[str]",
        "Table",
    )


# -- text-list and text-split -------------------------------------------------------------------------------


def test_text_list_gives_its_texts_once_and_once_per_item_of_what_is_wired_to_for_each():
    alone = compile_graph(
        text.text_list,
        shout,
        edges=[("text-list.result", "shout.text")],
        constants={"text-list.items": ["a", "b"]},
    )
    assert alone.describe().nodes["text-list"].per_item_of is None and alone.run().results["shout"][
        "result"
    ] == ["A", "B"]
    inside = compile_graph(
        words,
        text.text_list,
        edges=[("words.result", "text-list.for_each")],
        constants={"text-list.items": ["x", "y", "z"]},
    )
    assert inside.describe().nodes["text-list"].per_item_of == "words.result"
    assert inside.run().results["text-list"]["result"] == [["x", "y", "z"], ["x", "y", "z"]]


def test_text_split_is_titled_split_text_splits_on_a_newline_and_an_empty_text_has_no_parts():
    assert text.split.title == "Split text" and text.split.inputs["separator"].default == "\n"
    assert (
        text.split("a\nb") == ["a", "b"]
        and text.split("a,b", separator=",") == ["a", "b"]
        and text.split("") == []
    )
    assert (
        series.split.title == "Split"
        and tables.collect.title == "Collect into a table"
        and text.text_list.title == "Text list"
    )
