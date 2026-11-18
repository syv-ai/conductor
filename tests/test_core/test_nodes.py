"""Tests for ``@node``: what a node author declares, and what a host sees of it.

A node is a function under ``@node``. Its signature says the node's inputs and outputs. The tests cover those, along
with types, ``Series`` and ``Table``, ``Param`` and widgets, versions, and the hooks that work out inputs and outputs
when a graph is compiled.
"""

import dataclasses
import datetime
import json
import logging
import typing
import warnings
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum, IntEnum, StrEnum
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from annotated_types import Ge, Interval, Len, MaxLen, MinLen, Timezone
from conductor import (
    SKIPPED,
    Choice,
    Deprecation,
    ExternalError,
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeRecord,
    NodeRegistry,
    Param,
    Policy,
    Run,
    Series,
    Source,
    Table,
    Upgrade,
    host_type,
    make_graph,
    node,
    subgraph,
)
from conductor.widgets import Dropdown, FileUpload, ListInput, Textarea, TextInput, Widget
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema


def registry_of(*definitions):
    registry = NodeRegistry()
    registry.add(*definitions)
    return registry


def compiled_graph(*definitions, record=None, constants=None, outputs=None):
    """Compiles a graph against a registry that has only ``definitions``.

    The graph is ``record`` when it's given. Otherwise it's ``definitions`` joined in a chain, with ``outputs`` named
    as ``make_graph`` names them. A last node whose hook gives its outputs needs one named there.
    """
    return Graph.compile(
        record
        if record is not None
        else make_graph(*definitions, constants=constants or {}, outputs=outputs or {}),
        registry_of(*definitions),
    )


def node_record(definition_id, *, version=1, constants=None, graph_inputs=(), sources=None):
    """Builds one node of a graph record.

    Args:
        definition_id: The id of the node's definition.
        version: The definition's version.
        constants: A constant value for each input that gets one, by input name.
        graph_inputs: Inputs that each get a graph input of the same name.
        sources: For each input wired to an edge, the ``(node, output)`` the edge starts from.
    """
    inputs: dict[str, Any] = {
        name: InputEntry(constant=value) for name, value in (constants or {}).items()
    }
    inputs |= {name: InputEntry(graph_input=name) for name in graph_inputs}
    inputs |= {
        name: InputEntry(sources=(Source(*source),)) for name, source in (sources or {}).items()
    }
    return NodeRecord(node=definition_id, version=version, inputs=inputs)


def codes(graph):
    return [p.type for p in graph.problems]


def declared(function, **options):
    """Declares ``function`` as a node with the id ``x``.

    A fault in the declaration raises here, so a test can catch it.
    """
    title = function.__name__.replace("_", " ").strip().capitalize()
    return node("x", **{"title": title, "description": "For a test.", **options})(function)


def field_of(definition, name, side="inputs"):
    return next(f for f in getattr(definition.describe(), side) if f.name == name)


@host_type("Text", title="Text")
class Text(str):
    pass


@host_type("Whole", title="Whole")
class Whole(int):
    pass


class Invoice(TypedDict):
    name: str
    amount: int


# -- declaration errors ------------------------------------------------------------------------


def test_a_wrong_declaration_is_a_type_error_with_a_code_at_the_node():
    """A declaration problem's ``loc`` is the node, as a graph problem's is, and the error's text starts with it."""

    def bad(text: str):
        return text

    with pytest.raises(NodeDeclarationError) as raised:
        declared(bad)
    assert isinstance(raised.value, TypeError)
    assert (raised.value.problems[0].type, raised.value.problems[0].loc) == (
        "output_without_type",
        ("x",),
    )
    assert str(raised.value).startswith("x (version 1): the node has no return type.")


def test_two_versions_of_a_node_with_the_same_fault_read_apart_by_their_version():
    """The version is in the error's text after the node's id; the ``loc`` is the place alone."""

    def bad(text: str):
        return text

    texts = []
    for version in (1, 2):
        with pytest.raises(NodeDeclarationError) as raised:
            node("x", title="X", description="For a test.", version=version)(bad)
        assert raised.value.problems[0].loc == ("x",)
        texts.append(str(raised.value).split(":")[0])
    assert texts == ["x (version 1)", "x (version 2)"]


def test_a_node_id_must_have_something_in_it_and_no_character_a_source_or_waiting_id_uses():
    for wrong in ("", "a.b", "a/b", "a@b", "a~b"):
        with pytest.raises(NodeDeclarationError) as raised:
            node(wrong, title="Wrong", description="For a test.")
        # an id that isn't a name can't be a place, so the problem has no loc
        assert (raised.value.problems[0].type, raised.value.problems[0].loc) == (
            "invalid_node_id",
            (),
        )
        assert "can't contain any of '.', '/', '@', '~', since" in raised.value.problems[0].msg

    def amount() -> int:
        return 1

    assert node("Beløb i kr", title="Amount", description="For a test.")(amount).id == "Beløb i kr"


def test_node_given_a_record_in_place_of_a_function_raises_pythons_own_error_that_it_is_not_callable():
    with pytest.raises(TypeError, match="is not a callable object"):
        node("enrich", title="Enrich", description="For a test.")(
            GraphRecord.model_validate({"nodes": {}})
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("title", "description", "empty"),
    [
        ("", "Does something.", "title"),
        ("Something", "", "description"),
        ("   ", "Does something.", "title"),
        ("Something", "   ", "description"),
    ],
)
def test_an_empty_title_or_description_is_refused_where_a_node_or_a_subgraph_is_declared(
    title, description, empty
):
    for declaration in (
        lambda: node("x", title=title, description=description),
        lambda: subgraph("x", title=title, description=description),
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            declaration()
        assert (raised.value.problems[0].type, raised.value.problems[0].loc) == (
            "empty_title_or_description",
            ("x",),
        )
        assert f"('x') has no {empty}" in str(raised.value)


@pytest.mark.parametrize(
    ("title", "description", "not_text"),
    [(5, "Does something.", "title"), ("Something", None, "description")],
)
def test_a_title_or_description_that_isnt_text_is_refused_as_an_empty_one_is(
    title, description, not_text
):
    for declaration in (
        lambda: node("x", title=title, description=description),
        lambda: subgraph("x", title=title, description=description),
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            declaration()
        assert (raised.value.problems[0].type, raised.value.problems[0].ctx["empty"]) == (
            "empty_title_or_description",
            not_text,
        )


@pytest.mark.parametrize("version", [0, -3, True, "2", 1.5])
def test_a_version_that_isnt_a_whole_number_of_one_or_more_is_refused_where_the_node_is_declared(
    version,
):
    for declaration in (
        lambda: node("x", title="X", description="For a test.", version=version),
        lambda: subgraph("x", title="X", description="For a test.", version=version),
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            declaration()
        assert (raised.value.problems[0].type, raised.value.problems[0].loc) == (
            "invalid_version",
            ("x",),
        )


def test_a_node_id_that_isnt_text_is_an_invalid_node_id():
    with pytest.raises(NodeDeclarationError) as raised:
        node(5, title="Five", description="For a test.")  # type: ignore[arg-type]
    assert raised.value.problems[0].type == "invalid_node_id"


def test_a_node_id_in_a_record_with_a_tilde_is_refused_by_the_record_validator():
    registry = NodeRegistry()

    @registry.node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": {"a~b": {"node": "upper"}}}), registry)
    assert [(p.type, p.loc, p.ctx["error"]) for p in refused.value.problems] == [
        ("invalid_record", ("record", "nodes", "a~b", "[key]"), "not_a_name")
    ]


def test_an_async_function_is_refused():
    async def fetch(url: str) -> str:
        return url

    with pytest.raises(
        NodeDeclarationError, match=r"^x \(version 1\): a node is a plain def"
    ) as raised:
        declared(fetch)
    assert (raised.value.problems[0].type, raised.value.problems[0].loc) == ("async_run", ("x",))


def test_a_default_its_type_cannot_read_fails_where_it_is_written():
    def f(x: int = "abc") -> int:
        return x

    with pytest.raises(NodeDeclarationError, match="'x'|x:") as raised:
        declared(f)
    assert raised.value.problems[0].type == "default_invalid"


def test_a_default_outside_its_own_bounds_fails_where_it_is_written():
    def f(x: Annotated[int, Param(ge=0)] = -1) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.problems[0].type == "default_invalid"


def test_a_series_default_is_checked_against_the_limits_of_each_item():
    def f(xs: Annotated[Series[int], Param(ge=0)] = [1, -1]) -> int:  # noqa: B006
        return sum(xs)

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.problems[0].type == "default_invalid"


def test_an_unknown_param_argument_is_pythons_own_type_error():
    with pytest.raises(TypeError, match="'titel'"):
        Param(titel="Times")  # type: ignore[call-arg]


def test_a_bound_written_wrong_is_refused_where_it_is_written():
    def on_text(x: Annotated[str, Param(ge=1)]) -> str:
        return x

    def on_table(x: Annotated[Table, Param(ge=1)]) -> int:
        return 0

    def on_text_or_none(x: Annotated[str | None, Param(ge=0)] = None) -> str:
        return ""

    def on_texts(x: Annotated[Series[str], Param(ge=0)]) -> str:
        return ""

    for wrong in (on_text, on_table, on_text_or_none, on_texts):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        assert raised.value.problems[0].type == "param_limit_not_supported"


def test_an_annotated_types_limit_with_no_rule_for_its_type_is_refused_as_a_param_limit_is():
    """Pydantic builds ``Annotated[int, MaxLen(3)]`` and then refuses every value, so the graph would compile and
    never run. It builds a ``Timezone`` and never checks it, so a naive datetime would pass."""

    def on_number(x: Annotated[int, MaxLen(3)]) -> int:
        return x

    def on_items(x: list[Annotated[int, Len(1, 3)]]) -> int:
        return 0

    def on_text_or_none(x: Annotated[str, Interval(ge=0)] | None = None) -> str:
        return ""

    def on_a_timezone(x: Annotated[datetime.datetime, Timezone(...)]) -> str:
        return ""

    def on_a_listed_timezone(x: list[Annotated[datetime.datetime, Timezone("UTC")]]) -> str:
        return ""

    for wrong in (on_number, on_items, on_text_or_none, on_a_timezone, on_a_listed_timezone):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        assert raised.value.problems[0].type == "param_limit_not_supported"


def test_a_length_limit_on_a_table_counts_its_rows_written_with_param_or_annotated_types():
    class Row(TypedDict):
        a: int

    @node("few-rows", title="Few rows", description="For a test.")
    def few_rows(
        rows: Annotated[Table[Row], MinLen(1)], more: Annotated[Table, Param(max_length=1)]
    ) -> int:
        return rows.height

    graph = compiled_graph(few_rows)
    assert graph.run({"rows": [{"a": 1}], "more": []}).outputs == {"result": 1}
    for given, code in (
        ({"rows": [], "more": []}, "too_short"),
        ({"rows": [{"a": 1}], "more": [{"b": 1}] * 2}, "too_long"),
    ):
        with pytest.raises(InvalidCallError) as refused:
            graph.run(given)
        assert refused.value.problems[0].ctx["errors"][0]["type"] == code


def test_a_limit_on_one_member_of_a_union_limits_that_member_only():
    @node("int-or-text", title="Int or text", description="For a test.")
    def int_or_text(x: Annotated[int, Ge(0)] | str) -> str:
        return str(x)

    graph = compiled_graph(int_or_text)
    assert graph.run({"x": "a"}).outputs == {"result": "a"}
    assert graph.run({"x": 3}).outputs == {"result": "3"}
    with pytest.raises(InvalidCallError):
        graph.run({"x": -1})


def _not_negative(value: int) -> int:
    if value < 0:
        raise ValueError("below zero")
    return value


def test_metadata_on_a_series_is_for_its_items():
    """A validator, like a limit, on a ``Series`` checks each item; it was dropped and ``-1`` passed."""

    @node("every-item", title="Every item", description="For a test.")
    def every_item(xs: Annotated[Series[int], AfterValidator(_not_negative)]) -> int:
        return len(xs)

    graph = compiled_graph(every_item)
    assert graph.run({"xs": [1, 2]}).outputs == {"result": 2}
    with pytest.raises(InvalidCallError):
        graph.run({"xs": [1, -1]})


def test_a_table_row_that_names_no_columns_is_refused_where_the_node_is_declared():
    """A table read for a dataclass row would keep every column it was given, as a bare table does."""

    @dataclass(frozen=True)
    class Spot:
        x: float

    class Point(TypedDict):
        x: float

    class Holder(TypedDict):
        spots: Table[int]

    def on_dataclass(points: Table[Spot]) -> int:
        return 0

    def on_number_in_a_list() -> list[Table[int]]:
        return []

    def on_a_dict_of_one_type(points: Table[dict[str, int]] | None = None) -> int:
        return 0

    def in_a_record(holder: Holder) -> int:
        return 0

    type Spots = Table[Spot]

    def through_an_alias(spots: Spots) -> int:
        return 0

    for wrong in (
        on_dataclass,
        on_number_in_a_list,
        on_a_dict_of_one_type,
        in_a_record,
        through_an_alias,
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        assert raised.value.problems[0].type == "table_row_not_a_typed_dict", wrong

    @node("keep-point", title="Keep point", description="For a test.")
    def keep_point(points: Table[Point], anything: Table[dict[str, Any]]) -> Table[Point]:
        return pl.DataFrame({"x": [1.0], "y": [2.0]})

    run = compiled_graph(keep_point).run({"points": [{"x": 1.0}], "anything": [{"y": 1}]})
    assert run.outputs["result"].columns == ["x"]


def test_a_generic_typed_dict_row_names_its_columns_with_its_arguments_filled_in():
    class Cell[T](TypedDict):
        value: T

    @node("keep-cells", title="Keep cells", description="For a test.")
    def keep_cells(cells: Table[Cell[int]]) -> Table[Cell[int]]:
        return pl.DataFrame({"value": [1], "extra": ["x"]})

    graph = compiled_graph(keep_cells)
    assert graph.inputs["cells"].type == "Table[value: int]"
    assert graph.run({"cells": [{"value": 1}]}).outputs["result"].columns == ["value"]


def test_a_record_field_written_as_text_that_names_nothing_is_a_declaration_problem():
    class Unknown(TypedDict):
        x: "Missing"  # type: ignore[name-defined]  # noqa: F821

    def reads(row: Unknown) -> int:
        return 0

    with pytest.raises(NodeDeclarationError) as raised:
        declared(reads)
    assert raised.value.problems[0].type == "type_not_supported"


def test_a_dict_returned_is_one_output_and_only_a_loose_dataclass_is_refused():
    def summary(labels: Series[str]) -> dict[str, int]:
        return {label: labels.count(label) for label in labels}

    def anything(x: int) -> Any:
        return x

    for one in (summary, anything):
        assert list(declared(one).outputs) == ["result"], one
    assert compiled_graph(declared(summary)).run({"labels": ["a", "b", "a"]}).outputs == {
        "result": {"a": 2, "b": 1}
    }

    @dataclass
    class Loose:
        a: int

    def loose(x: int) -> Loose:
        return Loose(x)

    with pytest.raises(NodeDeclarationError) as raised:
        declared(loose)
    assert raised.value.problems[0].type == "outputs_not_frozen"


def test_a_class_pydantic_cannot_check_is_refused_where_the_class_is_declared():
    with pytest.raises(NodeDeclarationError) as raised:

        @host_type("thing", title="Thing")
        class Thing:
            pass

    assert raised.value.problems[0].type == "host_type_not_supported"


def test_a_series_of_tables_is_accepted_as_many_tables():
    def f(x: Series[Table[Invoice]]) -> int:
        return 1

    field = declared(f).inputs["x"]
    assert field.is_series is True and field.type.id == "Table[name: str, amount: int]"


def test_a_definition_is_frozen():
    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    with pytest.raises(FrozenInstanceError):
        upper.title = "Shout"  # type: ignore[misc]


# -- a node and what it declares -------------------------------------------------------------


def test_a_node_is_called_like_its_function_and_declares_from_its_signature():
    @node("split_lines", title="Split lines", description="Splits a text into lines.")
    def split_lines(text: str) -> Series[str]:
        """Splits a text into lines.

        One per line break."""
        return text.splitlines()

    assert split_lines("a\nb") == ["a", "b"]
    described = split_lines.describe()
    assert (described.title, described.description, described.category, described.version) == (
        "Split lines",
        "Splits a text into lines.",
        "",
        1,
    )
    assert [(f.name, f.required) for f in described.inputs] == [("text", True)]
    assert [f.name for f in described.outputs] == ["result"]


def test_describe_is_the_node_as_json_with_no_keys_beyond_its_own():
    @node("v", title="V1", description="For a test.")
    def v1(text: str) -> str:
        return text

    @node(
        "v",
        title="V2",
        description="For a test.",
        replaces=v1,
        tags=("words",),
        policy=Policy(retries=2),
    )
    def v2(text: Annotated[str, Param(widget=Textarea(rows=2))]) -> Text:
        return Text(text)

    described = json.loads(v2.describe().model_dump_json())
    assert set(described) == {
        "id",
        "version",
        "versions",
        "title",
        "description",
        "category",
        "tags",
        "extra",
        "docs",
        "deprecation",
        "inputs",
        "outputs",
        "takes_added_inputs",
        "computes_outputs",
        "asks",
    }
    assert (described["id"], described["version"], described["versions"], described["tags"]) == (
        "v",
        2,
        [2, 1],
        ["words"],
    )
    assert described["inputs"][0]["widget"] == {"type": "textarea", "rows": 2}
    assert described["outputs"][0]["type"] == "Text"


def test_a_sink_node_declares_no_outputs_and_runs():
    seen = []

    @node("log", title="Log", description="For a test.")
    def log(text: str) -> None:
        seen.append(text)

    graph = compiled_graph(log)
    assert log.outputs == {} and graph.problems == [] and list(graph.outputs) == []
    assert graph.run({"text": "hi"}).status == "complete" and seen == ["hi"]


def test_each_of_several_outputs_has_its_own_type_and_title():
    @dataclass(frozen=True)
    class Parts:
        name: str
        count: int | None
        case_sensitive: bool
        renamed: Annotated[float, Param(title="Score")]

    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Parts:
        return Parts(text, None, True, 1.0)

    outputs = {f.name: f for f in parts.describe().outputs}
    assert {n: f.type for n, f in outputs.items()} == {
        "name": "str",
        "count": "int",
        "case_sensitive": "bool",
        "renamed": "float",
    }
    assert {n: f.title for n, f in outputs.items()} == {
        "name": None,
        "count": None,
        "case_sensitive": None,
        "renamed": "Score",
    }
    # The output typed `| None` may be None. The one next to it may not.
    assert {"type": "null"} in outputs["count"].json_schema["anyOf"] and "anyOf" not in outputs[
        "name"
    ].json_schema
    assert compiled_graph(parts).run({"text": "a"}).results["parts"] == {
        "name": "a",
        "count": None,
        "case_sensitive": True,
        "renamed": 1.0,
    }


def test_an_untitled_field_has_no_title_and_none_is_guessed_from_its_name():
    @node("f", title="F", description="For a test.")
    def f(
        case_sensitive: bool, texts: Series[str], titled: Annotated[str, Param(title="Given")]
    ) -> str:
        return ""

    assert [i.title for i in f.describe().inputs] == [None, None, "Given"]


# -- the type of an input or output ------------------------------------------------------------


def test_an_input_or_output_that_may_be_none_is_described_by_its_type_without_none():
    @node("f", title="F", description="For a test.")
    def f(many: Series[Text | None], one: Text | None) -> Text | None:
        return None

    assert [(i.name, i.type, i.is_series) for i in f.describe().inputs] == [
        ("many", "Text", True),
        ("one", "Text", False),
    ]
    output = field_of(f, "result", "outputs")
    assert output.type == "Text" and {"type": "null"} in output.json_schema["anyOf"]
    assert compiled_graph(f, constants={"f.many": [None], "f.one": None}).run().results["f"] == {
        "result": None
    }


def test_a_none_default_on_an_optional_type_is_not_required_and_arrives_as_none():
    @node("f", title="F", description="For a test.")
    def f(
        x: int | None = None, picked: Annotated[str | None, Param(widget=TextInput())] = None
    ) -> str:
        return f"{x}-{picked}"

    assert [i.required for i in f.describe().inputs] == [False, False]
    assert compiled_graph(f).run().results["f"]["result"] == "None-None"


def test_a_series_that_may_be_none_takes_none_given_as_a_constant_or_left_out_but_no_item_may_be_none():
    @node("f", title="F", description="For a test.")
    def f(xs: Series[int] | None = None) -> str:
        return repr(xs)

    graph = Graph.compile(make_graph(f, defaults={"f.xs": None}), registry_of(f))
    assert compiled_graph(f).run().results["f"]["result"] == "None"
    assert graph.run().results["f"]["result"] == "None"
    assert graph.run({"xs": None}).results["f"]["result"] == "None"
    assert compiled_graph(f, constants={"f.xs": None}).run().results["f"]["result"] == "None"
    assert graph.run({"xs": [1, 2]}).results["f"]["result"] == "[1, 2]"
    with pytest.raises(InvalidCallError):
        graph.run({"xs": [1, None]})


def test_a_series_that_may_not_be_none_refuses_none_and_a_default_of_none():
    @node("f", title="F", description="For a test.")
    def f(xs: Series[int]) -> str:
        return repr(xs)

    with pytest.raises(InvalidCallError) as refused:
        compiled_graph(f).run({"xs": None})
    assert [p.type for p in refused.value.problems] == ["invalid_input"]
    assert codes(compiled_graph(f, constants={"f.xs": None})) == ["invalid_value"]

    def g(xs: Series[int] = None) -> int:  # type: ignore[assignment]
        return 0

    with pytest.raises(NodeDeclarationError) as raised:
        declared(g)
    assert raised.value.problems[0].type == "default_none_not_taken"


def test_none_on_an_input_that_takes_none_is_not_checked_against_its_limits():
    @node("f", title="F", description="For a test.")
    def f(x: Annotated[str | None, Param(max_length=2)] = "ab") -> str:
        return repr(x)

    graph = Graph.compile(make_graph(f, defaults={"f.x": "ab"}), registry_of(f))
    assert graph.run({"x": None}).results["f"]["result"] == "None"
    assert graph.run({"x": "ok"}).results["f"]["result"] == "'ok'"


def test_bounds_are_on_the_field_an_editor_reads():
    @node("f", title="F", description="For a test.")
    def f(
        n: Annotated[int, Param(ge=0, le=9)], s: Annotated[str, Param(max_length=3)] = "a"
    ) -> int:
        return n

    assert [i.json_schema for i in f.describe().inputs] == [
        {"type": "integer", "minimum": 0, "maximum": 9},
        {"type": "string", "maxLength": 3},
    ]


def test_a_constant_of_a_series_input_is_checked_item_by_item_at_compile_and_arrives_as_a_list():
    @node("total", title="Total", description="For a test.")
    def total(xs: Annotated[Series[int], Param(ge=0)]) -> str:
        return f"{type(xs).__name__}:{sum(xs)}"

    def with_constant(values):
        return compiled_graph(total, constants={"total.xs": values})

    assert codes(with_constant([1, -1])) == ["invalid_value"]
    assert with_constant([1, 2]).run().results["total"]["result"] == "list:3"


def test_a_subclass_of_a_builtin_is_checked_as_the_builtin_then_converted_to_the_subclass():
    @node("kind", title="Kind", description="For a test.")
    def kind(t: Text) -> str:
        return type(t).__name__

    assert compiled_graph(kind).run({"t": "a"}).results["kind"]["result"] == "Text"


def test_a_subclass_output_fits_an_input_of_its_parent_type_and_not_the_other_way():
    @node("make", title="Make", description="For a test.")
    def make(x: str) -> Text:
        return Text(x)

    @node("plain", title="Plain", description="For a test.")
    def plain(x: str) -> str:
        return x

    @node("on_str", title="On str", description="For a test.")
    def on_str(t: str) -> str:
        return type(t).__name__

    @node("on_text", title="On text", description="For a test.")
    def on_text(t: Text) -> str:
        return t

    assert compiled_graph(make, on_str).run({"x": "a"}).results["on_str"]["result"] == "Text"
    mistyped = make_graph(plain, on_text, edges=[("plain.result", "on_text.t")])
    assert codes(compiled_graph(plain, on_text, record=mistyped)) == ["type_mismatch"]


def test_a_bool_and_a_date_reach_the_next_node_as_their_own_type():
    @node("flags", title="Flags", description="For a test.")
    def flags(on: bool, day: datetime.date) -> bool:
        return on

    @node("flip", title="Flip", description="For a test.")
    def flip(on: bool) -> bool:
        return not on

    @node("when", title="When", description="For a test.")
    def when(day: datetime.date) -> datetime.date:
        return day + datetime.timedelta(days=1)

    graph = compiled_graph(
        flags,
        flip,
        when,
        record=GraphRecord(
            nodes={
                "f": node_record("flags", graph_inputs=("on", "day")),
                "n": node_record("flip", sources={"on": ("f", "result")}),
                "w": node_record("when", graph_inputs=("day",)),
            }
        ),
    )
    run = graph.run({"on": True, "day": datetime.date(2026, 1, 1)})
    assert run.results["n"]["result"] is False
    assert run.results["w"]["result"] == datetime.date(2026, 1, 2)


def test_a_pydantic_model_travels_on_an_edge_and_through_json():
    @host_type("Visit", title="Visit")
    class Visit(BaseModel):
        who: str
        when: datetime.date

    @node("make", title="Make", description="For a test.")
    def make(who: str) -> Visit:
        return Visit(who=who, when=datetime.date(2026, 1, 2))

    @node("show", title="Show", description="For a test.")
    def show(visit: Visit) -> str:
        return f"{visit.who} {visit.when.isoformat()}"

    graph = compiled_graph(make, show)
    run = graph.run({"who": "Bo"})
    back = Run.from_json(run.to_json(), graph)
    assert back.results == run.results and back.results["show"]["result"] == "Bo 2026-01-02"
    assert isinstance(back.results["make"]["result"], Visit)


def test_an_any_input_has_a_json_form_before_anything_is_wired():
    @node("f", title="F", description="For a test.")
    def f(x: Any) -> str:
        return ""

    assert json.loads(f.describe().model_dump_json(exclude_none=True))["inputs"][0] == {
        "name": "x",
        "is_series": False,
        "required": True,
        "type": "Any",
        "type_variables": [],
        "json_schema": {},
        "constant_only": False,
        "loops": [],
    }


def test_each_key_a_field_description_writes_is_its_python_name():
    """A field's JSON names each part as Python does, so ``json_schema`` is ``"json_schema"`` in both.

    A host moves between the Python objects and the served JSON, and one concept should have one name.
    """

    @node("f", title="F", description="For a test.")
    def f(x: int) -> str:
        return ""

    field = f.describe().inputs[0]
    written = json.loads(field.model_dump_json())
    assert set(written) <= set(type(field).model_fields)
    assert written["json_schema"] == field.json_schema == {"type": "integer"}


def test_describe_writes_each_default_as_its_type_writes_it_as_json():
    @host_type("point", title="Point")
    @dataclass
    class Point:
        x: int

    @node("f", title="F", description="For a test.")
    def f(b: list[Point] = [Point(1)]) -> str:  # noqa: B006, B008
        return ""

    written = json.loads(f.describe().model_dump_json())
    assert [i["default"] for i in written["inputs"]] == [[{"x": 1}]]


def test_a_parameter_named_like_a_pydantic_model_attribute_checks_and_runs_without_a_warning():
    @node("f", title="F", description="For a test.")
    def f(schema: int, copy: int, json: int, model_config: int, json_: int) -> str:
        return f"{schema}{copy}{json}{model_config}{json_}"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = compiled_graph(f).run(
            {"schema": 1, "copy": 2, "json": 3, "model_config": 4, "json_": 5}
        )
    assert run.results["f"]["result"] == "12345"


# -- Series and Table --------------------------------------------------------------------------


def test_a_series_input_takes_many_values_of_its_element_type_and_a_list_is_one_value():
    @node("f", title="F", description="For a test.")
    def f(texts: Series[Text], numbers: list[float]) -> str:
        return ""

    assert [(i.is_series, i.type) for i in f.describe().inputs] == [
        (True, "Text"),
        (False, "list[float]"),
    ]


def test_a_table_declares_its_columns_and_their_types():
    @node("read", title="Read", description="For a test.")
    def read(text: str) -> Table[Invoice]:
        return pl.DataFrame({"name": [text], "amount": [1]})

    @node("bare", title="Bare", description="For a test.")
    def bare(text: str) -> Table:
        return pl.DataFrame()

    output = field_of(read, "result", "outputs")
    columns = {
        c["properties"]["name"]["const"]: c["properties"]["values"]["items"]
        for c in output.json_schema["properties"]["columns"]["prefixItems"]
    }
    assert (
        not output.is_series
        and output.type == "Table[name: str, amount: int]"
        and columns == {"name": {"type": "string"}, "amount": {"type": "integer"}}
    )
    any_columns = field_of(bare, "result", "outputs")
    assert (any_columns.is_series, any_columns.type) == (
        False,
        "Table",
    ) and "prefixItems" not in any_columns.json_schema["properties"]["columns"]


# -- Param and widgets -------------------------------------------------------------------------


def test_a_param_with_a_widget_describes_with_one_title_on_the_field():
    languages = (Choice(id="en", title="English"), Choice(id="da", title="Dansk"))

    @node("f", title="F", description="For a test.")
    def f(
        lang: Annotated[
            str,
            Param(title="Language", description="Which one", choices=languages, widget=Dropdown()),
        ] = "en",
    ) -> str:
        return lang

    described = f.describe().inputs[0]
    assert (described.title, described.description, described.type) == (
        "Language",
        "Which one",
        "str",
    )
    assert described.widget == {"type": "dropdown"} and described.choices == (
        Choice("en", "English"),
        Choice("da", "Dansk"),
    )


def test_a_param_with_a_title_alone_is_an_input_with_no_control():
    @node("f", title="F", description="For a test.")
    def f(x: Annotated[str, Param(title="Text")], y: str) -> str:
        return x

    assert [i.widget is not None for i in f.describe().inputs] == [False, False]


def test_a_widget_is_frozen_keyword_only_and_refuses_a_key_it_does_not_have():
    with pytest.raises(ValidationError):
        Textarea(row=5)
    with pytest.raises(TypeError):
        Textarea(5)  # type: ignore[misc]
    shown = TextInput()
    with pytest.raises(ValidationError):
        shown.type = "textarea"  # type: ignore[misc]


def test_a_widget_dumps_its_type_and_its_own_config_only():
    assert Dropdown().model_dump(mode="json") == {"type": "dropdown"}
    assert ListInput().model_dump(mode="json") == {"type": "list-input"}
    assert Textarea().type == "textarea"


def test_a_file_upload_offers_every_kind_as_an_empty_tuple_and_takes_no_none():
    assert FileUpload().accept == ()
    assert FileUpload().model_dump(mode="json")["accept"] == []
    with pytest.raises(ValidationError):
        FileUpload(accept=None)  # type: ignore[arg-type]


def test_a_hosts_own_widget_compiles_runs_and_is_described_with_its_config():
    class EntityPicker(Widget):
        type: Literal["entity-picker"] = "entity-picker"
        entity_kind: str = ""

    @node("greet", title="Greet", description="For a test.")
    def greet(who: Annotated[str, Param(widget=EntityPicker(entity_kind="user"))]) -> str:
        return f"hi {who}"

    assert greet.describe().inputs[0].widget == {"type": "entity-picker", "entity_kind": "user"}
    graph = compiled_graph(greet)
    assert graph.problems == [] and graph.run({"who": "Bo"}).results["greet"]["result"] == "hi Bo"


class Client:
    """A host's class with no JSON form. Pydantic can only check it with ``isinstance``."""


@pytest.mark.parametrize(
    "hint", [Client, pl.DataFrame, type[str]], ids=["host class", "frame", "class itself"]
)
def test_a_hosts_widget_with_a_field_that_has_no_json_form_is_refused_where_its_class_ends(
    hint: Any,
) -> None:
    with pytest.raises(
        PydanticInvalidForJsonSchema,
        match=r"\.EntityPicker\.source has a type with no JSON form.*Field\(exclude=True\)",
    ):

        class EntityPicker(Widget):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            source: hint  # type: ignore[valid-type]
            type: Literal["entity-picker"] = "entity-picker"


def test_a_field_with_no_json_form_in_a_model_inside_a_hosts_widget_is_named_with_that_model() -> (
    None
):
    class Entity(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        source: Client

    with pytest.raises(
        PydanticInvalidForJsonSchema,
        match=r"\.Entity\.source\b.* has a type with no JSON form, but \S+\.EntityPicker is",
    ):

        class EntityPicker(Widget):
            entity: Entity
            type: Literal["entity-picker"] = "entity-picker"


def test_a_hosts_widget_may_keep_a_field_with_no_json_form_out_of_its_json() -> None:
    class EntityPicker(Widget):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        type: Literal["entity-picker"] = "entity-picker"
        source: Client | None = Field(default=None, exclude=True)

    assert EntityPicker(source=Client()).model_dump(mode="json") == {"type": "entity-picker"}


# -- versions, policy, deprecation -------------------------------------------------------------


def test_a_version_number_is_its_place_in_the_chain_and_each_keeps_its_own_settings():
    @node(
        "v",
        title="V1",
        description="For a test.",
        policy=Policy(retries=1),
        deprecation=Deprecation(header="Use v3"),
    )
    def v1(x: str) -> str:
        return "one"

    @node("v", title="V2", description="For a test.", replaces=v1, policy=Policy(retries=2))
    def v2(x: str) -> str:
        return "two"

    @node("v", title="V3", description="For a test.", replaces=v2, policy=Policy(concurrency=3))
    def v3(x: str) -> str:
        return "three"

    assert list(v3.versions) == [3, 2, 1] and v3("a") == "three"
    assert [v3.versions[n].policy for n in (1, 2, 3)] == [
        Policy(retries=1),
        Policy(retries=2),
        Policy(concurrency=3),
    ]
    assert v3.versions[1].deprecation == Deprecation(header="Use v3") and v3.deprecation is None
    assert (
        v3.versions[1].describe().deprecation.header == "Use v3"
        and v3.describe().deprecation is None
    )
    assert 4 not in v3.versions
    with pytest.raises(TypeError):  # a definition is frozen, so its versions can't be changed
        v3.versions[4] = v3  # type: ignore[index]
    for saved, body in ((1, "one"), (3, "three")):
        graph = compiled_graph(
            v3,
            record=GraphRecord(nodes={"n": node_record("v", version=saved, graph_inputs=("x",))}),
        )
        assert graph.run({"x": "a"}).results["n"]["result"] == body


def test_a_version_with_no_policy_gets_the_default_of_one_call_at_a_time():
    @node("f", title="F", description="For a test.")
    def f(x: str) -> str:
        return x

    assert f.policy == Policy() and f.policy.concurrency is None


def test_a_deprecation_is_kept_as_written_and_is_never_a_problem():
    @node("old", title="Old", description="For a test.", deprecation=Deprecation())
    def old(x: str) -> str:
        return x

    @node(
        "going",
        title="Going",
        description="For a test.",
        deprecation=Deprecation(header="Going", alternative="new"),
    )
    def going(x: str) -> str:
        return x

    assert old.deprecation == Deprecation() and going.deprecation.alternative == "new"
    assert compiled_graph(going).problems == []


def test_a_chain_written_wrong_is_refused_where_it_is_written():
    def f(x: int) -> int:
        return x

    def g(y: int) -> int:
        return y

    other = node("other", title="F", description="For a test.")(f)
    with pytest.raises(NodeDeclarationError) as raised:
        node("x", title="F", description="For a test.", replaces=other)(f)
    assert raised.value.problems[0].type == "replaces_another_node"
    with pytest.raises(NodeDeclarationError) as raised:
        node("x", title="F", description="For a test.", upgrade=Upgrade())(f)
    assert raised.value.problems[0].type == "upgrade_without_replaces"
    with pytest.raises(NodeDeclarationError) as raised:
        node(
            "x",
            title="G",
            description="For a test.",
            replaces=node("x", title="F", description="For a test.")(f),
        )(g)
    assert raised.value.problems[0].type == "upgrade_missing"


def test_a_subgraph_may_be_declared_at_a_version_of_its_own():
    @node("read", title="Read", description="For a test.")
    def read(text: str) -> str:
        return text

    inner = subgraph("inner", title="Inner", description="For a test.", version=3)(make_graph(read))
    assert inner.version == 3 and list(inner.versions) == [3]
    assert (
        next(n for n in registry_of(read, inner).describe().nodes if n.id == "inner").version == 3
    )
    graph = compiled_graph(
        read,
        inner,
        record=GraphRecord(nodes={"e": node_record("inner", version=3, graph_inputs=("text",))}),
    )
    assert graph.problems == [] and list(graph.inputs) == ["text"]


# -- hooks -------------------------------------------------------------------------------------


def test_a_hook_parameter_must_name_an_input_by_name():
    def f(x: int) -> dict:
        return {}

    def names_nothing(nope: int):
        return {}

    def star(*args):
        return {}

    def double_star(**kwargs):
        return {}

    for hook, code in (
        (names_nothing, "hook_parameter_unknown"),
        (star, "hook_parameter_not_by_name"),
        (double_star, "hook_parameter_var_keyword"),
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(f, compute_outputs=hook)
        assert raised.value.problems[0].type == code, hook


def test_a_compute_added_inputs_hook_needs_a_var_keyword_parameter():
    def f(x: int) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f, compute_added_inputs=lambda: {})
    assert raised.value.problems[0].type == "compute_added_inputs_without_var_keyword"


def test_an_added_input_whose_default_is_skipped_is_the_hooks_failure_and_the_function_is_never_called():
    called = []

    def added() -> dict[str, Any]:
        return {"extra": (str, SKIPPED)}

    @node("adds", title="Adds", description="For a test.", compute_added_inputs=added)
    def adds(text: str, **extra: str) -> str:
        called.append(extra)
        return text

    [problem] = compiled_graph(adds).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == (
        "compute_hook_returned_invalid",
        ("adds",),
        "default_skipped_on_input",
    )
    with pytest.raises(InvalidCallError):
        compiled_graph(adds).run({"text": "a"})
    assert called == []


def test_a_hook_gets_the_constant_or_else_the_declared_default():
    seen = []

    def outputs(n: int):
        seen.append(n)
        return {f"o{i}": int for i in range(n)}

    @node("fan", title="Fan", description="For a test.", compute_outputs=outputs)
    def fan(n: int = 2) -> dict:
        return {f"o{i}": i for i in range(n)}

    assert (
        compiled_graph(fan, outputs={"o0": "fan.o0"}).describe().nodes["fan"].outputs[1].name
        == "o1"
    )
    assert seen == [2]
    compiled_graph(fan, constants={"fan.n": 3}, outputs={"o0": "fan.o0"})
    assert seen == [2, 3]


def test_a_hook_is_not_called_when_a_value_it_needs_has_none():
    seen = []

    def outputs(n: int):
        seen.append(n)
        return {"result": int}

    @node("fan", title="Fan", description="For a test.", compute_outputs=outputs)
    def fan(n: int) -> dict:
        return {"result": n}

    graph = compiled_graph(fan, outputs={"result": "fan.result"})
    assert (
        seen == []
        and codes(graph) == ["input_has_no_value_constant_only"]
        and graph.problems[0].loc == ("fan", "n")
    )


def test_an_input_an_added_inputs_hook_reads_takes_no_edge_and_the_hook_is_not_called():
    seen = []

    def added(n: int = 2):
        seen.append(n)
        return {f"part{i}": str for i in range(n)}

    @node("number", title="Number", description="For a test.")
    def number(x: int) -> int:
        return x

    @node("joiner", title="Joiner", description="For a test.", compute_added_inputs=added)
    def joiner(n: int = 2, **parts: Any) -> str:
        return ",".join(parts.values())

    graph = compiled_graph(
        number, joiner, record=make_graph(number, joiner, edges=[("number.result", "joiner.n")])
    )
    assert codes(graph) == ["input_takes_only_a_constant"] and seen == []


def test_each_version_has_its_own_hooks():
    @node("h", title="H1", description="For a test.", compute_outputs=lambda: {"first": str})
    def h1(x: str) -> dict:
        return {"first": x}

    @node(
        "h",
        title="H2",
        description="For a test.",
        replaces=h1,
        compute_outputs=lambda: {"second": str},
    )
    def h2(x: str) -> dict:
        return {"second": x}

    for saved, name in ((1, "first"), (2, "second")):
        graph = compiled_graph(
            h2,
            record=GraphRecord(nodes={"n": node_record("h", version=saved, graph_inputs=("x",))}),
        )
        assert graph.run({"x": "a"}).results["n"] == {name: "a"}


def test_an_output_a_hook_declares_may_be_none_when_its_type_says_so():
    @node(
        "maybe",
        title="Maybe",
        description="For a test.",
        compute_outputs=lambda: {"result": int | None},
    )
    def maybe(x: str) -> dict:
        return {"result": None}

    graph = compiled_graph(maybe, outputs={"result": "maybe.result"})
    output = graph.describe().nodes["maybe"].outputs[0]
    assert output.type == "int" and {"type": "null"} in output.json_schema["anyOf"]
    assert graph.run({"x": "a"}).results["maybe"]["result"] is None


def test_a_hook_reads_the_element_type_a_series_input_receives():
    def typed_by(items: type):
        return {"result": items}

    @node("lengths", title="Lengths", description="For a test.")
    def lengths(text: str) -> Series[int]:
        return [len(w) for w in text.split()]

    @node("first", title="First", description="For a test.", compute_outputs=typed_by)
    def first(items: Series[Any]) -> dict:
        return {"result": items[0]}

    @node("double", title="Double", description="For a test.")
    def double(n: int) -> int:
        return n * 2

    graph = compiled_graph(
        lengths,
        first,
        double,
        record=make_graph(
            lengths,
            first,
            double,
            edges=[("lengths.result", "first.items"), ("first.result", "double.n")],
        ),
    )
    assert graph.problems == [] and graph.describe().nodes["first"].outputs[0].type == "int"
    assert graph.describe().nodes["first"].per_item_of is None
    assert graph.run({"text": "ab c"}).results["double"]["result"] == 4


def test_a_hook_gets_any_for_an_unwired_any_input_even_with_a_constant():
    seen = []

    def outputs(x: type):
        seen.append(x)
        return {"result": int}

    @node("f", title="F", description="For a test.", compute_outputs=outputs)
    def f(x: Any) -> dict:
        return {"result": 1}

    compiled_graph(f, outputs={"result": "f.result"})
    compiled_graph(f, constants={"f.x": "typed"}, outputs={"result": "f.result"})
    assert seen == [Any, Any]


def test_a_node_whose_hook_gives_no_outputs_leaves_the_graph_runnable():
    @node("empty", title="Empty", description="For a test.", compute_outputs=lambda: {})
    def empty(x: str) -> dict:
        return {}

    # make_graph refuses a node whose hook outputs nothing names, so the record is written out
    graph = compiled_graph(
        empty,
        record=GraphRecord.model_validate(
            {"nodes": {"empty": {"node": "empty", "inputs": {"x": {"graph_input": "x"}}}}}
        ),
    )
    assert (
        graph.problems == [] and graph.outputs == {} and graph.run({"x": "a"}).status == "complete"
    )


def _refuses():
    raise ValueError("cannot say yet")


@node("echo", title="Echo", description="For a test.")
def echo(v: str) -> str:
    return v


def test_a_hook_that_raises_is_its_nodes_one_problem_never_an_exception():
    @node("outs", title="Outs", description="For a test.", compute_outputs=_refuses)
    def outs(x: str) -> dict:
        return {}

    @node("adds", title="Adds", description="For a test.", compute_added_inputs=_refuses)
    def adds(**kwargs: Any) -> str:
        return ""

    for kind, definition, graph_inputs in (("outs", outs, ("x",)), ("adds", adds, ())):
        graph = compiled_graph(
            definition,
            echo,
            record=GraphRecord(
                nodes={
                    "n": node_record(kind, graph_inputs=graph_inputs),
                    "r": node_record("echo", sources={"v": ("n", "result")}),
                }
            ),
        )
        # The node after it has no problem of its own.
        assert [(p.type, p.loc) for p in graph.problems] == [("compute_hook_raised", ("n",))], kind
        assert graph.problems[0].ctx["error"] == "ValueError"


def test_a_hook_is_given_any_for_an_input_whose_edge_is_refused_as_for_one_nothing_is_wired_to():
    """A hook gets ``Any`` for an input whose edge is refused.

    A refused edge has no type to pass on, so the hook sees the input as if nothing were wired to it. The node still
    gets its inputs and outputs.
    """
    arrived: list[Any] = []

    def of_what_arrives(x: type) -> dict[str, Any]:
        arrived.append(x)
        return {"result": x}

    @node("outs", title="Outs", description="For a test.", compute_outputs=of_what_arrives)
    def outs(x: str) -> dict:
        return {}

    graph = compiled_graph(
        echo,
        outs,
        record=GraphRecord(
            nodes={
                "e": node_record("echo", graph_inputs=("v",)),
                "n": node_record("outs", sources={"x": ("e", "nothing")}),
            }
        ),
    )
    assert codes(graph) == ["unknown_output"] and arrived == [Any]
    assert [o.type for o in graph.describe().nodes["n"].outputs] == ["Any"]


def test_a_series_of_tables_from_a_hook_compiles_with_no_problems():
    def h(x: str) -> dict:
        return {}

    assert (
        codes(
            compiled_graph(
                node(
                    "h",
                    title="H",
                    description="For a test.",
                    compute_outputs=lambda: {"result": Series[Table]},
                )(h),
                outputs={"result": "h.result"},
            )
        )
        == []
    )


def test_a_hint_from_a_hook_naming_a_class_that_is_not_declared_is_the_hooks_problem():
    class Sheet(Table):
        pass

    def h(x: str) -> dict:
        return {}

    def giving(hint):
        return lambda: {"result": hint}

    for hint in (Sheet, Series[Sheet]):
        assert codes(
            compiled_graph(
                node("h", title="H", description="For a test.", compute_outputs=giving(hint))(h),
                outputs={"result": "h.result"},
            )
        ) == ["compute_hook_returned_invalid"], hint


def _computing(hook: Any) -> Any:
    """Compiles a graph of one node, ``h``, whose outputs come from ``hook``."""

    def h(x: str = "a") -> dict:
        return {}

    return compiled_graph(
        node("h", title="H", description="For a test.", compute_outputs=hook)(h),
        outputs={"result": "h.result"},
    )


@pytest.mark.parametrize("returned", [None, ["result"], "result"])
def test_a_hook_that_returns_no_mapping_is_the_hooks_failure_saying_what_a_hook_returns(
    returned: Any,
):
    [problem] = _computing(lambda: returned).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == (
        "compute_hook_returned_invalid",
        ("h",),
        "not_a_mapping",
    )
    assert f"returned a {type(returned).__name__}, not a mapping" in problem.msg


@pytest.mark.parametrize(
    ("returns", "written"),
    [(str, "str"), (int | None, "int | None"), (None, "None"), (list[str], "list[str]")],
)
def test_a_hooks_node_whose_return_type_cannot_be_a_mapping_is_refused_when_it_is_declared(
    returns: Any, written: str
):
    """The node returns the hook's outputs as a mapping, so ``-> str`` would fail on every call. It is refused at
    import, where its developer meets it, not on the first run."""

    def one(text: str) -> Any:
        return text

    one.__annotations__["return"] = returns
    with pytest.raises(NodeDeclarationError) as refused:
        node("one", title="One", description="For a test.", compute_outputs=lambda: {"a": int})(one)
    # the return type is written as conductor writes a type, never as NoneType or typing.Optional
    assert [(p.type, p.ctx["returns"]) for p in refused.value.problems] == [
        ("outputs_not_a_mapping", written)
    ]


type _Outputs = dict[str, Any]
type _MaybeOutputs = dict[str, Any] | None


@pytest.mark.parametrize(
    "returns",
    [dict, dict[str, Any], Mapping[str, int], Any, dict | None]
    + [object, _Outputs, _MaybeOutputs, JsonValue],
)
def test_a_hooks_node_may_declare_any_return_type_a_mapping_fits(returns: Any):
    """Whether a mapping fits is asked by the rule an edge is checked by, so an alias, ``object`` and ``JsonValue``
    are taken as an edge would take them."""

    def one(text: str) -> Any:
        return {"a": 1}

    one.__annotations__["return"] = returns
    node("one", title="One", description="For a test.", compute_outputs=lambda: {"a": int})(one)


@pytest.mark.parametrize(
    ("returned", "reason"),
    [
        ({}, "'a' is missing from the outputs the node returned"),
        ("a", "the node must return a mapping with one key per output (a), not a str"),
        ({"a": 1, "zzz": 2}, "'zzz' isn't one of the node's outputs"),
    ],
)
def test_a_return_that_does_not_fit_the_outputs_says_so_whatever_is_wrong_with_it(
    returned: Any, reason: str
):
    """The message opens with what is true of every such return, not with "returned something it doesn't declare",
    which is wrong for an output that is missing."""

    @node("h", title="H", description="For a test.", compute_outputs=lambda: {"a": int})
    def h(x: str = "a") -> dict:
        return returned  # type: ignore[no-any-return]

    cause = compiled_graph(h, outputs={"a": "h.a"}).run().cause
    assert (cause.type, cause.msg) == (
        "invalid_output",
        f"The node's return doesn't fit its outputs: {reason}. Fix what the node returns.",
    )


def test_a_hook_that_names_a_field_with_no_text_is_refused_by_that_name():
    assert [(p.type, p.loc, p.ctx["error"]) for p in _computing(lambda: {1: int}).problems] == [
        ("compute_hook_returned_invalid", ("h", "1"), "not_a_name")
    ]


@pytest.mark.parametrize("key", [1.5, None, (1, 2)], ids=["float", "none", "tuple"])
def test_a_hook_whose_field_name_is_no_text_or_int_is_the_hooks_failure_not_a_raise(key: Any):
    [problem] = _computing(lambda: {"ok": int, key: int}).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == (
        "compute_hook_returned_invalid",
        ("h", str(key)),
        "not_a_name",
    )


def test_a_bug_in_a_host_types_own_schema_hook_raises_as_itself_where_the_class_is_declared():
    """A bug in a host type's own schema code raises right away, where the class is declared.

    A hook fails with a problem only when it names a class with no ``@host_type``. An ``AttributeError`` in the host's
    own code is the host's bug, so it isn't turned into a problem.
    """
    with pytest.raises(AttributeError, match="no_such_method"):

        @host_type("buggy", title="Buggy")
        class Buggy:
            @classmethod
            def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
                return handler.no_such_method(str)


def test_a_hooks_own_exception_is_logged_with_its_traceback_as_a_nodes_own_failure_is(caplog):
    def boom() -> dict[str, Any]:
        return {}["missing"]

    with caplog.at_level(logging.ERROR, logger="conductor"):
        [problem] = _computing(boom).problems
    assert (problem.type, problem.ctx["error"]) == ("compute_hook_raised", "KeyError")
    logged = caplog.records[0]  # logged once each time the graph is compiled
    assert (
        logged.exc_info is not None
        and logged.exc_info[0] is KeyError
        and "'h'" in logged.getMessage()
        and "compute_outputs" in logged.getMessage()
    )


def test_a_hooks_own_exception_text_goes_to_the_log_and_never_into_its_problem(caplog):
    """A hook's exception other than a ``NodeError`` is ``compute_hook_raised``.

    The problem has only the exception's class, since its text might quote a value. The text goes to the log.
    """

    def leaks() -> dict[str, Any]:
        raise ValueError("the secret is 42")

    with caplog.at_level(logging.ERROR, logger="conductor"):
        [problem] = _computing(leaks).problems
    assert (problem.type, problem.ctx) == (
        "compute_hook_raised",
        {"error": "ValueError"},
    )
    assert "secret" not in problem.model_dump_json() and "the secret is 42" in caplog.text


def test_a_field_a_hook_declares_wrong_keeps_the_declarations_code_and_never_its_default():
    """A default ``@node`` would refuse makes ``compute_hook_returned_invalid`` with the declaration's code.

    The declaration's own message quotes the default, a value, so the problem names only the field.
    """
    [problem] = _computing(lambda: {"count": (int, "secret-default")}).problems
    assert (problem.type, problem.ctx["error"]) == (
        "compute_hook_returned_invalid",
        "default_invalid",
    )
    assert "secret-default" not in problem.model_dump_json() and "'count'" in problem.msg


@pytest.mark.parametrize(
    "not_a_type",
    [5, "str", object(), [int], {"a": int}, {int}],
    ids=["an-int", "a-str", "an-instance", "a-list", "a-dict", "a-set"],
)
def test_a_hook_that_gives_a_field_no_type_at_all_fails_with_the_declarations_code(not_a_type: Any):
    """A hook's value that is no type, like ``5`` or ``[int]``, is ``compute_hook_returned_invalid`` with ``type_not_supported``.

    It is the code ``@node`` gives the same value in a signature, and compiling reports it rather than raising.
    """
    [problem] = _computing(lambda: {"count": not_a_type}).problems
    assert (problem.type, problem.ctx["error"]) == (
        "compute_hook_returned_invalid",
        "type_not_supported",
    )


@pytest.mark.parametrize(
    "not_a_type",
    [5, object(), [int], {"a": int}, {int}],
    ids=["an-int", "an-instance", "a-list", "a-dict", "a-set"],
)
def test_a_signature_whose_type_is_no_type_at_all_is_refused_as_not_supported(not_a_type: Any):
    """``def f() -> 5`` or ``-> [int]`` is ``type_not_supported``, a ``NodeDeclarationError``, not a bare ``TypeError``.

    A string in a signature is a forward reference, which ``@node`` resolves, so it is left out here.
    """

    def f() -> int:
        return 1

    f.__annotations__["return"] = not_a_type
    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.problems[0].type == "type_not_supported"


# -- signatures conductor refuses, and signatures it takes as written ------------------------


def test_a_none_default_on_an_input_that_takes_no_none_is_refused():
    def f(x: int = None) -> int:  # type: ignore[assignment]
        return 1

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.problems[0].type == "default_none_not_taken"


def test_a_default_reaches_the_function_as_its_declared_type():
    @node("kind", title="Kind", description="For a test.")
    def kind(x: Whole = 3, t: Text = "a") -> str:  # type: ignore[assignment]
        return f"{type(x).__name__} {type(t).__name__}"

    assert compiled_graph(kind).run().results["kind"]["result"] == "Whole Text"


def test_annotated_metadata_that_is_not_a_param_is_pydantics_so_a_note_is_ignored_and_an_annotated_types_limit_is_checked():
    def noted(x: Annotated[int, "a note"]) -> int:
        return x

    def limited(x: Annotated[int, Ge(0)]) -> str:
        return str(x)

    assert field_of(declared(noted), "x").type == "int"
    assert compiled_graph(declared(noted)).run({"x": 3}).outputs == {"result": 3}
    graph = compiled_graph(
        declared(limited),
        record=GraphRecord(nodes={"l": node_record("x", constants={"x": -1})}),
    )
    assert codes(graph) == ["invalid_value"]


def test_params_stacked_on_an_alias_merge_as_pydantics_fields_do_the_later_one_winning():
    line = Annotated[str, Param(widget=TextInput(), title="Line")]

    def f(separator: Annotated[line, Param(min_length=1, title="Separator")] = ",") -> int:
        return 1

    (field,) = declared(f).describe().inputs
    assert (field.title, field.widget, field.json_schema) == (
        "Separator",
        {"type": "text-input"},
        {"type": "string", "minLength": 1},
    )
    assert codes(compiled_graph(declared(f), constants={"x.separator": ""})) == ["invalid_value"]


def test_a_param_on_a_series_item_is_the_fields_own_and_refuses_a_negative_int_or_a_bool():
    count = Annotated[int, Param(ge=0, title="Count")]
    registry = NodeRegistry()

    @registry.node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str]:
        return text.split()

    @registry.node("length", title="Length", description="For a test.")
    def length(text: str) -> int:
        return len(text)

    @registry.node("total", title="Total", description="For a test.")
    def total(n: Series[count], m: Series[count | None] | None = None) -> int:
        return sum(n)

    for name, may_be_none in (("n", False), ("m", True)):
        field = field_of(total, name)
        item_schema = {"type": "integer", "minimum": 0}
        assert (field.type, field.title, field.is_series, field.json_schema) == (
            "int",
            "Count",
            True,
            {"anyOf": [item_schema, {"type": "null"}]} if may_be_none else item_schema,
        )
        for wrong in ([-1], [True]):
            with pytest.raises(InvalidCallError) as refused:
                Graph.compile(make_graph(total, inputs={name: f"total.{name}"}), registry).run(
                    {"n": [1], name: wrong}
                )
            assert [problem.type for problem in refused.value.problems] == ["invalid_input"]
    graph = Graph.compile(
        make_graph(
            words,
            length,
            total,
            edges=[("words.result", "length.text"), ("length.result", "total.n")],
        ),
        registry,
    )
    assert graph.problems == []
    assert graph.run({"text": "ab c"}).outputs == {"result": 3}


def test_a_param_on_a_tables_column_limits_what_the_node_returns():
    class Row(TypedDict):
        n: Annotated[int, Param(ge=0)]

    @node("rows", title="Rows", description="For a test.")
    def rows(text: str) -> Table[Row]:
        return pl.DataFrame({"n": [-1]})

    assert (
        Graph.compile(
            GraphRecord(nodes={"r": node_record("rows", graph_inputs=("text",))}), registry_of(rows)
        ).problems
        == []
    )
    run = Graph.compile(
        GraphRecord(nodes={"r": node_record("rows", graph_inputs=("text",))}), registry_of(rows)
    ).run({"text": "x"})
    assert (run.status, run.cause.type) == ("failed", "invalid_output")


def test_a_params_choices_limit_a_hosts_own_model_as_they_limit_a_nodes_field():
    class Settings(BaseModel):
        colour: Annotated[str, Param(choices=["red", Choice("blue", "Blue")])] = "red"

    assert Settings(colour="blue").colour == "blue"
    with pytest.raises(ValidationError, match="not_a_choice"):
        Settings(colour="green")


def test_a_value_from_an_edge_is_checked_against_the_inputs_choices_as_against_its_limits():
    registry = NodeRegistry()

    @registry.node("give", title="Give", description="For a test.")
    def give(text: str) -> str:
        return text

    @registry.node("pick", title="Pick", description="For a test.")
    def pick(colour: Annotated[str, Param(choices=["red", "blue"])]) -> str:
        return colour

    graph = Graph.compile(make_graph(give, pick), registry)
    assert graph.run({"text": "red"}).outputs == {"result": "red"}
    run = graph.run({"text": "green"})
    assert (run.status, run.cause.type, run.cause.loc) == (
        "failed",
        "invalid_edge_value",
        ("pick",),
    )


def test_a_params_choices_limit_a_field_of_a_dataclass_input_as_its_limits_do():
    @host_type("Order", title="Order")
    @dataclass(frozen=True)
    class Order:
        colour: Annotated[str, Param(choices=["red", "blue"])]
        n: Annotated[int, Param(ge=0)]

    registry = NodeRegistry()

    @registry.node("take", title="Take", description="For a test.")
    def take(order: Order) -> str:
        return order.colour

    graph = Graph.compile(make_graph(take), registry)
    assert graph.run({"order": {"colour": "red", "n": 1}}).outputs == {"result": "red"}
    for wrong in ({"colour": "green", "n": 1}, {"colour": "red", "n": -1}):
        with pytest.raises(InvalidCallError) as raised:
            graph.run({"order": wrong})
        assert [problem.type for problem in raised.value.problems] == ["invalid_input"]


def test_a_series_of_series_does_not_exist():
    def f(x: Series[Series[int]]) -> int:
        return 1

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.problems[0].type == "nested_series"


def test_a_bare_series_is_many_values_of_anything():
    @node("count", title="Count", description="For a test.")
    def count(xs: Series) -> int:
        return len(xs)

    assert [(i.is_series, i.type) for i in count.describe().inputs] == [(True, "Any")]
    assert (
        compiled_graph(count, constants={"count.xs": [1, "a"]}).run().results["count"]["result"]
        == 2
    )


def test_a_series_that_may_be_none_is_still_many_values():
    @node("f", title="F", description="For a test.")
    def f(xs: Series[int] | None = None) -> str:
        return repr(xs)

    assert [(i.is_series, i.type) for i in f.describe().inputs] == [(True, "int")]


def test_a_limit_its_type_cannot_take_is_refused_naming_the_type_as_written():
    def f(x: Annotated[str | None, Param(ge=0)] = None) -> str:
        return ""

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert "pydantic has no rule for ge on str | None" in str(raised.value)


def test_a_wrongly_typed_limit_is_refused_as_a_bound():
    def f(x: Annotated[int, Param(ge="a")]) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.problems[0].type == "param_limit_value_invalid"


def test_a_hook_reads_a_whole_table_arriving_with_its_columns():
    seen = []

    def outputs(rows: type):
        seen.append(rows)
        return {"result": str}

    @node("read", title="Read", description="For a test.")
    def read(text: str) -> Table[Invoice]:
        return pl.DataFrame({"name": [text], "amount": [1]})

    @node("look", title="Look", description="For a test.", compute_outputs=outputs)
    def look(rows: Table) -> dict:
        return {"result": ",".join(rows.columns)}

    graph = compiled_graph(read, look, outputs={"result": "look.result"})
    row_type = typing.get_args(seen[0])[
        0
    ]  # in Table[R], R is a TypedDict of the columns the input gets
    assert graph.problems == [] and typing.get_type_hints(row_type) == {"name": str, "amount": int}
    assert graph.run({"text": "Bo"}).results["look"]["result"] == "name,amount"


def test_a_bare_series_is_many_values_of_any_type() -> None:
    @node("count", title="Count", description="For a test.")
    def count(items: Series) -> int:  # type: ignore[type-arg]
        return len(items)

    assert count.inputs["items"].is_series and count.inputs["items"].type.hint is Any


def test_a_literal_inputs_choices_are_in_its_record_and_a_widget_draws_them():
    @node("priority", title="Priority", description="For a test.")
    def priority(level: Annotated[Literal["low", "high"], Param(widget=Dropdown())]) -> str:
        return level

    described = priority.describe().inputs[0]
    assert described.choices == (Choice("low", "low"), Choice("high", "high"))
    assert described.widget == {"type": "dropdown"}


def test_a_fields_choices_on_its_plain_type_title_its_values_and_limit_what_it_takes():
    from conductor import Ask, InvalidCallError, NodeRegistry, make_graph

    titled = (Choice("low", "Lav"), Choice("high", "Høj"))

    @node("priority", title="Priority", description="For a test.")
    def priority(level: Annotated[str, Param(choices=titled, widget=Dropdown())]) -> str:
        return level

    @node("size", title="Size", description="For a test.")
    def size(name: Annotated[str, Param(choices=["S", "M"])]) -> str:
        return name

    @node("rate", title="Rate", description="For a test.", asks=True)
    def rate(text: str) -> Annotated[str, Param(choices=titled)]:
        raise Ask("How urgent?")

    assert priority.describe().inputs[0].choices == (Choice("low", "Lav"), Choice("high", "Høj"))
    assert size.describe().inputs[0].choices == (Choice("S", "S"), Choice("M", "M"))
    registry = NodeRegistry()
    registry.add(size, rate)
    with pytest.raises(InvalidCallError):
        Graph.compile(make_graph(size), registry).run({"name": "XL"})
    assert Graph.compile(make_graph(size), registry).run({"name": "M"}).outputs == {"result": "M"}
    asked = Graph.compile(make_graph(rate), registry).run({"text": "x"}).waiting[0].questions[0]
    assert asked.choices == (Choice("low", "Lav"), Choice("high", "Høj"))
    for wrong, code in (
        (("low", "low", "high"), "choices_invalid"),
        ((1, 2), "choices_invalid_not_a_value"),
    ):
        with pytest.raises(NodeDeclarationError) as raised:

            @node("bad", title="Bad", description="For a test.")
            def bad(level: Annotated[str, Param(choices=wrong)]) -> str:
                return level

        assert raised.value.problems[0].type == code
        assert ". Give " in str(raised.value)


@pytest.mark.parametrize(
    ("hint", "plain"),
    [
        (Literal["low", "high"], "str"),
        (Literal["low", None], "str | None"),
        (Literal["low"] | None, "str"),
        (Literal["low"] | Literal[1], "str | int"),
    ],
)
def test_choices_on_a_literal_are_refused_and_the_message_writes_its_plain_type_in_its_place(
    hint: Any, plain: str
) -> None:
    """A ``Literal``'s values are its choices, so choices on it would write each value twice."""
    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(level: Annotated[hint, Param(choices=[Choice("low", "Lav")])]) -> str:  # type: ignore[valid-type]
            return str(level)

    (problem,) = raised.value.problems
    assert (problem.type, problem.loc, problem.ctx) == (
        "choices_on_a_literal",
        ("bad", "level"),
        {"type": plain},
    )
    assert f"Write {plain} in place of the Literal and keep the Param(choices=[...])" in str(
        raised.value
    )


def test_choices_on_a_literal_inside_a_series_name_the_items_plain_type() -> None:
    """The plain type goes where the ``Literal`` is, so the field stays a ``Series``."""
    from conductor import Series

    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(levels: Annotated[Series[Literal["low"]], Param(choices=["low"])]) -> str:
            return ""

    assert raised.value.problems[0].ctx == {"type": "str"}


@pytest.mark.parametrize("hint", [Literal["auto"] | int, Literal["auto"] | str | None])
def test_choices_on_a_literal_beside_another_type_are_refused_and_told_to_go(hint: Any) -> None:
    """Choices there would limit the field to them, so the other type would take nothing more.

    Writing the plain type in the ``Literal``'s place would do the same, so the message says only to remove them,
    and the field without them takes ``"auto"`` and any value of the other type.
    """
    from conductor import make_graph

    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(count: Annotated[hint, Param(choices=[Choice("auto", "Automatic")])]) -> str:  # type: ignore[valid-type]
            return str(count)

    (problem,) = raised.value.problems
    assert (problem.type, problem.loc, problem.ctx) == (
        "choices_on_a_literal_beside_another_type",
        ("bad", "count"),
        {},
    )
    assert str(raised.value).endswith("Remove the Param(choices=).")

    @node("open", title="Open", description="For a test.")
    def open_count(count: Literal["auto"] | int) -> str:
        return str(count)

    graph = Graph.compile(make_graph(open_count), registry_of(open_count))
    assert [graph.run({"count": value}).outputs for value in ("auto", 5)] == [
        {"result": "auto"},
        {"result": "5"},
    ]


def test_the_choices_of_a_literal_are_its_values_and_none_is_one_only_when_it_lists_it() -> None:
    from conductor import make_graph

    registry = NodeRegistry()
    for hint, node_id, choices in (
        (Literal["low", None], "spelt-in", ("low", None)),
        (Literal["low"] | None, "beside", ("low",)),
    ):

        @node(node_id, title="Level", description="For a test.")
        def level(given: hint) -> str:  # type: ignore[valid-type]
            return str(given)

        assert tuple(choice.id for choice in level.describe().inputs[0].choices) == choices
        registry.add(level)
        graph = Graph.compile(make_graph(level), registry)
        assert [graph.run({"given": value}).outputs for value in ("low", None)] == [
            {"result": "low"},
            {"result": "None"},
        ]


def test_an_input_a_hook_adds_may_have_a_default_and_a_hook_may_give_its_own_code():
    from conductor import NodeError

    def added(names: str) -> dict[str, Any]:
        if names == "boom":
            raise NodeError("No such column.", code="no_column", ctx={"names": names})
        return {name: (str, "x") for name in names.split(",")}

    @node("glue", title="Glue", description="For a test.", compute_added_inputs=added)
    def glue(text: str, names: str = "a,b", **columns: str) -> str:
        return text + "".join(columns.values())

    registry = NodeRegistry()
    registry.add(glue)
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"g": {"node": "glue", "inputs": {"text": {"constant": "t"}}}},
                "outputs": {"result": {"node": "g", "output": "result"}},
            }
        ),
        registry,
    )
    assert graph.problems == [] and graph.run().outputs == {"result": "txx"}
    (problem,) = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "g": {
                        "node": "glue",
                        "inputs": {"text": {"constant": "t"}, "names": {"constant": "boom"}},
                    }
                }
            }
        ),
        registry,
    ).problems
    assert (problem.type, problem.ctx["error"], problem.ctx["error_ctx"], problem.msg) == (
        "compute_hook_failed",
        "no_column",
        {"names": "boom"},
        "The node couldn't work out its inputs and outputs: No such column.",
    )


@pytest.mark.parametrize("ctx", [{"reason": "r", "error": "e"}, {}])
def test_a_hooks_own_ctx_is_kept_whole_beside_the_reason_and_the_code(ctx: dict[str, Any]) -> None:
    from conductor import NodeError

    def added(names: str) -> dict[str, Any]:
        raise NodeError("No such column.", code="no_column", ctx=ctx)

    @node("glue", title="Glue", description="For a test.", compute_added_inputs=added)
    def glue(names: str, **columns: str) -> str:
        return names

    (problem,) = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"g": {"node": "glue", "inputs": {"names": {"constant": "a"}}}},
                "outputs": {},
            }
        ),
        registry_of(glue),
    ).problems
    assert problem.ctx == {
        "reason": "No such column.",
        "error": "no_column",
        **({"error_ctx": ctx} if ctx else {}),
    }


@pytest.mark.parametrize(
    ("raised", "code"),
    [(NodeError("No columns."), "node_error"), (ExternalError("No columns."), "external_failed")],
)
def test_a_hooks_error_without_a_code_of_its_own_says_the_code_its_class_gives_every_error_without_one(
    raised: NodeError, code: str
) -> None:
    def added(names: str) -> dict[str, Any]:
        raise raised

    @node("glue", title="Glue", description="For a test.", compute_added_inputs=added)
    def glue(names: str, **columns: str) -> str:
        return names

    (problem,) = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"g": {"node": "glue", "inputs": {"names": {"constant": "a"}}}},
                "outputs": {},
            }
        ),
        registry_of(glue),
    ).problems
    assert (raised.problems[0].type, problem.ctx["error"]) == (code, code)


def test_a_configuration_the_graphs_author_fixes_is_a_value_and_one_left_open_is_the_callers():
    @node("ask-model", title="Ask model", description="For a test.")
    def ask_model(prompt: str, model: str = "small") -> str:
        return f"{model}:{prompt}"

    registry = NodeRegistry()
    registry.add(ask_model)
    fixed = Graph.compile(make_graph(ask_model, constants={"ask-model.model": "big"}), registry)
    assert list(fixed.inputs) == ["prompt"] and fixed.run({"prompt": "hi"}).outputs == {
        "result": "big:hi"
    }
    open_to_caller = Graph.compile(
        make_graph(ask_model, defaults={"ask-model.model": "small"}), registry
    )
    assert list(open_to_caller.inputs) == ["prompt", "model"] and open_to_caller.run(
        {"prompt": "hi"}
    ).outputs == {"result": "small:hi"}
    assert open_to_caller.run({"prompt": "hi", "model": "big"}).outputs == {"result": "big:hi"}
    assert list(Graph.compile(make_graph(ask_model), registry).inputs) == ["prompt"]


@pytest.mark.parametrize(
    ("choices", "limits", "reason"),
    [
        ([True, 2], {}, "a yes or no is not a number"),
        (["1", 2], {}, "Input should be a valid integer"),
        ([-1, 1], {"ge": 0}, "Input should be greater than or equal to 0"),
    ],
    ids=["a bool", "a number as text", "below the field's limit"],
)
def test_a_declared_choice_is_read_strictly_and_checked_against_the_fields_limits(
    choices: list[object], limits: dict[str, int], reason: str
) -> None:
    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(n: Annotated[int, Param(choices=choices, **limits)]) -> int:  # type: ignore[arg-type]
            return n

    assert raised.value.problems[0].type == "choices_invalid_not_a_value"
    assert f"a choice isn't a value of the field's type: {reason}." in str(raised.value)


def test_a_declared_choice_is_read_as_an_answer_is_so_an_enums_value_or_its_member_is_a_choice() -> (
    None
):
    from enum import Enum

    @host_type("color", title="Color")
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    @node("by-value", title="By value", description="For a test.")
    def by_value(color: Annotated[Color, Param(choices=["red"])]) -> str:
        return color.value

    @node("by-member", title="By member", description="For a test.")
    def by_member(color: Annotated[Color, Param(choices=[Color.RED])]) -> str:
        return color.value

    @node("a-day", title="A day", description="For a test.")
    def a_day(day: Annotated[datetime.date, Param(choices=["2024-01-02"])]) -> str:
        return day.isoformat()

    assert [c.id for c in by_value.describe().inputs[0].choices or ()] == ["red"]
    assert [c.id for c in by_member.describe().inputs[0].choices or ()] == ["red"]
    assert a_day.describe().inputs[0].choices == (Choice("2024-01-02", "2024-01-02"),)


@host_type("hue", title="Hue")
class _Hue(Enum):
    RED = "red"
    BLUE = "blue"


class _HueRecord(TypedDict):
    hue: Literal[_Hue.RED]


class _RawRecord(TypedDict):
    raw: Literal[b"a"]


@pytest.mark.parametrize(
    ("hint", "code"),
    [
        (Literal[_Hue.RED], "named_value_invalid"),
        (list[Literal[_Hue.RED, _Hue.BLUE]], "named_value_invalid"),
        (Series[Literal[_Hue.RED]], "named_value_invalid"),
        (_HueRecord, "named_value_invalid"),
        (Table[_RawRecord], "named_value_invalid"),
        (Literal[b"a"], "named_value_invalid"),
        (Literal[float("nan")], "named_value_invalid"),
        (list[Literal[float("nan")]], "named_value_invalid"),
        (Literal[float("inf"), "Infinity"], "named_value_invalid_alike"),  # type: ignore[valid-type]
        (list[Literal[float("inf"), "Infinity"]], "named_value_invalid_alike"),  # type: ignore[valid-type]
    ],
)
def test_a_literal_whose_values_arent_plain_json_is_refused_at_any_depth(
    hint: Any, code: str
) -> None:
    """A ``Literal``'s values are text, numbers, yes or no, or ``None``, each with a JSON form of its own, on the
    field, a list's or a ``Series``'s item, a record's field and a table's column alike."""

    def give() -> str:
        return "x"

    give.__annotations__ = {"return": hint}
    with pytest.raises(NodeDeclarationError) as raised:
        node("give", title="Give", description="For a test.")(give)

    (problem,) = raised.value.problems
    assert problem.type == code


@host_type("pair", title="Pair")
class _Pair(Enum):
    LOW = (0, 1)


class _PairRecord(TypedDict):
    pair: _Pair


@pytest.mark.parametrize("hint", [_Pair, list[_Pair], dict[str, _Pair], Series[_Pair], _PairRecord])
def test_an_enum_that_cant_read_its_members_back_is_refused_at_any_depth_naming_the_enum(
    hint: Any,
) -> None:
    """An enum's members are its choices, so an enum that reads none back from its JSON, like one of tuples, is
    refused at ``@node`` wherever it sits, with a message about the enum and what to write instead."""

    def give() -> Any:
        return _Pair.LOW

    give.__annotations__ = {"return": hint}
    with pytest.raises(NodeDeclarationError) as raised:
        node("give", title="Give", description="For a test.")(give)

    (problem,) = raised.value.problems
    assert problem.type == "named_value_invalid_enum"
    assert "the enum _Pair's member" in problem.msg
    assert "take the values' type, such as str, with Param(choices=[...])" in problem.msg


class _IntHue(IntEnum):
    ONE = 1


class _StrHue(str, Enum):
    A = "a"


class _StrEnumHue(StrEnum):
    B = "b"


for _cls, _id in ((_IntHue, "int-hue"), (_StrHue, "str-hue"), (_StrEnumHue, "strenum-hue")):
    host_type(_id, title=_id)(_cls)


@pytest.mark.parametrize("member", [_IntHue.ONE, _StrHue.A, _StrEnumHue.B])
def test_a_literal_of_a_str_or_int_enums_member_is_declared_and_reads_back(member: Any) -> None:
    """A member that is also a ``str`` or an ``int`` reads back from its JSON as itself, so it may be a ``Literal``'s
    value, as it could before."""

    def give() -> Any:
        return member

    give.__annotations__ = {"return": Literal[member]}
    definition = node("give", title="Give", description="For a test.")(give)
    graph = Graph.compile(make_graph(definition), registry_of(definition))
    run = graph.run()

    read = Run.from_json(run.to_json(), graph).outputs["result"]
    assert read is member
    assert [choice.id for choice in definition.describe().outputs[0].choices or ()] == [
        member.value
    ]


def test_a_literal_of_a_plain_enums_member_is_told_to_take_the_enum_or_fix_its_values() -> None:
    """The advice names the enum as the type only when the enum itself reads its members back."""
    messages = []
    for member in (_Hue.RED, _Pair.LOW):

        def give() -> Any:
            return member

        give.__annotations__ = {"return": Literal[member]}
        with pytest.raises(NodeDeclarationError) as raised:
            node("give", title="Give", description="For a test.")(give)
        messages.append(raised.value.problems[0].msg)

    assert "Take the enum _Hue as the type" in messages[0]
    assert "Give the values of the enum _Pair as text" in messages[1]
    assert " a _" not in "".join(messages)


@host_type("cap-kind", title="Cap kind")
class _CapKind(Enum):
    NONE = float("inf")
    TEXT = "Infinity"


def test_an_enum_with_two_members_of_one_json_form_is_alike() -> None:
    """``inf`` and ``"Infinity"`` are written as one JSON, so a saved run couldn't tell the two members apart."""
    with pytest.raises(NodeDeclarationError) as raised:

        @node("give", title="Give", description="For a test.")
        def give() -> _CapKind:
            return _CapKind.NONE

    (problem,) = raised.value.problems
    assert problem.type == "named_value_invalid_alike"


def test_a_literal_of_an_enums_member_is_told_to_take_the_enum() -> None:
    with pytest.raises(NodeDeclarationError) as raised:

        @node("give", title="Give", description="For a test.")
        def give() -> Literal[_Hue.RED]:
            return _Hue.RED

    assert "Take the enum _Hue as the type" in str(raised.value)


@pytest.mark.parametrize(
    "hint",
    [Literal[1, True], Literal[1, 1.0], Literal["1", 1], Literal[None, "null"], Literal[0, False]],
)
def test_a_literal_of_plain_values_whose_json_differs_is_declared_and_reads_back(hint: Any) -> None:
    """``1`` and ``True`` are two values, written ``1`` and ``true``, so each comes back as itself."""
    value = typing.get_args(hint)[1]

    def give() -> Any:
        return value

    give.__annotations__ = {"return": hint}
    definition = node("give", title="Give", description="For a test.")(give)
    graph = Graph.compile(make_graph(definition), registry_of(definition))
    run = graph.run()

    read = Run.from_json(run.to_json(), graph).outputs["result"]
    assert (read, type(read)) == (value, type(value))


def test_a_default_or_a_choice_that_is_not_finite_is_refused_where_the_node_is_declared() -> None:
    """A graph record would write such a default, and a choice is given from its JSON form, as an answer is: no one
    could give either, since a number that enters is finite."""

    def capped(limit: float = float("inf")) -> str:
        return str(limit)

    def capped_in_a_list(limits: list[float] = [1.0, float("nan")]) -> str:  # noqa: B006
        return str(limits)

    def picked(limit: Annotated[float, Param(choices=[1.0, float("inf")])]) -> str:
        return str(limit)

    def picked_on_output() -> Annotated[float, Param(choices=[float("nan")])]:
        return 1.0

    refused = []
    for wrong in (capped, capped_in_a_list, picked, picked_on_output):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        refused.append(raised.value.problems[0].type)
    assert refused == [
        "default_invalid",
        "default_invalid",
        "choices_invalid_not_a_value",
        "choices_invalid_not_a_value",
    ]


@pytest.mark.parametrize(
    "given",
    [
        {"policy": Policy(concurrency=4)},
        {"compute_outputs": lambda: {"x": int}},
        {"compute_added_inputs": lambda: {"x": int}},
    ],
)
def test_a_subgraph_takes_no_policy_or_hook_as_its_inner_nodes_have_their_own(
    given: dict[str, Any],
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        subgraph("x", title="X", description="d", **given)


def test_a_limit_pydantic_supports_on_a_type_is_kept_and_one_it_does_not_is_refused_in_a_model_too():
    import dataclasses
    import datetime
    from decimal import Decimal

    def limited(
        price: Annotated[Decimal, Param(ge=0)],
        wait: Annotated[datetime.timedelta, Param(ge=datetime.timedelta(0))],
        tags: Annotated[list[str], Param(min_length=1)],
    ) -> int:
        return 0

    right = {"x.price": Decimal("1"), "x.wait": datetime.timedelta(1), "x.tags": ["a"]}
    assert compiled_graph(declared(limited), constants=right).run().outputs == {"result": 0}
    for wrong, reason in (
        ({"x.price": Decimal("-1")}, "greater than or equal"),
        ({"x.tags": []}, "at least 1"),
    ):
        (problem,) = compiled_graph(declared(limited), constants=right | wrong).problems
        assert problem.type == "invalid_value" and reason in problem.msg

    with pytest.raises(NodeDeclarationError) as raised:

        @host_type("Order", title="Order")
        @dataclasses.dataclass(frozen=True)
        class Order:
            code: Annotated[str, Param(ge=0)]

    assert raised.value.problems[0].type == "param_limit_not_supported"


def test_a_param_whose_choices_are_not_values_of_its_type_is_refused_at_node_in_a_model_a_node_takes_and_on_a_table():
    with pytest.raises(NodeDeclarationError) as raised:

        @host_type("Order", title="Order")
        @dataclasses.dataclass(frozen=True)
        class Order:
            n: Annotated[int, Param(choices=["one", "two"])]

    assert raised.value.problems[0].type == "choices_invalid_not_a_value"

    def on_table(x: Annotated[Table, Param(choices=["a"])]) -> int:
        return 0

    with pytest.raises(NodeDeclarationError) as raised:
        declared(on_table)
    assert raised.value.problems[0].type == "choices_invalid_not_a_value"


@pytest.mark.parametrize("default", [b"\xff\xfe", object()], ids=["bytes-not-utf-8", "any-object"])
def test_a_default_with_no_json_form_is_refused_where_the_node_is_declared(default: Any) -> None:
    with pytest.raises(NodeDeclarationError) as raised:

        @node("b", title="B", description="Takes a value.")
        def b(data: Any = default) -> int:
            return 1

    assert raised.value.problems[0].type == "default_invalid"


def test_a_default_is_described_as_its_json_and_a_graph_input_offers_the_records_default_as_its_json():
    class Row(TypedDict):
        n: int

    @node("t", title="T", description="Counts rows.")
    def t(
        rows: Table[Row] = pl.DataFrame({"n": [1, 2]}),
        when: datetime.date = datetime.date(2026, 1, 2),
    ) -> int:
        return len(rows)

    described = {field.name: field.default for field in t.describe().inputs}
    record = make_graph(t, defaults={"t.rows": [{"n": 3}], "t.when": datetime.date(2026, 1, 3)})
    offered = {
        name: given.default for name, given in Graph.compile(record, registry_of(t)).inputs.items()
    }

    assert described == {
        "rows": {"columns": [{"name": "n", "type": "int", "values": [1, 2]}]},
        "when": "2026-01-02",
    }
    assert offered == {
        "rows": {"columns": [{"name": "n", "type": "int", "values": [3]}]},
        "when": "2026-01-03",
    }


def test_a_reason_a_host_check_gave_is_quoted_with_one_period() -> None:
    """The text that replaces a host validator's message has no final period, so a message quoting it reads right."""
    from pydantic import AfterValidator

    def refuse(value: str) -> str:
        raise ValueError(f"no {value}")

    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(pick: Annotated[str, AfterValidator(refuse), Param(choices=["x"])]) -> str:
            return pick

    assert raised.value.problems[0].type == "choices_invalid_not_a_value"
    assert "did not pass its type's own check. Give each choice" in str(raised.value)
