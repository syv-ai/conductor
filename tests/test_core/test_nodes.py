"""Tests for ``@node``: what a node author declares, and what a host sees of it.

A node is a function under ``@node``. Its signature says the node's inputs and outputs. The tests cover those, along
with types, ``Series`` and ``Table``, ``Param`` and widgets, versions, and the hooks that work out inputs and outputs
when a graph is compiled.
"""

import dataclasses
import datetime
import json
import logging
import math
import re
import typing
import warnings
from collections.abc import Callable, Mapping, Set
from dataclasses import FrozenInstanceError, dataclass
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from fractions import Fraction
from types import MappingProxyType
from typing import Annotated, Any, Literal, NotRequired, TypedDict, TypeVar

import polars as pl
import pytest
from annotated_types import (
    Ge,
    Interval,
    IsDigit,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotFinite,
    Le,
    Len,
    LowerCase,
    MaxLen,
    MinLen,
    Not,
    Timezone,
    Unit,
)
from conductor import (
    JSON_CONFIG,
    SKIPPED,
    Ask,
    Choice,
    Deprecation,
    ExternalError,
    Graph,
    GraphInputRecord,
    GraphRecord,
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
from pydantic import (
    AfterValidator,
    AnyUrl,
    Base64Bytes,
    Base64Str,
    Base64UrlBytes,
    Base64UrlStr,
    BaseModel,
    ByteSize,
    ConfigDict,
    Discriminator,
    Field,
    ImportString,
    Json,
    JsonValue,
    PlainValidator,
    Secret,
    SecretBytes,
    SecretStr,
    StringConstraints,
    ValidationError,
    field_serializer,
    with_config,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic_core import core_schema

from tests.test_core.nodes import input_entry, registry_of


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
        name: input_entry({"constant": value}) for name, value in (constants or {}).items()
    }
    inputs |= {name: input_entry({"graph_input": name}) for name in graph_inputs}
    inputs |= {
        name: input_entry({"sources": (Source(*source),)})
        for name, source in (sources or {}).items()
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


def port_of(definition, name, side="inputs"):
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
        # an id that isn't a name can't be a place, so the problem has no loc
        assert _faults(lambda: node(wrong, title="Wrong", description="For a test."), ()) == [
            ("not_a_name", ["id"])
        ]

    def amount() -> int:
        return 1

    assert node("Beløb i kr", title="Amount", description="For a test.")(amount).id == "Beløb i kr"


def test_node_given_a_record_in_place_of_a_function_raises_pythons_own_error_that_it_is_not_callable():
    with pytest.raises(TypeError, match="is not a callable object"):
        node("enrich", title="Enrich", description="For a test.")(
            GraphRecord.model_validate({"nodes": {}})
        )  # type: ignore[arg-type]


class _Secret:
    """An argument whose repr must never reach an error."""

    def __repr__(self) -> str:
        return "SECRET-REPR"


def _whisper(text: str) -> str:
    return text.lower()


def _faults(
    declaration: Callable[[], object], loc: tuple[str, ...] = ("x",)
) -> list[tuple[str, list[str | int]]]:
    """The code and loc of every fault in the ``invalid_declaration_argument`` a declaration raises at ``loc``."""
    with pytest.raises(NodeDeclarationError) as raised:
        declaration()
    (problem,) = raised.value.problems
    assert (problem.type, problem.loc) == ("invalid_declaration_argument", loc)
    return [(e["type"], e["loc"]) for e in problem.ctx["errors"]]


@pytest.mark.parametrize(
    ("arguments", "faults"),
    [
        # tags="abc" was taken as ('a', 'b', 'c')
        ({"tags": "abc"}, [("sequence_str", ["tags"])]),
        ({"tags": ["a", 1]}, [("string_type", ["tags", 1])]),
        # a date, a tuple or an object in extra was taken, and failed only at registry.describe()
        (
            {"extra": {"when": datetime.date(2026, 1, 1)}},
            [("invalid-json-value", ["extra", "when"])],
        ),
        ({"extra": {"pair": (1, 2)}}, [("invalid-json-value", ["extra", "pair"])]),
        ({"extra": {1: "a"}}, [("string_type", ["extra", "[key]"])]),
        ({"extra": {"cap": math.inf}}, [("finite_number", ["extra"])]),
        ({"extra": {"cap": [1, {"x": math.nan}]}}, [("finite_number", ["extra"])]),
        ({"version": 0}, [("greater_than_equal", ["version"])]),
        ({"version": -3}, [("greater_than_equal", ["version"])]),
        ({"version": True}, [("bool_is_not_a_number", ["version"])]),
        ({"version": "2"}, [("int_type", ["version"])]),
        ({"version": 1.5}, [("int_type", ["version"])]),
        ({"title": 5}, [("string_type", ["title"])]),
        ({"description": None}, [("string_type", ["description"])]),
        ({"category": None}, [("string_type", ["category"])]),
        ({"docs": 5}, [("string_type", ["docs"])]),
        # a string deprecation was taken, and failed only at registry.describe()
        ({"deprecation": "soon"}, [("is_instance_of", ["deprecation"])]),
        ({"replaces": "x"}, [("not_a_node", ["replaces"])]),
        ({"upgrade": {}}, [("dataclass_exact_type", ["upgrade"])]),
        # an Upgrade's fields were taken as given, and failed only at registry.upgrade
        ({"upgrade": Upgrade(renamed_inputs=5)}, [("dict_type", ["upgrade", "renamed_inputs"])]),  # type: ignore[arg-type]
        (
            {"upgrade": Upgrade(renamed_outputs={"old": 1})},  # type: ignore[dict-item]
            [("string_type", ["upgrade", "renamed_outputs", "old"])],
        ),
        ({"upgrade": Upgrade(rewrite=5)}, [("callable_type", ["upgrade", "rewrite"])]),  # type: ignore[arg-type]
        # a key that isn't text is never quoted, inside a dataclass's field too
        (
            {"upgrade": Upgrade(renamed_inputs={1: "x"})},  # type: ignore[dict-item]
            [("string_type", ["upgrade", "renamed_inputs", "[key]"])],
        ),
        ({"title": ""}, [("blank_text", ["title"])]),
        ({"title": "   "}, [("blank_text", ["title"])]),
        ({"description": ""}, [("blank_text", ["description"])]),
        ({"description": "   "}, [("blank_text", ["description"])]),
        ({"description": "\t\n"}, [("blank_text", ["description"])]),
        (
            {"tags": "ab", "version": 0},
            [("greater_than_equal", ["version"]), ("sequence_str", ["tags"])],
        ),
    ],
)
def test_an_argument_of_the_wrong_type_is_refused_where_a_node_or_a_subgraph_is_declared(
    arguments, faults
):
    given = {"title": "X", "description": "For a test.", **arguments}
    assert _faults(lambda: node("x", **given)) == faults
    assert _faults(lambda: subgraph("x", **given)) == faults


@pytest.mark.parametrize(
    ("arguments", "faults"),
    [
        ({"title": "  "}, [("blank_text", ["title"])]),
        ({"description": ""}, [("blank_text", ["description"])]),
        (
            {"title": "\t", "description": " "},
            [("blank_text", ["title"]), ("blank_text", ["description"])],
        ),
    ],
)
def test_a_blank_title_or_description_of_a_param_is_refused_as_a_nodes_is(arguments, faults):
    assert _faults(lambda: Param(**arguments), ()) == faults


@pytest.mark.parametrize(
    ("arguments", "faults"),
    [
        # widget=5 was an AttributeError when the node was declared
        ({"widget": 5}, [("model_type", ["widget"])]),
        # these were taken, and meant nothing
        ({"constant_only": "yes"}, [("bool_type", ["constant_only"])]),
        ({"min_length": True}, [("bool_is_not_a_number", ["min_length"])]),
        # one fault for a limit of none of its kinds, not one per kind tried
        ({"pattern": 5}, [("not_a_limit", ["pattern"])]),
        ({"ge": "a"}, [("not_a_limit", ["ge"])]),
        ({"lt": True}, [("not_a_limit", ["lt"])]),
        ({"multiple_of": "2"}, [("not_a_limit", ["multiple_of"])]),
        (
            {"widget": "x", "max_length": "3"},
            [("model_type", ["widget"]), ("int_type", ["max_length"])],
        ),
    ],
)
def test_every_field_of_a_param_is_held_to_its_type_where_it_is_written(arguments, faults):
    """A ``Param`` is held to its own annotations, strictly, as ``@node``'s arguments are. Its choices are the one
    field left to the type it limits, which reads them (``choices_invalid``)."""
    assert _faults(lambda: Param(**arguments), ()) == faults


def test_a_params_pattern_may_be_compiled():
    @node("code", title="Code", description="For a test.")
    def code(text: Annotated[str, Param(pattern=re.compile("^[a-z]+$"))]) -> str:
        return text

    graph = compiled_graph(code)
    assert graph.run({"text": "abc"}).results["code"]["result"] == "abc"
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs({"text": "ABC"})
    assert refused.value.problems[0].ctx["errors"][0]["type"] == "string_pattern_mismatch"


@pytest.mark.parametrize("declare", [node, subgraph])
def test_every_fault_in_a_declaration_is_found_in_one_pass(declare: Callable[..., object]):
    given: dict[str, Any] = {"title": " ", "description": "For a test.", "tags": "ab"}
    assert _faults(lambda: declare("a.b", **given), ()) == [
        ("not_a_name", ["id"]),
        ("blank_text", ["title"]),
        ("sequence_str", ["tags"]),
    ]
    # the reason, which the message quotes, names every fault after its argument
    with pytest.raises(NodeDeclarationError) as raised:
        declare("a.b", **given)
    reason = raised.value.problems[0].ctx["reason"]
    assert [part.split(":")[0] for part in reason.split("; ")] == ["id", "title", "tags"]
    assert reason in str(raised.value)


def test_two_versions_of_a_node_with_a_blank_title_read_apart_by_their_version():
    texts = []
    for version in (1, 2):
        with pytest.raises(NodeDeclarationError) as raised:
            node("x", title=" ", description="For a test.", version=version)
        assert raised.value.problems[0].loc == ("x",)
        texts.append(str(raised.value).split(":")[0])
    assert texts == ["x (version 1)", "x (version 2)"]
    # a version that is refused itself can't name the fault
    with pytest.raises(NodeDeclarationError) as raised:
        node("x", title=" ", description="For a test.", version=0)
    assert str(raised.value).startswith("x: ")


@pytest.mark.parametrize(
    ("arguments", "faults"),
    [
        # policy={...} was an AttributeError
        ({"policy": {"retries": 1}}, [("is_instance_of", ["policy"])]),
        ({"asks": "yes"}, [("bool_type", ["asks"])]),
        ({"asks": 1}, [("bool_type", ["asks"])]),
        ({"compute_outputs": 3}, [("callable_type", ["compute_outputs"])]),
    ],
)
def test_an_argument_only_node_takes_is_held_to_its_type_too(arguments, faults):
    assert _faults(lambda: node("x", title="X", description="For a test.", **arguments)) == faults


def test_a_registry_holds_its_defaults_to_the_types_a_node_does():
    assert _faults(
        lambda: NodeRegistry(tags="ab", extra={"when": datetime.date(2026, 1, 1)}), ()
    ) == [  # type: ignore[arg-type]
        ("sequence_str", ["tags"]),
        ("invalid-json-value", ["extra", "when"]),
    ]


def test_a_refused_argument_is_never_quoted():
    secret = _Secret()
    for declaration in (
        lambda: node("x", title=secret, description="For a test."),  # type: ignore[arg-type]
        lambda: node("x", title="X", description="For a test.", extra={"k": secret}),  # type: ignore[dict-item]
        lambda: node("x", title="X", description="For a test.", extra={secret: 1}),  # type: ignore[dict-item]
        lambda: node(secret, title="X", description="For a test."),  # type: ignore[arg-type]
        lambda: node(
            "x",
            title="X",
            description="For a test.",
            extra={"a": {secret: datetime.date(2026, 1, 1)}},
        ),  # type: ignore[dict-item]
        lambda: node("x", title="X", description="For a test.", extra={"a": [{secret: 1}]}),  # type: ignore[dict-item]
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            declaration()
        assert "SECRET" not in str(raised.value)
        assert "SECRET" not in json.dumps(raised.value.problems[0].model_dump(mode="json"))


def test_lists_tuples_and_any_mapping_of_json_are_taken():
    declared_node = node(
        "x",
        title="X",
        description="For a test.",
        tags=["a", "b"],
        extra=MappingProxyType(
            {"icon": "star", "order": [1, 2], "beta": None, "nested": {"a": 1.5}}
        ),
    )(_whisper)
    assert declared_node.tags == ("a", "b")
    assert dict(declared_node.extra) == {
        "icon": "star",
        "order": [1, 2],
        "beta": None,
        "nested": {"a": 1.5},
    }
    assert node("y", title="Y", description="For a test.", tags=("a",))(_whisper).tags == ("a",)


def test_a_node_id_that_isnt_text_is_an_argument_of_the_wrong_type_placed_at_no_node():
    assert _faults(lambda: node(5, title="Five", description="For a test."), ()) == [  # type: ignore[arg-type]
        ("string_type", ["id"])
    ]


def test_an_argument_of_the_wrong_type_names_the_node():
    with pytest.raises(NodeDeclarationError) as raised:
        node("tagged_twice", title="X", description="For a test.", tags="x")  # type: ignore[arg-type]
    assert str(raised.value).startswith(
        "tagged_twice (version 1): @node can't take its arguments: tags:"
    )


@pytest.mark.parametrize(
    ("extra", "faults"),
    [
        # a key that isn't text, nested in extra, was quoted, and an int key read as an index
        ({"a": {_Secret(): datetime.date(2026, 1, 1)}}, [("string_type", ["extra", "a", "[key]"])]),
        ({"a": {7: 1}}, [("string_type", ["extra", "a", "[key]"])]),
        ({"a": [{7: (1,)}]}, [("string_type", ["extra", "a", 0, "[key]"])]),
        # JsonValue's labels, "dict" and "list", are left out of the loc
        ({"a": [1, (2,)]}, [("invalid-json-value", ["extra", "a", 1])]),
        (
            {"dict": {"b": [{"c": datetime.date(2026, 1, 1)}]}},
            [("invalid-json-value", ["extra", "dict", "b", 0, "c"])],
        ),
    ],
)
def test_a_fault_deep_in_extra_names_only_keys_and_indexes_and_quotes_no_key(extra, faults):
    assert _faults(lambda: node("x", title="X", description="For a test.", extra=extra)) == faults


def test_a_node_id_in_a_record_with_a_tilde_is_refused_by_the_record_validator():
    registry = NodeRegistry()

    @registry.node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": {"a~b": {"node": "upper"}}}), registry)
    assert [(p.type, p.loc, p.ctx["errors"][0]["type"]) for p in refused.value.problems] == [
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
    (problem,) = raised.value.problems
    assert (problem.type, [(e["type"], e["loc"]) for e in problem.ctx["errors"]]) == (
        "default_invalid",
        [("int_type", [])],
    )


def test_a_default_outside_its_own_bounds_fails_where_it_is_written():
    def f(x: Annotated[int, Param(ge=0)] = -1) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    (problem,) = raised.value.problems
    assert (problem.type, problem.ctx["errors"]) == (
        "default_invalid",
        [
            {
                "type": "greater_than_equal",
                "loc": [],
                "msg": "Input should be greater than or equal to 0",
                "ctx": {"ge": 0},
            }
        ],
    )


def test_a_series_default_is_checked_against_the_limits_of_each_item():
    def f(xs: Annotated[Series[int], Param(ge=0)] = [1, -1]) -> int:  # noqa: B006
        return sum(xs)

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    (problem,) = raised.value.problems
    assert (problem.type, [(e["type"], e["loc"]) for e in problem.ctx["errors"]]) == (
        "default_invalid",
        [("greater_than_equal", [1])],
    )


def test_an_unknown_param_argument_is_pythons_own_type_error():
    with pytest.raises(TypeError, match="'titel'"):
        Param(titel="Times")  # type: ignore[call-arg]


class _NamesNothing(TypedDict):
    x: "Missing"  # type: ignore[name-defined]  # noqa: F821


@pytest.mark.parametrize(
    ("hint", "defaults", "code"),
    [
        (Annotated[str, Param(ge=1)], (), "param_limit_not_supported"),
        (Annotated[Table, Param(ge=1)], (), "param_limit_not_supported"),
        (Annotated[str | None, Param(ge=0)], (None,), "param_limit_not_supported"),
        (Annotated[Series[str], Param(ge=0)], (), "param_limit_not_supported"),
        # Pydantic builds ``Annotated[int, MaxLen(3)]`` and then refuses every value, so the graph would compile
        # and never run.
        (Annotated[int, MaxLen(3)], (), "param_limit_not_supported"),
        (list[Annotated[int, Len(1, 3)]], (), "param_limit_not_supported"),
        (Annotated[str, Interval(ge=0)] | None, (None,), "param_limit_not_supported"),
        # A validator wraps the type's schema in a function, and a limit beside it was let through, so every value
        # failed at run time.
        (Annotated[str, AfterValidator(str.strip), Param(ge=0)], (), "param_limit_not_supported"),
        (Annotated[int, AfterValidator(abs), Param(max_length=3)], (), "param_limit_not_supported"),
        (Annotated[int, StringConstraints(pattern="^1")], (), "param_limit_not_supported"),
        (Annotated[str, PlainValidator(str), Param(ge=0)], (), "param_limit_not_supported"),
        (
            Annotated[int, StringConstraints(pattern="^1", strip_whitespace=True)],
            (),
            "param_limit_not_supported",
        ),
        # Pydantic builds a ``Timezone`` and never checks it, so a naive datetime would pass.
        (Annotated[datetime.datetime, Timezone(...)], (), "timezone_in_hint"),
        (list[Annotated[datetime.datetime, Timezone("UTC")]], (), "timezone_in_hint"),
        # ``Unit`` and a named predicate written as metadata are ignored by pydantic.
        (Annotated[float, Unit("m")], (), "unit_in_hint"),
        (Annotated[str, LowerCase], (), "predicate_alias_in_hint"),
        (list[Annotated[str, IsDigit]], (), "predicate_alias_in_hint"),
        # No number that enters a graph is ``nan`` or infinite.
        (IsNan[float], (), "predicate_refuses_every_number"),
        (list[IsInfinite[float]], (), "predicate_refuses_every_number"),
        (IsNotFinite[float], (), "predicate_refuses_every_number"),
        (Annotated[float, Not(math.isfinite)], (), "predicate_refuses_every_number"),
        (_NamesNothing, (), "type_not_supported"),
        (int, (None,), "default_none_not_taken"),
        (Series[Series[int]], (), "nested_series"),
    ],
    ids=[
        "a-bound-on-text",
        "a-bound-on-a-table",
        "a-bound-on-text-or-none",
        "a-bound-on-a-series-of-text",
        "max-len-on-a-number",
        "len-on-a-lists-number",
        "interval-on-text-or-none",
        "a-bound-beside-an-after-validator",
        "a-length-beside-an-after-validator",
        "a-pattern-on-a-number",
        "a-bound-beside-a-plain-validator",
        "a-pattern-that-strips-on-a-number",
        "a-timezone",
        "a-timezone-in-a-list",
        "a-unit",
        "a-predicate-alias",
        "a-predicate-alias-in-a-list",
        "is-nan",
        "is-infinite-in-a-list",
        "is-not-finite",
        "not-finite",
        "a-typed-dict-field-naming-nothing",
        "a-none-default-on-an-int",
        "a-series-of-series",
    ],
)
def test_an_input_type_or_default_conductor_cant_take_is_refused_at_node_with_its_code(
    hint: Any, defaults: tuple[Any, ...], code: str
) -> None:
    def takes(x: Any) -> int:
        return 0

    takes.__annotations__["x"] = hint
    takes.__defaults__ = defaults or None
    with pytest.raises(NodeDeclarationError) as raised:
        declared(takes)
    assert [problem.type for problem in raised.value.problems] == [code]


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


def test_a_param_hands_pydantic_its_limits_as_annotated_types_objects_and_the_objects_written_directly_read_the_same():
    """``Param(ge=0)`` is handed to pydantic as ``Ge(0)``, so the two spellings are one type."""
    param = Param(le=8, ge=0, pattern="^a")
    assert (param.ge, param.limits) == (0, (Ge(0), Le(8), StringConstraints(pattern="^a")))

    def by_keyword(
        x: Annotated[int, Param(ge=0, le=8)],
        y: Annotated[str, Param(pattern="^a")],
        z: Annotated[str, Param(max_length=3)] = "a",
    ) -> int:
        return x

    def by_object(
        x: Annotated[int, Interval(ge=0, le=8)],
        y: Annotated[str, StringConstraints(pattern="^a")],
        z: Annotated[str, MaxLen(3)] = "a",
    ) -> int:
        return x

    def schemas(definition):
        return [port.json_schema for port in definition.describe().inputs]

    assert schemas(declared(by_object)) == schemas(declared(by_keyword))
    assert schemas(declared(by_keyword)) == [
        {"type": "integer", "minimum": 0, "maximum": 8},
        {"type": "string", "pattern": "^a"},
        {"type": "string", "maxLength": 3},
    ]


@host_type("Money", title="Money")
class Money:
    """Cents read from text, compared as a number."""

    def __init__(self, cents: int) -> None:
        self.cents = cents

    def __ge__(self, other: int) -> bool:
        return self.cents >= other

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_after_validator_function(
            lambda text: cls(int(text)), core_schema.str_schema()
        )


def test_a_limit_pydantic_checks_is_taken_on_a_host_type_and_on_pydantics_own_types():
    """A limit is taken where the schema pydantic built shows it: as its own rule (a fraction, an address) or as its
    comparison over a host's own function (``Money``, ``ByteSize``)."""

    def limited(
        money: Annotated[Money, Param(ge=0)],
        part: Annotated[Fraction, Param(ge=0)],
        address: Annotated[AnyUrl, Param(max_length=20)],
        size: Annotated[ByteSize, Param(ge=1)],
    ) -> int:
        return 0

    graph = compiled_graph(declared(limited))
    right = {"money": "5", "part": "1/2", "address": "http://a.b", "size": "1KB"}
    assert graph.run(right).outputs == {"result": 0}
    for name, wrong in (
        ("money", "-5"),
        ("part", "-1/2"),
        ("address", "http://" + "a" * 40 + ".com"),
        ("size", 0),
    ):
        with pytest.raises(InvalidCallError) as refused:
            graph.run(right | {name: wrong})
        assert refused.value.problems[0].loc == ("inputs", name)


def test_an_output_may_declare_a_number_that_is_not_finite_since_a_runs_own_values_may_be_one():
    @node("not-a-number", title="Not a number", description="For a test.")
    def not_a_number() -> IsNan[float]:
        return math.nan

    assert math.isnan(compiled_graph(not_a_number).run().outputs["result"])


def test_a_predicate_is_pydantics_and_checks_each_value():
    @node("lower-only", title="Lower only", description="For a test.")
    def lower_only(x: LowerCase[str], y: IsFinite[float]) -> str:
        return x

    graph = compiled_graph(lower_only)
    assert graph.run({"x": "ab", "y": 1.0}).outputs == {"result": "ab"}
    with pytest.raises(InvalidCallError):
        graph.run({"x": "AB", "y": 1.0})


def test_a_length_in_a_string_constraints_is_a_param_length_so_the_last_one_written_wins():
    """``constr(...)`` writes a ``StringConstraints``; its length was kept apart from a ``Param``'s, and pydantic
    enforced whichever it was handed last."""

    def constrained_then_param(
        x: Annotated[str, StringConstraints(min_length=1), Param(min_length=3)],
    ) -> str:
        return x

    def param_then_constrained(
        x: Annotated[str, Param(min_length=3), StringConstraints(min_length=1)],
    ) -> str:
        return x

    assert port_of(declared(constrained_then_param), "x").json_schema["minLength"] == 3
    assert port_of(declared(param_then_constrained), "x").json_schema["minLength"] == 1
    assert compiled_graph(declared(param_then_constrained)).run({"x": "ab"}).outputs == {
        "result": "ab"
    }
    with pytest.raises(InvalidCallError):
        compiled_graph(declared(constrained_then_param)).run({"x": "ab"})


def test_two_limits_of_one_name_keep_the_last():
    def loosened(x: Annotated[int, Param(ge=5), Ge(0)]) -> int:
        return x

    assert compiled_graph(declared(loosened)).run({"x": 3}).outputs == {"result": 3}


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


class _Part(TypedDict):
    a: int
    b: NotRequired[int]


@host_type("keyed-wallet", title="Keyed wallet")
@dataclass(frozen=True)
class _KeyedWallet:
    amounts: dict[str, int]


@host_type("part-box", title="Part box")
@dataclass(frozen=True)
class _PartBox:
    part: _Part


type _Counts = dict[str, int]


@host_type("open-note", title="Open note")
class _OpenNote(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    text: str


@host_type("cat-pet", title="Cat pet")
class _Cat(BaseModel, frozen=True):
    kind: Literal["cat"]
    meow: str


@host_type("dog-pet", title="Dog pet")
class _Dog(BaseModel, frozen=True):
    kind: Literal["dog"]
    bark: int


type _Nest = list[_Nest] | int
type _Deep = list[_Deep]


@host_type("cell-colour", title="Cell colour")
class _Colour(Enum):
    RED = "red"


@host_type("cell-money", title="Cell money")
class _Money(BaseModel, frozen=True):
    amount: float


@with_config(ConfigDict(extra="allow"))
class _OpenRow(TypedDict):
    a: int


@host_type("open-point", title="Open point")
@pydantic_dataclass(frozen=True, config=ConfigDict(extra="allow"))
class _OpenPoint:
    x: float


@pytest.mark.parametrize(
    ("cell", "named", "reason"),
    [
        (dict[str, int], "dict[str, int]", "a table has no column type for it"),
        (dict, "dict", "a table has no column type for it"),
        (list[dict[str, int]] | None, "dict[str, int]", "a table has no column type for it"),
        (_Counts, "_Counts", "a table has no column type for it"),
        (_Counts | None, "_Counts", "a table has no column type for it"),
        (JsonValue, "JsonValue", "a value of a union may be of either type"),
        (_Part, "_Part", "a value may leave out its key 'b'"),
        (list[_Part], "_Part", "a value may leave out its key 'b'"),
        (_KeyedWallet, "dict[str, int]", "a table has no column type for it"),
        (_PartBox, "_Part", "a value may leave out its key 'b'"),
        (_OpenNote, "_OpenNote", "a value may have keys it doesn't declare"),
        (_OpenRow, "_OpenRow", "a value may have keys it doesn't declare"),
        (_OpenPoint, "_OpenPoint", "a value may have keys it doesn't declare"),
        (
            Annotated[_Cat | _Dog, Discriminator("kind")],
            "cat-pet | dog-pet",
            "a value of a union may be of either type",
        ),
        (_Money | str, "cell-money | str", "a value of a union may be of either type"),
        (int | str, "int | str", "a value of a union may be of either type"),
        (_Nest, "_Nest", "a value of a union may be of either type"),
        (tuple[int, str], "tuple[int, str]", "its items differ in type"),
        (tuple[_Money, _Colour], "tuple[cell-money, cell-colour]", "its items differ in type"),
        (Literal["a", 1], "str | int", "its values are of different classes"),
    ],
    ids=[
        "a dict",
        "a bare dict",
        "a dict in a list",
        "a dict through an alias",
        "an alias that may be None",
        "JsonValue",
        "a NotRequired field",
        "a NotRequired field in a list",
        "a dict in a dataclass",
        "a NotRequired field in a dataclass",
        "a model that takes extra keys",
        "a TypedDict that takes extra keys",
        "a pydantic dataclass that takes extra keys",
        "a discriminated union of models",
        "a union of a model and a text",
        "a union of numbers and texts",
        "an alias of a union",
        "a tuple of an int and a text",
        "a tuple of a model and an enum",
        "a Literal of two classes",
    ],
)
def test_a_table_cell_of_a_type_a_table_does_not_keep_is_refused_where_the_node_is_declared(
    cell: Any, named: str, reason: str
) -> None:
    """Polars holds a few of Python's own classes, and lists and structs of them. A cell of any other type would
    come back changed, so a table whose row declares one is refused. A Series of the same rows is a list, and is
    taken."""
    row: Any = TypedDict("Row", {"name": str, "cell": cell})  # type: ignore[operator]

    def table_of_rows(text: str) -> Table[row]:  # type: ignore[valid-type]
        return pl.DataFrame()

    def series_of_rows(text: str) -> Series[row]:  # type: ignore[valid-type]
        return []

    with pytest.raises(NodeDeclarationError) as raised:
        declared(table_of_rows)

    assert [(one.type, one.ctx["cell"], one.ctx["reason"]) for one in raised.value.problems] == [
        ("table_cell_not_kept", named, reason)
    ]
    declared(series_of_rows)


class _Branch(TypedDict):
    name: str
    branches: list["_Branch"]


class _Tree(TypedDict):
    root: _Branch


@pytest.mark.parametrize(
    "cell",
    [
        int,
        str | None,
        datetime.date,
        Decimal,
        bytes,
        Any,
        Literal["a", "b"],
        _Colour,
        list[int],
        tuple[int, ...],
        tuple[int, int],
        _Money,
        list[_Money | None],
        _Tree,
        _Deep,
        Annotated[int, Param(ge=0)],
    ],
    ids=[
        "an int",
        "a text that may be None",
        "a date",
        "a decimal",
        "bytes",
        "Any",
        "a Literal of texts",
        "an enum",
        "a list",
        "a tuple of one item type",
        "a tuple of two alike items",
        "a model",
        "a list of models that may be None",
        "a TypedDict that holds itself",
        "an alias that holds itself",
        "a limit",
    ],
)
def test_a_table_cell_of_a_type_a_table_keeps_is_declared(cell: Any) -> None:
    """A table keeps one of Python's own classes polars has a type for, a ``Literal`` or enum of one class, a list
    or tuple of one item type, a model, dataclass or ``TypedDict`` of such fields, and any of them ``| None``, at
    any depth, a type that holds itself too."""
    row: Any = TypedDict("Row", {"name": str, "cell": cell})  # type: ignore[operator]

    def tables(text: str) -> Table[row]:  # type: ignore[valid-type]
        return pl.DataFrame()

    declared(tables)


def test_a_generic_typed_dict_row_names_its_columns_with_its_arguments_filled_in():
    class Cell[T](TypedDict):
        value: T

    @node("keep-cells", title="Keep cells", description="For a test.")
    def keep_cells(cells: Table[Cell[int]]) -> Table[Cell[int]]:
        return pl.DataFrame({"value": [1], "extra": ["x"]})

    graph = compiled_graph(keep_cells)
    assert graph.inputs["cells"].type == "Table[value: int]"
    assert graph.run({"cells": [{"value": 1}]}).outputs["result"].columns == ["value"]


def test_a_dict_returned_is_one_output_and_a_dataclass_that_is_not_frozen_lists_outputs():
    def summary(labels: Series[str]) -> dict[str, int]:
        return {label: labels.count(label) for label in labels}

    def anything(x: int) -> Any:
        return x

    for one in (summary, anything):
        assert list(declared(one).outputs) == ["result"], one
    assert compiled_graph(declared(summary)).run({"labels": ["a", "b", "a"]}).outputs == {
        "result": {"a": 2, "b": 1}
    }

    # the dataclass is taken apart into its fields when the node returns, so no call can change it
    @dataclass
    class Loose:
        a: int

    def loose(x: int) -> Loose:
        return Loose(x)

    assert list(declared(loose).outputs) == ["a"]
    assert compiled_graph(declared(loose)).run({"x": 2}).outputs == {"a": 2}


def test_a_class_pydantic_cannot_check_is_refused_where_the_class_is_declared():
    with pytest.raises(NodeDeclarationError) as raised:

        @host_type("thing", title="Thing")
        class Thing:
            pass

    assert raised.value.problems[0].type == "host_type_not_supported"


def test_a_series_of_tables_is_accepted_as_many_tables():
    def f(x: Series[Table[Invoice]]) -> int:
        return 1

    port = declared(f).inputs["x"]
    assert port.is_series is True and port.type.id == "Table[name: str, amount: int]"


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
    output = port_of(f, "result", "outputs")
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
            inputs={"on": GraphInputRecord(), "day": GraphInputRecord()},
            nodes={
                "f": node_record("flags", graph_inputs=("on", "day")),
                "n": node_record("flip", sources={"on": ("f", "result")}),
                "w": node_record("when", graph_inputs=("day",)),
            },
        ),
    )
    run = graph.run({"on": True, "day": datetime.date(2026, 1, 1)})
    assert run.results["n"]["result"] is False
    assert run.results["w"]["result"] == datetime.date(2026, 1, 2)


def test_a_pydantic_model_travels_on_an_edge_and_through_json():
    @host_type("Visit", title="Visit")
    class Visit(BaseModel, frozen=True):
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
    """A port's JSON names each part as Python does, so ``json_schema`` is ``"json_schema"`` in both.

    A host moves between the Python objects and the served JSON, and one concept should have one name.
    """

    @node("f", title="F", description="For a test.")
    def f(x: int) -> str:
        return ""

    port = f.describe().inputs[0]
    written = json.loads(port.model_dump_json())
    assert set(written) <= set(type(port).model_fields)
    assert written["json_schema"] == port.json_schema == {"type": "integer"}


def test_describe_writes_each_default_as_its_type_writes_it_as_json():
    @host_type("point", title="Point")
    @dataclass(frozen=True)
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

    output = port_of(read, "result", "outputs")
    columns = {
        c["properties"]["name"]["const"]: c["properties"]["values"]["items"]
        for c in output.json_schema["properties"]["columns"]["prefixItems"]
    }
    assert (
        not output.is_series
        and output.type == "Table[name: str, amount: int]"
        and columns == {"name": {"type": "string"}, "amount": {"type": "integer"}}
    )
    any_columns = port_of(bare, "result", "outputs")
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
            record=GraphRecord(
                inputs={"x": GraphInputRecord()},
                nodes={"n": node_record("v", version=saved, graph_inputs=("x",))},
            ),
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
        record=GraphRecord(
            inputs={"text": GraphInputRecord()},
            nodes={"e": node_record("inner", version=3, graph_inputs=("text",))},
        ),
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
    assert (problem.type, problem.loc, problem.ctx["errors"][0]["type"]) == (
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
            record=GraphRecord(
                inputs={"x": GraphInputRecord()},
                nodes={"n": node_record("h", version=saved, graph_inputs=("x",))},
            ),
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
            {
                "inputs": {"x": {}},
                "nodes": {"empty": {"node": "empty", "inputs": {"x": {"graph_input": "x"}}}},
            }
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
                inputs={name: GraphInputRecord() for name in graph_inputs},
                nodes={
                    "n": node_record(kind, graph_inputs=graph_inputs),
                    "r": node_record("echo", sources={"v": ("n", "result")}),
                },
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
            inputs={"v": GraphInputRecord()},
            nodes={
                "e": node_record("echo", graph_inputs=("v",)),
                "n": node_record("outs", sources={"x": ("e", "nothing")}),
            },
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
    assert (problem.type, problem.loc, problem.ctx["errors"][0]["type"]) == (
        "compute_hook_returned_invalid",
        ("h",),
        "dict_type",
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
        ({"a": 1, "zzz": 2}, "a key of the mapping the node returned isn't one of its outputs"),
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


@pytest.mark.parametrize("key", [1, 1.5, None, (1, 2)], ids=["int", "float", "none", "tuple"])
def test_a_hook_whose_field_name_is_no_text_is_the_hooks_failure_named_by_that_name(key: Any):
    [problem] = _computing(lambda: {"ok": int, key: int}).problems
    assert (problem.type, problem.loc, problem.ctx["errors"][0]["type"]) == (
        "compute_hook_returned_invalid",
        ("h", str(key)),
        "string_type",
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


def test_a_port_a_hook_declares_wrong_keeps_the_declarations_code_and_never_its_default():
    """A default ``@node`` would refuse makes ``compute_hook_returned_invalid`` with the declaration's code.

    The declaration's own message quotes the default, a value, so the problem names only the port.
    """
    [problem] = _computing(lambda: {"count": (int, "secret-default")}).problems
    assert (problem.type, problem.ctx["errors"][0]["type"]) == (
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
    assert (problem.type, problem.ctx["errors"][0]["type"]) == (
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

    assert port_of(declared(noted), "x").type == "int"
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

    (port,) = declared(f).describe().inputs
    assert (port.title, port.widget, port.json_schema) == (
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
        port = port_of(total, name)
        item_schema = {"type": "integer", "minimum": 0}
        assert (port.type, port.title, port.is_series, port.json_schema) == (
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
            GraphRecord(
                inputs={"text": GraphInputRecord()},
                nodes={"r": node_record("rows", graph_inputs=("text",))},
            ),
            registry_of(rows),
        ).problems
        == []
    )
    run = Graph.compile(
        GraphRecord(
            inputs={"text": GraphInputRecord()},
            nodes={"r": node_record("rows", graph_inputs=("text",))},
        ),
        registry_of(rows),
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
    def f(x: Annotated[int, Param(ge=1.5)]) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    (problem,) = raised.value.problems
    assert problem.type == "param_limit_value_invalid"
    assert "error" not in problem.ctx and "coercible to an integer" in problem.ctx["reason"]


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


def test_a_literal_inputs_choices_are_in_its_record_and_a_widget_draws_them():
    @node("priority", title="Priority", description="For a test.")
    def priority(level: Annotated[Literal["low", "high"], Param(widget=Dropdown())]) -> str:
        return level

    described = priority.describe().inputs[0]
    assert described.choices == (Choice("low", "low"), Choice("high", "high"))
    assert described.widget == {"type": "dropdown"}


def test_a_fields_choices_on_its_plain_type_title_its_values_and_limit_what_it_takes():
    from conductor import Ask, InvalidCallError, make_graph

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
    registry = registry_of(size, rate)
    with pytest.raises(InvalidCallError):
        Graph.compile(make_graph(size), registry).run({"name": "XL"})
    assert Graph.compile(make_graph(size), registry).run({"name": "M"}).outputs == {"result": "M"}
    asked = Graph.compile(make_graph(rate), registry).run({"text": "x"}).waiting[0].questions[0]
    assert asked.choices == (Choice("low", "Lav"), Choice("high", "Høj"))
    for wrong, code in (
        (("low", "low", "high"), "choices_invalid"),
        ((1, 2), "choices_invalid"),
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
    """The plain type goes where the ``Literal`` is, so the port stays a ``Series``."""
    from conductor import Series

    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(levels: Annotated[Series[Literal["low"]], Param(choices=["low"])]) -> str:
            return ""

    assert raised.value.problems[0].ctx == {"type": "str"}


@pytest.mark.parametrize("hint", [Literal["auto"] | int, Literal["auto"] | str | None])
def test_choices_on_a_literal_beside_another_type_are_refused_and_told_to_go(hint: Any) -> None:
    """Choices there would limit the port to them, so the other type would take nothing more.

    Writing the plain type in the ``Literal``'s place would do the same, so the message says only to remove them,
    and the port without them takes ``"auto"`` and any value of the other type.
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

    registry = registry_of(glue)
    graph = Graph.compile(
        {
            "nodes": {"g": {"node": "glue", "inputs": {"text": {"constant": "t"}}}},
            "outputs": {"result": {"node": "g", "output": "result"}},
        },
        registry,
    )
    assert graph.problems == [] and graph.run().outputs == {"result": "txx"}
    (problem,) = Graph.compile(
        {
            "nodes": {
                "g": {
                    "node": "glue",
                    "inputs": {"text": {"constant": "t"}, "names": {"constant": "boom"}},
                }
            }
        },
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
        {
            "nodes": {"g": {"node": "glue", "inputs": {"names": {"constant": "a"}}}},
            "outputs": {},
        },
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
        {
            "nodes": {"g": {"node": "glue", "inputs": {"names": {"constant": "a"}}}},
            "outputs": {},
        },
        registry_of(glue),
    ).problems
    assert (raised.problems[0].type, problem.ctx["error"]) == (code, code)


def test_a_configuration_the_graphs_author_fixes_is_a_value_and_one_left_open_is_the_callers():
    @node("ask-model", title="Ask model", description="For a test.")
    def ask_model(prompt: str, model: str = "small") -> str:
        return f"{model}:{prompt}"

    registry = registry_of(ask_model)
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
    ids=["a bool", "a number as text", "below the port's limit"],
)
def test_a_declared_choice_is_read_strictly_and_checked_against_the_ports_limits(
    choices: list[object], limits: dict[str, int], reason: str
) -> None:
    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(n: Annotated[int, Param(choices=choices, **limits)]) -> int:  # type: ignore[arg-type]
            return n

    (problem,) = raised.value.problems
    assert (problem.type, problem.ctx["errors"][0]["loc"]) == ("choices_invalid", [0])
    assert f"the choices in int can't be used: {reason}." in str(raised.value)


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


@host_type("cap-kind", title="Cap kind")
class _CapKind(Enum):
    NONE = float("inf")
    TEXT = "Infinity"


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
        (_CapKind, "named_value_invalid_alike"),
    ],
    ids=[
        "an-enums-member",
        "in-a-list",
        "in-a-series",
        "a-records-field",
        "a-tables-column",
        "bytes",
        "nan",
        "nan-in-a-list",
        "inf-and-its-text",
        "inf-and-its-text-in-a-list",
        "an-enum-whose-members-write-one-json",
    ],
)
def test_a_literal_whose_values_arent_plain_json_is_refused_at_any_depth(
    hint: Any, code: str
) -> None:
    """A ``Literal``'s values are text, numbers, yes or no, or ``None``, each with a JSON form of its own, on the
    port, a list's or a ``Series``'s item, a record's field and a table's column alike. An enum's members are held to
    the same: ``inf`` and ``"Infinity"`` are written as one JSON, so a saved run couldn't tell the two apart."""

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

    def keyed(limits: dict[float, str] = {float("inf"): "top"}) -> str:  # noqa: B006
        return str(limits)

    def picked_by_key(
        limits: Annotated[dict[float, str], Param(choices=[{float("inf"): "top"}])],
    ) -> str:
        return str(limits)

    refused = []
    for wrong in (capped, capped_in_a_list, picked, picked_on_output, keyed, picked_by_key):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        refused.append(raised.value.problems[0].type)
    assert refused == [
        "default_invalid",
        "default_invalid",
        "choices_invalid",
        "choices_invalid",
        "default_invalid",
        "choices_invalid",
    ]


def test_a_default_or_a_choice_a_hosts_check_raises_on_is_refused_as_value_error() -> None:
    """A host's own check may raise something other than pydantic's error, such as ``float(None)``'s
    ``TypeError``. For a default and a choice alike, that is a refusal where the node is declared, not a crash."""

    def cannot_check(value: float) -> float:
        return float(None)  # type: ignore[arg-type]

    def with_default(limit: Annotated[float, AfterValidator(cannot_check)] = 1.0) -> str:
        return str(limit)

    def with_choice(
        limit: Annotated[float, AfterValidator(cannot_check), Param(choices=[1.0])],
    ) -> str:
        return str(limit)

    refused = []
    for wrong in (with_default, with_choice):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        (problem,) = raised.value.problems
        refused.append((problem.type, [error["type"] for error in problem.ctx["errors"]]))
    assert refused == [("default_invalid", ["value_error"]), ("choices_invalid", ["value_error"])]


@host_type("Shade", title="Shade")
class Shade:
    """A host's value pydantic can write only by its own class's schema, as pydantic-extra-types' ``Color`` is."""

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from_text = core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
        return core_schema.json_or_python_schema(
            json_schema=from_text,
            python_schema=core_schema.union_schema(
                [core_schema.is_instance_schema(cls), from_text]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda shade: shade.name
            ),
        )


def test_a_choice_of_a_hosts_class_is_written_by_its_own_schema_as_its_default_is() -> None:
    """A choice is read from its JSON form, as a default and an answer are, so a value pydantic can write only by
    its class's own schema is written by it there too."""

    def tinted(
        shade: Annotated[Shade, Param(choices=[Shade("red"), Shade("blue")])] = Shade("red"),
    ) -> str:
        return shade.name

    definition = declared(tinted)
    (described,) = definition.describe().inputs
    assert [choice.id for choice in described.choices or ()] == ["red", "blue"]
    assert compiled_graph(definition).run().outputs == {"result": "red"}

    def tint(text: str) -> Shade:
        raise Ask("Which shade?", choices=[Shade("red"), Shade("blue")])

    asking = declared(tint, asks=True)
    (question,) = compiled_graph(asking).run({"text": "wall"}).waiting[0].questions
    assert [choice.id for choice in question.choices or ()] == ["red", "blue"]


@pytest.mark.parametrize(
    ("hint", "default"),
    [
        (int, "3"),
        (int, 5.0),
        (bool, 1),
        (float, Decimal("1.5")),
        (datetime.date, datetime.datetime(2026, 1, 2)),
    ],
    ids=["text-for-int", "float-for-int", "int-for-bool", "decimal-for-float", "datetime-for-date"],
)
def test_a_default_is_read_strictly_as_its_json_so_one_a_graph_input_would_refuse_is_refused_at_node(
    hint: Any, default: Any
) -> None:
    """A node's own default is read as a caller's graph input and a record's constant are, from its JSON form: a
    graph record writes it and reads it back, so it reads the same from either."""

    def f(x: Any = default) -> int:
        return 1

    f.__annotations__["x"] = hint
    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.problems[0].type == "default_invalid"


def test_a_default_given_as_its_json_is_read_by_its_type() -> None:
    def f(day: datetime.date = "2026-01-02", n: float = 1) -> str:  # type: ignore[assignment]
        return f"{day!r} {n!r}"

    assert compiled_graph(declared(f)).run().outputs == {"result": "datetime.date(2026, 1, 2) 1.0"}


class _Tagged(TypedDict):
    tags: frozenset[str]


@pytest.mark.parametrize(
    ("hint", "code"),
    [
        (SecretStr, "type_not_supported_secret"),
        (SecretBytes, "type_not_supported_secret"),
        (Secret[int], "type_not_supported_secret"),
        (list[SecretStr], "type_not_supported_secret"),
        (ImportString, "type_not_supported_import_string"),
        (ImportString[float], "type_not_supported_import_string"),
        (Json, "type_not_supported_decoding"),
        (Json[int], "type_not_supported_decoding"),
        (Base64Bytes, "type_not_supported_decoding"),
        (Base64Str, "type_not_supported_decoding"),
        (Base64UrlBytes, "type_not_supported_decoding"),
        (Base64UrlStr | None, "type_not_supported_decoding"),
    ],
    ids=[
        "secret-str",
        "secret-bytes",
        "secret-of-int",
        "secret-in-a-list",
        "import-string",
        "import-string-of-float",
        "json",
        "json-of-int",
        "base64-bytes",
        "base64-str",
        "base64-url-bytes",
        "base64-url-str-or-none",
    ],
)
def test_a_secret_an_import_string_or_a_decoding_type_is_refused_at_node(
    hint: Any, code: str
) -> None:
    """A secret is saved and matched as its mask, an ``ImportString`` runs the code a record names, and a value a
    type decoded fails it the next time it is checked, so each is refused where the node is declared, on an input
    and an output alike."""

    def takes(x: Any) -> int:
        return 1

    def returns() -> Any:
        return None

    takes.__annotations__["x"] = hint
    returns.__annotations__["return"] = hint
    for wrong in (takes, returns):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        assert raised.value.problems[0].type == code


@host_type("tag-holder", title="Tag holder")
class _TagHolder(BaseModel, frozen=True):
    tags: set[str]


@host_type("number-bag", title="Number bag")
class _NumberBag(BaseModel, frozen=True):
    numbers: frozenset[int]


@host_type("tag-record", title="Tag record")
@dataclass(frozen=True)
class _TagRecord:
    tags: set[str]


@host_type("tag-shelf", title="Tag shelf")
class _TagShelf(BaseModel, frozen=True):
    holders: list[_TagHolder]


class _TagRow(TypedDict):
    tags: set[str]


@host_type("sorted-tags", title="Sorted tags")
class _SortedTags(BaseModel, frozen=True):
    tags: set[str]

    @field_serializer("tags")
    def _sorted(self, tags: set[str]) -> list[str]:
        return sorted(tags)


@pytest.mark.parametrize(
    ("hint", "where"),
    [
        (set[int], "set[int]"),
        (frozenset[str], "frozenset[str]"),
        (set, "set"),
        (Set[int], "frozenset[int]"),
        (int | set[str], "set[str]"),
        (list[set[str]], "set[str]"),
        (dict[str, frozenset[int]], "frozenset[int]"),
        (Series[frozenset[int]], "frozenset[int]"),
        (_Tagged, "_Tagged.tags"),
        (_TagHolder, "_TagHolder.tags"),
        (_NumberBag, "_NumberBag.numbers"),
        (_TagRecord, "_TagRecord.tags"),
        (list[_TagShelf], "_TagHolder.tags"),
        (Table[_TagRow], "_TagRow.tags"),
        (_SortedTags, "_SortedTags.tags"),
    ],
    ids=[
        "set",
        "frozenset",
        "bare-set",
        "abc-set",
        "in-a-union",
        "in-a-list",
        "a-dicts-value",
        "in-a-series",
        "in-a-typed-dict",
        "model-set",
        "model-frozenset",
        "dataclass-set",
        "set-in-a-list-of-models",
        "a-tables-row",
        "a-serializer-that-sorts-it",
    ],
)
def test_a_set_anywhere_is_refused_at_node_naming_the_field_or_the_set(
    hint: Any, where: str
) -> None:
    """A set has no order, so it has no JSON form that keeps it the same, and no default of a set type could be read
    from one. A host's class holds its value as pydantic writes it, so a set field there writes a list in a varying
    order, and its call hashes differently in each process. The field that holds it is named, the nearest class's, a
    table's row's too; with no class around it, the set as written. A serializer that sorts it doesn't make it
    taken: the rule is that a declared type holds no set."""

    def takes(x: Any) -> int:
        return 1

    def returns() -> Any:
        return None

    takes.__annotations__["x"] = hint
    returns.__annotations__["return"] = hint
    for wrong in (takes, returns):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        (problem,) = raised.value.problems
        assert (problem.type, problem.ctx) == ("type_not_supported_set", {"type": where})


@pytest.mark.parametrize(
    "default",
    [
        {"type": "set"},
        {"type": "bytes"},
        {"type": "x", "metadata": {"conductor_choices": ["a"]}},
        {"type": "definition-ref"},
        {"type": "dataclass-field"},
        {"type": "definitions"},
    ],
    ids=["set", "bytes", "choices", "definition-ref", "dataclass-field", "definitions"],
)
def test_a_fields_default_that_looks_like_a_schema_is_a_value_not_a_schema(
    default: dict[str, Any],
) -> None:
    """The walk of a type's schema reads only the keys that hold schemas, so a model field's default, a value, is
    never read as one."""

    @host_type(f"settings-{default['type']}", title="Settings")
    class Settings(BaseModel, frozen=True):
        model_config = ConfigDict(**JSON_CONFIG)

        meta: dict[str, Any] = default

    def takes(x: Settings) -> int:
        return 1

    declared(takes)


def _exclaimed(text: str) -> str:
    return text + "!"


_Exclaimed = Annotated[str, AfterValidator(_exclaimed), Param(choices=["calm", "glad"])]


@host_type("mood-card", title="Mood card")
class _MoodCard(BaseModel, frozen=True):
    mood: _Exclaimed


@pytest.mark.parametrize(
    ("hint", "where"),
    [
        (_Exclaimed, "str"),
        (list[_Exclaimed], "str"),
        (dict[str, _Exclaimed], "str"),
        (_MoodCard, "str"),
    ],
    ids=["the-port", "a-lists-item", "a-dicts-value", "a-models-field"],
)
def test_choices_another_check_on_the_hint_moves_off_are_refused_at_any_depth(
    hint: Any, where: str
) -> None:
    """Every value would fail at run time: the check adds "!" before the choices are compared. So the choices are
    read by the check around them wherever their ``Param`` sits, and the fix is the check, not the choices."""

    def takes(x: Any) -> int:
        return 1

    takes.__annotations__["x"] = hint
    with pytest.raises(NodeDeclarationError) as raised:
        declared(takes)
    (problem,) = raised.value.problems
    assert (problem.type, problem.ctx["type"], problem.ctx["errors"][0]["type"]) == (
        "choices_invalid",
        where,
        "not_a_choice",
    )
    assert problem.ctx["instead"].startswith("Give choices that every check on the value keeps")


def test_choices_inside_a_type_that_its_checks_keep_are_taken() -> None:
    """The walk reads nested choices by their own check, so ones that check keeps, a float's ``1`` and an enum's
    member inside a model, are taken."""

    class Size(Enum):
        SMALL = "s"
        LARGE = "l"

    @host_type("sized-card", title="Sized card")
    class SizedCard(BaseModel, frozen=True):
        size: Annotated[Size, Param(choices=[Size.SMALL])]
        weight: Annotated[float, Param(choices=[1, 2.5])]

    def takes(x: list[SizedCard], y: dict[str, Annotated[int, Param(choices=[1, 2])]]) -> int:
        return 1

    declared(takes)


_Bag = TypeVar("_Bag", bound=frozenset[int])


def test_a_set_in_a_type_variables_bound_or_in_an_input_a_hook_adds_is_refused_as_at_node() -> None:
    """A variable no edge fills in becomes its bound, and a hook's port is held to ``@node``'s rules."""

    def bagged(x: _Bag) -> _Bag:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(bagged)
    assert raised.value.problems[0].type == "type_not_supported_set"

    def adds() -> dict[str, Any]:
        return {"tags": set[str]}

    def tagged(**added: Any) -> int:
        return 1

    (problem,) = compiled_graph(
        node("tagged", title="Tagged", description="For a test.", compute_added_inputs=adds)(tagged)
    ).problems
    assert (problem.type, problem.ctx["errors"][0]["type"]) == (
        "compute_hook_returned_invalid",
        "type_not_supported_set",
    )


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

    assert raised.value.problems[0].type == "choices_invalid"

    def on_table(x: Annotated[Table, Param(choices=["a"])]) -> int:
        return 0

    with pytest.raises(NodeDeclarationError) as raised:
        declared(on_table)
    assert raised.value.problems[0].type == "choices_invalid"


def test_a_default_with_no_json_form_is_refused_where_the_node_is_declared() -> None:
    with pytest.raises(NodeDeclarationError) as raised:

        @node("b", title="B", description="Takes a value.")
        def b(data: Any = object()) -> int:
            return 1

    assert raised.value.problems[0].type == "default_invalid"


def test_a_default_of_a_hosts_class_with_no_json_form_is_refused_as_no_json_form_not_by_the_classs_check():
    @host_type("Handle", title="Handle")
    class Handle:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
            return core_schema.is_instance_schema(cls)

    with pytest.raises(NodeDeclarationError) as raised:

        @node("held", title="Held", description="Takes a handle.")
        def held(handle: Handle = Handle()) -> int:
            return 1

    (problem,) = raised.value.problems
    assert (problem.type, problem.ctx["reason"]) == (
        "default_invalid",
        "the value can't be written as JSON",
    )


def test_a_default_is_described_as_its_json_and_a_graph_input_offers_the_records_default_as_its_json():
    class Row(TypedDict):
        n: int

    @node("t", title="T", description="Counts rows.")
    def t(
        rows: Table[Row] = pl.DataFrame({"n": [1, 2]}),
        when: datetime.date = datetime.date(2026, 1, 2),
    ) -> int:
        return len(rows)

    described = {port.name: port.default for port in t.describe().inputs}
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


def test_every_fault_in_a_params_choices_is_one_choices_invalid_with_each_choice_by_its_index() -> (
    None
):
    """A choice with no JSON form, one of another type and one that isn't finite are one problem, each fault an
    error at the index of its choice, as an ``Ask``'s choices are reported."""
    with pytest.raises(NodeDeclarationError) as raised:

        @node("mixed", title="Mixed", description="For a test.")
        def mixed(pick: Annotated[float, Param(choices=[object(), "a", float("inf")])]) -> float:
            return pick

    (problem,) = raised.value.problems
    assert (problem.type, [(e["type"], e["loc"]) for e in problem.ctx["errors"]]) == (
        "choices_invalid",
        [("no_json_form", [0]), ("float_type", [1]), ("finite_number", [2])],
    )
    assert problem.ctx["reason"] == problem.ctx["errors"][0]["msg"]


def test_a_choice_with_a_set_in_it_is_refused_as_set_has_no_order() -> None:
    """A set's JSON form is a list, so a choice holding one, at any depth, would read back as another value. It is
    refused as a set is everywhere a value enters, even on ``Any``, which takes a list."""
    for choices in ([{"a"}], [["x", [frozenset({1})]]]):
        with pytest.raises(NodeDeclarationError) as raised:

            @node("loose", title="Loose", description="For a test.")
            def loose(pick: Annotated[Any, Param(choices=choices)]) -> Any:
                return pick

        (problem,) = raised.value.problems
        assert (problem.type, [(e["type"], e["loc"]) for e in problem.ctx["errors"]]) == (
            "choices_invalid",
            [("set_has_no_order", [0])],
        ), choices


def test_text_or_a_set_given_as_a_params_choices_is_refused_as_an_asks_are() -> None:
    """Choices are a list or a tuple. Text would offer each of its characters, and a set its items in no fixed order,
    so both are refused at ``@node``, by the rule an ``Ask``'s choices are read by."""
    for choices, code in (("ab", "list_type"), ({"a", "b"}, "set_has_no_order")):
        with pytest.raises(NodeDeclarationError) as raised:

            @node("letters", title="Letters", description="For a test.")
            def letters(pick: Annotated[str, Param(choices=choices)]) -> str:  # type: ignore[arg-type]
                return pick

        (problem,) = raised.value.problems
        assert (problem.type, [(e["type"], e["loc"]) for e in problem.ctx["errors"]]) == (
            "choices_invalid",
            [(code, [])],
        ), choices


@pytest.mark.parametrize(
    ("hint", "valid", "with_a_set"),
    [
        (str, ("a", "b"), {"a", "b"}),
        (Any, [["x", [[1]]]], [["x", [frozenset({1})]]]),
    ],
    ids=["top-level", "nested"],
)
@pytest.mark.parametrize("set_first", [False, True], ids=["valid-first", "set-first"])
def test_valid_choices_and_their_set_twin_are_told_apart_in_one_process(
    hint: Any, valid: Any, with_a_set: Any, set_first: bool
) -> None:
    """``typing`` keeps one ``Annotated[...]`` for arguments that compare equal, and a set's JSON text is a list's.
    So a ``Param`` whose choices hold a set must not equal its twin with a list or tuple there, or whichever is
    declared first would decide for both: the set taken, or the valid choices refused."""

    def declare(choices: Any) -> list[str]:
        try:

            @node("twin", title="Twin", description="For a test.")
            def twin(pick: Annotated[hint, Param(choices=choices)]) -> Any:
                return pick

        except NodeDeclarationError as raised:
            return [error["type"] for error in raised.problems[0].ctx["errors"]]
        return []

    order = [(with_a_set, ["set_has_no_order"]), (valid, [])]
    for choices, expected in order if set_first else order[::-1]:
        assert declare(choices) == expected, choices


def test_a_reason_a_host_check_gave_is_quoted_with_one_period() -> None:
    """The text that replaces a host validator's message has no final period, so a message quoting it reads right."""
    from pydantic import AfterValidator

    def refuse(value: str) -> str:
        raise ValueError(f"no {value}")

    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(pick: Annotated[str, AfterValidator(refuse), Param(choices=["x"])]) -> str:
            return pick

    assert raised.value.problems[0].type == "choices_invalid"
    assert "did not pass its type's own check. Give each choice" in str(raised.value)
