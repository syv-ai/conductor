"""Tests that the values a run passes around are checked against their types.

Two kinds of value are checked. The first is what a node returns, which must match its declared return type. The
second is a value on an edge, when the source's type is too loose to say whether the target's input takes it.

A value that doesn't fit fails a node for that item. A bad return fails the node that returned it, with
``invalid_output``. A bad value on an edge fails the node it goes into, with ``invalid_edge_value``. Neither message
quotes the value, since it may be private.

A value that is already of its declared class is kept as it is. Anything else is read strictly by its type. So a
model may be returned as a dict, but ``"12"`` is never an ``int``, and ``12.0`` never goes in an ``int`` column.
"""

import asyncio
import datetime
import json
import math
from dataclasses import dataclass, field
from threading import Event
from typing import Annotated, Any, Literal, TypedDict, TypeVar

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Policy,
    Problem,
    Run,
    Series,
    Source,
    Table,
    host_type,
    make_graph,
    node,
)
from pydantic import AfterValidator, BaseModel, ConfigDict, Discriminator, Field, field_validator
from pydantic_core import PydanticCustomError, core_schema

from tests.test_core.nodes import input_entry, registry_of

SECRET = "cpr-1234567890"


def _returning(returned: Any, return_type: Any) -> Any:
    """Runs a graph of one node, ``g``, that returns ``returned`` and is declared ``-> return_type``."""

    def give() -> return_type:  # type: ignore[valid-type]
        return returned

    definition = node("give", title="Give", description="For a test.")(give)
    outputs = {name: Source("g", name) for name in definition.outputs}
    return Graph.compile(
        GraphRecord(nodes={"g": NodeRecord(node="give")}, outputs=outputs), registry_of(definition)
    ).run()


def _feeding(returned: Any, return_type: Any, sink: Any) -> Any:
    """Runs node ``a``, which returns ``returned``, wired into the one input of ``sink`` as node ``b``."""

    def give() -> return_type:  # type: ignore[valid-type]
        return returned

    definition = node("give", title="Give", description="For a test.")(give)
    (name,) = sink.inputs
    return Graph.compile(
        GraphRecord(
            nodes={
                "a": NodeRecord(node="give"),
                "b": NodeRecord(
                    node=sink.id, inputs={name: input_entry({"sources": [Source("a", "result")]})}
                ),
            },
            outputs={output: Source("b", output) for output in sink.outputs},
        ),
        registry_of(definition, sink),
    ).run()


# -- what a node returns ----------------------------------------------------------


@pytest.mark.parametrize(
    ("returned", "return_type"),
    [(42, str), ("abc", Series[str]), (None, str), (float("nan"), int), ("x", None)],
    ids=[
        "an-int-for-a-str",
        "a-str-for-a-series",
        "none-without-the-hint",
        "nan-for-an-int",
        "a-value-from-no-outputs",
    ],
)
def test_a_return_that_is_not_the_declared_type_fails_the_node_that_returned_it(
    returned, return_type
):
    """A wrong return fails at ``g`` itself, never further on.

    So ``-> str`` returning ``42`` fails at ``g``, and ``"abc"`` for a ``Series`` is refused, not read as three items.
    """
    run = _returning(returned, return_type)

    assert (run.status, run.cause.type, run.cause.loc) == ("failed", "invalid_output", ("g",))
    assert run.results["g"] == ({"result": None} if return_type is not None else {})


def test_returning_none_needs_the_hint():
    assert _returning(None, str | None).status == "complete"
    assert _returning(None, str).cause.type == "invalid_output"


@pytest.mark.parametrize(
    ("returned", "return_type"),
    [([1], list[int] | None), ("a", Literal["a", "b"] | None)],
    ids=["a-list", "a-literal"],
)
def test_an_optional_output_of_a_type_that_is_not_a_class_is_read_by_its_type(
    returned, return_type
):
    """``-> list[int] | None`` returning ``[1]`` is read just as ``-> list[int]`` reads it.

    It isn't compared against the class ``NoneType`` and refused.
    """
    run = _returning(returned, return_type)

    assert run.status == "complete"
    assert run.results["g"]["result"] == returned


def test_a_value_read_as_its_type_is_kept_as_the_type():
    """A value is converted to the type its node declares.

    A model returned as a dict becomes the model. An int returned for a ``float`` becomes a float. A ``str`` returned
    for ``class Text(str)`` becomes a ``Text``.
    """

    @host_type("Point", title="Point")
    class Point(BaseModel, frozen=True):
        x: int

    @host_type("Text", title="Text")
    class Text(str):
        pass

    assert _returning({"x": 1}, Point).results["g"]["result"] == Point(x=1)
    assert type(_returning(3, float).results["g"]["result"]) is float
    assert type(_returning("a", Text).results["g"]["result"]) is Text


class Shape(dict):  # type: ignore[type-arg]
    pass


@host_type("Square", title="Square")
class Square(Shape):
    """A type whose own check takes any mapping with a ``side`` and gives it back as it is, so its values are never
    of the class ``Square``, as pydantic's ``EmailStr`` gives a ``str``."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def with_a_side(value: Any) -> Any:
            if not isinstance(value, dict) or "side" not in value:
                raise PydanticCustomError("no_side", "a square has a side")
            return value

        return core_schema.no_info_plain_validator_function(with_a_side)


def test_an_answer_is_what_its_types_own_check_gives_whatever_its_class():
    @node("square", title="Square", description="For a test.", asks=True)
    def square() -> Square:
        raise Ask("Which square?")

    waiting = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"g": {"node": "square"}},
                "outputs": {"result": {"node": "g", "output": "result"}},
            }
        ),
        registry_of(square),
    ).run()
    answered = waiting.with_answers(waiting.waiting[0].id, {"side": 2}).resume()

    assert (answered.status, answered.outputs) == ("complete", {"result": {"side": 2}})
    with pytest.raises(InvalidCallError) as refused:
        waiting.with_answers(waiting.waiting[0].id, {"corner": 2})
    assert [(p.type, p.ctx["reason"]) for p in refused.value.problems] == [
        ("invalid_answer", "The value did not pass its type's own check")
    ]


def test_an_asks_choice_is_what_its_types_own_check_gives_as_an_answer_is():
    @node("square", title="Square", description="For a test.", asks=True)
    def square() -> Square:
        raise Ask("Which square?", choices=[{"side": 2}])

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"g": {"node": "square"}},
                "outputs": {"result": {"node": "g", "output": "result"}},
            }
        ),
        registry_of(square),
    ).run()

    assert run.status == "waiting"
    assert [choice.id for choice in run.waiting[0].questions[0].choices] == [{"side": 2}]


def test_a_value_its_types_own_check_gives_as_another_class_is_returned_saved_and_read_back():
    run = _returning(Shape(side=2), Square)

    assert (run.status, run.outputs) == ("complete", {"result": {"side": 2}})
    assert Run.from_json(run.to_json(), run.graph).outputs == {"result": {"side": 2}}


@host_type("Shout", title="Shout")
class Shout(str):
    """A host's ``str`` subclass whose own check takes only capitals, while its constructor checks nothing."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def capitals(value: str) -> str:
            if not value.isupper():
                raise PydanticCustomError("not_shouted", "a shout is in capitals")
            return cls(value)

        return core_schema.no_info_after_validator_function(capitals, core_schema.str_schema())


def test_a_hosts_class_built_by_hand_is_still_checked_by_its_own_check():
    """A value of exactly a host's class is checked like any other, since its constructor may check nothing: a run
    that kept ``Shout("hi")`` as it is could not be read back."""
    refused = _returning(Shout("hi"), Shout)
    taken = _returning(Shout("HI"), Shout)

    assert (refused.status, refused.cause.type, refused.cause.ctx["errors"][0]["type"]) == (
        "failed",
        "invalid_output",
        "not_shouted",
    )
    assert type(taken.outputs["result"]) is Shout
    assert Run.from_json(taken.to_json(), taken.graph).outputs == {"result": "HI"}


@host_type("Hue", title="Hue")
class Hue:
    """A host's class whose own schema reads only text, so it refuses a ``Hue`` itself."""

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


def test_a_class_whose_own_schema_refuses_its_own_value_cant_be_returned_and_the_reason_says_so():
    """What a node returns is checked by its type's own schema, so a schema must take a value of its own class, as
    pydantic asks; the reason names that fault, since pydantic's own message speaks only of text."""
    run = _returning(Hue("red"), Hue)

    assert (run.status, run.cause.type, run.cause.ctx["errors"][0]["type"]) == (
        "failed",
        "invalid_output",
        "string_type",
    )
    assert run.cause.ctx["reason"] == (
        "the type's own check refuses a Hue, a value of its own class (Input should be a valid string)"
    )


@host_type("Tint", title="Tint")
class Tint:
    """A host's class whose own check refuses every value and writes the value into its message."""

    def __repr__(self) -> str:
        return f"Tint({SECRET})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def unread(value: Any) -> Any:
            raise ValueError(f"cannot read {value!r}")

        return core_schema.no_info_plain_validator_function(unread)


def test_the_reason_for_a_refused_value_of_its_own_class_quotes_nothing_of_the_value():
    """The reason quotes the error's message without the value, as every refusal does, since a host's check may write
    the value into its own message."""
    run = _returning(Tint(), Tint)

    assert run.cause.ctx["reason"] == (
        "the type's own check refuses a Tint, a value of its own class (The value did not pass its type's own check)"
    )
    assert SECRET not in run.cause.msg


@dataclass(frozen=True)
class Pair:
    name: str
    count: int


def test_a_node_returning_something_other_than_its_dataclass_fails_naming_the_class():
    run = _returning(("a", 1), Pair)

    assert (run.cause.type, run.cause.loc) == ("invalid_output", ("g",))
    assert "Pair" in run.cause.ctx["reason"] and "tuple" in run.cause.ctx["reason"]


@dataclass(frozen=True)
class OnlyResult:
    result: str


def test_a_dataclass_whose_one_field_is_named_result_is_spread_like_any_dataclass_of_outputs():
    assert _returning(OnlyResult("hi"), OnlyResult).results["g"] == {"result": "hi"}

    run = _returning("hi", OnlyResult)

    assert (run.cause.type, run.cause.loc) == (
        "invalid_output",
        ("g",),
    ) and "OnlyResult" in run.cause.ctx["reason"]


@dataclass(frozen=True)
class NoFields:
    pass


def test_a_dataclass_with_no_fields_is_returned_as_itself_and_none_is_refused():
    """A dataclass with no fields declares no outputs, like ``-> None``.

    But the node must still return the dataclass, never ``None``.
    """
    assert _returning(NoFields(), NoFields).results["g"] == {}

    run = _returning(None, NoFields)

    assert (run.cause.type, run.cause.loc) == (
        "invalid_output",
        ("g",),
    ) and "NoFields" in run.cause.ctx["reason"]


def test_a_hook_that_gives_the_one_output_result_is_returned_as_a_mapping():
    @node(
        "named-result",
        title="Named result",
        description="For a test.",
        compute_outputs=lambda: {"result": str},
    )
    def named_result() -> Any:
        return {"result": "hi"}

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"c": {"node": "named-result"}},
                "outputs": {"result": {"node": "c", "output": "result"}},
            }
        ),
        registry_of(named_result),
    )

    assert graph.run().results["c"] == {"result": "hi"}


def test_a_call_whose_second_output_is_invalid_writes_none_of_its_outputs():
    run = _returning(Pair("ok", "many"), Pair)  # type: ignore[arg-type]

    assert (run.cause.type, run.cause.ctx["output"]) == ("invalid_output", "count")
    assert run.results["g"] == {"name": None, "count": None}
    assert "g" not in json.loads(run.to_json())["outcomes"]


def test_a_refused_output_is_named_ctx_output_as_every_problem_names_a_nodes_output():
    run = _returning(Pair("ok", "many"), Pair)  # type: ignore[arg-type]

    assert list(run.cause.ctx) == ["reason", "output", "errors"]
    assert json.loads(run.cause.model_dump_json())["ctx"]["output"] == "count"


def _computed(returned: Any) -> Any:
    @node(
        "computed",
        title="Computed",
        description="For a test.",
        compute_outputs=lambda: {"name": str, "age": int},
    )
    def computed() -> Any:
        return returned

    return Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"c": {"node": "computed"}},
                "outputs": {
                    "name": {"node": "c", "output": "name"},
                    "age": {"node": "c", "output": "age"},
                },
            }
        ),
        registry_of(computed),
    ).run()


def test_a_node_with_computed_outputs_returns_exactly_them_as_a_mapping():
    assert _computed({"name": "Ida", "age": 3}).results["c"] == {"name": "Ida", "age": 3}
    for returned, named, error in (
        ({"name": "Ida"}, "age", {"type": "missing", "loc": ["age"], "msg": "Field required"}),
        (
            {"name": "Ida", "age": 3, "more": 1},
            None,
            {"type": "extra_forbidden", "loc": ["[key]"], "msg": "Extra inputs are not permitted"},
        ),
        (
            {"name": "Ida", "age": 3, 1: 1},
            None,
            {"type": "invalid_key", "loc": ["[key]"], "msg": "Keys should be strings"},
        ),
    ):
        run = _computed(returned)
        assert (run.cause.type, run.cause.ctx.get("output"), run.cause.ctx["errors"]) == (
            "invalid_output",
            named,
            [error],
        )
    assert _computed(["Ida", 3]).cause.type == "invalid_output"


def test_a_key_that_is_no_output_is_never_quoted_since_data_may_have_built_it():
    """A node may build its mapping's keys from data, such as a person's e-mail address. So a key that is no output is
    refused at ``"[key]"``, with no ``ctx.output``, and the message and reason don't quote it, as a key that isn't text
    is refused wherever conductor reads a mapping."""
    run = _computed({"ida@example.com": 1, "name": "Ida", "age": 3, "beta": 2})

    assert "ida@example.com" not in run.cause.model_dump_json()
    assert (run.cause.ctx.get("output"), run.cause.ctx["reason"], run.cause.ctx["errors"]) == (
        None,
        "a key of the mapping the node returned isn't one of its outputs",
        [{"type": "extra_forbidden", "loc": ["[key]"], "msg": "Extra inputs are not permitted"}],
    )


def test_a_mapping_may_leave_a_branch_out_as_an_answer_may():
    """A branch left out of a returned mapping is skipped, as when a person's answer to the node leaves it out."""

    @node(
        "routed_by_hook",
        title="Routed by hook",
        description="For a test.",
        compute_outputs=lambda: {"long": (str, SKIPPED), "short": (str, SKIPPED)},
    )
    def routed_by_hook() -> Any:
        return {"long": "a line"}

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"r": {"node": "routed_by_hook"}},
                "outputs": {
                    "long": {"node": "r", "output": "long"},
                    "short": {"node": "r", "output": "short"},
                },
            }
        ),
        registry_of(routed_by_hook),
    ).run()

    assert (run.status, run.results["r"]) == ("complete", {"long": "a line", "short": None})


@dataclass(frozen=True)
class Routed:
    count: int = 0
    long: str = SKIPPED
    short: str = SKIPPED


def test_a_return_that_gives_no_branch_a_value_skips_every_branch():
    """A call gives a value to at most one branch, so it may give none.

    A return with only ``count`` skips ``long`` and ``short``. Returning ``SKIPPED`` skips ``count`` too.
    """
    assert _returning(Routed(count=2), Routed).results["g"] == {
        "count": 2,
        "long": None,
        "short": None,
    }
    assert _returning(SKIPPED, Routed).results["g"] == {"count": None, "long": None, "short": None}


def test_a_return_that_skips_an_output_that_is_not_a_branch_is_refused():
    """Only a branch, an output whose default is ``SKIPPED``, may be ``SKIPPED`` in a return.

    ``count`` defaults to 0, so it isn't a branch. The reason says only what is wrong; ``invalid_output``'s fix on
    the problems page says what to write instead.
    """
    run = _returning(Routed(count=SKIPPED, long="a"), Routed)

    assert (run.cause.type, run.cause.ctx["output"], run.cause.ctx["reason"]) == (
        "invalid_output",
        "count",
        "'count' is SKIPPED, but it isn't a branch: its default isn't SKIPPED",
    )


@host_type("skip-parcel", title="Parcel")
@dataclass(frozen=True)
class Parcel:
    weight: float
    fragile: bool


@host_type("skip-box", title="Box")
@dataclass(frozen=True)
class Box:
    value: Any


@dataclass(frozen=True)
class Inner:
    count: int = SKIPPED


@host_type("skip-outer", title="Outer")
@dataclass(frozen=True)
class Outer:
    inner: Inner


@pytest.mark.parametrize(
    ("returned", "return_type"),
    [
        (Parcel(1.0, SKIPPED), Parcel),
        (Box(SKIPPED), Box),
        (Outer(Inner()), Outer),
        ([1, [2, SKIPPED]], Series[Any]),
        ({"a": SKIPPED}, dict[str, Any]),
    ],
    ids=["a-field", "an-any-field", "a-nested-default", "inside-a-series-item", "a-dict-value"],
)
def test_skipped_inside_a_returned_value_is_refused_when_the_run_is_saved(returned, return_type):
    """``SKIPPED`` is an output's whole value or an item of a ``Series``. Inside a value it has no JSON form, but a
    return isn't searched for it, so the run completes and saving it, or its outputs, is ``run_unwritable``, whose
    message names that cause. A model's or dataclass's instance is taken without its fields checked again, so a
    field built as ``SKIPPED`` gets this far, and so does a default of a nested dataclass."""
    run = _returning(returned, return_type)
    assert run.status == "complete"

    for save, where in ((run.to_json, "g.result"), (run.outputs_to_json, "result")):
        with pytest.raises(InvalidCallError) as refused:
            save()
        (problem,) = refused.value.problems
        assert (problem.type, problem.ctx["where"]) == ("run_unwritable", where)
        assert "SKIPPED inside a value" in problem.msg


@node("pass-on-one", title="Pass on one", description="For a test.")
def _pass_on_one(values: Any) -> Any:
    return values[1]


@node("pass-on-all", title="Pass on all", description="For a test.")
def _pass_on_all(values: Any) -> Series[Any]:
    return list(values)


@node("shown", title="Shown", description="For a test.")
def _shown(value: Any) -> str:
    return "shown"


@pytest.mark.parametrize(
    ("passer", "shown"),
    [(_pass_on_one, None), (_pass_on_all, ["shown", None])],
    ids=["whole", "as-a-series-item"],
)
def test_a_node_that_passes_on_skipped_from_inside_a_value_skips_what_it_feeds_until_the_run_is_saved(
    passer: Any, shown: Any
) -> None:
    """A node typed ``Any`` that passes on ``SKIPPED`` taken from inside a value, whole or as a ``Series`` item,
    skips what it feeds as a branch not taken. The run reads ``complete`` in memory, and only saving it is
    ``run_unwritable``, at the value that holds ``SKIPPED``."""
    give = _giving([1, SKIPPED], Any)
    run = Graph.compile(
        make_graph(give, passer, _shown, outputs={"shown": "shown.result"}),
        registry_of(give, passer, _shown),
    ).run()

    assert (run.status, run.outputs["shown"]) == ("complete", shown)
    with pytest.raises(InvalidCallError) as refused:
        run.to_json()
    (problem,) = refused.value.problems
    assert (problem.type, problem.ctx["where"]) == ("run_unwritable", "give.result")


@pytest.mark.parametrize(
    ("returned", "return_type", "output", "error"),
    [
        (Routed(count=SKIPPED, long="a"), Routed, "count", "skipped_not_a_branch"),
        (Routed(long="a", short="b"), Routed, "short", "several_branches_given"),
        ({"count": 1}, Routed, None, "not_the_class"),
        ("a", None, None, "none_required"),
    ],
    ids=[
        "skipped-not-a-branch",
        "two-branches",
        "not-the-class",
        "none-required",
    ],
)
def test_a_return_of_the_wrong_shape_says_why_in_one_error_by_its_code(
    returned: Any, return_type: Any, output: str | None, error: str
) -> None:
    """A fault in the shape of a return is ``invalid_output`` with its one error in ``ctx.errors``, as every value
    refusal has, so a host reads the code there, such as ``several_branches_given`` as an answer gives it."""
    cause = _returning(returned, return_type).cause

    assert (cause.type, cause.ctx.get("output"), [e["type"] for e in cause.ctx["errors"]]) == (
        "invalid_output",
        output,
        [error],
    )
    assert (
        cause.ctx["errors"][0]["loc"] == [] and cause.ctx["errors"][0]["msg"] == cause.ctx["reason"]
    )


def test_skipped_as_a_branch_and_as_a_series_item_is_still_taken():
    """The walk looks below an output's whole value and a ``Series``'s items, so a branch and a skipped item stay as
    they were, and a table's cells aren't walked."""
    assert _returning(Routed(long="a"), Routed).results["g"] == {
        "count": 0,
        "long": "a",
        "short": None,
    }
    assert _returning([1, SKIPPED, 3], Series[int]).outputs == {"result": [1, None, 3]}
    assert _returning(pl.DataFrame({"a": [1, 2]}), Table).status == "complete"


def test_a_return_of_several_outputs_is_checked_for_its_shape_before_its_values():
    """When both branches have a value, that is the problem reported.

    ``count`` is wrong too, and it is declared first, but the shape of the return is checked before its values.
    """
    run = _returning(Routed(long="a", short="b", count="many"), Routed)  # type: ignore[arg-type]

    assert (run.cause.type, run.cause.ctx["output"]) == ("invalid_output", "short")
    assert run.cause.ctx["reason"] == (
        "'long' and 'short' both have a value, but they default to SKIPPED, so they are branches of one node"
    )
    assert _returning(Routed(long="a", count=2), Routed).results["g"] == {
        "count": 2,
        "long": "a",
        "short": None,
    }


@dataclass(frozen=True)
class Routed3:
    long: str = SKIPPED
    medium: str = SKIPPED
    short: str = SKIPPED


def test_a_return_giving_three_branches_a_value_names_all_three():
    run = _returning(Routed3(long="a", medium="b", short="c"), Routed3)

    assert run.cause.type == "invalid_output"
    assert run.cause.ctx["reason"].startswith(
        "'long', 'medium' and 'short' all have a value, but they default to SKIPPED"
    )
    assert (
        _returning(Routed3(long="a", medium="b"), Routed3)
        .cause.ctx["reason"]
        .startswith("'long' and 'medium' both have a value")
    )


class Entry(TypedDict):
    item: str
    amount: int


@dataclass(frozen=True)
class Squares:
    first: Square
    lines: Series[Entry]


def test_an_output_of_several_is_refused_as_its_own_type_says_naming_the_output():
    run = _returning(Squares(Shape(corner=2), [{"item": "a", "amount": 1}]), Squares)  # type: ignore[arg-type]

    assert (run.cause.ctx["output"], run.cause.ctx["errors"][0]["type"]) == ("first", "no_side")
    run = _returning(
        Squares(Square(side=2), pl.DataFrame({"item": ["a"], "amount": [1.5]})), Squares
    )  # type: ignore[arg-type]

    assert (run.cause.ctx["output"], run.cause.ctx["column"]) == ("lines", "amount")
    assert run.cause.ctx["reason"].startswith("column 'amount': ")


def test_a_series_with_a_skipped_item_is_checked_around_the_gap():
    run = _returning([1, SKIPPED, 2], Series[int])

    assert run.status == "complete" and run.results["g"]["result"] == [1, None, 2]
    assert _returning([1, SKIPPED, "2"], Series[int]).cause.type == "invalid_output"


class Line(TypedDict):
    item: str
    amount: int


def test_a_fraction_in_an_int_column_fails_the_node_rather_than_being_cut_off():
    run = _returning(pl.DataFrame({"item": ["a"], "amount": [12.75]}), Table[Line])

    assert (run.cause.type, run.cause.loc, run.cause.ctx["output"], run.cause.ctx["column"]) == (
        "invalid_output",
        ("g",),
        "result",
        "amount",
    )


def test_a_float_column_for_an_int_one_fails_even_when_whole():
    """Outputs are read strictly. ``12.0`` is a float, so an ``int`` column refuses it.

    It's the same rule by which ``-> int`` refuses ``"12"``.
    """
    assert (
        _returning(pl.DataFrame({"item": ["a"], "amount": [12.0]}), Table[Line]).cause.type
        == "invalid_output"
    )


def test_a_column_may_have_none_only_where_its_type_takes_none():
    """``None`` in an ``int`` column is refused when the table is returned, and the problem names the column.

    A column that may have ``None`` is declared ``int | None``.
    """

    class Loose(TypedDict):
        item: str
        amount: int | None

    with_a_none = pl.DataFrame({"item": ["a", "b"], "amount": [12, None]})
    refused = _returning(with_a_none, Table[Line]).cause
    assert (refused.type, refused.ctx["column"]) == ("invalid_output", "amount")
    kept = _returning(with_a_none, Table[Loose]).results["g"]["result"]
    assert kept.to_dicts() == [{"item": "a", "amount": 12}, {"item": "b", "amount": None}]


def test_a_column_of_yes_or_no_is_never_an_int_column_returned_on_an_edge_or_given():
    """A column of yes or no is refused where an ``int`` column is expected, though Python counts ``True`` as 1.

    A column is checked the same way as a single value. That holds for a returned table, a table on an edge and a
    table given as a graph input.
    """
    yes_no = pl.DataFrame({"item": ["a", "b"], "amount": [True, False]})
    assert (
        _returning(yes_no, Table[Line]).cause.type,
        _returning(yes_no, Table[Line]).cause.ctx["column"],
    ) == ("invalid_output", "amount")

    @node("total", title="Total", description="For a test.")
    def total(lines: Table[Line]) -> int:
        return int(lines["amount"].sum())

    on_edge = _feeding(yes_no, Table, total)
    assert (on_edge.cause.type, on_edge.cause.loc, on_edge.cause.ctx["column"]) == (
        "invalid_edge_value",
        ("b",),
        "amount",
    )
    with pytest.raises(InvalidCallError) as given:
        Graph.compile(
            GraphRecord.model_validate(
                {
                    "inputs": {"lines": {}},
                    "nodes": {
                        "t": {"node": "total", "inputs": {"lines": {"graph_input": "lines"}}}
                    },
                }
            ),
            registry_of(total),
        ).run({"lines": yes_no})
    assert [p.type for p in given.value.problems] == ["invalid_input"]


class Graded(TypedDict):
    level: Literal[1, 2]
    score: float


@pytest.mark.parametrize("column", ["level", "score"])
def test_a_column_refuses_a_yes_or_no_wherever_a_field_of_its_type_does(column):
    """A column refuses ``True`` wherever a single value of its type would.

    That covers a ``Literal`` of numbers, and a number. Python counts ``True`` as 1, but a yes or no is not a number.
    The column and the single value follow the same rule.
    """

    @node("count_rows", title="Count rows", description="For a test.")
    def count_rows(rows: Table[Graded]) -> int:
        return rows.height

    given = pl.DataFrame({"level": [1], "score": [3]}).with_columns(pl.lit(True).alias(column))
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(
            GraphRecord.model_validate(
                {
                    "inputs": {"rows": {}},
                    "nodes": {
                        "c": {"node": "count_rows", "inputs": {"rows": {"graph_input": "rows"}}}
                    },
                    "outputs": {"result": {"node": "c", "output": "result"}},
                }
            ),
            registry_of(count_rows),
        ).run({"rows": given})
    assert [(p.type, p.ctx.get("column")) for p in refused.value.problems] == [
        ("invalid_input", column)
    ]


@pytest.mark.parametrize(
    ("returned", "return_type"),
    [("no", bool), ("12", int), ("2026-01-02", datetime.date), (12.0, int)],
    ids=["a-word-for-a-bool", "text-for-an-int", "text-for-a-date", "a-float-for-an-int"],
)
def test_an_output_is_read_strictly_never_from_text_that_spells_it(returned, return_type):
    assert _returning(returned, return_type).cause.type == "invalid_output"


def test_a_node_may_return_a_table_as_records_and_anything_else_fails_it():
    assert _returning([{"item": "a", "amount": 1}], Table[Line]).results["g"][
        "result"
    ].to_dicts() == [{"item": "a", "amount": 1}]
    assert _returning("item,amount", Table[Line]).cause.type == "invalid_output"


# -- a value on an edge ---------------------------------------------------------------------


@host_type("PickyId", title="Picky id", accepts=[object])
class PickyId(str):
    """A type whose own check quotes the value it refuses, as ordinary Python code often does."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def check(value: Any) -> Any:
            if isinstance(value, PickyId):
                return value
            raise ValueError(f"not a valid id: {value}")

        return core_schema.no_info_plain_validator_function(check)


def test_a_value_on_an_edge_that_the_input_refuses_fails_the_target_node_quoting_nothing():
    @node("sink", title="Sink", description="For a test.")
    def sink(identifier: PickyId) -> str:
        return identifier

    run = _feeding(SECRET, str, sink)

    assert (run.status, run.cause.type, run.cause.loc, run.cause.ctx["input"]) == (
        "failed",
        "invalid_edge_value",
        ("b",),
        "identifier",
    )
    assert SECRET not in run.cause.model_dump_json() and run.cause.ctx["reason"]
    assert run.results["a"] == {"result": SECRET}


def test_a_value_on_an_edge_outside_its_bounds_fails_the_node_for_that_item():
    @node("numbers", title="Numbers", description="For a test.")
    def numbers() -> Series[int]:
        return [5, 1, 7]

    @node("at_least_five", title="At least five", description="For a test.")
    def at_least_five(n: Annotated[int, Param(ge=5)]) -> int:
        return n

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "a": {"node": "numbers"},
                    "b": {
                        "node": "at_least_five",
                        "inputs": {"n": {"sources": [{"node": "a", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "b", "output": "result"}},
            }
        ),
        registry_of(numbers, at_least_five),
    ).run()

    assert (run.cause.type, run.cause.loc) == ("invalid_edge_value", ("b", 1))
    assert [(e["type"], e["msg"]) for e in run.cause.ctx["errors"]] == [
        ("greater_than_equal", "Input should be greater than or equal to 5")
    ]


def test_an_int_arriving_on_a_float_input_arrives_a_float():
    @node("half", title="Half", description="For a test.")
    def half(x: float) -> str:
        return type(x).__name__

    assert _feeding(3, int, half).results["b"] == {"result": "float"}


class Contacts(TypedDict):
    name: str
    email: str


def test_a_table_input_given_a_table_without_a_column_it_declares_fails_naming_the_column():
    @node("needs_email", title="Needs email", description="For a test.")
    def needs_email(contacts: Table[Contacts]) -> int:
        return contacts.height

    run = _feeding(pl.DataFrame({"name": ["Ida"]}), Table, needs_email)

    assert (run.cause.type, run.cause.loc, run.cause.ctx["input"], run.cause.ctx["column"]) == (
        "invalid_edge_value",
        ("b",),
        "contacts",
        "email",
    )


def test_a_callers_records_are_read_cell_by_cell_as_they_came_not_as_polars_made_them():
    @node(
        "count_contacts", title="Count contacts", description="Counts the rows of a contacts table."
    )
    def count_contacts(contacts: Table[Contacts]) -> int:
        return contacts.height

    graph = Graph.compile(make_graph(count_contacts), registry_of(count_contacts))
    with pytest.raises(InvalidCallError) as refused:
        graph.run(
            {"contacts": [{"name": "Ida", "email": "i@x"}, {"name": 5, "email": "b@x"}]}
        )  # Polars would turn the 5 into text.
    assert refused.value.problems[0].ctx["column"] == "name"


def test_a_table_whose_column_is_of_a_type_its_input_wont_take_is_a_type_mismatch_at_compile():
    """Compiling compares two tables by their columns, so a text column never fits an ``int`` column."""

    class Amounts(TypedDict):
        result: int

    class Texts(TypedDict):
        result: str

    @node("texts", title="Texts", description="For a test.")
    def texts() -> Table[Texts]:
        return pl.DataFrame({"result": ["1", "two"]})

    @node("total", title="Total", description="For a test.")
    def total(rows: Table[Amounts]) -> int:
        return int(rows["result"].sum())

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "a": {"node": "texts"},
                    "b": {
                        "node": "total",
                        "inputs": {"rows": {"sources": [{"node": "a", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "b", "output": "result"}},
            }
        ),
        registry_of(texts, total),
    )

    [problem] = graph.problems
    assert (problem.type, problem.loc, problem.ctx["got"], problem.ctx["expected"]) == (
        "type_mismatch",
        ("b", "rows"),
        "Table[result: str]",
        "Table[result: int]",
    )


def test_a_column_of_an_untyped_table_its_input_wont_take_fails_the_target_as_invalid_edge_value():
    class Amounts(TypedDict):
        result: int

    @node("total", title="Total", description="For a test.")
    def total(rows: Table[Amounts]) -> int:
        return int(rows["result"].sum())

    run = _feeding(pl.DataFrame({"result": ["1", "two"]}), Table, total)

    assert (run.cause.type, run.cause.loc, run.cause.ctx["column"]) == (
        "invalid_edge_value",
        ("b",),
        "result",
    )


# -- a failure says why, and nothing of the value ------------------------------------------


class Cat(BaseModel):
    kind: Literal["cat"]


class Dog(BaseModel):
    kind: Literal["dog"]


@host_type("Pet", title="Pet")
class Pet(BaseModel, frozen=True):
    pet: Annotated[Cat | Dog, Discriminator("kind")]


@host_type("Asserting", title="Asserting")
class Asserting(BaseModel, frozen=True):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        assert value == "ok", f"bad code {value}"
        return value


@host_type("ByPerson", title="By person")
class ByPerson(BaseModel, frozen=True):
    counts: dict[str, int]


@host_type("Bounded", title="Bounded")
class Bounded(BaseModel, frozen=True):
    number: int = Field(ge=5)


@pytest.mark.parametrize(
    ("returned", "return_type"),
    [
        (SECRET, PickyId),
        ({"pet": {"kind": SECRET}}, Pet),
        ({"code": SECRET}, Asserting),
        ({"counts": {SECRET: "many"}}, ByPerson),
    ],
    ids=["own-validator", "a-tag-pydantic-quotes", "an-assert", "a-key-of-a-dict"],
)
def test_what_a_node_returned_is_not_in_the_failure_of_its_output(returned, return_type):
    run = _returning(returned, return_type)

    assert run.cause.type == "invalid_output"
    assert SECRET not in run.cause.model_dump_json() and SECRET not in run.to_json()


def test_the_failure_of_an_output_says_which_check_refused_and_the_limit_it_broke():
    cause = _returning({"number": 1}, Bounded).cause

    assert [(error["type"], error["msg"]) for error in cause.ctx["errors"]] == [
        ("greater_than_equal", "Input should be greater than or equal to 5")
    ]


def test_a_field_of_a_model_is_named_in_the_failure_of_an_output():
    cause = _returning({"counts": {SECRET: "many"}}, ByPerson).cause

    assert [tuple(error["loc"]) for error in cause.ctx["errors"]] == [("counts",)]


def test_a_key_of_a_dict_given_is_never_in_the_errors_and_a_declared_field_or_an_index_is():
    @node("tally", title="Tally", description="For a test.")
    def tally(scores: dict[str, int], people: list[ByPerson]) -> str:
        return "ok"

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"scores": {}, "people": {}},
                "nodes": {
                    "t": {
                        "node": "tally",
                        "inputs": {
                            "scores": {"graph_input": "scores"},
                            "people": {"graph_input": "people"},
                        },
                    }
                },
                "outputs": {"result": {"node": "t", "output": "result"}},
            }
        ),
        registry_of(tally),
    )

    for given, loc in (
        ({"scores": {SECRET: "x"}, "people": []}, ()),
        ({"scores": {}, "people": [{"counts": {}}, {"counts": {SECRET: "x"}}]}, (1, "counts")),
    ):
        with pytest.raises(InvalidCallError) as refused:
            graph.run(given)
        (problem,) = refused.value.problems
        assert SECRET not in problem.model_dump_json() and [
            tuple(e["loc"]) for e in problem.ctx["errors"]
        ] == [loc]


# -- where the check runs ----------------------------------------------------------------------


@host_type("_SlowOutput", title="Slow output")
class _SlowOutput(str):
    """An output type whose check waits until the stream has reported that the other node finished."""

    check_began = Event()
    other_reported = Event()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def check(value: Any) -> Any:
            cls.check_began.set()
            assert cls.other_reported.wait(timeout=10)
            return cls(value)

        return core_schema.no_info_plain_validator_function(check)


def test_a_slow_output_check_runs_on_its_nodes_thread_and_delays_no_other_call():
    """A slow check on one node's output doesn't hold up another node.

    The other node finishes only after the check has started. The check passes only once the stream has reported the
    other node done.
    """

    @node("slow", title="Slow", description="For a test.", policy=Policy(concurrency=2))
    def slow() -> _SlowOutput:
        return "x"  # type: ignore[return-value]  # a plain str, so its check runs

    @node("other", title="Other", description="For a test.", policy=Policy(concurrency=2))
    def other() -> int:
        assert _SlowOutput.check_began.wait(timeout=10)
        return 1

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"s": {"node": "slow"}, "o": {"node": "other"}},
                "outputs": {
                    "o_result": {"node": "o", "output": "result"},
                    "s_result": {"node": "s", "output": "result"},
                },
            }
        ),
        registry_of(slow, other),
    )

    async def go() -> str:
        async for event in graph.with_inputs().stream():
            if event.type == "node_complete" and event.loc[0] == "o":
                _SlowOutput.other_reported.set()
            if event.type == "run_end":
                return event.status
        raise AssertionError("no run_end")

    assert asyncio.run(go()) == "complete"


def test_a_value_already_of_its_type_is_kept_as_returned():
    """A value that is already its type is kept as it is, so an infinite ``float`` stays infinite."""
    assert math.isinf(_returning(math.inf, float).results["g"]["result"])


@pytest.mark.parametrize("return_type", [Any, int | Any, Any | None])
@pytest.mark.parametrize("returned", [1.5, "s", {"k": [1]}, None, 3])
def test_a_node_typed_any_returns_anything_and_a_node_taking_any_receives_it(
    return_type: Any, returned: Any
) -> None:
    @node("take_any", title="Take any", description="For a test.")
    def take_any(value: Any) -> Any:
        return value

    run = _feeding(returned, return_type, take_any)
    assert run.status == "complete"
    assert run.results["b"]["result"] == returned


def test_a_yes_or_no_returned_for_a_literal_of_numbers_is_refused_as_for_a_number():
    for return_type in (int, float, Literal[1, 2]):
        run = _returning(True, return_type)
        assert (run.status, run.cause.type) == ("failed", "invalid_output"), return_type


@node("anything-in", title="Anything in", description="For a test.")
def _anything_in(x: Any) -> Any:
    return x


@node("all-of", title="All of", description="For a test.")
def _all_of(values: Series[Any]) -> int:
    return len(values)


def _giving(returned: Any, return_type: Any) -> Any:
    """Returns a node ``give`` that returns ``returned`` and is declared ``-> return_type``."""

    def give() -> return_type:  # type: ignore[valid-type]
        return returned

    return node("give", title="Give", description="For a test.")(give)


def _fails_with(graph: Any) -> tuple[Any, ...]:
    """Runs ``graph`` and returns how it ended: its status, and its cause's code, place, input and field."""
    run = graph.run()
    cause = run.cause
    return run.status, cause.type, cause.loc, cause.ctx["input"], cause.ctx["field"]


@pytest.mark.parametrize(
    ("declared", "returned"), [(dict, {"other": 1}), (Any, 5), (Any, {"other": 1})]
)
def test_a_field_named_on_an_edge_that_the_value_at_run_time_lacks_fails_the_reader_naming_the_field(
    declared: Any, returned: Any
):
    """An edge may read a field of a record whose type declares no fields. The field is only checked at run time.

    When the field is missing, the node the edge goes into fails with ``invalid_edge_value``.
    """
    give = _giving(returned, declared)
    graph = Graph.compile(
        make_graph(give, _anything_in, edges=[("give.result.name", "anything-in.x")]),
        registry_of(give, _anything_in),
    )
    assert graph.is_runnable
    assert _fails_with(graph) == ("failed", "invalid_edge_value", ("anything-in",), "x", "name")
    assert SECRET not in graph.run().cause.msg and "no field 'name'" in graph.run().cause.msg


def test_a_field_an_item_or_untyped_column_lacks_fails_the_target_at_that_item_or_the_collect_whole():
    items = _giving([{"name": "a"}, {"other": "b"}], Series[dict])
    per_item = Graph.compile(
        make_graph(items, _anything_in, edges=[("give.result.name", "anything-in.x")]),
        registry_of(items, _anything_in),
    )
    assert _fails_with(per_item) == (
        "failed",
        "invalid_edge_value",
        ("anything-in", 1),
        "x",
        "name",
    )
    collected = Graph.compile(
        make_graph(items, _all_of, edges=[("give.result.name", "all-of.values")]),
        registry_of(items, _all_of),
    )
    assert _fails_with(collected) == ("failed", "invalid_edge_value", ("all-of",), "values", "name")
    by_column = _giving(pl.DataFrame({"other": ["a", "b"]}), Series[Any])
    column_read = Graph.compile(
        make_graph(by_column, _anything_in, edges=[("give.result.name", "anything-in.x")]),
        registry_of(by_column, _anything_in),
    )
    assert _fails_with(column_read) == (
        "failed",
        "invalid_edge_value",
        ("anything-in", 0),
        "x",
        "name",
    )
    column_collected = Graph.compile(
        make_graph(by_column, _all_of, edges=[("give.result.name", "all-of.values")]),
        registry_of(by_column, _all_of),
    )
    assert _fails_with(column_collected) == (
        "failed",
        "invalid_edge_value",
        ("all-of",),
        "values",
        "name",
    )


@pytest.mark.parametrize(
    ("returned", "declared", "edge", "into", "loc"),
    [
        (Box(SKIPPED), Box, "give.result.value", "anything-in.x", ("anything-in",)),
        ({"name": SKIPPED}, dict[str, Any], "give.result.name", "anything-in.x", ("anything-in",)),
        (
            [{"name": "a"}, {"name": SKIPPED}],
            Series[dict],
            "give.result.name",
            "anything-in.x",
            ("anything-in", 1),
        ),
        (
            [{"name": "a"}, {"name": SKIPPED}],
            Series[dict],
            "give.result.name",
            "all-of.values",
            ("all-of",),
        ),
    ],
    ids=["a-model-field", "a-dict-key", "a-series-item-field", "a-collected-field"],
)
def test_a_field_an_edge_reads_that_is_skipped_fails_the_reader_instead_of_skipping_it(
    returned: Any, declared: Any, edge: str, into: str, loc: tuple[Any, ...]
) -> None:
    """Only an output's whole value is a skipped branch. A field inside a value that is ``SKIPPED`` has no JSON form,
    so the edge that reads it refuses it, and the node it feeds isn't skipped as if a branch weren't chosen."""
    give = _giving(returned, declared)
    reader = _anything_in if into.startswith("anything-in") else _all_of
    run = Graph.compile(
        make_graph(give, reader, edges=[(edge, into)]), registry_of(give, reader)
    ).run()

    cause = run.cause
    assert (run.status, cause.type, cause.loc, cause.ctx["field"]) == (
        "failed",
        "invalid_edge_value",
        loc,
        edge.rsplit(".", 1)[1],
    )
    assert [error["type"] for error in cause.ctx["errors"]] == ["skipped_inside"]


@dataclass(frozen=True)
class _YesNo:
    yes: str = SKIPPED
    no: str = SKIPPED


@node("decide", title="Decide", description="For a test.")
def _decide(text: str) -> _YesNo:
    return _YesNo(text, SKIPPED)


@pytest.mark.parametrize("values_first", [True, False])
def test_a_skipped_input_skips_the_call_also_when_an_input_before_it_names_a_field_a_record_lacks(
    values_first: bool,
):
    """A skipped input skips the call, even when another input has a fault.

    ``both`` collects ``give.result.name``, and one of the records has no ``name``. Its other input, ``one``, is on
    a skipped branch. The skip wins over the fault, whichever input comes first in the signature. So the call is
    skipped, and the missing field fails nothing.
    """
    items = _giving([{"name": "a"}, {"other": "b"}], Series[dict])
    if values_first:

        @node("both", title="Both", description="For a test.")
        def both(values: Series[Any], one: str) -> int:
            return len(values)
    else:

        @node("both", title="Both", description="For a test.")
        def both(one: str, values: Series[Any]) -> int:
            return len(values)

    edges = [("give.result.name", "both.values"), ("decide.no", "both.one")]
    graph = Graph.compile(
        make_graph(_decide, items, both, edges=edges), registry_of(_decide, items, both)
    )
    run = graph.run({"text": "ok"})
    assert (run.status, json.loads(run.to_json())["outcomes"]["both"]["states"]) == (
        "complete",
        ["skipped"],
    )


def test_the_first_fault_in_the_signature_fails_a_call_that_no_skip_settles():
    """When two inputs read fields their records lack, the call fails on the first in the signature, ``first``."""
    items = _giving([{"name": "a"}, {"other": "b"}], Series[dict])

    @node("two", title="Two", description="For a test.")
    def two(first: Series[Any], second: Series[Any]) -> int:
        return len(first)

    edges = [("give.result.email", "two.first"), ("give.result.name", "two.second")]
    graph = Graph.compile(make_graph(items, two, edges=edges), registry_of(items, two))
    run = graph.run()
    assert (run.status, run.cause.type, run.cause.ctx["input"], run.cause.ctx["field"]) == (
        "failed",
        "invalid_edge_value",
        "first",
        "email",
    )


@node("words", title="Words", description="For a test.")
def _words(text: str) -> Series[str]:
    return text.split()


@node("with-it", title="With it", description="For a test.")
def _with_it(word: str, x: Any) -> str:
    return word


@node("with-them", title="With them", description="For a test.")
def _with_them(word: str, values: Series[Any]) -> str:
    return word


@pytest.mark.parametrize(
    ("returned", "declared", "reader", "input_name"),
    [
        ({"other": 1}, dict, _with_it, "x"),
        ([{"name": "a"}, {"other": "b"}], Series[dict], _with_them, "values"),
        (pl.DataFrame({"other": ["a"]}), Series[Any], _with_them, "values"),
    ],
)
def test_a_run_failed_on_a_field_a_record_lacks_shows_none_for_each_item_never_called(
    returned: Any,
    declared: Any,
    reader: Any,
    input_name: str,
):
    """The failure becomes the run's cause.

    Reading the run afterwards shows ``None`` for each call that read the field.
    """
    give = _giving(returned, declared)
    edges = [
        ("words.result", f"{reader.id}.word"),
        ("give.result.name", f"{reader.id}.{input_name}"),
        (f"{reader.id}.result", "all-of.values"),
    ]
    graph = Graph.compile(
        make_graph(_words, give, reader, _all_of, edges=edges),
        registry_of(_words, give, reader, _all_of),
    )
    run = graph.run({"text": "a b"})
    assert (run.status, run.cause.type, run.cause.ctx["field"]) == (
        "failed",
        "invalid_edge_value",
        "name",
    )
    for read in (run, Run.from_json(run.to_json(), graph)):
        assert read.results["words"] == {"result": ["a", "b"]}
        assert read.results[reader.id] == {"result": [None, None]} and read.results["all-of"] == {
            "result": None
        }


T = TypeVar("T")


@node("text-a", title="Text a", description="For a test.")
def _text_a() -> str:
    return "a"


@node("same", title="Same", description="For a test.")
def _same(value: T = 3) -> T:  # type: ignore[assignment]  # the type an edge binds may refuse this default
    return value


@node("first-of", title="First of", description="For a test.")
def _first_of(first: T, second: T = 3) -> T:  # type: ignore[assignment]
    return first


def test_a_declared_default_is_read_by_the_type_the_edges_bind_and_one_it_refuses_is_a_problem_not_an_exception():
    """An edge that binds a type variable checks the default against the bound type.

    A default the bound type refuses is a problem of the graph. It isn't a ``NodeDeclarationError``.
    """
    registry = registry_of(_text_a, _same, _first_of)
    wired = Graph.compile(
        make_graph(_text_a, _same, edges=[("text-a.result", "same.value")]), registry
    )
    assert wired.problems == [] and wired.run().outputs == {
        "result": "a"
    }  # An edge gives the value, so the default is never used.
    unwired = Graph.compile(
        make_graph(_text_a, _first_of, edges=[("text-a.result", "first-of.first")]), registry
    )
    assert [(found.type, found.loc, found.ctx["type"]) for found in unwired.problems] == [
        ("invalid_value_bound_default", ("first-of", "second"), "str"),
    ]


@node("first-of-whole", title="First of whole", description="For a test.")
def _first_of_whole(first: T, second: T = 5.0) -> T:  # type: ignore[assignment]
    return first


@node("whole-3", title="Whole 3", description="For a test.")
def _whole_3() -> int:
    return 3


def test_a_declared_default_is_read_strictly_by_the_type_the_edges_bind():
    """``5.0`` fits ``T`` alone, but once an edge binds ``T`` to ``int`` it is read as its JSON is, where ``5.0``
    is no ``int``."""
    graph = Graph.compile(
        make_graph(_whole_3, _first_of_whole, edges=[("whole-3.result", "first-of-whole.first")]),
        registry_of(_whole_3, _first_of_whole),
    )
    assert [(found.type, found.loc) for found in graph.problems] == [
        ("invalid_value_bound_default", ("first-of-whole", "second")),
    ]


@node("same-or-blank", title="Same or blank", description="For a test.")
def _same_or_blank(value: T = None) -> T:  # type: ignore[assignment]  # None fits T until an edge binds T
    return value


@node("first-or-blank", title="First or blank", description="For a test.")
def _first_or_blank(first: T, second: T = None) -> T:  # type: ignore[assignment]
    return first


@dataclass(frozen=True)
class _Kept[V]:
    value: V
    other: V = None  # type: ignore[assignment]


@node("kept", title="Kept", description="For a test.")
def _kept(value: T) -> _Kept[T]:
    return _Kept(value)


def test_a_declared_default_of_none_the_bound_type_does_not_take_is_a_problem_of_the_graph_not_an_exception():
    """A default of ``None`` is checked by the bound type like any other default, and compiling never raises for it."""
    registry = registry_of(_text_a, _same_or_blank, _first_or_blank, _kept)
    wired = Graph.compile(
        make_graph(_text_a, _same_or_blank, edges=[("text-a.result", "same-or-blank.value")]),
        registry,
    )
    assert wired.problems == [] and wired.run().outputs == {"result": "a"}
    unwired = Graph.compile(
        make_graph(_text_a, _first_or_blank, edges=[("text-a.result", "first-or-blank.first")]),
        registry,
    )
    output = Graph.compile(
        make_graph(_text_a, _kept, edges=[("text-a.result", "kept.value")]), registry
    )
    assert [
        (found.type, found.loc, found.ctx["type"])
        for found in [*unwired.problems, *output.problems]
    ] == [
        ("invalid_value_bound_default", ("first-or-blank", "second"), "str"),
        ("output_default_bound_invalid", ("kept", "other"), "str"),
    ]
    (refused,) = output.problems
    assert refused.ctx["output"] == "other" and "input" not in refused.ctx
    assert refused.msg.endswith("Change its default, or declare it with no default.")


@node("lines", title="Lines", description="For a test.")
def _lines(item: str) -> Table[Line]:
    return pl.DataFrame({"item": [item], "amount": [1]})


@node("first-line", title="First line", description="For a test.")
def _first_line(one: T, other: T = [{"item": "x"}]) -> T:  # type: ignore[assignment]
    return one


@dataclass(frozen=True)
class _KeptLines[V]:
    value: V
    other: V = field(default_factory=lambda: [{"item": "x"}])  # type: ignore[assignment]


@node("kept-lines", title="Kept lines", description="For a test.")
def _kept_lines(value: T) -> _KeptLines[T]:
    return _KeptLines(value)


def test_a_bound_default_that_is_a_table_missing_a_column_names_the_column():
    """A default the edges make a ``Table[R]`` is refused for a column it lacks, and the problem names that column."""
    registry = registry_of(_lines, _first_line, _kept_lines)
    found = [
        problem
        for second, into in ((_first_line, "first-line.one"), (_kept_lines, "kept-lines.value"))
        for problem in Graph.compile(
            make_graph(_lines, second, edges=[("lines.result", into)]), registry
        ).problems
    ]
    assert [(p.type, p.loc, p.ctx["column"]) for p in found] == [
        ("invalid_value_bound_default", ("first-line", "other"), "amount"),
        ("output_default_bound_invalid", ("kept-lines", "other"), "amount"),
    ]


# -- what a caller gives --------------------------------------------------------------------------------------


@host_type("Celsius", title="Celsius")
@dataclass(frozen=True)
class Celsius:
    degrees: float


@host_type("Label", title="Label")
class Label:
    """A host's class with its own schema: read from its text, written as its text."""

    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
        def read(value: Any) -> "Label":
            if isinstance(value, Label):
                return value
            if isinstance(value, str):
                return Label(value)
            raise ValueError("not a label")

        return core_schema.no_info_plain_validator_function(
            read,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda held: held.text),
        )


@host_type("Socket", title="Socket")
class Socket:
    """An object held in memory, which pydantic checks with isinstance and can't write as JSON."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls)


class Priced(TypedDict):
    name: str
    amount: int


@host_type("Gauge", title="Gauge")
@dataclass(frozen=True)
class Gauge:
    level: float
    label: str


@host_type("Dial", title="Dial")
class Dial(BaseModel, frozen=True):
    level: float
    label: str


@host_type("Meter", title="Meter")
class Meter(BaseModel, frozen=True):
    """A host's model whose own config writes ``inf`` as ``null``, pydantic's default, into an optional field."""

    level: float | None = None


@host_type("Strict", title="Strict")
class Strict:
    """A host's class whose own check raises a ``TypeError`` for what it can't read, as ``float(None)`` does."""

    def __init__(self, level: Any) -> None:
        self.level = float(level)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, Strict) else Strict(value),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda held: held.level),
        )


#: The doors a value enters a graph by, and the problem each gives a value its type refuses.
_DOORS = {"input": "invalid_input", "constant": "invalid_value", "default": "invalid_value"}


def _given(value: Any, input_type: Any, door: str = "input") -> Any:
    """Gives ``value`` to the one input, ``value``, of a graph of one node, ``take``, that takes ``input_type``,
    and returns what the node received.

    ``door`` says how: as the graph input a caller gives, or as a constant or a graph input's default that
    ``make_graph`` writes into the record. A value refused raises ``InvalidCallError``, a compile problem too.
    """
    received: list[Any] = []

    def take(value: input_type) -> str:  # type: ignore[valid-type]
        received.append(value)
        return "ok"

    definition = node("take", title="Take", description="For a test.")(take)
    if door == "input":
        Graph.compile(make_graph(definition), registry_of(definition)).run({"value": value})
    else:
        graph = Graph.compile(
            make_graph(definition, **{f"{door}s": {"take.value": value}}), registry_of(definition)
        )
        if graph.problems:
            raise InvalidCallError(graph.problems)
        graph.run()
    (taken,) = received
    return taken


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize(
    ("given", "input_type"),
    [
        ("5", int),
        (5.0, int),
        ("5.5", float),
        ("yes", bool),
        (1, bool),
        (["1"], list[int]),
        ({"name": "a", "amount": "1"}, Priced),
        ({"degrees": "1"}, Celsius),
        (datetime.datetime(2026, 1, 2), datetime.date),
        (1767312000, datetime.datetime),
        ("Infinity", int),
        ("Infinity", float),
        ("inf", float),
    ],
    ids=[
        "text-for-int",
        "float-for-int",
        "text-for-float",
        "word-for-bool",
        "number-for-bool",
        "text-in-a-list",
        "text-in-a-record",
        "text-in-a-dataclass",
        "datetime-for-date",
        "number-for-datetime",
        "infinity-for-int",
        "infinity-for-float",
        "inf-for-float",
    ],
)
def test_a_value_is_read_strictly_as_its_json_by_every_door_so_a_lax_conversion_is_refused(
    given: Any, input_type: Any, door: str
):
    with pytest.raises(InvalidCallError) as refused:
        _given(given, input_type, door)

    (problem,) = refused.value.problems
    assert (problem.type, problem.loc) == (
        _DOORS[door],
        ("inputs", "value") if door == "input" else ("take", "value"),
    )


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize(
    ("given", "input_type", "received"),
    [
        (5, float, 5.0),
        ("2026-01-02", datetime.date, datetime.date(2026, 1, 2)),
        ("2026-01-02T03:04:00", datetime.datetime, datetime.datetime(2026, 1, 2, 3, 4)),
        ({"degrees": 1}, Celsius, Celsius(1.0)),
        ((1, 2), list[int], [1, 2]),
        ({"a": datetime.date(2026, 1, 2)}, Any, {"a": "2026-01-02"}),
        (None, str | None, None),
        ([{"name": "a", "amount": 1}], list[Priced], [{"name": "a", "amount": 1}]),
        ("Infinity", float | str, "Infinity"),
        ({"level": 1.5, "label": "NaN"}, Gauge, Gauge(1.5, "NaN")),
    ],
    ids=[
        "int-for-float",
        "iso-date",
        "iso-datetime",
        "dict-for-dataclass",
        "tuple-for-list",
        "any",
        "none",
        "records",
        "a-text-a-float-is-written-as-is-text",
        "a-text-a-float-is-written-as-in-a-dataclass",
    ],
)
def test_a_value_is_read_as_its_json_would_be_by_every_door(
    given: Any, input_type: Any, received: Any, door: str
):
    assert _given(given, input_type, door) == received


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize("given", [{"1": "a"}, {1: "a"}], ids=["text-key", "int-key"])
def test_a_dict_with_number_keys_reads_each_key_from_its_text_by_every_door(given: Any, door: str):
    """A JSON object's keys are text, so ``dict[int, str]`` is given ``{"1": "a"}``, and ``{1: "a"}`` is written so."""
    assert _given(given, dict[int, str], door) == {1: "a"}


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize(
    "given",
    [{math.inf: "a"}, {"Infinity": "a"}, {"NaN": "a"}, {"inf": "a"}],
    ids=["inf", "infinity-text", "nan-text", "inf-text"],
)
def test_a_number_key_that_is_not_finite_is_refused_by_every_door(given: Any, door: str):
    """A key is text in the JSON, so only the key the type reads shows it isn't finite."""
    with pytest.raises(InvalidCallError) as refused:
        _given(given, dict[float, str], door)

    (problem,) = refused.value.problems
    assert problem.ctx["errors"][0]["type"] == "finite_number"


@pytest.mark.parametrize("door", _DOORS)
def test_a_yes_or_no_is_not_a_number_key(door: str):
    with pytest.raises(InvalidCallError) as refused:
        _given({True: "a"}, dict[int, str], door)

    (problem,) = refused.value.problems
    assert problem.ctx["errors"][0]["type"] == "int_parsing"


class Levels(TypedDict):
    level: float


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize(
    ("given", "input_type"),
    [
        (math.inf, float),
        (math.nan, float),
        (-math.inf, Any),
        ({"a": [math.nan]}, Any),
        ([1.0, math.inf], list[float]),
        ([math.nan], Series[float]),
        ({"level": math.inf}, Levels),
        (Gauge(math.inf, "a"), Gauge),
        ({"level": math.nan, "label": "a"}, Dial),
        (Dial(level=math.inf, label="a"), Dial),
        (Meter(level=math.inf), Meter),
        ({"dial": Dial(level=-math.inf, label="a")}, Any),
        ((math.inf, "a"), tuple[float, str]),
        ([{"level": math.nan}], Table[Levels]),
    ],
    ids=[
        "inf",
        "nan",
        "any",
        "deep-in-any",
        "list",
        "series",
        "typed-dict",
        "host-dataclass",
        "host-model",
        "host-model-itself",
        "host-model-optional-field",
        "host-model-in-any",
        "tuple",
        "table-records",
    ],
)
def test_a_number_that_enters_must_be_finite_at_any_depth_by_every_door(
    given: Any, input_type: Any, door: str
):
    with pytest.raises(InvalidCallError) as refused:
        _given(given, input_type, door)

    (problem,) = refused.value.problems
    assert (problem.type, [error["type"] for error in problem.ctx["errors"]]) == (
        _DOORS[door],
        ["finite_number"],
    )


def _greater_than_seen(value: str) -> str:
    """A host's check that raises pydantic's own ``greater_than`` with the value in a ``ctx`` key its message never
    reads."""
    raise PydanticCustomError(
        "greater_than", "Input should be greater than {gt}", {"gt": 0, "seen": value}
    )


@pytest.mark.parametrize(
    ("given", "input_type", "error"),
    [
        (
            2,
            Annotated[int, Param(ge=3)],
            {
                "type": "greater_than_equal",
                "loc": [],
                "msg": "Input should be greater than or equal to 3",
                "ctx": {"ge": 3},
            },
        ),
        (
            [1, "x"],
            list[Annotated[str, Param(max_length=2)]],
            {"type": "string_type", "loc": [0], "msg": "Input should be a valid string"},
        ),
        (
            {"level": "high"},
            Levels,
            {"type": "float_type", "loc": ["level"], "msg": "Input should be a valid number"},
        ),
        (
            {"cpr-0102031234": "x"},
            dict[str, int],
            {"type": "int_type", "loc": [], "msg": "Input should be a valid integer"},
        ),
        (
            "x",
            Annotated[str, AfterValidator(_greater_than_seen)],
            {
                "type": "greater_than",
                "loc": [],
                "msg": "Input should be greater than 0",
                "ctx": {"gt": 0},
            },
        ),
    ],
    ids=["limit-ctx", "list-index", "record-field", "dict-key-not-kept", "host-ctx-key-not-read"],
)
def test_each_error_of_a_refusal_keeps_its_type_ctx_and_loc_within_the_value(
    given: Any, input_type: Any, error: dict[str, Any]
):
    """The ``ctx`` of a type's own error, like ``{"ge": 3}``, is the type's, so a host can write its own message by
    the code. A list's index and a record's field are steps the type declares; a dict's key is part of the value, so
    the ``loc`` stops there."""
    with pytest.raises(InvalidCallError) as refused:
        _given(given, input_type)

    (problem,) = refused.value.problems
    assert problem.ctx["errors"] == [error]


@pytest.mark.parametrize(
    "frame",
    [pl.DataFrame({"level": [1.0, math.nan]}), pl.DataFrame({"level": [math.inf]})],
    ids=["nan", "inf"],
)
def test_a_frame_given_for_a_table_with_a_float_cell_not_finite_is_refused(frame: pl.DataFrame):
    with pytest.raises(InvalidCallError) as refused:
        _given(frame, Table[Levels])

    (problem,) = refused.value.problems
    assert ([error["type"] for error in problem.ctx["errors"]], problem.ctx["column"]) == (
        ["finite_number"],
        "level",
    )


def test_a_node_computes_a_float_that_is_not_finite_and_the_run_keeps_it():
    assert math.isnan(_returning(math.nan, float).results["g"]["result"])
    assert math.isinf(_returning(-math.inf, float).results["g"]["result"])


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize("given", [None, {"level": 1}], ids=["none", "mapping"])
def test_a_host_check_that_raises_a_type_error_is_a_refusal_never_an_exception(
    given: Any, door: str
):
    with pytest.raises(InvalidCallError) as refused:
        _given(given, Strict, door)

    (problem,) = refused.value.problems
    assert (problem.type, [error["type"] for error in problem.ctx["errors"]]) == (
        _DOORS[door],
        ["value_error"],
    )


def _refuses_with_conductors_error(value: int) -> int:
    """A host's check that calls conductor and lets its ``InvalidCallError`` out, with a value in its message and
    ``ctx``."""
    raise InvalidCallError(
        [
            Problem(
                type="invalid_input", loc=("inputs", "pin"), msg=f"pin {value}", ctx={"pin": value}
            )
        ]
    )


@pytest.mark.parametrize("door", _DOORS)
def test_a_host_check_that_raises_conductors_own_error_is_refused_as_any_check_is(door: str):
    """An ``InvalidCallError`` is also a ``ValidationError``, which pydantic folds into its own errors when a check
    raises it. So the door still refuses the value with its own problem, at its own place. The host's problem keeps
    only its code, as any error a host's check raises does: its message and ``ctx`` quote the value."""
    with pytest.raises(InvalidCallError) as refused:
        _given(4711, Annotated[int, AfterValidator(_refuses_with_conductors_error)], door)

    (problem,) = refused.value.problems
    assert (problem.type, problem.loc[-1], problem.ctx["errors"]) == (
        _DOORS[door],
        "value",
        [
            {
                "type": "invalid_input",
                "loc": [],
                "msg": "The value did not pass its type's own check",
            }
        ],
    )
    assert "4711" not in str(refused.value)


def test_a_list_given_to_a_host_check_that_raises_a_type_error_is_a_list_for_one_value():
    with pytest.raises(InvalidCallError) as refused:
        _given([1], Strict)

    assert refused.value.problems[0].type == "invalid_input_list_for_one_value"


@pytest.mark.parametrize("door", _DOORS)
def test_a_host_class_given_as_itself_is_written_by_its_own_schema_and_read_back(door: str):
    taken = _given(Label("urgent"), Label, door)

    assert (type(taken), taken.text) == (Label, "urgent")


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize("given", [Socket(), object()], ids=["host-object", "any-object"])
def test_a_value_json_cannot_write_is_refused_where_it_is_given(given: Any, door: str):
    with pytest.raises(InvalidCallError) as refused:
        _given(given, Socket if isinstance(given, Socket) else Any, door)

    (problem,) = refused.value.problems
    assert (problem.type, [error["type"] for error in problem.ctx["errors"]]) == (
        _DOORS[door],
        ["no_json_form"],
    )


@pytest.mark.parametrize("door", _DOORS)
@pytest.mark.parametrize("input_type", [list[str], Series[str], Any], ids=["list", "series", "any"])
def test_a_set_given_is_refused_as_set_has_no_order(input_type: Any, door: str):
    with pytest.raises(InvalidCallError) as refused:
        _given({"a", "b"}, input_type, door)

    (problem,) = refused.value.problems
    assert (problem.type, [error["type"] for error in problem.ctx["errors"]]) == (
        _DOORS[door],
        ["set_has_no_order"],
    )


@host_type("Tagged", title="Tagged")
@dataclass(frozen=True)
class Tagged:
    tags: list[str]


def test_a_set_inside_a_dataclass_given_is_refused_as_set_has_no_order():
    with pytest.raises(InvalidCallError) as refused:
        _given(Tagged(tags={"pear", "apple"}), Tagged)  # type: ignore[arg-type]

    assert refused.value.problems[0].ctx["errors"][0]["type"] == "set_has_no_order"


@pytest.mark.parametrize(
    "given",
    [pl.DataFrame({"a": [1]}), {"t": pl.DataFrame({"a": [1]})}, [pl.DataFrame({"a": [1]})]],
    ids=["frame", "in-a-dict", "in-a-list"],
)
def test_a_frame_given_where_the_type_is_no_table_is_refused_wherever_it_sits(given: Any):
    with pytest.raises(InvalidCallError) as refused:
        _given(given, Any)

    (problem,) = refused.value.problems
    assert (problem.type, [error["type"] for error in problem.ctx["errors"]]) == (
        "invalid_input",
        ["no_json_form"],
    )


def test_a_frame_given_for_a_table_keeps_only_its_columns_so_one_a_saved_run_cant_keep_is_dropped():
    frame = pl.DataFrame(
        {
            "name": ["a"],
            "amount": pl.Series([1], dtype=pl.Int32),
            "when": [datetime.datetime(2026, 1, 1)],
        }
    ).with_columns(pl.col("when").dt.replace_time_zone("Europe/Copenhagen"))

    taken = _given(frame, Table[Priced])

    assert (taken.schema, taken.rows()) == ({"name": pl.String, "amount": pl.Int64}, [("a", 1)])


def test_a_wrong_container_given_names_what_was_wanted_in_pydantics_own_words():
    with pytest.raises(InvalidCallError) as refused:
        _given("a", Series[str])

    assert refused.value.problems[0].ctx["reason"] == "Input should be a valid array"


def test_a_value_that_holds_itself_is_refused_as_no_json_form():
    looped: list[Any] = []
    looped.append(looped)

    with pytest.raises(InvalidCallError) as refused:
        _given(looped, Any)

    (problem,) = refused.value.problems
    assert (problem.type, [error["type"] for error in problem.ctx["errors"]]) == (
        "invalid_input",
        ["no_json_form"],
    )


class PlainName(TypedDict):
    name: str


class LowerName(TypedDict):
    __pydantic_config__ = ConfigDict(str_to_lower=True)  # type: ignore[misc]
    name: str


@node("plain-name", title="Plain name", description="For a test.")
def plain_name(k: int) -> PlainName:
    return {"name": "ANN"}


@node("plain-names", title="Plain names", description="For a test.")
def plain_names(k: int) -> Series[PlainName]:
    return [{"name": "ANN"} for _ in range(k)]


@node("lower-name", title="Lower name", description="For a test.")
def lower_name(row: LowerName) -> list[str]:
    return [row["name"]]


@node("lower-series", title="Lower series", description="For a test.")
def lower_series(rows: Series[LowerName]) -> list[str]:
    return [row["name"] for row in rows]


@node("lower-table", title="Lower table", description="For a test.")
def lower_table(table: Table[LowerName]) -> list[str]:
    return table["name"].to_list()


@pytest.mark.parametrize(
    "nodes",
    [(plain_name, lower_name), (plain_names, lower_series), (plain_names, lower_table)],
    ids=["a row", "a series of rows", "rows collected into a table"],
)
def test_a_row_read_by_a_config_of_its_own_reads_each_value_of_another_row_type_by_it(
    nodes,
) -> None:
    graph = Graph.compile(make_graph(*nodes), registry_of(*nodes))
    run = graph.run({"k": 1})

    assert (run.status, run.outputs["result"]) == ("complete", ["ann"])


@host_type("lower-who", title="Lower who")
class LowerWho(BaseModel):
    model_config = ConfigDict(
        frozen=True, str_to_lower=True, revalidate_instances="subclass-instances"
    )
    name: str


@host_type("cased-who", title="Cased who")
class CasedWho(LowerWho):
    model_config = ConfigDict(frozen=True, str_to_lower=False)


class CasedRow(TypedDict):
    who: CasedWho


class LowerRow(TypedDict):
    who: LowerWho


@node("cased-who-one", title="Cased who one", description="For a test.")
def cased_who_one(k: int) -> CasedWho:
    return CasedWho(name="ANN")


@node("cased-rows", title="Cased rows", description="For a test.")
def cased_rows(k: int) -> Series[CasedRow]:
    return [{"who": CasedWho(name="ANN")} for _ in range(k)]


@node("lower-who-name", title="Lower who name", description="For a test.")
def lower_who_name(who: LowerWho) -> list[str]:
    return [who.name]


@node("lower-row-names", title="Lower row names", description="For a test.")
def lower_row_names(rows: Series[LowerRow]) -> list[str]:
    return [row["who"].name for row in rows]


@node("lower-table-names", title="Lower table names", description="For a test.")
def lower_table_names(table: Table[LowerRow]) -> list[str]:
    return [row["who"]["name"] for row in table.to_dicts()]


@pytest.mark.parametrize(
    "nodes",
    [
        (cased_who_one, lower_who_name),
        (cased_rows, lower_row_names),
        (cased_rows, lower_table_names),
    ],
    ids=["a model", "a series of rows", "rows collected into a table"],
)
def test_a_subclass_of_a_model_that_revalidates_instances_is_read_by_the_models_own_config(
    nodes,
) -> None:
    graph = Graph.compile(make_graph(*nodes), registry_of(*nodes))
    run = graph.run({"k": 1})

    assert (run.status, run.outputs["result"]) == ("complete", ["ann"])
