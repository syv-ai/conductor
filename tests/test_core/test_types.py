"""Tests that a type means the same thing wherever conductor reads it.

The same type hint can turn up in many places: on an edge, inside a list, under a limit, in a person's answer. Each
test picks one place where a hint could be read two ways, and checks that it is read only one way.
"""

import datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict

import polars as pl
import pytest
from conductor import (
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRegistry,
    Param,
    Run,
    Series,
    Table,
    host_type,
    make_graph,
    node,
)
from conductor.widgets import Widget
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    JsonValue,
    model_validator,
    with_config,
)
from pydantic_core import CoreSchema, core_schema
from typing_extensions import ReadOnly

from tests.test_core.nodes import registry_of


@host_type("text", title="Text")
class Text(str):
    """A host's own text, which says itself how pydantic reads it: as a ``str``, then made a ``Text``."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))


@host_type("money", title="Money")
class Money(BaseModel, frozen=True):
    """A host's own record."""

    amount: float = 0


def _run(definition: Any, *more: Any, edges: list[list[str]] | None = None, **values: Any) -> Any:
    """Runs a small graph to its end, and returns the run.

    ``definition`` is the node ``q``, with ``values`` as its constants. Each of ``more`` is a node named by its id.
    The outputs of ``q`` are the graph's outputs.
    """
    nodes: dict[str, Any] = {
        "q": {
            "node": definition.id,
            "inputs": {name: {"constant": value} for name, value in values.items()},
        },
        **{other.id: {"node": other.id} for other in more},
    }
    for source, target in edges or []:
        (source_node, output), (target_node, name) = source.split("."), target.split(".")
        nodes[target_node]["inputs"][name] = {"sources": [{"node": source_node, "output": output}]}
    outputs = {name: {"node": "q", "output": name} for name in definition.outputs}
    graph = Graph.compile(
        GraphRecord.model_validate({"nodes": nodes, "outputs": outputs}),
        registry_of(definition, *more),
    )
    assert not graph.problems, graph.problems
    return graph.run().resume()


def test_an_answer_to_a_series_of_dates_is_read_from_its_json() -> None:
    @node("days", title="Days", description="For a test.", asks=True)
    def days() -> Series[datetime.date]:
        raise Ask("Which days?")

    run = _run(days)
    assert run.with_answers("q", ["2026-01-01"]).resume().outputs == {
        "result": [datetime.date(2026, 1, 1)]
    }


def test_a_limit_on_a_hosts_text_refuses_a_constant_over_it() -> None:
    @node("short", title="Short", description="For a test.")
    def short(text: Annotated[Text, Param(max_length=3)]) -> str:
        return text

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"q": {"node": "short", "inputs": {"text": {"constant": "too long"}}}},
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(short),
    )
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


def test_a_list_of_a_hosts_text_is_a_type_like_any_other() -> None:
    @node("texts", title="Texts", description="For a test.")
    def texts(items: list[Text]) -> int:
        return sum(type(item) is Text for item in items)

    assert _run(texts, items=["a", "b"]).outputs == {"result": 2}


def test_a_model_output_cannot_be_wired_into_a_json_value_input() -> None:
    @node("money", title="Money", description="For a test.")
    def money() -> Money:
        return Money()

    @node("json", title="JSON", description="For a test.")
    def json(value: JsonValue) -> str:
        return str(value)

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "m": {"node": "money"},
                    "j": {
                        "node": "json",
                        "inputs": {"value": {"sources": [{"node": "m", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "j", "output": "result"}},
            }
        ),
        registry_of(money, json),
    )
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


def test_a_hosts_own_method_named_accepts_is_nothing_to_conductor() -> None:
    @host_type("offer", title="Offer")
    class Offer(BaseModel, frozen=True):
        price: float = 0

        def accepts(self, counter: float) -> bool:
            return counter >= self.price

    @node("money", title="Money", description="For a test.")
    def money() -> Money:
        return Money()

    @node("take", title="Take", description="For a test.")
    def take(offer: Offer) -> Offer:
        return offer

    registry = NodeRegistry()
    registry.add(money, take)
    registry.describe()
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "m": {"node": "money"},
                    "t": {
                        "node": "take",
                        "inputs": {"offer": {"sources": [{"node": "m", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "t", "output": "result"}},
            }
        ),
        registry,
    )
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


def test_a_bug_in_a_hosts_schema_hook_is_raised_as_itself() -> None:
    with pytest.raises(AttributeError, match="no_such_helper"):

        @host_type("buggy", title="Buggy")
        class Buggy:
            @classmethod
            def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
                return cls.no_such_helper()  # type: ignore[attr-defined]


def test_a_widget_with_no_json_form_is_a_declaration_problem_naming_the_field() -> None:
    class Catalogue:
        pass

    class Picker(Widget):
        type: Literal["picker"] = "picker"
        source: Any = None

    def pick(value: Annotated[str, Param(widget=Picker(source=Catalogue()))]) -> str:
        return value

    with pytest.raises(
        NodeDeclarationError,
        match=r"^pick \(version 1\)\.value: the widget Picker has a value that can't be written as JSON",
    ) as raised:
        node("pick", title="Pick", description="For a test.")(pick)
    assert [(p.type, p.loc, p.ctx["widget"]) for p in raised.value.problems] == [
        ("widget_without_json_form", ("pick", "value"), "Picker")
    ]


def test_a_yes_or_no_is_no_number_inside_a_list_either() -> None:
    @node("total", title="Total", description="For a test.")
    def total(numbers: list[int]) -> int:
        return sum(numbers)

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"q": {"node": "total", "inputs": {"numbers": {"constant": [True]}}}},
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(total),
    )
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


def test_a_list_of_whole_numbers_can_be_wired_into_a_list_of_numbers_but_never_a_list_of_texts() -> (
    None
):
    @node("ints", title="Ints", description="For a test.")
    def ints() -> list[int]:
        return [1, 2]

    @node("mean", title="Mean", description="For a test.")
    def mean(numbers: list[float]) -> float:
        return sum(numbers) / len(numbers)

    @node("words", title="Words", description="For a test.")
    def words(texts: list[str]) -> int:
        return len(texts)

    assert _run(mean, ints, edges=[["ints.result", "q.numbers"]]).outputs == {"result": 1.5}
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "i": {"node": "ints"},
                    "w": {
                        "node": "words",
                        "inputs": {"texts": {"sources": [{"node": "i", "output": "result"}]}},
                    },
                }
            }
        ),
        registry_of(ints, words),
    )
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


@pytest.mark.parametrize(
    "takes",
    [tuple[str, ...], int | list[str], list[str] | None],
    ids=["tuple", "a union with a list", "an optional list"],
)
def test_a_list_into_an_input_that_takes_a_list_is_the_plain_mismatch_as_a_call_agrees(
    takes: Any,
) -> None:
    """An input takes no list only when every option of its type refuses one as a whole. Compiling and a call read
    that one rule: such an edge is plain ``type_mismatch``, never the advice to split the list, and a list given
    to the input as a graph input is never ``invalid_input_list_for_one_value``."""

    @node("ints", title="Ints", description="For a test.")
    def ints() -> list[int]:
        return [1, 2]

    registry = registry_of(ints)

    def taking(x: Any) -> int:
        return 1

    taking.__annotations__ = {"x": takes, "return": int}
    registry.add(node("takes", title="Takes", description="For a test.")(taking))
    wired = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "i": {"node": "ints"},
                    "t": {
                        "node": "takes",
                        "inputs": {"x": {"sources": [{"node": "i", "output": "result"}]}},
                    },
                }
            }
        ),
        registry,
    )
    assert [problem.type for problem in wired.problems] == ["type_mismatch"]
    given = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"t": {"node": "takes", "inputs": {"x": {"graph_input": "x"}}}},
                "outputs": {"result": {"node": "t", "output": "result"}},
            }
        ),
        registry,
    )
    with pytest.raises(InvalidCallError) as refused:
        given.run({"x": [1, 2]})  # refused for its items, not as a list
    assert [p.type for p in refused.value.problems] == ["invalid_input"]


def test_a_pydantic_type_keeps_the_limits_it_has() -> None:
    from annotated_types import Gt
    from pydantic import PositiveInt, constr

    @node("count", title="Count", description="For a test.")
    def count(
        times: PositiveInt, least: Annotated[int, Gt(1)] = 2, word: constr(min_length=1) = "a"
    ) -> str:  # type: ignore[valid-type]
        return word * times * least

    for wrong in ({"times": 0}, {"times": 1, "least": 1}, {"times": 1, "word": ""}):
        graph = Graph.compile(
            GraphRecord.model_validate(
                {
                    "nodes": {
                        "q": {
                            "node": "count",
                            "inputs": {name: {"constant": value} for name, value in wrong.items()},
                        }
                    }
                }
            ),
            registry_of(count),
        )
        assert [problem.type for problem in graph.problems] == ["invalid_value"], wrong
    assert _run(count, times=3).outputs == {"result": "aaaaaa"}


def test_a_pydantic_field_in_a_hint_is_refused_showing_the_param_to_write() -> None:
    from pydantic import Field

    limit = Field(ge=0)
    hints = (
        Annotated[int, limit],
        Annotated[int, limit] | None,
        Annotated[int, Field(title="Times")],
        Annotated[Annotated[int, limit], Param(title="Times")],
        Series[Annotated[int, limit]],
        list[Annotated[int, limit]],
        dict[str, Annotated[int, limit]],
    )
    for hint in hints:
        with pytest.raises(
            NodeDeclarationError,
            match=r"value: the hint uses pydantic's Field\(\.\.\.\).*Annotated\[int, Param\(ge=0\)\]",
        ) as raised:
            node("count", title="Count", description="For a test.")(_returning_its_input(hint))
        assert raised.value.problems[0].type == "pydantic_field_in_hint", hint

    class Row(TypedDict):
        times: Annotated[int, limit]

    for returned in (Annotated[int, limit], Table[Row]):  # an output, and a column of a table
        with pytest.raises(NodeDeclarationError) as raised:
            node("count", title="Count", description="For a test.")(_returning(returned))
        assert raised.value.problems[0].type == "pydantic_field_in_hint", returned


def test_a_field_of_a_bare_typed_dict_is_held_to_the_rules_a_column_of_a_table_row_is() -> None:
    """A hint a node may not write is refused at the same depth in a ``TypedDict`` input as in a table's row."""
    import math

    from annotated_types import LowerCase, Predicate, Timezone, Unit
    from pydantic import Field

    @host_type("tabby", title="Tabby")
    class Tabby(BaseModel, frozen=True):
        kind: Literal["tabby"]

    @host_type("siamese", title="Siamese")
    class Siamese(BaseModel, frozen=True):
        kind: Literal["siamese"]

    faults = {
        "pydantic_field_in_hint": Annotated[int, Field(ge=0)],
        "pydantic_discriminator_in_hint": Annotated[Tabby | Siamese, Field(discriminator="kind")],
        "timezone_in_hint": Annotated[datetime.datetime, Timezone(None)],
        "unit_in_hint": Annotated[float, Unit("m")],
        "predicate_alias_in_hint": Annotated[str, LowerCase],
        "type_not_supported_set": set[int],
    }
    for kind, hint in faults.items():
        record = TypedDict("Record", {"part": hint})  # type: ignore[misc]
        for declared in (record, Table[record], list[record]):  # type: ignore[valid-type]
            with pytest.raises(NodeDeclarationError) as raised:
                node("count", title="Count", description="For a test.")(
                    _returning_its_input(declared)
                )
            assert raised.value.problems[0].type == kind, (kind, declared)
    # a predicate pydantic checks, written as the metadata itself, is taken
    taken = TypedDict("Taken", {"part": Annotated[float, Predicate(math.isfinite)]})  # type: ignore[misc]
    node("count", title="Count", description="For a test.")(_returning_its_input(taken))


def test_a_series_limit_is_for_its_items_inside_a_list_too() -> None:
    """``Annotated[Series[int], Param(ge=0)]`` limits each item wherever the Series is read, so a list of them takes
    the limit and checks it on every item."""

    @node("lists", title="Lists", description="For a test.")
    def lists(groups: list[Annotated[Series[int], Param(ge=0)]]) -> int:
        return len(groups)

    assert _run(lists, groups=[[1, 2], [3]]).outputs == {"result": 2}
    record = make_graph(lists, constants={"lists.groups": [[1, -1]]})
    graph = Graph.compile(record, registry_of(lists))
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


def test_choices_a_converting_check_moves_off_are_refused_when_the_node_is_declared() -> None:
    """A check that converts a value, like one that adds ``"!"``, would make every choice refuse itself, so the
    choices are read through the type's whole check and refused at ``@node``, not at every call."""

    def shouted(value: str) -> str:
        return value + "!"

    for hint in (
        Annotated[str, AfterValidator(shouted), Param(choices=["a"])],
        Annotated[str, Param(choices=["a"]), AfterValidator(shouted)],
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            node("count", title="Count", description="For a test.")(_returning_its_input(hint))
        assert raised.value.problems[0].type == "choices_invalid", hint


def test_choices_a_converting_check_moves_off_are_refused_on_a_series_inside_a_list() -> None:
    """A Series' items are read as one type wherever the Series is, so a converting check's choices are refused in a
    list of Series as on a Series field."""

    def shouted(value: str) -> str:
        return value + "!"

    for hint in (
        Series[Annotated[str, AfterValidator(shouted), Param(choices=["a"])]],
        list[Series[Annotated[str, AfterValidator(shouted), Param(choices=["a"])]]],
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            node("count", title="Count", description="For a test.")(_returning_its_input(hint))
        assert raised.value.problems[0].type == "choices_invalid", hint


def test_a_pydantic_field_that_tells_a_union_apart_is_refused_showing_the_discriminator_to_write() -> (
    None
):
    from pydantic import Discriminator, Field

    @host_type("cat", title="Cat")
    class Cat(BaseModel, frozen=True):
        kind: Literal["cat"] = "cat"

    @host_type("dog", title="Dog")
    class Dog(BaseModel, frozen=True):
        kind: Literal["dog"] = "dog"

    with pytest.raises(
        NodeDeclarationError,
        match=r"value: the hint uses pydantic's Field\(discriminator=\.\.\.\).*Annotated\[Cat \| Dog, Discriminator\(\"kind\"\)\]",
    ) as raised:
        node("pet", title="Pet", description="For a test.")(
            _returning_its_input(Annotated[Cat | Dog, Field(discriminator="kind")])
        )
    assert raised.value.problems[0].type == "pydantic_discriminator_in_hint" and "Param" not in str(
        raised.value
    )

    pet = node("pet", title="Pet", description="For a test.")(
        _returning_its_input(Annotated[Cat | Dog, Discriminator("kind")])
    )
    assert _run(pet, value={"kind": "dog"}).outputs == {"result": "kind='dog'"}


def test_a_pydantic_field_inside_a_hosts_own_model_is_the_models() -> None:
    from pydantic import Field

    @host_type("stock", title="Stock")
    class Stock(BaseModel, frozen=True):
        count: int = Field(ge=0)
        name: Annotated[str, Field(min_length=1)] = "a"

    @node("restock", title="Restock", description="For a test.")
    def restock(stock: Stock) -> int:
        return stock.count

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "q": {"node": "restock", "inputs": {"stock": {"constant": {"count": -1}}}}
                },
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(restock),
    )
    assert [problem.type for problem in graph.problems] == ["invalid_value"]
    assert _run(restock, stock={"count": 2}).outputs == {"result": 2}


def test_a_subclass_of_a_builtin_with_no_schema_of_its_own_is_validated_as_the_builtin_and_made_the_subclass() -> (
    None
):
    @host_type("bare", title="Bare")
    class Bare(str):
        pass

    @node("kinds", title="Kinds", description="For a test.")
    def kinds(one: Bare, many: list[Bare]) -> str:
        return " ".join(type(value).__name__ for value in [one, *many])

    assert _run(kinds, one="x", many=["x"]).outputs == {"result": "Bare Bare"}


# -- a table is a type like any other -------------------------------------------------------------


class Named(TypedDict):
    name: str


class Dated(TypedDict):
    name: str
    born: datetime.date


def test_a_table_read_as_its_row_type_has_exactly_its_columns_also_on_a_nodes_own_return() -> None:
    @node("people", title="People", description="For a test.")
    def people() -> Table[Named]:
        return pl.DataFrame({"name": ["Ann"], "email": ["ann@x.dk"]})

    @node("all", title="All", description="For a test.")
    def everything() -> Table:
        return pl.DataFrame({"name": ["Ann"], "email": ["ann@x.dk"]})

    assert _run(people).results["q"]["result"].columns == ["name"]
    assert _run(everything).results["q"]["result"].columns == ["name", "email"]


def test_a_table_of_no_rows_has_every_declared_column_of_its_declared_type() -> None:
    @node("nobody", title="Nobody", description="For a test.")
    def nobody() -> Table[Dated]:
        return pl.DataFrame()

    @node("count-people", title="Count people", description="For a test.")
    def count_people(people: Table[Dated]) -> int:
        assert people.schema == pl.Schema({"name": pl.String, "born": pl.Date})
        return people.height

    assert _run(nobody).results["q"]["result"].schema == pl.Schema(
        {"name": pl.String, "born": pl.Date}
    )
    assert _run(count_people, people=[]).outputs == {"result": 0}


def test_rows_collected_into_a_table_are_read_once_each_cell_by_its_columns_type() -> None:
    @node("person", title="Person", description="For a test.")
    def person() -> Series[Dated]:
        return [
            {"name": "Ann", "born": datetime.date(1990, 1, 2)},
            {"name": "Bo", "born": datetime.date(1991, 3, 4)},
        ]

    @node("oldest", title="Oldest", description="For a test.")
    def oldest(people: Table[Dated]) -> datetime.date:
        return people["born"].min()

    run = _run(oldest, person, edges=[["person.result", "q.people"]])
    assert run.outputs == {"result": datetime.date(1990, 1, 2)}


def test_a_table_type_works_inside_a_union_and_inside_a_series() -> None:
    @node("size", title="Size", description="For a test.")
    def size(value: Table[Named] | list[str]) -> int:
        return len(value)

    @node("each", title="Each", description="For a test.")
    def each() -> Series[Table[Named]]:
        return [pl.DataFrame({"name": ["Ann"]}), pl.DataFrame({"name": ["Bo", "Cy"]})]

    assert _run(size, value=[{"name": "Ann"}]).outputs == {"result": 1}
    assert _run(size, value=["a", "b"]).outputs == {"result": 2}
    run = _run(size, each, edges=[["each.result", "q.value"]])
    assert run.outputs == {"result": [1, 2]}
    back = Run.from_json(run.to_json(), Graph.compile(run.graph.record, registry_of(size, each)))
    assert [frame.to_dicts() for frame in back.results["each"]["result"]] == [
        [{"name": "Ann"}],
        [{"name": "Bo"}, {"name": "Cy"}],
    ]


def test_a_record_with_a_date_is_no_json_value() -> None:
    @node("dated", title="Dated", description="For a test.")
    def dated() -> Dated:
        return {"name": "Ann", "born": datetime.date(1990, 1, 2)}

    @node("named", title="Named", description="For a test.")
    def named() -> Named:
        return {"name": "Ann"}

    @node("json", title="JSON", description="For a test.")
    def json(value: JsonValue) -> str:
        return str(value)

    def problems(source: Any) -> list[str]:
        graph = Graph.compile(
            GraphRecord.model_validate(
                {
                    "nodes": {
                        "s": {"node": source.id},
                        "j": {
                            "node": "json",
                            "inputs": {"value": {"sources": [{"node": "s", "output": "result"}]}},
                        },
                    }
                }
            ),
            registry_of(source, json),
        )
        return [problem.type for problem in graph.problems]

    assert problems(dated) == ["type_mismatch"] and problems(named) == []


def test_a_models_class_variables_are_not_its_fields() -> None:
    @host_type("address", title="Address")
    class Address(BaseModel, frozen=True):
        city: str = "Aarhus"

    @node("address", title="Address", description="For a test.")
    def address() -> Address:
        return Address()

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    def compiled(field: str) -> Any:
        return Graph.compile(
            GraphRecord.model_validate(
                {
                    "nodes": {
                        "a": {"node": "address"},
                        "s": {
                            "node": "shout",
                            "inputs": {
                                "text": {
                                    "sources": [{"node": "a", "output": "result", "field": field}]
                                }
                            },
                        },
                    },
                    "outputs": {"result": {"node": "s", "output": "result"}},
                }
            ),
            registry_of(address, shout),
        )

    assert compiled("city").run().outputs == {"result": "AARHUS"}
    for wrong in ("town", "type_id"):
        (problem,) = compiled(wrong).problems
        assert (problem.type, problem.ctx["outputs"]) == ("unknown_output", ["a.result.city"])


def test_a_host_type_names_the_other_classes_it_takes() -> None:
    @host_type("celsius", title="Celsius")
    class Celsius(BaseModel, frozen=True):
        degrees: float

    @host_type("kelvin", title="Kelvin", accepts=[Celsius])
    class Kelvin(BaseModel, frozen=True):
        degrees: float

        @model_validator(mode="before")
        @classmethod
        def _from_celsius(cls, value: Any) -> Any:
            return {"degrees": value.degrees + 273.15} if isinstance(value, Celsius) else value

    @node("outside", title="Outside", description="For a test.")
    def outside() -> Celsius:
        return Celsius(degrees=20)

    @node("absolute", title="Absolute", description="For a test.")
    def absolute(temperature: Kelvin) -> float:
        return temperature.degrees

    @node("warm", title="Warm", description="For a test.")
    def warm(temperature: Celsius) -> bool:
        return temperature.degrees > 15

    assert _run(absolute, outside, edges=[["outside.result", "q.temperature"]]).outputs == {
        "result": 293.15
    }
    registry = registry_of(outside, absolute, warm)
    assert {described.id: described.accepted_as for described in registry.describe().types}[
        "celsius"
    ] == ["celsius", "kelvin"]


def test_a_widget_in_annotated_outside_param_is_refused_saying_where_it_goes() -> None:
    from conductor.widgets import Textarea

    for hint in (
        Annotated[str, Textarea()],
        Series[Annotated[str, Textarea()]],
        Annotated[str, Textarea()] | None,
    ):
        with pytest.raises(NodeDeclarationError, match=r"Param\(widget=") as raised:
            node("note", title="Note", description="For a test.")(_returning_its_input(hint))
        assert raised.value.problems[0].type == "widget_outside_param"

    @node("note", title="Note", description="For a test.")
    def note(text: Annotated[str, Param(widget=Textarea())]) -> str:
        return text

    assert note.describe().inputs[0].widget["type"] == "textarea"


def _returning(hint: Any) -> Any:
    def passes(value: int):  # type: ignore[no-untyped-def]
        return value

    passes.__annotations__["return"] = hint
    return passes


def _returning_its_input(hint: Any) -> Any:
    def passes(value: Any) -> str:
        return str(value)

    passes.__annotations__ = {"value": hint, "return": str}
    return passes


def _shout_checked_by(check: Any, calls: list[str]) -> Graph[None]:
    """Returns a graph of one node, ``shout``, whose input is checked by ``check``.

    Each call of ``shout`` appends its text to ``calls``.
    """

    @node("shout", title="Shout", description="For a test.")
    def shout(text: Annotated[str, AfterValidator(check)]) -> str:
        calls.append(text)
        return text.upper()

    return Graph.compile(make_graph(shout), registry_of(shout))


def test_a_function_in_a_hint_is_told_apart_by_where_it_is_written() -> None:
    """Two functions of the same name in a hint are told apart by where they are written.

    With the same function in its hint, a node reuses its previous call. Declared again with another function of the
    same name, it is another node, so its call is made again.
    """

    def local(text: str) -> str:
        return text

    def another_of_the_same_name() -> Any:
        def local(text: str) -> str:
            return text

        return local

    calls: list[str] = []
    first = _shout_checked_by(local, calls).run({"text": "a"})
    _shout_checked_by(local, calls).with_inputs({"text": "a"}, previous=first).resume()
    assert calls == ["a"]
    _shout_checked_by(another_of_the_same_name(), calls).with_inputs(
        {"text": "a"}, previous=first
    ).resume()
    assert calls == ["a", "a"]


_A_NODE_WITH_A_FUNCTION_IN_ITS_HINT = """
import sys
from typing import Annotated

from conductor import Graph, NodeRegistry, Run, make_graph
from pydantic import AfterValidator

calls = []
padding = [object() for _ in range(int(sys.argv[2]))]  # so the function below is at another address in each process


def stripped(text: str) -> str:
    return text.strip()


class Marked:  # no repr of its own: Python writes it with its memory address
    pass


registry = NodeRegistry()


@registry.node("shout", title="Shout", description="For a test.")
def shout(
    text: Annotated[str, AfterValidator(stripped), AfterValidator(lambda text: text), Marked()],
) -> str:
    calls.append(text)
    return text.upper()


graph = Graph.compile(make_graph(shout), registry)
if sys.argv[2] == "0":
    open(sys.argv[1], "w").write(graph.run({"text": " a "}).to_json())
else:
    again = Run.from_json(open(sys.argv[1]).read(), graph).again({"text": " a "})
    print(again.outputs, "calls:", len(calls))
"""


def test_a_run_saved_in_one_process_is_reused_in_another_when_a_hint_has_a_function(
    tmp_path: Any,
) -> None:
    import subprocess
    import sys

    program, saved = tmp_path / "shout.py", tmp_path / "run.json"
    program.write_text(_A_NODE_WITH_A_FUNCTION_IN_ITS_HINT)
    for padding, printed in (("0", ""), ("100000", "{'result': 'A'} calls: 0\n")):
        child = subprocess.run(
            [sys.executable, str(program), str(saved), padding],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout == printed


def test_a_series_is_written_by_its_public_name() -> None:
    assert repr(Series[int]) == "conductor.Series[int]"


class _Pair[T](TypedDict):
    name: str
    value: T


class _PlainPair(TypedDict):
    name: str
    value: int


class _FixedPair(TypedDict):
    name: ReadOnly[str]
    value: int


@with_config(ConfigDict(title="Configured pair"))
class _ConfiguredPair(TypedDict):
    name: str
    value: int


class _PartialPair(TypedDict, total=False):
    name: str
    value: int


class _SomePair(TypedDict):
    name: str
    value: NotRequired[int]


def _declared(name: str, function: Any) -> Any:
    return node(name, title=name.title(), description="For a test.")(function)


def test_a_generic_typed_dict_is_a_row_on_an_input_an_output_a_series_and_a_table() -> None:
    """``Row[int]`` is taken wherever a ``TypedDict`` is, with ``value: T`` read as ``int``.

    It was taken in a ``Table`` but refused on a field with ``type_not_named``, whose fix a ``TypedDict`` can't take.
    """

    def on_input(pair: _Pair[int]) -> int:
        return pair["value"]

    def on_output() -> _Pair[int]:
        return {"name": "ida", "value": 1}

    def in_a_series() -> Series[_Pair[int]]:
        return [{"name": "ida", "value": 1}]

    def in_a_table() -> Table[_Pair[int]]:
        return pl.DataFrame({"name": ["ida"], "value": [1]})

    described = {
        name: _declared(name, function).describe()
        for name, function in [
            ("on-input", on_input),
            ("on-output", on_output),
            ("in-a-series", in_a_series),
            ("in-a-table", in_a_table),
        ]
    }
    assert described["on-input"].inputs[0].type == "row"
    assert described["on-input"].inputs[0].json_schema["properties"]["value"] == {
        "title": "Value",
        "type": "integer",
    }
    assert described["on-output"].outputs[0].type == "row"
    assert described["in-a-series"].outputs[0].type == "row"
    assert described["in-a-series"].outputs[0].is_series
    assert described["in-a-table"].outputs[0].type == "Table[name: str, value: int]"


def test_an_edge_reads_a_field_of_a_generic_typed_dict_by_its_filled_in_type() -> None:
    """An edge from ``make.result.value`` of a ``Row[int]`` carries an ``int``, and one to a field it lacks is refused."""

    @node("make", title="Make", description="For a test.")
    def make() -> _Pair[int]:
        return {"name": "ida", "value": 2}

    @node("double", title="Double", description="For a test.")
    def double(n: int) -> int:
        return n * 2

    registry = registry_of(make, double)
    graph = Graph.compile(
        make_graph(make, double, edges=[("make.result.value", "double.n")]), registry
    )
    assert graph.problems == []
    assert graph.run({}).results["double"]["result"] == 4
    unknown = Graph.compile(
        make_graph(make, double, edges=[("make.result.size", "double.n")]), registry
    )
    assert [problem.type for problem in unknown.problems] == ["unknown_output"]


def test_a_generic_typed_dict_left_unfilled_reads_its_open_field_as_any() -> None:
    """A bare ``Row``, its ``T`` given no argument, is a row whose ``value`` is ``Any`` in a table and on an edge."""

    @node("make", title="Make", description="For a test.")
    def make() -> _Pair:  # type: ignore[type-arg]
        return {"name": "ida", "value": "x"}

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    def in_a_table() -> Table[_Pair]:  # type: ignore[type-arg]
        return pl.DataFrame({"name": ["ida"], "value": [1]})

    assert _declared("unfilled-table", in_a_table).describe().outputs[0].type == (
        "Table[name: str, value: Any]"
    )
    graph = Graph.compile(
        make_graph(make, shout, edges=[("make.result.value", "shout.text")]),
        registry_of(make, shout),
    )
    assert graph.problems == []
    assert graph.run({}).results["shout"]["result"] == "X"


@pytest.mark.parametrize(
    "row",
    [_PlainPair, _Pair[int], _FixedPair, _ConfiguredPair],
    ids=["plain", "generic", "read-only", "with-config"],
)
def test_every_reader_of_a_rows_fields_finds_the_same_fields(row: Any) -> None:
    """A row's fields are the same to its JSON schema, a table's columns, an edge's field and a frame for a ``Series``.

    Each of these once read the fields its own way, and a generic ``TypedDict`` or a ``ReadOnly`` field fell between
    them.
    """

    def take(pair: row) -> int:  # type: ignore[valid-type]
        return 0

    def as_a_table() -> Table[row]:  # type: ignore[valid-type]
        return pl.DataFrame({"name": ["ida"], "value": [1]})

    @node("make", title="Make", description="For a test.")
    def make() -> Series[row]:  # type: ignore[valid-type]
        return pl.DataFrame({"name": ["ida", "bo"], "value": [1, 2]})

    @node("double", title="Double", description="For a test.")
    def double(n: int) -> int:
        return n * 2

    assert set(_declared("take", take).describe().inputs[0].json_schema["properties"]) == {
        "name",
        "value",
    }
    assert _declared("as-a-table", as_a_table).describe().outputs[0].type == (
        "Table[name: str, value: int]"
    )
    graph = Graph.compile(
        make_graph(make, double, edges=[("make.result.value", "double.n")]),
        registry_of(make, double),
    )
    assert graph.problems == []
    assert graph.run({}).results["double"]["result"] == [2, 4]


@pytest.mark.parametrize("row", [_PartialPair, _SomePair], ids=["total-false", "not-required"])
def test_a_row_that_may_leave_a_field_out_is_read_alike_by_an_edge_and_refused_as_a_table(
    row: Any,
) -> None:
    """A ``total=False`` or ``NotRequired`` field is a field to an edge, and no column of a table.

    A row with every field fits such an input, and an edge may read the field. A table refuses the row, since every
    row of a table has every column.
    """

    @node("make", title="Make", description="For a test.")
    def make() -> _PlainPair:
        return {"name": "ida", "value": 3}

    @node("take", title="Take", description="For a test.")
    def take(pair: row) -> int:  # type: ignore[valid-type]
        return pair.get("value", 0)

    @node("give", title="Give", description="For a test.")
    def give() -> row:  # type: ignore[valid-type]
        return {"name": "ida", "value": 3}

    @node("double", title="Double", description="For a test.")
    def double(n: int) -> int:
        return n * 2

    whole = Graph.compile(
        make_graph(make, take, edges=[("make.result", "take.pair")]), registry_of(make, take)
    )
    assert whole.problems == []
    assert whole.run({}).results["take"]["result"] == 3
    picked = Graph.compile(
        make_graph(give, double, edges=[("give.result.value", "double.n")]),
        registry_of(give, double),
    )
    assert picked.problems == []

    def as_a_table() -> Table[row]:  # type: ignore[valid-type]
        return pl.DataFrame()

    with pytest.raises(NodeDeclarationError) as refused:
        _declared("as-a-table", as_a_table)
    assert refused.value.problems[0].type == "row_field_optional"


class _Named(TypedDict):
    name: str


@pytest.mark.parametrize("row", [_PartialPair, _SomePair], ids=["total-false", "not-required"])
def test_a_row_without_a_field_its_input_may_do_without_fits_it(row: Any) -> None:
    """A row with only ``name`` fits an input whose ``value`` may be left out, since the input's check takes it."""

    @node("make", title="Make", description="For a test.")
    def make() -> _Named:
        return {"name": "ida"}

    @node("take", title="Take", description="For a test.")
    def take(pair: row) -> int:  # type: ignore[valid-type]
        return pair.get("value", 0)

    graph = Graph.compile(
        make_graph(make, take, edges=[("make.result", "take.pair")]), registry_of(make, take)
    )
    assert graph.problems == []
    assert graph.run({}).results["take"]["result"] == 0
    # the other way round, a row that may leave out a field the input requires doesn't fit it

    @node("give", title="Give", description="For a test.")
    def give() -> row:  # type: ignore[valid-type]
        return {"name": "ida"}

    @node("need", title="Need", description="For a test.")
    def need(pair: _PlainPair) -> int:
        return pair["value"]

    backwards = Graph.compile(
        make_graph(give, need, edges=[("give.result", "need.pair")]), registry_of(give, need)
    )
    assert [problem.type for problem in backwards.problems] == ["type_mismatch"]


def test_an_edge_reads_a_field_of_a_generic_pydantic_model_by_its_filled_in_type() -> None:
    """``Box[int]``'s field ``value: T`` is an ``int`` to an edge, as a generic ``TypedDict``'s is.

    Pydantic makes a class of its own for ``Box[int]``, so its arguments are read from what pydantic keeps of them.
    """

    @host_type("box-nfr", title="Box")
    class Box[T](BaseModel, frozen=True):
        value: Annotated[T, Param(title="Value")]

    @node("make", title="Make", description="For a test.")
    def make() -> Box[int]:
        return Box[int](value=2)

    @node("double", title="Double", description="For a test.")
    def double(n: int) -> int:
        return n * 2

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    registry = registry_of(make, double, shout)
    taken = Graph.compile(
        make_graph(make, double, edges=[("make.result.value", "double.n")]), registry
    )
    assert taken.problems == []
    assert taken.run({}).results["double"]["result"] == 4
    refused = Graph.compile(
        make_graph(make, shout, edges=[("make.result.value", "shout.text")]), registry
    )
    assert [problem.type for problem in refused.problems] == ["type_mismatch"]
    unknown = Graph.compile(
        make_graph(make, double, edges=[("make.result.size", "double.n")]), registry
    )
    assert unknown.problems[0].ctx["outputs"] == ["make.result.value"]
