"""A type means the same wherever it is read: on an edge, in a list, under a limit, in an answer.

Each test names one place where reading an annotation a second way gave a second answer.
"""

import datetime
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import Ask, NodeDeclarationError, NodeRegistry, Param, Series, Table, host_type, node
from conductor.widgets import Widget
from pydantic import BaseModel, GetCoreSchemaHandler, JsonValue, model_validator
from pydantic_core import CoreSchema, PydanticSerializationError, core_schema

from tests.test_core.nodes import registry_of


@host_type("text", title="Text")
class Text(str):
    """A host's own text, which says itself how pydantic reads it: as a ``str``, then made a ``Text``."""


    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))


@host_type("money", title="Money")
class Money(BaseModel):
    """A host's own record."""

    amount: float = 0


def _run(definition: Any, *more: Any, edges: list[list[str]] | None = None, **values: Any) -> Any:
    """``definition`` as node ``q`` with ``values`` typed in, and each of ``more`` as a node named by its id; the run, resumed."""
    nodes = [{"id": "q", "node": definition.id, "values": values}, *({"id": other.id, "node": other.id} for other in more)]
    graph = registry_of(definition, *more).load({"nodes": nodes, "edges": edges or []})
    assert not graph.problems, graph.problems
    return graph.run().resume()


def test_an_answer_to_a_series_of_dates_is_read_from_its_json() -> None:
    @node("days", title="Days", description="")
    def days() -> Series[datetime.date]:
        raise Ask("Which days?")

    run = _run(days)
    assert run.answer("q", ["2026-01-01"]).resume().outputs == {"q.result": [datetime.date(2026, 1, 1)]}


def test_a_limit_on_a_hosts_text_is_held() -> None:
    @node("short", title="Short", description="")
    def short(text: Annotated[Text, Param(max_length=3)]) -> str:
        return text

    graph = registry_of(short).load({"nodes": [{"id": "q", "node": "short", "values": {"text": "too long"}}]})
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


def test_a_list_of_a_hosts_text_is_a_type_like_any_other() -> None:
    @node("texts", title="Texts", description="")
    def texts(items: list[Text]) -> int:
        return sum(type(item) is Text for item in items)

    assert _run(texts, items=["a", "b"]).outputs == {"q.result": 2}


def test_a_model_does_not_land_on_a_json_value() -> None:
    @node("money", title="Money", description="")
    def money() -> Money:
        return Money()

    @node("json", title="JSON", description="")
    def json(value: JsonValue) -> str:
        return str(value)

    graph = registry_of(money, json).load({
        "nodes": [{"id": "m", "node": "money"}, {"id": "j", "node": "json"}],
        "edges": [["m.result", "j.value"]],
    })
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


def test_a_hosts_own_method_named_accepts_is_nothing_to_conductor() -> None:
    @host_type("offer", title="Offer")
    class Offer(BaseModel):
        price: float = 0

        def accepts(self, counter: float) -> bool:
            return counter >= self.price

    @node("money", title="Money", description="")
    def money() -> Money:
        return Money()

    @node("take", title="Take", description="")
    def take(offer: Offer) -> Offer:
        return offer

    registry = NodeRegistry()
    registry.add(money, take)
    registry.describe()
    graph = registry.load({"nodes": [{"id": "m", "node": "money"}, {"id": "t", "node": "take"}], "edges": [["m.result", "t.offer"]]})
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


def test_a_bug_in_a_hosts_schema_hook_is_raised_as_itself() -> None:
    with pytest.raises(AttributeError, match="no_such_helper"):

        @host_type("buggy", title="Buggy")
        class Buggy:

            @classmethod
            def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
                return cls.no_such_helper()  # type: ignore[attr-defined]


def test_a_widget_with_no_json_form_fails_where_the_node_is_declared() -> None:
    class Catalogue:
        pass

    class Picker(Widget):
        type: Literal["picker"] = "picker"
        source: Any = None

    with pytest.raises(PydanticSerializationError):

        @node("pick", title="Pick", description="")
        def pick(value: Annotated[str, Param(widget=Picker(source=Catalogue()))]) -> str:
            return value


def test_a_yes_or_no_is_no_number_inside_a_list_either() -> None:
    @node("total", title="Total", description="")
    def total(numbers: list[int]) -> int:
        return sum(numbers)

    graph = registry_of(total).load({"nodes": [{"id": "q", "node": "total", "values": {"numbers": [True]}}]})
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


def test_a_list_of_whole_numbers_lands_on_a_list_of_numbers_and_never_on_a_list_of_texts() -> None:
    @node("ints", title="Ints", description="")
    def ints() -> list[int]:
        return [1, 2]

    @node("mean", title="Mean", description="")
    def mean(numbers: list[float]) -> float:
        return sum(numbers) / len(numbers)

    @node("words", title="Words", description="")
    def words(texts: list[str]) -> int:
        return len(texts)

    assert _run(mean, ints, edges=[["ints.result", "q.numbers"]]).outputs == {"q.result": 1.5}
    graph = registry_of(ints, words).load({
        "nodes": [{"id": "i", "node": "ints"}, {"id": "w", "node": "words"}],
        "edges": [["i.result", "w.texts"]],
    })
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


def test_a_limit_spelled_as_pydantic_spells_it_is_carried() -> None:
    from pydantic import Field, PositiveInt

    @node("count", title="Count", description="")
    def count(times: PositiveInt, least: Annotated[int, Field(ge=2)] = 2) -> int:
        return times * least

    graph = registry_of(count).load({"nodes": [{"id": "q", "node": "count", "values": {"times": 0}}]})
    assert [problem.type for problem in graph.problems] == ["invalid_value"]
    assert _run(count, times=3).outputs == {"q.result": 6}


def test_a_subclass_of_a_builtin_with_no_schema_of_its_own_is_validated_as_the_builtin_and_made_the_subclass() -> None:
    @host_type("bare", title="Bare")
    class Bare(str):
        pass

    from conductor._internal.value_type import Type

    assert type(Type(Bare).read("x")) is Bare and Type(list[Bare]).read(["x"]) == ["x"]


# -- a table is a type like any other -------------------------------------------------------------


class Named(TypedDict):
    name: str


class Dated(TypedDict):
    name: str
    born: datetime.date


def test_a_table_read_as_its_row_type_holds_exactly_its_columns_a_nodes_own_return_included() -> None:
    @node("people", title="People", description="")
    def people() -> Table[Named]:
        return pl.DataFrame({"name": ["Ann"], "email": ["ann@x.dk"]})

    @node("all", title="All", description="")
    def everything() -> Table:
        return pl.DataFrame({"name": ["Ann"], "email": ["ann@x.dk"]})

    assert _run(people).results["q"]["result"].columns == ["name"]
    assert _run(everything).results["q"]["result"].columns == ["name", "email"]


def test_a_table_of_no_rows_holds_every_declared_column_of_its_declared_type() -> None:
    @node("nobody", title="Nobody", description="")
    def nobody() -> Table[Dated]:
        return pl.DataFrame()

    @node("gathered", title="Gathered", description="")
    def gathered(people: Table[Dated]) -> int:
        assert people.schema == pl.Schema({"name": pl.String, "born": pl.Date})
        return people.height

    assert _run(nobody).results["q"]["result"].schema == pl.Schema({"name": pl.String, "born": pl.Date})
    assert _run(gathered, people=[]).outputs == {"q.result": 0}


def test_records_gathered_into_a_table_are_read_once_each_cell_by_its_columns_type() -> None:
    @node("person", title="Person", description="")
    def person() -> Series[Dated]:
        return [{"name": "Ann", "born": datetime.date(1990, 1, 2)}, {"name": "Bo", "born": datetime.date(1991, 3, 4)}]

    @node("oldest", title="Oldest", description="")
    def oldest(people: Table[Dated]) -> datetime.date:
        return people["born"].min()

    run = _run(oldest, person, edges=[["person.result", "q.people"]])
    assert run.outputs == {"q.result": datetime.date(1990, 1, 2)}


def test_a_table_stands_in_a_union_and_in_a_series() -> None:
    @node("size", title="Size", description="")
    def size(value: Table[Named] | list[str]) -> int:
        return len(value)

    @node("each", title="Each", description="")
    def each() -> Series[Table[Named]]:
        return [pl.DataFrame({"name": ["Ann"]}), pl.DataFrame({"name": ["Bo", "Cy"]})]

    assert _run(size, value=[{"name": "Ann"}]).outputs == {"q.result": 1}
    assert _run(size, value=["a", "b"]).outputs == {"q.result": 2}
    run = _run(size, each, edges=[["each.result", "q.value"]])
    assert run.outputs == {"q.result": [1, 2]}
    back = registry_of(size, each).load(run.graph.record).load_run(run.to_json())
    assert [frame.to_dicts() for frame in back.results["each"]["result"]] == [[{"name": "Ann"}], [{"name": "Bo"}, {"name": "Cy"}]]


def test_a_record_with_a_date_is_no_json_value() -> None:
    @node("dated", title="Dated", description="")
    def dated() -> Dated:
        return {"name": "Ann", "born": datetime.date(1990, 1, 2)}

    @node("named", title="Named", description="")
    def named() -> Named:
        return {"name": "Ann"}

    @node("json", title="JSON", description="")
    def json(value: JsonValue) -> str:
        return str(value)

    def problems(source: Any) -> list[str]:
        graph = registry_of(source, json).load({"nodes": [{"id": "s", "node": source.id}, {"id": "j", "node": "json"}], "edges": [["s.result", "j.value"]]})
        return [problem.type for problem in graph.problems]

    assert problems(dated) == ["type_mismatch"] and problems(named) == []


def test_a_models_class_variables_are_not_its_fields() -> None:
    @host_type("address", title="Address")
    class Address(BaseModel):
        city: str = "Aarhus"

    @node("address", title="Address", description="")
    def address() -> Address:
        return Address()

    @node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    def loaded(field: str) -> Any:
        return registry_of(address, shout).load({"nodes": [{"id": "a", "node": "address"}, {"id": "s", "node": "shout"}], "edges": [[f"a.result.{field}", "s.text"]]})

    assert loaded("city").run().outputs == {"s.result": "AARHUS"}
    for wrong in ("town", "type_id"):
        (problem,) = loaded(wrong).problems
        assert (problem.type, problem.ctx["outputs"]) == ("unknown_output", ["a.result.city"])


def test_a_host_type_names_the_other_classes_it_takes() -> None:
    @host_type("celsius", title="Celsius")
    class Celsius(BaseModel):
        degrees: float

    @host_type("kelvin", title="Kelvin", accepts=[Celsius])
    class Kelvin(BaseModel):
        degrees: float

        @model_validator(mode="before")
        @classmethod
        def _from_celsius(cls, value: Any) -> Any:
            return {"degrees": value.degrees + 273.15} if isinstance(value, Celsius) else value

    @node("outside", title="Outside", description="")
    def outside() -> Celsius:
        return Celsius(degrees=20)

    @node("absolute", title="Absolute", description="")
    def absolute(temperature: Kelvin) -> float:
        return temperature.degrees

    @node("warm", title="Warm", description="")
    def warm(temperature: Celsius) -> bool:
        return temperature.degrees > 15

    assert _run(absolute, outside, edges=[["outside.result", "q.temperature"]]).outputs == {"q.result": 293.15}
    registry = registry_of(outside, absolute, warm)
    assert {described.id: described.accepted_as for described in registry.describe().types}["celsius"] == ["celsius", "kelvin"]


def test_one_hint_is_read_once() -> None:
    from conductor._internal.value_type import Type

    hint = Annotated[int, Param(ge=0)]
    assert Type(hint) is Type(Annotated[int, Param(ge=0)]) and Type(hint).adapter is Type(hint).adapter


def test_a_widget_in_annotated_outside_param_is_refused_saying_where_it_goes() -> None:
    from conductor.widgets import Textarea

    for hint in (Annotated[str, Textarea()], Series[Annotated[str, Textarea()]], Annotated[str, Textarea()] | None):
        with pytest.raises(NodeDeclarationError, match=r"Param\(widget=") as raised:
            node("note", title="Note", description="")(_returning_its_input(hint))
        assert raised.value.code == "widget_outside_param"

    @node("note", title="Note", description="")
    def note(text: Annotated[str, Param(widget=Textarea())]) -> str:
        return text

    assert note.describe().inputs[0].widget["type"] == "textarea"


def _returning_its_input(hint: Any) -> Any:
    def passes(value: Any) -> str:
        return str(value)

    passes.__annotations__ = {"value": hint, "return": str}
    return passes


def test_what_a_reuse_digest_says_of_a_series_and_a_table_names_no_internal_module() -> None:
    """A unit's reuse digest is made of each field's ``reads_as``, and a stored waiting id of that digest. So the
    text must not change when the class behind ``Series`` or ``Table`` moves to another file."""
    from conductor._internal.value_type import Type

    assert Type(Series[int]).reads_as == "conductor.Series[builtins.int]"
    assert Type(Table).reads_as == "conductor.Table"
    assert repr(Series[int]) == "conductor.Series[int]"
