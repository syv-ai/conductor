"""A type means the same wherever it is read: on an edge, in a list, under a limit, in an answer.

Each test names one place where reading an annotation a second way gave a second answer.
"""

import datetime
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    Ask,
    Graph,
    GraphRecord,
    NodeDeclarationError,
    NodeRegistry,
    Param,
    Run,
    Series,
    Table,
    host_type,
    node,
)
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
    nodes = [{"id": "q", "node": definition.id, "constants": values}, *({"id": other.id, "node": other.id} for other in more)]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": nodes, "edges": edges or []}), registry_of(definition, *more))
    assert not graph.problems, graph.problems
    return graph.run().resume()


def test_an_answer_to_a_series_of_dates_is_read_from_its_json() -> None:
    @node("days", title="Days", description="For a test.")
    def days() -> Series[datetime.date]:
        raise Ask("Which days?")

    run = _run(days)
    assert run.answer("q", ["2026-01-01"]).resume().outputs == {"q.result": [datetime.date(2026, 1, 1)]}


def test_a_limit_on_a_hosts_text_is_held() -> None:
    @node("short", title="Short", description="For a test.")
    def short(text: Annotated[Text, Param(max_length=3)]) -> str:
        return text

    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "q", "node": "short", "constants": {"text": "too long"}}]}), registry_of(short))
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


def test_a_list_of_a_hosts_text_is_a_type_like_any_other() -> None:
    @node("texts", title="Texts", description="For a test.")
    def texts(items: list[Text]) -> int:
        return sum(type(item) is Text for item in items)

    assert _run(texts, items=["a", "b"]).outputs == {"q.result": 2}


def test_a_model_does_not_land_on_a_json_value() -> None:
    @node("money", title="Money", description="For a test.")
    def money() -> Money:
        return Money()

    @node("json", title="JSON", description="For a test.")
    def json(value: JsonValue) -> str:
        return str(value)

    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "m", "node": "money"}, {"id": "j", "node": "json"}],
        "edges": [["m.result", "j.value"]],
    }), registry_of(money, json))
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


def test_a_hosts_own_method_named_accepts_is_nothing_to_conductor() -> None:
    @host_type("offer", title="Offer")
    class Offer(BaseModel):
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
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "m", "node": "money"}, {"id": "t", "node": "take"}], "edges": [["m.result", "t.offer"]]}), registry)
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

        @node("pick", title="Pick", description="For a test.")
        def pick(value: Annotated[str, Param(widget=Picker(source=Catalogue()))]) -> str:
            return value


def test_a_yes_or_no_is_no_number_inside_a_list_either() -> None:
    @node("total", title="Total", description="For a test.")
    def total(numbers: list[int]) -> int:
        return sum(numbers)

    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "q", "node": "total", "constants": {"numbers": [True]}}]}), registry_of(total))
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


def test_a_list_of_whole_numbers_lands_on_a_list_of_numbers_and_never_on_a_list_of_texts() -> None:
    @node("ints", title="Ints", description="For a test.")
    def ints() -> list[int]:
        return [1, 2]

    @node("mean", title="Mean", description="For a test.")
    def mean(numbers: list[float]) -> float:
        return sum(numbers) / len(numbers)

    @node("words", title="Words", description="For a test.")
    def words(texts: list[str]) -> int:
        return len(texts)

    assert _run(mean, ints, edges=[["ints.result", "q.numbers"]]).outputs == {"q.result": 1.5}
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "i", "node": "ints"}, {"id": "w", "node": "words"}],
        "edges": [["i.result", "w.texts"]],
    }), registry_of(ints, words))
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


def test_a_pydantic_type_is_carried_with_the_limits_it_has() -> None:
    from annotated_types import Gt
    from pydantic import PositiveInt, constr

    @node("count", title="Count", description="For a test.")
    def count(times: PositiveInt, least: Annotated[int, Gt(1)] = 2, word: constr(min_length=1) = "a") -> str:  # type: ignore[valid-type]
        return word * times * least

    for wrong in ({"times": 0}, {"times": 1, "least": 1}, {"times": 1, "word": ""}):
        graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "q", "node": "count", "constants": wrong}]}), registry_of(count))
        assert [problem.type for problem in graph.problems] == ["invalid_value"], wrong
    assert _run(count, times=3).outputs == {"q.result": "aaaaaa"}


def test_a_pydantic_field_in_a_hint_is_refused_showing_the_param_to_write() -> None:
    from pydantic import Field

    limit = Field(ge=0)
    hints = (
        Annotated[int, limit], Annotated[int, limit] | None, Annotated[int, Field(title="Times")], Annotated[Annotated[int, limit], Param(title="Times")],
        Series[Annotated[int, limit]], list[Annotated[int, limit]], dict[str, Annotated[int, limit]],
    )
    for hint in hints:
        with pytest.raises(NodeDeclarationError, match=r"value: pydantic's Field\(\.\.\.\) in the hint.*Annotated\[int, Param\(ge=0\)\]") as raised:
            node("count", title="Count", description="For a test.")(_returning_its_input(hint))
        assert raised.value.code == "pydantic_field_in_hint", hint

    class Row(TypedDict):
        times: Annotated[int, limit]

    for returned in (Annotated[int, limit], Table[Row]):  # an output, and a column of a table
        with pytest.raises(NodeDeclarationError) as raised:
            node("count", title="Count", description="For a test.")(_returning(returned))
        assert raised.value.code == "pydantic_field_in_hint", returned


def test_a_pydantic_field_that_tells_a_union_apart_is_refused_showing_the_discriminator_to_write() -> None:
    from pydantic import Discriminator, Field

    @host_type("cat", title="Cat")
    class Cat(BaseModel):
        kind: Literal["cat"] = "cat"

    @host_type("dog", title="Dog")
    class Dog(BaseModel):
        kind: Literal["dog"] = "dog"

    with pytest.raises(NodeDeclarationError, match=r"value: pydantic's Field\(discriminator=\.\.\.\) in the hint.*Annotated\[Cat \| Dog, Discriminator\(\"kind\"\)\]") as raised:
        node("pet", title="Pet", description="For a test.")(_returning_its_input(Annotated[Cat | Dog, Field(discriminator="kind")]))
    assert raised.value.code == "pydantic_field_in_hint" and "Param" not in str(raised.value)

    pet = node("pet", title="Pet", description="For a test.")(_returning_its_input(Annotated[Cat | Dog, Discriminator("kind")]))
    assert _run(pet, value={"kind": "dog"}).outputs == {"q.result": "kind='dog'"}


def test_a_pydantic_field_inside_a_hosts_own_model_is_the_models() -> None:
    from pydantic import Field

    @host_type("stock", title="Stock")
    class Stock(BaseModel):
        count: int = Field(ge=0)
        name: Annotated[str, Field(min_length=1)] = "a"

    @node("restock", title="Restock", description="For a test.")
    def restock(stock: Stock) -> int:
        return stock.count

    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "q", "node": "restock", "constants": {"stock": {"count": -1}}}]}), registry_of(restock))
    assert [problem.type for problem in graph.problems] == ["invalid_value"]
    assert _run(restock, stock={"count": 2}).outputs == {"q.result": 2}


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
    @node("people", title="People", description="For a test.")
    def people() -> Table[Named]:
        return pl.DataFrame({"name": ["Ann"], "email": ["ann@x.dk"]})

    @node("all", title="All", description="For a test.")
    def everything() -> Table:
        return pl.DataFrame({"name": ["Ann"], "email": ["ann@x.dk"]})

    assert _run(people).results["q"]["result"].columns == ["name"]
    assert _run(everything).results["q"]["result"].columns == ["name", "email"]


def test_a_table_of_no_rows_holds_every_declared_column_of_its_declared_type() -> None:
    @node("nobody", title="Nobody", description="For a test.")
    def nobody() -> Table[Dated]:
        return pl.DataFrame()

    @node("gathered", title="Gathered", description="For a test.")
    def gathered(people: Table[Dated]) -> int:
        assert people.schema == pl.Schema({"name": pl.String, "born": pl.Date})
        return people.height

    assert _run(nobody).results["q"]["result"].schema == pl.Schema({"name": pl.String, "born": pl.Date})
    assert _run(gathered, people=[]).outputs == {"q.result": 0}


def test_records_gathered_into_a_table_are_read_once_each_cell_by_its_columns_type() -> None:
    @node("person", title="Person", description="For a test.")
    def person() -> Series[Dated]:
        return [{"name": "Ann", "born": datetime.date(1990, 1, 2)}, {"name": "Bo", "born": datetime.date(1991, 3, 4)}]

    @node("oldest", title="Oldest", description="For a test.")
    def oldest(people: Table[Dated]) -> datetime.date:
        return people["born"].min()

    run = _run(oldest, person, edges=[["person.result", "q.people"]])
    assert run.outputs == {"q.result": datetime.date(1990, 1, 2)}


def test_a_table_stands_in_a_union_and_in_a_series() -> None:
    @node("size", title="Size", description="For a test.")
    def size(value: Table[Named] | list[str]) -> int:
        return len(value)

    @node("each", title="Each", description="For a test.")
    def each() -> Series[Table[Named]]:
        return [pl.DataFrame({"name": ["Ann"]}), pl.DataFrame({"name": ["Bo", "Cy"]})]

    assert _run(size, value=[{"name": "Ann"}]).outputs == {"q.result": 1}
    assert _run(size, value=["a", "b"]).outputs == {"q.result": 2}
    run = _run(size, each, edges=[["each.result", "q.value"]])
    assert run.outputs == {"q.result": [1, 2]}
    back = Run.from_json(run.to_json(), Graph.compile(run.graph.record, registry_of(size, each)))
    assert [frame.to_dicts() for frame in back.results["each"]["result"]] == [[{"name": "Ann"}], [{"name": "Bo"}, {"name": "Cy"}]]


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
        graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": source.id}, {"id": "j", "node": "json"}], "edges": [["s.result", "j.value"]]}), registry_of(source, json))
        return [problem.type for problem in graph.problems]

    assert problems(dated) == ["type_mismatch"] and problems(named) == []


def test_a_models_class_variables_are_not_its_fields() -> None:
    @host_type("address", title="Address")
    class Address(BaseModel):
        city: str = "Aarhus"

    @node("address", title="Address", description="For a test.")
    def address() -> Address:
        return Address()

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    def loaded(field: str) -> Any:
        return Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "address"}, {"id": "s", "node": "shout"}], "edges": [[f"a.result.{field}", "s.text"]]}), registry_of(address, shout))

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

    @node("outside", title="Outside", description="For a test.")
    def outside() -> Celsius:
        return Celsius(degrees=20)

    @node("absolute", title="Absolute", description="For a test.")
    def absolute(temperature: Kelvin) -> float:
        return temperature.degrees

    @node("warm", title="Warm", description="For a test.")
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
            node("note", title="Note", description="For a test.")(_returning_its_input(hint))
        assert raised.value.code == "widget_outside_param"

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


def _stripped(text: str) -> str:
    return text.strip()


class _Marked:
    """Metadata with no ``repr`` of its own: Python writes it with its memory address."""


def test_what_a_code_fingerprint_says_of_a_hint_holds_no_memory_address() -> None:
    """A function or an object in ``Annotated`` is written by its name, never by ``repr``, which gives its address in
    this process. The fingerprint is saved with a run, so it has to come out the same in the process that loads it."""
    from conductor._internal.value_type import Type
    from pydantic import AfterValidator

    def local(text: str) -> str:
        return text

    def another_of_the_same_name() -> Any:
        def local(text: str) -> str:
            return text

        return local

    assert Type(Annotated[str, AfterValidator(_stripped)]).reads_as == f"builtins.str['AfterValidator(func={__name__}._stripped)']"
    assert Type(Annotated[str, _Marked()]).reads_as == f"builtins.str['{__name__}._Marked']"
    here = f"{__name__}.test_what_a_code_fingerprint_says_of_a_hint_holds_no_memory_address.<locals>"
    assert Type(Annotated[str, AfterValidator(local)]).reads_as == f"builtins.str['AfterValidator(func={here}.local)']"
    assert Type(Annotated[str, AfterValidator(lambda text: text)]).reads_as == f"builtins.str['AfterValidator(func={here}.<lambda>)']"
    assert Type(Annotated[str, AfterValidator(another_of_the_same_name())]).reads_as != Type(Annotated[str, AfterValidator(local)]).reads_as  # named by where it is written
    assert Type(Annotated[str, AfterValidator(str.strip)]).reads_as == "builtins.str[\"AfterValidator(func=<method 'strip' of 'str' objects>)\"]"  # a repr with no address stays


_A_NODE_WITH_A_FUNCTION_IN_ITS_HINT = """
import sys
from typing import Annotated

from conductor import Graph, NodeRegistry, Run, make_graph
from pydantic import AfterValidator

calls = []
padding = [object() for _ in range(int(sys.argv[2]))]  # so the function below is at another address in each process


def stripped(text: str) -> str:
    return text.strip()


registry = NodeRegistry()


@registry.node("shout", title="Shout", description="For a test.")
def shout(text: Annotated[str, AfterValidator(stripped)]) -> str:
    calls.append(text)
    return text.upper()


graph = Graph.compile(make_graph(shout), registry)
if sys.argv[2] == "0":
    open(sys.argv[1], "w").write(graph.run(text=" a ").to_json())
else:
    again = Run.from_json(open(sys.argv[1]).read(), graph).again(text=" a ")
    print(again.outputs, "calls:", len(calls))
"""


def test_a_run_saved_in_one_process_is_reused_in_another_when_a_hint_holds_a_function(tmp_path: Any) -> None:
    import subprocess
    import sys

    program, saved = tmp_path / "shout.py", tmp_path / "run.json"
    program.write_text(_A_NODE_WITH_A_FUNCTION_IN_ITS_HINT)
    for padding, printed in (("0", ""), ("100000", "{'shout.result': 'A'} calls: 0\n")):
        child = subprocess.run([sys.executable, str(program), str(saved), padding], capture_output=True, text=True, timeout=120, check=False)
        assert child.returncode == 0, child.stderr
        assert child.stdout == printed


def test_what_a_code_fingerprint_says_of_a_series_and_a_table_names_no_internal_module() -> None:
    """A unit's reuse digest is made of each field's ``reads_as``, and a stored waiting id of that digest. So the
    text must not change when the class behind ``Series`` or ``Table`` moves to another file."""
    from conductor._internal.value_type import Type

    assert Type(Series[int]).reads_as == "conductor.Series[builtins.int]"
    assert Type(Table).reads_as == "conductor.Table"
    assert repr(Series[int]) == "conductor.Series[int]"
