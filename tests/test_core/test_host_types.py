"""``@host_type``: how a host declares a class of its own, and that a wrong declaration fails at the class statement."""

import dataclasses
from enum import Enum
from ipaddress import IPv4Address
from typing import Any, Generic, TypeVar

import pytest
from conductor import Graph, NodeDeclarationError, NodeRegistry, host_type, make_graph, node
from pydantic import BaseModel
from pydantic_core import core_schema

from tests.test_core.test_nodes import declared


@host_type("text", title="Text")
class Text(str):
    def __new__(cls, value: str) -> "Text":
        if len(value) > 20:
            raise ValueError("a text is at most 20 characters")
        return super().__new__(cls, value)


@host_type("money", title="Money")
class Money(BaseModel):
    amount: float


@host_type("point", title="Point")
@dataclasses.dataclass
class Point:
    x: float
    y: float


@host_type("colour", title="Colour")
class Colour(Enum):
    RED = "red"


@host_type("upload", title="Upload")
class Upload:
    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls, serialization=core_schema.plain_serializer_function_ser_schema(lambda held: held.name))


@host_type("celsius", title="Celsius")
class Celsius(BaseModel):
    degrees: float


@host_type("temperature", title="Temperature", accepts=[Celsius])
class Temperature(BaseModel):
    degrees: float


IP = host_type("ip", title="IP address")(IPv4Address)

T = TypeVar("T")


@host_type("box", title="Box")
class Box(BaseModel, Generic[T]):
    inside: T


@pytest.mark.parametrize(("held", "type_id"), [
    (Text, "text"), (Money, "money"), (Point, "point"), (Colour, "colour"), (Upload, "upload"), (Temperature, "temperature"), (IP, "ip"), (list[Point], "list[point]"),
])
def test_a_class_declared_with_host_type_is_a_fields_type_under_its_id(held: Any, type_id: str) -> None:
    def f(value: held) -> str:  # type: ignore[valid-type]
        return ""

    (described,) = declared(f).describe().inputs
    assert described.type == type_id


def test_a_subclass_of_a_builtin_needs_no_schema_of_its_own_and_its_own_check_runs() -> None:
    @node("short", title="Short", description="For a test.")
    def short(text: Text) -> Text:
        return text

    registry = NodeRegistry()
    registry.add(short)
    graph = Graph.compile(make_graph(short), registry)

    assert type(graph.run(text="hello").outputs["short.result"]) is Text
    with pytest.raises(Exception) as refused:
        graph.run(text="x" * 21)
    assert refused.value.errors()[0].type == "invalid_input"


def test_a_type_that_accepts_another_takes_its_values_on_an_edge() -> None:
    from conductor._internal.value_type import Type

    assert Type(Temperature).fits(Type(Celsius)) is not None and Type(Celsius).fits(Type(Temperature)) is None


def test_a_generic_pydantic_model_is_named_by_its_declared_id_and_its_arguments() -> None:
    def f(a: Box[int], b: Box[str]) -> str:
        return ""

    registry = NodeRegistry()
    registry.add(declared(f))
    assert [i.type for i in declared(f).describe().inputs] == ["box[int]", "box[str]"]
    assert {"box[int]", "box[str]"} <= set(registry.types)


def _refused(build: Any) -> NodeDeclarationError:
    with pytest.raises(NodeDeclarationError) as refused:
        build()
    return refused.value


def test_an_empty_title_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("thing", title="")
        class Thing(BaseModel):
            pass

    assert _refused(build).code == "type_declaration_invalid"


@pytest.mark.parametrize("wrong_id", [5, "str", "Series", "Table", "record", "list[int]", "a b", "a|b", ""], ids=repr)
def test_an_id_that_is_no_word_or_is_one_of_conductors_own_fails_at_the_class_statement(wrong_id: Any) -> None:
    def build() -> None:
        @host_type(wrong_id, title="Thing")
        class Thing(BaseModel):
            pass

    assert _refused(build).code == "type_declaration_invalid"


def test_accepts_that_holds_something_other_than_classes_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("thing", title="Thing", accepts=["celsius"])  # type: ignore[list-item]
        class Thing(BaseModel):
            pass

    assert _refused(build).code == "type_declaration_invalid"


def test_a_plain_class_pydantic_cant_validate_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("handle", title="Handle")
        class Handle:
            def __init__(self, path: str) -> None:
                self.path = path

    refused = _refused(build)
    assert refused.code == "type_not_carried" and "Handle" in str(refused)


def test_host_type_written_under_dataclass_fails_at_the_class_statement_and_says_to_put_it_on_top() -> None:
    def build() -> None:
        @dataclasses.dataclass
        @host_type("pair", title="Pair")
        class Pair:
            x: float

    refused = _refused(build)
    assert refused.code == "type_not_carried" and "@host_type above @dataclass" in str(refused)


def test_a_class_python_lets_nobody_write_on_is_refused_with_the_way_out() -> None:
    refused = _refused(lambda: host_type("whole", title="Whole")(int))
    assert refused.code == "type_declaration_invalid" and "a class of your own" in str(refused)


def test_a_class_that_is_not_declared_fails_at_node_naming_the_class_and_the_line_to_write() -> None:
    @dataclasses.dataclass
    class Plain:
        x: float

    def f(values: list[Plain]) -> str:
        return ""

    refused = _refused(lambda: declared(f))
    assert refused.code == "type_not_named" and "Plain" in str(refused) and '@host_type("...", title="...")' in str(refused)


def test_a_subclass_does_not_inherit_its_parents_declaration() -> None:
    class Rich(Money):
        note: str = ""

    def f(value: Rich) -> str:
        return ""

    refused = _refused(lambda: declared(f))
    assert refused.code == "type_not_named" and "Money's declaration is its own" in str(refused)


def test_a_hook_that_returns_a_class_that_is_not_declared_is_the_hooks_failure() -> None:
    @dataclasses.dataclass
    class Plain:
        x: float

    @node("made", title="Made", description="For a test.", compute_outputs=lambda: {"result": Plain})
    def made() -> dict[str, Any]:
        return {}

    registry = NodeRegistry()
    registry.add(made)
    (problem,) = Graph.compile(make_graph(made), registry).problems
    assert (problem.type, problem.ctx["error"]) == ("compute_hook_failed", "type_not_named")


@pytest.mark.parametrize("accepts", [None, int, 5], ids=repr)
def test_accepts_that_is_no_list_of_classes_fails_at_the_class_statement(accepts: Any) -> None:
    def build() -> None:
        @host_type("thing", title="Thing", accepts=accepts)
        class Thing(BaseModel):
            pass

    assert _refused(build).code == "type_declaration_invalid"


def test_a_title_of_only_spaces_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("thing", title="   ")
        class Thing(BaseModel):
            pass

    assert _refused(build).code == "type_declaration_invalid"


def test_host_type_on_something_that_is_no_class_or_is_a_typed_dict_is_refused() -> None:
    from typing import TypedDict

    class Row(TypedDict):
        n: int

    def a_function() -> None:
        pass

    for wrong in (42, list[int], a_function, Row):
        refused = _refused(lambda wrong=wrong: host_type("thing", title="Thing")(wrong))
        assert refused.code == "type_declaration_invalid", wrong
