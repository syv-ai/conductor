"""A node's fields, and what a return declaration produces."""

import dataclasses
from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from conductor import DType, Output, Result, Series
from conductor._internal.node_signature import _return_shape_and_outputs, split_return_into_outputs
from pydantic import TypeAdapter


def _read_return_hint(hint):
    return _return_shape_and_outputs({"return": hint})


class Text(DType, str):
    id = "returns-test-text"
    display_title = "Text"


def test_an_output_has_no_widget_no_default_no_optional():
    """Not blank there — meaningless there. Nobody supplies a result."""
    output = Output("result", Text, title="R")

    assert not hasattr(output, "widget")
    assert not hasattr(output, "default")
    assert not hasattr(output, "optional")


def test_an_output_says_nothing_about_downloading():
    """The dtype decides, and a file carries its own name."""
    output = Output("result", Text, title="R")

    assert not hasattr(output, "download")
    assert not hasattr(output, "filename")


def test_the_record_is_the_schema():
    """Dumping an Output yields the wire, with the dtype as its description."""
    output = Output("result", Series[Text], title="R")

    assert TypeAdapter(Output).dump_python(output, mode="json") == {
        "name": "result",
        "dtype": {"id": "series", "of": {"id": "returns-test-text"}},
        "title": "R",
        "description": None,
        "choice": None,
        "takes_blank": False,
    }


def test_the_record_publishes_a_json_schema():
    schema = TypeAdapter(Output).json_schema()

    assert schema["properties"]["dtype"]["type"] == "object"
    assert set(schema["required"]) == {"name", "dtype", "title"}



class Num(DType, float):
    id = "returns-test-num"
    display_title = "Number"


@dataclasses.dataclass(frozen=True)
class Pair:
    text: Annotated[Text, Result(title="Text")]
    number: Annotated[Num, Result(title="Number")]


def test_a_dtype_return_declares_one_output_named_result():
    returns, outputs = _read_return_hint(Annotated[Text, Result(title="Opsummering")])

    assert returns is Text
    assert [(o.name, o.title, o.dtype) for o in outputs] == [("result", "Opsummering", Text)]


def test_a_declaration_says_nothing_about_downloading():
    """A value is downloadable because of its type, not because of a flag."""
    _, outputs = _read_return_hint(Annotated[Text, Result(title="Rapport")])

    assert not hasattr(outputs[0], "download")
    assert not hasattr(outputs[0], "filename")


def test_a_record_return_declares_one_output_per_field_named_by_the_field():
    returns, outputs = _read_return_hint(Pair)

    assert returns is Pair
    assert [(o.name, o.title, o.dtype) for o in outputs] == [("text", "Text", Text), ("number", "Number", Num)]


def test_a_record_field_declares_its_choice():
    """The branches of a deciding node are one ``choice`` group."""

    @dataclasses.dataclass(frozen=True)
    class Branches:
        if_true: Annotated[Text, Result(title="If true", choice="branches")]
        if_false: Annotated[Text, Result(title="If false", choice="branches")]

    _, outputs = _read_return_hint(Branches)

    assert [(o.name, o.choice) for o in outputs] == [("if_true", "branches"), ("if_false", "branches")]


def test_a_pass_through_return_declares_any():
    """A node that routes what it does not read returns ``Any``; the
    output carries it until the node's hook types it from what arrives."""
    returns, outputs = _read_return_hint(Annotated[Any, Result(title="Videre")])

    assert returns.id == "any"
    assert outputs[0].dtype.id == "any"


def test_a_mapping_return_declares_no_outputs():
    """The interface is computed; the declaration only says 'by name'."""
    assert _read_return_hint(Mapping[str, Any]) == (Mapping, ())


def test_a_return_that_is_none_of_the_three_is_refused():
    with pytest.raises(TypeError, match="a frozen dataclass of them"):
        _read_return_hint(tuple[Text, Text])


def test_unpack_puts_a_dtype_return_on_its_one_output():
    _, outputs = _read_return_hint(Annotated[Text, Result(title="R")])

    assert split_return_into_outputs(Text, "hello", outputs, outputs) == {"result": "hello"}


def test_unpack_reads_a_record_by_field():
    _, outputs = _read_return_hint(Pair)

    assert split_return_into_outputs(Pair, Pair(text=Text("a"), number=Num(1)), outputs, outputs) == {"text": "a", "number": 1}


def test_unpack_refuses_a_value_that_is_not_the_record():
    """The node is broken, and it says so here rather than downstream."""
    _, outputs = _read_return_hint(Pair)

    with pytest.raises(ValueError, match="must return a Pair"):
        split_return_into_outputs(Pair, ("a", 1), outputs, outputs)


def test_unpack_reads_a_mapping_against_the_roster():
    interface = (Output("name", Text, title="Name"), Output("age", Num, title="Age"))

    assert split_return_into_outputs(Mapping, {"name": "Ida", "age": 3}, interface, interface) == {"name": "Ida", "age": 3}
    with pytest.raises(ValueError, match="exactly the outputs"):
        split_return_into_outputs(Mapping, {"name": "Ida"}, interface, interface)
