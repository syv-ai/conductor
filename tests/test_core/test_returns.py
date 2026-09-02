"""A node's fields, and what a return declaration produces."""

import dataclasses

import pytest
from pydantic import TypeAdapter

from conductor.dtype import DType
from conductor.metadata import Field, Output
from conductor.series import Series


class Text(DType, str):
    id = "returns-test-text"
    title = "Tekst"


def test_an_output_is_a_field_and_adds_one_contract_fact():
    """``choice`` names the group of exclusive alternatives an output
    belongs to. Compile's contract derivation and the editor read it; the
    engine never does."""
    output = Output(name="result", dtype=Text, title="Resultat")

    assert isinstance(output, Field)
    assert {f.name for f in dataclasses.fields(Output)} == {
        f.name for f in dataclasses.fields(Field)
    } | {"choice"}
    assert output.choice is None
    assert Output(name="if_true", dtype=Text, title="Hvis sand", choice="grene").choice == "grene"


def test_a_field_is_name_dtype_title_description():
    """`optional` is an `Input` fact, not a field's."""
    assert [f.name for f in dataclasses.fields(Field)] == [
        "name",
        "dtype",
        "title",
        "description",
    ]


def test_an_output_carries_what_both_sides_of_a_node_have():
    output = Output(
        name="result", dtype=Text, title="Resultat", description="Svaret"
    )

    assert (output.name, output.dtype, output.title) == ("result", Text, "Resultat")
    assert output.description == "Svaret"


def test_an_output_has_no_widget_no_default_no_optional():
    """Not blank there — meaningless there. Nobody supplies a result."""
    output = Output(name="result", dtype=Text, title="R")

    assert not hasattr(output, "widget")
    assert not hasattr(output, "default")
    assert not hasattr(output, "optional")


def test_an_output_says_nothing_about_downloading():
    """The dtype decides, and a file carries its own name."""
    output = Output(name="result", dtype=Text, title="R")

    assert not hasattr(output, "download")
    assert not hasattr(output, "filename")


def test_a_field_is_frozen_and_keyword_only():
    """`kw_only` is load-bearing, not style: it is what lets `Input` add a
    *required* `widget` after this parent's defaulted fields."""
    output = Output(name="result", dtype=Text, title="R")

    with pytest.raises(Exception):
        output.name = "other"
    with pytest.raises(TypeError):
        Output("result", Text, "R")


def test_the_record_is_the_schema():
    """Dumping an Output yields the wire, with the dtype as its wire id."""
    output = Output(name="result", dtype=Series[Text], title="R")

    assert TypeAdapter(Output).dump_python(output, mode="json") == {
        "name": "result",
        "dtype": {"id": "series", "of": {"id": "returns-test-text", "accepted_as": ["returns-test-text"]}},
        "title": "R",
        "description": None,
        "choice": None,
    }


def test_the_record_publishes_a_json_schema():
    schema = TypeAdapter(Output).json_schema()

    assert schema["properties"]["dtype"]["anyOf"][0]["type"] == "object"
    assert set(schema["required"]) == {"name", "dtype", "title"}


from collections.abc import Mapping
from typing import Annotated, Any

from conductor.returns import Result, outputs_of, unpack


class Num(DType, float):
    id = "returns-test-num"
    title = "Tal"


@dataclasses.dataclass(frozen=True)
class Pair:
    text: Annotated[Text, Result(title="Tekst")]
    number: Annotated[Num, Result(title="Tal")]


def test_a_dtype_return_declares_one_output_named_result():
    returns, outputs = outputs_of(Annotated[Text, Result(title="Opsummering")])

    assert returns is Text
    assert [(o.name, o.title, o.dtype) for o in outputs] == [("result", "Opsummering", Text)]


def test_a_declaration_says_nothing_about_downloading():
    """A value is downloadable because of its type, not because of a flag."""
    _, outputs = outputs_of(Annotated[Text, Result(title="Rapport")])

    assert not hasattr(outputs[0], "download")
    assert not hasattr(outputs[0], "filename")


def test_a_record_return_declares_one_output_per_field_named_by_the_field():
    returns, outputs = outputs_of(Pair)

    assert returns is Pair
    assert [(o.name, o.title, o.dtype) for o in outputs] == [("text", "Tekst", Text), ("number", "Tal", Num)]


def test_a_record_field_declares_its_choice():
    """The branches of a deciding node are one ``choice`` group."""

    @dataclasses.dataclass(frozen=True)
    class Branches:
        if_true: Annotated[Text, Result(title="Hvis sand", choice="grene")]
        if_false: Annotated[Text, Result(title="Hvis falsk", choice="grene")]

    _, outputs = outputs_of(Branches)

    assert [(o.name, o.choice) for o in outputs] == [("if_true", "grene"), ("if_false", "grene")]


def test_a_pass_through_return_declares_any():
    """A node that routes what it does not read returns ``Any``; the
    output carries it until the node's hook types it from what arrives."""
    returns, outputs = outputs_of(Annotated[Any, Result(title="Videre")])

    assert returns is Any
    assert outputs[0].dtype is Any


def test_a_mapping_return_declares_no_outputs():
    """The roster is computed; the declaration only says 'by name'."""
    assert outputs_of(Mapping[str, Any]) == (Mapping, ())


def test_a_dtype_return_without_a_result_is_refused():
    with pytest.raises(TypeError, match="Result"):
        outputs_of(Text)


def test_a_return_that_is_none_of_the_three_is_refused():
    with pytest.raises(TypeError, match="DType, a dataclass"):
        outputs_of(tuple[Text, Text])


def test_unpack_puts_a_dtype_return_on_its_one_output():
    _, outputs = outputs_of(Annotated[Text, Result(title="R")])

    assert unpack(Text, "hej", outputs) == {"result": "hej"}


def test_unpack_reads_a_record_by_field():
    _, outputs = outputs_of(Pair)

    assert unpack(Pair, Pair(text=Text("a"), number=Num(1)), outputs) == {"text": "a", "number": 1}


def test_unpack_refuses_a_value_that_is_not_the_record():
    """The node is broken, and it says so here rather than downstream."""
    _, outputs = outputs_of(Pair)

    with pytest.raises(ValueError, match="must return a Pair"):
        unpack(Pair, ("a", 1), outputs)


def test_unpack_reads_a_mapping_against_the_roster():
    roster = (Output(name="navn", dtype=Text, title="Navn"), Output(name="alder", dtype=Num, title="Alder"))

    assert unpack(Mapping, {"navn": "Ida", "alder": 3}, roster) == {"navn": "Ida", "alder": 3}
    with pytest.raises(ValueError, match="exactly the outputs"):
        unpack(Mapping, {"navn": "Ida"}, roster)


def test_nothing_is_positional():
    """A field's name is the persisted binding key and the Ref an author
    wires, so there is no Results carrying titles by position."""
    import conductor.returns as returns_module

    assert not hasattr(returns_module, "Results")
