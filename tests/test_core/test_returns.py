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
