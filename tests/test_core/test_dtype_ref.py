"""A record carries a declared type on the wire as its description."""

from typing import Any

from conductor import DType
from conductor._internal.dtype import DTypeRef, description_of, setting_type
from pydantic import TypeAdapter


class Text(DType, str):
    """A host-side type; the id is test-scoped because the registry is process-global."""

    id = "dtype-ref-test-text"
    title = "Text"


def test_a_dtype_ref_dumps_as_the_description():
    assert TypeAdapter(DTypeRef).dump_python(Text, mode="json") == {"id": "dtype-ref-test-text"}


def test_a_dtype_ref_keeps_the_declared_class_in_python():
    """The class is read off a signature, never off the wire, so validation is identity."""
    assert TypeAdapter(DTypeRef).validate_python(Text) is Text


def test_an_any_input_dumps_as_any():
    """A palette shows an if/else node with ``value: Any`` before anything is
    connected, so the declaration needs a JSON form — and it says "any", never
    a dtype id."""

    declared = TypeAdapter(DTypeRef).validate_python(Any)
    assert TypeAdapter(DTypeRef).dump_python(declared, mode="json") == {"id": "any"}


def test_a_closed_inputs_plain_type_dumps_as_a_setting():
    """A closed input may hold a type that is not a ``DType`` (a schema, a set
    of branches). It is wrapped in a private ``DType``, which says so on the wire."""

    class Schema:
        pass

    assert description_of(setting_type(Schema)) == {"id": "setting", "of": "Schema"}


def test_a_dtype_ref_has_a_json_schema():
    """So a record holding one can be a FastAPI response model."""
    wire = TypeAdapter(DTypeRef).json_schema()

    assert wire["type"] == "object"
    assert wire["properties"]["id"] == {"type": "string"}
    assert "of" in wire["properties"]
