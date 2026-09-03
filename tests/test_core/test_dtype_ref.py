"""A record carries a declared type on the wire as its description."""

from typing import Any

from conductor.dtype import DType
from conductor.dtype_ref import DTypeRef, description_of
from pydantic import TypeAdapter


class Text(DType, str):
    """A host-side type; the id is test-scoped because the registry is process-global."""

    id = "dtype-ref-test-text"
    title = "Text"


def test_a_dtype_ref_dumps_as_the_description():
    assert TypeAdapter(DTypeRef).dump_python(Text, mode="json") == {"id": "dtype-ref-test-text", "accepted_as": ["dtype-ref-test-text"]}


def test_a_dtype_ref_keeps_the_declared_class_in_python():
    """The class is read off a signature, never off the wire, so validation is identity."""
    assert TypeAdapter(DTypeRef).validate_python(Text) is Text


def test_an_any_input_dumps_as_any():
    """A palette shows an if/else node with ``value: Any`` before anything is
    wired, so the declaration needs a JSON form — and it says "any", never
    a dtype id."""

    assert description_of(Any) == {"id": "any"}
    assert TypeAdapter(DTypeRef).dump_python(Any, mode="json") == {"id": "any"}


def test_a_static_type_has_no_wire_form():
    """A handle-less input may declare a type that is not a ``DType``
    — a schema, a set of branches. Nothing travels on it, so the wire says
    ``null`` rather than inventing an id that no cable could carry."""

    class Schema:
        pass

    assert description_of(Schema) is None
    assert TypeAdapter(DTypeRef).dump_python(Schema, mode="json") is None


def test_a_dtype_ref_has_a_json_schema():
    """So a record holding one can be a FastAPI response model. The
    ``null`` arm is the static type of a handle-less input."""
    schema = TypeAdapter(DTypeRef).json_schema()

    wire, nothing = schema["anyOf"]
    assert nothing == {"type": "null"}
    assert wire["type"] == "object"
    assert wire["properties"]["id"] == {"type": "string"}
    assert "of" in wire["properties"]
