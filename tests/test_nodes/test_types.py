"""What the catalog is declared with: plain types, and one of its own."""

from datetime import date

import conductor_nodes
from conductor import DType
from conductor_nodes.types import Json
from pydantic import BaseModel


def test_the_vocabulary_is_the_plain_types_and_json():
    """Each is declared by some node, so a registry of the standard nodes holds it."""
    vocabulary = conductor_nodes.registry().types
    assert {"str", "float", "bool", "json"} <= vocabulary.keys()
    assert issubclass(vocabulary["str"], DType) and vocabulary["str"].plain is str
    assert vocabulary["json"] is Json


def test_json_wraps_any_json_value():
    assert Json({"a": [1, 2]}).value == {"a": [1, 2]}
    assert Json(None).value is None
    assert Json([1]) == Json([1])


def test_pydantic_validates_json_into_the_dtype():
    class M(BaseModel):
        j: Json
        when: date

    assert M(j={"k": 1}, when=date(2026, 1, 1)).j == Json({"k": 1})
