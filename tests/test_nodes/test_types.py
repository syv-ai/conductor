"""The four types the catalog is declared with."""

from conductor.dtype import DType, registered_dtypes
from conductor_nodes.types import Flag, Json, Number, Text
from pydantic import BaseModel


def test_the_vocabulary_is_four_dtypes():
    for dtype, dtype_id, title in (
        (Text, "text", "Text"),
        (Number, "number", "Number"),
        (Flag, "flag", "Flag"),
        (Json, "json", "JSON"),
    ):
        assert issubclass(dtype, DType)
        assert dtype.id == dtype_id
        assert dtype.title == title
        assert dtype in registered_dtypes()


def test_text_and_number_are_their_builtins():
    assert isinstance(Text("a"), str)
    assert isinstance(Number(2.5), float)
    assert Number("3") == 3.0


def test_a_flag_is_a_boolean_built_on_int():
    """``bool`` cannot be subclassed; ``int`` can, and a ``bool`` is an ``int``."""
    assert bool(Flag(True)) is True
    assert bool(Flag(0)) is False
    assert Flag(True) == Flag(1)


def test_json_wraps_any_json_value():
    assert Json({"a": [1, 2]}).value == {"a": [1, 2]}
    assert Json(None).value is None
    assert Json([1]) == Json([1])


def test_pydantic_validates_each_into_the_dtype():
    class M(BaseModel):
        t: Text
        n: Number
        f: Flag
        j: Json

    v = M(t="hi", n="3.5", f=False, j={"k": 1})

    assert isinstance(v.t, Text)
    assert isinstance(v.n, Number) and v.n == 3.5
    assert isinstance(v.f, Flag) and bool(v.f) is False
    assert isinstance(v.j, Json) and v.j.value == {"k": 1}
