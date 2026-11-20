"""Tests of ``type_fields``, the fields a pydantic model, a dataclass or a ``TypedDict`` declares, as a hook reads them.

Each test checks one thing its docstring promises, through the public name.
"""

from dataclasses import InitVar, dataclass, field
from typing import Annotated, ClassVar, NotRequired, TypedDict

from conductor import Param, type_fields
from pydantic import BaseModel, PrivateAttr

Count = Annotated[int, Param(ge=0)]


def test_a_generic_models_fields_have_its_argument_filled_in_and_keep_their_limits():
    class Box[T](BaseModel):
        item: T
        count: Count
        _secret: str = PrivateAttr("x")
        kind: ClassVar[str] = "box"

    assert type_fields(Box[int]) == {"item": int, "count": Count, "_secret": str}


def test_a_dataclass_gives_every_annotation_but_a_classvar():
    @dataclass
    class Order:
        total: float
        scale: InitVar[int]
        lines: int = field(init=False, default=0)
        kind: ClassVar[str] = "order"

    fields = type_fields(Order)
    # an InitVar compares by identity, so its own type is compared
    assert list(fields) == ["total", "scale", "lines"] and (
        fields["total"],
        fields["scale"].type,
        fields["lines"],
    ) == (float, int, int)


def test_a_typed_dicts_field_a_value_may_leave_out_is_given_as_its_type():
    class Row[T](TypedDict):
        value: T
        note: NotRequired[str]

    assert type_fields(Row[int]) == {"value": int, "note": str}


def test_a_type_that_declares_no_fields_gives_none():
    assert type_fields(str) == {} and type_fields(dict[str, int]) == {}
