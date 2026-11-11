"""``Field``: one input or output of a node, made of its name, its type hint and its default."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, TypeVar, get_args, get_origin

from pydantic import JsonValue
from pydantic_core import PydanticSerializationError

from conductor._internal.field.describe import FieldDescription
from conductor._internal.problem import NodeDeclarationError
from conductor._internal.sentinels import REQUIRED, SKIPPED
from conductor._internal.value_type import (
    FieldValueError,
    Param,
    Type,
    list_of,
    split_hint,
    union_members,
    union_of,
)
from conductor.widgets import Widget


@dataclass(frozen=True)
class Field:
    """One input or output of a node: ``Field("times", Annotated[int, Param(ge=0)], 1)``.

    The constructor is the only way to a field: one renamed, given another default or bound to a type variable
    is a new ``Field`` of the changed ``name``, ``default`` or ``hint``, so nothing it holds can go stale.
    What the hint costs to read is kept per hint, so a second field of it is cheap.

    Attributes:
        name: The parameter's name, or the output's.
        hint: The annotation as written.
        default: The default as its type holds it; ``REQUIRED`` for none, ``SKIPPED`` for an output on a branch
            not taken, ``None`` for a blank.
        param: The ``Param``s of the hint as one; an empty one when it has none.
        type: The type of one value: a ``Series``' item, else ``whole``.
        whole: The type of the field's whole value, which reads it and writes it: a ``Series``' list.
        many: Whether it is a ``Series``: an output that makes items, an input that gathers them.
        default_json: The default as JSON; ``None`` where there is none to propose.
    """

    name: str
    hint: Any = field(repr=False)
    default: Any = REQUIRED
    param: Param = field(init=False, compare=False, repr=False)
    type: Type = field(init=False, compare=False)
    whole: Type = field(init=False, compare=False, repr=False)
    many: bool = field(init=False, compare=False)
    default_json: JsonValue = field(init=False, compare=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Reads the hint and the default once, refusing what conductor can't carry.

        Raises:
            NodeDeclarationError: Naming the field: a type pydantic has no schema for, a ``Series`` of ``Series``,
                a ``Param`` that doesn't fit, or a default its type doesn't take or can't write as JSON.
            PydanticSerializationError: The widget has no JSON form.
        """
        keep = functools.partial(object.__setattr__, self)
        try:
            param, whole, one = _field_parts(self.hint)
            keep("param", param)
            keep("whole", whole)
            keep("type", one)
            keep("many", one is not whole)
            whole.adapter  # noqa: B018  a type pydantic can't carry fails here
            whole.id  # noqa: B018  a class with no @host_type fails here (type_not_named)
            if self.default is None and not whole.optional:
                raise NodeDeclarationError("the default None is a blank; write its type with | None", code="default_invalid")
            if self.default is not REQUIRED and self.default is not SKIPPED and self.default is not None:
                try:
                    keep("default", whole.read(self.default))
                    keep("default_json", whole.to_json(self.default))
                except (FieldValueError, PydanticSerializationError, UnicodeDecodeError) as error:
                    raise NodeDeclarationError(
                        f"the default {self.default!r} is not a value of {whole.id} that it writes as JSON, so give one that is: {error}", code="default_invalid",
                    ) from None
        except NodeDeclarationError as error:
            raise NodeDeclarationError(f"{self.name}: {error.message}", code=error.code) from None
        if self.param.widget is not None:
            self.description  # noqa: B018  a widget with no JSON form fails here

    def bind(self, bound: Mapping[TypeVar, Any]) -> Field:
        """Gives this field with its type variables replaced by what ``bound`` binds them to, one nothing binds by
        its bound, else ``Any``; the field itself when its hint holds none."""
        return Field(self.name, Type(self.hint).bind(bound).hint, self.default) if self.whole.variables else self

    @functools.cached_property
    def keeps_blanks(self) -> bool:
        """Whether a field that gathers keeps the place of an item with no value, as a blank: a ``Series`` whose
        item type takes one. A table never does, as a record is never blank."""
        return self.many and self.type.optional

    @functools.cached_property
    def slot(self) -> Type:
        """The type of the field's value on one unit, as a saved run and a run's written outputs hold it: blank
        where the call returned none, else the form its type stores (``SeriesItems`` or ``Records`` for a ``Series``,
        the value itself for any other type)."""
        return Type(union_of((self.whole.stored_hint, type(None))))

    @functools.cached_property
    def column(self) -> Type:
        """The type of the field's values over many units: a list of what ``slot`` is the type of."""
        return Type(list_of(self.slot.hint))

    @functools.cached_property
    def description(self) -> FieldDescription:
        """The field as an editor reads it, made once."""
        choices, widget = self.type.choices, self.param.widget
        return FieldDescription(
            name=self.name, title=self.param.title or self.name.replace("-", " ").replace("_", " ").strip().capitalize(),
            description=self.param.description, many=self.many, type=self.type.id, type_variables=[variable.__name__ for variable in self.whole.variables],
            schema=self.type.json_schema, required=self.default is REQUIRED, default=self.default_json,
            choices=None if choices is None else [choice.model_dump(mode="json") for choice in choices],
            widget=None if widget is None else widget.model_dump(mode="json"), exclusive_group=self.param.exclusive_group,
        )


def _field_parts(hint: Any) -> tuple[Param, Type, Type]:
    """Reads a field's hint: its ``Param``, the type of its whole value and the type of one value.

    A type is made of the hint and what a value must keep (limits, choices, other metadata), so fields that
    differ only in what a person reads share one. A ``Series``' limits and choices are its items', wherever in
    the hint they were written.

    Raises:
        NodeDeclarationError: A ``Series`` of ``Series`` (``nested_series``), or a widget in ``Annotated`` that is
            not on a ``Param`` (``widget_outside_param``).
    """
    bare, param, others = split_hint(hint)
    beside_param = others  # what Annotated holds that is no Param, on the field and on a Series' item
    written, of_items = bare, Type(bare).item is not None
    if of_items:  # the item as written, its type variables still open
        (items,) = (option for option in union_members(written) if option is not type(None))
        bare, of_item, others = split_hint((get_args(items) or (Any,))[0])
        beside_param = (*beside_param, *others)
        if Type(bare).item is not None:
            raise NodeDeclarationError("a Series of Series; a field holds one level of items, so return one Series", code="nested_series")
        param = dataclasses.replace(of_item, **{f.name: value for f in dataclasses.fields(param) if (value := getattr(param, f.name)) is not None})
    if any(isinstance(mark, Widget) for mark in beside_param):  # pydantic would pass it by, and the field would show no control
        raise NodeDeclarationError("a widget in Annotated goes on Param: Annotated[..., Param(widget=...)]", code="widget_outside_param")
    kept = {**param.limits, **({} if param.choices is None else {"choices": param.choices})}
    metadata = (*others, *([Param(**kept)] if kept else []))
    one = Type(Annotated[(bare, *metadata)] if metadata else bare)
    if not of_items:
        return param, one, one
    whole = union_of((get_origin(option) or option)[one.hint] if option is not type(None) else option for option in union_members(written))
    return param, Type(whole), one
