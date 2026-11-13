"""``Field``: one input or output of a node, made of its name, its type hint and its default."""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args, get_origin

from pydantic import JsonValue

from conductor._internal.base import REQUIRED, SKIPPED
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.types import (
    Param,
    Type,
    split_hint,
    union_members,
    union_of,
)
from conductor.widgets import Widget

from .describe import FieldDescription


@dataclass(frozen=True, eq=False)
class Field:
    """One input or output of a node: ``Field("times", Annotated[int, Param(ge=0)], 1)``.

    The constructor is the only way to a field: one renamed, given another default or bound to a type variable
    is a new ``Field`` of the changed ``name``, ``default`` or ``hint``, so nothing it holds can go stale. Two fields
    are equal only when they are the same object (``eq=False``), so a field is a weak dictionary key even when its
    default or hint can't be hashed (a list default): ``run/json/codec.py`` keeps a saved run's types per field so.
    What the hint costs to read is kept per hint, so a second field of it is cheap.

    Attributes:
        name: The parameter's name, or the output's.
        hint: The annotation as written.
        default: The default as its type holds it, already read by it and with a JSON form (``gate.declared_field``
            reads a default a node declares); ``REQUIRED`` for none, ``SKIPPED`` for an output on a branch not taken,
            ``None`` for a blank.
        param: The ``Param``s of the hint as one; an empty one when it has none.
        type: The type of one value: a ``Series``' item, else ``whole``.
        whole: The type of the field's whole value, which reads it and writes it: a ``Series``' list.
        is_series: Whether it is declared ``Series[T]``: an output that returns a list a graph can loop over, or an input
            that takes every item of a loop as one list.
    """

    name: str
    hint: Any = field(repr=False)
    default: Any = REQUIRED
    param: Param = field(init=False, compare=False, repr=False)
    type: Type = field(init=False, compare=False)
    whole: Type = field(init=False, compare=False, repr=False)
    is_series: bool = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Reads the hint once, refusing a type conductor can't carry.

        Raises:
            NodeDeclarationError: Naming the field: a type pydantic has no schema for, a ``Series`` of ``Series``, or
                a ``Param`` that doesn't fit.
            PydanticSerializationError: The widget has no JSON form.
        """
        keep = functools.partial(object.__setattr__, self)
        try:
            param, whole, one = _field_parts(self.hint)
            keep("param", param)
            keep("whole", whole)
            keep("type", one)
            keep("is_series", one is not whole)
            whole.schema  # noqa: B018  a type pydantic can't carry fails here
            whole.returned_schema  # noqa: B018  a Series of records may be kept as a table, whose row type fails here
            whole.id  # noqa: B018  a class with no @host_type fails here (type_not_named)
        except NodeDeclarationError as error:
            raise error.within(self.name) from None
        if self.param.widget is not None:
            self.description  # noqa: B018  a widget with no JSON form fails here

    @functools.cached_property
    def default_json(self) -> JsonValue:
        """The default as JSON; ``None`` where there is none to propose."""
        return None if self.default is REQUIRED or self.default is SKIPPED else self.whole.to_json(self.default)

    @functools.cached_property
    def accepts_none_items(self) -> bool:
        """Whether a field that gathers keeps the place of an item with no value, as a blank: a ``Series`` whose
        item type takes one. A table never does, as a record is never blank."""
        return self.is_series and self.type.optional

    @functools.cached_property
    def description(self) -> FieldDescription:
        """The field as an editor reads it, made once."""
        widget = self.param.widget
        return FieldDescription(
            name=self.name, title=self.param.title or self.name.replace("-", " ").replace("_", " ").strip().capitalize(),
            description=self.param.description, is_series=self.is_series, type=self.type.id, type_variables=[variable.__name__ for variable in self.whole.variables],
            schema=self.type.json_schema, required=self.default is REQUIRED, default=self.default_json,
            choices=self.type.choices,
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
            raise NodeDeclarationError(declaration_problems.nested_series())
        param = dataclasses.replace(of_item, **{f.name: value for f in dataclasses.fields(param) if (value := getattr(param, f.name)) is not None})
    if any(isinstance(metadata, Widget) for metadata in beside_param):  # pydantic would pass it by, and the field would show no control
        raise NodeDeclarationError(declaration_problems.widget_outside_param())
    kept = {**param.limits, **({} if param.choices is None else {"choices": param.choices})}
    metadata = (*others, *([Param(**kept)] if kept else []))
    one = Type(Annotated[(bare, *metadata)] if metadata else bare)
    if not of_items:
        return param, one, one
    whole = union_of((get_origin(option) or option)[one.hint] if option is not type(None) else option for option in union_members(written))
    return param, Type(whole), one
