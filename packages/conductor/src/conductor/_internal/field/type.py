"""``Field``: one input or output of a node, made of its name, its type hint and its default."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args, get_origin

from pydantic import MISSING, JsonValue
from pydantic_core import PydanticSerializationError

from conductor._internal.base import REQUIRED, SKIPPED
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.types import (
    Choice,
    Param,
    Type,
    merged_param,
    split_hint,
    union_members,
    union_of,
)
from conductor.widgets import Widget

from .describe import FieldDescription


@dataclass(frozen=True, eq=False)
class Field:
    """One input or output of a node: ``Field("times", Annotated[int, Param(ge=0)], 1, is_output=False)``, and
    ``Field("result", str | None, is_output=True)``.

    The constructor is the only way to a field: one renamed, given another default or bound to a type variable
    is a new ``Field`` of the changed ``name``, ``default`` or ``hint``, so nothing it holds can go stale. Two fields
    are equal only when they are the same object (``eq=False``), so a field is a weak dictionary key even when its
    default or hint can't be hashed (a list default): ``run/json/codec.py`` keeps a saved run's types per field so.
    What the hint costs to read is kept per hint, so a second field of it is cheap.

    Attributes:
        name: The parameter's name, or the output's.
        hint: The annotation as written.
        default: The default as its type holds it, already read by it and with a JSON form (``DeclaredInputValidator``
            and ``DeclaredOutputValidator`` in ``validator.py`` read a default a node declares); ``REQUIRED`` for none, ``SKIPPED`` for an
            output on a branch not taken, ``None`` for a blank.
        is_output: Whether it is an output, which its description shows as the question an answer to the node fills.
            Every field says which it is, since an input and an output whose type takes ``None`` differ in what
            leaving them out gives.
        param: The ``Param``s of the hint as one; an empty one when it has none.
        type: The type of one value: a ``Series``' item, else ``whole``.
        whole: The type of the field's whole value, which reads it and writes it: a ``Series``' list.
    """

    name: str
    hint: Any = field(repr=False)
    default: Any = REQUIRED
    is_output: bool = field(kw_only=True)
    param: Param = field(init=False, compare=False, repr=False)
    type: Type = field(init=False, compare=False)
    whole: Type = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Reads the hint once, refusing a type conductor can't carry.

        Raises:
            NodeDeclarationError: Naming the field: a type pydantic has no schema for, a ``Series`` of ``Series``, a
                ``Param`` that doesn't fit, or a widget with no JSON form.
        """
        keep = functools.partial(object.__setattr__, self)
        try:
            param, whole, one = _field_parts(self.hint)
            keep("param", param)
            keep("whole", whole)
            keep("type", one)
            whole.schema  # noqa: B018  a type pydantic can't carry fails here
            whole.returned_schema  # noqa: B018  a Series of records may be kept as a table, whose row type fails here
            whole.id  # noqa: B018  a class with no @host_type fails here (type_not_named)
        except NodeDeclarationError as error:
            raise error.within(self.name) from None

    @property
    def is_branch(self) -> bool:
        """Whether it is an output on a branch: one the node may give no value on a call, ``SKIPPED`` instead. An
        output of an exclusive group is one, and so is one whose default is ``SKIPPED``. Left out of an answer, or on a
        call the node did not make, a branch is not chosen."""
        return self.param.exclusive_group is not None or self.default is SKIPPED

    @property
    def is_series(self) -> bool:
        """Whether it is declared ``Series[T]``: an output that returns a list a graph can loop over, or an input that
        takes every item of a loop as one list."""
        return self.type is not self.whole

    @functools.cached_property
    def default_json(self) -> JsonValue | MISSING:
        """The default as JSON: ``"!"`` for ``mark: str = "!"``, ``None`` for ``note: str | None = None``, and
        ``MISSING`` for no default (``REQUIRED``) and for ``SKIPPED``, which have no JSON form."""
        return (
            MISSING
            if self.default is REQUIRED or self.default is SKIPPED
            else self.whole.to_json(self.default)
        )

    @functools.cached_property
    def keeps_none_items(self) -> bool:
        """Whether a ``Series`` input that collects a loop keeps an item that is ``None``, or drops it.

        A loop over ``["a", "b", "c"]`` whose node returns ``None`` for ``"b"``: an input ``Series[str | None]``
        collects ``["a", None, "c"]``, and an input ``Series[str]`` collects ``["a", "c"]``. True only for a
        ``Series`` whose item type takes ``None``; never for a table, as a record is never ``None``."""
        return self.is_series and self.type.optional

    @functools.cached_property
    def description(self) -> FieldDescription:
        """The field as it is declared, as an editor reads it, made once: ``description_with`` the choices its type
        names.
        A palette shows it so, and a waiting call asks an output so unless its ``Ask`` sets choices or a default."""
        return self.description_with(self.type.choices)

    def description_with(self, choices: tuple[Choice, ...] | None) -> FieldDescription:
        """The field as an editor reads it, its values limited to ``choices`` (``None`` for none), with
        ``required`` and ``default`` saying what leaving it out gives:

        - its own default, as JSON: ``mark: str = "!"`` gives ``"!"``;
        - ``None`` for an output that has no default, whose type takes ``None``, that no choices limit, and that is
          neither of an exclusive group nor a ``Series``: ``-> str | None``;
        - a branch not chosen (no default, ``required`` false) for an output of an exclusive group or whose default
          is ``SKIPPED``;
        - else nothing: leaving it out is refused (no default, ``required`` true), as for ``text: str`` and ``-> str``.

        An input whose type takes ``None`` and that has no default is ``required``: an edge or a constant gives it.
        """
        branch = self.is_branch
        default: JsonValue | MISSING
        if self.default is not REQUIRED:
            default = self.default_json  # MISSING for SKIPPED
        elif (
            self.is_output
            and not branch
            and not self.is_series
            and self.whole.optional
            and choices is None
        ):
            default = None
        else:
            default = MISSING
        widget = self.param.widget
        return FieldDescription(
            name=self.name,
            title=self.param.title,
            description=self.param.description,
            is_series=self.is_series,
            type=self.type.id,
            type_variables=[variable.__name__ for variable in self.whole.variables],
            schema=self.type.json_schema,
            required=default is MISSING and not branch,
            default=default,
            choices=choices,
            widget=None if widget is None else widget.model_dump(mode="json"),
            exclusive_group=self.param.exclusive_group,
            constant_only=self.param.constant_only is True,
        )


def _field_parts(hint: Any) -> tuple[Param, Type, Type]:
    """Reads a field's hint: its ``Param``, the type of its whole value and the type of one value.

    A type is made of the hint and what a value must keep (limits, choices, other metadata), so fields that
    differ only in what a person reads share one. A ``Series``' limits and choices are its items', wherever in
    the hint they were written.

    Raises:
        NodeDeclarationError: A ``Series`` of ``Series`` (``nested_series``), a widget in ``Annotated`` that is
            not on a ``Param`` (``widget_outside_param``), or a widget with no JSON form (``widget_without_json_form``).
    """
    bare, param, others = split_hint(hint)
    beside_param = (
        others  # what Annotated holds that is no Param, on the field and on a Series' item
    )
    written, of_items = bare, Type(bare).item is not None
    if of_items:  # the item as written, its type variables still open
        (items,) = (option for option in union_members(written) if option is not type(None))
        bare, of_item, others = split_hint((get_args(items) or (Any,))[0])
        beside_param = (*beside_param, *others)
        if Type(bare).item is not None:
            raise NodeDeclarationError(declaration_problems.nested_series())
        param = merged_param(of_item, param)  # the field's own Param after its item's
    if any(
        isinstance(metadata, Widget) for metadata in beside_param
    ):  # pydantic would pass it by, and the field would show no control
        raise NodeDeclarationError(declaration_problems.widget_outside_param())
    if param.widget is not None:
        try:
            param.widget.model_dump(mode="json")
        except PydanticSerializationError:
            raise NodeDeclarationError(
                declaration_problems.widget_without_json_form(widget=type(param.widget).__name__)
            ) from None
    kept = {**param.limits, **({} if param.choices is None else {"choices": param.choices})}
    metadata = (*others, *([Param(**kept)] if kept else []))
    one = Type(Annotated[(bare, *metadata)] if metadata else bare)
    if not of_items:
        return param, one, one
    whole = union_of(
        (get_origin(option) or option)[one.hint] if option is not type(None) else option
        for option in union_members(written)
    )
    return param, Type(whole), one
