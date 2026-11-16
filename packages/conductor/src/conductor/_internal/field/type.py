"""``Field``, one input or output of a node.

A field is a name, a type hint and maybe a default. ``Field`` works out everything else conductor needs from the
hint once, when the field is made: the type of its value, its ``Param`` and whether it is a ``Series``.
"""

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
    """One input or output of a node.

    Here are an input with a limit and a default, and an output that may be ``None``::

        Field("times", Annotated[int, Param(ge=0)], 1, is_output=False)
        Field("result", str | None, is_output=True)

    A field never changes once it is made. To rename one, give it another default or fill in a type variable, you
    make a new ``Field``. That way nothing worked out from the hint can go out of date.

    Two fields are equal only when they are the same object. So a field can be a key in a weak dictionary, even
    when its default is a list and can't be hashed. A saved run uses this to keep the types it stores each field's
    values by.

    Attributes:
        name: The parameter's name, or the output's.
        hint: The type hint as written.
        default: The default, already checked by the field's type and converted to it. ``REQUIRED`` when there is
            none, and ``SKIPPED`` for an output on a branch that may not be taken. ``validator.py`` checks the
            defaults a node declares.
        is_output: Whether it is an output. An input and an output whose type takes ``None`` differ when they are
            left out, so every field says which it is.
        param: The hint's ``Param``, all of them merged into one. An empty ``Param`` when the hint has none.
        type: The type of one value. For a ``Series``, that is the type of one item.
        whole: The type of the field's whole value. For a ``Series``, that is the list. Values are checked and
            written as JSON by this type.
    """

    name: str
    hint: Any = field(repr=False)
    default: Any = REQUIRED
    is_output: bool = field(kw_only=True)
    param: Param = field(init=False, compare=False, repr=False)
    type: Type = field(init=False, compare=False)
    whole: Type = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Works out the field's types from its hint, and refuses a type conductor can't pass along an edge.

        Raises:
            NodeDeclarationError: Naming the field. The hint is a type pydantic has no schema for, a ``Series`` of
                ``Series``, a class not declared with ``@host_type``, a ``Param`` that doesn't fit, or a widget with
                no JSON form.
        """
        keep = functools.partial(object.__setattr__, self)
        try:
            param, whole, one = _field_parts(self.hint)
            keep("param", param)
            keep("whole", whole)
            keep("type", one)
            whole.schema  # noqa: B018  fails for a type pydantic has no schema for
            whole.returned_schema  # noqa: B018  a Series of records may be stored as a table, and fails here if its rows can't be
            whole.id  # noqa: B018  fails for a class with no @host_type (type_not_named)
        except NodeDeclarationError as error:
            raise error.within(self.name) from None

    @property
    def is_branch(self) -> bool:
        """Whether it is a branch: an output whose default is ``SKIPPED``.

        The rule branches follow is in the reference, under "Branches".
        """
        return self.default is SKIPPED

    @property
    def is_series(self) -> bool:
        """Whether it is declared ``Series[T]``.

        A ``Series`` output returns a list that the graph can loop over. A ``Series`` input collects every item of a
        loop into one list.
        """
        return self.type is not self.whole

    @functools.cached_property
    def default_json(self) -> JsonValue | MISSING:
        """The default as JSON.

        ``mark: str = "!"`` gives ``"!"``, and ``note: str | None = None`` gives ``None``. A field with no default, or
        with the default ``SKIPPED``, gives ``MISSING``, since neither can be written as JSON.
        """
        return (
            MISSING
            if self.default is REQUIRED or self.default is SKIPPED
            else self.whole.to_json(self.default)
        )

    @functools.cached_property
    def keeps_none_items(self) -> bool:
        """Whether a ``Series`` input that collects a loop keeps the items that are ``None``.

        Say a loop runs over ``["a", "b", "c"]``, and the node in it returns ``None`` for ``"b"``. An input
        ``Series[str | None]`` collects ``["a", None, "c"]``. An input ``Series[str]`` drops the ``None`` and collects
        ``["a", "c"]``. A table never keeps one, since a record is never ``None``.
        """
        return self.is_series and self.type.optional

    @functools.cached_property
    def description(self) -> FieldDescription:
        """The field as JSON for an editor, with the choices its own type names.

        A palette shows a node's fields this way. A node that raises ``Ask`` asks about its outputs this way too,
        unless the ``Ask`` sets choices or a default of its own.
        """
        return self.description_with(self.type.choices)

    def description_with(self, choices: tuple[Choice, ...] | None) -> FieldDescription:
        """The field as JSON for an editor, with its values limited to ``choices``.

        ``required`` and ``default`` say what happens when the field is left out. There are four cases:

        - A field with a default gets it. ``mark: str = "!"`` gets ``"!"``.
        - An output like ``-> str | None`` gets ``None``. That holds for an output with no default whose type takes
          ``None``, unless it has choices, is a ``Series`` or is a branch.
        - A branch, an output whose default is ``SKIPPED``, is skipped. It has no default, and isn't required.
        - Anything else can't be left out. ``text: str`` and ``-> str`` are required.

        An input whose type takes ``None`` but has no default is still required. An edge or a constant gives it its
        value.

        Args:
            choices: The values the field takes, or ``None`` for no limit.
        """
        branch = self.is_branch
        default: JsonValue | MISSING
        if self.default is not REQUIRED:
            default = self.default_json  # MISSING when the default is SKIPPED
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
            json_schema=self.type.json_schema,
            required=default is MISSING and not branch,
            default=default,
            choices=choices,
            widget=None if widget is None else widget.model_dump(mode="json"),
            constant_only=self.param.constant_only is True,
        )


def _field_parts(hint: Any) -> tuple[Param, Type, Type]:
    """Splits a field's hint into its ``Param``, the type of its whole value and the type of one value.

    The types keep only what limits a value: its limits, its choices and any other metadata. A title or a widget is
    left out of them, so two fields that differ only in what a person reads get the same type.

    On a ``Series``, the limits and choices are for its items. ``Annotated[Series[int], Param(ge=0)]`` and
    ``Series[Annotated[int, Param(ge=0)]]`` both say each item is at least 0.

    Raises:
        NodeDeclarationError: ``nested_series`` for a ``Series`` of ``Series``. ``widget_outside_param`` for a
            widget put in ``Annotated`` on its own, not on a ``Param``. ``widget_without_json_form`` for a widget
            that can't be written as JSON.
    """
    bare, param, others = split_hint(hint)
    beside_param = (
        others  # the metadata in Annotated that isn't a Param, on the field and on a Series' item
    )
    written, of_items = bare, Type(bare).item is not None
    if of_items:  # read the item's own hint, with its type variables not yet filled in
        (items,) = (option for option in union_members(written) if option is not type(None))
        bare, of_item, others = split_hint((get_args(items) or (Any,))[0])
        beside_param = (*beside_param, *others)
        if Type(bare).item is not None:
            raise NodeDeclarationError(declaration_problems.nested_series())
        param = merged_param(of_item, param)  # the field's own Param wins over its item's
    if any(
        isinstance(metadata, Widget) for metadata in beside_param
    ):  # pydantic would ignore it, and describe() would show no widget for the field
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
