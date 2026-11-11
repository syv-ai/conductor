"""``Param``: the title, description, widget, limits and choices of a field, written inside ``Annotated``.
``split_hint`` separates a hint into its plain type and its ``Param``.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import functools
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin

from pydantic import Field as PydanticField
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, TypeAdapter, ValidationError
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import (
    CoreSchema,
    PydanticCustomError,
    PydanticSerializationError,
    core_schema,
    to_json,
)

from conductor._internal.problem import NodeDeclarationError, value_free_errors
from conductor._internal.value_type.choice import Choice
from conductor._internal.value_type.hint import READ_AS, union_members, union_of
from conductor.widgets import Widget

#: The limits ``Param`` takes, by the words pydantic's ``Field`` uses.
_LIMITS = ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length", "pattern")

#: What a bound may be: a number's, a date's, a time's or a duration's.
_Bound = float | decimal.Decimal | datetime.date | datetime.time | datetime.timedelta


@dataclass(frozen=True, kw_only=True)
class Param:
    """What a person reads about a field, and the limits a value must keep, inside ``Annotated``::

        times: Annotated[int, Param(title="Times", ge=0)] = 1

    The limits and choices hold wherever pydantic reads the type, in a field or in a host's own model.

    Attributes:
        title: What a person reads for the field; its name in everyday words when not given.
        description: A longer text about the field, for its help.
        widget: The ``conductor.widgets`` control a person edits the field with (on an output, answers it with); a
            host's own is a subclass of ``Widget``. Carried to the host and never read here.
        exclusive_group: Outputs of one node that share it are alternatives: on each call the node gives a value on
            one of them and ``SKIPPED`` on the rest.
        choices: The values the field takes, each given as a value or as a ``Choice`` to title it, and held as a
            ``Choice``. On a ``Literal`` field they title its values, one each; on any other they limit what it
            takes. A ``Dropdown`` draws them.
        ge: The least value taken. ``gt``, ``le`` and ``lt`` are the other bounds, and ``multiple_of`` a step; each
            fits a number, a date, a time or a duration.
        min_length: The fewest letters, items or keys taken; ``max_length`` the most. Each fits a text, bytes or a
            collection.
        pattern: The regular expression a text matches.
    """

    title: str | None = None
    description: str | None = None
    widget: Widget | None = None
    exclusive_group: str | None = None
    choices: list[Any] | tuple[Any, ...] | None = None
    ge: _Bound | None = None
    gt: _Bound | None = None
    le: _Bound | None = None
    lt: _Bound | None = None
    multiple_of: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None

    def __post_init__(self) -> None:
        """Holds each choice as a ``Choice``, a bare value titled by its text."""
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(c if isinstance(c, Choice) else Choice(c, str(c)) for c in self.choices))

    @functools.cached_property
    def limits(self) -> dict[str, Any]:
        """The limits set, by pydantic's words, as ``pydantic.Field`` takes them."""
        return {key: value for key in _LIMITS if (value := getattr(self, key)) is not None}

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema of ``source`` held to these limits and choices.

        Args:
            source: The type the ``Param`` annotates.
            handler: Pydantic's, which builds the schema of a type.

        Returns:
            The schema pydantic reads a value by, in a field of a node or of a host's own model.

        Raises:
            NodeDeclarationError: A limit pydantic holds no rule for on ``source`` (``bound_invalid``), or choices
                that aren't each one value of it, once (``choices_invalid``).
        """
        limited_hint: Any = Annotated[source, PydanticField(**self.limits)] if self.limits else source
        limited = handler(limited_hint)
        # A limit pydantic holds no rule for on a type becomes a function of its own round the type's schema, which
        # fails on every value. A further function is a host's own schema, whose values only the host knows.
        held, functions, layer = set[str](), 0, limited
        while True:
            held |= layer.keys()
            functions += layer["type"].startswith("function")
            if "schema" not in layer:
                break
            layer = layer["schema"]  # type: ignore[typeddict-item]  # every schema that wraps another names it so
        unheld = sorted(self.limits.keys() - held)
        if unheld and functions <= len(unheld):
            written = source.__name__ if isinstance(source, type) else str(source)  # as its developer wrote it
            raise NodeDeclarationError(
                f"{unheld} can't apply to {written}, as pydantic holds no such limit on it; take them off, or check the value in the node",
                code="bound_invalid",
            )
        if self.choices is None:
            return limited
        reader = TypeAdapter(limited_hint)
        try:
            ids = [reader.validate_json(to_json(choice.id, inf_nan_mode="constants"), strict=True) for choice in self.choices]
        except PydanticSerializationError:
            raise NodeDeclarationError("a choice has no JSON form; give each choice once, as a value of the field's type", code="choices_invalid") from None
        except ValidationError as refused:
            raise NodeDeclarationError(
                f"a choice is not a value of the field's type: {value_free_errors(refused)[0]['msg']}; give each choice once, as a value of the field's type",
                code="choices_invalid",
            ) from None
        literal = [value for option in union_members(source) if get_origin(option) is Literal for value in get_args(option)]
        distinct = {repr(i) for i in ids}
        if not ids or len(distinct) != len(ids):
            raise NodeDeclarationError("the choices hold an id twice, or none; give each choice once, as a value of the field's type", code="choices_invalid")
        if literal and distinct != {repr(value) for value in literal}:
            raise NodeDeclarationError(
                f"on a Literal the choices title its values, one each: {literal}; give each choice once, as a value of the field's type", code="choices_invalid",
            )
        if literal:  # a Literal's choices title its values
            return limited

        def one_of_the_choices(value: Any) -> Any:
            if value is not None and value not in ids:
                raise PydanticCustomError("not_a_choice", "the value is not one of the choices")
            return value

        return core_schema.no_info_after_validator_function(one_of_the_choices, limited)

    def __get_pydantic_json_schema__(self, schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Gives the JSON schema of the limited type, with this ``title`` and ``description`` where a model's field carries them."""
        written = handler(schema)
        for key in ("title", "description"):
            if getattr(self, key) is not None:
                written[key] = getattr(self, key)
        return written


_NO_PARAM = Param()


def split_hint(hint: Any) -> tuple[Any, Param, tuple[Any, ...]]:
    """Takes a hint apart into its bare type, its ``Param`` and its other metadata.

    ``Annotated[X, Param(...)] | None`` is ``X | None`` with that ``Param``; several ``Param``s are one, each
    argument the last one given. A ``polars.DataFrame`` written as a hint is read as ``Table``.

    Returns:
        The hint with every ``Annotated`` taken off its members, the ``Param``s as one (an empty one for none),
        and whatever else ``Annotated`` held, which pydantic reads.
    """
    if type(hint) is type and hint not in READ_AS:  # a plain class has nothing to take off, and most hints are one
        return hint, _NO_PARAM, ()
    options, params, others = [], list[Param](), list[Any]()
    for option in union_members(hint):
        if get_origin(option) is Annotated:
            option, *metadata = get_args(option)
            params += [m for m in metadata if isinstance(m, Param)]
            others += [m for m in metadata if not isinstance(m, Param)]
            inner, inner_param, inner_others = split_hint(option)
            option, params, others = inner, [inner_param, *params], [*inner_others, *others]
        options.append(READ_AS.get(option, option) if type(option) is type else option)
    merged = {f.name: value for param in params for f in dataclasses.fields(param) if (value := getattr(param, f.name)) is not None}
    return union_of(options), Param(**merged) if merged else _NO_PARAM, tuple(others)
