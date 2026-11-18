"""``Param``: what a node author adds to a type hint, inside ``Annotated``.

A plain hint like ``int`` says what kind of value a field takes. ``Param`` says the rest: the title and
description a person sees, the widget they edit it with, limits like ``ge=0``, and choices to pick from::

    times: Annotated[int, Param(title="Times", ge=0)] = 1

``split_hint`` takes a hint apart into its plain type and its ``Param``. The full rules are in the reference:
``python -m conductor.about types``.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, get_args, get_origin

from annotated_types import Ge, GroupedMetadata, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
from pydantic import Field as PydanticField
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, JsonValue
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import (
    CoreSchema,
    PydanticSerializationError,
    core_schema,
    to_jsonable_python,
)

from conductor._internal.base import canonical_json
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor.widgets import Widget

from .choice import check_one_of, declared_choices, pick_texts
from .hint import READ_AS, literal_classes, union_members, union_of

#: The limits ``Param`` takes, named as pydantic's ``Field`` names them.
_LIMITS = ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length", "pattern")

#: annotated-types' own limits, each the same as the ``Param`` limit its one attribute is named for, so
#: ``Annotated[int, Ge(0)]`` is ``Annotated[int, Param(ge=0)]``.
_ANNOTATED_LIMITS = {
    Gt: "gt",
    Ge: "ge",
    Lt: "lt",
    Le: "le",
    MultipleOf: "multiple_of",
    MinLen: "min_length",
    MaxLen: "max_length",
}

#: The key of a core schema's metadata that says its values are sized, so ``min_length`` and ``max_length`` count
#: them with ``len()``: a table's schema sets it, and a table's length is its rows.
SIZED = "conductor_sized"

#: What a bound like ``ge`` may be: a number, a date, a time or a duration.
_Bound = float | Decimal | datetime.date | datetime.time | datetime.timedelta


@dataclass(frozen=True, kw_only=True, eq=False)
class Param:
    """A field's title, description, widget, limits and choices, written inside ``Annotated``.

    For example::

        times: Annotated[int, Param(title="Times", ge=0)] = 1

    The limits and choices apply wherever pydantic checks the type, whether on a node's field or in a host's own
    model. On a node's input or output, ``Param`` is the only way to write them. Pydantic's ``Field(...)`` is
    refused there (``pydantic_field_in_hint``).

    Attributes:
        title: What a person sees for the field. ``None`` if not given, and nothing is guessed from the name.
        description: A longer text about the field, shown as help.
        widget: The ``conductor.widgets`` control a person edits the field with, or answers it with on an output.
            A host's own control is a subclass of ``Widget``. Conductor passes it on to the host and never uses it.
        constant_only: Whether an input takes only a constant, from the graph record or its default, and never an
            edge or a graph input. ``None`` means no. An editor offers no edge for it.
        choices: The values the field takes, each a plain value or a ``Choice`` with a title. They are kept as
            given, and ``Type.choices`` checks them. They go on a plain type, like ``str``: a ``Literal``'s values
            are its choices already, so ``Param(choices=)`` on one is refused (``choices_on_a_literal``). A
            ``Dropdown`` shows them.
        ge: The smallest value allowed. ``gt``, ``le`` and ``lt`` are the other bounds, and ``multiple_of`` is a
            step. Each works on a number, a date, a time or a duration.
        min_length: The fewest characters, items or keys allowed, and ``max_length`` the most. Each works on text,
            bytes or a collection.
        pattern: A regular expression the text must match.
    """

    title: str | None = None
    description: str | None = None
    widget: Widget | None = None
    constant_only: bool | None = None
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
        """Stores the choices as a tuple, in the order given."""
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Param) and self._compared == other._compared

    def __hash__(self) -> int:
        return hash(self._compared)

    @functools.cached_property
    def _compared(self) -> tuple[Any, ...]:
        """The values two ``Param``s are compared by.

        The choices are compared by their JSON text (``canonical_json``) and by the class of each. The class
        counts, so two ``Param``s that differ only in a ``Choice`` subclass aren't taken for one, and a host is
        shown the right class.
        """
        choices: Any = self.choices
        if choices is not None:
            try:
                choices = (
                    canonical_json(to_jsonable_python(choices)),
                    tuple(type(choice) for choice in choices),
                )
            # no JSON form: compare as given; the choices are refused later, when the type is built
            except PydanticSerializationError:
                pass
        return (
            *(getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "choices"),
            choices,
        )

    @functools.cached_property
    def limits(self) -> dict[str, Any]:
        """The limits that are set, by name, ready to pass to ``pydantic.Field``."""
        return {key: value for key in _LIMITS if (value := getattr(self, key)) is not None}

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Returns the schema of ``source``, with these limits and choices checked.

        Args:
            source: The type the ``Param`` is on.
            handler: Pydantic's handler, which builds the schema of a type.

        Returns:
            The schema pydantic checks a value with, on a node's field or in a host's own model.

        Raises:
            NodeDeclarationError: ``param_limit_not_supported`` if pydantic has no rule for a limit on ``source``, like
                ``ge`` on a ``str``. ``param_limit_value_invalid`` if a limit's value doesn't fit ``source``.
                ``choices_on_a_literal`` if ``source`` is a ``Literal``, and
                ``choices_on_a_literal_beside_another_type`` if it is a union of one with another type.
                ``choices_invalid`` or one of the ``choices_invalid_...`` codes if the choices aren't each a value of
                ``source``, given once.
        """
        limited_hint: Any = (
            Annotated[source, PydanticField(**self.limits)] if self.limits else source
        )
        limited = handler(limited_hint)
        # When pydantic has no rule for a limit on a type, it wraps the type's schema in one function per such
        # limit, and that function compares each value directly. On a type like ``str`` that fails for every value.
        # So each limit with no rule shows up as a missing key plus one function layer. Any further function layer
        # is a host's own schema. Only the host knows whether its values support the comparison, so the limit is
        # let through.
        held, functions, layer = set[str](), 0, limited
        while True:
            held |= layer.keys()
            functions += layer["type"].startswith("function")
            if "schema" not in layer:
                break
            layer = layer["schema"]  # type: ignore[typeddict-item]  # a wrapping schema keeps the inner one here
        unknown_limits = sorted(self.limits.keys() - held)
        # A type whose schema says its values are sized, as a table's does, takes a length limit too: pydantic's
        # own check of it, one function layer, counts len() of the value, a table's rows.
        if layer.get("metadata", {}).get(SIZED):
            lengths = [limit for limit in unknown_limits if limit in ("min_length", "max_length")]
            unknown_limits = [limit for limit in unknown_limits if limit not in lengths]
            functions -= len(lengths)
        if unknown_limits and functions <= len(unknown_limits):
            raise NodeDeclarationError(
                declaration_problems.param_limit_not_supported(
                    unknown_limits=unknown_limits, type=_as_written(source)
                )
            )
        if self.choices is None:
            return limited
        # a Literal's values are its choices already, so choices on it would write each value twice
        members = union_members(source)
        literals = [option for option in members if get_origin(option) is Literal]
        if literals and len(literals) + (type(None) in members) == len(members):
            raise NodeDeclarationError(
                declaration_problems.choices_on_a_literal(
                    type=_as_written(literal_classes(union_of(literals)))
                )
            )
        if literals:
            raise NodeDeclarationError(
                declaration_problems.choices_on_a_literal_beside_another_type()
            )
        # Type is built on Param, so it is imported here
        from .type import Type

        # the choices are read by the type's own argument check, limits included, as an answer is
        value_type = Type(Annotated[source, Param(**self.limits)]) if self.limits else Type(source)

        def as_json(value: Any) -> JsonValue:
            """Returns ``value`` as plain JSON data.

            A float that is not finite stays a float, and ``canonical_json`` writes it as text.
            """
            written: JsonValue = value_type.serializer.to_python(value, mode="json")
            return written

        read = declared_choices(self.choices, value_type.argument_validator, as_json)

        def one_of_the_choices(value: Any) -> Any:
            check_one_of(read, pick_texts(as_json(value), several=False))
            return value

        checked = core_schema.no_info_after_validator_function(one_of_the_choices, limited)
        # The schema's enum is the choices' ids, with None when the type takes it, since None is taken beside
        # them. A schema may say less than the run checks, but never list a value the run refuses, so a choice
        # that isn't finite, like inf, leaves no enum: JSON Schema would read its "Infinity" as text.
        enum = [choice.id for choice in read]
        try:
            json.dumps(enum, allow_nan=False)
        except ValueError:  # a float that is not finite, at any depth
            return checked
        if value_type.optional and None not in enum:
            enum.append(None)
        # kept for __get_pydantic_json_schema__, which pydantic calls with this schema
        return {**checked, "metadata": {**checked.get("metadata", {}), _SCHEMA_ENUM: enum}}  # type: ignore[typeddict-unknown-key]

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Returns the JSON schema of the limited type, with this ``title``, ``description`` and choices added.

        The choices are the schema's ``enum``: their ids, with ``None`` when the type takes it. This matters at any
        depth: on a list's item, on a field of a record or of a host's own model, and on a table's column, the JSON
        schema is where they show up.

        A schema that is a ``$ref``, like an enum's, is put in an ``allOf`` before anything is added beside it. JSON
        Schema draft-07 ignores every key beside a ``$ref``, so a form reading ``{"$ref": ..., "enum": ["red"]}`` that
        way would offer every member; ``{"allOf": [{"$ref": ...}], "enum": ["red"]}`` means the same in every draft.
        """
        written = handler(schema)
        enum = schema.get("metadata", {}).get(_SCHEMA_ENUM)
        added = {
            **({} if enum is None else {"enum": enum}),
            **{
                key: getattr(self, key)
                for key in ("title", "description")
                if getattr(self, key) is not None
            },
        }
        if added and "$ref" in written:
            written = {"allOf": [{"$ref": written.pop("$ref")}], **written}
        return {**written, **added}


#: The key of a schema's metadata that keeps the ``enum`` its JSON schema is written with.
_SCHEMA_ENUM = "conductor_enum"


_NO_PARAM = Param()


def _as_written(hint: Any) -> str:
    """Returns ``hint`` as its developer wrote it: ``str`` for a class, ``str | None`` for a union."""
    return hint.__name__ if isinstance(hint, type) else str(hint)


def split_hint(hint: Any) -> tuple[Any, Param, tuple[Any, ...]]:
    """Takes a hint apart into its bare type, its ``Param`` and its other metadata.

    For example, ``Annotated[X, Param(title="X")] | None`` gives ``X | None`` and that ``Param``. Several
    ``Param``s are merged into one (``merged_param``). A ``polars.DataFrame`` written as a hint is read as
    ``Table``.

    An annotated-types limit, like ``Ge(0)`` or ``Len(1, 3)``, is read as the ``Param`` limit of the same name, so
    ``Annotated[int, Ge(0)]`` and ``Annotated[int, Param(ge=0)]`` are one type, and a limit that doesn't fit its
    type is refused the same way (``param_limit_not_supported``).

    A ``Param`` is lifted from a union's member onto the whole, since the field has one title and one set of
    choices, but for its limits, which stay on the member they are written on, as pydantic reads them:
    ``Annotated[int, Ge(0)] | str`` limits only the ``int``. Any other metadata stays on its member, where it checks only that member's values: in
    ``Annotated[str, AfterValidator(f)] | None`` the validator never sees ``None``, and in JsonValue's union each
    member keeps the ``Tag`` its ``Discriminator`` picks it by.

    Returns:
        The hint with every ``Param`` taken off, all of them merged into one (an empty one if there are none), and
        the hint's own other ``Annotated`` metadata, which is left for pydantic.
    """
    # a plain class has nothing to take off, and most hints are a plain class
    if type(hint) is type and hint not in READ_AS:
        return hint, _NO_PARAM, ()
    members = union_members(hint)
    if len(members) == 1:
        if get_origin(hint) is not Annotated:
            return (READ_AS.get(hint, hint) if type(hint) is type else hint), _NO_PARAM, ()
        inner, *given = get_args(hint)
        metadata = list(_limits_as_params(given))
        bare, inner_param, inner_others = split_hint(inner)
        params = [m for m in metadata if isinstance(m, Param)]
        own = [m for m in metadata if not isinstance(m, Param)]
        return bare, merged_param(inner_param, *params), (*inner_others, *own)
    options, params = [], list[Param]()
    for member in members:
        option, param, kept = split_hint(member)
        # a limit belongs to the member it is written on: Annotated[int, Ge(0)] | str limits only the int
        limits = param.limits
        params.append(dataclasses.replace(param, **dict.fromkeys(limits)))
        on_member = (*kept, *([Param(**limits)] if limits else []))
        options.append(Annotated[(option, *on_member)] if on_member else option)
    return union_of(options), merged_param(*params), ()


def _limits_as_params(metadata: Iterable[Any]) -> Iterator[Any]:
    """Yields ``metadata`` with each annotated-types limit in it as a ``Param`` of that one limit, in its place.

    A group of them, like ``Len(1, 3)`` or ``Interval(ge=0, lt=5)``, is taken apart into its limits first. Any other
    metadata is yielded as it is.
    """
    for one in metadata:
        if isinstance(one, GroupedMetadata):
            yield from _limits_as_params(one)
        elif (name := _ANNOTATED_LIMITS.get(type(one))) is not None:
            yield Param(**{name: getattr(one, name)})
        else:
            yield one


def merged_param(*params: Param) -> Param:
    """Merges several ``Param``s on one hint into one.

    For each argument, the last ``Param`` that sets it wins, and ``None`` counts as not set. So
    ``Annotated[Annotated[int, Param(constant_only=True)], Param(title="Times")]`` gives
    ``Param(title="Times", constant_only=True)``. A later ``Param(constant_only=False)`` turns it off again, but a
    later ``title=None`` leaves the title alone.
    """
    arguments = {
        f.name: value
        for param in params
        for f in dataclasses.fields(param)
        if (value := getattr(param, f.name)) is not None
    }
    return Param(**arguments) if arguments else _NO_PARAM
