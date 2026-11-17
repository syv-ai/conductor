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
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, get_args, get_origin

from pydantic import Field as PydanticField
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, JsonValue, TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import (
    CoreSchema,
    PydanticSerializationError,
    SchemaValidator,
    core_schema,
    to_jsonable_python,
)

from conductor._internal.base import canonical_json
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor.widgets import Widget

from .choice import Choice, check_one_of, choice_texts, declared_choices, pick_texts
from .hint import READ_AS, union_members, union_of
from .rules import with_rules

#: The limits ``Param`` takes, named as pydantic's ``Field`` names them.
_LIMITS = ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length", "pattern")

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
            given, and ``Type.choices`` checks them. On a ``Literal`` field they give its values titles, one each.
            On any other field they limit what it takes. A ``Dropdown`` shows them.
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
            NodeDeclarationError: ``limit_not_supported`` if pydantic has no rule for a limit on ``source``, like
                ``ge`` on a ``str``. ``choices_invalid`` if the choices aren't each a value of ``source``, given
                once.
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
        if unknown_limits and functions <= len(unknown_limits):
            # the type as its developer wrote it
            written = source.__name__ if isinstance(source, type) else str(source)
            raise NodeDeclarationError(
                declaration_problems.limit_not_supported(
                    unknown_limits=unknown_limits, type=written
                )
            )
        if self.choices is None:
            return limited
        reader = TypeAdapter(limited_hint)

        def as_json(value: Any) -> JsonValue:
            """Returns ``value`` as plain JSON data.

            A float that is not finite stays a float, and ``canonical_json`` writes it as text.
            """
            return reader.dump_python(value, mode="json")

        # read by the type's own check with conductor's rules, as an answer is, so an enum member is read from its JSON
        read = declared_choices(
            self.choices, SchemaValidator(with_rules(reader.core_schema)), as_json, literal=False
        )
        texts = choice_texts(read)
        literal = [  # None is not a pick, so it is never a choice
            value
            for option in union_members(source)
            if get_origin(option) is Literal
            for value in get_args(option)
            if value is not None
        ]
        if literal and texts != {canonical_json(as_json(value)) for value in literal}:
            raise NodeDeclarationError(
                declaration_problems.choices_invalid_on_a_literal(values=literal)
            )
        if literal:  # on a Literal, the choices only give its values titles
            checked = limited
        else:

            def one_of_the_choices(value: Any) -> Any:
                check_one_of(texts, pick_texts(as_json(value), several=False))
                return value

            checked = core_schema.no_info_after_validator_function(one_of_the_choices, limited)
        enum = _enum_of(read, reader)
        if enum is None:
            return checked
        # kept for __get_pydantic_json_schema__, which pydantic calls with this schema
        return {**checked, "metadata": {**checked.get("metadata", {}), _ENUM: enum}}  # type: ignore[typeddict-unknown-key]

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Returns the JSON schema of the limited type, with this ``title``, ``description`` and choices added.

        The choices are the schema's ``enum`` (``_enum_of``). This matters at any depth: on a list's item, on a field
        of a record or of a host's own model, and on a table's column, the JSON schema is where they show up.
        """
        written = handler(schema)
        enum = schema.get("metadata", {}).get(_ENUM)
        if enum is not None:
            written["enum"] = enum
        for key in ("title", "description"):
            if getattr(self, key) is not None:
                written[key] = getattr(self, key)
        return written


#: The key of a schema's metadata that keeps the ``enum`` its JSON schema is written with.
_ENUM = "conductor_enum"


def _enum_of(choices: tuple[Choice, ...], reader: TypeAdapter[Any]) -> list[JsonValue] | None:
    """Returns the ``enum`` of a JSON schema whose values are ``choices``: their ids, as JSON.

    ``None`` is added when the type takes it, since ``None`` is taken beside the choices. Whether it does is asked of
    pydantic, as ``Type.optional`` asks it. A schema may say less than the run checks, but never list a value the run
    refuses. So when a choice is not finite, like ``inf``, there is no ``enum``: JSON Schema would read ``"Infinity"``
    as text, which a number refuses.

    Returns:
        The ids, with ``None`` when the type takes it, or ``None`` for no ``enum``.
    """
    ids = [choice.id for choice in choices]
    try:
        json.dumps(ids, allow_nan=False)
    except ValueError:  # a float that is not finite, at any depth
        return None
    try:
        reader.validate_python(None, strict=True)
    except Exception:  # noqa: BLE001  whatever the check raises, it doesn't take None
        return ids
    return ids if None in ids else [*ids, None]


_NO_PARAM = Param()


def split_hint(hint: Any) -> tuple[Any, Param, tuple[Any, ...]]:
    """Takes a hint apart into its bare type, its ``Param`` and its other metadata.

    For example, ``Annotated[X, Param(title="X")] | None`` gives ``X | None`` and that ``Param``. Several
    ``Param``s are merged into one (``merged_param``). A ``polars.DataFrame`` written as a hint is read as
    ``Table``.

    Returns:
        The hint with every ``Annotated`` taken off its members, all its ``Param``s merged into one (an empty one
        if there are none), and any other ``Annotated`` metadata, which is left for pydantic.
    """
    # a plain class has nothing to take off, and most hints are a plain class
    if type(hint) is type and hint not in READ_AS:
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
    return union_of(options), merged_param(*params), tuple(others)


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
