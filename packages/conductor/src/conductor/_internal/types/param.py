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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, get_args, get_origin

from annotated_types import Ge, GroupedMetadata, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, JsonValue, StringConstraints
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import (
    CoreSchema,
    PydanticSerializationError,
    core_schema,
)

from conductor._internal.base import canonical_json, plain_json
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor.widgets import Widget

from .choice import check_one_of, declared_choices, pick_texts
from .hint import READ_AS, literal_classes, union_members, union_of

#: The limits ``Param`` takes: by the keyword that writes each, the class pydantic is handed it as, whose attribute
#: of that name is its value. ``Param(ge=0)`` is handed over as ``Ge(ge=0)``, and ``Param(pattern="x")`` as
#: pydantic's own ``StringConstraints(pattern="x")``, since annotated-types has no pattern.
LIMITS: dict[str, type[Any]] = {
    "ge": Ge,
    "gt": Gt,
    "le": Le,
    "lt": Lt,
    "multiple_of": MultipleOf,
    "min_length": MinLen,
    "max_length": MaxLen,
    "pattern": StringConstraints,
}

#: The keyword of each class in ``LIMITS``.
_KEYWORDS = {kind: keyword for keyword, kind in LIMITS.items()}

#: The key of a core schema's metadata that says its values are sized, so they take ``min_length`` and
#: ``max_length``, which count them with ``len()``: a table's schema sets it, and a table's length is its rows.
SIZED = "conductor_sized"

#: What a bound like ``ge`` may be: a number, a date, a time or a duration.
_Bound = float | Decimal | datetime.date | datetime.time | datetime.timedelta


@dataclass(frozen=True, kw_only=True, eq=False)
class Param:
    """A field's title, description, widget, limits and choices, written inside ``Annotated``.

    For example::

        times: Annotated[int, Param(title="Times", ge=0)] = 1

    The limits and choices apply wherever pydantic checks the type, whether on a node's field or in a host's own
    model. On a node's input or output, pydantic's ``Field(...)`` is refused (``pydantic_field_in_hint``). An
    annotated-types limit written in the hint, like ``Ge(0)``, ``Interval(ge=0, le=8)`` or ``Len(1, 3)``, is read as
    the same ``Param``.

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
        ge: The smallest value allowed. ``gt``, ``le`` and ``lt`` are the other bounds. Each works on a number, a
            date, a time or a duration. ``multiple_of`` is a step, on a number or a duration.
        min_length: The fewest characters, items or keys allowed, and ``max_length`` the most. Each works on text,
            bytes or a collection, and on a table counts its rows.
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
    multiple_of: float | datetime.timedelta | None = None
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
                    canonical_json(plain_json(choices)),
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
    def limits(self) -> tuple[Any, ...]:
        """The limits that are set, as the objects pydantic is handed: ``(Ge(ge=0), MaxLen(max_length=8))``.

        Each is an annotated-types limit, and a pattern pydantic's ``StringConstraints``.
        """
        return tuple(
            kind(**{keyword: value})
            for keyword, kind in LIMITS.items()
            if (value := getattr(self, keyword)) is not None
        )

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
                ``choices_invalid`` if the choices aren't each a value of ``source``, given once.
        """
        limited = handler(Annotated[(source, *self.limits)] if self.limits else source)
        names = [_KEYWORDS[type(limit)] for limit in self.limits]
        if unknown := not_taken(limited, names):
            raise NodeDeclarationError(
                declaration_problems.param_limit_not_supported(
                    unknown_limits=unknown, type=as_written(source)
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
                    type=as_written(literal_classes(union_of(literals)))
                )
            )
        if literals:
            raise NodeDeclarationError(
                declaration_problems.choices_on_a_literal_beside_another_type()
            )
        # Type is built on Param, so it is imported here
        from .type import Type

        # the choices are read by the type's own argument check, limits included, as an answer is
        value_type = Type(Annotated[(source, *self.limits)] if self.limits else source)

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
        # them. A choice is finite (declared_choices), so JSON Schema reads each as the run does.
        enum = [choice.id for choice in read]
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


def as_written(hint: Any) -> str:
    """Returns ``hint`` as its developer wrote it: ``str`` for a class, ``str | None`` for a union."""
    return hint.__name__ if isinstance(hint, type) else str(hint)


def split_hint(hint: Any) -> tuple[Any, Param, tuple[Any, ...]]:
    """Takes a hint apart into its bare type, its ``Param`` and its other metadata.

    For example, ``Annotated[X, Param(title="X")] | None`` gives ``X | None`` and that ``Param``. Several
    ``Param``s are merged into one (``merged_param``). A class that means another hint (``READ_AS``) is read as
    that hint: a ``polars.DataFrame`` as ``Table``, and pydantic's ``EmailStr`` as a ``str`` with its check.

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
    # a class that means another hint is taken apart as that hint: EmailStr as Annotated[str, EmailStr]
    if isinstance(hint, type) and hint in READ_AS:
        return split_hint(READ_AS[hint])
    members = union_members(hint)
    if len(members) == 1:
        if get_origin(hint) is not Annotated:
            return hint, _NO_PARAM, ()
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
        params.append(dataclasses.replace(param, **dict.fromkeys(LIMITS)))
        on_member = (*kept, *param.limits)
        options.append(Annotated[(option, *on_member)] if on_member else option)
    return union_of(options), merged_param(*params), ()


def _limits_as_params(metadata: Iterable[Any]) -> Iterator[Any]:
    """Yields ``metadata`` with each limit in it as a ``Param`` of its limits, in its place.

    An annotated-types limit, like ``Ge(0)``, is a ``Param`` of that one limit, and a group of them, like
    ``Len(1, 3)`` or ``Interval(ge=0, lt=5)``, is taken apart into its limits first. A ``StringConstraints``, which
    ``constr(...)`` writes, gives a ``Param`` of its lengths and pattern, and a ``StringConstraints`` of the rest,
    such as ``to_lower``, which is pydantic's. So ``constr(min_length=1)`` is ``Param(min_length=1)``, and of two
    lengths the last one written wins wherever each is written. Any other metadata is yielded as it is.
    """
    for one in metadata:
        if isinstance(one, StringConstraints):
            given = {
                f.name: value
                for f in dataclasses.fields(one)
                if (value := getattr(one, f.name)) is not None
            }
            limits = {
                name: given.pop(name)
                for name in ("min_length", "max_length", "pattern")
                if name in given
            }
            yield from ([Param(**limits)] if limits else [])
            yield from ([StringConstraints(**given)] if given else [])
        elif (keyword := _KEYWORDS.get(type(one))) is not None:
            yield Param(**{keyword: getattr(one, keyword)})
        elif isinstance(one, GroupedMetadata):
            yield from _limits_as_params(one)
        else:
            yield one


def not_taken(limited: CoreSchema, names: Iterable[str]) -> list[str]:
    """Returns the limits, of ``names``, that the type pydantic built in ``limited`` doesn't take.

    The schema pydantic built says which limits it took. Where it has a rule for a limit on a kind of value, it sets
    the limit on that value's schema as a key: ``{"type": "int", "ge": 0}``. Where it has none, it wraps the schema
    in a function that compares each value with the limit. That comparison is taken over a host's own function,
    since only the host knows whether its values compare, and over a schema that says its values are sized, as a
    table's does, for a length. Over one of pydantic's own kinds, such as ``str`` or ``bool``, it would fail on every
    value or take ``True`` as a number, so the limit is not taken. A limit that is nowhere in the schema, as when a
    ``PlainValidator`` replaced the schema it was put on, is not taken either.

    Args:
        limited: The schema pydantic built for a type with these limits.
        names: The limits, by the ``Param`` keyword of each.

    Returns:
        The limits not taken, sorted.
    """
    wanted, taken, layer = set(names), set[str](), limited
    while True:
        taken |= wanted & layer.keys()
        if (compared := _compared_limit(layer)) is not None:
            inner = layer["schema"]  # type: ignore[typeddict-item]  # a function after keeps the one it wraps here
            while _compared_limit(inner) is not None:
                inner = inner["schema"]
            metadata = inner.get("metadata", {})
            sized = compared in ("min_length", "max_length") and metadata.get(SIZED)
            if inner["type"].startswith("function") or sized:
                taken.add(compared)
        if "schema" not in layer:
            return sorted(wanted - taken)
        layer = layer["schema"]  # type: ignore[typeddict-item]  # a wrapping schema keeps the inner one here


def _compared_limit(schema: CoreSchema) -> str | None:
    """Returns the limit ``schema`` compares each value with, when it is pydantic's comparison for a limit.

    Pydantic writes that comparison as a function after the schema it wraps: a ``functools.partial`` of its own
    validator, given the limit by name (``ge=0``), with the limit kept for the JSON schema in its metadata
    (``pydantic_js_updates``). ``None`` for any other schema.
    """
    function = schema.get("function", {}).get("function")
    if (
        schema["type"] == "function-after"
        and isinstance(function, functools.partial)
        and len(function.keywords) == 1
        and "pydantic_js_updates" in schema.get("metadata", {})
    ):
        return str(next(iter(function.keywords)))
    return None


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
