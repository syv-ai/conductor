"""Choices: a fixed list of values a person picks from.

A type can say which values it takes, with ``Param(choices=...)`` or a ``Literal``. A node that raises ``Ask``
can offer choices too. Each one is a ``Choice``, an id with a title::

    Annotated[str, Param(choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])]

Two steps use them. First, the choices are checked against the type and converted to JSON (``choices_schema``,
which ``read_choices`` checks by). A list of choices offers at least one and none twice (``choice_list``), and a
question a saved run keeps declares its choices as ``Choices``, held to that rule too.
Later, each value is checked against the choices. That second step compares JSON texts: ``choice_texts`` gives
the text of each choice, ``pick_texts`` the text of each value picked, and ``check_one_of`` refuses a pick that
isn't one of the choices. A message names the choices with ``choices_in_message``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence, Set
from enum import Enum
from typing import Annotated, Any, cast

from pydantic import ConfigDict, GetPydanticSchema, JsonValue, ValidationError
from pydantic_core import (
    CoreSchema,
    PydanticKnownError,
    PydanticSerializationError,
    SchemaValidator,
    core_schema,
)

from conductor._internal.base import (
    WRITE_BY_NAME,
    ConductorModel,
    canonical_json,
    holds_a_set,
    plain_json,
)
from conductor._internal.problem import (
    NodeDeclarationError,
    conductor_error,
    declaration_problems,
    no_json_form,
    set_has_no_order,
    value_free_errors,
)


class Choice(ConductorModel):
    """One option a person can pick, with an id and a title: ``Choice("ida", "Ida B.")``.

    A host can add more fields to say more about an option, such as an icon. Pass them as keywords, or declare
    them on a subclass. A choice is kept as its JSON form, as a saved run keeps it: once a type or an ``Ask`` has
    read it, it is a plain ``Choice`` with each other field a keyword, its value written as JSON, so a ``date`` is
    its ISO text. So a question, a description and a run read back show the same choice. A keyword the class doesn't
    declare is kept, not refused, so a misspelt one is kept too.

    Attributes:
        id: The value an answer has when this choice is picked.
        title: What a person sees for it.
    """

    model_config = ConfigDict(extra="allow")

    id: Any
    title: str

    def __init__(self, id: Any, title: str, **more: Any) -> None:  # noqa: A002  "id" is what a host calls it
        """Makes a choice from its id and title, given in that order.

        Args:
            id: The value an answer has when this choice is picked.
            title: What a person sees for it.
            **more: Any other fields a host gives it.
        """
        fields: dict[str, Any] = {
            "id": id,
            "title": title,
            **more,
        }  # passed as keywords, since the base class's own signature names no fields
        super().__init__(**fields)

    # Pydantic reads a choice from data, such as a saved run's, field by field, and doesn't call __init__ for it.
    # Called with the fields as keywords, __init__ would raise a TypeError for a missing title, where pydantic says
    # "missing" at the field.
    __init__.__pydantic_base_init__ = True  # type: ignore[attr-defined]


def choices_schema(
    read: Callable[[Any], Any],
    write: Callable[[Any], JsonValue],
    known: Iterable[Choice] = (),
) -> CoreSchema:
    """Builds the pydantic schema that checks a list of choices of one type, and gives them back with each id in its
    JSON form.

    The choices are a list or a tuple, with at least one choice. Each id is checked by ``read``, strictly, the way its
    JSON text would be. Then ``write`` turns it back into JSON. A bare value that is the id of a ``known`` choice
    becomes that choice, title and all. Any other bare value becomes a ``Choice`` titled by its text, and an enum
    member by its id's text (``"red"`` for ``Color.RED``, ``"1"`` for a member whose value is 1). A ``Choice`` keeps
    all its fields, written as JSON, on a plain ``Choice``, so a subclass's own fields are its keywords, as a saved
    run reads them back.

    For a ``float``, the schema reads ``[1, Choice(2, "Two")]`` as ``(Choice(1.0, "1"), Choice(2.0, "Two"))``.

    Two ids are the same choice when their JSON texts match (``choice_texts``). So ``{"a": 1, "b": 2}`` and
    ``{"b": 2, "a": 1}`` are one choice given twice, which is refused.

    The caller's ``read`` checks one id. A ``Param``'s choices are read by ``declared_choices`` and an ``Ask``'s by
    the output's type (``read_as_output``). Both read an id as an answer is read, by one reader
    (``read_entering_value``), so a choice a person couldn't give, such as ``inf``, is refused either way.

    Args:
        read: Checks one id by the type the choices are values of, raising pydantic's ``ValidationError`` for one
            the type refuses.
        write: Converts a value ``read`` gave back to JSON.
        known: Choices already read, which a bare value given by its id stands for. An ``Ask`` passes its
            output's own choices, so a bare ``"ida"`` is the ``Choice("ida", "Ida B.")`` the output's type gives.

    Returns:
        A schema whose errors are pydantic's. An error of one choice has the choice's index first in its ``loc``,
        then the steps of ``read``'s own error: the type's refusal, or ``no_json_form`` if the id can't be written as
        JSON. A ``Choice`` with a set in any of its fields is ``set_has_no_order`` at its index. An error of the whole list has the ``loc`` ``()``: ``set_has_no_order`` for a set, ``list_type`` for
        anything else that isn't a list or a tuple, text included, ``too_short`` for no choices, and
        ``choice_twice`` for an id given twice.
    """
    known_by_text = {canonical_json(choice.id): choice for choice in known}

    def a_list(given: Any) -> Any:
        if isinstance(given, Set):
            raise set_has_no_order()
        if isinstance(given, str | bytes) or not isinstance(given, Sequence):
            raise PydanticKnownError("list_type")
        return given

    def one(choice: Any) -> Choice:
        try:
            if isinstance(choice, Choice):
                # JSON writes a set as a list in no fixed order, so the choice would differ between processes
                if holds_a_set(choice):
                    raise set_has_no_order()
                # a model's fields are a JSON object
                fields = cast(
                    "dict[str, Any]",
                    plain_json(choice.model_dump(exclude={"id"}, **WRITE_BY_NAME)),
                )
                return Choice(write(read(choice.id)), **fields)
            written = write(read(choice))
        except PydanticSerializationError:
            raise no_json_form() from None
        return known_by_text.get(
            canonical_json(written),
            Choice(written, str(written) if isinstance(choice, Enum) else str(choice)),
        )

    return core_schema.no_info_before_validator_function(
        a_list, choice_list(core_schema.no_info_plain_validator_function(one))
    )


def choice_list(choice: CoreSchema) -> CoreSchema:
    """Builds the schema of a list of choices, each read by ``choice``: the one rule that a list offers at least one
    choice and none twice.

    It gives the choices back as a tuple, and writes them as a list. Two choices are the same when their ids' JSON
    texts match (``choice_texts``).

    Returns:
        A schema whose error of the whole list is pydantic's ``too_short`` for no choices, or ``choice_twice`` for an
        id given twice, at the list's own place.
    """

    def each_once(chosen: list[Choice]) -> tuple[Choice, ...]:
        if len(choice_texts(chosen)) != len(chosen):
            raise conductor_error("choice_twice", "a choice is given twice")
        return tuple(chosen)

    return core_schema.no_info_after_validator_function(
        each_once,
        core_schema.list_schema(choice, min_length=1),
        # what the list schema writes, given the tuple each_once returns
        serialization=core_schema.plain_serializer_function_ser_schema(
            list, return_schema=core_schema.list_schema(choice)
        ),
    )


#: The choices a question offers for one output, as a run keeps them: at least one, none twice (``choice_list``).
#: ``Unanswered`` declares its choices with it, so a question a saved run reads back is held to the rule an ``Ask``'s
#: choices are. A question the run makes in memory is a plain dataclass, and isn't checked again.
Choices = Annotated[
    tuple[Choice, ...],
    GetPydanticSchema(lambda _source, handler: choice_list(handler.generate_schema(Choice))),
]


def read_choices(
    given: Any,
    read: Callable[[Any], Any],
    write: Callable[[Any], JsonValue],
    known: Iterable[Choice] = (),
) -> tuple[Choice, ...]:
    """Checks the choices of a type by ``choices_schema``, and returns them with each id in its JSON form.

    Args:
        given: The choices as a developer or a node gave them, each a value or a ``Choice``.
        read: As ``choices_schema`` takes it.
        write: As ``choices_schema`` takes it.
        known: As ``choices_schema`` takes it.

    Raises:
        ValidationError: ``choices_schema``'s errors.
    """
    chosen: tuple[Choice, ...] = SchemaValidator(
        choices_schema(read, write, known)
    ).validate_python(given)
    return chosen


def declared_choices(
    given: Any,
    reader: SchemaValidator,
    write: Callable[[Any], JsonValue],
    *,
    where: str,
) -> tuple[Choice, ...]:
    """Reads the choices a type declares, as ``read_choices`` does, and refuses them as a fault in the declaration.

    ``Param`` reads its choices with it, ``Type.choices`` a ``Literal``'s values or an enum's members, and
    ``Type.check_declared`` a ``Param``'s choices again by every check on the hint, so a type whose choices can't be
    read is refused by ``@node``, and nothing that reads them later meets a bare pydantic error.

    Args:
        given: The choices, as ``Param``, the ``Literal`` or the enum gives them. ``Param``'s are as a developer
            wrote them, so text or a set is refused here.
        reader: The validator of the type the choices are values of.
        write: Converts a value ``reader`` gave back to JSON.
        where: Where the choices are, which the problem names: a field of a class, as ``Order.size``, or a type.

    A choice is read from its JSON form, as an answer is (``read_entering_value``). So ``inf`` and ``nan`` are
    refused, a dict's key included: no one could pick them. A choice with a set in it is refused too
    (``set_has_no_order``), since its JSON form would be a list. A host's object is written by its own class's
    schema, as its default is.

    Raises:
        NodeDeclarationError: ``choices_invalid``, with the fault of each choice that doesn't read in
            ``ctx.errors``, or the fault of the whole list once every choice reads.
    """

    # Type is built on choices, so the reader is imported here
    from .read import read_entering_value

    def read(value: Any) -> Any:
        return read_entering_value(reader, value, title="choice")

    try:
        return read_choices(given, read, write)
    except ValidationError as refused:
        # an error of one choice has its index first in its loc; one of the whole list has none
        errors = [
            {**free, "loc": list(raw["loc"][:1])}
            for raw, free in zip(refused.errors(), value_free_errors(refused), strict=True)
        ]
    # a check on the hint that changes a value moves a choice off itself, so the fix is the check, not the choice
    instead = (
        "Give choices that every check on the value keeps as they are, or change the value in the node instead."
        if errors[0]["type"] == "not_a_choice"
        else f"Give each choice once, as a value of {where} that can be written as JSON."
    )
    raise NodeDeclarationError(
        declaration_problems.choices_invalid(
            type=where, reason=errors[0]["msg"], errors=errors, instead=instead
        )
    ) from None


def choice_texts(choices: Iterable[Choice]) -> frozenset[str]:
    """Returns the ids of ``choices`` as the JSON texts a pick is compared with (``canonical_json``).

    Two ids are one choice when their texts match. On a ``float``, ``1`` and ``1.0`` are both checked as ``1.0``.
    And ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` have the same text, since the keys are sorted.
    """
    return frozenset(canonical_json(choice.id) for choice in choices)


def pick_texts(written: Any, *, several: bool) -> list[str]:
    """Returns the values picked in ``written``, as JSON texts to compare with the choices (``canonical_json``).

    ``written`` is one value in its JSON form, or a list of them when ``several`` is set. ``None`` is not a pick: a
    type that allows ``None`` takes it besides its choices. For example,
    ``pick_texts(["a", None, {"b": 1}], several=True)`` is ``['"a"', '{"b":1}']``, and
    ``pick_texts(None, several=False)`` is ``[]``.
    """
    if written is None:
        return []
    return [
        canonical_json(item) for item in (written if several else [written]) if item is not None
    ]


def check_one_of(choices: Sequence[Choice], picks: list[str]) -> None:
    """Checks that every pick is one of ``choices``.

    ``picks`` comes from ``pick_texts``. For example, an input ``Annotated[str | None, Param(choices=["a"])]`` takes
    ``"a"``, and ``None`` too, since ``None`` is not a pick.

    Raises:
        PydanticCustomError: ``not_a_choice`` if a pick is not one of the choices. The message names the choices,
            as ``choices_in_message`` does, and quotes nothing of the value: ``the value is not "ida" or "bo"``.
    """
    if not choice_texts(choices).issuperset(picks):
        raise conductor_error("not_a_choice", f"the value is not {choices_in_message(choices)}")


def choices_in_message(choices: Sequence[Choice]) -> str:
    """Names ``choices`` in a message, by their ids as JSON text, joined with "or".

    ``"ida" or "bo"``. Past five choices the rest are counted, so the message stays one line:
    ``"a", "b", "c", "d", "e" or 295 more``.
    """
    named = [canonical_json(choice.id) for choice in choices]
    if len(named) > 5:
        return f"{', '.join(named[:5])} or {len(named) - 5} more"
    *others, last = named
    return f"{', '.join(others)} or {last}" if others else last
