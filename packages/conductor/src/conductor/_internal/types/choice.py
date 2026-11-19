"""Choices: a fixed list of values a person picks from.

A type can say which values it takes, with ``Param(choices=...)`` or a ``Literal``. A node that raises ``Ask``
can offer choices too. Each one is a ``Choice``, an id with a title::

    Annotated[str, Param(choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])]

Two steps use them. First, the choices are checked against the type and converted to JSON (``read_choices``).
Later, each value is checked against the choices. That second step compares JSON texts: ``choice_texts`` gives
the text of each choice, ``pick_texts`` the text of each value picked, and ``check_one_of`` refuses a pick that
isn't one of the choices. A message names the choices with ``choices_in_message``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from typing import Any

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError, SchemaValidator

from conductor._internal.base import (
    ConductorModel,
    canonical_json,
    finite_json_text,
)
from conductor._internal.problem import (
    NodeDeclarationError,
    conductor_error,
    declaration_problems,
    no_json_form,
    value_free_errors,
)

#: Why a list of choices is refused when one of them can't be written as JSON.
_ONE_HAS_NO_JSON_FORM = "one can't be written as JSON"


class Choice(ConductorModel):
    """One option a person can pick, with an id and a title: ``Choice("ida", "Ida B.")``.

    A host can add more fields to say more about an option, such as an icon. Pass them as keywords, or declare
    them on a subclass. They are passed on as given to the question and to a dropdown, and they survive a saved run
    and a description. A keyword the class doesn't declare is kept, not refused, so a misspelt one is kept too.

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


def read_choices(
    given: Iterable[Any],
    read: Callable[[Any], Any],
    write: Callable[[Any], JsonValue],
    known: Iterable[Choice] = (),
) -> tuple[Choice, ...]:
    """Checks the choices of a type, and returns them with each id in its JSON form.

    Each id is checked by ``read``, strictly, the way its JSON text would be. Then ``write`` turns it back into
    JSON. A bare value that is the id of a ``known`` choice becomes that choice, title and all. Any other bare value becomes a ``Choice`` titled by its text, and an enum member by its id's text
    (``"red"`` for ``Color.RED``, ``"1"`` for a member whose value is 1). A ``Choice`` keeps its class and all its
    fields.

    For a ``float``, ``read_choices([1, Choice(2, "Two")], read, write)`` gives
    ``(Choice(1.0, "1"), Choice(2.0, "Two"))``.

    Two ids are the same choice when their JSON texts match (``choice_texts``). So ``{"a": 1, "b": 2}`` and
    ``{"b": 2, "a": 1}`` are one choice given twice, which is refused.

    The caller's ``read`` decides the rules. ``Param`` reads by its own type, limits included
    (``validate_as_json``). The ``Ask`` validator reads by the output's type as an answer is read
    (``read_as_json``), so a choice a person couldn't give, such as ``inf``, is refused.

    Args:
        given: The choices as a developer or a node gave them, each a value or a ``Choice``.
        read: Reads one id by the type the choices are values of, raising pydantic's ``ValidationError`` for one
            the type refuses, ``no_json_form`` among them.
        write: Converts a value ``read`` gave back to JSON.
        known: Choices already read, which a bare value given by its id stands for. An ``Ask`` passes its
            output's own choices, so a bare ``"ida"`` is the ``Choice("ida", "Ida B.")`` the output's type gives.

    Raises:
        PydanticCustomError: ``no_json_form`` if an id can't be written as JSON. ``not_its_type`` if an id isn't
            a value of the type; its message is the type's reason, which quotes no value. ``no_choices`` if the
            list is empty. ``choice_twice`` if an id is given twice.
    """
    known_by_text = {canonical_json(choice.id): choice for choice in known}
    chosen: list[Choice] = []
    try:
        for choice in given:
            if isinstance(choice, Choice):
                written = write(read(choice.id))
                chosen.append(choice.model_copy(update={"id": written}))
            else:
                written = write(read(choice))
                chosen.append(
                    known_by_text.get(
                        canonical_json(written),
                        Choice(written, str(written) if isinstance(choice, Enum) else str(choice)),
                    )
                )
    except PydanticSerializationError:
        raise no_json_form(_ONE_HAS_NO_JSON_FORM) from None
    except ValidationError as refused:
        first = value_free_errors(refused)[0]
        # ``read`` refused it as no JSON could hold it, which is said of one choice among the rest
        if first["type"] == "no_json_form":
            raise no_json_form(_ONE_HAS_NO_JSON_FORM) from None
        raise conductor_error("not_its_type", first["msg"]) from None
    if not chosen:
        raise conductor_error("no_choices", "none is given")
    if len(choice_texts(chosen)) != len(chosen):
        raise conductor_error("choice_twice", "one is given twice")
    return tuple(chosen)


def declared_choices(
    given: Iterable[Any],
    reader: SchemaValidator | TypeAdapter[Any],
    write: Callable[[Any], JsonValue],
) -> tuple[Choice, ...]:
    """Reads the choices a type declares, as ``read_choices`` does, and refuses them as a fault in the declaration.

    ``Param`` reads its choices with it, and ``Type.choices`` a ``Literal``'s values or an enum's members, so a type
    whose choices can't be read is refused by ``@node``, and nothing that reads them later meets a bare pydantic
    error.

    Args:
        given: The choices, as ``Param``, the ``Literal`` or the enum gives them.
        reader: The validator of the type the choices are values of.
        write: Converts a value ``reader`` gave back to JSON.

    A choice is read from its JSON form, as an answer is, so ``inf`` and ``nan`` are refused: no one could pick them.

    Raises:
        NodeDeclarationError: ``choices_invalid``, or ``choices_invalid_no_json_form`` or ``choices_invalid_not_a_value`` when one fits.
    """

    def read(value: Any) -> Any:
        # a choice is given as an answer is, from its JSON form, so one that isn't finite could never be picked
        return reader.validate_json(finite_json_text(value, "choice"), strict=True)

    try:
        return read_choices(given, read, write)
    except PydanticCustomError as refused:
        match refused.type:
            case "no_json_form":
                problem = declaration_problems.choices_invalid_no_json_form()
            case "not_its_type":
                problem = declaration_problems.choices_invalid_not_a_value(reason=refused.message())
            case _:  # no choices, or an id given twice
                problem = declaration_problems.choices_invalid()
        raise NodeDeclarationError(problem) from None


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
