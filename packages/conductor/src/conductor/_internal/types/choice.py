"""Choices: a fixed list of values a person picks from.

A type can say which values it takes, with ``Param(choices=...)`` or a ``Literal``. A node that raises ``Ask``
can offer choices too. Each one is a ``Choice``, an id with a title::

    Annotated[str, Param(choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])]

Two steps use them. First, the choices are checked against the type and converted to JSON (``read_choices``).
Later, each value is checked against the choices. That second step compares JSON texts: ``choice_texts`` gives
the text of each choice, ``pick_texts`` the text of each value picked, and ``check_one_of`` refuses a pick that
isn't one of the choices.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, SchemaValidator

from conductor._internal.base import ConductorModel, canonical_json, validate_as_json
from conductor._internal.problem import conductor_error, value_free_errors


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
    reader: SchemaValidator | TypeAdapter[Any],
    write: Callable[[Any], JsonValue],
) -> tuple[Choice, ...]:
    """Checks the choices of a type, and returns them with each id in its JSON form.

    Each id is checked by ``reader`` strictly, the way its JSON text would be (``validate_as_json``). Then
    ``write`` turns it back into JSON. A bare value becomes a ``Choice`` titled by its text. A ``Choice`` keeps its
    class and all its fields.

    For a ``float``, ``read_choices([1, Choice(2, "Two")], adapter, write)`` gives
    ``(Choice(1.0, "1"), Choice(2.0, "Two"))``.

    Two ids are the same choice when their JSON texts match (``choice_texts``). So ``{"a": 1, "b": 2}`` and
    ``{"b": 2, "a": 1}`` are one choice given twice, which is refused.

    The caller's ``reader`` decides the rules. ``Param`` passes its own type, limits included. The ``Ask``
    validator passes the output's validator, which also applies ``validation.rules``: a yes or no is not a
    number.

    Args:
        given: The choices as a developer or a node gave them, each a value or a ``Choice``.
        reader: The validator of the type the choices are values of.
        write: Converts a value ``reader`` gave back to JSON.

    Raises:
        PydanticCustomError: ``no_json_form`` if an id can't be written as JSON. ``not_its_type`` if an id isn't
            a value of the type; its message is the type's reason, which quotes no value. ``no_choices`` if the
            list is empty. ``choice_twice`` if an id is given twice.
    """
    choices = [
        choice if isinstance(choice, Choice) else Choice(choice, str(choice)) for choice in given
    ]
    try:
        read = tuple(
            choice.model_copy(update={"id": write(validate_as_json(reader, choice.id))})
            for choice in choices
        )
    except PydanticSerializationError:
        raise conductor_error("no_json_form", "one can't be written as JSON") from None
    except ValidationError as refused:
        raise conductor_error("not_its_type", value_free_errors(refused)[0]["msg"]) from None
    if not read:
        raise conductor_error("no_choices", "none is given")
    if len(choice_texts(read)) != len(read):
        raise conductor_error("choice_twice", "one is given twice")
    return read


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


def check_one_of(texts: frozenset[str], picks: list[str]) -> None:
    """Checks that every pick is one of the choices.

    ``texts`` comes from ``choice_texts`` and ``picks`` from ``pick_texts``. For example, an input
    ``Annotated[str | None, Param(choices=["a"])]`` takes ``"a"``, and ``None`` too, since ``None`` is not a pick.

    Raises:
        PydanticCustomError: ``not_a_choice`` if a pick is not one of the choices. The message quotes nothing of
            the value.
    """
    if not texts.issuperset(picks):
        raise conductor_error("not_a_choice", "the value is not one of the choices")
