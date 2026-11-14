"""``Choice``: one option a person can pick, with an id and a title; ``read_choices`` reads the choices of a type,
``choice_texts`` keys them, ``pick_texts`` keys the picks a value makes, and ``check_one_of`` holds the picks to
the choices."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError, SchemaValidator

from conductor._internal.base import ConductorModel, canonical_json, validate_as_json
from conductor._internal.problem import value_free_errors


class Choice(ConductorModel):
    """One option a person may pick: ``Choice("ida", "Ida B.")``.

    A host gives it more fields to carry more about an option, as keywords or on a subclass of its own; they reach
    the question, and a dropdown, as written, and a saved run or a description reads them back. A keyword it does
    not declare is kept, never refused, so a misspelt one is kept too.

    Attributes:
        id: The value an answer picking it holds.
        title: What a person reads for it.
    """

    model_config = ConfigDict(extra="allow")

    id: Any
    title: str

    def __init__(self, id: Any, title: str, **more: Any) -> None:  # noqa: A002  the word a host's option has
        """Makes a choice from its id and title given in that order.

        Args:
            id: The value an answer picking it holds.
            title: What a person reads for it.
            **more: The other fields a host gives it.
        """
        fields: dict[str, Any] = {
            "id": id,
            "title": title,
            **more,
        }  # the base's own signature names no fields
        super().__init__(**fields)


def read_choices(
    given: Iterable[Any],
    reader: SchemaValidator | TypeAdapter[Any],
    write: Callable[[Any], JsonValue],
) -> tuple[Choice, ...]:
    """Reads the choices of a type: each id read by ``reader`` strictly from its JSON form (``validate_as_json``)
    and written back in its JSON form by ``write``.

    ``read_choices([1, Choice(2, "Two")], adapter, write)`` with the adapter and writer of ``float`` is
    ``(Choice(1.0, "1"), Choice(2.0, "Two"))``: a bare value is titled by its text, and a ``Choice`` keeps its class
    and every field it has. Two ids are the same choice when their texts are (``choice_texts``), so
    ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` are one choice given twice. Who reads decides the rules:
    ``Param`` reads by its limited type, the ``Ask`` door by the gate's validator of the output, which holds every
    value to the gate's rules (a yes or no is not a number).

    Args:
        given: The choices as a developer or a node gave them, each a value or a ``Choice``.
        reader: The validator of the type the choices are values of.
        write: Writes a value ``reader`` gave in its JSON form.

    Raises:
        PydanticCustomError: An id has no JSON form (``no_json_form``) or is not a value of the type
            (``not_its_type``, whose message is the type's reason, quoting no value), or there are no choices
            (``no_choices``) or an id is given twice (``choice_twice``).
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
        raise PydanticCustomError("no_json_form", "one has no JSON form") from None
    except ValidationError as refused:
        raise PydanticCustomError("not_its_type", value_free_errors(refused)[0]["msg"]) from None
    if not read:
        raise PydanticCustomError("no_choices", "none is given")
    if len(choice_texts(read)) != len(read):
        raise PydanticCustomError("choice_twice", "one is given twice")
    return read


def choice_texts(choices: Iterable[Choice]) -> frozenset[str]:
    """The ids of ``choices`` as the texts a pick is looked up by (``canonical_json``), each id in its JSON form.

    Two ids are one choice when their texts are: ``1.0`` and ``1`` on a ``float`` are both read as ``1.0``, and
    ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` are one text.
    """
    return frozenset(canonical_json(choice.id) for choice in choices)


def pick_texts(written: Any, *, several: bool) -> list[str]:
    """The picks ``written`` makes, each as the text it is looked up by (``canonical_json``): ``written`` is one
    value in its JSON form, or with ``several`` a list of them.

    ``None``, the value or an item, is no pick: a type that takes ``None`` takes it beside its choices. So
    ``pick_texts(["a", None, {"b": 1}], several=True)`` is ``['"a"', '{"b":1}']``, and
    ``pick_texts(None, several=False)`` is ``[]``.
    """
    if written is None:
        return []
    return [
        canonical_json(item) for item in (written if several else [written]) if item is not None
    ]


def check_one_of(texts: frozenset[str], picks: list[str]) -> None:
    """Refuses ``picks`` (``pick_texts``) unless each is one of the choices whose ``choice_texts`` are ``texts``.

    An input ``Annotated[str | None, Param(choices=["a"])]`` takes ``"a"`` and ``None``, which is no pick.

    Raises:
        PydanticCustomError: A pick is not one of the choices (``not_a_choice``); the message quotes nothing of the
            value.
    """
    if not texts.issuperset(picks):
        raise PydanticCustomError("not_a_choice", "the value is not one of the choices")
