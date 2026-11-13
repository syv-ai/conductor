"""``Choice``: one option a person can pick, with an id and a title; ``read_choices`` reads the choices of a type."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError, SchemaValidator, to_json

from conductor._internal.base import ConductorModel, validate_as_json
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
        fields: dict[str, Any] = {"id": id, "title": title, **more}  # the base's own signature names no fields
        super().__init__(**fields)


def read_choices(given: Iterable[Any], reader: SchemaValidator | TypeAdapter[Any], write: Callable[[Any], JsonValue]) -> tuple[Choice, ...]:
    """Reads the choices of a type: each id read by ``reader`` strictly from its JSON form (``validate_as_json``)
    and written back in its JSON form by ``write``.

    ``read_choices([1, Choice(2, "Two")], adapter, write)`` with the adapter and writer of ``float`` is
    ``(Choice(1.0, "1"), Choice(2.0, "Two"))``: a bare value is titled by its text, and a ``Choice`` keeps its class
    and every field it has. Two ids are the same choice when their JSON is. Who reads decides the rules: ``Param``
    reads by its limited type, the ``Ask`` door by the gate's validator of the output, which holds every value to
    the gate's rules (a yes or no is not a number).

    Args:
        given: The choices as a developer or a node gave them, each a value or a ``Choice``.
        reader: The validator of the type the choices are values of.
        write: Writes a value ``reader`` gave in its JSON form.

    Raises:
        PydanticCustomError: An id has no JSON form (``no_json_form``) or is not a value of the type
            (``not_its_type``, whose message is the type's reason, quoting no value), or there are no choices
            (``no_choices``) or an id is given twice (``choice_twice``).
    """
    choices = [choice if isinstance(choice, Choice) else Choice(choice, str(choice)) for choice in given]
    try:
        read = tuple(choice.model_copy(update={"id": write(validate_as_json(reader, choice.id))}) for choice in choices)
    except PydanticSerializationError:
        raise PydanticCustomError("no_json_form", "one has no JSON form") from None
    except ValidationError as refused:
        raise PydanticCustomError("not_its_type", value_free_errors(refused)[0]["msg"]) from None
    if not read:
        raise PydanticCustomError("no_choices", "none is given")
    if len({to_json(choice.id) for choice in read}) != len(read):
        raise PydanticCustomError("choice_twice", "one is given twice")
    return read
