"""``FieldDescription``: one input or output of a node as JSON, for an editor to show."""

from __future__ import annotations

from typing import Any, Self

from pydantic import MISSING, ConfigDict, JsonValue, SerializeAsAny, model_validator
from pydantic import Field as PydanticField
from pydantic_core import PydanticCustomError

from conductor._internal.base import ConductorModel
from conductor._internal.types import Choice


class FieldDescription(ConductorModel):
    """An input or output as an editor reads it, and a question a waiting call asks. It reads back what it wrote.

    ``required`` and ``default`` say what leaving the field out gives, and a host tells the three cases apart by them:

    - ``"default": "a"`` or ``"default": null``: the value it then has, ``null`` being ``None`` and nothing else.
      ``required`` is false.
    - no ``default`` key and ``"required": true``: leaving it out is refused, as for ``text: str``.
    - no ``default`` key and ``"required": false``: left out, it is a branch not chosen (``SKIPPED``): an output in
      an ``exclusive_group``, or an output whose default is ``SKIPPED`` (a field of a returned dataclass, or one a
      ``compute_outputs`` hook gives as ``(type, SKIPPED)``).

    In Python a ``default`` that is not there is pydantic's ``MISSING``, as an ``InputEntry``'s is. The record refuses
    parts that contradict each other: ``required`` with a ``default``, an output of an ``exclusive_group`` that is
    ``required`` or has a ``default``, and ``loops`` on a field that is not ``is_series``.

    Attributes:
        name: The parameter's name, or the output's.
        title: What a person reads as its name, as its ``Param(title=)`` declares; ``None`` when it declares none.
        description: A longer text for its help.
        is_series: Whether its value is a list a graph can loop over: an output declared ``Series[T]``, an input that
            takes every item of a loop as one list, or an output of a subgraph returned inside loops, a list per loop
            (``loops`` names them).
        type: The id of one value's type, the innermost item's: a ``Series``' item's, and for an output of a subgraph
            the value the node inside returns for one item of its innermost loop; a table's names its columns.
        type_variables: The names of the type variables its hint leaves open, which an edge binds; two fields that
            share a name have the same type once it is bound.
        json_schema: The JSON schema of one value, the same one value as ``type``, written as ``schema``; a table's
            columns are in it.
        required: Whether leaving it out is refused: an input with no default; an output whose question an answer
            must give (``-> str`` must, ``-> str | None`` may be left out and is then ``None``).
        default: What leaving it out gives, as JSON; ``MISSING``, and no key in the JSON, when that is no value:
            when it is ``required``, or left out it is a branch not chosen.
        choices: The values it takes when it names them, each a ``Choice`` with every field a host gave it: from a
            ``Literal``, ``Param(choices=)`` or, on a question, the ``Ask``.
        widget: The control a person edits it with, as JSON.
        exclusive_group: The group of alternative outputs it belongs to.
        constant_only: Whether an input takes only a constant, its field's ``constant_only``, so an editor offers it
            no edge and no graph input.
        loops: For an output of a subgraph returned inside loops: the lists it is one list per item of, outermost
            first, each as its source inside the subgraph is written, then the list the node inside returns when it
            returns a ``Series``. ``documents -> lines -> shout`` returning one ``str`` per line is
            ``["documents.result", "lines.result"]``, a list of lists of ``str``. How many levels of list the value
            has: one per entry of ``loops`` when it has any, else one when ``is_series`` (a ``Series`` returned outside
            every loop has ``loops`` ``[]`` and is one list), else none. Two outputs
            whose loops start the same are returned in the same loops, so a node outside that reads both is
            called once per item with the values of that item. Empty for every other field.
    """

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, serialize_by_alias=True
    )

    name: str
    title: str | None = None
    description: str | None = None
    is_series: bool
    type: str
    type_variables: list[str] = []
    json_schema: dict[str, Any] = PydanticField(alias="schema")
    required: bool
    default: JsonValue | MISSING = MISSING
    choices: tuple[SerializeAsAny[Choice], ...] | None = None
    widget: dict[str, Any] | None = None
    exclusive_group: str | None = None
    constant_only: bool = False
    loops: list[str] = []

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        """Refuses a field whose parts contradict each other, each with its own error type."""
        has_default = self.default is not MISSING
        in_group = self.exclusive_group is not None
        if self.required and has_default:
            raise PydanticCustomError(
                "required_with_default",
                "a required field has no default; leaving out a field with a default gives the default",
            )
        if self.required and in_group:
            raise PydanticCustomError(
                "required_in_exclusive_group",
                "an output of an exclusive group is never required; left out, it is a branch not chosen",
            )
        if has_default and in_group:
            raise PydanticCustomError(
                "default_in_exclusive_group",
                "an output of an exclusive group has no default; left out, it is a branch not chosen",
            )
        if self.loops and not self.is_series:
            raise PydanticCustomError(
                "loops_without_series",
                "a field with loops is a list (is_series), one level of list per loop",
            )
        return self
