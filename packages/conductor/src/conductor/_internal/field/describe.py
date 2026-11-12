"""``FieldDescription``: one input or output of a node as JSON, for an editor to show."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, JsonValue
from pydantic import Field as PydanticField

from conductor._internal.base import ConductorModel


class FieldDescription(ConductorModel):
    """An input or output as an editor reads it, and a question a waiting call asks. It reads back what it wrote.

    Attributes:
        name: The parameter's name, or the output's.
        title: What a person reads as its name.
        description: A longer text for its help.
        is_series: Whether it is declared ``Series[T]``: an output that returns a list a graph can loop over, or an input
            that takes every item of a loop as one list.
        type: The id of one value's type, a ``Series``' item's; a table's names its columns.
        type_variables: The names of the type variables its hint leaves open, which an edge binds; two fields that
            share a name have the same type once it is bound.
        json_schema: The JSON schema of one value, written as ``schema``; a table's columns are in it.
        required: Whether nothing defaults it.
        default: The default as JSON; ``None`` when ``required``.
        choices: The values it takes when it names them, each ``{"id", "title"}`` with whatever a host's ``Choice``
            subclass adds: from a ``Literal``, ``Param(choices=)`` or, on a question, the ``Ask``.
        widget: The control a person edits it with, as JSON.
        exclusive_group: The group of alternative outputs it belongs to.
        loops: For an output of a subgraph returned inside loops: the lists looped over, outermost first,
            each by its address inside the subgraph, ``["documents.result", "lines.result"]``. Two outputs
            whose loops start the same are returned in the same loops, so a node outside that reads both is
            called once per item with the values of that item. Empty for every other field.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True, ser_json_inf_nan="strings",
    )

    name: str
    title: str
    description: str | None = None
    is_series: bool
    type: str
    type_variables: list[str] = []
    json_schema: dict[str, Any] = PydanticField(alias="schema")
    required: bool
    default: JsonValue = None
    choices: list[dict[str, Any]] | None = None
    widget: dict[str, Any] | None = None
    exclusive_group: str | None = None
    loops: list[str] = []
