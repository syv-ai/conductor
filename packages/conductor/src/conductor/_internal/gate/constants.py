"""The constants door: a constant or a graph input's default written in a graph record, read by its input's type
when the graph is compiled; and a default a node declares, read again by the type an edge binds its type variable to.

Example: ``shout(line: str, mark: str = "!")`` with ``{"node": "shout", "inputs": {"mark": {"constant": "?"}}}`` gets
``"?"`` on every call. ``{"mark": {"constant": 3}}`` is refused with ``invalid_value`` at ``("shout", "mark")``,
which says what the type refused and quotes nothing of the value. A record holds JSON, so a value is read as its
JSON reads: ``"2026-01-02"`` is a ``date`` for an input that takes one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import JsonValue, ValidationError

from conductor._internal.base import REQUIRED, SKIPPED
from conductor._internal.field import Field
from conductor._internal.problem import catalogue as problem
from conductor._internal.types import Type

from .refused import RefusedError
from .rules import validator_of


def read_constant(node_id: str, field: Field, value: JsonValue) -> Any:
    """Returns ``value``, the constant or graph input's default the graph record gives the input ``field`` of the node ``node_id``,
    read by the input's type.

    Raises:
        RefusedError: The type refuses the value: one ``invalid_value`` problem at ``(node_id, input)``.
    """
    try:
        return validator_of(field.whole).validate_python(value)
    except ValidationError as error:
        refused = field.whole.refusal(error)
        raise RefusedError([problem.invalid_value(node_id, field.name, reason=refused.reason, errors=refused.errors, column=refused.column)]) from None


def bound_field(node_id: str, field: Field, bound: Mapping[TypeVar, Any]) -> Field:
    """Returns ``field`` with its type variables replaced by what the edges into the node ``node_id`` bind them to
    (``bound``), and its default read by the type it then has; ``field`` itself when its type has no variable.

    For ``same(value: T = "-")`` wired from an output of ``str``, ``T`` is bound to ``str`` and the default stays
    ``"-"``. A variable no edge binds is read as its bound, else ``Any``.

    Raises:
        RefusedError: The bound type refuses the default the node declares: one ``invalid_value`` problem (variant
            ``bound_default``) at ``(node_id, field)``.
    """
    if not field.whole.variables:
        return field
    hint = Type(field.hint).bind(bound).hint
    if field.default is REQUIRED or field.default is SKIPPED or field.default is None:
        return Field(field.name, hint, field.default)
    whole = Field(field.name, hint).whole
    try:
        return Field(field.name, hint, validator_of(whole).validate_python(field.default))
    except ValidationError as error:
        refused = whole.refusal(error)
        raise RefusedError([problem.invalid_value_bound_default(node_id, field.name, type=whole.id, reason=refused.reason, errors=refused.errors)]) from None
