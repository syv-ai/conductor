"""The constants door: a constant or a graph input's default written in a graph record, read by its input's type
when the graph is compiled.

Example: ``shout(line: str, mark: str = "!")`` with ``{"node": "shout", "inputs": {"mark": {"constant": "?"}}}`` gets
``"?"`` on every call. ``{"mark": {"constant": 3}}`` is refused with ``invalid_value`` at ``("shout", "mark")``,
which says what the type refused and quotes nothing of the value. A record holds JSON, so a value is read as its
JSON reads: ``"2026-01-02"`` is a ``date`` for an input that takes one.
"""

from __future__ import annotations

from typing import Any

from pydantic import JsonValue, ValidationError

from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, compile_problems

from .rules import validator_of


def read_constant(node_id: str, field: Field, value: JsonValue) -> Any:
    """Returns ``value``, the constant or graph input's default the graph record gives the input ``field`` of the node ``node_id``,
    read by the input's type.

    Raises:
        InvalidCallError: The type refuses the value: one ``invalid_value`` problem at ``(node_id, input)``.
    """
    try:
        return validator_of(field.whole).validate_python(value)
    except ValidationError as error:
        refused = field.whole.refusal(error)
        raise InvalidCallError([compile_problems.invalid_value(node_id, field.name, reason=refused.reason, errors=refused.errors, column=refused.column)]) from None

