"""The constants door: a constant or default written in a graph record, read by its input's type when the graph is
compiled.

Example: ``shout(line: str, mark: str = "!")`` with ``{"id": "shout", "node": "shout", "constants": {"mark": "?"}}``
gets ``"?"`` on every call. ``{"constants": {"mark": 3}}`` is refused with ``invalid_value`` at ``("shout", "mark")``,
which says what the type refused and quotes nothing of the value. A record holds JSON, so a value is read as its
JSON reads: ``"2026-01-02"`` is a ``date`` for an input that takes one.
"""

from __future__ import annotations

from typing import Any

from pydantic import JsonValue, ValidationError

from conductor._internal.field import Field
from conductor._internal.gate.refused import RefusedError
from conductor._internal.problem import catalogue as problem


def read_constant(node_id: str, field: Field, value: JsonValue) -> Any:
    """Returns ``value``, the constant or default the graph record gives the input ``field`` of the node ``node_id``,
    read by the input's type.

    Raises:
        RefusedError: The type refuses the value: one ``invalid_value`` problem at ``(node_id, input)``.
    """
    try:
        return field.whole.read(value)
    except ValidationError as error:
        refused = field.whole.refusal(error.errors(include_url=False, include_input=False))
        raise RefusedError([problem.invalid_value(node_id, field.name, reason=refused.reason, errors=refused.errors, column=refused.column)]) from None
