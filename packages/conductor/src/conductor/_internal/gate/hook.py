"""The hook door: what a node's ``compute_added_inputs`` or ``compute_outputs`` hook returns, read into fields when
the graph is compiled.

Example: a node ``fields(names: str)`` whose ``compute_outputs`` hook, given ``names="a,b"``, returns
``{"a": str, "b": (int, 0)}`` gets two outputs, ``a: str`` and ``b: int`` with the default 0. A hook returns each
field's type, or ``(type, default)`` as pydantic's ``create_model`` takes one. Given ``names="a.b"`` it returns
``{"a.b": str}``, which is the hook's failure, ``compute_hook_failed`` with ``error`` ``not_a_name``: a name holds none
of the characters places in a graph are written with (``Name``, the rule every name in a graph record is held to).
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

from conductor._internal.base import REQUIRED, Name
from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, NodeDeclarationError, compile_problems

from .declared import declared_field

#: What a hook returns: a type, or ``(type, default)``, for each name.
_HOOK_FIELDS = TypeAdapter(dict[Name, Any])


def read_hook_fields(node_id: str, kind: str, returned: Any) -> dict[str, Field]:
    """Returns what the hook ``kind`` of the node ``node_id`` returned, as fields by name.

    Raises:
        InvalidCallError: ``compute_hook_failed``, one at each name that cannot be one (``error`` ``not_a_name``), or
            one at the node when the hook returned no mapping (``error`` ``not_a_mapping``) or a type conductor does
            not support (``error`` the declaration code, as ``type_not_carried``).
    """
    try:
        hints = _HOOK_FIELDS.validate_python(returned)
    except ValidationError as invalid:
        if not invalid.errors()[0]["loc"]:  # the value returned is not a mapping
            reason = f"its {kind} hook returned a {type(returned).__name__}; a hook returns its fields by name, as a mapping of each name to its type"
            raise InvalidCallError([compile_problems.compute_hook_failed(node_id, reason=reason, error="not_a_mapping")]) from None
        raise InvalidCallError(
            compile_problems.compute_hook_failed(node_id, str(fault["loc"][0]), reason=f"what its {kind} hook returned has a field name it cannot have ({fault['msg']})", error="not_a_name")
            for fault in invalid.errors()
        ) from None
    try:
        return {name: declared_field(name, *_hint_and_default(hint)) for name, hint in hints.items()}
    except NodeDeclarationError as error:  # a type conductor does not support is the hook's failure
        raise InvalidCallError([compile_problems.compute_hook_failed(node_id, reason=str(error), error=error.problems[0].type)]) from None


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """Reads what a hook returns for one field: ``(type, default)``, or a type alone, which then has no default."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED
