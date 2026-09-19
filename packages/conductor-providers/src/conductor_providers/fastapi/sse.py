"""Server-Sent Events framing for conductor's execution stream, and the one JSON form a frame's values take."""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python


def _jsonable(value: Any) -> Any:
    """``value`` as JSON-ready data, the one place a frame or a result is serialised.

    A record (an ``ErrorCause``, an ``Input`` among a pending unit's
    questions) dumps through pydantic. A value pydantic has no rule for by
    itself — a ``Series`` in a result, a type a host declared — dumps
    through its own type's schema, so a series keeps its index and rows,
    and so does whatever that value holds: a series of ``Json`` from a node
    that iterates dumps each ``Json`` through its type in turn. A float that
    is not a number is ``null``, as it is inside a series. A value with no
    JSON form raises rather than turning into its repr.
    """
    return to_jsonable_python(value, inf_nan_mode="null", fallback=_through_its_type)


def _through_its_type(value: Any) -> Any:
    """``value`` dumped through ``TypeAdapter(type(value))``, and anything
    inside it the same way; a value its own type cannot dump raises."""

    def inner(unknown: Any) -> Any:
        if unknown is value:
            raise TypeError(f"a {type(value).__name__} has no JSON form")
        return _through_its_type(unknown)

    return TypeAdapter(type(value)).dump_python(value, mode="json", fallback=inner)


def sse_frame(event: Any) -> str:
    """Serialize a single conductor ``ExecutionEvent`` as an SSE ``data:`` frame.

    The frame ends with ``\\n\\n`` per the SSE spec.
    """
    return f"data: {json.dumps(_jsonable(event))}\n\n"
