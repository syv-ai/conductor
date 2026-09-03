"""Compile-endpoint result type.

The ``{status, errors}`` envelope ``/compile`` returns: ``status`` is
``"error"`` when compilation raised (unknown node type, missing edge
endpoint, cycle, unknown compound) and the messages are in ``errors``;
``"ok"`` when the graph compiled. Type problems between placements are
not reported here until the compiler answers them itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CompileResult(BaseModel):
    """Envelope returned by the ``/compile`` endpoint."""

    status: Literal["ok", "error"]
    errors: list[str] = []
