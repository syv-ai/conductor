"""Compile-endpoint result types.

Mirrors ``conductor.graph.type_check.TypeWarning`` as pydantic for
JSON serialization at the transport boundary, plus a top-level
``{status, errors, warnings}`` envelope distinguishing hard
``CompilationError``s from soft type warnings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CompileWarning(BaseModel):
    """Per-edge or per-consume warning from conductor's type checker."""

    edge_id: str
    code: str
    message: str
    source_node: str
    source_output: str
    source_type: str
    target_node: str
    target_input: str
    target_type: str


class CompileResult(BaseModel):
    """Envelope returned by the ``/compile`` endpoint.

    ``status`` is ``"error"`` when compilation itself raised
    (unknown node type, missing edge endpoint, cycle, unknown compound) —
    those go in ``errors``. ``status`` is ``"ok"`` when the graph
    compiled, even if there were soft type warnings; those go in
    ``warnings`` so the frontend can paint offending edges red without
    blocking the Run button.
    """

    status: Literal["ok", "error"]
    errors: list[str] = []
    warnings: list[CompileWarning] = []
