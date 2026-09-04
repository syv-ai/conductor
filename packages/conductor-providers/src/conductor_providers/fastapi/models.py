"""The pydantic payload for conductor's HTTP surface.

The long-lived contract between any conductor-backed server and the
clients that talk to it; bumping it is a breaking change for every host
using ``conductor_router``. The graph half of the payload is the ``Flow``
record itself — ``TypeAdapter(Flow)`` is the schema, so nothing here
restates a field of it.
"""

from __future__ import annotations

from typing import Any

from conductor.graph.model import Flow
from pydantic import BaseModel, ConfigDict


class ExecuteRequest(BaseModel):
    """The POST body shared by ``/execute``, ``/execute-stream``, and ``/compile``."""

    model_config = ConfigDict(extra="ignore")

    flow: Flow
    # Optional precomputed node results, keyed by node id. Nodes listed here
    # are seeded as already-completed (the engine emits ``node_complete`` with
    # ``cached=True`` and skips running them), so a host can reuse outputs from
    # a previous run instead of recomputing the whole graph. Unset = run all.
    cache: dict[str, Any] | None = None
