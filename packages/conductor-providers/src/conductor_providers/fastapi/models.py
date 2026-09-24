"""The pydantic payload for conductor's HTTP surface.

The long-lived contract between any conductor-backed server and the
clients that talk to it; bumping it is a breaking change for every host
using ``conductor_router``. The graph half of the payload is the ``Graph``
record itself — ``Graph`` is the schema, so nothing here
restates a field of it.
"""

from __future__ import annotations

from typing import Any

from conductor.execution.state import RunState
from conductor.graph.model import Graph
from pydantic import BaseModel, ConfigDict


class ExecuteRequest(BaseModel):
    """The POST body shared by ``/execute``, ``/execute-stream``, and ``/compile``, which reads only ``graph``."""

    model_config = ConfigDict(extra="ignore")

    graph: Graph
    # Outputs recorded by node id without running the node: a person's
    # answers to a pending leg, or an earlier run's results. The engine reports
    # a node the cache completes as ``node_complete`` with ``cached=True``. For
    # a node that runs per row, each output is a series in the form one is
    # dumped in, ``{"rows": [...], "values": [...]}``, naming only the rows it
    # answers. Unset = run all.
    cache: dict[str, Any] | None = None
    # The ``state`` of an earlier leg's ending, handed back so this leg goes
    # on from where that one stopped. Unset = a fresh run.
    state: RunState | None = None
