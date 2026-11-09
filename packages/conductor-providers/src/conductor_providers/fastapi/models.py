"""The pydantic payload for conductor's HTTP surface.

The long-lived contract between any conductor-backed server and the
clients that talk to it; bumping it is a breaking change for every host
using ``conductor_router``. The graph half of the payload is the ``Graph``
record itself — ``Graph`` is the schema, so nothing here
restates a field of it.
"""

from __future__ import annotations

from typing import Any

from conductor import Graph, RunState
from pydantic import BaseModel, ConfigDict, Field


class RunRequest(BaseModel):
    """The POST body shared by ``/run``, ``/run-stream``, and ``/compile``, which reads only ``graph``."""

    model_config = ConfigDict(extra="ignore")

    graph: Graph
    #: The state of an earlier leg, handed back so this leg goes on from where
    #: that one stopped. Unset = a fresh run.
    state: RunState | None = None
    #: A person's answers, by ``WaitingUnit.id``: a value per output, or
    #: ``null`` for no value on that row. Needs ``state``.
    answers: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
