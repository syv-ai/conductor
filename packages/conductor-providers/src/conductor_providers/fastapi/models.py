"""The body of ``/run`` and ``/run-stream``, and what ``/run`` answers.

The graph half of the body is the ``Graph`` record itself, so nothing here restates a field of it.
"""

from __future__ import annotations

from typing import Any

from conductor import Graph, RunState
from pydantic import BaseModel, ConfigDict, Field


class RunRequest(BaseModel):
    """The POST body of ``/run`` and ``/run-stream``."""

    model_config = ConfigDict(extra="forbid")

    graph: Graph
    #: The graph's inputs, as ``compiled.with_inputs`` takes them, by name or address.
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: The state of an earlier leg, handed back so this leg goes on from where
    #: that one stopped. Unset = a fresh run.
    state: RunState | None = None
    #: A person's answers, by ``WaitingUnit.id``: a value per output, or
    #: ``null`` for no value on that row. Needs ``state``.
    answers: dict[str, dict[str, Any] | None] = Field(default_factory=dict)


class RunResult(BaseModel):
    """What ``/run`` returns: the leg's ``state`` and every node's ``results``, as JSON."""

    state: RunState
    results: dict[str, dict[str, Any]]
