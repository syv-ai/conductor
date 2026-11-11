"""How a run stands: going on, or ended early with a cause."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import ConfigDict

from conductor._internal.problem import Problem

#: How a run stands while nothing has ended it early: not started, in a leg, waiting for a person, or done.
GoingStatus = Literal["ready", "running", "complete", "waiting"]
#: How a leg ended before the run was done: a node failed, a Series outgrew ``max_items``, a cancel, or the deadline.
EndedStatus = Literal["failed", "stopped", "cancelled", "timed_out"]
#: How a run stands after a leg.
RunStatus = GoingStatus | EndedStatus


@dataclass(frozen=True, slots=True)
class Going:
    """A run nothing has ended early.

    Attributes:
        status: Not started (``ready``), in a leg (``running``), waiting for a person, or ``complete``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")  # a save that gives it a cause is refused
    status: GoingStatus


@dataclass(frozen=True, slots=True)
class EndedEarly:
    """A run whose last leg ended before it was done, and why; ``resume`` goes on from where it stopped.

    Attributes:
        status: ``failed`` (a node failed), ``stopped`` (a Series outgrew ``max_items``), ``cancelled`` or
            ``timed_out``.
        cause: Why, as a problem with its place and code.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    status: EndedStatus
    cause: Problem


#: How a run stands: going on, or ended early with a cause.
RunState = Going | EndedEarly

READY, RUNNING, WAITING, COMPLETE = Going("ready"), Going("running"), Going("waiting"), Going("complete")
