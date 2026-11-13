"""The state of a run: going on, or ended early, with the problem that ended it.

``resume()`` calls every node that is ready, until the run is complete, waiting, failed or stopped. Before it
starts the state is ``Going("ready")``; after ``graph.run(text="a\\nb\\nc")`` on the README graph it is
``Going("complete")``. When ``shout`` raises for one line the state is ``EndedEarly("failed", problem)``, and
``resume()`` calls ``shout`` for that line again.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import ConfigDict

from conductor._internal.problem import Problem, run_problems

#: The status of a run nothing has ended early: not started, being run by ``resume()``, waiting for a person, or complete.
GoingStatus = Literal["ready", "running", "complete", "waiting"]
#: Why ``resume()`` stopped before the run was complete: a node raised, a series is longer than the item limit
#: (``max_items``), the run was cancelled, or its ``timeout`` passed.
EndedStatus = Literal["failed", "stopped", "cancelled", "timed_out"]
#: The status of a run: ``run.status``.
RunStatus = GoingStatus | EndedStatus


@dataclass(frozen=True, slots=True)
class Going:
    """A run nothing has ended early: ``Going("complete")``.

    Attributes:
        status: ``ready`` (not started), ``running`` (``resume()`` is calling nodes), ``waiting`` (for a person) or
            ``complete``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")  # a saved run that gives it a cause is refused
    status: GoingStatus


@dataclass(frozen=True, slots=True)
class EndedEarly:
    """A run that ``resume()`` stopped before it was complete, and why. ``resume()`` goes on from where it stopped.

    Attributes:
        status: ``failed`` (a node raised), ``stopped`` (a series is longer than the item limit), ``cancelled`` or
            ``timed_out``.
        cause: The problem that ended it, with its ``loc`` and its code.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    status: EndedStatus
    cause: Problem


#: The state of a run: going on, or ended early with a cause.
RunState = Going | EndedEarly

READY, RUNNING, WAITING, COMPLETE = Going("ready"), Going("running"), Going("waiting"), Going("complete")


def deadline(timeout: float | None) -> Callable[[], EndedEarly | None] | None:
    """A check a run makes before each call: ``EndedEarly("timed_out", ...)`` once ``timeout`` seconds have passed
    since this was called, else ``None``. ``None`` in place of the check when there is no timeout."""
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: EndedEarly("timed_out", run_problems.run_timed_out(timeout=timeout)) if time.monotonic() > end else None
