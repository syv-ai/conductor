"""How a run stands: one rule reads its status from its outcomes and from how the last ``resume()`` left it.

A run's outcomes are its own. On the README graph, before any call the run is ``ready``; after
``graph.run(text="a\\nb\\nc")`` every node has an outcome for every call, so it is ``complete``; a run with a call that
waits for a person is ``waiting``, and once that call is answered it is ``ready`` again, or ``complete`` when nothing
is left to call. How the last ``resume()`` left the run is kept beside the outcomes: ``"running"`` in a copy made
while it went on (in memory only: a save never says it), or how it ended early. A call that failed (``failed``) or a series over the item limit
(``stopped``) is the run's: it stands until a ``resume()`` tries again, whatever is answered meanwhile. That the
resume was cancelled or timed out is that resume's: an answer is a step after it, so the answered run is read from
its outcomes, as is a run whose every call returned however the resume ended.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import ConfigDict

from conductor._internal.base import ItemIndexes, NodePath
from conductor._internal.problem import Problem, run_problems

from .flat import FlatFunctionNode
from .outcome import Outcome, Unanswered
from .progress import is_complete

#: The status of a run: ``run.status``. ``ready``, ``waiting`` and ``complete`` are read from its outcomes;
#: ``running`` is a copy made while a ``resume()`` went on; ``failed``, ``stopped``, ``cancelled`` and ``timed_out``
#: say how the last ``resume()`` ended it early.
RunStatus = Literal["ready", "running", "waiting", "complete", "failed", "stopped", "cancelled", "timed_out"]


@dataclass(frozen=True, slots=True)
class EndedEarly:
    """How ``resume()`` ended a run before it was complete, and why. ``resume()`` goes on from where it stopped.

    Attributes:
        status: ``failed`` (a node raised), ``stopped`` (a series is longer than the item limit, ``max_items``),
            ``cancelled`` (the stream's ``cancel`` was set) or ``timed_out`` (the ``timeout`` passed).
        cause: The problem that ended it, with its ``loc`` and its code.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    status: Literal["failed", "stopped", "cancelled", "timed_out"]
    cause: Problem


#: How the last ``resume()`` left a run: ``"running"`` in a copy made while it went on (a checkpoint, or the run a
#: stream hands back), how it ended early, or ``None`` when it made every call that was ready.
LastResume = EndedEarly | Literal["running"] | None


def status_of(
    last_resume: LastResume, nodes: Iterable[FlatFunctionNode], outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
) -> tuple[RunStatus, Problem | None]:
    """The status of a run, and the problem that ended it early when that is its status.

    ``running`` in a copy made while a ``resume()`` went on; else ``complete`` once every node is complete
    (``is_complete``); else how the last ``resume()`` ended early, with its cause; else
    ``waiting`` while a call waits for a person, and ``ready`` before that.

    Args:
        last_resume: How the last ``resume()`` left the run.
        nodes: Every function node of the run's graph and of its subgraphs.
        outcomes: The run's outcomes, by node path and item indexes.
    """
    if last_resume == "running":
        return "running", None
    if all(is_complete(outcomes, node) for node in nodes):
        return "complete", None
    if last_resume is not None:
        return last_resume.status, last_resume.cause
    waiting = any(type(outcome) is Unanswered for of_node in outcomes.values() for outcome in of_node.values())
    return ("waiting" if waiting else "ready"), None


def kept_by_an_answer(last_resume: LastResume) -> LastResume:
    """What a run keeps of how the last ``resume()`` left it once a call is answered: a call that failed, or a series
    over the item limit, which stand until a ``resume()`` tries again. That the resume was going on, was cancelled or
    timed out is that resume's, and the answered run comes after it."""
    return last_resume if isinstance(last_resume, EndedEarly) and last_resume.status in ("failed", "stopped") else None


def deadline(timeout: float | None) -> Callable[[], EndedEarly | None] | None:
    """A check a run makes before each call: ``EndedEarly("timed_out", ...)`` once ``timeout`` seconds have passed
    since this was called, else ``None``. ``None`` in place of the check when there is no timeout."""
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: EndedEarly("timed_out", run_problems.run_timed_out(timeout=timeout)) if time.monotonic() > end else None
