"""What a node's author sets beside its function and its fields: ``Policy``, how the engine calls the node
(``Policy(retries=3)``), and ``Deprecation``, the notice that a node is going away, which an editor shows.
"""

from __future__ import annotations

import functools

from pydantic import ConfigDict
from pydantic import Field as PydanticField

from conductor._internal.base import ConductorModel


class Policy(ConductorModel):
    """How the engine calls a node, set by its author: ``@node("fetch", policy=Policy(retries=3))``.

    ``concurrency`` calls are made at once, on threads; with 1 and no
    ``attempt_timeout`` the node is called in order, on the scheduler's thread.
    A call that raises ``ExternalError``, or one of the ``retry_on`` classes
    of the node's own client, is tried ``retries`` more times, waiting from
    ``retry_delay`` seconds (the reference has the schedule); anything else
    fails at once, since repeating a computation repeats its failure. A
    ``NodeError`` other than ``ExternalError``, an output its type refuses
    and an ``Ask`` are never tried again, whatever ``retry_on`` names. An
    attempt that runs past ``attempt_timeout`` fails the call, finally: its
    thread can't be stopped, and a second call beside it would repeat its
    effect. ``reuse_when_inputs_match=False`` is for a node that reads the
    world, not only its inputs: ``run.again`` calls it for every item.
    ``costs_nothing=True`` is for a node whose call costs nothing and touches
    nothing outside the run (one that hands on an upload's files, splits a
    table, or returns a list given to it): ``run.resume(only_costs_nothing=True)``
    calls only such nodes, so a host learns how many calls the costly nodes will
    have (``run.progress``) before any of them is made.

    Only a function node has one: a subgraph is never called, and each node inside it runs by its own policy. As
    JSON it leaves ``retry_on`` out, since a class has no JSON form. Its values are read strictly: a whole number is
    no yes or no and no text that spells one.

    Attributes:
        concurrency: How many calls are made at once; at least 1.
        retries: How many more times a call that failed outside the node's control is tried; at least 0.
        retry_delay: The seconds the first retry waits; finite and at least 0.
        attempt_timeout: The seconds one attempt may take; finite and above 0, or ``None`` for no clock.
        retry_on: The exception classes of the node's own client that are tried again like an ``ExternalError``.
        reuse_when_inputs_match: Whether ``run.again`` reuses a result when a call receives what it received before.
        costs_nothing: Whether a call costs nothing, so ``run.resume(only_costs_nothing=True)`` makes it. A node
            that takes the run's context is refused it.
    """

    model_config = ConfigDict(strict=True)

    concurrency: int = PydanticField(default=1, ge=1)
    retries: int = PydanticField(default=0, ge=0)
    retry_delay: float = PydanticField(default=1.0, ge=0, allow_inf_nan=False)
    attempt_timeout: float | None = PydanticField(default=None, gt=0, allow_inf_nan=False)
    retry_on: tuple[type[BaseException], ...] = PydanticField(default=(), exclude=True)
    reuse_when_inputs_match: bool = True
    costs_nothing: bool = False

    @functools.cached_property
    def threaded(self) -> bool:
        """Whether a call goes to a thread: several at once, or a clock on each attempt."""
        return self.concurrency > 1 or self.attempt_timeout is not None


class Deprecation(ConductorModel):
    """A notice that a version of a node is going away, which an editor shows beside the node.

    A graph that uses a deprecated node is not wrong, so this is never a ``Problem``. Every part is optional prose.

    Attributes:
        header: The notice in one line, as a badge or a title shows it: ``"Use version 2"``.
        description: Why the version is going away, in a sentence or two.
        alternative: The id of the node to use instead.
        migration: How to move a graph off this version.
    """

    header: str | None = None
    description: str | None = None
    alternative: str | None = None
    migration: str | None = None
