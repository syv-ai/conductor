"""What a node's author sets beside its function and its fields: ``Policy``, how the engine calls the node
(``Policy(retries=3)``), and ``Deprecation``, the notice that a node is going away, which an editor shows.
"""

from __future__ import annotations

import functools

from pydantic import ConfigDict
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

from conductor._internal.base import ConductorModel


@dataclass(frozen=True, config=ConfigDict(strict=True))
class Policy:
    """How the engine calls a node, set by its author: ``@node("fetch", policy=Policy(retries=3))``.

    By default (``concurrency=None``) a run calls the node one item at a
    time, on its own thread, and takes no notice of other runs. A node that
    sets ``concurrency`` has at most that many calls running at once in the
    whole process, counted over every run and every place a graph uses it:
    with ``concurrency=2``, two runs that each use the node twice make two
    calls at once, not eight. That is how a node keeps an outside service's
    quota. Above 1 the calls are made on threads; with 1, or ``None``, and no
    ``attempt_timeout`` the node is called on the scheduler's own thread.

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

    Only a function node has one: a subgraph is never called, and each node inside it runs by its own policy. It is
    a frozen pydantic dataclass, never written as JSON, and its values are read strictly when it is made: a whole
    number is no yes or no and no text that spells one.

    Attributes:
        concurrency: How many calls of the node may run at once in the whole process; at least 1. ``None``, the
            default, sets no limit across runs: each run calls the node one item at a time. Setting it on a node
            whose function runs a graph that uses the node again (or a node that uses this one back) is a
            deadlock: the outer calls hold every place while the inner run waits for one. ``None`` has no such
            limit.
        retries: How many more times a call that failed outside the node's control is tried; at least 0.
        retry_delay: The seconds the first retry waits; finite and at least 0.
        attempt_timeout: The seconds one attempt may take; finite and above 0, or ``None`` for no clock.
        retry_on: The exception classes of the node's own client that are tried again like an ``ExternalError``.
        reuse_when_inputs_match: Whether ``run.again`` reuses a result when a call receives what it received before.
        costs_nothing: Whether a call costs nothing, so ``run.resume(only_costs_nothing=True)`` makes it. A node
            that takes the run's context is refused it.
    """

    concurrency: int | None = PydanticField(default=None, ge=1)
    retries: int = PydanticField(default=0, ge=0)
    retry_delay: float = PydanticField(default=1.0, ge=0, allow_inf_nan=False)
    attempt_timeout: float | None = PydanticField(default=None, gt=0, allow_inf_nan=False)
    retry_on: tuple[type[BaseException], ...] = ()
    reuse_when_inputs_match: bool = True
    costs_nothing: bool = False

    @functools.cached_property
    def most_at_once(self) -> int:
        """How many calls may run at once: ``concurrency``, or 1 when it is ``None``, counted then in each run and
        place in a graph on its own."""
        return 1 if self.concurrency is None else self.concurrency

    @functools.cached_property
    def threaded(self) -> bool:
        """Whether a call goes to a thread: several at once, or a clock on each attempt."""
        return self.most_at_once > 1 or self.attempt_timeout is not None


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
