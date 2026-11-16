"""Settings a node's author gives beside the function: ``Policy`` and ``Deprecation``.

``Policy`` says how the engine calls the node: how many calls may run at once, and whether a failed call is tried
again. ``Deprecation`` is a notice, shown in an editor, that a version of the node is going away.
"""

from __future__ import annotations

import functools
import threading
from typing import Final

from pydantic import ConfigDict
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

from conductor._internal.base import ConductorModel

#: The longest wait, in seconds, a policy may set and the scheduler makes: ``threading.TIMEOUT_MAX``, the most a
#: thread can wait for anything, about 292 years. A retry's delay, doubled at each retry, stops growing there.
LONGEST_WAIT: Final = threading.TIMEOUT_MAX


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class Policy:
    """How the engine calls a node, as its author sets it.

    For example, ``@node("fetch", policy=Policy(concurrency=2, retries=3))`` makes at most two calls of ``fetch``
    at once, and tries a call that failed outside the node's control three more times.

    ``concurrency`` limits calls across the whole process. With ``concurrency=2``, two runs that each use ``fetch``
    twice still make only two calls at once, not four. That is how a node stays inside an outside service's quota.
    Without it, each run calls the node one item at a time and ignores other runs.

    Only some failures are tried again: an ``ExternalError``, or an exception class named in ``retry_on``. Anything
    else fails at once, since running the same computation again fails the same way. A ``NodeError`` other than
    ``ExternalError``, an output its type refuses and an ``Ask`` are never tried again, even with
    ``retry_on=(Exception,)``.

    ``reuse_when_inputs_match=False`` is for a node that reads the outside world, not only its inputs. A run made
    from a previous one (``run.again``, ``run.with_inputs``, ``graph.with_inputs(previous=)`` or ``run.forget``)
    then calls it again for every item. ``costs_nothing=True`` is for a cheap node that touches nothing outside the
    run, like one that splits a table. ``run.resume(only_costs_nothing=True)`` calls only such nodes, so a host can
    see how many calls the costly nodes will make before any of them runs.

    Only a function node has a policy. A subgraph is never called itself, and each node inside it has its own. The
    values are checked strictly when the policy is made, so ``1`` isn't a yes or no, and a keyword it
    doesn't have, like ``timeout``, is refused. The full rules are in the
    reference: ``python -m conductor.about versions-and-policies`` and ``retries-and-timeouts``.

    Attributes:
        concurrency: How many calls of the node may run at once in the whole process, at least 1. ``None``, the
            default, means each run calls the node one item at a time. Careful with a node whose function runs a
            graph that uses this node again, or a node that uses this one back. Once the outer calls take every
            place, the inner run waits for one forever. With ``concurrency=2``, two items or two runs calling the
            node at once are already enough.
        retries: How many more times a failed call is tried, at least 0.
        retry_delay: How many seconds the first retry waits. Each later retry waits twice as long, up to
            ``LONGEST_WAIT``.
        attempt_timeout: How many seconds one attempt may take, or ``None`` for no limit. An attempt that runs over
            fails the call, and the run, even after a cancel, and isn't retried. Python can't stop a thread, so the
            attempt is abandoned, not killed: it runs on until its function returns, and a later ``resume()`` may
            make the call again beside it. A node with side effects guards against that itself, or sets
            ``concurrency``, which counts the abandoned attempt until it returns.
        retry_on: Exception classes of the node's own client to try again, like an ``ExternalError``.
        reuse_when_inputs_match: Whether a run made from a previous one reuses a result when a call gets the same
            inputs as before. It is checked when the new run is made, and again by ``Run.from_json``, which drops
            the outcomes a saved run could still reuse for a node that says ``False`` now.
        costs_nothing: Whether a call is cheap enough for ``run.resume(only_costs_nothing=True)`` to make. A node
            that takes the run's context can't set it.
    """

    concurrency: int | None = PydanticField(default=None, ge=1)
    retries: int = PydanticField(default=0, ge=0)
    retry_delay: float = PydanticField(default=1.0, ge=0, le=LONGEST_WAIT)
    attempt_timeout: float | None = PydanticField(default=None, gt=0, le=LONGEST_WAIT)
    retry_on: tuple[type[BaseException], ...] = ()
    reuse_when_inputs_match: bool = True
    costs_nothing: bool = False

    @functools.cached_property
    def most_at_once(self) -> int:
        """How many calls may run at once: ``concurrency``, or 1 when it is ``None``.

        With ``None``, the 1 counts within one run and one place in a graph, not across the process.
        """
        return 1 if self.concurrency is None else self.concurrency

    @functools.cached_property
    def threaded(self) -> bool:
        """Whether calls are made on threads: when several may run at once, or each attempt has a time limit."""
        return self.most_at_once > 1 or self.attempt_timeout is not None


class Deprecation(ConductorModel):
    """A notice that a version of a node is going away, for an editor to show beside the node.

    A graph that uses a deprecated node still works, so this is never a ``Problem``. Every part is optional text.

    Attributes:
        header: The notice in one line, for a badge or a title, like ``"Use version 2"``.
        description: Why the version is going away, in a sentence or two.
        alternative: The id of the node to use instead.
        migration: How to move a graph off this version.
    """

    header: str | None = None
    description: str | None = None
    alternative: str | None = None
    migration: str | None = None
