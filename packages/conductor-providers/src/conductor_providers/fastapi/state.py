"""The two JSON columns the worker keeps in a run's row, and what a reader is shown.

A row's ``state`` is ``RunState``: where the run is, the replies queued while it runs, and a cancel asked of it. It is
small however large the run is. So queuing an answer or asking a cancel writes only it and the row's ``status``,
though the read before the write fetches the whole row. A row's ``outcome`` is ``RunOutcome``: what waits for a
person, the outputs, and how far the run is. It can be megabytes, so it is written only when the run stops and with
each checkpoint. A store keeps both as text and never reads them.

Every change of a state is a method here that returns a new state, so a test can call it on a ``Run`` with no store
at all. The one that takes a run, ``ended_with``, also returns the row's ``run`` and ``outcome`` to write with it.
``RunView`` is what a reader sees: the run's status and outcome, without the worker's own bookkeeping.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal

from conductor import (
    PROBLEMS,
    InvalidCallError,
    Problem,
    Run,
    RunProgress,
    RunStatus,
    Waiting,
)
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import from_json

from conductor_providers.fastapi.problems import raised_outside_a_node

log = logging.getLogger("conductor_providers.fastapi")

#: The cause of a run a cancel ended: conductor's own ``run_cancelled``, the cause ``run.stream(cancel=)`` gives.
_RUN_CANCELLED = {kind.code: kind for kind in PROBLEMS}["run_cancelled"]


class RunOutcome(BaseModel):
    """What a reader gets of a run without loading it: a row's ``outcome``, as JSON.

    Attributes:
        waiting: What waits for a person.
        outputs: The graph's outputs by name once the row's status is ``complete``; ``None`` until then.
        progress: ``run.progress``: how many calls of each node have an outcome, and how many there will be. It is
            written with each checkpoint too, so while the run is running it is as of the last checkpoint. ``None``
            for a run whose row couldn't be read as a run.
    """

    waiting: list[Waiting]
    outputs: dict[str, Any] | None
    progress: RunProgress | None

    @classmethod
    def of(cls, run: Run[Any], status: RunStatus) -> RunOutcome:
        """The outcome of a run: what waits in it, its progress and, once its row says it is complete, its outputs.

        A row's status can differ from its run's: a run that completes after a cancel was asked ends ``cancelled``,
        and one given its queue as its resume call fails ends ``failed``. Its outputs show only once the row says
        ``complete``, so a reader never sees outputs beside another status.

        Args:
            run: The run.
            status: The status of the row the outcome is written with.

        Returns:
            Its outcome, each output written by its type as the run writes it.

        Raises:
            InvalidCallError: An output can't be written as JSON (``run_unwritable``).
        """
        outputs = from_json(run.outputs_to_json()) if status == "complete" else None
        return cls(waiting=run.waiting, outputs=outputs, progress=run.progress)


class QueuedAnswer(BaseModel):
    """An answer to one question of a running run, kept in the run's state until the run stops.

    It reads ``{"kind": "answer", "waiting_id": "approve~3fa2c0d1", "answer": true}``.

    Attributes:
        kind: ``"answer"``, which tells it from a ``QueuedDecline`` in the row's JSON. A key it doesn't have, such
            as a ``reason``, is refused, so a row written wrong fails when it is read.
        waiting_id: The question it answers, as the run's ``waiting`` lists it.
        answer: The answer, as the JSON data ``run.with_answers`` reads it from.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["answer"] = "answer"
    waiting_id: str
    answer: Any


class QueuedDecline(BaseModel):
    """A decline of one question of a running run, kept in the run's state until the run stops.

    It reads ``{"kind": "decline", "waiting_id": "approve~3fa2c0d1", "reason": "Over budget"}``.

    Attributes:
        kind: ``"decline"``, which tells it from a ``QueuedAnswer`` in the row's JSON. A key it doesn't have, such
            as an ``answer``, is refused, so a row written wrong fails when it is read.
        waiting_id: The question it declines, as the run's ``waiting`` lists it.
        reason: Why the person declined, as ``run.with_decline`` keeps it; ``None`` for no reason given.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["decline"] = "decline"
    waiting_id: str
    reason: str | None


class RunState(BaseModel):
    """Where a run is, and what was asked of it while it was running: a row's ``state``, as JSON.

    For a running run that one answer was sent to, it reads ``{"status": "running", "cause": null,
    "queued": [{"kind": "answer", "waiting_id": "approve~3fa2c0d1", "answer": true}], "refused": {},
    "cancel_asked": false}``.

    ``status`` and ``cause`` repeat the run's own, so a route reads them without loading a run that can be
    megabytes. Three values are the worker's own, which no saved run can hold: ``running``, while a worker resumes the
    run or is about to; ``failed`` with a cause of the worker's, when a resume call raised and returned no run; and
    ``cancelled`` over a run that completed after a cancel was asked of it (see ``ended_with``).

    Attributes:
        status: ``running`` while a worker resumes the run or is about to; else the run's status as it was stored.
        cause: Why the run ended early; ``None`` for a running run, and for one that didn't end early.
        queued: The replies that came for the run's questions while it was running, one entry per reply, each to
            one waiting id, in the order they came. The run is given them when it stops.
        refused: By waiting id, the problems of the last answer or decline to that question, when the run refused
            it.
        cancel_asked: Whether someone asked the running run to stop. A waiting run that is cancelled is made
            running with it set, so its resume call ends before any node call.
    """

    status: RunStatus
    cause: Problem | None
    queued: list[Annotated[QueuedAnswer | QueuedDecline, Field(discriminator="kind")]]
    refused: dict[str, list[Problem]]
    cancel_asked: bool

    @classmethod
    def new(cls) -> RunState:
        """The state of a run just stored: running, since a worker is about to resume it, and nothing queued, refused
        or asked."""
        return cls(status="running", cause=None, queued=[], refused={}, cancel_asked=False)

    @property
    def is_running(self) -> bool:
        """Whether a worker resumes the run, or is about to."""
        return self.status == "running"

    def running(self) -> RunState:
        """This state with the run running again, for a worker to resume."""
        return self.model_copy(update={"status": "running", "cause": None})

    def with_queued(
        self,
        answers: Mapping[str, Any] = MappingProxyType({}),
        /,
        *,
        declines: Mapping[str, Any] = MappingProxyType({}),
    ) -> RunState:
        """This state with more replies queued after what was queued before: one per answer, then one per decline.

        What is queued is the JSON data ``Run.queued_as_json`` makes of the answers and the reasons, which the run
        will be given. That also refuses what JSON can't write, as the run would.

        Args:
            answers: Answers by waiting id, as ``run.with_answers`` takes them.
            declines: Reasons by waiting id, as ``run.with_decline`` takes them.

        Raises:
            InvalidCallError: An answer can't be written as JSON (``invalid_answer``), or a reason can't
                (``invalid_decline_reason``).
        """
        answered, reasons = Run.queued_as_json(answers, declines=declines)
        added = [
            *(QueuedAnswer(waiting_id=key, answer=value) for key, value in answered.items()),
            *(QueuedDecline(waiting_id=key, reason=reason) for key, reason in reasons.items()),
        ]
        return self.model_copy(update={"queued": [*self.queued, *added]})

    def asked_to_stop(self) -> RunState:
        """This state with a cancel asked."""
        return self.model_copy(update={"cancel_asked": True})

    def answered(self, waiting_ids: Collection[str]) -> RunState:
        """This state once the run took answers to some questions: nothing is refused for them any more.

        Args:
            waiting_ids: The waiting ids the run took answers to.

        Returns:
            The state without them in ``refused``.
        """
        refused = {key: value for key, value in self.refused.items() if key not in waiting_ids}
        return self.model_copy(update={"refused": refused})

    def ended_with(
        self, run: Run[Any] | None, *, saved: str | None = None, cause: Problem | None = None
    ) -> tuple[RunState, str | None, RunOutcome]:
        """This state once the run has stopped, given what was queued meanwhile, in the order it came.

        A run stops when its resume call ends or raises. A run that isn't running also stops here once it takes a
        person's decline, with nothing queued. Either way the row that stops holds no queue.
        ``python -m conductor.about answers-and-declines`` has the queue's rules. A reply whose check raises
        something other than a refusal (a fault in a host's type) is logged and refused as ``raised_outside_a_node``.

        A run that takes a decline has ended for good: it is ``declined``, and a reply queued after the decline is
        refused, as the run refuses it. Else a call that raised leaves the run ``failed`` with ``cause``. An answer
        to a question the run asked after its last checkpoint is then refused as ``not_waiting``, and that question
        waits again on the next resume. When the row's run couldn't be loaded, there is no run to give the queue
        to, so each queued reply is refused with ``cause``.

        Else a run that takes any answer is running again, for the worker to resume at once. A cancel asked that the
        run didn't end on then stays asked, and ends the next resume call before any node call. So does one asked of
        a run that ends ``waiting``. A run that completes with a cancel asked ends ``cancelled``. The cancel came
        before the run's end was written, so the run ends as the cancel's answer said it would, even though its last
        calls were made: another worker hears a cancel only at its next checkpoint or lease renewal. Every call's
        outcome is kept, so a resume makes no call and completes the run. Like any row that isn't complete, it shows
        no outputs until then. A run that ended early keeps how it ended, and any other cancel is cleared.

        The row shows the outcome of the run it keeps, the one given the queue. A run that is running again shows
        the run as its call returned it instead, before the answers it took, so what a reader sees changes only when
        the run stops again.

        Args:
            run: The run that stopped: the run a resume call returned, the run as the row has it (its last
                checkpoint) for a call that raised, or a run that took a decline. ``None`` when the row's run
                couldn't be loaded.
            saved: ``run`` as JSON, when the row doesn't keep it yet; ``None`` when it does.
            cause: Why the resume call raised; ``None`` for one that didn't.

        Returns:
            The new state, the row's ``run`` (as JSON) to write with it, ``None`` to keep the row's, and its
            ``outcome``.
        """
        if run is None:
            given, taken, refused = None, False, dict(self.refused)
        else:
            given, taken, refused = self._given_to(run)
        # a cancel asked that the run didn't end on: this resume call heard it too late, or not at all
        unheard = self.cancel_asked and run is not None and run.status != "cancelled"
        status: RunStatus
        if given is not None and given.status == "declined":
            status, why, cancel_asked = given.status, given.cause, False
        elif cause is not None:
            status, why, cancel_asked = "failed", cause, False
            if given is None:
                refused |= {reply.waiting_id: [cause] for reply in self.queued}
        elif given is None:
            raise ValueError("ended_with takes the cause when the row's run couldn't be loaded")
        elif taken or (unheard and given.status == "waiting"):
            status, why, cancel_asked = "running", None, unheard
        elif unheard and given.status == "complete":
            status, why, cancel_asked = "cancelled", _RUN_CANCELLED(), False
        else:
            status, why, cancel_asked = given.status, given.cause, False
        state = self.model_copy(
            update={
                "status": status,
                "cause": why,
                "queued": [],
                "refused": refused,
                "cancel_asked": cancel_asked,
            }
        )
        if run is None or given is None:
            return state, None, RunOutcome(waiting=[], outputs=None, progress=None)
        shown = run if state.is_running else given
        return state, given.to_json() if taken else saved, RunOutcome.of(shown, status)

    def _given_to(self, run: Run[Any]) -> tuple[Run[Any], bool, dict[str, list[Problem]]]:
        """``run`` once it is given the queued replies, in the order they came.

        Args:
            run: The run to give it to.

        Returns:
            The run given it, whether it took any of it, and ``refused`` with what it refused.
        """
        given, taken, refused = run, False, dict(self.refused)
        for reply in self.queued:
            try:
                given = (
                    given.with_decline(reply.waiting_id, reason=reply.reason)
                    if isinstance(reply, QueuedDecline)
                    else given.with_answers(reply.waiting_id, reply.answer)
                )
            except InvalidCallError as error:
                refused[reply.waiting_id] = error.problems
            except Exception as error:
                log.exception(
                    "the answer or decline queued for %s could not be checked", reply.waiting_id
                )
                refused[reply.waiting_id] = [raised_outside_a_node(error=type(error).__name__)]
            else:
                taken = True
                refused.pop(reply.waiting_id, None)
        return given, taken, refused


class RunView(RunOutcome):
    """A run as a reader sees it, as it was when it last stopped running, read without loading the run.

    ``Worker.read`` gives it, and every route under ``/runs`` but the events route answers with it, so an action
    answers with the run as it wrote it. It shows only what a reader needs: the replies a running run has queued, and a cancel asked of it, are the worker's own business and aren't shown.

    Attributes:
        id: The run's id.
        metadata: The run's metadata, as the JSON object the row keeps: what the host gave when the run started,
            written by alias. The worker reads it back by its ``metadata_type`` for the host's own code.
        status: Where the run is: ``running`` while a worker resumes it or is about to, ``waiting`` for answers,
            ``complete``, how it ended early (``failed``, ``stopped``, ``cancelled``, ``timed_out``), which
            ``Worker.resume`` goes on from, or ``declined``, which a person's decline ended for good.
        cause: Why the run ended early or was declined; ``None`` while it runs, waits or once complete.
        waiting: What waits for a person, as of when the run last stopped running; each ``id`` is what
            ``Worker.answer`` takes. When the worker raised while resuming the run, what waits in the run's last
            checkpoint.
        outputs: The graph's outputs by name once the run is complete, each as its type writes it (a table as its
            columns); ``None`` until then.
        progress: How many calls of each node have an outcome, and how many there will be, as ``run.progress``
            says it: as of the last checkpoint while the run is running.
        refused: By waiting id, the problems of the last answer or decline sent to that question while the run
            was running, when the run refused it once it stopped.
    """

    id: str
    metadata: dict[str, Any]
    status: RunStatus
    cause: Problem | None
    refused: dict[str, list[Problem]]
