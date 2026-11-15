"""The two JSON columns the worker keeps in a run's row, every way the first one changes, and what a reader is shown.

A row's ``state`` is ``RunState``: where the run is, the answers queued while it runs, and a cancel asked of it. It
is small however large the run is, so queuing an answer or asking a cancel reads and writes only it. A row's
``outcome`` is ``RunOutcome``: what waits for a person, the outputs, and how far the run is. It can be megabytes,
so it is written only when the run stops and with each checkpoint. A store keeps both as text and never reads them.

Every change of a state is a method here that returns a new state, so a test can call it on a ``Run`` with no store
at all. ``RunView`` is what a reader sees of the two: the run's status and outcome, without the worker's own
bookkeeping.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any

from conductor import InvalidCallError, Problem, Run, RunProgress, RunStatus, Waiting
from pydantic import BaseModel
from pydantic_core import from_json
from typing_extensions import TypeVar

from conductor_providers.fastapi.problems import raised_outside_a_node
from conductor_providers.fastapi.store import RunChanges

#: The type of a run's metadata: what the host keeps on it. ``dict[str, str]`` unless the host names a model.
M = TypeVar("M", default=dict[str, str])

log = logging.getLogger("conductor_providers.fastapi")


class RunOutcome(BaseModel):
    """What a reader gets of a run without loading it: a row's ``outcome``, as JSON.

    Attributes:
        waiting: What waits for a person.
        outputs: The graph's outputs by name once the run is complete; ``None`` until then.
        progress: ``run.progress``: how many calls of each node have an outcome, and how many there will be. It is
            written with each checkpoint too, so while the run is running it is as of the last checkpoint. ``None``
            for a run whose row couldn't be read as a run.
    """

    waiting: list[Waiting]
    outputs: dict[str, Any] | None
    progress: RunProgress | None

    @classmethod
    def of(cls, run: Run[Any]) -> RunOutcome:
        """The outcome of a run: what waits in it, its progress and, once it is complete, its outputs.

        Args:
            run: The run.

        Returns:
            Its outcome, each output written by its type as the run writes it.

        Raises:
            InvalidCallError: An output can't be written as JSON (``run_unwritable``).
        """
        outputs = from_json(run.outputs_to_json()) if run.status == "complete" else None
        return cls(waiting=run.waiting, outputs=outputs, progress=run.progress)


class RunState(BaseModel):
    """Where a run is, and what was asked of it while it was running: a row's ``state``, as JSON.

    For a running run that one answer was sent to, it reads ``{"status": "running", "cause": null,
    "queued_answers": [{"approve~3fa2c0d1": true}], "refused": {}, "cancel_asked": false}``.

    ``status`` and ``cause`` repeat the run's own, so a route reads them without loading a run that can be
    megabytes. Two values are the worker's own, which no saved run can hold: ``running``, while a worker resumes the
    run or is about to, and ``failed`` with a cause of the worker's, when a resume call raised and returned no run.

    Attributes:
        status: ``running`` while a worker resumes the run or is about to; else the run's status as it was stored.
        cause: Why the run ended early; ``None`` for a running run, and for one that didn't end early.
        queued_answers: The answers that came while the run was running, each mapping as one call sent it, in the
            order they came. The run is given them when it stops.
        refused: By waiting id, the problems of the last answer to that question, when the run refused it.
        cancel_asked: Whether someone asked the running run to stop. A waiting run that is cancelled is made
            running with it set, so its resume call ends before any node call.
    """

    status: RunStatus
    cause: Problem | None
    queued_answers: list[dict[str, Any]]
    refused: dict[str, list[Problem]]
    cancel_asked: bool

    @classmethod
    def started(cls) -> RunState:
        """The state of a run that has just started: running, and nothing queued, refused or asked."""
        return cls(status="running", cause=None, queued_answers=[], refused={}, cancel_asked=False)

    @property
    def is_running(self) -> bool:
        """Whether a worker resumes the run, or is about to."""
        return self.status == "running"

    def running(self) -> RunState:
        """This state with the run running again, for a worker to resume."""
        return self.model_copy(update={"status": "running", "cause": None})

    def queued(self, answers: Mapping[str, Any]) -> RunState:
        """This state with one more call's answers queued, after those queued before."""
        return self.model_copy(update={"queued_answers": [*self.queued_answers, dict(answers)]})

    def asked_to_stop(self) -> RunState:
        """This state with a cancel asked."""
        return self.model_copy(update={"cancel_asked": True})

    def answered(self, waiting_ids: Collection[str]) -> RunState:
        """This state once the run took answers to some questions: nothing is refused or queued for them any more.

        Args:
            waiting_ids: The waiting ids the run took answers to.

        Returns:
            The state without them in ``refused`` and ``queued_answers``.
        """
        queued = [
            left
            for answers in self.queued_answers
            if (left := {key: value for key, value in answers.items() if key not in waiting_ids})
        ]
        refused = {key: value for key, value in self.refused.items() if key not in waiting_ids}
        return self.model_copy(update={"queued_answers": queued, "refused": refused})

    def ended_with(self, run: Run[Any], saved: str, outcome: str) -> tuple[RunState, RunChanges]:
        """This state once a resume call returned ``run``, after the run was given the answers queued meanwhile.

        The queued answers are given to the run in the order they came, one waiting id at a time, with
        ``run.answer``. So a wrong answer is refused alone. An answer the run refuses goes to ``refused`` with its
        problems, and one it takes clears what ``refused`` had for its question. A second answer to a question the
        first one answered is refused by conductor as ``not_waiting``. An answer whose check raises something other
        than a refusal (a fault in a host's type) is logged and refused as ``raised_outside_a_node``.

        If the run takes any answer, it is running again, so the worker resumes it at once, with the answered run as
        the row's ``run``. A cancel that was asked but didn't end this resume call then stays asked, and ends the
        next one before any node call. So does a cancel that came after the call's last node call, when the run
        ends ``waiting``: it is running again, and ends ``cancelled``, as a cancel of a waiting run does. Otherwise
        the cancel is cleared: a run that is complete or ended early has nothing to cancel.

        Args:
            run: The run the resume call returned.
            saved: ``run``, as JSON.
            outcome: The row's ``outcome`` for ``run``, as JSON.

        Returns:
            The new state, and the row's ``run`` and ``outcome`` to write with it.
        """
        answered, taken, refused = run, False, dict(self.refused)
        for answers in self.queued_answers:
            for waiting_id, value in answers.items():
                try:
                    answered = answered.answer(waiting_id, value)
                except InvalidCallError as error:
                    refused[waiting_id] = error.problems
                except Exception as error:
                    log.exception("the answer queued for %s could not be checked", waiting_id)
                    refused[waiting_id] = [raised_outside_a_node(error=type(error).__name__)]
                else:
                    taken = True
                    refused.pop(waiting_id, None)
        # a cancel this resume call didn't hear is kept, so the next one ends cancelled before any node call
        unheard = self.cancel_asked and run.status != "cancelled"
        if taken or (unheard and run.status == "waiting"):
            status: RunStatus = "running"
            cause = None
            cancel_asked = unheard
        else:
            status, cause, cancel_asked = run.status, run.cause, False
        state = self.model_copy(
            update={
                "status": status,
                "cause": cause,
                "queued_answers": [],
                "refused": refused,
                "cancel_asked": cancel_asked,
            }
        )
        return state, {"run": answered.to_json() if taken else saved, "outcome": outcome}

    def failed(self, cause: Problem, outcome: str) -> tuple[RunState, RunChanges]:
        """This state once a resume call raised, or conductor refused it, so it returned no run.

        The row keeps its last checkpoint as ``run``. The answers queued meanwhile stay queued until the run is
        resumed and stops again, and a cancel asked is cleared, since the run has stopped.

        Args:
            cause: Why the resume call failed.
            outcome: The row's ``outcome`` for the run as the row has it, as JSON.

        Returns:
            The state, ``failed`` with ``cause``, and the row's ``outcome`` to write with it.
        """
        state = self.model_copy(update={"status": "failed", "cause": cause, "cancel_asked": False})
        return state, {"outcome": outcome}

    def columns(self) -> RunChanges:
        """The row's ``status`` and ``state`` for this state, written together in one ``update``.

        ``status`` is for the host's own queries: the worker and the routes read only ``state``.
        """
        return {"status": self.status, "state": self.model_dump_json()}


class RunView(RunOutcome):
    """A run as a reader sees it, as it was when it last stopped running, read without loading the run.

    ``Worker.read`` gives it, and every route under ``/runs`` answers with it, so an action answers with the run as
    it wrote it. It shows only what a reader needs: the answers a running run has queued, and a cancel asked of it,
    are the worker's own business and aren't shown.

    Attributes:
        id: The run's id.
        metadata: The run's metadata, as the JSON object the row keeps: what the host gave when the run started,
            written by alias. The worker reads it back by its ``metadata_type`` for the host's own code.
        status: Where the run is: ``running`` while a worker resumes it or is about to, ``waiting`` for answers,
            ``complete``, or how it ended early (``failed``, ``stopped``, ``cancelled``, ``timed_out``), which
            ``Worker.resume`` goes on from.
        cause: Why the run ended early; ``None`` while it runs, waits or once complete.
        waiting: What waits for a person, as of when the run last stopped running; each ``id`` is what
            ``Worker.answer`` takes. When the worker raised while resuming the run, what waits in the run's last
            checkpoint.
        outputs: The graph's outputs by name once the run is complete, each as its type writes it (a table as its
            columns); ``None`` until then.
        progress: How many calls of each node have an outcome, and how many there will be, as ``run.progress``
            says it: as of the last checkpoint while the run is running.
        refused: By waiting id, the problems of the last answer sent to that question while the run was running,
            when the run refused it once it stopped.
    """

    id: str
    metadata: dict[str, Any]
    status: RunStatus
    cause: Problem | None
    refused: dict[str, list[Problem]]
