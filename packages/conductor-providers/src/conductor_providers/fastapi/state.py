"""A run's row as the worker reads it, the two JSON columns the worker keeps in it, and what a reader is shown.

A row's ``state`` is ``RunState``: where the run is, the replies queued while it runs, and a cancel asked of it. It is
small however large the run is. So queuing an answer or asking a cancel writes only it and the row's ``status``,
though the read before the write fetches the whole row. A row's ``outcome`` is ``RunOutcome``: what waits for a
person, the outputs, and how far the run is. It can be megabytes, so it is written only when the run stops and with
each checkpoint. A store keeps both as text and never reads them.

``RunRow`` is a row with its JSON read into these types, as the worker's store gives it. Every change of a state is a
method here that returns a new state, so a test can call it on a ``Run`` with no store at all. ``RunView`` is what a
reader sees: the run's status and outcome, without the worker's own bookkeeping.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Any, Generic, Literal

from conductor import (
    PROBLEMS,
    Graph,
    InvalidCallError,
    NodeRegistry,
    Problem,
    Run,
    RunProgress,
    RunStatus,
    Waiting,
)
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_core import from_json
from typing_extensions import TypeVar

from conductor_providers.fastapi.problems import raised_outside_a_node

#: The type of a run's metadata: what the host keeps on it. ``dict[str, str]`` unless the host names a model.
M = TypeVar("M", default=dict[str, str])

log = logging.getLogger("conductor_providers.fastapi")

#: The cause of a run a cancel ended: conductor's own ``run_cancelled``, the cause ``run.stream(cancel=)`` gives.
_RUN_CANCELLED = {kind.code: kind for kind in PROBLEMS}["run_cancelled"]


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


class QueuedAnswer(BaseModel):
    """An answer to one question of a running run, kept in the run's state until the run stops.

    It reads ``{"kind": "answer", "waiting_id": "approve~3fa2c0d1", "answer": true}``.

    Attributes:
        kind: ``"answer"``, which tells it from a ``QueuedDecline`` in the row's JSON. A key it doesn't have, such
            as a ``reason``, is refused, so a row written wrong fails when it is read.
        waiting_id: The question it answers, as the run's ``waiting`` lists it.
        answer: The answer, as the JSON data ``run.answer`` reads it from.
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
        reason: Why the person declined, as ``run.decline`` keeps it; ``None`` for no reason given.
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
    def started(cls) -> RunState:
        """The state of a run that has just started: running, and nothing queued, refused or asked."""
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
            answers: Answers by waiting id, as ``run.answer`` takes them.
            declines: Reasons by waiting id, as ``run.decline`` takes them.

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

    def declined(self, run: Run[Any]) -> RunState:
        """This state once a person's decline ended the run for good: ``declined``, with nothing queued or asked.

        Args:
            run: The run that took the decline.

        Returns:
            The state, ``declined`` with the run's ``answer_declined`` as its cause.
        """
        return self.model_copy(
            update={"status": run.status, "cause": run.cause, "queued": [], "cancel_asked": False}
        )

    def ended_with(
        self, run: Run[Any], saved: str, outcome: RunOutcome
    ) -> tuple[RunState, str, RunOutcome]:
        """This state once a resume call stops and the run is given what was queued meanwhile, in the order it came.

        ``python -m conductor.about answers-and-declines`` has the queue's rules. A reply whose check raises
        something other than a refusal (a fault in a host's type) is logged and refused as ``raised_outside_a_node``.

        A run that takes a decline has ended for good: it is ``declined``, and a reply queued after the decline is
        refused, as the run refuses it. Else a run that takes any answer is running again, for the worker to resume
        at once. A cancel asked that the run didn't end on then stays asked, and ends the next resume call before any
        node call. So does one asked of a run that ends ``waiting``.

        A run that completes with a cancel asked ends ``cancelled``. The cancel came before the run's end was
        written, so the run ends as the cancel's answer said it would, even though its last calls were made: another
        worker hears a cancel only at its next checkpoint or lease renewal. Every call's outcome is kept, so a resume
        makes no call and completes the run. Like any run that isn't complete, it shows no outputs until then. A run
        that ended early keeps how it ended, and the cancel is cleared.

        Args:
            run: The run the resume call returned.
            saved: ``run``, as JSON.
            outcome: The row's outcome for ``run``.

        Returns:
            The new state, and the row's ``run`` (as JSON) and ``outcome`` to write with it.
        """
        given, taken, refused = self._given_to(run)
        if given.status == "declined":
            ended = self.model_copy(update={"refused": refused}).declined(given)
            return ended, given.to_json(), RunOutcome.of(given)
        # a cancel asked that the run didn't end on: this resume call heard it too late, or not at all
        unheard = self.cancel_asked and run.status != "cancelled"
        if taken or (unheard and run.status == "waiting"):
            status, cause, cancel_asked = "running", None, unheard
        elif unheard and run.status == "complete":
            status, cause, cancel_asked = "cancelled", _RUN_CANCELLED(), False
            outcome = outcome.model_copy(update={"outputs": None})
        else:
            status, cause, cancel_asked = run.status, run.cause, False
        state = self.model_copy(
            update={
                "status": status,
                "cause": cause,
                "queued": [],
                "refused": refused,
                "cancel_asked": cancel_asked,
            }
        )
        return state, given.to_json() if taken else saved, outcome

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
                    given.decline(reply.waiting_id, reason=reply.reason)
                    if isinstance(reply, QueuedDecline)
                    else given.answer(reply.waiting_id, reply.answer)
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

    def failed(self, cause: Problem, run: Run[Any] | None) -> tuple[RunState, Run[Any] | None]:
        """This state once a resume call raised, or conductor refused it, so it returned no run.

        The run as the row has it, its last checkpoint, is given the replies queued meanwhile then and there, as
        ``ended_with`` gives them. So a row that stops never holds a queue, and a later answer, decline or resume
        acts on the run alone. A run that takes a decline is ``declined``; else it is ``failed`` with ``cause``. An
        answer to a question the run asked after its last checkpoint is refused as ``not_waiting`` in ``refused``,
        and that question waits again on the next resume. A cancel asked is cleared, since the run has stopped.

        When the row's run can't be loaded, there is no run to give the queue to, so each queued reply is refused
        with ``cause``, in ``refused`` by its waiting id.

        Args:
            cause: Why the resume call failed.
            run: The run as the row has it, or ``None`` when it couldn't be loaded.

        Returns:
            The state, and the run given the queue when it took any of it, for the row to keep; ``None`` to keep the
            row's run as it is.
        """
        if run is None:
            given, taken = None, False
            refused = {**self.refused, **{reply.waiting_id: [cause] for reply in self.queued}}
        else:
            given, taken, refused = self._given_to(run)
        if given is not None and given.status == "declined":
            return self.model_copy(update={"refused": refused}).declined(given), given
        state = self.model_copy(
            update={
                "status": "failed",
                "cause": cause,
                "queued": [],
                "refused": refused,
                "cancel_asked": False,
            }
        )
        return state, given if taken else None


@dataclass(frozen=True)
class RunRow(Generic[M]):
    """A run's row as the worker reads it: a ``StoredRun`` with its ``state`` read, and its metadata and outcome read
    when they are first asked for.

    The worker's store reads each row into one as it gets it from the host's store, and writes the changes of one
    back as text. So nothing else in the worker parses a row's JSON. The ``outcome`` can be megabytes, and only a
    reader's view and a resume call need it. The ``metadata`` is read on first use so that a row whose metadata the
    worker's type refuses can still be taken over, and its resume call then fails it. A row has no ``status`` of its
    own: the ``status`` column is written from ``state``.

    Attributes:
        id: The run's id.
        graph: The ``GraphRecord`` the run is of, as JSON.
        metadata_json: The run's metadata, as the JSON object the row keeps; ``metadata`` reads it.
        metadata_type: Reads ``metadata_json``: the worker's ``metadata_type``.
        run: ``run.to_json()``: the last checkpoint while the run is running, the whole run once it stops.
        state: Where the run is, and what was asked of it while it was running.
        outcome_json: What a reader gets without loading the run, as JSON; ``outcome`` reads it.
        rev: Moved one on by every update.
        owner: The worker resuming it.
        lease_until: Until when the owner's lease holds, as seconds since the epoch; ``None`` when the run is not
            running.
    """

    id: str
    graph: str
    metadata_json: str
    run: str
    state: RunState
    outcome_json: str
    rev: int
    owner: str | None
    lease_until: float | None
    metadata_type: TypeAdapter[M] = field(repr=False, compare=False, kw_only=True)

    @cached_property
    def metadata(self) -> M:
        """The run's metadata, read from ``metadata_json`` by its type the first time it is asked for.

        Raises:
            ValidationError: The worker's ``metadata_type`` refuses the row's metadata.
        """
        return self.metadata_type.validate_json(self.metadata_json)

    @cached_property
    def outcome(self) -> RunOutcome:
        """What a reader gets without loading the run, read from ``outcome_json`` the first time it is asked for.

        Raises:
            ValidationError: The row's outcome can't be read.
        """
        return RunOutcome.model_validate_json(self.outcome_json)

    def loaded(self, nodes: NodeRegistry[Any]) -> Run[Any]:
        """The run this row keeps, loaded on ``nodes``.

        Raises:
            InvalidCallError: The graph doesn't compile on ``nodes``, or the saved run can't be read.
        """
        return Run.from_json(self.run, Graph.compile(self.graph, nodes))


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
