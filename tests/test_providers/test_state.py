"""Tests of ``RunState``, which tracks a served run's status, its queued answers and any cancel asked for.

Most tests end a resume call on a plain ``Run``, with no store behind it, and check the new state and the changes it
writes. One test checks a held run whose store fails.
"""

from __future__ import annotations

import asyncio
from asyncio import Event as AsyncioEvent
from typing import Any

from conductor import Graph, Problem
from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.bodies import Decline
from conductor_providers.fastapi.held_run import HeldRun, ResumeSettings
from conductor_providers.fastapi.state import QueuedAnswers, QueuedDecline, RunOutcome, RunState
from conductor_providers.fastapi.store import GuardedStore
from pydantic import TypeAdapter
from pytest import LogCaptureFixture

from tests.test_providers.in_memory import MemoryStore, asking, left_running


def _waiting() -> tuple[Any, str]:
    """Returns a run of the ``asking`` graph that waits on its question, and the question's waiting id."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    return Graph.compile(graph, registry).run(inputs), waiting_id


def _ended_with(state: RunState, run: Any) -> tuple[RunState, RunChanges]:
    return state.ended_with(run, run.to_json(), RunOutcome.of(run).model_dump_json())


def test_a_run_that_stopped_with_nothing_queued_takes_its_status() -> None:
    run, _ = _waiting()

    state, changes = _ended_with(RunState.started(), run)

    assert (state.status, state.cause, state.cancel_asked, state.queued_answers) == (
        "waiting",
        None,
        False,
        [],
    )
    assert changes == {"run": run.to_json(), "outcome": RunOutcome.of(run).model_dump_json()}


def test_a_queued_answer_the_run_takes_makes_it_running_again_with_the_answered_run() -> None:
    run, waiting_id = _waiting()

    state, changes = _ended_with(RunState.started().queued({waiting_id: True}), run)

    assert (state.status, state.refused, state.queued_answers) == ("running", {}, [])
    assert changes["run"] == run.answer(waiting_id, True).to_json()


def test_of_two_answers_to_one_question_the_first_is_taken_and_the_second_refused_as_not_waiting() -> (
    None
):
    run, waiting_id = _waiting()
    queued = RunState.started().queued({waiting_id: True}).queued({waiting_id: False})

    state, changes = _ended_with(queued, run)

    assert state.status == "running"
    assert [(problem.type, problem.ctx["status"]) for problem in state.refused[waiting_id]] == [
        ("not_waiting", "ready")
    ]
    assert changes["run"] == run.answer(waiting_id, True).to_json()


def test_a_wrong_answer_then_a_right_one_leaves_nothing_refused() -> None:
    run, waiting_id = _waiting()
    queued = RunState.started().queued({waiting_id: "not a bool"}).queued({waiting_id: True})

    state, _ = _ended_with(queued, run)

    assert (state.status, state.refused) == ("running", {})


def test_a_cancel_asked_while_the_run_ran_stays_asked_for_the_resume_a_taken_answer_starts() -> (
    None
):
    run, waiting_id = _waiting()

    state, _ = _ended_with(RunState.started().queued({waiting_id: True}).asked_to_stop(), run)

    assert (state.status, state.cancel_asked) == ("running", True)


def test_a_failed_resume_call_keeps_the_queued_answers_and_writes_only_the_outcome() -> None:
    cause = Problem(type="raised_outside_a_node", msg="It raised.", ctx={"error": "KeyError"})
    queued = RunState.started().queued({"approve~1": True}).asked_to_stop()

    state, changes = queued.failed(cause, '{"waiting": [], "outputs": null, "progress": null}')

    assert (state.status, state.cause, state.cancel_asked, state.queued_answers) == (
        "failed",
        cause,
        False,
        [QueuedAnswers(answers={"approve~1": True})],
    )
    assert changes == {"outcome": '{"waiting": [], "outputs": null, "progress": null}'}


def test_answers_taken_clear_what_was_queued_and_refused_for_their_questions() -> None:
    refused = RunState.started().model_copy(
        update={"refused": {"a": [Problem(type="invalid_answer", msg="No.")], "b": []}}
    )
    state = refused.queued({"a": 1, "c": 3}).queued({"a": 2}).answered({"a"})

    assert (state.queued_answers, list(state.refused)) == (
        [QueuedAnswers(answers={"c": 3})],
        ["b"],
    )


def test_a_queued_decline_the_run_takes_fails_it_and_writes_the_declined_run_with_nothing_waiting() -> (
    None
):
    run, waiting_id = _waiting()

    state, changes = _ended_with(
        RunState.started().queued(Decline(waiting_id=waiting_id, reason="No")), run
    )

    declined = run.decline(waiting_id, reason="No")
    assert (state.status, state.cause, state.queued_answers) == ("failed", declined.cause, [])
    assert changes == {
        "run": declined.to_json(),
        "outcome": RunOutcome.of(declined).model_dump_json(),
    }
    assert RunOutcome.model_validate_json(changes["outcome"]).waiting == []


def test_of_an_answer_and_a_decline_to_one_question_the_first_queued_is_taken() -> None:
    """The second finds nothing waiting, so conductor refuses it as ``not_waiting``, as a second answer is."""
    run, waiting_id = _waiting()
    decline = Decline(waiting_id=waiting_id)

    answered_first, _ = _ended_with(
        RunState.started().queued({waiting_id: True}).queued(decline), run
    )
    declined_first, _ = _ended_with(
        RunState.started().queued(decline).queued({waiting_id: True}), run
    )

    assert answered_first.status == "running"
    assert (declined_first.status, declined_first.cause.type) == ("failed", "answer_declined")
    for state in (answered_first, declined_first):
        assert [problem.type for problem in state.refused[waiting_id]] == ["not_waiting"]


def test_each_queued_entry_reads_back_from_the_rows_json_by_its_kind_alone() -> None:
    """An answer mapping whose keys are ``waiting_id`` and ``reason`` is still read back as answers."""
    looks_like_a_decline = {"waiting_id": "b", "reason": "No"}
    state = RunState.started().queued(looks_like_a_decline).queued(Decline(waiting_id="b"))

    read = RunState.model_validate_json(state.model_dump_json())

    assert read.queued_answers == [
        QueuedAnswers(answers=looks_like_a_decline),
        QueuedDecline(waiting_id="b"),
    ]
    assert read.answered({"b"}).queued_answers == [
        QueuedAnswers(answers={"waiting_id": "b", "reason": "No"})
    ]


class _DownStore(MemoryStore):
    """A store that raises on every method, like a database that has gone away."""

    async def create(self, run: StoredRun) -> None:
        raise ConnectionError("the database is down")

    async def get(self, run_id: str) -> StoredRun | None:
        raise ConnectionError("the database is down")

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        raise ConnectionError("the database is down")

    async def expired(self, now: float) -> list[StoredRun]:
        raise ConnectionError("the database is down")

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        raise ConnectionError("the database is down")

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        raise ConnectionError("the database is down")


async def test_a_held_run_whose_store_fails_gives_the_run_up_and_leaves_it_running(
    caplog: LogCaptureFixture,
) -> None:
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = _DownStore()
    row = left_running(
        "left", graph, Graph.compile(graph, registry).with_inputs(inputs), owner="me"
    )
    store.rows[row.id] = row
    settings = ResumeSettings(
        worker_id="me",
        nodes_for=lambda metadata: registry,
        context=None,
        metadata=TypeAdapter(dict[str, str]),
        lease=30.0,
        checkpoint_every=5.0,
        resume_timeout=None,
        handing_back=AsyncioEvent(),
    )

    async with asyncio.timeout(10):
        await HeldRun(GuardedStore(store), row, settings).run()

    assert store.rows[row.id] == row and row.status == "running"
    assert any(
        "the store raised while run left was resumed" in record.message for record in caplog.records
    )


def test_a_cancel_heard_only_as_the_run_ends_waiting_keeps_it_running_to_end_cancelled() -> None:
    run, _ = _waiting()

    state, changes = _ended_with(RunState.started().asked_to_stop(), run)

    assert (state.status, state.cancel_asked) == ("running", True)
    assert changes["run"] == run.to_json()


def test_a_cancel_asked_of_a_run_that_completes_is_cleared() -> None:
    run, waiting_id = _waiting()
    complete = run.answer(waiting_id, True).resume()

    state, _ = _ended_with(RunState.started().asked_to_stop(), complete)

    assert (state.status, state.cancel_asked) == ("complete", False)
