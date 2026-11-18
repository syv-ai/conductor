"""Tests of ``RunState``, which tracks a served run's status, its queued answers and any cancel asked for.

Most tests end a resume call on a plain ``Run``, with no store behind it, and check the new state and the changes it
writes. One test checks a held run whose store fails.
"""

from __future__ import annotations

import asyncio
import json
from asyncio import Event as AsyncioEvent
from typing import Any

import pytest
from conductor import Graph, Problem
from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.held_run import HeldRun, ResumeSettings
from conductor_providers.fastapi.state import RunOutcome, RunState
from conductor_providers.fastapi.store import GuardedStore
from pydantic import TypeAdapter, ValidationError
from pytest import LogCaptureFixture

from tests.test_providers.in_memory import MemoryStore, asking, left_running


def _waiting() -> tuple[Any, str]:
    """Returns a run of the ``asking`` graph that waits on its question, and the question's waiting id."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    return Graph.compile(graph, registry).run(inputs), waiting_id


def _ended_with(state: RunState, run: Any) -> tuple[RunState, dict[str, Any]]:
    """The new state, and the row's ``run`` and ``outcome`` written with it."""
    new, saved, outcome = state.ended_with(run, run.to_json(), RunOutcome.of(run))
    return new, {"run": saved, "outcome": outcome}


def test_a_run_that_stopped_with_nothing_queued_takes_its_status() -> None:
    run, _ = _waiting()

    state, changes = _ended_with(RunState.started(), run)

    assert (state.status, state.cause, state.cancel_asked, state.queued) == (
        "waiting",
        None,
        False,
        [],
    )
    assert changes == {"run": run.to_json(), "outcome": RunOutcome.of(run)}


def test_a_queued_answer_the_run_takes_makes_it_running_again_with_the_answered_run() -> None:
    run, waiting_id = _waiting()

    state, changes = _ended_with(RunState.started().with_queued({waiting_id: True}), run)

    assert (state.status, state.refused, state.queued) == ("running", {}, [])
    assert changes["run"] == run.answer(waiting_id, True).to_json()


def test_of_two_answers_to_one_question_the_first_is_taken_and_the_second_refused_as_not_waiting() -> (
    None
):
    run, waiting_id = _waiting()
    queued = RunState.started().with_queued({waiting_id: True}).with_queued({waiting_id: False})

    state, changes = _ended_with(queued, run)

    assert state.status == "running"
    assert [(problem.type, problem.ctx["status"]) for problem in state.refused[waiting_id]] == [
        ("not_waiting", "ready")
    ]
    assert changes["run"] == run.answer(waiting_id, True).to_json()


def test_a_wrong_answer_then_a_right_one_leaves_nothing_refused() -> None:
    run, waiting_id = _waiting()
    queued = (
        RunState.started().with_queued({waiting_id: "not a bool"}).with_queued({waiting_id: True})
    )

    state, _ = _ended_with(queued, run)

    assert (state.status, state.refused) == ("running", {})


def test_a_cancel_asked_while_the_run_ran_stays_asked_for_the_resume_a_taken_answer_starts() -> (
    None
):
    run, waiting_id = _waiting()

    state, _ = _ended_with(RunState.started().with_queued({waiting_id: True}).asked_to_stop(), run)

    assert (state.status, state.cancel_asked) == ("running", True)


def test_a_failed_resume_call_gives_the_run_its_queue_then_and_there_and_clears_a_cancel_asked() -> (
    None
):
    run, waiting_id = _waiting()
    cause = Problem(type="raised_outside_a_node", msg="It raised.", ctx={"error": "KeyError"})

    answered, given = (
        RunState.started().with_queued({waiting_id: True}).asked_to_stop().failed(cause, run)
    )
    declined, ended = RunState.started().with_queued(declines={waiting_id: "No"}).failed(cause, run)

    assert (answered.status, answered.cause, answered.cancel_asked, answered.queued) == (
        "failed",
        cause,
        False,
        [],
    )
    assert given is not None and given.to_json() == run.answer(waiting_id, True).to_json()
    assert (declined.status, declined.cause.type, declined.queued) == (
        "declined",
        "answer_declined",
        [],
    )
    assert ended is not None and ended.status == "declined"


def test_a_failed_resume_call_whose_run_cant_be_loaded_refuses_each_queued_reply_with_its_cause() -> (
    None
):
    """There is no run to give the queue to, so each reply is refused loudly rather than kept or dropped unseen."""
    cause = Problem(type="saved_run_unreadable", msg="It can't be read.", ctx={})

    state, given = (
        RunState.started()
        .with_queued({"a~1": True})
        .with_queued(declines={"b~2": None})
        .failed(cause, None)
    )

    assert (state.status, state.cause, state.queued, given) == ("failed", cause, [], None)
    assert state.refused == {"a~1": [cause], "b~2": [cause]}


def test_answers_taken_clear_what_was_refused_for_their_questions() -> None:
    refused = RunState.started().model_copy(
        update={"refused": {"a": [Problem(type="invalid_answer", msg="No.")], "b": []}}
    )

    assert list(refused.answered({"a"}).refused) == ["b"]


def test_a_queued_decline_the_run_takes_ends_it_declined_and_writes_the_declined_run_with_nothing_waiting() -> (
    None
):
    run, waiting_id = _waiting()

    state, changes = _ended_with(RunState.started().with_queued(declines={waiting_id: "No"}), run)

    declined = run.decline(waiting_id, reason="No")
    assert (state.status, state.cause, state.queued) == ("declined", declined.cause, [])
    assert changes == {"run": declined.to_json(), "outcome": RunOutcome.of(declined)}
    assert changes["outcome"].waiting == []


def test_a_decline_the_run_takes_clears_a_cancel_asked_since_the_run_has_ended() -> None:
    """The run is declined and never resumed, so a cancel left asked would mean nothing."""
    run, waiting_id = _waiting()

    state, _ = _ended_with(
        RunState.started().asked_to_stop().with_queued(declines={waiting_id: None}), run
    )

    assert (state.status, state.cancel_asked) == ("declined", False)


def test_of_an_answer_and_a_decline_to_one_question_the_first_queued_is_taken() -> None:
    """The second finds nothing waiting, so conductor refuses it as ``not_waiting``, as a second answer is."""
    run, waiting_id = _waiting()
    decline = {waiting_id: None}

    answered_first, _ = _ended_with(
        RunState.started().with_queued({waiting_id: True}).with_queued(declines=decline), run
    )
    declined_first, _ = _ended_with(
        RunState.started().with_queued(declines=decline).with_queued({waiting_id: True}), run
    )

    assert answered_first.status == "running"
    assert (declined_first.status, declined_first.cause.type) == ("declined", "answer_declined")
    assert [
        [(problem.type, problem.ctx["status"]) for problem in state.refused[waiting_id]]
        for state in (answered_first, declined_first)
    ] == [[("not_waiting", "ready")], [("not_waiting", "declined")]]


def test_the_queue_is_stored_as_one_reply_per_waiting_id_and_reads_back_as_it_was() -> None:
    """One call's answers are one reply each, and a decline is one reply too, each tagged with its ``kind`` and
    holding only what that kind has."""
    state = (
        RunState.started()
        .with_queued({"a": 1, "b": {"waiting_id": "c", "reason": "No"}})
        .with_queued(declines={"c": "No"})
    )

    stored = json.loads(state.model_dump_json())

    assert stored["queued"] == [
        {"kind": "answer", "waiting_id": "a", "answer": 1},
        {"kind": "answer", "waiting_id": "b", "answer": {"waiting_id": "c", "reason": "No"}},
        {"kind": "decline", "waiting_id": "c", "reason": "No"},
    ]
    assert RunState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.parametrize(
    "entry",
    [
        {"kind": "answer", "waiting_id": "a~1", "answer": 1, "reason": "x"},
        {"kind": "decline", "waiting_id": "a~1", "reason": "No", "answer": 1},
    ],
)
def test_a_queued_reply_with_a_key_its_kind_doesnt_have_is_refused_when_the_row_is_read(
    entry: dict[str, object],
) -> None:
    stored = json.dumps({**json.loads(RunState.started().model_dump_json()), "queued": [entry]})

    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs are not permitted"):
        RunState.model_validate_json(stored)


class _DownStore(MemoryStore):
    """A store that raises on every method, like a database that has gone away."""

    async def create(self, run: StoredRun) -> None:
        raise ConnectionError("the database is down")

    async def get(self, run_id: str) -> StoredRun | None:
        raise ConnectionError("the database is down")

    async def update(
        self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
    ) -> bool:
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
    metadata = TypeAdapter(dict[str, str])
    # the row as the worker reads it, from a store that is up
    readable = MemoryStore()
    readable.rows[row.id] = row
    held = await GuardedStore(readable, metadata).get(row.id)
    assert held is not None
    settings = ResumeSettings(
        worker_id="me",
        nodes_for=lambda metadata: registry,
        context=None,
        lease=30.0,
        checkpoint_every=5.0,
        resume_timeout=None,
        handing_back=AsyncioEvent(),
    )

    async with asyncio.timeout(10):
        await HeldRun(GuardedStore(store, metadata), held, settings).run()

    assert store.rows[row.id] == row and row.status == "running"
    assert any(
        "the store raised while run left was resumed" in record.message for record in caplog.records
    )


def test_a_cancel_heard_only_as_the_run_ends_waiting_keeps_it_running_to_end_cancelled() -> None:
    run, _ = _waiting()

    state, changes = _ended_with(RunState.started().asked_to_stop(), run)

    assert (state.status, state.cancel_asked) == ("running", True)
    assert changes["run"] == run.to_json()


def test_a_run_that_completes_with_a_cancel_asked_ends_cancelled_with_every_outcome_kept_and_no_outputs() -> (
    None
):
    run, waiting_id = _waiting()
    complete = run.answer(waiting_id, True).resume()

    state, changes = _ended_with(RunState.started().asked_to_stop(), complete)

    assert state.cause is not None
    assert (state.status, state.cause.type, state.cancel_asked) == (
        "cancelled",
        "run_cancelled",
        False,
    )
    assert changes["run"] == complete.to_json()
    assert (changes["outcome"].outputs, changes["outcome"].progress) == (None, complete.progress)
