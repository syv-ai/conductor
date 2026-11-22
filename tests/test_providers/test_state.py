"""Tests of ``RunState``, which tracks a served run's status, its queued answers and any cancel asked for.

Only what no worker or route test reaches is tested here: a run whose row can't be loaded when its resume call fails,
a decline that clears a cancel asked, and the queue's stored form. Everything else ``RunState`` does is tested through
the worker, in ``test_worker.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conductor import Graph, Problem
from conductor_providers.fastapi.state import RunOutcome, RunState
from pydantic import ValidationError

from tests.test_providers.in_memory import asking


def _waiting() -> tuple[Any, str]:
    """Returns a run of the ``asking`` graph that waits on its question, and the question's waiting id."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    return Graph.compile(graph, registry).run(inputs), waiting_id


def _ended_with(state: RunState, run: Any) -> tuple[RunState, dict[str, Any]]:
    """The new state, and the row's ``run`` and ``outcome`` written with it."""
    new, saved, outcome = state.ended_with(run, run.to_json())
    return new, {"run": saved, "outcome": outcome}


def test_a_failed_resume_call_whose_run_cant_be_loaded_refuses_each_queued_reply_with_its_cause() -> (
    None
):
    """There is no run to give the queue to, so each reply is refused loudly rather than kept or dropped unseen."""
    cause = Problem(type="saved_run_unreadable", msg="It can't be read.", ctx={})

    state, given, outcome = (
        RunState.new()
        .with_queued({"a~1": True})
        .with_queued(declines={"b~2": None})
        .failed(cause, None)
    )

    assert (state.status, state.cause, state.queued, given) == ("failed", cause, [], None)
    assert outcome == RunOutcome(waiting=[], outputs=None, progress=None)
    assert state.refused == {"a~1": [cause], "b~2": [cause]}


def test_a_decline_the_run_takes_clears_a_cancel_asked_since_the_run_has_ended() -> None:
    """The run is declined and never resumed, so a cancel left asked would mean nothing."""
    run, waiting_id = _waiting()

    state, _ = _ended_with(
        RunState.new().asked_to_stop().with_queued(declines={waiting_id: None}), run
    )

    assert (state.status, state.cancel_asked) == ("declined", False)


def test_the_queue_is_stored_as_one_reply_per_waiting_id_and_reads_back_as_it_was() -> None:
    """One call's answers are one reply each, and a decline is one reply too, each tagged with its ``kind`` and
    holding only what that kind has."""
    state = (
        RunState.new()
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
    stored = json.dumps({**json.loads(RunState.new().model_dump_json()), "queued": [entry]})

    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs are not permitted"):
        RunState.model_validate_json(stored)
