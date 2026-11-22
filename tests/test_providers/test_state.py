"""The one test of ``RunState`` that reads names ``conductor_providers.fastapi`` doesn't make public.

Everything else ``RunState`` does is tested through the worker, in ``test_worker.py``. A decline that clears a cancel
asked is the one thing nothing outside a row's state shows: a declined run is never resumed, so the cancel left asked
would never act. It is tested here, on ``RunState`` itself, until that line of production code is looked at again.
"""

from __future__ import annotations

from typing import Any

from conductor import Graph
from conductor_providers.fastapi.state import RunState

from tests.test_providers.in_memory import asking


def _waiting() -> tuple[Any, str]:
    """Returns a run of the ``asking`` graph that waits on its question, and the question's waiting id."""
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    return Graph.compile(graph, registry).run(inputs), waiting_id


def test_a_decline_the_run_takes_clears_a_cancel_asked_since_the_run_has_ended() -> None:
    """The run is declined and never resumed, so a cancel left asked would mean nothing."""
    run, waiting_id = _waiting()

    asked = RunState.new().asked_to_stop().with_queued(declines={waiting_id: None})

    state, _, _ = asked.ended_with(run, run.to_json())

    assert (state.status, state.cancel_asked) == ("declined", False)
