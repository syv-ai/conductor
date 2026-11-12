"""``check_run_store``: one test that a host's ``RunStore`` does what the runs router relies on.

A host puts one test in its suite, on an empty database of the test's own::

    from conductor_providers.fastapi.testing import check_run_store

    async def test_run_store():
        assert await check_run_store(PostgresRunStore(test_database_url)) == []

It writes runs and events through the store's six methods and reads them back. It checks that a run is kept as
written; that ``get`` gives ``None`` for an id no run has; that ``update`` writes only at the ``rev`` it is given, and
that of several updates at one ``rev`` made at once exactly one is written; that ``expired`` gives the runs whose
lease ran out and no other; that ``add_event`` appends only for the run's owner, numbering each run's events from ``0``;
and that ``events`` gives them in order above the number asked. A store that raises where an answer is expected is
reported too, since the router takes an exception from the store for an outage.

Every problem is a ``Problem`` whose ``loc`` is the method at fault, as ``("update",)``. Its kinds are in
``RUN_STORE_CHECK_PROBLEMS`` (``conductor_providers.fastapi.testing.problems``), and the page on serving runs has an
example of each. The runs it writes stay in the store, so a worker that serves the same database would take some of
them over: give it a database of its own.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid
from collections.abc import Awaitable, Callable

from conductor import Problem

from conductor_providers.fastapi.store import RunStore
from conductor_providers.fastapi.stored_run import Outcome, RunChanges, StoredRun, StoredState
from conductor_providers.fastapi.testing import problems

__all__ = ["check_run_store"]

#: How many updates at one ``rev`` the check makes at once, as workers that claim one run at the same moment do.
_AT_ONCE = 8

#: How many times the updates at once are made, so a store that lets two through rarely gets past every round.
_ROUNDS = 5


class _StoreRaisedError(Exception):
    """A method of the store raised.

    Attributes:
        method: The method's name, as ``"update"``.
        error: What it raised.
    """

    def __init__(self, method: str, error: Exception) -> None:
        super().__init__(f"{method} raised {error!r}")
        self.method = method
        self.error = error


async def _answer[**P, T](method: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
    """What one method of the store answers; ``_StoreRaisedError`` with the method's name when it raises.

    ``await _answer(store.get, run_id)`` is ``await store.get(run_id)``.
    """
    try:
        return await method(*args, **kwargs)
    except Exception as error:
        raise _StoreRaisedError(method.__name__, error) from error


def _new_run(*, owner: str | None, lease_until: float | None) -> StoredRun:
    """A run as the router creates it, under an id that names the check, with texts as the router writes them."""
    written: RunChanges = {
        "run": '{"outcomes": {}}', **StoredState.started().columns(), "outcome": Outcome.empty().model_dump_json(),
        "owner": owner, "lease_until": lease_until,
    }
    return StoredRun(id=f"check-run-store-{uuid.uuid4().hex}", graph='{"nodes": [], "edges": []}', caller='"Zoë"', rev=0, **written)


def _columns_not_kept(found: StoredRun, written: StoredRun) -> list[str]:
    """The columns of ``found`` that differ from ``written``, in the order ``StoredRun`` declares them."""
    return [field.name for field in dataclasses.fields(StoredRun) if getattr(found, field.name) != getattr(written, field.name)]


async def _read_back(store: RunStore, written: StoredRun, method: str) -> Problem | None:
    """Whether ``get`` gives ``written`` back as it is: ``None``, else the problem with it.

    Args:
        store: The store.
        written: The run as ``method`` just wrote it.
        method: ``create`` or ``update``.
    """
    found = await _answer(store.get, written.id)
    if found is None:
        return problems.run_not_found(method)
    if columns := _columns_not_kept(found, written):
        return problems.run_differs(method, columns=columns)
    return None


async def _created_runs_are_kept(store: RunStore) -> list[Problem]:
    """A run created with a large text, a text that is not ASCII, an owner and a lease is read back as written, and
    so is one with neither; an id no run has is read as ``None``."""
    held = dataclasses.replace(
        _new_run(owner="worker-a", lease_until=time.time() + 3600.25), run=json.dumps({"outcomes": {"x": "é" * 500_000}}),
    )
    found = []
    for run in (held, _new_run(owner=None, lease_until=None)):
        await _answer(store.create, run)
        found.append(await _read_back(store, run, "create"))
    if await _answer(store.get, f"check-run-store-{uuid.uuid4().hex}") is not None:
        found.append(problems.missing_run_found("get"))
    return [problem for problem in found if problem is not None]


async def _updates_are_compared_and_set(store: RunStore) -> list[Problem]:
    """``update`` writes at the run's ``rev`` and moves it one on, leaves the run's other columns and other runs as they
    are, loses at an older ``rev``, and of several made at once at one ``rev``, exactly one is written."""
    run, beside = _new_run(owner="worker-a", lease_until=12.5), _new_run(owner="worker-a", lease_until=12.5)
    for each in (run, beside):
        await _answer(store.create, each)
    writes: tuple[RunChanges, ...] = (
        {"status": "waiting", "outcome": '{"waiting": [], "outputs": null}', "owner": None, "lease_until": None},
        {"owner": "worker-b", "lease_until": 7.75},
    )
    for changes in writes:
        if not await _answer(store.update, run.id, run.rev, changes):
            return [problems.update_not_written("update")]
        run = dataclasses.replace(run, **changes, rev=run.rev + 1)
        if problem := await _read_back(store, run, "update") or await _read_back(store, beside, "update"):
            return [problem]
    if await _answer(store.update, run.id, run.rev - 1, {"owner": "worker-c"}):
        return [problems.stale_update_written("update", rev=run.rev - 1, current=run.rev)]
    if problem := await _read_back(store, run, "update"):
        return [problem]
    for _ in range(_ROUNDS):
        claims: list[RunChanges] = [{"owner": f"worker-{number}"} for number in range(_AT_ONCE)]
        written = await asyncio.gather(*(_answer(store.update, run.id, run.rev, claim) for claim in claims))
        if sum(map(bool, written)) != 1:
            if not any(written):
                return [problems.update_not_written("update")]
            return [problems.two_updates_written("update", written=sum(map(bool, written)), tried=_AT_ONCE)]
        winner = next(claim for claim, won in zip(claims, written, strict=True) if won)
        run = dataclasses.replace(run, **winner, rev=run.rev + 1)
    return []


async def _expired_gives_the_runs_whose_lease_ran_out(store: RunStore) -> list[Problem]:
    """Of four runs, ``expired`` gives the one whose lease ran out and the one handed back, and neither the one whose
    lease holds nor the one that is not running. Runs that are not the check's own are left out of the comparison."""
    now = time.time()
    due = {lease: _new_run(owner="worker-a", lease_until=lease) for lease in (now - 60, 0.0)}
    held_until = now + 60
    held, not_running = _new_run(owner="worker-a", lease_until=held_until), _new_run(owner=None, lease_until=None)
    for run in (*due.values(), held, not_running):
        await _answer(store.create, run)
    given = {run.id for run in await _answer(store.expired, now)}
    found = [problems.expired_missed("expired", lease_until=lease) for lease, run in due.items() if run.id not in given]
    if held.id in given:
        found.append(problems.expired_not_due("expired", lease_until=held_until))
    if not_running.id in given:
        found.append(problems.expired_not_running("expired"))
    return found


async def _events_are_numbered_and_kept_for_the_owner(store: RunStore) -> list[Problem]:
    """The run's owner appends events, numbered from ``0`` for each run, and ``events`` gives those above a number, in
    order. A worker that does not hold the run, or any worker on a run that is not running, gets ``False`` and appends
    nothing, also once the run has another owner."""
    other, run, not_running = (_new_run(owner=owner, lease_until=None) for owner in ("worker-b", "worker-a", None))
    for each in (other, run, not_running):
        await _answer(store.create, each)
    bodies = ['{"type": "run_start"}', '{"type": "node_complete", "node_id": "ü"}', '{"type": "run_end"}']
    added = [await _answer(store.add_event, other.id, "worker-b", bodies[0])]
    added += [await _answer(store.add_event, run.id, "worker-a", body) for body in bodies]
    if not all(added):
        return [problems.event_refused("add_event")]
    for after in (-1, 0, 2):
        if (given := await _answer(store.events, run.id, after)) != list(enumerate(bodies))[after + 1:]:
            return [problems.events_not_as_added("events", after=after, numbers=[number for number, _ in given])]
    if not await _answer(store.update, run.id, run.rev, {"owner": "worker-b"}):
        return []  # the check of update reports it
    for not_held in (run, other, not_running):  # by worker-a, which holds none of them now
        if await _answer(store.add_event, not_held.id, "worker-a", '{"type": "run_end"}'):
            return [problems.event_added_by_another("add_event")]
    if await _answer(store.events, run.id, -1) != list(enumerate(bodies)):
        return [problems.event_added_by_another("add_event")]
    return []


async def check_run_store(store: RunStore) -> list[Problem]:
    """Every way ``store`` differs from what the runs router relies on.

    ``assert await check_run_store(SqliteRunStore("check.db")) == []`` passes for the store on the page on serving
    runs. Each check writes runs of its own, under ids that start ``check-run-store-``, and stops at its first
    problem, so one fault is reported once. A method that raises is reported as ``store_raised`` at that method.

    Args:
        store: The store, on a database of its own: the runs the check writes stay in it.

    Returns:
        The problems, each once; empty when the store does all the router relies on.
    """
    found: list[Problem] = []
    for check in (
        _created_runs_are_kept,
        _updates_are_compared_and_set,
        _expired_gives_the_runs_whose_lease_ran_out,
        _events_are_numbered_and_kept_for_the_owner,
    ):
        try:
            made = await check(store)
        except _StoreRaisedError as raised:
            made = [problems.store_raised(raised.method, error=type(raised.error).__name__, reason=str(raised.error))]
        found += [problem for problem in made if problem not in found]
    return found
