"""``check_run_store``: one test that your ``RunStore`` does what the worker relies on.

Put one test like this in your suite, on an empty database of the test's own::

    from conductor_providers.fastapi.testing import check_run_store

    async def test_run_store():
        assert await check_run_store(PostgresRunStore(test_database_url)) == []

The check writes runs and events through the store's six methods and reads them back. Each fault it finds is one
plain message that starts with the method at fault, such as ``"update: wrote at rev 1 while the run was at rev 2;
write only while the run is at the rev given"``. The runs it writes stay in the database, so don't point a serving
worker at the same one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid

from conductor_providers.fastapi.state import RunOutcome, RunState
from conductor_providers.fastapi.store import (
    GuardedStore,
    RunChanges,
    RunStore,
    StoredRun,
    StoreFailedError,
)

__all__ = ["check_run_store"]

#: How many updates at one ``rev`` the check sends at once, as several workers that claim one run at the same moment do.
_AT_ONCE = 8

#: How many times it sends them, so a store that sometimes lets two through is very likely caught.
_ROUNDS = 5

#: The columns that hold JSON. They are compared as the JSON values they read back as, so a ``jsonb`` column passes.
_JSON_COLUMNS = frozenset({"graph", "metadata", "run", "state", "outcome"})


def _new_run(*, owner: str | None, lease_until: float | None, stopped: bool = False) -> StoredRun:
    """A run like one the worker creates, under an id that starts ``check-run-store-``; ``stopped``, one that waits."""
    state = RunState.started()
    written: RunChanges = {
        "run": '{"outcomes": {}}',
        **(state.model_copy(update={"status": "waiting"}) if stopped else state).columns(),
        "outcome": RunOutcome(waiting=[], outputs=None, progress=None).model_dump_json(),
        "owner": owner,
        "lease_until": lease_until,
    }
    return StoredRun(
        id=f"check-run-store-{uuid.uuid4().hex}",
        graph='{"nodes": [], "edges": []}',
        metadata='{"user": "Zoë"}',
        rev=0,
        **written,
    )


def _as_json_value(text: str) -> str:
    """The JSON value ``text`` reads as, written one fixed way (keys sorted, no spaces); text that isn't JSON as itself.

    So ``'{"b": 1, "a": 2.0}'`` and the same object read back from a ``jsonb`` column both give ``'{"a":2.0,"b":1}'``.
    """
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except ValueError:
        return text


def _columns_not_kept(found: StoredRun, written: StoredRun) -> list[str]:
    """The columns of ``found`` that differ from ``written``, in the order ``StoredRun`` declares them."""

    def kept_as(name: str, run: StoredRun) -> object:
        value = getattr(run, name)
        return _as_json_value(value) if name in _JSON_COLUMNS else value

    return [
        field.name
        for field in dataclasses.fields(StoredRun)
        if kept_as(field.name, found) != kept_as(field.name, written)
    ]


def _as_events(given: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Events with each body as the JSON value it reads as."""
    return [(number, _as_json_value(body)) for number, body in given]


async def _read_back(store: GuardedStore, written: StoredRun, method: str) -> str | None:
    """What is wrong with the run ``get`` gives back for ``written``; ``None`` when it is the run as written.

    Args:
        store: The store.
        written: The run as ``method`` just wrote it.
        method: ``create`` or ``update``.
    """
    found = await store.get(written.id)
    if found is None:
        return f"{method}: get gave None for the run just written; read from the database that create and update write to"
    if columns := _columns_not_kept(found, written):
        return (
            f"{method}: get gave a run whose {', '.join(columns)} are not what was written; keep every column as "
            "written, and add one to rev on each update"
        )
    return None


async def _created_runs_are_kept(store: GuardedStore) -> list[str]:
    """A run with a large text that isn't ASCII, an owner and a lease reads back as written, and so does one with
    neither. An id no run has reads as ``None``."""
    held = dataclasses.replace(
        _new_run(owner="worker-a", lease_until=time.time() + 3600.25),
        run=json.dumps({"outcomes": {"x": "é" * 500_000}}),
    )
    found = []
    for run in (held, _new_run(owner=None, lease_until=None)):
        await store.create(run)
        found.append(await _read_back(store, run, "create"))
    if await store.get(f"check-run-store-{uuid.uuid4().hex}") is not None:
        found.append(
            "get: gave a run for an id no run has; give None when there is no run by the id"
        )
    return [fault for fault in found if fault is not None]


async def _updates_are_compared_and_set(store: GuardedStore) -> list[str]:
    """``update`` writes every column it is given, at the run's ``rev``, and moves the ``rev`` on by one. It leaves the
    other columns and the other runs alone, refuses an older ``rev``, and of several updates sent at once at one ``rev``, lets exactly one through."""
    run, beside = (
        _new_run(owner="worker-a", lease_until=12.5),
        _new_run(owner="worker-a", lease_until=12.5),
    )
    for each in (run, beside):
        await store.create(each)
    writes: tuple[RunChanges, ...] = (
        {
            "run": json.dumps({"outcomes": {"y": "ü" * 200_000}}),
            "status": "waiting",
            "state": RunState.started().model_copy(update={"status": "waiting"}).model_dump_json(),
            "outcome": '{"waiting": [], "outputs": null}',
            "owner": None,
            "lease_until": None,
        },
        {"owner": "worker-b", "lease_until": 7.75},
    )
    not_written = (
        "update: gave False for a run at the rev it was given; write the changes and give True while the run is at "
        "the rev given"
    )
    for changes in writes:
        if not await store.update(run.id, run.rev, changes):
            return [not_written]
        run = dataclasses.replace(run, **changes, rev=run.rev + 1)
        if fault := await _read_back(store, run, "update") or await _read_back(
            store, beside, "update"
        ):
            return [fault]
    if await store.update(run.id, run.rev - 1, {"owner": "worker-c"}):
        return [
            f"update: wrote at rev {run.rev - 1} while the run was at rev {run.rev}; write only while the run is "
            "at the rev given, and give False otherwise"
        ]
    if fault := await _read_back(store, run, "update"):
        return [fault]
    for _ in range(_ROUNDS):
        claims: list[RunChanges] = [{"owner": f"worker-{number}"} for number in range(_AT_ONCE)]
        written = await asyncio.gather(*(store.update(run.id, run.rev, claim) for claim in claims))
        if not any(written):
            return [not_written]
        if sum(written) > 1:
            return [
                f"update: {sum(written)} of {_AT_ONCE} updates at one rev, sent at once, were written; check the "
                "rev and write in one statement, so that only one is"
            ]
        winner = next(claim for claim, won in zip(claims, written, strict=True) if won)
        run = dataclasses.replace(run, **winner, rev=run.rev + 1)
    return []


async def _expired_gives_the_runs_whose_lease_ran_out(store: GuardedStore) -> list[str]:
    """Of four runs, ``expired`` gives the one whose lease ran out and the one handed back (``0.0``). It gives neither
    the one whose lease holds nor the one that isn't running. Runs that aren't the check's own are ignored."""
    now = time.time()
    due = {lease: _new_run(owner="worker-a", lease_until=lease) for lease in (now - 60, 0.0)}
    held_until = now + 60
    held, not_running = (
        _new_run(owner="worker-a", lease_until=held_until),
        _new_run(owner=None, lease_until=None),
    )
    for run in (*due.values(), held, not_running):
        await store.create(run)
    given = {run.id for run in await store.expired(now)}
    found = [
        f"expired: left out a run whose lease_until, {lease}, is before now; give every run whose lease_until is "
        "before the time given"
        for lease, run in due.items()
        if run.id not in given
    ]
    if held.id in given:
        found.append(
            f"expired: gave a run whose lease_until, {held_until}, is not before now; give only the runs whose "
            "lease_until is before the time given"
        )
    if not_running.id in given:
        found.append(
            "expired: gave a run whose lease_until is None, which isn't running; never give a run that has no "
            "lease_until"
        )
    return found


async def _events_are_numbered_and_kept_for_the_owner(store: GuardedStore) -> list[str]:
    """The run's owner adds events, numbered from ``0`` for each run, and ``events`` gives those above a number, in
    order. A worker that doesn't hold the run gets ``False`` and adds nothing, also once the run has a new owner, and
    also on a run that isn't running, with an owner or with none."""
    other, run, not_running = (
        _new_run(owner=owner, lease_until=None) for owner in ("worker-b", "worker-a", None)
    )
    stopped = [
        _new_run(owner=owner, lease_until=None, stopped=True) for owner in ("worker-b", None)
    ]
    for each in (other, run, not_running, *stopped):
        await store.create(each)
    bodies = [
        '{"type": "run_start"}',
        '{"type": "node_complete", "node_id": "ü"}',
        '{"type": "run_end"}',
    ]
    added = [await store.add_event(other.id, "worker-b", bodies[0])]
    added += [await store.add_event(run.id, "worker-a", body) for body in bodies]
    if not all(added):
        return [
            "add_event: gave False for the run's owner; add the event and give True while the owner given is the "
            "run's owner"
        ]
    for after in (-1, 0, 2):
        given = await store.events(run.id, after)
        if _as_events(given) != _as_events(list(enumerate(bodies))[after + 1 :]):
            numbers = ", ".join(str(number) for number, _ in given)
            return [
                f"events: events(run_id, {after}) gave the events numbered [{numbers}]; give each event numbered "
                "above after, in order, as it was added"
            ]
    if not await store.update(run.id, run.rev, {"owner": "worker-b"}):
        return []  # the check of update reports it
    by_another = (
        "add_event: added an event for a worker that doesn't hold the run; add one only while the owner given is "
        "the run's owner"
    )
    for not_held in (
        run,
        other,
        not_running,
        *stopped,
    ):  # by worker-a, which holds none of them now
        if await store.add_event(not_held.id, "worker-a", '{"type": "run_end"}'):
            return [by_another]
    if _as_events(await store.events(run.id, -1)) != _as_events(list(enumerate(bodies))):
        return [by_another]
    return []


async def _an_update_with_an_event_writes_both_or_neither(store: GuardedStore) -> list[str]:
    """An ``update`` with an event appends it, numbered after the run's last, exactly when the row is written: at the
    run's ``rev``, whoever its owner is. At an older ``rev`` it writes neither."""
    run = _new_run(owner="worker-a", lease_until=None)
    await store.create(run)
    first, ended = '{"type": "run_start"}', '{"type": "run_end", "status": "failed"}'
    if not await store.add_event(run.id, "worker-a", first):
        return []  # the check of add_event reports it
    if not await store.update(run.id, run.rev, {"lease_until": 5.5}) or await store.update(
        run.id, run.rev, {"lease_until": 6.5}
    ):
        return []  # the check of update reports it
    run = dataclasses.replace(run, rev=run.rev + 1)
    if not await store.update(run.id, run.rev, {"owner": "worker-b", "status": "failed"}, ended):
        return [
            "update: gave False for an update with an event at the run's rev; write the changes, append the event "
            "and give True while the run is at the rev given"
        ]
    if _as_events(await store.events(run.id, -1)) != _as_events([(0, first), (1, ended)]):
        return [
            "update: an update with an event, at the run's rev, did not append the event numbered after the run's "
            "last; append it in the same statement or transaction as the write"
        ]
    await store.update(run.id, run.rev, {"status": "complete"}, '{"type": "run_end"}')
    if len(await store.events(run.id, -1)) != 2:
        return [
            "update: appended the event of an update that was not written; append it only when the write lands, "
            "in the same statement or transaction"
        ]
    return []


async def check_run_store(store: RunStore) -> list[str]:
    """Every way ``store`` differs from what the worker relies on, as plain messages.

    It checks that a run reads back as written, that ``update`` writes only at the ``rev`` it is given (and lets
    one of several sent at once through), that ``expired`` gives exactly the runs whose lease ran out, that
    ``add_event`` and ``events`` number a run's events and add them only for its owner, and that an ``update`` with an
    event writes both or neither. A method that raises is
    reported once, at that method, since the worker reads an exception from the store as an outage.

    Args:
        store: Your store, on a database of its own. The runs the check writes stay there.

    Returns:
        One message per fault, each starting with the method at fault; empty when the store does all the worker
        relies on.
    """
    guarded = GuardedStore(store)
    found: list[str] = []
    for check in (
        _created_runs_are_kept,
        _updates_are_compared_and_set,
        _expired_gives_the_runs_whose_lease_ran_out,
        _events_are_numbered_and_kept_for_the_owner,
        _an_update_with_an_event_writes_both_or_neither,
    ):
        try:
            faults = await check(guarded)
        except StoreFailedError as failed:
            faults = [
                f"{failed.method}: raised {type(failed.error).__name__}: {failed.error}; answer None or False for "
                "what isn't so, and raise only when the database fails"
            ]
        found += [fault for fault in faults if fault not in found]
    return found
