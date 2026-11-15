"""``check_run_store`` passes the stores the suite has, and gives one message, naming the method, for each kind of
fault a store can have."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.testing import check_run_store
from pytest import MonkeyPatch

from tests.test_providers.in_memory import MemoryStore

PAGE = Path(__file__).parents[2] / "docs" / "serving-runs.md"


async def test_the_store_in_memory_does_all_the_worker_relies_on() -> None:
    assert await check_run_store(MemoryStore()) == []


async def test_the_pages_test_of_its_sqlite_store_passes(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    text = PAGE.read_text()
    section = text[text.index("## Writing a store") : text.index("## The smallest app")]
    page = ModuleType("serving_runs_page")
    monkeypatch.setitem(sys.modules, page.__name__, page)
    exec(
        compile(
            "\n".join(re.findall(r"```python\n(.*?)```", section, re.S)),
            "serving runs: a store",
            "exec",
        ),
        page.__dict__,
    )  # noqa: S102

    await page.test_run_store(tmp_path)


async def test_runs_of_the_hosts_own_in_the_store_are_left_out() -> None:
    store = MemoryStore()
    await store.create(
        StoredRun(
            "theirs", "{}", '{"user": "anna"}', "{}", "running", "{}", "{}", 0, "worker-z", 1.0
        )
    )

    assert await check_run_store(store) == []


class _KeepsJsonAsJsonb(MemoryStore):
    """Writes every JSON text back another way, keys sorted and spaced, as a Postgres ``jsonb`` column gives it."""

    @staticmethod
    def _rewritten(text: str) -> str:
        return json.dumps(json.loads(text), sort_keys=True, indent=1)

    async def create(self, run: StoredRun) -> None:
        columns = {
            name: self._rewritten(getattr(run, name))
            for name in ("graph", "metadata", "run", "state", "outcome")
        }
        await super().create(dataclasses.replace(run, **columns))

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        rewritten = {
            name: self._rewritten(value) if name in ("run", "state", "outcome") else value
            for name, value in changes.items()
        }
        return await super().update(run_id, rev, rewritten)  # type: ignore[arg-type]

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        return await super().add_event(run_id, owner, self._rewritten(body))


async def test_a_store_that_writes_its_json_another_way_passes() -> None:
    assert await check_run_store(_KeepsJsonAsJsonb()) == []


# -- one broken store for each kind of fault ------------------------------------------------------


class _ReadsAReplicaThatLags(MemoryStore):
    """Misses the first run written, as a replica that is a moment behind the database written to."""

    lagged = False

    async def get(self, run_id: str) -> StoredRun | None:
        if not self.lagged:
            self.lagged = True
            return None
        return await super().get(run_id)


class _SkipsNone(MemoryStore):
    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        written = {name: value for name, value in changes.items() if value is not None}
        return await super().update(run_id, rev, written)  # type: ignore[arg-type]


class _GivesAnEmptyRun(MemoryStore):
    async def get(self, run_id: str) -> StoredRun | None:
        found = await super().get(run_id)
        return found or StoredRun(run_id, "", "", "", "running", "", "", 0, None, None)


class _SaysFalseThoughItWrote(MemoryStore):
    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        await super().update(run_id, rev, changes)
        return False


class _ForgetsTheRev(MemoryStore):
    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        return await super().update(run_id, self.rows[run_id].rev, changes)


class _ReadsThenWrites(MemoryStore):
    """Reads the ``rev`` and writes in two steps, with an ``await`` between them, as two statements would be."""

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        row = self.rows.get(run_id)
        if row is None or row.rev != rev:
            return False
        await asyncio.sleep(0)
        self.rows[run_id] = dataclasses.replace(row, **changes, rev=rev + 1)
        return True


class _ZeroIsNoLease(MemoryStore):
    async def expired(self, now: float) -> list[StoredRun]:
        return [row for row in await super().expired(now) if row.lease_until]


class _AnHourOut(MemoryStore):
    """Compares the time an hour out, as when one side is local time and the other UTC."""

    async def expired(self, now: float) -> list[StoredRun]:
        return await super().expired(now + 3600)


class _NoLeaseIsZero(MemoryStore):
    async def expired(self, now: float) -> list[StoredRun]:
        return [row for row in self.rows.values() if (row.lease_until or 0) < now]


class _RefusesEveryEvent(MemoryStore):
    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        await super().add_event(run_id, owner, body)
        return False


class _AnyoneMayAdd(MemoryStore):
    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        self.written[run_id].append(body)
        return True


class _FromAfterOn(MemoryStore):
    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        return await super().events(run_id, after - 1)


class _CreateRaises(MemoryStore):
    async def create(self, run: StoredRun) -> None:
        raise ConnectionError("the database is down")


@pytest.mark.parametrize(
    ("store", "message"),
    [
        (_ReadsAReplicaThatLags, "create: get gave None for the run just written"),
        (_SkipsNone, "update: get gave a run whose owner, lease_until are not what was written"),
        (_GivesAnEmptyRun, "get: gave a run for an id no run has"),
        (_SaysFalseThoughItWrote, "update: gave False for a run at the rev it was given"),
        (_ForgetsTheRev, "update: wrote at rev 1 while the run was at rev 2"),
        (_ReadsThenWrites, "update: 8 of 8 updates at one rev, sent at once, were written"),
        (_ZeroIsNoLease, "expired: left out a run whose lease_until, 0.0, is before now"),
        (_AnHourOut, "expired: gave a run whose lease_until, "),
        (_NoLeaseIsZero, "expired: gave a run whose lease_until is None"),
        (_RefusesEveryEvent, "add_event: gave False for the run's owner"),
        (_AnyoneMayAdd, "add_event: added an event for a worker that doesn't hold the run"),
        (_FromAfterOn, "events: events(run_id, 0) gave the events numbered [0, 1, 2]"),
        (_CreateRaises, "create: raised ConnectionError: the database is down"),
    ],
    ids=lambda case: case.__name__.strip("_") if isinstance(case, type) else "",
)
async def test_each_fault_is_one_message_naming_its_method_and_saying_what_to_do(
    store: type[MemoryStore], message: str
) -> None:
    (found,) = await check_run_store(store())

    assert found.startswith(message), found
    assert "; " in found  # what is wrong, then what to do
