"""``check_run_store`` on the stores the suite has: the one in memory and the page's SQLite one pass, and a fault is
reported once, at the method that has it.

Each kind's own broken store is its example on the page on serving runs, run by ``test_problems_page``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import sys
from pathlib import Path
from types import ModuleType

from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.testing import check_run_store
from conductor_providers.fastapi.testing.problems import RUN_STORE_CHECK_PROBLEMS
from pytest import MonkeyPatch

from tests.test_providers.in_memory import MemoryStore

PAGE = Path(__file__).parents[2] / "docs" / "serving-runs.md"


async def test_the_store_in_memory_does_all_the_router_relies_on() -> None:
    assert await check_run_store(MemoryStore()) == []


async def test_the_pages_test_of_its_sqlite_store_passes(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    text = PAGE.read_text()
    section = text[text.index("## A store"):text.index("## The app")]
    page = ModuleType("serving_runs_page")
    monkeypatch.setitem(sys.modules, page.__name__, page)
    exec(compile("\n".join(re.findall(r"```python\n(.*?)```", section, re.S)), "serving runs: a store", "exec"), page.__dict__)  # noqa: S102

    await page.test_run_store(tmp_path)


async def test_runs_of_the_hosts_own_in_the_store_are_left_out() -> None:
    store = MemoryStore()
    await store.create(StoredRun("theirs", "{}", '"anna"', "{}", "running", "{}", "{}", 0, "worker-z", 1.0))

    assert await check_run_store(store) == []


class _CreateRaises(MemoryStore):
    async def create(self, run: StoredRun) -> None:
        raise ConnectionError("the database is down")


async def test_a_method_that_raises_is_reported_once_at_that_method() -> None:
    (problem,) = await check_run_store(_CreateRaises())

    assert (problem.type, problem.loc, problem.ctx) == ("store_raised", ("create",), {"error": "ConnectionError", "reason": "the database is down"})


class _ReadsThenWrites(MemoryStore):
    """Reads the ``rev`` and writes in two steps, with an ``await`` between them, as two statements would be."""

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        row = self.rows.get(run_id)
        if row is None or row.rev != rev:
            return False
        await asyncio.sleep(0)
        self.rows[run_id] = dataclasses.replace(row, **changes, rev=rev + 1)
        return True


async def test_updates_at_once_that_all_get_through_are_reported_with_how_many() -> None:
    (problem,) = await check_run_store(_ReadsThenWrites())

    assert (problem.type, problem.loc, problem.ctx) == ("two_updates_written", ("update",), {"written": 8, "tried": 8})


def test_every_kind_says_what_to_do() -> None:
    """A clause after what is wrong, as conductor's own messages have."""
    assert [kind.code for kind in RUN_STORE_CHECK_PROBLEMS if "; " not in kind.sentence] == []
