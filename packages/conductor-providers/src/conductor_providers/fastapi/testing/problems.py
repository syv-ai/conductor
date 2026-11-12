"""The kinds of problem ``check_run_store`` returns, one function each, in ``RUN_STORE_CHECK_PROBLEMS``.

A kind is called to make its ``Problem``, as the kinds of ``conductor.PROBLEMS`` are (``ProblemCatalogue.problem``
gives the rules)::

    update_not_written("update")
    # Problem(type="update_not_written", loc=("update",), msg="update at the run's own rev gave False; ...", ctx={})

Every problem here is about one method of the store, so its ``loc`` is that method's name. A developer reads them when
a test fails; a host never gets one from a run, so they are in neither ``conductor.PROBLEMS`` nor ``ROUTER_PROBLEMS``.
Each kind's docstring is its entry on the page on serving runs (``docs/serving-runs.md``); its example runs after the
page's section on writing a store, and a test checks that it makes this kind of problem and no other.
"""

from __future__ import annotations

from conductor import ProblemCatalogue

#: The kinds of problem ``check_run_store`` returns.
RUN_STORE_CHECK_PROBLEMS = ProblemCatalogue()


@RUN_STORE_CHECK_PROBLEMS.problem()
def run_not_found(*loc: str | int) -> str:
    """``get`` gave ``None`` for a run that ``create`` or ``update`` had just written.

    The router reads a run right after it writes it, often on another worker. So ``get`` reads what was written
    before it: from the database written to, not from a cache or a replica that lags behind.

    Args:
        loc: The method that wrote the run.

    Example:
        class ReadsAnotherDatabase(SqliteRunStore):
            async def get(self, run_id: str) -> StoredRun | None:
                return await SqliteRunStore("replica.db").get(run_id)


        asyncio.run(check_run_store(ReadsAnotherDatabase("runs.db")))
    """
    return "get gave None for a run that was just written; read from the database that create and update write to"


@RUN_STORE_CHECK_PROBLEMS.problem()
def run_differs(*loc: str | int, columns: list[str]) -> str:
    """``get`` gave a run whose columns are not what ``create`` or ``update`` wrote.

    A store keeps each column as it is written: a JSON column as the same JSON value (in any form of JSON), ``None``
    as ``None``, a time to the fraction of a second. ``update`` writes the columns it is given, leaves the others as they are and adds one to ``rev``. In the
    example, ``update`` leaves out every column it is given ``None`` for, so a run given back keeps its ``owner``.

    Args:
        loc: The method that wrote the run.
        columns: The columns that are not what was written.

    Example:
        class SkipsNone(SqliteRunStore):
            async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
                written = {name: value for name, value in changes.items() if value is not None}
                return await super().update(run_id, rev, written)  # type: ignore[arg-type]


        asyncio.run(check_run_store(SkipsNone("runs.db")))
    """
    return f"get gave a run whose {', '.join(columns)} are not what was written; keep every column as written, and add one to rev on each update"


@RUN_STORE_CHECK_PROBLEMS.problem()
def missing_run_found(*loc: str | int) -> str:
    """``get`` gave a run for an id that no run has.

    The router answers 404 when ``get`` gives ``None``, and reads a run it gets as the caller's. In the example,
    ``get`` gives an empty run in place of ``None``.

    Args:
        loc: ``get``.

    Example:
        class GivesAnEmptyRun(SqliteRunStore):
            async def get(self, run_id: str) -> StoredRun | None:
                found = await super().get(run_id)
                return found or StoredRun(run_id, "", "", "", "running", "", "", 0, None, None)


        asyncio.run(check_run_store(GivesAnEmptyRun("runs.db")))
    """
    return "get gave a run for an id no run has; give None when there is no run by the id"


@RUN_STORE_CHECK_PROBLEMS.problem()
def update_not_written(*loc: str | int) -> str:
    """``update`` gave ``False`` for a run that is at the ``rev`` it was given.

    A worker that is told its write lost reads the run again and tries again, or gives the run up, so a store that
    always says ``False`` stops every run. In the example, an ``UPDATE`` without ``RETURNING`` gives no rows, so the
    store says ``False`` though it wrote.

    Args:
        loc: ``update``.

    Example:
        class ReturnsNoRows(SqliteRunStore):
            async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
                assignments = "".join(f"{name} = ?, " for name in changes)
                changed = await self.sql(
                    f"UPDATE runs SET {assignments}rev = rev + 1 WHERE id = ? AND rev = ?", *changes.values(), run_id, rev
                )
                return bool(changed)


        asyncio.run(check_run_store(ReturnsNoRows("runs.db")))
    """
    return "update at the run's own rev gave False; write the changes and give True while the run is at the rev given"


@RUN_STORE_CHECK_PROBLEMS.problem()
def stale_update_written(*loc: str | int, rev: int, current: int) -> str:
    """``update`` wrote a run that is no longer at the ``rev`` it was given.

    This is the one rule that lets several workers act on one run: of two writes made from the same read, the second
    must lose. Write only where the run's ``rev`` is the one given, in the same statement.

    Args:
        loc: ``update``.
        rev: The ``rev`` the update was given.
        current: The ``rev`` the run was at.

    Example:
        class ForgetsTheRev(SqliteRunStore):
            async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
                assignments = "".join(f"{name} = ?, " for name in changes)
                changed = await self.sql(
                    f"UPDATE runs SET {assignments}rev = rev + 1 WHERE id = ? RETURNING id", *changes.values(), run_id
                )
                return bool(changed)


        asyncio.run(check_run_store(ForgetsTheRev("runs.db")))
    """
    return f"update at rev {rev} wrote a run that was at rev {current}; write only while the run is at the rev given, and give False otherwise"


@RUN_STORE_CHECK_PROBLEMS.problem()
def two_updates_written(*loc: str | int, written: int, tried: int) -> str:
    """Of several updates at one ``rev``, made at once, more than one was written.

    Two workers that claim one run at the same moment both send an ``update`` at the ``rev`` they read; only one may
    win, or both call its nodes. A store that reads the ``rev`` and then writes, in two statements, lets both through.
    Check the ``rev`` and write in one statement, or hold a lock on the row between the two.

    Args:
        loc: ``update``.
        written: How many were written.
        tried: How many were made at once.

    Example:
        class ReadsThenWrites(SqliteRunStore):
            async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
                found = await self.get(run_id)
                if found is None or found.rev != rev:
                    return False
                assignments = "".join(f"{name} = ?, " for name in changes)
                await self.sql(f"UPDATE runs SET {assignments}rev = ? WHERE id = ?", *changes.values(), rev + 1, run_id)
                return True


        asyncio.run(check_run_store(ReadsThenWrites("runs.db")))
    """
    return f"{written} of {tried} updates at one rev, made at once, were written; check the rev and write in one statement, so only one is"


@RUN_STORE_CHECK_PROBLEMS.problem()
def expired_missed(*loc: str | int, lease_until: float) -> str:
    """``expired`` left out a run whose ``lease_until`` is before the time it was given.

    Such a run's worker died, or handed it back with a ``lease_until`` of ``0.0``, and no worker goes on with it
    until ``expired`` gives it. In the example, ``0.0`` is taken to mean no lease.

    Args:
        loc: ``expired``.
        lease_until: The run's ``lease_until``.

    Example:
        class ZeroIsNoLease(SqliteRunStore):
            async def expired(self, now: float) -> list[StoredRun]:
                found = await self.sql("SELECT * FROM runs WHERE lease_until > 0 AND lease_until < ?", now)
                return [StoredRun(*values) for values in found]


        asyncio.run(check_run_store(ZeroIsNoLease("runs.db")))
    """
    return f"expired left out a run whose lease_until, {lease_until}, is before now; give every run whose lease_until is before the time given"


@RUN_STORE_CHECK_PROBLEMS.problem()
def expired_not_due(*loc: str | int, lease_until: float) -> str:
    """``expired`` gave a run whose ``lease_until`` is not before the time it was given.

    Its worker is still resuming it, and another worker would take it over and call its nodes again. In the example,
    the time is compared an hour out, as when one side is local time and the other UTC.

    Args:
        loc: ``expired``.
        lease_until: The run's ``lease_until``.

    Example:
        class AnHourOut(SqliteRunStore):
            async def expired(self, now: float) -> list[StoredRun]:
                found = await self.sql("SELECT * FROM runs WHERE lease_until < ?", now + 3600)
                return [StoredRun(*values) for values in found]


        asyncio.run(check_run_store(AnHourOut("runs.db")))
    """
    return f"expired gave a run whose lease_until, {lease_until}, is not before now; give only the runs whose lease_until is before the time given"


@RUN_STORE_CHECK_PROBLEMS.problem("expired_not_due", variant="not_running")
def expired_not_running(*loc: str | int) -> str:
    """``expired`` gave a run whose ``lease_until`` is ``None``: a run that is not running.

    It waits for a person, is complete or ended early, and a worker that took it would resume it. In the example, no
    lease is read as ``0``.

    Args:
        loc: ``expired``.

    Example:
        class NoLeaseIsZero(SqliteRunStore):
            async def expired(self, now: float) -> list[StoredRun]:
                found = await self.sql("SELECT * FROM runs WHERE COALESCE(lease_until, 0) < ?", now)
                return [StoredRun(*values) for values in found]


        asyncio.run(check_run_store(NoLeaseIsZero("runs.db")))
    """
    return "expired gave a run whose lease_until is None, which is not running; never give a run that has no lease_until"


@RUN_STORE_CHECK_PROBLEMS.problem()
def event_refused(*loc: str | int) -> str:
    """``add_event`` gave ``False`` when the worker that holds the run added an event.

    A worker told so takes it that another worker holds the run, and stops resuming it. In the example, an
    ``INSERT`` without ``RETURNING`` gives no rows.

    Args:
        loc: ``add_event``.

    Example:
        class InsertsWithoutReturning(SqliteRunStore):
            async def add_event(self, run_id: str, owner: str, body: str) -> bool:
                added = await self.sql(
                    "INSERT INTO events SELECT ?, COALESCE((SELECT MAX(seq) + 1 FROM events WHERE run_id = ?), 0), ?"
                    " WHERE EXISTS (SELECT 1 FROM runs WHERE id = ? AND owner = ?)",
                    run_id, run_id, body, run_id, owner,
                )
                return bool(added)


        asyncio.run(check_run_store(InsertsWithoutReturning("runs.db")))
    """
    return "add_event by the run's owner gave False; append the event and give True while the owner given is the run's owner"


@RUN_STORE_CHECK_PROBLEMS.problem()
def event_added_by_another(*loc: str | int) -> str:
    """``add_event`` appended an event for a worker that does not hold the run, or for a run that is not running.

    A worker whose lease ran out may still be calling nodes while another worker goes on with the run. Its events
    must be refused, so that followers see one run's events and the worker learns to stop. Check the run's ``owner``
    in the statement that appends.

    Args:
        loc: ``add_event``.

    Example:
        class AnyoneMayAdd(SqliteRunStore):
            async def add_event(self, run_id: str, owner: str, body: str) -> bool:
                added = await self.sql(
                    "INSERT INTO events SELECT ?, COALESCE((SELECT MAX(seq) + 1 FROM events WHERE run_id = ?), 0), ?"
                    " RETURNING seq",
                    run_id, run_id, body,
                )
                return bool(added)


        asyncio.run(check_run_store(AnyoneMayAdd("runs.db")))
    """
    return "add_event appended an event for a worker that does not hold the run; append only while the owner given is the run's owner"


@RUN_STORE_CHECK_PROBLEMS.problem()
def events_not_as_added(*loc: str | int, after: int, numbers: list[int]) -> str:
    """``events`` did not give the run's events above ``after``, as they were added and numbered.

    The first event of a run is numbered ``0`` and each next one is one more. ``events(run_id, after)`` gives every
    event of that run numbered above ``after``, in order, each with its body as added. A follower that reconnects
    says the last number it has, so an event given twice is shown twice and one left out is never shown. In the
    example, ``events`` gives the event numbered ``after`` too.

    Args:
        loc: ``events``.
        after: The ``after`` that ``events`` was given.
        numbers: The numbers of the events it gave.

    Example:
        class FromAfterOn(SqliteRunStore):
            async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
                found = await self.sql("SELECT seq, body FROM events WHERE run_id = ? AND seq >= ? ORDER BY seq", run_id, after)
                return [(seq, body) for seq, body in found]


        asyncio.run(check_run_store(FromAfterOn("runs.db")))
    """
    return f"events(run_id, {after}) gave the events numbered [{', '.join(map(str, numbers))}]; give each event numbered above after, in order, as it was added"


@RUN_STORE_CHECK_PROBLEMS.problem()
def store_raised(*loc: str | int, error: str, reason: str) -> str:
    """A method of the store raised an exception where the router expects an answer.

    The router takes an exception from the store for an outage: it stops the run and waits for its lease to run out.
    So a store answers what it is asked, also when the answer is no: ``None`` from ``get`` for an id no run has,
    ``False`` from ``update`` for a run at another ``rev`` and from ``add_event`` for a worker that does not hold the
    run. In the example, ``get`` unpacks one row, which raises when there is none.

    Args:
        loc: The method that raised.
        error: The exception's class name.
        reason: The exception's message.

    Example:
        class OneRowOrRaise(SqliteRunStore):
            async def get(self, run_id: str) -> StoredRun | None:
                (found,) = await self.sql("SELECT * FROM runs WHERE id = ?", run_id)
                return StoredRun(*found)


        asyncio.run(check_run_store(OneRowOrRaise("runs.db")))
    """
    return f"the store raised {error}: {reason}; answer None or False for what is not so, and raise only when the database fails"
