"""``RunningCalls``: how many calls of one node are running now, in every run of the process.

``@node("fetch", policy=Policy(concurrency=2))`` makes one, and the definition keeps it. Two runs at once, of graphs
that each use ``fetch`` in two places, still make at most two calls of ``fetch`` at once.
"""

from __future__ import annotations

from threading import Condition, Lock
from typing import Protocol


class Waiter(Protocol):
    """A run that waits to start a call of a node: its scheduler.

    Attributes:
        parked: Whether the run's thread is waiting for news right now, so a run that is woken asks at once. False
            while it is busy elsewhere, calling another node, say.
    """

    parked: bool

    def wake(self) -> None:
        """A call of the node ended: ask again. Called from the thread that ends the call, so it only hands the
        news to the run."""
        ...


class RunningCalls:
    """How many calls of one node are running now, in every run of the process, and which runs wait to start one.

    Example: ``fetch`` has ``Policy(concurrency=2)``, and two calls of it are running. Run B asks ``try_start(2, b)``,
    which returns False and keeps B; run C is refused the same way. When one of the two calls returns, ``finish()``
    wakes B and C, and both ask again: the first to ask starts its call, and the other waits for the next. A free call
    is never kept for a run, so a run whose thread is busy elsewhere when it is woken holds up no other.

    A run that makes its calls on its own thread could take the call back before the runs it woke had a chance to
    ask, and keep them waiting until it has no calls left. So ``finish(give_parked_runs_a_turn=True)`` waits until
    each run it woke that was parked (waiting for news, so it asks at once) has asked, left, or turned to something
    else. With no parked run it does not wait at all.

    The run that starts a call calls ``finish`` once when the call is over. A run that ends calls ``stop_waiting``.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._turns_taken = Condition(self._lock)  # notified when a run leaves ``_turns``
        self._running = 0
        self._waiting: dict[Waiter, None] = {}  # a dict, not a set: woken in the order they asked
        self._turns: set[Waiter] = set()  # parked runs woken by a finish that have not asked, left or turned away

    def try_start(self, concurrency: int, waiter: Waiter) -> bool:
        """Count one more running call, if fewer than ``concurrency`` are running.

        Args:
            concurrency: The most calls of the node that may run at once: its ``Policy``'s ``concurrency``.
            waiter: The run asking. It is kept, to be woken when a call of the node ends, if the call can't start now.

        Returns:
            Whether the call may start.
        """
        with self._lock:
            self._take_turn(waiter)
            if self._running < concurrency:
                self._running += 1
                self._waiting.pop(waiter, None)
                return True
            self._waiting[waiter] = None
            return False

    def finish(self, give_parked_runs_a_turn: bool = False) -> None:
        """Count one call less, and wake every run that waits to start one.

        Args:
            give_parked_runs_a_turn: Return only once each woken run that was parked has asked, left or turned to
                something else. The run that made the call on its own thread passes it.
        """
        with self._lock:
            self._running -= 1
            waiting = list(self._waiting)
            parked = {waiter for waiter in waiting if waiter.parked} if give_parked_runs_a_turn else set()
            self._turns |= parked
        rest = iter(waiting)
        try:
            for waiter in rest:
                waiter.wake()
        finally:  # an exception raised in this thread while it wakes them (a KeyboardInterrupt) still wakes the rest
            for waiter in rest:
                waiter.wake()
        if parked:
            with self._turns_taken:
                self._turns_taken.wait_for(lambda: not parked & self._turns)

    def turned_away(self, waiter: Waiter) -> None:
        """``waiter`` was woken by news other than a call ending: it is busy for now and is not waited for."""
        with self._lock:
            self._take_turn(waiter)

    def stop_waiting(self, waiter: Waiter) -> None:
        """Forget ``waiter``, if it is kept: its run has ended."""
        with self._lock:
            self._waiting.pop(waiter, None)
            self._take_turn(waiter)

    def _take_turn(self, waiter: Waiter) -> None:
        """``waiter`` has had its turn: a ``finish`` waiting for it goes on. Called with the lock held."""
        if waiter in self._turns:
            self._turns.discard(waiter)
            self._turns_taken.notify_all()
