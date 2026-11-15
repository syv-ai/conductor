"""``RunningCalls``, a counter of how many calls of one node are running, across every run in the process.

A node with ``Policy(concurrency=2)`` may have at most two calls running at once, however many runs use it.
``@node`` makes one ``RunningCalls`` for such a node, and the definition keeps it. Say two runs go at once, each of a
graph that uses ``fetch`` in two places. They still make at most two calls of ``fetch`` at once.
"""

from __future__ import annotations

from threading import Condition, Lock
from typing import Protocol


class Waiter(Protocol):
    """A run waiting to start a call of a node. In practice, the run's scheduler.

    Attributes:
        parked: Whether the run's thread is idle, waiting for news. A parked run that is woken asks again at once.
            It is false while the run is busy, for example calling another node.
    """

    parked: bool

    def wake(self) -> None:
        """Tells the run that a call of the node has ended, so it can ask again.

        It is called from the thread that ended the call, so it should only pass the news on to the run.
        """
        ...


class RunningCalls:
    """Counts the running calls of one node across the process, and keeps the runs waiting to start one.

    Say ``fetch`` has ``Policy(concurrency=2)``, and two calls of it are running. Run B asks ``try_start(2, b)``. It
    gets ``False``, and B is kept as waiting. Run C is turned down the same way. When one of the two calls ends,
    ``finish()`` wakes B and C, and both ask again. The first to ask starts its call, and the other waits for the next
    one. A free place is never held for a particular run. So a run that is busy elsewhere when it is woken doesn't
    hold up the others.

    There is one catch. A run that makes its calls on its own thread could start its next call before the runs it
    woke got a chance to ask. It could keep them waiting until it has no calls left. So
    ``finish(give_parked_runs_a_turn=True)`` waits until each parked run it woke has asked, left, or turned to
    something else. If no run was parked, it doesn't wait at all.

    The run that starts a call calls ``finish`` once when the call is over. A run that ends calls ``stop_waiting``.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._turns_taken = Condition(self._lock)  # notified when a run is taken out of ``_turns``
        self._running = 0
        self._waiting: dict[Waiter, None] = {}  # a dict keeps the order they asked in
        self._turns: set[Waiter] = (
            set()
        )  # parked runs that a finish woke, and that haven't yet asked, left or turned away

    def try_start(self, concurrency: int, waiter: Waiter) -> bool:
        """Counts one more running call, if fewer than ``concurrency`` are running.

        Args:
            concurrency: The most calls of the node that may run at once, from its ``Policy``.
            waiter: The run asking. If the call can't start now, the run is kept and woken when a call ends.

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
        """Counts one call less, and wakes every run waiting to start one.

        Args:
            give_parked_runs_a_turn: Whether to wait until each parked run it woke has asked, left or turned to
                something else. A run that made the call on its own thread passes ``True``.
        """
        with self._lock:
            self._running -= 1
            waiting = list(self._waiting)
            parked = (
                {waiter for waiter in waiting if waiter.parked}
                if give_parked_runs_a_turn
                else set()
            )
            self._turns |= parked
        rest = iter(waiting)
        try:
            for waiter in rest:
                waiter.wake()
        finally:  # if waking one raises here, say a KeyboardInterrupt, the rest are still woken
            for waiter in rest:
                waiter.wake()
        if parked:
            with self._turns_taken:
                self._turns_taken.wait_for(lambda: not parked & self._turns)

    def turned_away(self, waiter: Waiter) -> None:
        """Notes that ``waiter`` was woken by other news, not a call ending, so it is busy and isn't waited for."""
        with self._lock:
            self._take_turn(waiter)

    def stop_waiting(self, waiter: Waiter) -> None:
        """Forgets ``waiter``, whose run has ended."""
        with self._lock:
            self._waiting.pop(waiter, None)
            self._take_turn(waiter)

    def _take_turn(self, waiter: Waiter) -> None:
        """Notes that ``waiter`` has had its turn, so a ``finish`` waiting for it can go on.

        The caller must hold the lock.
        """
        if waiter in self._turns:
            self._turns.discard(waiter)
            self._turns_taken.notify_all()
