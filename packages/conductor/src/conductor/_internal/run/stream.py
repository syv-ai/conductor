"""``run.stream()``, which runs a graph and gives the host each event as it happens.

A host iterates the stream with ``async for``, on an asyncio event loop. The nodes themselves are plain functions,
and some are slow. So the scheduler runs on a thread of its own, and calls the nodes there. Each event it makes is
handed to an asyncio queue, and the stream gives them out from that queue. That way a slow node never blocks the
host's event loop.
"""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop, Queue
from asyncio import Event as AsyncioEvent
from collections.abc import AsyncGenerator, Callable
from threading import Event as ThreadEvent
from threading import Thread
from typing import Any

from conductor._internal.problem import run_problems

from .events import Event, EventBase
from .state import EndedEarly, LastResume, deadline


async def stream(
    run_scheduler: Callable[[Callable[[EventBase], None], Callable[[], LastResume]], object],
    timeout: float | None,
    cancel: AsyncioEvent | ThreadEvent | None,
    hand_back: AsyncioEvent | None,
) -> AsyncGenerator[Event]:
    """Runs ``run_scheduler`` on a thread of its own, and gives out its events as they come.

    This is what ``run.stream()`` returns. For the README graph, the events are ``RunStart``, then ``NodeStart`` and
    ``NodeComplete`` for ``lines``, ``shout`` and ``join``, then ``RunEnd``.

    ``run_scheduler`` is the streamed run's ``Run._run_scheduler``, with the run's context and settings already
    given. It is called with two things: the function to hand each event to, and the check to make before each
    call. The check is built here, from ``cancel``, ``hand_back`` and ``timeout``. ``Run.stream`` says what each
    argument does.

    If the host stops iterating, the run is cancelled. If the scheduler itself raises, the stream raises the same
    exception once the events before it are given out.
    """
    loop, events = asyncio.get_running_loop(), Queue["Event | None"]()
    cancelled = ThreadEvent()
    check_deadline = deadline(timeout)

    def stop() -> LastResume:
        if cancelled.is_set() or (cancel is not None and cancel.is_set()):
            return EndedEarly(status="cancelled", cause=run_problems.run_cancelled())
        if hand_back is not None and hand_back.is_set():
            return "running"  # the scheduler stops, and another process goes on with the run
        return check_deadline() if check_deadline is not None else None

    failed: list[BaseException] = []

    def call_scheduler() -> None:
        try:
            run_scheduler(lambda event: _hand_over(loop, events, event), stop)
        except BaseException as error:  # an engine bug: the host gets it after the events
            failed.append(error)
        finally:
            _hand_over(loop, events, None)

    thread = Thread(target=call_scheduler, name="conductor-scheduler", daemon=True)
    thread.start()
    try:
        while (event := await events.get()) is not None:
            yield event
        if failed:
            raise failed[0]
    finally:
        cancelled.set()


def _hand_over(loop: AbstractEventLoop, events: Queue[Any], event: Any) -> None:
    """Puts an event on the listener's queue, from the scheduler's thread.

    If the listener's event loop has closed, the listener is gone, and the event is dropped.
    """
    try:
        loop.call_soon_threadsafe(events.put_nowait, event)
    except RuntimeError:
        pass
