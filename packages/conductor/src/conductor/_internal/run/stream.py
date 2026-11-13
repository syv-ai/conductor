"""``run.stream()``: the scheduler on a thread of its own, its events handed to an asyncio queue and yielded.

The nodes are called on that thread, so a slow node never blocks the event loop the host iterates the events on.
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
from .state import RUNNING, EndedEarly, RunState, deadline


async def stream(
    run_scheduler: Callable[[Any, Callable[[EventBase], None] | None, Callable[[], RunState | None] | None, float | None], Any],
    context: Any, timeout: float | None, cancel: AsyncioEvent | None, hand_back: AsyncioEvent | None,
    checkpoint_every: float | None,
) -> AsyncGenerator[Event]:
    """What ``run.stream()`` returns: ``run_scheduler`` called on a thread of its own, its events yielded here.

    For the README graph the events are ``RunStart``, then ``NodeStart`` and ``NodeComplete`` for ``lines``, ``shout``
    and ``join``, then ``RunEnd``. ``run_scheduler`` is ``Run._run_scheduler`` of the run streamed, called with the
    run's context, the function each event is handed to, the check made before each call (``cancel``,
    ``hand_back`` and ``timeout``) and ``checkpoint_every``. ``Run.stream`` says what each argument does.
    """
    loop, events = asyncio.get_running_loop(), Queue["Event | None"]()
    cancelled = ThreadEvent()
    check_deadline = deadline(timeout)

    def stop() -> RunState | None:
        if cancelled.is_set() or (cancel is not None and cancel.is_set()):
            return EndedEarly("cancelled", run_problems.run_cancelled())
        if hand_back is not None and hand_back.is_set():
            return RUNNING  # the scheduler stops here and another process goes on with the run
        return check_deadline() if check_deadline is not None else None

    failed: list[BaseException] = []

    def call_scheduler() -> None:
        try:
            run_scheduler(context, lambda event: _hand_over(loop, events, event), stop, checkpoint_every)
        except BaseException as error:  # the engine's own defect reaches the caller
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
    """Put an event on the listener's queue from the scheduler's thread; a listener whose loop has closed is gone, and hears nothing."""
    try:
        loop.call_soon_threadsafe(events.put_nowait, event)
    except RuntimeError:
        pass
