"""Signal / event nodes — pause until an external event arrives.

Signal nodes are the general-purpose cousin of HITL: they pause the
flow, checkpoint state, and hand off to the host. The host inspects the
checkpoint's ``signal_name`` / ``correlation`` / timeout fields and
wires those into its own message bus / webhook endpoint / timer daemon.
When the signal fires, the host calls ``resume_sync(compiled,
checkpoint, payload)`` to continue.

Two kinds are registered:

* ``signal-wait`` — waits for an explicit named event (``kind="event"``)
  or a timer (``kind="timer"``). One node type, a ``kind`` discriminator.
* ``signal-timer`` — convenience alias that only does time-based waits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from conductor.errors import SignalRequired
from conductor.types import NodeCategory
from conductor.widgets import CodeEditor, Dropdown, Number, Output, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register the signal / timer / event nodes."""

    @registry.node(
        "signal-wait", version=1, name="Wait for Signal",
        description=(
            "Pauses the flow until an external signal arrives. The host "
            "reads (signal_name, correlation, timeout_seconds) from the "
            "checkpoint and resumes the flow when the signal fires."
        ),
        category=NodeCategory.EVENT,
        is_signal=True,
    )
    def signal_wait(
        kind: Annotated[
            str,
            Dropdown(label="Kind", choices=["event", "timer"]),
        ] = "event",
        signal_name: Annotated[str, Text(label="Signal name")] = "",
        correlation: Annotated[
            str, CodeEditor(label="Correlation (CEL)", description="Optional CEL expression")
        ] = "",
        timeout_seconds: Annotated[
            float, Number(label="Timeout (seconds)")
        ] = 0,
    ) -> Annotated[Any, Output(label="Payload")]:
        # If already resumed with a payload, surface it here. The engine
        # injects the payload by pre-seeding results, so this function is
        # only called on the initial pause.
        raise SignalRequired(
            signal_name or kind,
            correlation=correlation or None,
            timeout_seconds=timeout_seconds or None,
        )

    @registry.node(
        "signal-timer", version=1, name="Timer",
        description="Wait for a duration, then continue. Host fires the timer.",
        category=NodeCategory.EVENT,
        is_signal=True,
    )
    def signal_timer(
        seconds: Annotated[float, Number(label="Seconds")] = 60,
    ) -> Annotated[Any, Output(label="Fired at")]:
        raise SignalRequired(
            "timer",
            correlation=None,
            timeout_seconds=float(seconds),
        )
