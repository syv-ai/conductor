"""What goes wrong at run time, and how it is reported.

Two kinds of wrong, kept apart. A graph that says something impossible is
data — a ``Problem`` on the ``CompiledGraph``, the record of everything
compile learned about the graph — so an editor can show it mid-edit. A
run that goes wrong raises, and what it raises carries an ``ErrorCause``
— a stable code, a message for a person and whatever details the failure
knows — so a host can act on it without guessing which diagnostic
belonged to which failure.

    ConductorError
    ├── CompilationError        a caller asked to run a flow compile rejected
    ├── NodeError               one node failed; carries node_id and a cause
    │   ├── NodeValidationError     its inputs were wrong (not retried)
    │   ├── NodeExecutionError      its body raised
    │   ├── NodeTimeoutError        it exceeded its policy's timeout
    │   └── NodeConnectionError     an external call failed (retried)
    ├── FlowExecutionError      execute_sync: the flow did not complete
    └── FlowPendingError        execute_sync: the run stopped to wait for a person; carries the questions and the record so far

A pause is not an error and nothing raises for one: a node returns
``Asks`` (the value that says a person must answer before the flow can
continue) and the leg ends pending — a leg being one call of ``execute``;
a run takes several when a person must answer in between.
``FlowPendingError`` exists only so the synchronous wrapper has a way to
hand that back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from conductor.series import Row

if TYPE_CHECKING:
    from conductor.graph.problem import Problem


class ConductorError(Exception):
    """Base for every engine error."""


class CompilationError(ConductorError):
    """A caller asked the engine to run a flow that compile rejected.

    Compile itself never raises this; what is wrong with a graph is data on
    the ``CompiledGraph``. ``execute`` raises it when a caller ignored
    ``is_runnable``, with the problems attached.
    """

    def __init__(self, message: str, *, problems: tuple[Problem, ...] = ()) -> None:
        self.problems = problems
        super().__init__(message)


#: Every ``ErrorCause.code`` the engine itself emits, declared once. A host
#: that translates causes by code covers exactly these; a code a node
#: raised with its own cause is the node's. A test checks this set against
#: the literals at the sites that raise them.
CODES: frozenset[str] = frozenset({
    "engine_error",
    "execution_failed",
    "failed",
    "invalid_input",
    "row_covered_twice",
    "timeout",
})


@dataclass(frozen=True)
class ErrorCause:
    """Why a node failed, in a shape a caller can act on.

    Created by the engine where the failure is known and carried two ways:
    on the ``NodeError`` and on the ``node_error`` / ``flow_error`` events.
    A host reads it to decide what to say and serialises it at its own
    edge. ``Problem`` is the compile-time counterpart: that one is about a
    graph that cannot run, this one about a run that went wrong::

        ErrorCause(code="rate_limited", message="The model rejected the call.",
                   details={"retry_after": 30}, row=(2, 0))

    ``code`` is stable and what a frontend keys on; ``message`` is for a
    person; ``details`` is whatever else the failure knows — a retry delay,
    a rejected field, an upstream request id; ``row`` is the row a node
    running once per row was on when it failed, as a path of positions:
    ``(2,)`` for the third row, ``(2, 0)`` for the first row nested under
    that one.
    """

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    row: Row | None = None


class NodeError(ConductorError):
    """One node failed. Carries which node (``node_id``) and why (``cause``).

    ``retryable`` is read by the engine's retry loop: a subclass for a
    failure retrying cannot fix sets it ``False``; an instance may override it.
    """

    retryable: bool = True

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        original: Exception | None = None,
        cause: ErrorCause | None = None,
    ) -> None:
        self.node_id = node_id
        self.original = original
        self.cause = cause
        super().__init__(message)


class NodeValidationError(NodeError):
    """The inputs themselves were wrong. Retrying cannot help."""

    retryable: bool = False


class NodeExecutionError(NodeError):
    """The node's body raised."""


class NodeTimeoutError(NodeError):
    """The node exceeded its policy's timeout."""


class NodeConnectionError(NodeError):
    """An external call inside a node failed — worth retrying."""


class FlowExecutionError(ConductorError):
    """``execute_sync``: the flow did not complete."""

    def __init__(self, message: str, *, node_id: str | None = None, cause: ErrorCause | None = None) -> None:
        self.node_id = node_id
        self.cause = cause
        super().__init__(message)


class FlowPendingError(ConductorError):
    """Raised by ``execute_sync`` when the leg ended waiting on a person.

    Carries the questions of every node — or every row of a node that
    runs per row — that is waiting (``pending``), and the record of
    everything the run has produced so far (``cells``); the caller answers
    and calls again with both.
    """

    def __init__(self, pending: list[dict[str, Any]], cells: dict[str, Any]) -> None:
        self.pending = pending
        self.cells = cells
        super().__init__("The flow is waiting for an answer.")
