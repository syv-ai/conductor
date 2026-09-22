"""What goes wrong at run time, and how it is reported.

Two kinds of wrong, kept apart. A graph that says something impossible is
data — a ``Problem`` on the ``CompiledGraph``, the record of everything
compile learned about the graph — so an editor can show it mid-edit. A
run that goes wrong raises, and what it raises carries an ``ErrorCause``
— a stable code, a message for a person and whatever details the failure
knows — so a host can act on it without guessing which diagnostic
belonged to which failure.

    ConductorError
    ├── CompilationError        a caller asked to run a graph compile rejected
    ├── NodeError               one node failed; carries node_id and a cause. Internal: never retried
    │   ├── ExternalFailure         the outside world failed; the one family the engine retries
    │   ├── NodeValidationError     its inputs were wrong
    │   ├── NodeExecutionError      its body raised something that is not a NodeError
    │   └── NodeTimeoutError        the leg stopped waiting for it

**Two families.** A node's failure is internal — a bug, bad data, a
refused schema — unless the outside world caused it: a network error, a
rate limit, an upstream 5xx. Only ``ExternalFailure`` is retried under the
version's ``Policy``; a node raises it itself, or names the exception
classes of the client it calls in ``Policy(retry_on=...)`` and the engine
wraps those. Every other exception a node lets out is wrapped as
``NodeExecutionError`` and runs once. The family is the class, so there is
no flag beside it to disagree with.

**What people are told.** A cause the engine writes carries the generic
message for its code, from ``MESSAGES``; a ``NodeError`` the node raised
keeps the message the node chose, since the node wrote it for people. The
text of a foreign exception is on the wrapping error's ``original`` for a
log and never in a cause, because a host streams causes to a browser.

A pause is not an error and nothing raises for one: a node returns
``Asks`` (the value that says a person must answer before the graph can
continue) and the leg ends pending — a leg being one call of ``execute``;
a run takes several when a person must answer in between. Nor is a leg
that ends in error an exception: ``run`` returns the ending event, and
the caller reads its ``type``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field

from conductor.model import ConductorModel
from conductor.series import Row

if TYPE_CHECKING:
    from conductor.graph.problem import Problem


class ConductorError(Exception):
    """Base for every engine error."""


class Refuses(ConductorError):
    """A node's hook or a type cannot answer for what it was given, and says why.

    Raised by ``compute_inputs`` and ``compute_outputs`` when the values
    or what arrives do not fit, and by ``DType.refuses_whole`` when a value
    of the type cannot be handed over whole. The compiler catches it and
    reports ``code`` and ``message`` — both the host's own, in the host's
    language — as a fatal problem on the node or the field. Not an error a
    run raises: a graph that meets one does not run.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class StartRefused(ConductorError, ValueError):
    """A run cannot start from what its caller handed it: a ``cache`` or a ``record`` that does not fit the graph.

    Raised by the ledger while it seeds a leg — an answer for a node the
    graph does not have, an output left out or not declared, a value that
    is not its output's type, a row the run has not produced, or a record
    cell whose value does not read back as its field's type. The caller's
    fault rather than the engine's, so a provider answers it as a 422; a
    ``ValueError`` too, so a host that caught that before still does.
    """


class CompilationError(ConductorError):
    """A caller asked the engine to run a graph that compile rejected.

    Compile itself never raises this; what is wrong with a graph is data on
    the ``CompiledGraph``. ``execute`` raises it when a caller ignored
    ``is_runnable``, with the problems attached.
    """

    def __init__(self, message: str, *, problems: tuple[Problem, ...] = ()) -> None:
        self.problems = problems
        super().__init__(message)

    def __str__(self) -> str:
        """The message, then one line per fatal problem: where, its code, what it says."""
        lines = [
            f"  {p.node_id}{'' if p.field is None else '.' + p.field} — {p.code}: {p.message}"
            for p in self.problems if p.fatal
        ]
        head = super().__str__()
        return head if not lines else "\n".join([f"{head}:", *lines])


#: Every ``ErrorCause.code`` the engine itself emits, declared once. A host
#: that translates causes by code covers exactly these; a code a node
#: raised with its own cause is the node's. A test checks this set against
#: the literals at the sites that raise them.
CODES: frozenset[str] = frozenset({
    "engine_error",
    "execution_failed",
    "external_failed",
    "failed",
    "invalid_input",
    "invalid_output",
    "row_covered_twice",
    "timeout",
})

#: What a person reads for each cause the engine writes on its own, by
#: code. The engine builds those causes from this table, so the text is
#: the same at every site, and a host translating by code sees the English
#: it replaces. ``failed`` is not here: it is a ``NodeError`` the node raised
#: without a cause, and its message is the node's own. ``row_covered_twice``
#: is the ledger's and carries the rows in its text.
MESSAGES: dict[str, str] = {
    "engine_error": "An error in the engine stopped the run.",
    "execution_failed": "The node failed.",
    "external_failed": "An outside service did not answer.",
    "invalid_input": "The node received a value it cannot use.",
    "invalid_output": "The node returned a value that is not what it declared.",
    "timeout": "The node did not answer in time.",
}


class ErrorCause(ConductorModel):
    """Why a node failed, in a shape a caller can act on.

    Created by the engine where the failure is known and carried two ways:
    on the ``NodeError`` and on the ``node_error`` / ``graph_error`` events.
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
    details: Mapping[str, Any] = Field(default_factory=dict)
    row: Row | None = None


class NodeError(ConductorError):
    """One node failed. Carries which node (``node_id``) and why (``cause``).

    The base of the internal family, and what a node raises for a failure
    of its own with a message for people: ``NodeError("The document could
    not be read.")`` reaches the event as code ``failed`` with that text, or
    with the ``cause`` the node built. The engine never retries it; a
    failure the outside world caused is an ``ExternalFailure``.

    ``original`` is the foreign exception the engine wrapped, when there
    was one, for a log; its text is not in ``message`` or ``cause``.
    """

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


class ExternalFailure(NodeError):
    """The outside world failed: a network error, a rate limit, an upstream 5xx.

    The one family the engine retries, under the version's ``Policy``.
    A node raises it where it knows the failure is external; the engine
    raises it for a foreign exception whose class the policy's
    ``retry_on`` names, with code ``external_failed`` and the generic
    message. Not for a bug in the node or a value it cannot use — those
    are internal, and retrying them repeats the same failure.
    """


class NodeValidationError(NodeError):
    """The inputs themselves were wrong (code ``invalid_input``)."""


class NodeExecutionError(NodeError):
    """The node's body raised something that is not a ``NodeError``.

    The engine's wrapping of a foreign exception the policy does not name
    (code ``execution_failed``); the exception itself is on ``original``.
    A node may raise it directly with a cause of its own.
    """


class NodeTimeoutError(NodeError):
    """The leg stopped waiting for the node (code ``timeout``).

    Final: the attempt is not retried, and the thread it ran on is not
    interrupted. Neither family's: the outside world may or may not be the
    reason, and the engine cannot tell.
    """
