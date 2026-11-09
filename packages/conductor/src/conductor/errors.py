"""What goes wrong at run time, and how it is reported.

Two kinds of wrong, kept apart. A graph that says something impossible is
data — a ``Problem`` on the ``CompiledGraph``, the record of everything
compile learned about the graph — so an editor can show it mid-edit. A
run that goes wrong raises, and what it raises carries an ``ErrorCause``
— a stable code, a message for a person and whatever details the failure
knows — so a host can act on it without guessing which diagnostic
belonged to which failure.

    ConductorError
    ├── CompilationError        a caller asked for something compile's problems rule out
    │   ├── NodeResolutionError     read off a node compile could not resolve
    │   └── NodeWiringError         read what the connections decide off a node whose wiring failed
    ├── InvalidCallError        a verb on a compiled graph was given what it cannot act on
    ├── NodeKindError           a caller read off a compiled node what a node of its kind does not have
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
``AskPerson`` and the leg ends with the status ``waiting``. Nor is a leg that
ends in error an exception: ``run`` returns its ``RunState``, whose ``status``
and ``failure`` say so. Asking for what the run cannot give is the caller's
fault, and raises ``InvalidCallError`` from the call that asked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

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
    reports ``code`` and ``message`` — both the host's own — as a fatal
    problem on the node or the field. Not an error a run raises: a graph
    that meets one does not run.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class InvalidCallDetail(TypedDict):
    """One fault of a refused call, in pydantic's shape: a stable ``type`` a host
    translates, the ``loc`` it is about, ``msg`` for a person, ``ctx`` for detail.
    A graph that cannot run gives one per fatal compile ``Problem``, its code as ``type``."""

    type: str
    loc: tuple[str | int, ...]
    msg: str
    ctx: dict[str, Any]


class InvalidCallError(ConductorError, ValueError):
    """A verb on a compiled graph was given what it cannot act on; nothing happened.

    Raised by ``run``, ``stream``, ``answer``, ``outputs``, ``results`` and
    ``with_inputs`` from the call that caused it, before anything runs, and
    listing every fault at once. The state the caller holds is untouched. A provider answers it as a
    422 with ``errors()`` as the detail; a host translates by each ``type``,
    every one in ``INVALID_CALL_CODES`` or a compile ``Problem``'s code. A
    ``ValueError`` too, as pydantic's error is.
    """

    def __init__(self, details: Sequence[InvalidCallDetail]) -> None:
        self._details = list(details)
        super().__init__("\n".join([f"{len(self._details)} refused:", *(f"  {'.'.join(str(at) for at in d['loc'])} — {d['type']}: {d['msg']}" for d in self._details)]))

    def errors(self) -> list[InvalidCallDetail]:
        """Every fault, in the order the call met them."""
        return list(self._details)


class CompilationError(ConductorError):
    """Something was asked of a graph, or of one of its nodes, that compile could not make work.

    Compile itself never raises; what is wrong with a graph is data on the
    ``CompiledGraph``, as ``Problem`` values. This is raised later, when a
    caller asks for something those problems rule out, and ``problems``
    carries the ones that say why. Its two subclasses are raised when a caller reads one broken
    node: ``NodeResolutionError`` and ``NodeWiringError``, each carrying
    the one problem behind that node's state.
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


class NodeResolutionError(CompilationError):
    """A caller read something off a node that compile could not make sense of at all.

    Raised by ``CompiledNode`` and ``CompiledGraph.field`` when the node's
    ``state`` is ``"resolution_failed"``: its type or version is unknown, it
    sits on a cycle, its id was refused, or its ``compute_inputs`` refused
    the values typed into it. Compile never learned what inputs and outputs
    the node has, so there is nothing to read. ``problems`` holds the one
    fatal ``Problem`` that says why, as it stands in ``compiled.problems``.

    An editor asks ``node.state`` first and never meets it; the engine never
    does either, since it runs only runnable graphs. Not a ``KeyError``:
    that is for an id the graph does not have.
    """

    def __init__(self, node_id: str, asked: str, problem: Problem) -> None:
        self.node_id = node_id
        super().__init__(f"{node_id!r} has no {asked}: compile could not resolve it", problems=(problem,))


class NodeWiringError(CompilationError):
    """A caller read what a node's connections decide, on a node whose connections could not be worked out.

    Raised by ``CompiledNode`` and ``CompiledField`` when the node's
    ``state`` is ``"wiring_failed"``: one of its own edges is broken, a node
    upstream of it is broken, or the embedded graph around it does not line
    up. What the node has on its own still answers — its inputs and
    outputs, its version, the values typed into it. What only its
    connections can decide does not: the type that arrives on each field,
    how each input receives it, and whether the node runs once per row.
    ``problems`` holds the one fatal ``Problem`` that explains it — the
    node's own, the one upstream of it, or the one on the graph around it —
    as it stands in ``compiled.problems``. Who meets it, and why it is not a
    ``KeyError``, as for ``NodeResolutionError``.
    """

    def __init__(self, node_id: str, asked: str, problem: Problem) -> None:
        self.node_id = node_id
        super().__init__(f"{node_id!r} has no {asked}: compile could not work out its connections", problems=(problem,))


class NodeKindError(ConductorError):
    """A caller read off a compiled node what a node of its kind does not have.

    Raised by ``CompiledNode.runner``, ``validate`` and ``fingerprint`` on a
    node whose ``kind`` is ``graph``: compile embedded its graph's nodes, so
    its inner nodes run in its place, each with a runner, a call and a
    fingerprint of its own. ``kind`` is the node's. Not a ``CompilationError``:
    nothing is wrong with the graph — the node is healthy and simply is not
    the kind that runs as one unit — so it carries no ``Problem``. The engine
    never meets it, since it runs only nodes of kind ``node``.
    """

    def __init__(self, node_id: str, asked: str, kind: str) -> None:
        self.node_id = node_id
        self.kind = kind
        super().__init__(f"{node_id!r} has no {asked}: a node of kind {kind!r} does not run as one unit")


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
    "run_cancelled",
    "run_timed_out",
    "timeout",
})

#: Every ``type`` a ``InvalidCallError`` carries of its own, beside a compile
#: ``Problem``'s code for a graph that cannot run.
INVALID_CALL_CODES: frozenset[str] = frozenset({
    "choice_unanswered",
    "invalid_value",
    "missing",
    "not_asked",
    "not_offered",
    "not_waiting",
    "run_not_complete",
    "state_unreadable",
})

#: What a person reads for each cause the engine writes on its own, by
#: code. The engine builds those causes from this table, so the text is
#: the same at every site, and a host translating by code sees the English
#: it replaces. ``failed`` is not here: it is a ``NodeError`` the node raised
#: without a cause, and its message is the node's own. ``row_covered_twice``
#: is the run values' and carries the rows in its text.
MESSAGES: dict[str, str] = {
    "engine_error": "An error in the engine stopped the run.",
    "execution_failed": "The node failed.",
    "external_failed": "An outside service did not answer.",
    "invalid_input": "The node received a value it cannot use.",
    "invalid_output": "The node returned a value that is not what it declared.",
    "run_cancelled": "The run was cancelled.",
    "run_timed_out": "The run took too long and was stopped.",
    "timeout": "The node did not answer in time.",
}


class ErrorCause(ConductorModel):
    """Why a node failed, in a shape a caller can act on.

    Created by the engine where the failure is known and carried two ways:
    on the ``NodeError`` and on the ``node_failed`` event and ``RunState.failure``.
    A host reads it to decide what to say and serialises it at its own
    edge. ``Problem`` is the compile-time counterpart: that one is about a
    graph that cannot run, this one about a run that went wrong::

        ErrorCause(code="rate_limited", message="The model rejected the call.",
                   details={"retry_after": 30}, row=(2, 0))

    ``code`` is stable and what a frontend keys on; ``message`` is for a
    person; ``details`` is whatever else the failure knows — a retry delay,
    a rejected field, an upstream request id; ``node_id`` the node, when a
    node caused it; ``row`` is the row a node
    running once per row was on when it failed, as a path of positions:
    ``(2,)`` for the third row, ``(2, 0)`` for the first row nested under
    that one, and the empty row ``()`` for a node that runs once.
    """

    code: str
    message: str
    details: Mapping[str, Any] = Field(default_factory=dict)
    row: Row = ()
    node_id: str | None = None


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
