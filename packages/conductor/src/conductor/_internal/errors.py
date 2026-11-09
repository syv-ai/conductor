"""What goes wrong, in one record, and how it is reported.

Everything wrong that a person has to hear about is a ``Problem``: what a
compile found (``CompiledGraph.problems``), what a refused call lists
(``InvalidCallError.errors()``), and why a node or a run failed
(``NodeFailed.cause``, ``RunState.failure``). So a host has one type to
translate, and one table to translate it with: ``PROBLEM_CODES`` is every
code conductor itself writes. Who meets what:

    who                   what                       when
    a developer, importing  NodeDeclarationError     a node or type is declared wrong
    an author, editing      Problem on the graph     compile; never raised
    a caller                InvalidCallError         run, answer, with_inputs, ...
    whoever reads a run     RunState.failure         a leg that ended in error

A graph that says something impossible is data, so an editor can show it
mid-edit; a caller asking for what cannot be done raises, from the call
that asked. A leg that ends in error is not an exception either: ``run``
returns its ``RunState``.

    ConductorError
    ├── NodeDeclarationError    a node or a type was declared wrong; a TypeError, met at import
    ├── CompileProblem          a node's hook cannot answer for what it was given
    ├── InvalidCallError        a verb on a compiled graph was given what it cannot act on
    ├── NodeError               one node failed; carries node_id. Internal: never retried
    │   ├── ExternalFailure         the outside world failed; the one family the engine retries
    │   ├── NodeValidationError     its inputs were wrong
    │   ├── NodeExecutionError      its body raised something that is not a NodeError
    │   └── NodeTimeoutError        the leg stopped waiting for it

**Two families of failure.** A node's failure is internal — a bug, bad
data, a refused schema — unless the outside world caused it: a network
error, a rate limit, an upstream 5xx. Only ``ExternalFailure`` is retried
under the version's ``Policy``; a node raises it itself, or names the
exception classes of the client it calls in ``Policy(retry_on=...)`` and
the engine wraps those. Every other exception a node lets out is wrapped as
``NodeExecutionError`` and runs once. The family is the class, so there is
no flag beside it to disagree with.

**What people are told.** A problem the engine writes carries the English
for its code, from ``PROBLEM_CATALOGUE``; a ``NodeError`` the node raised keeps
the message the node chose, since the node wrote it for people. The text
of a foreign exception goes to ``logging.getLogger("conductor")``, with its
traceback, and never in a problem, because a host streams problems to a browser;
the problem's ``ctx`` has its class name.

A pause is not an error and nothing raises for one: a node returns
``AskPerson`` and the leg ends with the status ``waiting``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from pydantic import ConfigDict, Field, computed_field

from conductor._internal.conductor_model import ConductorModel

#: Every code conductor itself writes, on one row each: its English, with
#: the ``ctx`` keys as slots, and whether it stops what it is about. A
#: compile problem's slots also take ``{node_id}`` and ``{field}``, the
#: first two places of its ``loc``. The non-fatal ones are states an author
#: can leave a graph in and still run it; every code a call or a run
#: writes is fatal, as it is what stopped them. A code that is not here is a
#: host's own, from a hook or a node, and is always fatal. A host that
#: translates by code covers ``PROBLEM_CODES``.
_PROBLEM_MESSAGES: Mapping[str, tuple[str, bool]] = {
    # -- what compile finds in a graph -----------------------------------------
    "cycle": ("The node is part of a cycle, so the graph cannot run.", True),
    "duplicate_field_name": ("The node has two fields named '{field}'.", True),
    "duplicate_node_id": ("Two nodes have the id '{node_id}'.", True),
    "embedded_graph_broken": ("The graph '{graph}' has {problems} problem(s) to fix before it can be used here.", True),
    "edge_into_unconnectable_input": ("Nothing can be connected to the field '{field}'.", True),
    "output_has_no_type": ("The output '{field}' has no type that can travel on an edge.", True),
    "invalid_node_id": ("'{node_id}' cannot be a node id: '/', '.' and '@' each mean something in an address.", True),
    "invalid_static": ("The value {input} in '{field}' cannot be read as {expected}. {reason}", True),
    "lists_on_a_per_row_node": (
        "Only one list can be set directly on a node that runs per row; '{a}' and '{b}' are both lists.", True,
    ),
    "no_rows_in_common": (
        "'{source}' shares no rows with what else is wired into '{field}', so its cells will be blank beside them; "
        "to join unrelated tables, match them on a key.", False,
    ),
    "table_column_missing": ("'{field}' needs the column '{column}', which the table it received doesn't have.", True),
    "series_of_tables": (
        "The output '{field}' is a series of tables; a table already holds many rows, so return one table.", True,
    ),
    "view_name_taken": (
        "The output '{field}' has the name the column '{column}' of the table '{table}' is offered under.", True,
    ),
    "misaligned": ("'{a}' and '{b}' get their rows from different sources, so the node cannot run per row.", True),
    "no_outputs": ("The node has no fields to pass on, so nothing can be connected from it.", False),
    "one_edge_per_parameter": ("Parameter '{field}' takes one edge; an extra edge is an extra parameter.", True),
    "parameter_name_invalid": (
        "'{field}' cannot be a parameter name; use letters, digits and underscores, and start with a letter.", True,
    ),
    "stale_binding": ("Field '{field}' is not an input of the node; it has {inputs}.", True),
    "type_mismatch": ("'{source}' is {source_said}, but '{node_id}.{field}' takes {target_said}.", True),
    "unbound_required": ("Nothing is connected to the field.", True),
    "union_needs_one_index": (
        "Field '{field}' has several edges; that only works when they are all rows of one table.", True,
    ),
    "unknown_locked_field": ("The lock on '{node_id}.{field}' points at a field the node does not have.", False),
    "unknown_node_type": ("Node type '{node_type}' does not exist; the types are {choices}.", True),
    "unknown_node_version": ("'{node_type}' has no version {version}.", True),
    "unknown_ref_node": ("Field '{field}' is connected to '{source_node}', which is not in the graph.", True),
    "unknown_ref_output": ("Field '{field}' is connected to '{source}', which is not an output of that node; it has {outputs}.", True),
    "unused_binding": (
        "Field '{field}' is not an input of the node as it is set up, so its value is not used; it has {inputs}.", False,
    ),
    # -- what a call to a compiled graph is refused for ------------------------
    "cannot_move_to_version": ("'{node_type}' cannot move from version {version} to {target}; it has {versions}.", True),
    "from_run_missing": ("'{node}' needs a {needs} for '{parameter}', and the call did not give one in from_run.", True),
    "from_run_not_a_type": ("from_run is keyed by type; {key} is not one.", True),
    "choice_unanswered": ("Answer exactly one of {alternatives}; {answered} answered.", True),
    "invalid_value": ("{reason}", True),
    "missing": ("'{name}' was not answered.", True),
    "node_not_in_graph": ("'{node}' is not a node of this graph; it has {nodes}.", True),
    "not_asked": ("'{name}' was not asked.", True),
    "not_offered": ("{reason}", True),
    "not_waiting": ("Nothing is waiting on an answer as {waiting_id!r}.", True),
    "run_not_complete": ("The run is {status}, so it has no outputs yet. {reason}", True),
    "state_unreadable": ("{reason}", True),
    # -- why a node or a run failed --------------------------------------------
    "engine_error": ("An error in the engine stopped the run.", True),
    "execution_failed": ("The node failed.", True),
    "external_failed": ("An outside service did not answer.", True),
    "failed": ("The node reported a failure.", True),
    "invalid_input": ("The node received a value it cannot use.", True),
    "invalid_output": ("The node returned a value that is not what it declared.", True),
    "row_covered_twice": ("The row comes from more than one edge; a merge may cover each row only once.", True),
    "row_ids_lost": (
        "The node returned a table whose rows no longer say which rows they are, and it is merged with the table it was built from.", True,
    ),
    "run_cancelled": ("The run was cancelled.", True),
    "run_timed_out": ("The run took too long and was stopped.", True),
    "timeout": ("The node did not answer in time.", True),
}


class CatalogueEntry(NamedTuple):
    """One code's English, with its ``ctx`` keys as slots, and whether a problem of that code stops what it is about."""

    message: str
    fatal: bool


#: Every code conductor writes, as its catalogue entry.
PROBLEM_CATALOGUE: Mapping[str, CatalogueEntry] = {code: CatalogueEntry(*entry) for code, entry in _PROBLEM_MESSAGES.items()}

#: The codes a host translating by code has to cover.
PROBLEM_CODES: frozenset[str] = frozenset(PROBLEM_CATALOGUE)


class Problem(ConductorModel):
    """One thing wrong, in pydantic's shape: ``type``, ``loc``, ``msg`` and ``ctx``.

    ``type`` is the stable code a host keys on. ``msg`` is for a person,
    never parsed; a host with its own wording translates by ``type`` and fills
    its sentence from ``ctx``, the values the message was made from under
    stable keys. ``loc`` says what it is about, and how to read it depends on
    who reported it, as it does between pydantic's body and query:

    ========================  ==================================================
    a compile problem         ``(node_id)`` or ``(node_id, field)``
    a node's failure          ``(node_id, *row)``: ``("b", 2, 0)`` is the first
                              row under the third row of what ``b`` runs on
    a run stopped as a whole  ``()``
    a refused call            ``("inputs", name)``, ``("answers", id, output)``,
                              ``("state", ...)``; a graph that can't run is
                              refused with its own compile problems
    ========================  ==================================================

    A compile problem's field need not exist: half of them are about a
    binding to a removed input or an edge to an unknown output.

    ``fatal`` is not stored: it is the row of ``type`` in ``PROBLEM_CATALOGUE``, and
    ``True`` for a code that isn't there. It stops a run for a compile
    problem; on the others it is simply true, since they are what stopped
    something. A dump carries it, for a reader with no catalogue, and reading the
    dump back ignores it.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str
    loc: tuple[str | int, ...] = ()
    msg: str
    ctx: Mapping[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fatal(self) -> bool:
        """Whether this stops what it is about, by its code's row."""
        entry = PROBLEM_CATALOGUE.get(self.type)
        return entry is None or entry.fatal


#: The one slot a caller may leave unfilled: the sentence a type's own
#: constructor raised, which not every constructor gives. Any other slot a
#: row names must be supplied, or ``make_problem`` raises where it is called — a
#: typo in a row or a forgotten keyword is a programming error, not an
#: empty message.
_OPTIONAL_MESSAGE_ARGUMENTS: frozenset[str] = frozenset({"reason"})


class _MessageArguments(dict[str, Any]):
    """Formatting slots: an optional slot the caller left out renders empty;
    any other missing slot raises."""

    def __missing__(self, key: str) -> str:
        if key in _OPTIONAL_MESSAGE_ARGUMENTS:
            return ""
        raise KeyError(f"the message for this problem needs {key!r}")


def make_problem(code: str, /, *loc: str | int | None, **ctx: Any) -> Problem:
    """The problem ``code`` at ``loc``, its message made from ``ctx``, which the record keeps.

    A ``None`` place is left out, so a site with an optional field passes it
    as it is. A code not in ``PROBLEM_CATALOGUE``, or a slot its row names that
    ``ctx`` does not fill, is a programming error and raises.
    """
    location = tuple(place for place in loc if place is not None)
    arguments = _MessageArguments(node_id=location[0] if location else None, field=location[1] if len(location) > 1 else None, **ctx)
    return Problem(type=code, loc=location, msg=PROBLEM_CATALOGUE[code].message.format_map(arguments).rstrip(), ctx=ctx)


class ConductorError(Exception):
    """Base for every engine error."""


class NodeDeclarationError(ConductorError, TypeError):
    """A node or a type was declared wrong; the developer writing it meets this at import.

    Raised by ``NodeDefinition``, a ``DType`` subclass, the signature a node
    declares and ``NodeRegistry``. Fixed in code, never by whoever draws
    a graph, so there is no ``Problem`` for it and the text stays English.
    ``code`` is stable, as pydantic's ``PydanticUserError`` has one; ``node_id``
    is the node's id when one is known.
    """

    def __init__(self, message: str, *, code: str, node_id: str | None = None) -> None:
        self.code = code
        self.node_id = node_id
        super().__init__(message)


class ComputeHookError(ConductorError):
    """A node's hook cannot answer for what it was given, and says why.

    Raised by ``compute_inputs`` and ``compute_outputs`` when the values or
    what arrives do not fit. Compile records it as a fatal ``Problem`` on
    the node, or on ``field`` when one is named, with ``code`` and
    ``message`` as the hook wrote them. Anything else a hook raises is a bug
    in the hook, not a fault in the graph, and leaves compile as it is.

    A code may be one conductor uses too, but not one the catalogue says is
    non-fatal: the node has no answer, so the graph must not run. Raising
    one is a ``ValueError``, from the hook.
    """

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        entry = PROBLEM_CATALOGUE.get(code)
        if entry is not None and not entry.fatal:
            raise ValueError(f"{code!r} is a code for a graph that can still run; a hook's problem always stops it")
        self.code = code
        self.message = message
        self.field = field
        super().__init__(message)

    def as_problem_on_node(self, node_id: str) -> Problem:
        """The ``Problem`` this is on ``node_id``."""
        return Problem(type=self.code, msg=self.message, loc=(node_id,) if self.field is None else (node_id, self.field))


class InvalidCallError(ConductorError, ValueError):
    """A verb on a compiled graph was given what it cannot act on; nothing happened.

    Raised by ``run``, ``stream``, ``answer``, ``outputs``, ``results``,
    ``with_inputs`` and ``upgrade_node`` from the call that caused it, before anything runs, and
    listing every fault at once. The state the caller holds is untouched. A
    ``ValueError`` too, as pydantic's error is.
    """

    def __init__(self, problems: Iterable[Problem]) -> None:
        self._problems = list(problems)
        super().__init__("\n".join([f"{len(self._problems)} problem(s):", *(f"  {'.'.join(str(place) for place in problem.loc)} — {problem.type}: {problem.msg}" for problem in self._problems)]))

    def errors(self) -> list[Problem]:
        """Every fault, in the order the call met them."""
        return list(self._problems)


class NodeError(ConductorError):
    """One node failed: what a node raises for a failure of its own with a message for people.

    ``NodeError("The document could not be read.")`` reaches the event as
    a ``Problem`` of code ``failed`` with that text; ``code`` and ``ctx`` give it
    a code of the node's own and what the host needs beyond the text.
    The engine adds the node and the row. It never retries a ``NodeError``;
    a failure the outside world caused is an ``ExternalFailure``.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        ctx: Mapping[str, Any] | None = None,
        node_id: str | None = None,
    ) -> None:
        self.code = code
        self.ctx = ctx or {}
        self.node_id = node_id
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
    (code ``execution_failed``); the exception itself is in the log, with ``__cause__`` to reach it.
    A node may raise it directly with a code of its own.
    """


class NodeTimeoutError(NodeError):
    """The leg stopped waiting for the node (code ``timeout``).

    Final: the attempt is not retried, and the thread it ran on is not
    interrupted. Neither family's: the outside world may or may not be the
    reason, and the engine cannot tell.
    """
