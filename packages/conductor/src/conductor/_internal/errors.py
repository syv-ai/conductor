"""What goes wrong, in one record, and how it is reported.

Everything wrong that a person has to hear about is a ``Problem``: what a
compile found (``CompiledGraph.problems``), what a rejected call lists
(``InvalidCallError.errors()``), and why a node or a run failed
(``NodeFailed.cause``, ``RunState.cause``). So a host has one type to
translate, and one table to translate it with: ``PROBLEM_CODES`` is every
code conductor itself writes. Who meets what:

    who                   what                       when
    a developer, importing  NodeDeclarationError     a node or type is declared wrong
    an author, editing      Problem on the graph     compile; never raised
    a caller                InvalidCallError         run, answer, with_inputs, ...
    whoever reads a run     RunState.cause         a leg that ended in error

A graph that says something impossible is data, so an editor can show it
mid-edit; a caller asking for what cannot be done raises, from the call
that asked. A leg that ends in error is not an exception either: ``run``
returns its ``RunState``.

    ConductorError
    ├── NodeDeclarationError    a node or a type was declared wrong; a TypeError, met at import
    ├── ComputeHookError        a node's hook cannot answer for what it was given
    ├── InvalidCallError        a verb on a compiled graph was given what it cannot act on
    └── NodeError               one node failed, with a message for people. Internal: never retried
        └── ExternalFailure         the outside world failed; the one family the engine retries

**Two families of failure.** A node's failure is internal — a bug, bad
data, a schema that didn't fit — unless the outside world caused it: a network
error, a rate limit, an upstream 5xx. Only ``ExternalFailure`` is retried
under the version's ``Policy``; a node raises it itself, or names the
exception classes of the client it calls in ``Policy(retry_on=...)`` and
the engine wraps those. Every other exception a node lets out is the code
``execution_failed`` and runs once. The family is the class, so there is
no flag beside it to disagree with.

**What people are told.** A problem the engine writes carries the English
for its code, from ``PROBLEM_CATALOGUE``; a ``NodeError`` the node raised keeps
the message the node chose, since the node wrote it for people. The text
of a foreign exception goes to ``logging.getLogger("conductor")``, with its
traceback, and never in a problem, because a host streams problems to a browser;
the problem's ``ctx`` has its class name. A value a run handled stays out of a
problem for the same reason: ``value_free_errors`` is what pydantic's account of
a refused value may say there.

A pause is not an error and nothing raises for one: a node returns
``AskPerson`` and the leg ends with the status ``waiting``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from pydantic import ConfigDict, Field, ValidationError, computed_field

from conductor._internal.conductor_model import ConductorModel

#: Every code conductor itself writes, on one row each: its English, with
#: the ``ctx`` keys as slots. A compile problem's slots also take
#: ``{node_id}`` and ``{field}``, the first two places of its ``loc``. A code
#: that is not here is a host's own, from a hook or a node. A host that
#: translates by code covers ``PROBLEM_CODES``.
PROBLEM_CATALOGUE: Mapping[str, str] = {
    # -- what compile finds in a graph -----------------------------------------
    "cycle": "The node is part of a cycle, so the graph cannot run.",
    "compute_hook_failed": "The node could not work out its fields from what it was given.",
    "duplicate_field_name": "The node has two fields named '{field}'.",
    "duplicate_node_id": "Two nodes have the id '{node_id}'.",
    "embedded_graph_broken": "The graph '{graph}' has {problems} problem(s) to fix before it can be used here.",
    "edge_into_unconnectable_input": "Nothing can be connected to the field '{field}'.",
    "output_has_no_type": "The output '{field}' has no type that can travel on an edge.",
    "invalid_node_id": "'{node_id}' cannot be a node id: '/', '.' and '@' each mean something in an address.",
    "invalid_static": "The value {input} in '{field}' cannot be read as {expected}. {reason}",
    "lists_on_a_per_row_node": "Only one list can be set directly on a node that runs per row; '{a}' and '{b}' are both lists.",
    "no_rows_in_common": (
        "'{source}' shares no rows with what else is wired into '{field}'. Unrelated tables have nothing to match on: "
        "use a node that joins them on a key column."
    ),
    "series_inputs_never_share_a_row": (
        "'{a}' and '{b}' read branches that never both have a value on a row, so no row would reach the node. "
        "Write Series[X | None] on them to keep every row, or take the branches through one input."
    ),
    "table_column_missing": "'{field}' needs the column '{column}', which the table it received doesn't have.",
    "series_of_tables": "The output '{field}' is a series of tables; a table already holds many rows, so return one table.",
    "view_name_taken": "The output '{field}' has the name the column '{column}' of the table '{table}' is offered under.",
    "misaligned": "'{a}' and '{b}' get their rows from different sources, so the node cannot run per row.",
    "no_outputs": "The node has no fields to pass on, so nothing can be connected from it.",
    "one_edge_per_parameter": "Parameter '{field}' takes one edge; an extra edge is an extra parameter.",
    "parameter_name_invalid": "'{field}' cannot be a parameter name; use letters, digits and underscores, and start with a letter.",
    "stale_binding": "Field '{field}' is not an input of the node; it has {inputs}.",
    "type_mismatch": "'{source}' is {source_said}, but '{node_id}.{field}' takes {target_said}.",
    "unbound_required": "Nothing is connected to the field.",
    "union_needs_one_index": "Field '{field}' has several edges; that only works when they are all rows of one table.",
    "unknown_locked_field": "The lock on '{node_id}.{field}' points at a field the node does not have.",
    "unknown_node_type": "Node type '{node_type}' does not exist; the types are {choices}.",
    "unknown_node_version": "'{node_type}' has no version {version}.",
    "unknown_ref_node": "Field '{field}' is connected to '{source_node}', which is not in the graph.",
    "unknown_ref_output": "Field '{field}' is connected to '{source}', which is not an output of that node; it has {outputs}.",
    "unused_binding": "Field '{field}' is not an input of the node as it is set up, so its value is not used; it has {inputs}.",
    # -- what a call to a compiled graph is rejected for -----------------------
    "cannot_move_to_version": "'{node_type}' cannot move from version {version} to {target}; it has {versions}.",
    "from_run_missing": "'{node}' needs a {needs} for '{parameter}', and the call did not give one in from_run.",
    "from_run_not_a_type": "from_run is keyed by type; {key} is not one.",
    "choice_unanswered": "Answer exactly one of {alternatives}; {answered} answered.",
    "invalid_value": "That is not a valid answer: {reason}",
    "missing": "'{name}' was not answered.",
    "node_not_in_graph": "'{node}' is not a node of this graph; it has {nodes}.",
    "not_asked": "'{name}' was not asked.",
    "not_offered": "This graph does not offer that input: {reason}",
    "not_waiting": "Nothing is waiting on an answer as {waiting_id!r}.",
    "run_not_complete": "The run is {status}, so it has no outputs yet. {reason}",
    "state_unreadable": "The saved state does not fit this graph: {reason}",
    # -- why a node or a run failed --------------------------------------------
    "engine_error": "An error in the engine stopped the run.",
    "execution_failed": "The node failed.",
    "external_failed": "An outside service did not answer.",
    "failed": "The node reported a failure.",
    "invalid_input": "The node received a value it cannot use.",
    "invalid_output": "The node returned a value that is not what it declared.",
    "row_covered_twice": "The row comes from more than one edge; a merge may cover each row only once.",
    "row_ids_lost": "The node returned a table whose rows no longer say which rows they are, and it is merged with the table it was built from.",
    "run_cancelled": "The run was cancelled.",
    "run_timed_out": "The run took too long and was stopped.",
    "timeout": "The node did not answer in time.",
}

#: The codes that don't stop what they are about: states an author can leave a
#: graph in and still run it. Every other code stops it, a host's own included.
_CODES_A_GRAPH_CAN_RUN_WITH = frozenset({"no_outputs", "unknown_locked_field", "unused_binding"})

#: The codes a host translating by code has to cover.
PROBLEM_CODES: frozenset[str] = frozenset(PROBLEM_CATALOGUE)


_SAID_OF_A_VALIDATOR = "The value did not pass its type's own check."


def value_free_errors(invalid: ValidationError) -> list[dict[str, Any]]:
    """pydantic's errors as a problem may hold them: nothing of the value.

    Each keeps its ``type``, and ``loc`` as far as the field it is on, since
    deeper a key of the value can stand there. ``msg`` is pydantic's own when
    that speaks of the type alone, such as a limit; a sentence built from what
    a validator raised or from the value's own tag is replaced, because it may
    quote the value.
    """
    return [
        {"type": error["type"], "loc": error["loc"][:1], "msg": _SAID_OF_A_VALIDATOR if {"error", "tag"} & error.get("ctx", {}).keys() else error["msg"]}
        for error in invalid.errors(include_url=False, include_input=False)
    ]


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
    a rejected call           ``("inputs", name)``, ``("answers", id, output)``,
                              ``("state", ...)``; a graph that can't run is
                              rejected with its own compile problems
    ========================  ==================================================

    A compile problem's field need not exist: half of them are about a
    binding to a removed input or an edge to an unknown output.

    ``fatal`` is not stored: it follows from ``type``, and is
    ``True`` for a code conductor doesn't know. It stops a run for a compile
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
        return self.type not in _CODES_A_GRAPH_CAN_RUN_WITH


def make_problem(code: str, /, *loc: str | int | None, **ctx: Any) -> Problem:
    """The problem ``code`` at ``loc``, its message made from ``ctx``, which the record keeps.

    A ``None`` place is left out, so a site with an optional field passes it
    as it is. A code not in ``PROBLEM_CATALOGUE``, or a slot its row names that
    ``ctx`` does not fill, is a programming error and raises; only ``reason`` may be
    left out, the sentence a type's own constructor raised, which not every one gives.
    """
    location = tuple(place for place in loc if place is not None)
    slots = {"reason": "", "node_id": location[0] if location else None, "field": location[1] if len(location) > 1 else None, **ctx}
    return Problem(type=code, loc=location, msg=PROBLEM_CATALOGUE[code].format_map(slots).rstrip(), ctx=ctx)


class ConductorError(Exception):
    """Base for every error conductor raises. One way to write one, whichever an author raises::

        raise ComputeHookError("The schema has no columns.", code="schema_empty", ctx={"schema": name}, field="schema")

    ``message`` is for a person. ``code`` is the stable word a host keys on
    and ``ctx`` the values its own sentence is filled from.
    """

    #: The ``Problem`` code of one that names no ``code`` of its own.
    _code_when_none_given: ClassVar[str]

    def __init__(self, message: str, *, code: str | None = None, ctx: Mapping[str, Any] | None = None) -> None:
        self.message = message
        self.code = code
        self.ctx = ctx or {}
        super().__init__(message)

    def _as_problem_at(self, *loc: str | int | None) -> Problem:
        """The ``Problem`` this is at ``loc``; a ``None`` place is left out, as in ``make_problem``."""
        return Problem(type=self.code or self._code_when_none_given, loc=tuple(place for place in loc if place is not None), msg=self.message, ctx=self.ctx)


class NodeDeclarationError(ConductorError, TypeError):
    """A node or a type was declared wrong; the developer writing it meets this at import.

    Raised by ``NodeDefinition``, a ``DType`` subclass, the signature a node
    declares and ``NodeRegistry``. Fixed in code, never by whoever draws
    a graph, so there is no ``Problem`` for it and the text stays English.
    ``code`` is stable, as pydantic's ``PydanticUserError`` has one.
    """


class ComputeHookError(ConductorError):
    """A node's hook cannot answer for what it was given, and says why.

    Raised by ``compute_added_inputs`` and ``compute_outputs`` when the values or
    what arrives do not fit. Compile records it as a fatal ``Problem`` on
    the node, or on ``field`` when one is named, with the message, code
    (``compute_hook_failed`` unless given) and ``ctx`` as the hook wrote them. Anything
    else a hook raises is a bug in the hook, not a problem in the graph, and
    leaves compile as it is.

    A code may be one conductor uses too, but not one the catalogue says is
    non-fatal: the node has no answer, so the graph must not run. Raising
    one is a ``ValueError``, from the hook.
    """

    _code_when_none_given = "compute_hook_failed"

    def __init__(self, message: str, *, code: str | None = None, ctx: Mapping[str, Any] | None = None, field: str | None = None) -> None:
        if code in _CODES_A_GRAPH_CAN_RUN_WITH:
            raise ValueError(f"{code!r} is a code for a graph that can still run; a hook's problem always stops it")
        super().__init__(message, code=code, ctx=ctx)
        self.field = field


class InvalidCallError(ConductorError, ValueError):
    """A verb on a compiled graph was given what it cannot act on; nothing happened.

    Raised by ``run``, ``stream``, ``answer``, ``outputs``, ``results``,
    ``with_inputs`` and ``upgrade_node`` from the call that caused it, before anything runs, and
    listing every problem at once. The state the caller holds is untouched. A
    ``ValueError`` too, as pydantic's error is.
    """

    def __init__(self, problems: Iterable[Problem]) -> None:
        self._problems = list(problems)
        super().__init__("\n".join([f"{len(self._problems)} problem(s):", *(f"  {'.'.join(str(place) for place in problem.loc)} — {problem.type}: {problem.msg}" for problem in self._problems)]))

    def errors(self) -> list[Problem]:
        """Every problem, in the order the call met them."""
        return list(self._problems)


class NodeError(ConductorError):
    """One node failed: what a node raises for a failure of its own with a message for people.

    ``NodeError("The document could not be read.")`` reaches the event as
    a ``Problem`` of code ``failed`` with that text; ``code`` and ``ctx`` give it
    a code of the node's own and what the host needs beyond the text.
    The engine adds the node and the row. It never retries a ``NodeError``;
    a failure the outside world caused is an ``ExternalFailure``.
    """

    _code_when_none_given = "failed"


class ExternalFailure(NodeError):
    """The outside world failed: a network error, a rate limit, an upstream 5xx.

    The one family the engine retries, under the version's ``Policy``.
    A node raises it where it knows the failure is external; the engine
    treats a foreign exception whose class the policy's ``retry_on`` names
    the same way, with code ``external_failed`` and the generic message. Not for a bug in the node or a value it cannot use — those
    are internal, and retrying them repeats the same failure.
    """

    _code_when_none_given = "external_failed"
