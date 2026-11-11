"""What goes wrong, in one record, and how it is reported.

Everything wrong that a person has to hear about is a ``Problem``: what
loading a graph found (``graph.problems``), what a refused call lists
(``InvalidCallError.errors()``), and why a node or a run stopped
(``NodeFailed.cause``, an ``EndedEarly`` ``run.state``'s ``cause``). So a host has one type to
translate, and one table to translate it with: ``PROBLEM_CODES`` is every
code conductor itself writes. Who meets what:

    who                     what                     when
    a developer, importing  NodeDeclarationError     a node is declared wrong
    an author, editing      Problem on the graph     loading; never raised
    a caller                InvalidCallError         run, answer, with_inputs, ...
    whoever reads a run     run.state                a leg that ended early

A graph that says something impossible is data, so an editor can show it
mid-edit; a caller asking for what cannot be done raises, from the call
that asked. A leg that ends in error is not an exception either: it returns
the ``Run``, whose ``cause`` says why.

    ConductorError
    ├── NodeDeclarationError    a node was declared wrong; a TypeError, met at import
    ├── InvalidCallError        a call on a graph or run was given what it cannot act on
    └── NodeError               one node failed, with a message for people. Internal: never retried
        └── ExternalError         the outside world failed; the one family the engine retries

**Two families of failure.** A node's failure is internal — a bug, bad
data, a schema that didn't fit — unless the outside world caused it: a network
error, a rate limit, an upstream 5xx. Only ``ExternalError`` is retried
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

A pause is not an error: a node raises ``Ask``, that call waits, and the leg
ends with the status ``waiting``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, final

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from conductor._internal.conductor_model import ConductorModel

#: Every code conductor itself writes, one entry each: its English, with
#: the ``ctx`` keys as slots. A compile problem's slots also take
#: ``{node_id}`` and ``{field}``, the first two places of its ``loc``. A code
#: that is not here is a node's own, from a ``NodeError``. A host that
#: translates by code covers ``PROBLEM_CODES``.
PROBLEM_CATALOGUE: Mapping[str, str] = {
    # -- what loading finds in a graph -----------------------------------------
    "compute_hook_failed": "The node could not work out its fields from what it was given: {reason}. Check what is typed into the node and what is wired to it.",
    "cycle": "The node is part of a cycle, so the graph cannot run; remove one of the edges that close it.",
    "duplicate_node_id": "Two nodes have the id '{node_id}'; rename one.",
    "edge_names_missing_node": "The edge from '{source}' to '{target}' names '{node_id}', which is not a node of the graph; add that node or remove the edge.",
    "embeds_itself": "The node '{node}' is a graph that contains itself; embed a graph that does not contain this one.",
    "default_on_later_node": "'{field}' is not an open input of the graph, so a caller can't change its default; make it a value.",
    "hook_needs_value": "'{field}' must be typed in: the node works out its fields from it when the graph loads.",
    "invalid_node_id": "'{node_id}' cannot be a node id: '.', '/', '@' and '~' each mean something in an address or a waiting id; rename it without them.",
    "invalid_value": "The value typed into '{field}' is not valid: {reason}. Type a value of the field's type.",
    "parameter_name_invalid": "'{field}' cannot be a field name: '.', '/' and '@' each mean something in an address, and a name can't be empty; rename the field.",
    "repeated_edge": "The edge from '{source}' into '{field}' is drawn twice; remove one.",
    "alternatives_in_different_series": (
        "'{field}' takes alternatives that run per item of different Series, {series}, so a call for an item of one has no place "
        "for the other's value. Bring them to items of one Series first."
    ),
    "several_edges_into_one_input": (
        "More than one of the edges into '{field}', from {sources}, can hold a value on the same call. "
        "Feed it from opposite branches of one decision, or put a node between that has an input for each."
    ),
    "series_inputs_never_share_an_item": (
        "{inputs} read branches that never both have a value for an item, so no item would reach the node. "
        "Write Series[X | None] on them to keep every item, or take the branches through one input."
    ),
    "series_not_nested": (
        "'{node_id}' reads items of {series}, and neither is made inside the other, so nothing says which item goes with which. "
        "Make one inside the other (wire '{wire_from}' into '{wire_into}', the node that makes the other), or give one of them whole to a Series input."
    ),
    "type_mismatch": "'{source}' gives {got}, but '{node_id}.{field}' takes {expected}; connect an output that gives {expected}, or convert it in a node between.",
    "unbound_on_later_node": "Nothing feeds '{field}', and only a first node takes inputs from the caller; draw an edge into it or type a value in.",
    "unknown_input": "'{node_id}' has no input '{field}'; it has {inputs}.",
    "unknown_node": "There is no node '{node}'; the nodes are {choices}.",
    "unknown_output": "'{source}' is not something that node gives; it gives {outputs}.",
    "unknown_version": "'{node}' has no version {version}; the newest is {newest}.",
    "value_and_default": "Field '{field}' has both a value and a default; keep one.",
    "value_and_edge": "Field '{field}' has an edge and a typed-in value; keep one.",
    # -- what a call is refused for ----------------------------------------------
    "invalid_record": "The record is not a graph record at '{at}': {reason}. Fix the record's JSON.",
    "answer_stale": "The answer was given for {waiting_id!r}, and another question waits at that place by now, {current!r}; show that one and answer it.",
    "cannot_move_to_version": "'{node}' cannot move from version {version} to {target}; it has {versions}.",
    "from_run_missing": "'{node_id}' needs a {needs} for '{parameter}', and from_run gives none (it gives {given}); pass from_run={{{needs}: ...}}.",
    "invalid_answer": "That is not a valid answer: {reason}",
    "input_given_twice": "'{name}' is given twice, by its name and by its address '{address}'; give it once.",
    "invalid_edge_value": "The value an edge brought to '{name}' is not one it takes: {reason}. Fix the node that made it, or connect an output of the type the input takes.",
    "invalid_input": "The value given for '{name}' is not valid: {reason}. Give a value of the type the input takes.",
    "invalid_max_items": "max_items is at least 1; it was {max_items}.",
    "invalid_checkpoint_every": "checkpoint_every is a number of seconds above 0; it was {checkpoint_every}.",
    "invalid_timeout": "timeout is a number of seconds above 0; it was {timeout}.",
    "missing_input": "'{name}' is required; give it in the call.",
    "not_a_chain": (
        "'{node_id}' can't follow '{before}' in a chain: a chain takes its one required open input (it has {inputs}) "
        "from the one output of the node before it (that has {outputs}); pass edges= to make_graph."
    ),
    "node_not_in_graph": "'{node_id}' is not a node of this graph; it has {nodes}.",
    "not_offered": "The graph has no input '{name}'; it has {inputs}.",
    "not_waiting": "Nothing is waiting on an answer as {waiting_id!r} in a run that is {status}; run.waiting lists what is.",
    "run_not_complete": "The run is {status}, so it has no outputs yet; run.results has what it did. {reason}",
    "state_of_another_graph": "The saved run is of another graph ({saved}) than this one ({graph}); an edited graph starts a new run.",
    "state_unreadable": "The saved run does not fit this graph at '{at}': {reason}. Load it into the graph it was saved from.",
    "state_unwritable": "The run can't be written as JSON at '{at}': {reason}. Give that field a type pydantic writes as JSON.",
    # -- why a node or a run stopped -------------------------------------------
    "execution_failed": "The node failed; its developer finds why in the conductor log.",
    "external_failed": "An outside service did not answer.",
    "failed": "The node reported a failure.",
    "invalid_output": "The node returned a value that is not what it declares: {reason}. Fix what the node returns.",
    "question_unwritable": "The node asked a person, but what it received at '{input}' has no JSON form to show them: {reason}. Give that input a type pydantic writes as JSON.",
    "invalid_ask": "The node asked a question its outputs can't be answered with: {reason}. Fix the node's Ask.",
    "item_ceiling": "{series} has {items} items, more than the {max_items} this run may work through; resume it with a larger max_items.",
    "run_cancelled": "The run was cancelled. Resume it to carry on, or start a new run.",
    "run_timed_out": "The run took too long and was stopped; run.resume() carries on.",
    "timeout": "The node did not answer in time; its Policy(attempt_timeout=) sets how long it may take.",
}

#: The English of a problem whose ``ctx.hint`` names the fix, by its code and that hint, in place of the code's own
#: entry: the same code, said for the one mistake the hint is for.
HINTED_CATALOGUE: Mapping[tuple[str, str], str] = {
    ("type_mismatch", "split_first"): (
        "'{source}' is a whole {whole} and '{node_id}.{field}' takes one {expected}. To run '{node_id}' once for each record or item "
        "in it, put a Split between them and wire Split's result, or one field of it."
    ),
    ("type_mismatch", "add_column"): (
        "'{node_id}.{field}' needs the column '{column}' ({column_type}); '{source}' has {has}. "
        "Add the column in a node between, or connect a table that has it."
    ),
    ("invalid_input", "make_a_series"): (
        "'{name}' takes one {expected} and was given a list of {length}. A graph runs once per item only after a node that makes "
        "a Series, such as a list or an upload node: give the list to such a node, or give one value."
    ),
    ("unknown_output", "split_first"): (
        "'{source}' names a field of a whole {whole}, which has no field to wire. Split it first: put a Split after it and wire "
        "the field from Split's result."
    ),
}

#: The codes a host translating by code has to cover.
PROBLEM_CODES: frozenset[str] = frozenset(PROBLEM_CATALOGUE)


_SAID_OF_A_VALIDATOR = "The value did not pass its type's own check."


def value_free_errors(invalid: ValidationError) -> list[dict[str, Any]]:
    """pydantic's errors as a problem may hold them: nothing of the value.

    Each keeps its ``type``; its ``loc`` is an empty list, since a key of the value can stand in pydantic's, and a caller
    that knows the type it read gives back the steps the type declares. ``msg`` is pydantic's own when that speaks of the type alone,
    such as a limit; a sentence built from what a validator raised or from the value's own tag is replaced, because
    it may quote the value.
    """
    return [
        {"type": error["type"], "loc": [], "msg": _SAID_OF_A_VALIDATOR if {"error", "tag"} & error.get("ctx", {}).keys() else error["msg"]}
        for error in invalid.errors(include_url=False, include_input=False)
    ]


#: Reads the ``ctx`` a node gives its error: JSON values only, since a problem is saved and streamed as JSON.
_CTX = TypeAdapter(dict[str, JsonValue])


@final
class Problem(ConductorModel):
    """One thing wrong, in pydantic's shape: ``type``, ``loc``, ``msg`` and ``ctx``.

    ``type`` is the stable code a host keys on. ``msg`` is for a person,
    never parsed; a host with its own wording translates by ``type`` and fills
    its sentence from ``ctx``, the values the message was made from under
    stable keys. ``loc`` says what it is about, and how to read it depends on
    who reported it, as it does between pydantic's body and query:

    ========================  ==================================================
    a problem of a graph      ``(node_id)`` or ``(node_id, field)``; of a graph
                              embedded in it, the same with the id of the node
                              that embeds it in front: ``("e", "upper", "text")``
    a node's failure          ``(node_id, *item)``: ``("b", 2, 0)`` is the first
                              item inside the third of what ``b`` runs per.
                              Inside an embedded graph, each embedding node and
                              its item come first: ``("e", 1, "b", 2)``
    a run stopped as a whole  ``()``
    a refused call            ``("inputs", name)``, ``(node_id, *item)`` for an
                              answer, ``("state",)``, ``("record", *where)``
                              for a record that isn't one; a graph that can't
                              run is refused with its own problems
    ========================  ==================================================

    A graph problem's field need not exist: several are about a value for a
    removed input or an edge to an unknown output.
    """

    type: str
    loc: tuple[str | int, ...] = ()
    msg: str
    ctx: Mapping[str, JsonValue] = Field(default_factory=dict)


#: A value ``make_problem`` is given for a problem's ``ctx``: JSON, written by the code that made the problem. Read-only
#: where pydantic's ``JsonValue`` is a ``list``, so a caller's ``list[str]`` is one.
CtxValue: TypeAlias = "str | int | float | bool | None | Sequence[CtxValue] | Mapping[str, CtxValue]"


def make_problem(code: str, /, *loc: str | int, **ctx: CtxValue) -> Problem:
    """The problem ``code`` at ``loc``, its message made from ``ctx``, which the record keeps as it is given.

    ``ctx`` is conductor's own, typed as JSON where it is written, so the record is built without reading it again.

    A list in ``ctx`` stays a list, and the message writes it joined by
    commas (``none`` when empty); in a list of Series, ``None`` stands for a
    node that runs once. A ``hint`` in ``ctx`` picks the code's entry in ``HINTED_CATALOGUE``.
    A code not in ``PROBLEM_CATALOGUE``, or a slot its entry
    names that ``ctx`` does not fill, is a programming error and raises; only
    ``reason`` may be left out, the sentence a type's own constructor raised,
    which not every one gives.
    """
    said = {key: _said(value) for key, value in ctx.items()}
    reason = said.get("reason")
    if isinstance(reason, str):
        said["reason"] = reason.rstrip(".")  # the entry's own sentence goes on after it
    slots = {"reason": "", "node_id": loc[0] if loc else None, "field": loc[1] if len(loc) > 1 else None, **said}
    entry = HINTED_CATALOGUE[code, str(ctx["hint"])] if "hint" in ctx else PROBLEM_CATALOGUE[code]
    return Problem.model_construct(type=code, loc=loc, msg=entry.format_map(slots).rstrip(), ctx=ctx)


def _said(value: CtxValue) -> CtxValue:
    """A ``ctx`` value as its message writes it: a list joined by commas, ``None`` in it a node that runs once."""
    if not isinstance(value, (list, tuple)):
        return value
    return ", ".join("none (it runs once)" if item is None else str(item) for item in value) or "none"


class ConductorError(Exception):
    """Base for every error conductor raises; catching it catches each of them.

    Attributes:
        message: What went wrong, for whoever meets the error.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


#: Every code a ``NodeDeclarationError`` has, so a type checker refuses one that is misspelled where it is raised.
DeclarationCode = Literal[
    "arriving_on_compute_added_inputs",
    "async_run",
    "bound_invalid",
    "choices_invalid",
    "compute_added_inputs_without_var_keyword",
    "default_invalid",
    "duplicate_node_id",
    "duplicate_type_id",
    "exclusive_group_on_input",
    "from_run_misplaced",
    "from_run_not_a_class",
    "hook_parameter_unknown",
    "input_without_type",
    "invalid_node_id",
    "nested_series",
    "not_for_a_graph",
    "output_without_type",
    "outputs_not_frozen",
    "parameter_not_by_name",
    "replaces_another_node",
    "type_not_carried",
    "type_variable_constrained",
    "type_variable_only_in_return",
    "upgrade_missing",
    "upgrade_without_replaces",
    "var_keyword_without_hook",
    "version_not_after",
    "widget_outside_param",
]


class NodeDeclarationError(ConductorError, TypeError):
    """A node or a type was declared wrong; the developer writing it meets this at import.

    Raised by ``@node`` for the signature a node declares, by ``Param`` and
    ``Policy``, and by ``NodeRegistry``. Fixed in code, never by whoever draws
    a graph, so there is no ``Problem`` for it and the text stays English.
    ``code`` is stable, as pydantic's ``PydanticUserError`` has one.

    Attributes:
        code: The stable word for what was declared wrong, one of ``DeclarationCode``; ``docs/problems.md`` has each.
    """

    def __init__(self, message: str, *, code: DeclarationCode) -> None:
        super().__init__(message)
        self.code = code


class InvalidCallError(ConductorError, ValueError):
    """A call on a graph or a run was given what it cannot act on; nothing happened.

    Raised by ``graph.run``, ``graph.with_inputs``, ``run.resume``,
    ``run.stream``, ``run.answer``, ``run.with_inputs``, ``run.again``, ``run.forget``,
    ``run.outputs``, ``graph.load_run``, ``registry.load`` (for a record
    that isn't one), ``registry.upgrade`` and ``make_graph`` from the call
    that caused it, before anything runs,
    listing every problem at once. The run the caller holds is untouched. A
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
    The engine adds the node and the item. It never retries a ``NodeError``;
    a failure the outside world caused is an ``ExternalError``::

        raise NodeError("The document could not be read.", code="unreadable", ctx={"pages": 0})

    Attributes:
        code: The stable word a host keys on: the node's own, else ``failed`` (an ``ExternalError``'s
            ``external_failed``).
        ctx: The values a host fills its own sentence from, as JSON.

    Raises:
        pydantic.ValidationError: When ``ctx`` holds a value that has no JSON form, such as an object, a set or a
            tuple. It is raised where the error is made, so a node that does this fails as ``execution_failed`` with a
            traceback at its own ``raise``.
    """

    def __init__(self, message: str, *, code: str = "failed", ctx: Mapping[str, JsonValue] = MappingProxyType({})) -> None:
        super().__init__(message)
        self.code = code
        self.ctx = _CTX.validate_python(ctx)

    def _as_problem_at(self, *loc: str | int) -> Problem:
        """The ``Problem`` this is at ``loc``, built from what the error already read."""
        return Problem.model_construct(type=self.code, loc=loc, msg=self.message, ctx=self.ctx)


class ExternalError(NodeError):
    """The outside world failed: a network error, a rate limit, an upstream 5xx.

    The one failure the engine retries, under the node's ``Policy``. A node
    raises it where it knows the failure is external. An exception whose
    class the policy's ``retry_on`` names is treated the same way, with code
    ``external_failed`` and a generic message. Not for a bug in the node or a
    value it cannot use: retrying those repeats the same failure.
    """

    def __init__(self, message: str, *, code: str = "external_failed", ctx: Mapping[str, JsonValue] = MappingProxyType({})) -> None:
        super().__init__(message, code=code, ctx=ctx)
