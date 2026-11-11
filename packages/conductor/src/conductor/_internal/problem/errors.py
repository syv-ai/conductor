"""The exceptions conductor raises, and what a failure may say.

Everything wrong that a person has to hear about is a ``Problem``: what
loading a graph found (``graph.problems``), what a refused call lists
(``InvalidCallError.errors()``), and why a node or a run stopped
(``NodeFailed.cause``, an ``EndedEarly`` ``run.state``'s ``cause``). So a host has one type to
translate. Who meets what:

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

**What people are told.** A ``NodeError`` the node raised keeps
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

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

from conductor._internal.problem.catalogue import external_failed, failed
from conductor._internal.problem.type import Problem

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
    "empty_title_or_description",
    "exclusive_group_on_input",
    "from_run_misplaced",
    "from_run_not_a_class",
    "hook_parameter_unknown",
    "input_without_type",
    "invalid_node_id",
    "nested_series",
    "output_without_type",
    "outputs_not_frozen",
    "parameter_not_by_name",
    "replaces_another_node",
    "type_declaration_invalid",
    "type_not_carried",
    "type_not_named",
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


class UnitFailedError(Exception):
    """A unit's call ends on a problem conductor names itself: what the node was to be given, or what it returned, is refused.

    Raised where the fault is met, which knows what is wrong and not which unit it is; the leg gives the problem
    its place and records it as the unit's failure. A caller never meets it: a run read afterwards
    (``run.results``) reads a blank where the fault was.

    Attributes:
        problem: The problem, with no ``loc``.
    """

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem.msg)
        self.problem = problem


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

    def __init__(self, message: str, *, code: str = failed.code, ctx: Mapping[str, JsonValue] = MappingProxyType({})) -> None:
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

    def __init__(self, message: str, *, code: str = external_failed.code, ctx: Mapping[str, JsonValue] = MappingProxyType({})) -> None:
        super().__init__(message, code=code, ctx=ctx)
