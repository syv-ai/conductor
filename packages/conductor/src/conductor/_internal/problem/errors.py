"""The exceptions conductor raises, and how a failure is kept free of the values a run handled.

Everything wrong that a person needs to hear about is a ``Problem``, so a host has one type to translate. Who meets
which, and when:

    who                       what                                        when
    a node's developer        NodeDeclarationError (raised)               the module that declares the node is imported
    a graph's author          a Problem in graph.problems (not raised)    the graph is compiled
    a caller                  InvalidCallError (raised)                   graph.run, run.answer, ... is given a wrong value
    whoever reads a run       a Problem as run.state's cause              a run ended early

A graph with problems is data, not an exception, so an editor can show it while it is being drawn. A call that cannot
be done raises, from that call. A run that ends early does not raise either: ``run.resume()`` returns the run, and
its ``state`` says why.

    ConductorError
    ├── NodeDeclarationError    a node was declared wrong; also a TypeError
    ├── InvalidCallError        a call on a graph or a run was given what it cannot act on; also a ValueError
    └── NodeError               a node failed, with a message for people; never retried
        └── ExternalError         the outside world failed; the only failure that is retried

**What is retried.** A node fails because of a bug or bad data, or because the outside world failed: a network
error, a rate limit, a server error. Only the second is worth retrying, under the node's ``Policy``. A node says so
by raising ``ExternalError``, or by naming the exception classes of the client it calls in
``Policy(retry_on=...)``. Any other exception is the code ``execution_failed`` and is not retried.

**What a person is told.** A ``NodeError`` keeps the message the node gave it, since the node wrote it for people. Any
other exception's text, which may quote a value, goes to ``logging.getLogger("conductor")`` with its traceback, and
the problem has only its class name: a host may stream problems to a browser. For the same reason
``value_free_errors`` takes the value out of pydantic's account of a refused value.

Asking a person is not an error: a node raises ``Ask``, that call waits, and the run's status is ``waiting``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import JsonValue, TypeAdapter
from pydantic_core import ErrorDetails

from conductor._internal.problem.catalogue import external_failed, failed
from conductor._internal.problem.type import Problem

_VALIDATOR_PREFIX = "The value did not pass its type's own check."


def value_free_errors(errors: list[ErrorDetails]) -> list[dict[str, Any]]:
    """pydantic's errors with nothing of the refused value in them, for a problem's ``ctx.errors``.

    ``errors`` is what ``ValidationError.errors(include_url=False, include_input=False)`` returns. For ``int`` on
    ``"many"`` the result is ``[{"type": "int_parsing", "loc": [], "msg": "Input should be a valid integer, unable to
    parse string as an integer"}]``. Each error keeps its ``type``. Its ``loc`` is emptied,
    since pydantic's may contain a key of the value; a caller that knows the type puts back the steps the type
    declares. Its ``msg`` is kept when it speaks only of the type, as a limit does, and replaced when it was made
    from what a validator raised or from a tag of the value, since those may quote the value.
    """
    return [
        {"type": error["type"], "loc": [], "msg": _VALIDATOR_PREFIX if {"error", "tag"} & error.get("ctx", {}).keys() else error["msg"]}
        for error in errors
    ]


#: Validates the ``ctx`` a node gives its ``NodeError``: JSON values only, since a problem is saved and sent as JSON.
_CTX = TypeAdapter(dict[str, JsonValue])


class ConductorError(Exception):
    """The base class of every exception conductor raises.

    Attributes:
        message: What went wrong, for whoever meets the error.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


#: Every code a ``NodeDeclarationError`` can have, so a type checker refuses a misspelled one where it is raised.
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
    "context_misplaced",
    "context_not_a_class",
    "hook_parameter_unknown",
    "input_without_type",
    "invalid_node_id",
    "nested_series",
    "output_without_type",
    "outputs_not_frozen",
    "parameter_not_by_name",
    "pydantic_field_in_hint",
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
    """A node or a type was declared wrong; raised when the module that declares it is imported.

    Raised by ``@node`` for the node's signature, by ``Param`` and ``Policy``, and by ``NodeRegistry``. Only the
    node's developer can fix it, in code, so it is an exception and not a ``Problem``. Like pydantic's
    ``PydanticUserError``, it has a code that does not change between versions.

    Attributes:
        code: What was declared wrong, one of ``DeclarationCode``; ``python -m conductor.about problems`` has each.
    """

    def __init__(self, message: str, *, code: DeclarationCode) -> None:
        super().__init__(message)
        self.code = code


class InvalidCallError(ConductorError, ValueError):
    """A call on a graph or a run was given something it cannot act on, and nothing was done.

    ``graph.run(n="many")`` on a graph whose input ``n`` is an ``int`` raises it with one ``invalid_input`` problem.
    Raised before anything runs, listing every problem at once, by ``graph.run``, ``graph.with_inputs``,
    ``run.resume``, ``run.stream``, ``run.answer``, ``run.with_inputs``, ``run.again``, ``run.forget``,
    ``run.outputs``, ``Run.from_json``, ``Graph.compile`` (for JSON that is not a graph record), ``registry.upgrade``
    and ``make_graph``. The run the caller has is unchanged. It is a ``ValueError`` too, as pydantic's error is.
    """

    def __init__(self, problems: Iterable[Problem]) -> None:
        self._problems = list(problems)
        super().__init__("\n".join([f"{len(self._problems)} problem(s):", *(f"  {'.'.join(str(place) for place in problem.loc)} — {problem.type}: {problem.msg}" for problem in self._problems)]))

    def errors(self) -> list[Problem]:
        """Every problem, in the order the call found them."""
        return list(self._problems)


class NodeError(ConductorError):
    """What a node raises when it fails, with a message for people::

        raise NodeError("The document could not be read.", code="unreadable", ctx={"pages": 0})

    The call fails with a ``Problem`` of that code, message and ``ctx``; conductor adds the node and the item indexes
    as its ``loc``. Without a ``code`` the problem's code is ``failed``. A ``NodeError`` is never retried: a failure
    of the outside world is an ``ExternalError``.

    Attributes:
        code: The problem's code: the node's own, else ``failed`` (``external_failed`` for an ``ExternalError``).
        ctx: The values a host makes its own message from, as JSON.

    Raises:
        pydantic.ValidationError: ``ctx`` has a value that is not JSON, such as an object, a set or a tuple. It is
            raised by the node's own ``raise``, so the call fails as ``execution_failed``.
    """

    def __init__(self, message: str, *, code: str = failed.code, ctx: Mapping[str, JsonValue] = MappingProxyType({})) -> None:
        super().__init__(message)
        self.code = code
        self.ctx = _CTX.validate_python(ctx)

    def _as_problem_at(self, *loc: str | int) -> Problem:
        """This error as a ``Problem`` at ``loc``; ``ctx`` was validated when the error was made."""
        return Problem.model_construct(type=self.code, loc=loc, msg=self.message, ctx=self.ctx)


class ExternalError(NodeError):
    """What a node raises when the outside world failed: a network error, a rate limit, a server error.

    The only failure that is retried, under the node's ``Policy``. An exception whose class the policy's
    ``retry_on`` names is retried the same way, and fails with the code ``external_failed`` and a general message.
    Not for a bug in the node or a value it cannot use: a retry would fail the same way.
    """

    def __init__(self, message: str, *, code: str = external_failed.code, ctx: Mapping[str, JsonValue] = MappingProxyType({})) -> None:
        super().__init__(message, code=code, ctx=ctx)
