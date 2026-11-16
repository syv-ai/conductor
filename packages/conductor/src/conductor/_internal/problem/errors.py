"""The exceptions conductor raises, and how it keeps a run's values out of what it reports.

Every ``ConductorError`` has ``Problem`` records, just like a graph's problems. So a host has one type to
translate. Conductor can also raise Python's own exceptions, such as a ``TypeError`` or a ``KeyError``, and
pydantic's ``ValidationError``. Here is who meets what, and when:

    who                       what                                        when
    a node's developer        NodeDeclarationError (raised)               the module that declares the node is imported
    a graph's author          a Problem in graph.problems (not raised)    the graph is compiled
    a caller                  InvalidCallError (raised)                   graph.run, run.answer, ... is given a wrong value
    whoever reads a run       a Problem as run.cause                      a run ended early

Only two of these are raised. A graph with problems is data, so an editor can show the problems while the graph is
being drawn. A run that ends early doesn't raise either: ``run.resume()`` returns the run, and ``run.cause`` says
why it stopped.

    ConductorError
    ├── NodeDeclarationError    a node was declared wrong; also a TypeError
    ├── InvalidCallError        a call on a graph or a run was given a value it can't use; also a ValueError
    └── NodeError               a node failed, with a message for people; not retried
        └── ExternalError         the outside world failed; retried

**What is retried.** A node can fail because of a bug or bad data. It can also fail because the outside world did,
with a network error, a rate limit or a server error. Only the second is worth trying again. A node says that a
failure is of this sort by raising ``ExternalError``, or by naming the client's exception classes in
``Policy(retry_on=...)``. Any other exception fails the call with ``execution_failed``, and is not retried.

**What a person is told.** A problem may be shown to anyone, since a host may stream problems to a browser. So
conductor keeps a node's values out of a failed call's problem. A ``NodeError`` keeps its message, since the node
wrote it for people. Any other exception's text might quote a value, so it goes to the ``conductor`` logger with
its traceback, and the problem names only the exception's class. ``value_free_errors`` does the same for pydantic's
errors. A few kinds do quote an exception's text. ``compute_hook_failed`` quotes the hook's, and
``question_unwritable`` quotes why an input can't be written as JSON. Of ``check_registry``'s kinds, ``hook_failed``
and ``run_failed`` quote a ``compute_hook_failed`` message, and ``upgrade_broken`` (its variant ``raised``) quotes
the upgrade's exception.

Asking a person isn't an error. A node raises ``Ask``, its call waits, and the run's status is ``waiting``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from pydantic import JsonValue, TypeAdapter, ValidationError

from .run_problems import external_failed, node_error
from .type import Problem

_VALIDATOR_PREFIX = "The value did not pass its type's own check."


def value_free_errors(error: ValidationError) -> list[dict[str, Any]]:
    """Returns pydantic's errors with nothing of the refused value left in them.

    A problem's ``ctx.errors`` is made from this. Each error keeps its ``type``. Its ``loc`` is emptied, since
    pydantic's ``loc`` may contain a key of the value. A caller that knows the type can put back the steps the type
    declares. Its ``msg`` is kept when it only talks about the type, as a limit's message does. It is replaced when it
    came from what a validator raised or from a tag of the value, since those may quote the value.

    For example, an ``int`` refusing ``"many"`` gives::

        [{"type": "int_parsing", "loc": [], "msg": "Input should be a valid integer, unable to parse string as an integer"}]

    Args:
        error: The error pydantic raised.

    Returns:
        One dict per error, with the keys ``type``, ``loc`` and ``msg``.
    """
    return [
        {
            "type": fault["type"],
            "loc": [],
            "msg": _VALIDATOR_PREFIX
            if {"error", "tag"} & fault.get("ctx", {}).keys()
            else fault["msg"],
        }
        for fault in error.errors()
    ]


#: Checks the ``ctx`` a node gives its ``NodeError``. It may only have JSON values, since a problem is saved and
#: sent as JSON.
_CTX = TypeAdapter(dict[str, JsonValue])


class ConductorError(Exception):
    """The base class of every exception conductor raises.

    Each one says what went wrong as a list of ``Problem`` records. So a host translates an exception the same way
    it translates a graph's problems.

    Attributes:
        problems: What went wrong, in the order it was found.
    """

    def __init__(self, problems: Iterable[Problem], text: str) -> None:
        """Makes the exception from its problems.

        Args:
            problems: What went wrong.
            text: The exception's text, made from the problems, for whoever reads a traceback.
        """
        self.problems = list(problems)
        super().__init__(text)


class NodeDeclarationError(ConductorError, TypeError):
    """A node or a type was declared wrong. It is raised as soon as the module that declares it is imported.

    Here ``fetch`` is an ``async def``, which a node can't be::

        @node("fetch", title="Fetch", description="Fetches a page.")
        async def fetch(url: str) -> str: ...
        # NodeDeclarationError: @node('fetch') on fetch: a node is a plain def, not an async def. For slow calls, ...

    ``@node``, ``subgraph``, ``Param``, ``@host_type`` and ``NodeRegistry`` raise it. Only the node's developer can
    fix it, in code, so it is an exception rather than one of a graph's problems. Its code, like ``async_run``, is
    ``error.problems[0].type``, and it stays the same between versions.

    Attributes:
        problems: One problem, whose ``type`` is the code, like ``async_run`` above. The codes are in
            ``DECLARATION_PROBLEMS``, and ``python -m conductor.about problems`` has each with an example.
        where: Where the fault is, outermost first, when ``@node`` or ``subgraph`` finds it while building the node:
            in the function or its signature, a hook, or a missing ``Upgrade``. It starts with the node, like
            ``"@node('fetch') on fetch"``, and for a fault in one input it then names the input. The error's text
            is these and the problem's message, joined by ``": "``. Other faults, such as an empty title or a node
            id used twice, have no ``where``; their message names the node, or nothing.
    """

    def __init__(self, problem: Problem, *where: str) -> None:
        super().__init__([problem], ": ".join([*where, problem.msg]))
        self.where = where

    def within(self, place: str) -> NodeDeclarationError:
        """Returns this error with ``place`` added in front of ``where``.

        A fault found in one input is raised first with the input's name. The code that checks the whole node then
        adds the node in front, like ``@node('fetch') on fetch``.
        """
        return NodeDeclarationError(self.problems[0], place, *self.where)


class InvalidCallError(ConductorError, ValueError):
    """A value given to conductor was refused, and nothing was done with it.

    Say a graph's input ``n`` is an ``int``. Then ``graph.run(n="many")`` raises it, with one ``invalid_input``
    problem. It is raised before anything runs, and the run the caller has stays as it was. A call's arguments are
    checked in stages, and the first stage that finds a fault raises with every problem of that stage. So
    ``graph.run(n="many", max_items=0)`` raises only ``invalid_limit``. Like pydantic's error, it is also a
    ``ValueError``.

    Most calls a host makes can raise it: ``graph.run``, ``graph.with_inputs``, ``run.resume``, ``run.stream``,
    ``run.answer``, ``run.with_inputs``, ``run.again``, ``run.forget``, ``run.outputs``, ``Run.from_json``,
    ``Graph.compile`` (for JSON that isn't a graph record), ``read_json``, ``registry.upgrade`` and ``make_graph``.

    Every validator raises it when it refuses a value from outside. A value that came from a graph or a run, not
    from a caller, isn't raised. Compiling puts its problems in ``graph.problems``, and a run keeps the problem as
    the outcome of the call it belongs to.

    Attributes:
        problems: What is wrong, in the order it was found.
    """

    def __init__(self, problems: Iterable[Problem]) -> None:
        problems = list(problems)
        super().__init__(
            problems,
            "\n".join(
                [
                    f"{len(problems)} problem(s):",
                    *(
                        f"  {'.'.join(str(place) for place in problem.loc)} — {problem.type}: {problem.msg}"
                        for problem in problems
                    ),
                ]
            ),
        )


class NodeError(ConductorError):
    """Raised by a node when it fails, with a message for people.

    For example::

        raise NodeError("The document could not be read.", code="unreadable", ctx={"pages": 0})

    The error carries one problem, made when it is raised: ``Problem(type="unreadable", loc=(), msg="The document
    could not be read.", ctx={"pages": 0})``. The call fails with that problem, and conductor fills in its ``loc``
    with the node and the item indexes. Without a ``code``, the code is ``node_error``.

    A ``NodeError`` is never retried. For a failure of the outside world, raise ``ExternalError``.

    Attributes:
        problems: The one problem, with an empty ``loc``. Its ``type`` is the code, its ``msg`` the message, and its
            ``ctx`` the values, as JSON, that a host can make its own message from.

    Raises:
        pydantic.ValidationError: ``ctx`` has a value that isn't JSON, such as an object, a set or a tuple. It is
            raised from the node's own ``raise``, so the call fails with ``execution_failed``.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = node_error.code,
        ctx: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> None:
        super().__init__(
            [
                Problem.model_construct(
                    type=code, loc=(), msg=message, ctx=_CTX.validate_python(ctx)
                )
            ],
            message,
        )


class ExternalError(NodeError):
    """Raised by a node when the outside world failed, such as a network error, a rate limit or a server error.

    It is retried, as the node's ``Policy`` says. Without a ``code``, the code is ``external_failed``. An exception
    whose class is named in the policy's ``retry_on`` is retried the same way, and fails with ``external_failed``
    and a general message.

    Don't raise it for a bug in the node or for a value it can't use. Trying again would only fail the same way.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = external_failed.code,
        ctx: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> None:
        super().__init__(message, code=code, ctx=ctx)
