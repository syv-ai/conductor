"""The exceptions conductor raises, and how it keeps a run's values out of what it reports.

Every ``ConductorError`` has a list of ``Problem``, just like a graph's problems. So a host has one type to
translate. Conductor can also raise Python's own exceptions, such as a ``TypeError`` or a ``KeyError``, and
pydantic's ``ValidationError``. Here is who meets what, and when:

    who                       what                                        when
    a node's developer        NodeDeclarationError (raised)               the module that declares the node is imported
    a graph's author          a Problem in graph.problems (not raised)    the graph is compiled
    a caller                  InvalidCallError (raised)                   graph.run, run.resume, ... is given a wrong value
    whoever reads a run       a Problem as run.cause                      a run ended early

Only two of these are raised. A graph with problems is data, so an editor can show the problems while the graph is
being drawn. A run that ends early doesn't raise either: ``run.resume()`` returns the run, and ``run.cause`` says
why it stopped.

    ConductorError
    ├── NodeDeclarationError    a node was declared wrong; also a TypeError
    ├── InvalidCallError        a call on a graph or a run was given a value it can't use; also pydantic's
    │                           ValidationError, and so a ValueError
    └── NodeError               a node failed, with a message for people; not retried
        └── ExternalError         the outside world failed; retried

**What is retried.** A node can fail because of a bug or bad data. It can also fail because the outside world did,
with a network error, a rate limit or a server error. Only the second is worth trying again. A node says that a
failure is of this sort by raising ``ExternalError``, or by naming the client's exception classes in
``Policy(retry_on=...)``. Any other exception fails the call with ``execution_failed``, and is not retried.

**What a person is told.** A problem may be shown to anyone, since a host may stream problems to a browser. So a
problem never quotes a value a graph carries: a graph input or its default, a constant, an output, an answer or a
value in a saved run. Where a fault is inside such a value, the problem says where by list indexes, declared fields and
positions, never by the value's own dict keys. A problem never quotes the text of an exception other than a
``NodeError`` either, since it might quote a value. Such an exception from a node, a hook or an ``Upgrade``'s rewrite
goes to the ``conductor`` logger with its traceback, and the problem names only the exception's class:
``author_code_problem`` is that rule. ``value_free_errors`` does the same for pydantic's errors.

A problem may name what a developer or a host wrote: node ids, input, output and field names, places, waiting ids,
the limits a type declares and where a JSON text stops being JSON. A field name a ``compute_...`` hook returns is
among them, even one it builds from a constant: it is in every ``loc``, in ``describe()`` and on edges. A
``NodeError`` keeps its message and its ``ctx``, since the node's developer wrote them for people.

Asking a person isn't an error. A node raises ``Ask``, its call waits, and the run's status is ``waiting``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Self, cast

from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticCustomError, PydanticKnownError
from pydantic_core.core_schema import ErrorType

from conductor._internal.base import place_text, plain_json

from .run_problems import external_failed, node_error
from .type import Problem

_logger = logging.getLogger("conductor")

#: The message that replaces one that might quote the value. Like pydantic's own, it has no final period, since a
#: problem's message quotes it as its reason.
_VALIDATOR_PREFIX = "The value did not pass its type's own check"

#: The ``ctx`` keys by which some of pydantic's own messages quote the value or an exception's text: ``error`` (a
#: validator's exception in ``value_error``, a parser's in ``uuid_parsing`` or ``date_parsing``), ``tag`` (a union's
#: tag, read from the value), ``encoding_error`` (the byte that isn't base64) and ``tz_actual`` (a datetime's offset).
_QUOTING_KEYS = frozenset({"error", "tag", "encoding_error", "tz_actual"})

#: The ``ctx`` that marks a pydantic error whose message conductor wrote to quote no value. ``conductor_error`` adds it
#: to an error conductor raises, and a schema's ``custom_error_context`` gives it to the error the schema reports.
OWN_MESSAGE: Mapping[str, bool] = MappingProxyType({"conductor": True})


def conductor_error(
    error_type: str, message: str, ctx: Mapping[str, Any] | None = None
) -> PydanticCustomError:
    """Makes a pydantic error whose message conductor wrote.

    Every pydantic error conductor raises is made here, whether it refuses a value, a graph record or what a node returned.
    Conductor writes ``message`` to name only what a developer or a host wrote, such as a column of a row type, and
    never a value. So ``value_free_errors`` keeps it, where it replaces the message of an error a host's validator
    raised.

    Args:
        error_type: The error's type, like ``"column_missing"``.
        message: The message, a template with ``{key}`` for each key of ``ctx``.
        ctx: The values the message is made from.

    Returns:
        The error to raise.
    """
    return PydanticCustomError(error_type, message, {**(ctx or {}), **OWN_MESSAGE})


def is_conductor_error(fault: ErrorDetails) -> bool:
    """Says whether conductor raised one of pydantic's errors, with ``conductor_error`` or a schema's ``OWN_MESSAGE``.

    Only such an error's message and ``ctx`` are conductor's own words, so only they are read as conductor wrote
    them: its message is kept, and a ``column`` in its ``ctx`` names a table's column.
    """
    return OWN_MESSAGE.items() <= fault.get("ctx", {}).items()


def no_json_form(reason: str = "the value can't be written as JSON") -> PydanticCustomError:
    """The refusal of a value that can't be written as JSON, such as a ``str`` with a lone surrogate, or an object
    held in memory that pydantic only checks with ``isinstance``.

    Args:
        reason: The message. A frame given where the type is no table says how to give it instead, and a list of
            choices says that one of them can't be written.
    """
    return conductor_error("no_json_form", reason)


def set_has_no_order() -> PydanticCustomError:
    """The refusal of a set where a value is read from its JSON form, written into a graph record, or given for a
    ``Series``.

    A set's order changes from one process to the next, so its items would come in a different order each time,
    and the same set would be written as a different JSON text.
    """
    return conductor_error(
        "set_has_no_order",
        "a set has no order, so its items could come in a different order each time",
    )


def value_free_errors(error: ValidationError) -> list[dict[str, Any]]:
    """Returns pydantic's errors with nothing of the refused value left in them.

    A problem's ``ctx.errors`` is made from this. Each error keeps its ``type``. Its ``loc`` is emptied, since
    pydantic's ``loc`` may contain a key of the value. A caller that knows the type puts back the steps the type
    declares (``Type.refusal``). Its ``msg`` and its ``ctx`` are kept when conductor wrote them, or when pydantic
    wrote them about the type, as a limit's are: ``{"ge": 3}`` is the type's, and lets a host write its own message
    by the code. Of pydantic's ``ctx`` only the keys its message reads are kept (``_ctx_its_message_reads``). Any
    other message is replaced and its ``ctx`` dropped: one a host's validator raised, whatever type it gives, and one
    of pydantic's that quotes the value or an exception's text, like ``value_error``'s.

    For example, an ``Annotated[int, Param(ge=3)]`` refusing ``1`` gives::

        [{"type": "greater_than_equal", "loc": [], "msg": "Input should be greater than or equal to 3", "ctx": {"ge": 3}}]

    Args:
        error: The error pydantic raised.

    Returns:
        One dict per error, with the keys ``type``, ``loc`` and ``msg``, and ``ctx`` when it is kept and not empty.
    """
    return [
        _written_error(fault["type"], [], fault["msg"], _ctx_its_message_reads(fault))
        if _message_holds_no_value(fault)
        else _written_error(fault["type"], [], _VALIDATOR_PREFIX, {})
        for fault in error.errors()
    ]


def shape_error(
    fault: ErrorDetails | PydanticCustomError | PydanticKnownError, loc: Iterable[str | int] = ()
) -> dict[str, Any]:
    """Returns one error of a problem's ``ctx.errors`` for a fault in the shape of what a host or a developer wrote.

    Such a fault is in a graph record, in what a ``compute_...`` hook or an ``Upgrade``'s rewrite returned, in what a
    node returned, or in a saved run's lists. So its message and ``ctx`` are kept as pydantic or conductor wrote them:
    they name what was written, such as a node id, and never a value. A record's fault inside a constant is the one
    that could quote a value, and ``NodeRecord`` frees it of the value where it reads the constant (``_CARRIED``,
    ``_one_fault`` in ``record/node.py``), so it reaches here without one. Its ``loc`` is ``loc``,
    the steps from where the problem's own ``loc`` points, often none. A fault found outside a pydantic schema, such as
    a key written twice in a record's text, is given as the error pydantic or conductor would raise for it.

    For example, ``shape_error(conductor_error("repeated_key", "a key is written twice in one object"))`` is
    ``{"type": "repeated_key", "loc": [], "msg": "a key is written twice in one object"}``.
    """
    if isinstance(fault, (PydanticCustomError, PydanticKnownError)):
        return _written_error(fault.type, list(loc), fault.message(), fault.context or {})
    return _written_error(fault["type"], list(loc), fault["msg"], fault.get("ctx", {}))


def argument_errors(
    error: ValidationError, arguments: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Returns the faults pydantic found in the arguments a developer gave one of conductor's own functions, by
    argument, each as ``shape_error`` writes it, its ``loc`` the steps inside the argument.

    ``make_graph``, ``@node``, ``subgraph`` and ``NodeRegistry`` check their arguments this way. A key of a mapping
    that isn't text is part of what was given, and pydantic writes it into a ``loc`` as text, its ``repr`` for an
    object, at any depth. So the key's own fault stops at it, ``[..., "[key]"]``, and a fault in the value under such
    a key is left out (``_steps_given``).

    Args:
        error: What pydantic raised, for a schema whose fields are the arguments.
        arguments: The arguments, by name.
    """
    faults: dict[str, list[dict[str, Any]]] = {}
    for fault in error.errors():
        argument, *steps = fault["loc"]
        given = _steps_given(arguments[str(argument)], steps)
        if given is not None:
            faults.setdefault(str(argument), []).append(shape_error(fault, given))
    return faults


def _steps_given(given: Any, steps: list[str | int]) -> list[str | int] | None:
    """Returns the steps of a fault's ``loc`` that name a place in ``given``, walking ``given`` along them.

    A step is a key of a mapping, as text, or an index of a sequence. Pydantic's label of a union's member, ``"dict"``
    or ``"list"`` in a ``JsonValue``, names no place and is left out, so ``{"a": [1, (2,)]}`` is refused at
    ``["a", 1]``. A step that is neither is a key that isn't text: its own fault ends at ``"[key]"``, and a fault
    under it gives ``None``. A step into anything else, such as a model, is kept as pydantic wrote it.
    """
    kept: list[str | int] = []
    for at, step in enumerate(steps):
        if step == "[key]":
            return [*kept, "[key]"]
        if isinstance(given, Mapping):
            if isinstance(step, str) and step in given:
                kept.append(step)
                given = given[step]
            elif step != "dict":
                return [*kept, "[key]"] if steps[at + 1 :] == ["[key]"] else None
        elif isinstance(given, Sequence) and not isinstance(given, (str, bytes)):
            if isinstance(step, int):
                kept.append(step)
                given = given[step]
            elif step != "list":
                return [*kept, *steps[at:]]
        else:
            return [*kept, *steps[at:]]
    return kept


def _ctx_its_message_reads(fault: ErrorDetails) -> Mapping[str, Any]:
    """Returns the ``ctx`` of an error whose message is kept, with only the keys its message is made from.

    A message that matches pydantic's own proves only the keys it reads. A host's validator may raise
    ``PydanticCustomError("greater_than", "Input should be greater than {gt}", {"gt": 0, "seen": value})``: the
    message is pydantic's, and ``seen`` holds the value. So for one of pydantic's errors a key is kept only when
    pydantic's message for the error's type changes, or can't be written, without it. A ``ctx`` conductor wrote is
    kept whole: conductor names in it only what a developer wrote.
    """
    ctx = fault.get("ctx", {})
    if is_conductor_error(fault):
        return ctx

    def message(given: Mapping[str, Any]) -> str | None:
        try:
            return PydanticKnownError(cast(ErrorType, fault["type"]), dict(given) or None).message()
        except (KeyError, TypeError):
            return None

    whole = message(ctx)
    return {
        key: value
        for key, value in ctx.items()
        if message({other: kept for other, kept in ctx.items() if other != key}) != whole
    }


def _written_error(
    error_type: str, loc: list[str | int], msg: str, ctx: Mapping[str, Any]
) -> dict[str, Any]:
    """Returns one error of a problem's ``ctx.errors``, with ``ctx`` as plain JSON, without conductor's mark, and
    left out when it is empty."""
    kept = {key: value for key, value in ctx.items() if key not in OWN_MESSAGE}
    written: dict[str, Any] = {"type": error_type, "loc": loc, "msg": msg}
    if kept:
        written["ctx"] = plain_json(kept)
    return written


def _message_holds_no_value(fault: ErrorDetails) -> bool:
    """Says whether one error's message was written by conductor, or by pydantic about the type, so it quotes no value.

    A message is pydantic's only when pydantic writes that same text for the error's type and ``ctx``, for a value
    read from Python or from JSON. The type alone proves nothing, since a host's validator may raise
    ``PydanticCustomError("value_error", f"{value} is bad")``.

    A host that puts its value in the ``ctx`` of one of pydantic's own errors on purpose, as
    ``PydanticKnownError("greater_than", {"gt": value})``, gets pydantic's own text back, value included. So does one
    that puts conductor's mark, ``OWN_MESSAGE``, in its own error's ``ctx``.
    """
    if is_conductor_error(fault):
        return True
    ctx = fault.get("ctx", {})
    if _QUOTING_KEYS & ctx.keys():
        return False
    known: InitErrorDetails = {"type": cast(ErrorType, fault["type"]), "loc": (), "input": None}
    if ctx:
        known["ctx"] = ctx
    try:
        # pydantic words some errors one way for a Python value and another for JSON: a list is an array there
        return fault["msg"] in {
            ValidationError.from_exception_data("", [known], input_type=read_from).errors()[0][
                "msg"
            ]
            for read_from in ("python", "json")
        }
    # the type is any text a validator gave: one pydantic doesn't know (KeyError), or a ctx without the keys its
    # message needs (TypeError)
    except (KeyError, TypeError):
        return False


#: Checks the ``ctx`` a node gives its ``NodeError``. It may only have JSON values, since a problem is saved and
#: sent as JSON.
_CTX = TypeAdapter(dict[str, JsonValue])


class ConductorError(Exception):
    """The base class of every exception conductor raises.

    Each one says what went wrong as a list of ``Problem``. So a host translates an exception the same way
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

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickles the exception as its class, its text and its problems, so a worker process can send it back.

        Python's own pickling calls the class again with the exception's text, which a subclass's ``__init__``
        doesn't take. So the exception is made again without calling ``__init__``, and its problems are put back.
        """
        return type(self).__new__, (type(self), *self.args), self.__dict__


class NodeDeclarationError(ConductorError, TypeError):
    """A node or a type was declared wrong. It is raised as soon as the module that declares it is imported.

    Here ``fetch`` is an ``async def``, which a node can't be::

        @node("fetch", title="Fetch", description="Fetches a page.")
        async def fetch(url: str) -> str: ...
        # NodeDeclarationError: fetch (version 1): a node is a plain def, not an async def. For slow calls, ...

    ``@node``, ``subgraph``, ``Param``, ``@host_type`` and ``NodeRegistry`` raise it. Only the node's developer can
    fix it, in code, so it is an exception rather than one of a graph's problems. Its code, like ``async_run``, is
    ``error.problems[0].type``, and it stays the same between versions.

    The problem's ``loc`` says where the fault is, as a graph problem's does: ``("fetch",)`` for the node, and
    ``("fetch", "url")`` for one of its inputs or outputs. It is ``()`` for a fault that is about no one node, such as
    a ``@host_type`` or two registries that disagree. The error's text is the ``loc``, written as a place is, with
    the node's version after its id, like ``fetch (version 2).url``, and then the message. So two versions of one node
    read apart.

    ``loc[0]`` is the node definition's id, such as ``"upper"``, where a graph problem's is a node's id in the graph
    record. The version is in the text only: ``loc`` is the place, as a graph problem's is. The text doesn't name the
    function; the traceback's decorator line shows it.

    Attributes:
        problems: One problem, whose ``type`` is the code, like ``async_run`` above. The codes are in
            ``DECLARATION_PROBLEMS``, and ``python -m conductor.about problems`` has each with an example.
    """

    def __init__(self, problem: Problem, *place: str, version: int | None = None) -> None:
        """Makes the error from its problem, with ``place`` in front of the problem's ``loc``.

        Args:
            problem: The fault, as its kind makes it.
            place: Where it is, outermost first, when the code that finds it knows: a node's id, a field's name.
            version: The node's version, given with the node's id. ``None`` while the fault is placed only in a
                field, before the code that checks the whole node puts the node in front.
        """
        located = problem.model_copy(update={"loc": (*place, *problem.loc)})
        text = located.msg
        if located.loc:
            # a declaration's loc is a node's id, then maybe one of its fields, as a graph problem names a place
            node, *field = located.loc
            named = str(node) if version is None else f"{node} (version {version})"
            text = f"{place_text((named,), *map(str, field))}: {located.msg}"
        super().__init__([located], text)

    def within(self, place: str, version: int | None = None) -> NodeDeclarationError:
        """Returns this error with ``place`` in front of its problem's ``loc``.

        A fault found in one input is raised first with the input's name, like ``("url",)``. The code that checks the
        whole node then puts the node's id in front, with its version: ``("fetch", "url")``, written
        ``fetch (version 1).url``.
        """
        return NodeDeclarationError(self.problems[0], place, version=version)


class InvalidCallError(ValidationError, ConductorError):
    """A value given to conductor was refused, and nothing was done with it.

    Say a graph's input ``n`` is an ``int``. Then ``graph.run({"n": "many"})`` raises it, with one ``invalid_input``
    problem. It is raised before anything runs, and the run the caller has stays as it was. A call's arguments are
    checked in stages, and the first stage that finds a fault raises with every problem of that stage. So
    ``graph.run({"n": "many"}, max_items=0)`` raises only ``invalid_limit``.

    It is also pydantic's ``ValidationError``, and so a ``ValueError``. ``except ValidationError`` catches it, and
    ``errors()`` gives one of pydantic's errors per problem, with the problem's ``type``, ``loc`` and ``msg``. Each
    error's ``ctx`` is the problem's ``ctx`` with the message added under ``msg``, since pydantic makes an error's
    message from its ``ctx``; no kind of problem has a ``ctx`` key of that name. Nothing it carries quotes a refused
    value, and its ``input`` is always ``None``. Neither ``NodeDeclarationError`` nor ``NodeError`` is a
    ``ValidationError``: one is a developer's fault and the other a node's failure, not a refused value.

    Most calls a host makes can raise it: ``graph.run``, ``graph.with_inputs``, ``run.resume``, ``run.stream``,
    ``run.with_answers``, ``run.with_inputs``, ``run.again``, ``run.with_forgotten``, ``run.outputs``,
    ``Run.from_json``, ``Graph.compile`` (for JSON that isn't a graph record), ``read_json``, ``registry.upgrade``
    and ``make_graph``.

    Every validator raises it when it refuses a value from outside. A value that came from a graph or a run, not
    from a caller, isn't raised. Compiling puts its problems in ``graph.problems``, and a run keeps the problem as
    the outcome of the call it belongs to.

    Attributes:
        problems: What is wrong, in the order it was found.
    """

    def __new__(cls, *args: Any) -> Self:
        """Makes the exception from its problems: ``InvalidCallError(problems)``.

        Pydantic makes it too, with its own four arguments, when it copies or unpickles one. Those are passed on.
        """
        if len(args) != 1:
            return super().__new__(cls, *args)
        (problems,) = args
        return cls.from_exception_data(
            cls.__name__,
            [
                {
                    # pydantic fills an error's message in from its ctx, so a message with braces in it is
                    # kept only as the last ctx value, filled into a template that reads nothing else
                    "type": PydanticCustomError(
                        problem.type, "{msg}", {**problem.ctx, "msg": problem.msg}
                    ),
                    "loc": problem.loc,
                    "input": None,
                }
                for problem in problems
            ],
            hide_input=True,
        )

    def __init__(self, *args: Any) -> None:
        """Does nothing: ``__new__`` made the exception, and its problems are read from pydantic's errors."""

    # ConductorError's __init__ stores its problems, but pydantic's errors are what this class pickles and copies,
    # so the problems are read from them
    @property
    def problems(self) -> list[Problem]:  # type: ignore[override]
        """What is wrong, in the order it was found: each of pydantic's errors, without the ``msg`` in its ``ctx``."""
        return [
            Problem.model_construct(
                type=error["type"],
                loc=error["loc"],
                msg=error["msg"],
                ctx={key: value for key, value in error["ctx"].items() if key != "msg"},
            )
            for error in self.errors(include_url=False, include_input=False)
        ]

    def __str__(self) -> str:
        """The problems, one per line, each with its ``loc``, its code and its message."""
        problems = self.problems
        return "\n".join(
            [
                f"{len(problems)} problem(s):",
                *(
                    f"  {'.'.join(str(place) for place in problem.loc)} — {problem.type}: {problem.msg}"
                    for problem in problems
                ),
            ]
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


def author_code_problem(
    error: BaseException,
    log_message: str,
    *log_args: object,
    raised: Callable[..., Problem],
    failed: Callable[..., Problem] | None = None,
    log_level: int = logging.ERROR,
) -> Problem:
    """Returns the problem for an exception that code a developer wrote raised, and logs it if it isn't a ``NodeError``.

    Conductor calls code it didn't write: a node's function, a ``compute_...`` hook, the ``rewrite`` of an
    ``Upgrade``, and a type's own JSON writing. That code may raise anything, and conductor reports it by one rule.

    A ``NodeError`` keeps its code, message and ``ctx``, since its developer wrote them for people. With ``failed``,
    they go into that kind's ``ctx``: ``failed`` is called with ``reason`` (the message), ``error`` (the code) and
    ``error_ctx`` (the ``ctx``, or ``None`` when it is empty). Without it, the ``NodeError``'s own problem is the
    problem, as when a node's call fails.

    Any other exception might quote a value in its text. So it goes to the ``conductor`` logger with its traceback,
    and ``raised`` is called with ``error``, the exception's class name, and nothing else of it.

    Args:
        error: What the code raised.
        log_message: The log line for an exception that isn't a ``NodeError``, a ``%`` template as ``logging`` takes.
        log_args: The values for ``log_message``.
        raised: The kind of problem for an exception that isn't a ``NodeError``, with its other ``ctx`` given.
        failed: The kind of problem a ``NodeError`` is reported as, with its other ``ctx`` given.
        log_level: The level of the log line.

    Returns:
        The problem. Its ``loc`` is the one ``raised`` or ``failed`` gives it, and empty for a ``NodeError``'s own.
    """
    if isinstance(error, NodeError):
        (problem,) = error.problems
        if failed is None:
            return problem
        return failed(reason=problem.msg, error=problem.type, error_ctx=dict(problem.ctx) or None)
    _logger.log(log_level, log_message, *log_args, exc_info=error)
    return raised(error=type(error).__name__)
