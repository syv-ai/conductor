"""``Problem``, the record of one thing that is wrong, and how a package declares its kinds of problem.

A problem looks like one of pydantic's errors. It has the same four fields: ``type`` (a code), ``loc`` (where),
``msg`` (the message) and ``ctx`` (the values the message was made from).

A package declares each kind of problem it reports as a function. The function's name is the code, its parameters
are the ``loc`` and the ``ctx``, and it returns the message. The package keeps its kinds in a ``ProblemCatalogue``,
and calling a kind makes a ``Problem``::

    PROBLEMS = ProblemCatalogue()

    @PROBLEMS.problem
    def unknown_input(node_id: str, input: str, /, *, inputs: list[str]) -> str:
        return f"'{node_id}' has no input '{input}'. Its inputs are {comma_separated(inputs)}."

    unknown_input("upper", "txet", inputs=["text"])
    # Problem(type="unknown_input", loc=("upper", "txet"), msg="'upper' has no input 'txet'. Its inputs are text.",
    #         ctx={"node_id": "upper", "input": "txet", "inputs": ["text"]})

Since each kind is a plain function, a type checker checks every call to it. A host that writes its own messages
goes through the catalogue to find each code and the keys of its ``ctx``.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Iterator, Mapping
from inspect import Parameter
from typing import Any, NotRequired, TypedDict, Union, final

from pydantic import Field, JsonValue

from conductor._internal.base import ConductorModel


@final
class Problem(ConductorModel):
    """One thing that is wrong, to be shown to a person.

    It has the same four fields as a pydantic error. Here is one::

        Problem(type="unknown_input", loc=("upper", "txet"), msg="'upper' has no input 'txet'. Its inputs are text.",
                ctx={"node_id": "upper", "input": "txet", "inputs": ["text"]})

    ``loc`` says where the problem is. What it names depends on where the problem was found:

    ==========================  ==============================================================================
    a problem of a graph        ``(node_id,)`` or ``(node_id, input)``. For a node inside a subgraph, the
                                subgraph node's id comes first: ``("loud", "shout", "line")``
    a call that failed          ``(node_id, *item_indexes)``: ``("shout", 1)`` is the call of ``shout`` for
                                the second item. Inside a subgraph, the subgraph node and its item indexes
                                come first: ``("loud", 1, "shout")``
    a run that stopped          ``()``, when no one call stopped it (a timeout, a cancel)
    a refused call              the argument the call got wrong: ``("inputs", name)`` for a graph input,
                                ``("saved_run",)`` for a saved run, ``("record", *steps)`` for JSON given as a
                                graph record and the steps to the fault in it, or the node and item indexes of
                                the call an answer (``run.answer``) is for
    ==========================  ==============================================================================

    The input or output named in a graph problem's ``loc`` may not exist. An edge to an output the node doesn't have
    is reported at that output's name.

    Attributes:
        type: The code, such as ``"unknown_input"``. It stays the same between versions, so a host can translate by
            it.
        loc: Where the problem is.
        msg: The message for a person, in English. Show it, but don't parse it.
        ctx: The values the message was made from, by name, so a host can write its own message.
    """

    type: str
    loc: tuple[str | int, ...] = ()
    msg: str
    ctx: Mapping[str, JsonValue] = Field(default_factory=dict)


@final
class ProblemKind[**Params]:
    """One kind of problem, such as ``unknown_input``.

    It wraps the function that writes the message, and calling it makes a ``Problem``. ``@catalogue.problem``
    builds one from a function, and works out a few things from the function's signature. For ``unknown_input``
    above, ``code`` is ``"unknown_input"``, ``sentence`` is ``"'{node_id}' has no input '{input}'. Its inputs
    are {inputs}."``, and ``ctx_type`` is a ``TypedDict`` with the keys ``node_id``, ``input`` and ``inputs``.

    Attributes:
        code: The problem's ``type``: the function's name.
        sentence: The message with a placeholder such as ``{node_id}`` where each ``ctx`` value goes. A host can use
            it as a template.
        ctx_type: The keys of ``ctx`` and their types, as a ``TypedDict``. A key that may be left out is not
            required.
        section: The title of the catalogue the kind was declared on, or empty if it has none.
    """

    def __init__(self, write_message: Callable[Params, str], section: str) -> None:
        self.code, self.section = write_message.__name__, section
        self.__doc__ = write_message.__doc__
        self._message = write_message
        signature = inspect.signature(write_message, eval_str=True)
        parameters = list(signature.parameters.values())
        for parameter in parameters:
            _check_parameter(write_message.__name__, parameter)
        if signature.return_annotation is not str:
            raise TypeError(
                f"{write_message.__name__}: 'return' is the message a person reads; annotate it -> str"
            )
        self._loc_parameters = tuple(p.name for p in parameters if p.kind is p.POSITIONAL_ONLY)
        self._has_optional_ctx = any(p.default is None for p in parameters)
        keys = [p for p in parameters if p.kind is not p.VAR_POSITIONAL]
        # Build the sentence by calling the function with a placeholder "{key}" for each ctx value. A key that may be
        # left out isn't passed, so the sentence is the message written when that key is missing.
        replacement_fields = {
            p.name: ["{" + p.name + "}"]
            if typing.get_origin(p.annotation) is list
            else "{" + p.name + "}"
            for p in keys
            if p.default is not None
        }
        loc_replacement_fields = [replacement_fields.pop(name) for name in self._loc_parameters]
        self.sentence = typing.cast("Callable[..., str]", write_message)(
            *loc_replacement_fields, **replacement_fields
        )
        self.ctx_type: Any = typing.cast("Any", TypedDict)(
            self.code,
            {  # the fields come from the signature, which a type checker can't follow
                p.name: NotRequired[_without_none(p.annotation)]
                if p.default is None
                else p.annotation
                for p in keys
            },
        )

    def __call__(self, *loc: Params.args, **ctx: Params.kwargs) -> Problem:
        """Makes the problem, with its ``loc`` and ``ctx`` taken from the arguments.

        The values aren't validated here. A type checker already checks each call against the function's signature.
        """
        given = (
            {key: value for key, value in ctx.items() if value is not None}
            if self._has_optional_ctx
            else ctx
        )
        return Problem.model_construct(
            type=self.code,
            loc=loc,
            msg=self._message(*loc, **ctx),
            ctx={**dict(zip(self._loc_parameters, loc)), **given},
        )


def _check_parameter(function: str, parameter: Parameter) -> None:
    """Checks that one parameter of a kind's function is of a sort ``ProblemCatalogue.problem`` allows.

    There are three sorts: a part of ``loc``, the rest of ``loc`` as ``*loc``, and a ``ctx`` key.
    ``ProblemCatalogue.problem`` explains each one.

    Args:
        function: The function's name, for the message.
        parameter: One of its parameters.

    Raises:
        TypeError: The parameter is none of the three, or its type or default is wrong for its sort.
    """
    prefix = f"{function}: parameter '{parameter.name}'"
    match parameter.kind:
        case parameter.POSITIONAL_OR_KEYWORD:
            raise TypeError(
                f"{prefix} can be given by name; a part of loc is positional-only, so put a / after it"
            )
        case parameter.VAR_KEYWORD:
            raise TypeError(
                f"{prefix} is **{parameter.name}; a ctx key is a keyword-only parameter of its own, with its type"
            )
        case parameter.VAR_POSITIONAL:
            if parameter.annotation != str | int:
                raise TypeError(
                    f"{prefix} is the parts of loc; annotate it *{parameter.name}: str | int"
                )
        case parameter.POSITIONAL_ONLY:
            if parameter.annotation not in (str, int):
                raise TypeError(
                    f"{prefix} is a part of loc, which is a str or an int; annotate it so, or make it keyword-only for a ctx key"
                )
            if parameter.default is not parameter.empty:
                raise TypeError(
                    f"{prefix} is a part of loc and has a default; remove the default, since a part of loc is always given"
                )
        case parameter.KEYWORD_ONLY:
            if parameter.annotation is parameter.empty:
                raise TypeError(f"{prefix} is a ctx key without a type; annotate it")
            if parameter.default is not parameter.empty and parameter.default is not None:
                raise TypeError(
                    f"{prefix} defaults to {parameter.default!r}; a ctx key is always given, or defaults to None and may be left out"
                )


def _without_none(annotation: Any) -> Any:
    """Turns ``X | None`` into ``X``.

    A ``ctx`` key that may be left out is missing from ``ctx`` when it isn't given. It is never ``None``, so its type
    in ``ctx_type`` leaves ``None`` out.
    """
    return Union[
        tuple(option for option in typing.get_args(annotation) if option is not type(None))
    ]


@final
class ProblemCatalogue:
    """The kinds of problem one package reports, in the order they were declared.

    A host that writes its own messages goes through it. Each kind has its ``code``, its ``sentence`` and its
    ``ctx_type``.

    A catalogue can have a title. Then it is a section of the problems page: the page prints the title as a heading,
    then the section's text, then the kinds. ``ProblemCatalogue.joined`` makes one catalogue from several sections.

    Attributes:
        sections: The text of each section, by title, in the order of the kinds. Empty for a catalogue with no
            title.
    """

    def __init__(self, title: str = "", text: str = "") -> None:
        """Makes an empty catalogue.

        Args:
            title: The heading of its section on the problems page. Leave it empty for a catalogue the page prints
                under a heading of its own.
            text: What the kinds have in common: where a host meets them, and what their ``loc`` is.
        """
        self._title = title
        self._kinds: dict[str, ProblemKind[...]] = {}
        self.sections: dict[str, str] = {title: text} if title else {}

    @classmethod
    def joined(cls, *sections: ProblemCatalogue) -> ProblemCatalogue:
        """Makes one catalogue from several sections.

        The kinds keep their order, and each stays in the section it was declared in.

        Raises:
            TypeError: Two sections declare a kind with the same code.
        """
        catalogue = cls()
        for section in sections:
            catalogue.sections |= section.sections
            for kind in section:
                catalogue._add(kind)
        return catalogue

    def problem[**Params](self, write_message: Callable[Params, str]) -> ProblemKind[Params]:
        """Turns the function under it into a kind of problem, whose code is the function's name.

        Use it as a decorator. It returns a ``ProblemKind``, and calling that makes a ``Problem``.

        The function returns the message, made from its parameters and nothing else. Each parameter is one of three
        things:

        - **A keyword-only parameter is a key of** ``ctx``, with its type. If it defaults to ``None``, it may be left
          out, and is then missing from ``ctx``. No other default is allowed::

              def timeout(*loc: str | int, seconds: float) -> str: ...
              timeout("fetch", 2, seconds=30.0)                    # ctx == {"seconds": 30.0}

        - **A positional-only parameter is a part of** ``loc`` **and also a key of** ``ctx``. It is a ``str`` or an
          ``int``. It can't be given by name, so ``loc`` and ``ctx`` always agree::

              def unknown_input(node_id: str, input: str, /, *, inputs: list[str]) -> str: ...
              unknown_input("upper", "txet", inputs=["text"])      # loc == ("upper", "txet"), ctx has node_id and input

        - ``*loc: str | int`` **is the rest of** ``loc``, the parts the message doesn't use. They are in ``loc``
          only::

              def cycle(*loc: str | int) -> str: ...
              cycle("a")                                           # loc == ("a",), ctx == {}

        Each mistake a host may want to tell apart is a kind of its own, with its own code. Kinds that belong
        together share the start of their code, like ``type_mismatch`` and ``type_mismatch_make_a_series``.

        Raises:
            TypeError: When the function is declared, if a parameter is none of the three, if the return type isn't
                ``str``, or if the catalogue already has a kind with the function's name.
        """
        kind = ProblemKind(write_message, self._title)
        self._add(kind)
        return kind

    def _add(self, kind: ProblemKind[...]) -> None:
        """Adds one kind to the catalogue.

        Raises:
            TypeError: The catalogue already has a kind with this code.
        """
        if kind.code in self._kinds:
            raise TypeError(
                f"{kind.code}: a kind with this code is declared before it; give the function another name"
            )
        self._kinds[kind.code] = kind

    def __iter__(self) -> Iterator[ProblemKind[...]]:
        """Goes through the kinds in the order they were declared."""
        return iter(self._kinds.values())

    @property
    def codes(self) -> frozenset[str]:
        """Every code in the catalogue."""
        return frozenset(self._kinds)
