"""A problem, and how a package declares the kinds of problem it reports.

Everything wrong that a person needs to hear about is one record, a ``Problem``, with the four fields of a pydantic
error: ``type`` (a code), ``loc``, ``msg`` and ``ctx``. A package declares each kind of problem it reports as a
function, and keeps them in a ``ProblemCatalogue``::

    PROBLEMS = ProblemCatalogue()

    @PROBLEMS.problem()
    def unknown_input(node_id: str, field: str, /, *, inputs: list[str]) -> str:
        return f"'{node_id}' has no input '{field}'; it has {comma_separated(inputs)}."

    unknown_input("upper", "txet", inputs=["text"])
    # Problem(type="unknown_input", loc=("upper", "txet"), msg="'upper' has no input 'txet'; it has text.",
    #         ctx={"node_id": "upper", "field": "txet", "inputs": ["text"]})

The function's name is the code, its parameters are ``loc`` and ``ctx``, and it returns the message. A type checker
checks every call against the function's signature, and a host that writes its own messages iterates the catalogue
for each code and the keys of its ``ctx``.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Iterator, Mapping
from inspect import Parameter
from typing import Any, Literal, NotRequired, TypedDict, Union, final

from pydantic import Field, JsonValue

from conductor._internal.base import ConductorModel


@final
class Problem(ConductorModel):
    """One thing that is wrong, with the four fields of a pydantic error::

        Problem(type="unknown_input", loc=("upper", "txet"), msg="'upper' has no input 'txet'; it has text.",
                ctx={"node_id": "upper", "field": "txet", "inputs": ["text"]})

    ``loc`` says where, and what it names depends on where the problem is reported:

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

    The input or output in a graph problem's ``loc`` may not exist: an edge to an output the node does not have is
    reported at that name.

    Attributes:
        type: The code, which does not change between versions; a host translates by it.
        loc: Where the problem is.
        msg: The message, in English, for a person; never parsed.
        ctx: The values the message is made from, by name, so a host can write its own message.
    """

    type: str
    loc: tuple[str | int, ...] = ()
    msg: str
    ctx: Mapping[str, JsonValue] = Field(default_factory=dict)




@final
class ProblemKind[**Params]:
    """One kind of problem: the function that writes its message, and what is read from that function's signature.

    ``@catalogue.problem()`` makes one from a function; calling it makes a ``Problem``. For ``unknown_input`` above:
    ``code`` is ``"unknown_input"``, ``sentence`` is ``"'{node_id}' has no input '{field}'; it has {inputs}."``, and
    ``ctx_type`` is a ``TypedDict`` with the keys ``node_id``, ``field`` and ``inputs``.

    Attributes:
        code: The problem's ``type``.
        variant: For a second message under a code, for one particular mistake: the value of ``ctx.variant``.
            ``None`` for a code's main message.
        sentence: The message with a replacement field ``{key}`` in place of each ``ctx`` value.
        ctx_type: The keys of ``ctx`` and their types, as a ``TypedDict``. A key that may be left out is not required.
        section: The title of the catalogue the kind is declared on; empty when it has none.
    """

    def __init__(self, write_message: Callable[Params, str], code: str, variant: str | None, section: str) -> None:
        self.code, self.variant, self.section = code, variant, section
        self.__doc__ = write_message.__doc__
        self._message = write_message
        signature = inspect.signature(write_message, eval_str=True)
        parameters = list(signature.parameters.values())
        for parameter in parameters:
            _check_parameter(write_message.__name__, parameter)
        if signature.return_annotation is not str:
            raise TypeError(f"{write_message.__name__}: 'return' is the message a person reads; annotate it -> str")
        self._loc_parameters = tuple(p.name for p in parameters if p.kind is p.POSITIONAL_ONLY)
        self._variant_ctx = {"variant": variant} if variant else {}
        self._has_optional_ctx = any(p.default is None for p in parameters)
        keys = [p for p in parameters if p.kind is not p.VAR_POSITIONAL]
        # The message with a replacement field "{key}" for each ctx value. A key that may be left out is not passed, so the function writes
        # the message it writes when the key is absent.
        replacement_fields = {p.name: ["{" + p.name + "}"] if typing.get_origin(p.annotation) is list else "{" + p.name + "}" for p in keys if p.default is not None}
        loc_replacement_fields = [replacement_fields.pop(name) for name in self._loc_parameters]
        self.sentence = typing.cast("Callable[..., str]", write_message)(*loc_replacement_fields, **replacement_fields)
        self.ctx_type: Any = typing.cast("Any", TypedDict)(code, {  # fields built from the signature, which a type checker cannot follow
            **{p.name: NotRequired[_without_none(p.annotation)] if p.default is None else p.annotation for p in keys},
            **({"variant": Literal[variant]} if variant else {}),
        })

    def __call__(self, *loc: Params.args, **ctx: Params.kwargs) -> Problem:
        """Makes the problem at ``loc``. The values are not validated: every call is checked by a type checker."""
        given = {key: value for key, value in ctx.items() if value is not None} if self._has_optional_ctx else ctx
        return Problem.model_construct(type=self.code, loc=loc, msg=self._message(*loc, **ctx), ctx={**dict(zip(self._loc_parameters, loc)), **given, **self._variant_ctx})


def _check_parameter(function: str, parameter: Parameter) -> None:
    """Checks that a parameter of a kind's function is one of the three ``ProblemCatalogue.problem`` allows.

    Args:
        function: The function's name, for the message.
        parameter: One of its parameters.

    Raises:
        TypeError: The parameter is none of the three.
    """
    prefix = f"{function}: parameter '{parameter.name}'"
    match parameter.kind:
        case parameter.POSITIONAL_OR_KEYWORD:
            raise TypeError(f"{prefix} can be given by name; a part of loc is positional-only, so put a / after it")
        case parameter.VAR_KEYWORD:
            raise TypeError(f"{prefix} is **{parameter.name}; a ctx key is a keyword-only parameter of its own, with its type")
        case parameter.VAR_POSITIONAL:
            if parameter.annotation != str | int:
                raise TypeError(f"{prefix} is the parts of loc; annotate it *{parameter.name}: str | int")
        case parameter.POSITIONAL_ONLY:
            if parameter.annotation not in (str, int):
                raise TypeError(f"{prefix} is a part of loc, which is a str or an int; annotate it so, or make it keyword-only for a ctx key")
            if parameter.default is not parameter.empty:
                raise TypeError(f"{prefix} is a part of loc and has a default; remove the default, since a part of loc is always given")
        case parameter.KEYWORD_ONLY:
            if parameter.annotation is parameter.empty:
                raise TypeError(f"{prefix} is a ctx key without a type; annotate it")
            if parameter.default is not parameter.empty and parameter.default is not None:
                raise TypeError(f"{prefix} defaults to {parameter.default!r}; a ctx key is always given, or defaults to None and may be left out")


def _without_none(annotation: Any) -> Any:
    """``X | None`` as ``X``: a ``ctx`` key that may be left out is then absent, never ``None``."""
    return Union[tuple(option for option in typing.get_args(annotation) if option is not type(None))]


@final
class ProblemCatalogue:
    """The kinds of problem one package reports, in the order they are declared.

    A host that writes its own messages iterates it: each kind has its ``code``, its ``sentence`` and its
    ``ctx_type``. A catalogue with a title is a section of the problems page, which prints the title as a heading
    and the text under it, then the kinds; ``ProblemCatalogue.joined`` makes one catalogue of several sections.

    Attributes:
        sections: The text of each section, by title, in the order of its kinds; empty for a catalogue with no title.
    """

    def __init__(self, title: str = "", text: str = "") -> None:
        """Makes an empty catalogue.

        Args:
            title: The heading of its section on the problems page; empty for a catalogue printed under a heading
                of the page's own.
            text: What its kinds have in common: where they are met, and what their ``loc`` is.
        """
        self._title = title
        self._kinds: dict[tuple[str, str | None], ProblemKind[...]] = {}
        self.sections: dict[str, str] = {title: text} if title else {}

    @classmethod
    def joined(cls, *sections: ProblemCatalogue) -> ProblemCatalogue:
        """One catalogue of the kinds of ``sections``, in their order, each kind in the section it is declared in.

        Raises:
            TypeError: Two sections declare a kind with one code and variant.
        """
        catalogue = cls()
        for section in sections:
            catalogue.sections |= section.sections
            for kind in section:
                catalogue._add(kind)
        return catalogue

    def problem[**Params](self, code: str | None = None, *, variant: str | None = None) -> Callable[[Callable[Params, str]], ProblemKind[Params]]:
        """Makes the function under it a kind of problem, and returns the kind, which is called to make a problem.

        The function returns the message, made from its parameters and nothing else. Each parameter is one of three
        things:

        - **A keyword-only parameter is a key of** ``ctx``, with its type. One that defaults to ``None`` may be left
          out, and is then absent from ``ctx``; no other default is allowed::

              def timeout(*loc: str | int, seconds: float) -> str: ...
              timeout("fetch", 2, seconds=30.0)                    # ctx == {"seconds": 30.0}

        - **A positional-only parameter is a part of** ``loc`` **and a key of** ``ctx``, a ``str`` or an ``int``.
          It cannot be given by name, so the two always agree::

              def unknown_input(node_id: str, field: str, /, *, inputs: list[str]) -> str: ...
              unknown_input("upper", "txet", inputs=["text"])      # loc == ("upper", "txet"), ctx has node_id and field

        - ``*loc: str | int`` **is the parts of** ``loc`` **the message does not use**; they are in ``loc`` only::

              def cycle(*loc: str | int) -> str: ...
              cycle("a")                                           # loc == ("a",), ctx == {}

        A code has one kind with no variant, declared first, and every other kind under it names the code and a
        variant of its own: ``@catalogue.problem("type_mismatch", variant="make_a_series")``.

        Args:
            code: The problem's ``type``; the function's name when left out.
            variant: For a second message under a code declared before it: the value of ``ctx.variant``.

        Raises:
            TypeError: When the function is declared, for a parameter that is none of the three, a return type other
                than ``str``, a second kind under a code with no ``variant`` or with a variant declared before it, or
                a ``variant`` for a code not declared before it.
        """
        def declared(write_message: Callable[Params, str]) -> ProblemKind[Params]:
            kind = ProblemKind(write_message, code or write_message.__name__, variant, self._title)
            self._add(kind)
            return kind

        return declared

    def _add(self, kind: ProblemKind[...]) -> None:
        """Adds ``kind``, the one kind of its code with no variant or a variant the code has no other kind of.

        Raises:
            TypeError: The catalogue has a kind of the code and variant, or ``kind`` is a variant of a code it has no
                kind of.
        """
        name = kind._message.__name__
        if (kind.code, kind.variant) in self._kinds:
            taken = f"the variant {kind.variant!r}" if kind.variant else "a kind"
            raise TypeError(f"{name}: '{kind.code}' has {taken} declared before it; give this kind a variant of its own")
        if kind.variant and (kind.code, None) not in self._kinds:
            raise TypeError(f"{name}: a variant is a second message under a code, and '{kind.code}' is not declared before it")
        self._kinds[kind.code, kind.variant] = kind

    def __iter__(self) -> Iterator[ProblemKind[...]]:
        return iter(self._kinds.values())

    @property
    def codes(self) -> frozenset[str]:
        """Every code of the catalogue's kinds."""
        return frozenset(code for code, _ in self._kinds)

