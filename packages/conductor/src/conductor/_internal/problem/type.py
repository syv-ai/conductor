"""A problem, and how a package declares the kinds of problem it writes.

Everything wrong that a person has to hear about is one record, ``Problem``. A package that writes problems holds a
``ProblemCatalogue`` and declares each kind with one function under ``@catalogue.problem()``: the function's name is the
code, its signature is what the problem holds, and it returns the English. Calling the function makes the problem, so
a type checker holds every problem made to its declaration, and a host that translates by code reads the catalogue
for the codes and the keys of each one's ``ctx``.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, NotRequired, TypedDict, final

from pydantic import Field, JsonValue

from conductor._internal.conductor_model import ConductorModel


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
                              Inside a subgraph, each subgraph node and
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




@final
class ProblemKind[**Said]:
    """One kind of problem, declared by the function that says it; calling it makes the ``Problem``.

    ``ProblemCatalogue.problem`` makes one and says how the function's signature is read.

    Attributes:
        code: The problem's ``type``.
        hint: What ``ctx.hint`` says when the kind is a code said for one mistake, else ``None``.
        of_a_hook: Whether the fault is in a node's ``compute_...`` hook, which the node's developer fixes, where any
            other fault of loading is fixed in the graph.
        sentence: The English with ``{key}`` where a ``ctx`` value goes.
        ctx_type: The ``ctx`` the kind writes, as a ``TypedDict``; a key that may be left out is not required.
    """

    def __init__(self, say: Callable[Said, str], code: str, hint: str | None, of_a_hook: bool) -> None:
        self.code, self.hint, self.of_a_hook = code, hint, of_a_hook
        self.__doc__ = say.__doc__
        self._say = say
        signature = inspect.signature(say, eval_str=True)
        parameters = list(signature.parameters.values())
        for parameter in parameters:
            _refuse_outside_the_rules(say.__name__, parameter)
        if signature.return_annotation is not str:
            raise TypeError(f"{say.__name__}: 'return' is the English a person reads; annotate it -> str")
        self._places = tuple(p.name for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD)
        self._hint = {"hint": hint} if hint else {}
        self._may_leave_out = any(p.default is None for p in parameters)
        keys = [p for p in parameters if p.kind is not p.VAR_POSITIONAL]
        slots = {p.name: ["{" + p.name + "}"] if typing.get_origin(p.annotation) is list else "{" + p.name + "}" for p in keys if p.default is not None}
        self.sentence = typing.cast("Callable[..., str]", say)(**slots)
        self.ctx_type: Any = typing.cast("Any", TypedDict)(code, {  # the fields are read off a signature, which a type checker takes only as a literal
            **{p.name: NotRequired[_without_none(p.annotation)] if p.default is None else p.annotation for p in keys},
            **({"hint": Literal[hint]} if hint else {}),
        })

    def __call__(self, *loc: Said.args, **ctx: Said.kwargs) -> Problem:
        """The problem at ``loc``. What is given is conductor's own, typed where it is written, and is not read again."""
        given = {key: value for key, value in ctx.items() if value is not None} if self._may_leave_out else ctx
        return Problem.model_construct(type=self.code, loc=loc, msg=self._say(*loc, **ctx), ctx={**dict(zip(self._places, loc)), **given, **self._hint})


def _refuse_outside_the_rules(function: str, parameter: inspect.Parameter) -> None:
    """Refuses a parameter of a kind's function that is none of a place, ``*loc`` and a ``ctx`` key.

    Args:
        function: The function's name, for the message.
        parameter: One of its parameters.

    Raises:
        TypeError: The parameter fits none of the three rules ``ProblemCatalogue.problem`` states.
    """
    said = f"{function}: parameter '{parameter.name}'"
    match parameter.kind:
        case parameter.POSITIONAL_ONLY:
            raise TypeError(f"{said} is positional-only; a place in loc is a named positional parameter, so drop the /")
        case parameter.VAR_KEYWORD:
            raise TypeError(f"{said} is **{parameter.name}; a ctx key is a keyword-only parameter of its own, with its type")
        case parameter.VAR_POSITIONAL:
            if parameter.annotation != str | int:
                raise TypeError(f"{said} holds places of loc; annotate it *{parameter.name}: str | int")
        case parameter.POSITIONAL_OR_KEYWORD:
            if parameter.annotation not in (str, int):
                raise TypeError(f"{said} is a place in loc, which is a str or an int; annotate it so, or make it keyword-only for a ctx key")
            if parameter.default is not parameter.empty:
                raise TypeError(f"{said} is a place in loc and has a default; a place is always given")
        case parameter.KEYWORD_ONLY:
            if parameter.annotation is parameter.empty:
                raise TypeError(f"{said} is a ctx key without a type; annotate it")
            if parameter.default is not parameter.empty and parameter.default is not None:
                raise TypeError(f"{said} defaults to {parameter.default!r}; a ctx key is always given, or defaults to None and may be left out")


def _without_none(annotation: Any) -> Any:
    """``X | None`` as ``X``: a key that may be left out is absent, never ``None``."""
    return typing.Union[tuple(option for option in typing.get_args(annotation) if option is not type(None))]


@final
class ProblemCatalogue:
    """The kinds of problem one package writes, in the order they are declared.

    A host that translates by code iterates it: each kind has its ``code``, its ``sentence`` and the type of its ``ctx``.
    """

    def __init__(self) -> None:
        self._kinds: list[ProblemKind[...]] = []

    def problem[**Said](self, code: str | None = None, *, hint: str | None = None, of_a_hook: bool = False) -> Callable[[Callable[Said, str]], ProblemKind[Said]]:
        """Declares the function under it a kind of problem, and gives back what a caller calls to make one.

        The function returns the English a person reads, built from its parameters and nothing else. Each parameter
        is one of three things:

        - **A keyword-only parameter is a key of** ``ctx``, with its type. One that defaults to ``None`` may be left
          out, and is then absent from ``ctx``; no other default is taken::

              def timeout(*loc: str | int, seconds: float) -> str: ...
              timeout("fetch", 2, seconds=30.0)                    # ctx == {"seconds": 30.0}

        - **A named positional parameter is a place in** ``loc`` **and a key of** ``ctx``, a ``str`` or an ``int``,
          so the two can't disagree::

              def unknown_input(node_id: str, field: str, *, inputs: list[str]) -> str: ...
              unknown_input("upper", "txet", inputs=["text"])      # loc == ("upper", "txet"), ctx has node_id and field

        - ``*loc: str | int`` **is a place the sentence doesn't read**; it is in ``loc`` alone::

              def cycle(*loc: str | int) -> str: ...
              cycle("a")                                           # loc == ("a",), ctx == {}

        Args:
            code: The problem's ``type``; the function's name when left out.
            hint: What ``ctx.hint`` says, for a second sentence under a code the catalogue has declared.
            of_a_hook: Whether the fault is in a node's ``compute_...`` hook.

        Raises:
            TypeError: When the function is declared, for a parameter that is none of the three, a return type other
                than ``str``, or a ``hint`` for a code the catalogue has not declared.
        """
        def declared(say: Callable[Said, str]) -> ProblemKind[Said]:
            kind = ProblemKind(say, code or say.__name__, hint, of_a_hook)
            if hint and kind.code not in self.codes:
                raise TypeError(f"{say.__name__}: a hint is a second sentence under a code, and '{kind.code}' is not declared before it")
            self._kinds.append(kind)
            return kind

        return declared

    def __iter__(self) -> Iterator[ProblemKind[...]]:
        return iter(self._kinds)

    @property
    def codes(self) -> frozenset[str]:
        """Every code the catalogue's kinds write."""
        return frozenset(kind.code for kind in self._kinds)


def listed(items: Sequence[object]) -> str:
    """Items as a sentence writes them: joined by commas, ``none`` when there are none."""
    return ", ".join(str(item) for item in items) or "none"
