"""What a function's signature declares, read once.

A node declares its inputs and outputs by annotating a function::

    @node("translate")
    def translate(
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        language: Annotated[Text, Param(widget=Dropdown(choices=...))] = Text("en"),
    ) -> Annotated[Text, Result(title="Translation")]:
        ...

``read_signature`` walks that signature once and produces the input
records, the output records, what the caller must provide and how a
return is split across the outputs. Nothing else reads a signature, so
there is one place the meaning of an annotation is decided, and it is here.

What it understands:

* a ``DType``, a plain type (``str``, a pydantic model) or ``Any`` on a
  parameter, with at most one ``Param`` beside it;
* a type no edge can carry (``list[Branch]``), which closes the input;
* ``FromRun[Clock]``: not an input, a value the run supplies by type;
* ``**x: T``: every edge an input declared ``T``, unless a
  ``compute_inputs`` hook says which inputs land there. ``AsItArrives``
  never runs the node per row, ``Series[Any]`` reduces, ``Any`` or a type
  runs it per row;
* on the return: a type (one output), a frozen dataclass (one output
  per field), ``Mapping`` (the node's ``compute_outputs`` says which),
  ``None`` (no outputs), and ``X | AskPerson`` read as ``X``.

Anything else inside ``Annotated`` is refused where it is written.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import NoneType, UnionType
from typing import Annotated, Any, TypedDict, TypeVar, Union, get_args, get_origin, get_type_hints

from pydantic import ConfigDict, TypeAdapter, with_config

from conductor._internal.dtype import (
    DType,
    _AsItArrivesMark,
    _ReceivedAsItArrives,
    dtype_of,
    read_blank,
)
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.field_records import Input, Output, Param, Result
from conductor._internal.skip_and_ask import AskPerson
from conductor.widgets import Widget

_T = TypeVar("_T")

RESULT_KEY = "result"


@dataclass(frozen=True)
class _FromRunMark:
    """What makes ``FromRun[T]`` a value the run supplies."""


#: A parameter the caller of the graph supplies, not the graph:
#: ``clock: FromRun[Clock]`` is no input (no widget, no edge, no binding) but
#: a value the host hands to ``compiled.run(from_run={Clock: ...})``, which
#: the engine passes in by type. A run will not start a graph whose nodes
#: need a type it was not given.
FromRun = Annotated[_T, _FromRunMark()]


@dataclass(frozen=True)
class Interface:
    """A node's inputs, outputs and needs.

    Three things carry one. What one version declares; what one node in a
    graph actually has once the compiler has asked its hooks and typed its
    edges (``CompiledGraph.node(node_id).interface``); and what a whole
    graph takes and returns (``CompiledGraph.interface``), with inputs
    named by address. Derived, never written by hand; frozen, so the
    derivation is the only writer.
    """

    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    #: The types the caller must provide, from every ``FromRun`` parameter.
    needs: frozenset[type] = frozenset()


@dataclass(frozen=True)
class ReadSignature:
    """What ``read_signature`` found: the interface, and how the function's return and ``**x`` work."""

    interface: Interface
    #: The ``FromRun`` parameters: the value of each type goes under this name in a call.
    from_run: dict[str, type]
    #: How a return splits across the outputs: a ``DType``, a record class,
    #: ``Mapping``, or ``None`` for a function that declares no outputs.
    returns: Any
    #: The name and hint of a ``**x`` parameter, ``None`` when it has none.
    var_keyword: tuple[str, Any] | None


def read_signature(function: Callable[..., Any]) -> ReadSignature:
    """Read ``function``'s signature. Every fault is a ``NodeDeclarationError``, at import."""
    hints = get_type_hints(function, include_extras=True)
    inputs: list[Input] = []
    needs: dict[str, type] = {}
    var_keyword: tuple[str, Any] | None = None
    for name, parameter in inspect.signature(function).parameters.items():
        hint = hints.get(name, parameter.annotation)
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            raise NodeDeclarationError(
                f"parameter *{name}: a node's inputs are named, so an edge or a binding can land on each; *args names none",
                code="var_positional_parameter",
            )
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword = (name, Any if hint is inspect.Parameter.empty else hint)
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise NodeDeclarationError(f"parameter {name!r} is positional-only; the engine passes every input by name", code="positional_only_parameter")
        if name.startswith("_"):
            raise NodeDeclarationError(f"parameter {name!r}: an input's name cannot start with an underscore", code="parameter_name_private")
        declared, extras = _unwrapped(hint)
        if any(isinstance(extra, _FromRunMark) for extra in extras):
            needs[name] = declared
            continue
        inputs.append(_input(name, declared, extras, parameter))
    returns, outputs = _read_return(hints)
    taken = {i.name for i in inputs} & {o.name for o in outputs}
    if taken:
        name = sorted(taken)[0]
        raise NodeDeclarationError(
            f"the parameter {name!r} and the output {name!r} share a name, and a field name is unique within a node; "
            f"rename the parameter, or name the output with Result(name=...)",
            code="field_on_both_sides",
        )
    return ReadSignature(Interface(inputs=tuple(inputs), outputs=outputs, needs=frozenset(needs.values())), needs, returns, var_keyword)


def open_input_of(name: str, hint: Any) -> Input:
    """The record every edge into ``**name: hint`` is an input like: named on the edge when compile makes one."""
    declared, extras = _unwrapped(hint)
    if any(isinstance(extra, _AsItArrivesMark) for extra in extras):
        return Input("", _ReceivedAsItArrives)
    dtype = dtype_of(declared)
    if dtype is None or dtype is DType:
        raise NodeDeclarationError(
            f"**{name}: {declared!r} is no type an edge carries; each edge is an input, declared a type, Any or AsItArrives",
            code="open_interface_annotation",
        )
    return Input("", dtype)


def _input(name: str, declared: Any, extras: tuple[Any, ...], parameter: inspect.Parameter) -> Input:
    """The ``Input`` of one named parameter, its ``Annotated`` extras checked."""
    for extra in extras:
        if isinstance(extra, Param):
            continue
        if isinstance(extra, _AsItArrivesMark):
            raise NodeDeclarationError(
                f"parameter {name!r}: AsItArrives is spelled on **inputs only; a named parameter declares a DType, or Any for whatever arrives",
                code="as_it_arrives_on_named_parameter",
            )
        if isinstance(extra, Widget):
            raise NodeDeclarationError(
                f"parameter {name!r} carries a bare widget; the widget goes on the Param — Annotated[Text, Param(title=..., widget=Textarea())]",
                code="widget_outside_param",
            )
        if isinstance(extra, Result):
            raise NodeDeclarationError(f"parameter {name!r} carries a Result; a parameter carries a Param, and Result belongs on the return", code="result_on_parameter")
        raise NodeDeclarationError(f"parameter {name!r} carries {extra!r}, which conductor does not read", code="annotated_extra_unknown")
    if declared is inspect.Parameter.empty:
        raise NodeDeclarationError(f"parameter {name!r} has no type; it must declare a type (str, a pydantic model, a DType) or Any", code="parameter_needs_type")
    if declared is DType:
        raise NodeDeclarationError(f"parameter {name!r} must declare a concrete DType, or Any for whatever arrives; the base would accept anything", code="parameter_base_dtype")
    param = next((extra for extra in extras if isinstance(extra, Param)), Param())
    has_default = parameter.default is not inspect.Parameter.empty
    return Input(
        name, declared, title=param.title, description=param.description, widget=param.widget, connectable=param.connectable,
        **({"default": parameter.default} if has_default else {}),
    )


def _read_return(hints: Mapping[str, Any]) -> tuple[Any, tuple[Output, ...]]:
    """The declared return's split and the outputs it declares."""
    if "return" not in hints:
        raise NodeDeclarationError("a node's function must declare a return type; -> None declares no outputs", code="missing_return_type")
    hint = _without_ask_person(hints["return"])
    if hint is None or hint is NoneType:
        return None, ()
    declared, extras = _unwrapped(hint)
    if any(isinstance(extra, Param) for extra in extras):
        raise NodeDeclarationError("the return carries a Param; a return carries a Result, and Param belongs on a parameter", code="param_on_return")
    bare, blank = read_blank(declared)
    if (dtype := dtype_of(bare)) is not None:
        return dtype, (_output(RESULT_KEY, dtype | None if blank else bare, _result_in(extras)),)
    if blank:
        raise NodeDeclarationError(f"{bare!r} can't take a blank: only one value, or a field of a returned record, is blank", code="blank_on_a_record")
    if isinstance(declared, type) and is_dataclass(declared):
        return declared, _record_outputs(declared)
    if declared is Mapping or get_origin(declared) is Mapping:
        return Mapping, ()
    raise NodeDeclarationError(f"a node returns a type an edge can carry (str, a pydantic model, a DType), Series[X] for many of them, a frozen dataclass of them, Mapping[str, Any] or None — got {declared!r}", code="unsupported_return_type")


def _record_outputs(record: type) -> tuple[Output, ...]:
    """One output per field of a returned dataclass."""
    hints = get_type_hints(record, include_extras=True)
    outputs = []
    for each in fields(record):
        declared, extras = _unwrapped(hints[each.name])
        if any(isinstance(extra, Param) for extra in extras):
            raise NodeDeclarationError(f"{record.__name__}.{each.name} carries a Param; a record's field carries a Result", code="param_on_record_field")
        bare, blank = read_blank(declared)
        result = _result_in(extras)
        if dtype_of(bare) is None:
            raise NodeDeclarationError(f"{record.__name__}.{each.name} must be a type an edge can carry — got {declared!r}", code="record_field_needs_dtype")
        dtype = declared
        if result.name is not None:
            raise NodeDeclarationError(f"{record.__name__}.{each.name}: a record's field is named by the field, so Result(name=...) has nothing to name", code="result_name_on_record_field")
        outputs.append(_output(each.name, dtype, result))
    return tuple(outputs)


def _output(default_name: str, dtype: Any, result: Result) -> Output:
    return Output(result.name or default_name, dtype, title=result.title, description=result.description, choice=result.choice)


def _result_in(extras: tuple[Any, ...]) -> Result:
    return next((extra for extra in extras if isinstance(extra, Result)), Result())


def _without_ask_person(hint: Any) -> Any:
    """``X | AskPerson`` is read as ``X``: a node that pauses to ask a person declares no output for it."""
    if get_origin(hint) in (Union, UnionType) and AskPerson in get_args(hint):
        members = tuple(member for member in get_args(hint) if member is not AskPerson)
        return Union[members]  # noqa: UP007  (the members are runtime values)
    return hint


def _unwrapped(hint: Any) -> tuple[Any, tuple[Any, ...]]:
    """The type under ``Annotated[...]`` with its ``| None`` kept beside it, and what the ``Annotated`` holds.

    ``Annotated[Text, Param()] | None`` and ``Annotated[Text | None, Param()]`` read alike."""
    written, blank = read_blank(hint)
    if get_origin(written) is Annotated:
        declared, *extras = get_args(written)
        return (declared | None if blank else declared), tuple(extras)
    return hint, ()


def unpack(returns: Any, value: Any, outputs: tuple[Output, ...], declared: tuple[Output, ...]) -> dict[str, Any]:
    """Split what a node returned into ``{output name: value}``.

    ``returns`` is ``ReadSignature.returns``, ``outputs`` what the node
    has in its graph and ``declared`` what its signature declared. A value
    of the wrong shape is a ``ValueError`` that never quotes the value.
    """
    if returns is None:
        if value is not None:
            raise ValueError(f"the node declares no outputs and returned a {type(value).__name__}")
        return {}
    if isinstance(returns, type) and issubclass(returns, DType):
        return {declared[0].name: value}
    if returns is Mapping:
        names = {output.name for output in outputs}
        if not isinstance(value, Mapping) or set(value) != names:
            given = sorted(value) if isinstance(value, Mapping) else type(value).__name__
            raise ValueError(f"the node must return exactly the outputs its interface names, {sorted(names)}, not {given}")
        return dict(value)
    if not isinstance(value, returns):
        raise ValueError(f"the node must return a {returns.__name__}, not a {type(value).__name__}")
    return {each.name: getattr(value, each.name) for each in fields(returns)}


def build_call_validator(inputs: tuple[Input, ...]) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """The check a call of ``inputs`` goes through: values in, keyword arguments out.

    The function it returns takes the values a call was given, by input
    name, and returns one value per input as its type validated it, a
    default filled in where the call gave none. It raises pydantic's
    ``ValidationError`` when a value is not its input's type or a required
    input is missing, and ignores names that are not inputs.
    ``CompiledNode.validate`` checks every call of a node through one, built
    the first time and kept.

    A default goes through its input's type like a value the call gave, so
    ``n: Integer = 3`` reaches the node as an ``Integer``. Inputs are the
    keys of a ``TypedDict``, so any name an input can carry is a key
    (``schema``, ``json``, ``inner.text``) and the result is a plain dict.
    """
    adapter = TypeAdapter(
        with_config(ConfigDict(extra="ignore", arbitrary_types_allowed=True))(
            TypedDict("Inputs", {inp.name: inp.dtype for inp in inputs})  # type: ignore[operator]
        )
    )
    defaults = {inp.name: inp.default for inp in inputs if not inp.required}

    def check(values: Mapping[str, Any]) -> dict[str, Any]:
        return adapter.validate_python({**defaults, **values})

    return check
