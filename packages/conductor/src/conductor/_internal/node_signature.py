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
  ``compute_added_inputs`` hook says which inputs land there. ``AsItArrives``
  never runs the node per row, ``Series[Any]`` reduces, ``Any`` or a type
  runs it per row;
* on the return: a type (one output), a frozen dataclass (one output
  per field), ``Mapping`` (the node's ``compute_outputs`` says which),
  ``None`` (no outputs), and ``X | AskPerson`` read as ``X``.

Anything else inside ``Annotated`` raises where it is written.
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
    read_field_annotation,
    split_off_none,
)
from conductor._internal.errors import NodeDeclarationError, Problem
from conductor._internal.field_records import Input, Output, Param, Result, type_read_by_input
from conductor._internal.skip_and_ask import AskPerson
from conductor.widgets import Widget

_T = TypeVar("_T")

DEFAULT_OUTPUT_NAME = "result"


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
class ParsedSignature:
    """What ``read_signature`` found: the interface, and how the function's return and ``**x`` work."""

    interface: Interface
    #: The ``FromRun`` parameters: the value of each type goes under this name in a call.
    from_run: dict[str, type]
    #: How a return splits across the outputs: a ``DType``, a record class,
    #: ``Mapping``, or ``None`` for a function that declares no outputs.
    return_shape: Any
    #: The name and hint of a ``**x`` parameter, ``None`` when it has none.
    var_keyword_name_and_hint: tuple[str, Any] | None


def read_signature(function: Callable[..., Any]) -> ParsedSignature:
    """Read ``function``'s signature. Every mistake is a ``NodeDeclarationError``, at import."""
    hints = get_type_hints(function, include_extras=True)
    inputs: list[Input] = []
    from_run_parameters: dict[str, type] = {}
    var_keyword_name_and_hint: tuple[str, Any] | None = None
    for name, parameter in inspect.signature(function).parameters.items():
        hint = hints.get(name, parameter.annotation)
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            raise NodeDeclarationError(
                f"parameter *{name}: a node's inputs are named, so an edge or a binding can land on each; *args names none",
                code="var_positional_parameter",
            )
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword_name_and_hint = (name, Any if hint is inspect.Parameter.empty else hint)
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise NodeDeclarationError(f"parameter {name!r} is positional-only; the engine passes every input by name", code="positional_only_parameter")
        if name.startswith("_"):
            raise NodeDeclarationError(f"parameter {name!r}: an input's name cannot start with an underscore", code="parameter_name_private")
        declared, extras = _split_annotated(hint)
        if any(isinstance(extra, _FromRunMark) for extra in extras):
            from_run_parameters[name] = declared
            continue
        inputs.append(_input_from_parameter(name, declared, extras, parameter))
    return_shape, outputs = _return_shape_and_outputs(hints)
    names_on_both_sides = {i.name for i in inputs} & {o.name for o in outputs}
    if names_on_both_sides:
        name = sorted(names_on_both_sides)[0]
        raise NodeDeclarationError(
            f"the parameter {name!r} and the output {name!r} share a name, and a field name is unique within a node; "
            f"rename the parameter, or name the output with Result(name=...)",
            code="field_on_both_sides",
        )
    return ParsedSignature(Interface(inputs=tuple(inputs), outputs=outputs, needs=frozenset(from_run_parameters.values())), from_run_parameters, return_shape, var_keyword_name_and_hint)


def input_template_for_var_keyword(name: str, hint: Any) -> Input:
    """The record every edge into ``**name: hint`` is an input like: named on the edge when compile makes one."""
    declared, extras = _split_annotated(hint)
    if any(isinstance(extra, _AsItArrivesMark) for extra in extras):
        return Input("", _ReceivedAsItArrives)
    one_value, _, _ = read_field_annotation(declared)
    dtype = dtype_of(one_value)
    if dtype is None or dtype is DType:
        raise NodeDeclarationError(
            f"**{name}: {declared!r} is no type an edge carries; each edge is an input, declared a type, Any or AsItArrives",
            code="open_interface_annotation",
        )
    return Input("", declared)


def _input_from_parameter(name: str, declared: Any, extras: tuple[Any, ...], parameter: inspect.Parameter) -> Input:
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
        **param.bounds, **({"default": parameter.default} if has_default else {}),
    )


def _return_shape_and_outputs(hints: Mapping[str, Any]) -> tuple[Any, tuple[Output, ...]]:
    """The declared return's split and the outputs it declares."""
    if "return" not in hints:
        raise NodeDeclarationError("a node's function must declare a return type; -> None declares no outputs", code="missing_return_type")
    hint = _without_ask_person(hints["return"])
    if hint is None or hint is NoneType:
        return None, ()
    declared, extras = _split_annotated(hint)
    if any(isinstance(extra, Param) for extra in extras):
        raise NodeDeclarationError("the return carries a Param; a return carries a Result, and Param belongs on a parameter", code="param_on_return")
    bare, _, blank = read_field_annotation(declared)
    if (dtype := dtype_of(bare)) is not None:
        return dtype, (_output_from_result(DEFAULT_OUTPUT_NAME, declared, _result_annotation_in(extras)),)
    if blank:
        raise NodeDeclarationError(f"{bare!r} can't take a blank: only one value, or a field of a returned record, is blank", code="blank_on_a_record")
    if isinstance(declared, type) and is_dataclass(declared):
        return declared, _outputs_of_returned_dataclass(declared)
    if declared is Mapping or get_origin(declared) is Mapping:
        return Mapping, ()
    raise NodeDeclarationError(f"a node returns a type an edge can carry (str, a pydantic model, a DType), Series[X] for many of them, a frozen dataclass of them, Mapping[str, Any] or None — got {declared!r}", code="unsupported_return_type")


def _outputs_of_returned_dataclass(record: type) -> tuple[Output, ...]:
    """One output per field of a returned dataclass."""
    hints = get_type_hints(record, include_extras=True)
    outputs = []
    for record_field in fields(record):
        declared, extras = _split_annotated(hints[record_field.name])
        if any(isinstance(extra, Param) for extra in extras):
            raise NodeDeclarationError(f"{record.__name__}.{record_field.name} carries a Param; a record's field carries a Result", code="param_on_record_field")
        bare, _, _ = read_field_annotation(declared)
        result = _result_annotation_in(extras)
        if dtype_of(bare) is None:
            raise NodeDeclarationError(f"{record.__name__}.{record_field.name} must be a type an edge can carry — got {declared!r}", code="record_field_needs_dtype")
        if result.name is not None:
            raise NodeDeclarationError(f"{record.__name__}.{record_field.name}: a record's field is named by the field, so Result(name=...) has nothing to name", code="result_name_on_record_field")
        outputs.append(_output_from_result(record_field.name, declared, result))
    return tuple(outputs)


def _output_from_result(default_name: str, dtype: Any, result: Result) -> Output:
    return Output(result.name or default_name, dtype, title=result.title, description=result.description, exclusive_group=result.exclusive_group)


def _result_annotation_in(extras: tuple[Any, ...]) -> Result:
    return next((extra for extra in extras if isinstance(extra, Result)), Result())


def _without_ask_person(hint: Any) -> Any:
    """``X | AskPerson`` is read as ``X``: a node that pauses to ask a person declares no output for it."""
    if get_origin(hint) in (Union, UnionType) and AskPerson in get_args(hint):
        members = tuple(member for member in get_args(hint) if member is not AskPerson)
        return Union[members]  # noqa: UP007  (the members are runtime values)
    return hint


def _split_annotated(hint: Any) -> tuple[Any, tuple[Any, ...]]:
    """The type under ``Annotated[...]`` with its ``| None`` kept beside it, and what the ``Annotated`` holds.

    ``Annotated[Text, Param()] | None`` and ``Annotated[Text | None, Param()]`` read alike."""
    written, blank = split_off_none(hint)
    if get_origin(written) is Annotated:
        declared, *extras = get_args(written)
        return (declared | None if blank else declared), tuple(extras)
    return hint, ()


def split_return_into_outputs(return_shape: Any, value: Any, outputs: tuple[Output, ...], declared: tuple[Output, ...]) -> dict[str, Any] | list[Problem]:
    """Split what a node returned into ``{output name: value}``.

    ``return_shape`` is ``ParsedSignature.return_shape``, ``outputs`` what the node
    has in its graph and ``declared`` what its signature declared. A value
    of the wrong shape is a list of one problem that never quotes the value.
    """
    if return_shape is None:
        if value is not None:
            return _wrong_shape(f"the node declares no outputs and returned a {type(value).__name__}")
        return {}
    if isinstance(return_shape, type) and issubclass(return_shape, DType):
        return {declared[0].name: value}
    if return_shape is Mapping:
        names = {output.name for output in outputs}
        if not isinstance(value, Mapping) or set(value) != names:
            given = sorted(value) if isinstance(value, Mapping) else type(value).__name__
            return _wrong_shape(f"the node must return exactly the outputs its interface names, {sorted(names)}, not {given}")
        return dict(value)
    if not isinstance(value, return_shape):
        return _wrong_shape(f"the node must return a {return_shape.__name__}, not a {type(value).__name__}")
    return {record_field.name: getattr(value, record_field.name) for record_field in fields(return_shape)}


def _wrong_shape(message: str) -> list[Problem]:
    return [Problem(type="invalid_output", msg=message)]


def build_call_validator(inputs: tuple[Input, ...]) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """The check the values of ``inputs`` go through at a call: values in, one per input as its type read it.

    It raises pydantic's ``ValidationError`` when a value is not its input's
    type or one is missing, and ignores names that are not among ``inputs``.
    Inputs are the keys of a ``TypedDict``, so any name an input can carry is
    a key (``schema``, ``json``, ``inner.text``) and the result is a plain dict.
    """
    adapter = TypeAdapter(
        with_config(ConfigDict(extra="ignore", arbitrary_types_allowed=True))(
            TypedDict("Inputs", {input_.name: type_read_by_input(input_) for input_ in inputs})  # type: ignore[operator]
        )
    )
    return adapter.validate_python
