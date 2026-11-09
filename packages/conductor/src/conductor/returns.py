"""What a node returns, read off ``run``'s return annotation.

The return type *is* the output declaration. Three shapes are read::

    def run(self, text: Text) -> Annotated[Text, Result(title="Summary")]:
        ...   # one output, named "result"

    @dataclass(frozen=True)
    class Parts:
        head: Annotated[Text, Result(title="Head")]
        tail: Annotated[Text, Result(title="Tail")]

    def run(self, text: Text) -> Parts:
        ...   # one output per field: "head" and "tail"

    def run(self, **inputs: Single) -> Mapping[str, Any]:
        ...   # the outputs were computed for the node; returned by name

``outputs_of`` turns the annotation into ``Output`` records and ``unpack``
splits a returned value across those outputs by name; each value is then
checked against its output at the door to the state
(``conductor.execution.validate_unit_outputs``). Nothing is
positional and nothing is auto-named: an output's name is what other nodes
edge to, so it is always a name the author chose. A ``run`` that returns
the wrong shape raises at the point of disagreement.

``Result`` (in ``conductor.metadata``, beside its twin ``Param``) is what
the author writes — title, description and, for one of several exclusive
branches, its ``choice`` group. ``Output`` is the record the node ends up
with. A return may be ``Any`` in place of a ``DType`` when the node passes
a value through without reading it.

A ``run`` that may return ``AskPerson`` — the value a node returns when a
person must answer before the graph can continue — says so in the same
place, as a union: ``-> Annotated[Text, Result(...)] | AskPerson``. The union
member declares no output; ``outputs_of`` reads the declaration past it.
It is the one thing a class says about asking, and it is a fact about the
return, not a capability: the engine still acts on the value alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from conductor._sentinel import AskPerson
from conductor.dtype import DType, dtype_of
from conductor.metadata import Output, Param, Result

RESULT_KEY = "result"


def outputs_of(return_hint: Any) -> tuple[Any, tuple[Output, ...]]:
    """Read ``run``'s return annotation: the declared type, and the outputs it declares.

    Returns ``(declared, outputs)``. ``declared`` is the annotation with
    ``Annotated`` stripped — a ``DType`` or ``Any``, a record class, or
    ``Mapping`` — and is what ``unpack`` later switches on. A ``DType``
    return declares one output named ``"result"``; a frozen dataclass
    declares one output per field; ``Mapping`` declares none (the
    node's computed outputs are used). Anything else is a
    ``TypeError``, as is a ``DType`` or a field without a ``Result``.

    ``X | AskPerson`` is read as ``X``: a node that pauses to ask a person
    declares no output for it.
    """
    if get_origin(return_hint) in (Union, UnionType):
        members = tuple(member for member in get_args(return_hint) if member is not AskPerson)
        if len(members) == 1 and AskPerson in get_args(return_hint):
            return outputs_of(members[0])
    if Param.on(return_hint) is not None:
        raise TypeError("the return carries a Param; a return carries a Result, and Param belongs on a parameter")
    declared = get_args(return_hint)[0] if get_origin(return_hint) is Annotated else return_hint
    dtype = dtype_of(declared)
    if dtype is not None:
        result = Result.on(return_hint)
        if result is None:
            raise TypeError("a one-output run() must be annotated Annotated[DType, Result(title=...)]")
        return dtype, (result.output(RESULT_KEY, dtype),)
    if isinstance(declared, type) and is_dataclass(declared):
        hints = get_type_hints(declared, include_extras=True)
        outputs = []
        for field in fields(declared):
            hint = hints[field.name]
            field_dtype, result = dtype_of(hint), Result.on(hint)
            if Param.on(hint) is not None:
                raise TypeError(f"{declared.__name__}.{field.name} carries a Param; a record's field carries a Result")
            if field_dtype is None or result is None:
                raise TypeError(
                    f"{declared.__name__}.{field.name} must be Annotated[DType, Result(title=...)] — got {hint!r}"
                )
            outputs.append(result.output(field.name, field_dtype))
        return declared, tuple(outputs)
    if declared is Mapping or get_origin(declared) is Mapping:
        return Mapping, ()
    raise TypeError(
        f"run() must return a DType, a dataclass of them, or Mapping[str, Any] — got {declared!r}"
    )


def unpack(returns: Any, value: Any, outputs: tuple[Output, ...]) -> dict[str, Any]:
    """Split what ``run`` returned into ``{output name: value}``.

    ``returns`` is the ``declared`` half of ``outputs_of``'s answer. A
    ``DType`` (or ``Any``) return lands on ``"result"``; a record is read
    field by field; a ``Mapping`` must name exactly the declared outputs. A
    value of the wrong shape is a ``ValueError`` that never quotes the value.
    """
    if returns is Any or (isinstance(returns, type) and issubclass(returns, DType)):
        return {RESULT_KEY: value}
    if returns is Mapping:
        names = {output.name for output in outputs}
        if not isinstance(value, Mapping) or set(value) != names:
            given = sorted(value) if isinstance(value, Mapping) else type(value).__name__
            raise ValueError(f"run() must return exactly the outputs its interface names, {sorted(names)}, not {given}")
        return dict(value)
    if not isinstance(value, returns):
        raise ValueError(f"run() must return a {returns.__name__}, not a {type(value).__name__}")
    return {field.name: getattr(value, field.name) for field in fields(returns)}
