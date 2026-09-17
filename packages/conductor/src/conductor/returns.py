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
splits a returned value across those outputs by name. Nothing is
positional and nothing is auto-named: an output's name is what other nodes
edge to, so it is always a name the author chose. A ``run`` that returns
the wrong shape raises at the point of disagreement.

``Result`` is what the author writes — title, description and, for one of
several exclusive branches, its ``choice`` group. ``Output`` is the record
the node ends up with. A return may be ``Any`` in place of a ``DType``
when the node passes a value through without reading it.

A ``run`` that may return ``Asks`` — the value a node returns when a
person must answer before the graph can continue — says so in the same
place, as a union: ``-> Annotated[Text, Result(...)] | Asks``. The union
member declares no output; ``outputs_of`` reads the declaration past it.
It is the one thing a class says about asking, and it is a fact about the
return, not a capability: the engine still acts on the value alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from conductor._sentinel import Asks, is_skipped
from conductor.codec import from_wire
from conductor.dtype import DType, dtype_of
from conductor.metadata import Output
from conductor.series import Series

RESULT_KEY = "result"


@dataclass(frozen=True)
class Result:
    """What the author says about an output: its title, description and ``choice``.

    Written inside ``Annotated[...]`` on ``run``'s return type or on a field
    of the returned record. ``choice`` groups outputs that are exclusive
    alternatives — exactly one of the group is produced per run, as with
    the two branches of an if/else node::

        if_true: Annotated[Text, Result(title="If true", choice="branches")]
        if_false: Annotated[Text, Result(title="If false", choice="branches")]

    ``outputs_of`` reads it once and produces an ``Output``; the ``Result``
    itself is not kept.
    """

    title: str
    description: str | None = None
    choice: str | None = None

    @classmethod
    def on(cls, hint: Any) -> Result | None:
        """The ``Result`` written on ``hint``, or ``None`` if it carries none."""
        if get_origin(hint) is not Annotated:
            return None
        return next((extra for extra in get_args(hint)[1:] if isinstance(extra, cls)), None)

    def output(self, name: str, dtype: Any) -> Output:
        """The ``Output`` this declares for the field ``name`` of type ``dtype``."""
        return Output(name=name, dtype=dtype, title=self.title, description=self.description, choice=self.choice)


def outputs_of(return_hint: Any) -> tuple[Any, tuple[Output, ...]]:
    """Read ``run``'s return annotation: the declared type, and the outputs it declares.

    Returns ``(declared, outputs)``. ``declared`` is the annotation with
    ``Annotated`` stripped — a ``DType`` or ``Any``, a record class, or
    ``Mapping`` — and is what ``unpack`` later switches on. A ``DType``
    return declares one output named ``"result"``; a frozen dataclass
    declares one output per field; ``Mapping`` declares none (the
    node's computed outputs are used). Anything else is a
    ``TypeError``, as is a ``DType`` or a field without a ``Result``.

    ``X | Asks`` is read as ``X``: a node that pauses to ask a person
    declares no output for it.
    """
    if get_origin(return_hint) in (Union, UnionType):
        members = tuple(member for member in get_args(return_hint) if member is not Asks)
        if len(members) == 1 and Asks in get_args(return_hint):
            return outputs_of(members[0])
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
    """Split what ``run`` returned into ``{output name: value}``, each checked against its output.

    ``returns`` is the ``declared`` half of ``outputs_of``'s answer. A
    ``DType`` (or ``Any``) return lands on ``"result"``; a record is read
    field by field; a ``Mapping`` must name exactly the declared outputs.
    Each value is then read as the type its output declares, the way an
    input is on the way in, so a ``-> Text`` that returns ``42`` fails
    here and not in the node that reads it. A value of the wrong shape or
    type is a ``ValueError`` naming the output and never quoting the value.
    """
    if returns is Any or (isinstance(returns, type) and issubclass(returns, DType)):
        named = {RESULT_KEY: value}
    elif returns is Mapping:
        names = {output.name for output in outputs}
        if not isinstance(value, Mapping) or set(value) != names:
            given = sorted(value) if isinstance(value, Mapping) else type(value).__name__
            raise ValueError(f"run() must return exactly the outputs its interface names, {sorted(names)}, not {given}")
        named = dict(value)
    elif not isinstance(value, returns):
        raise ValueError(f"run() must return a {returns.__name__}, not a {type(value).__name__}")
    else:
        named = {field.name: getattr(value, field.name) for field in fields(returns)}
    return {name: _declared_type(name, given, outputs) for name, given in named.items()}


def _declared_type(name: str, value: Any, outputs: tuple[Output, ...]) -> Any:
    """``value`` as the type output ``name`` declares; ``SKIPPED`` and an
    ``Any`` output pass as they are, and an output the interface does not
    name is a ``ValueError``."""
    output = next((out for out in outputs if out.name == name), None)
    if output is None:
        raise ValueError(f"run() returned '{name}', which is not one of its outputs")
    if is_skipped(value) or output.dtype is Any:
        return value
    element = getattr(output.dtype, "element", None)
    if element is None:
        return _read_as(name, value, output.dtype)
    # A series output: each element is read as the element type, and a
    # skipped element stays skipped — that row is absent downstream.
    if isinstance(value, Series):
        return Series(value.index, [_read_as(name, item, element) for item in value.values], rows=value.rows)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_read_as(name, item, element) for item in value]
    raise ValueError(f"'{name}' is a series output and must be a sequence, not a {type(value).__name__}")


def _read_as(name: str, value: Any, dtype: Any) -> Any:
    """``value`` as ``dtype``; ``SKIPPED`` and ``Any`` pass; anything the type refuses names the output."""
    if is_skipped(value) or dtype is Any:
        return value
    try:
        return from_wire(value, dtype)
    except ValueError as invalid:
        raise ValueError(f"'{name}' must be a {dtype.__name__}, not a {type(value).__name__}") from invalid
