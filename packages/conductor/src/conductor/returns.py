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
        ...   # the outputs were computed for the placement; returned by name

``outputs_of`` turns the annotation into ``Output`` records and ``unpack``
splits a returned value across those outputs by name. Nothing is
positional and nothing is auto-named: an output's name is what other nodes
wire to, so it is always a name the author chose. A ``run`` that returns
the wrong shape raises at the point of disagreement.

``Result`` is what the author writes — title, description and, for one of
several exclusive branches, its ``choice`` group. ``Output`` is the record
the node ends up with. A return may be ``Any`` in place of a ``DType``
when the node passes a value through without reading it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from conductor.dtype import DType, dtype_of
from conductor.metadata import Output

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
    placement's computed outputs are used). Anything else is a
    ``TypeError``, as is a ``DType`` or a field without a ``Result``.
    """
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
    """Split what ``run`` returned into ``{output name: value}``.

    ``returns`` is the ``declared`` half of ``outputs_of``'s answer. A
    ``DType`` (or ``Any``) return lands on ``"result"``; a record is read
    field by field; a ``Mapping`` must name exactly the declared outputs.
    A value of the wrong shape is a ``ValueError``.
    """
    if returns is Any or (isinstance(returns, type) and issubclass(returns, DType)):
        return {RESULT_KEY: value}
    if returns is Mapping:
        names = {output.name for output in outputs}
        if not isinstance(value, Mapping) or set(value) != names:
            raise ValueError(
                f"run() must return exactly the outputs its roster names, {sorted(names)} — got {value!r}"
            )
        return dict(value)
    if not isinstance(value, returns):
        raise ValueError(f"run() must return a {returns.__name__} — got {value!r}")
    return {field.name: getattr(value, field.name) for field in fields(returns)}
