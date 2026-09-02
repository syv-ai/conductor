"""What a node returns.

The return type is the declaration: what ``run`` is annotated to
return says what outputs the node has and how the returned value splits
across them. There are three readings, each visible in the annotation:

* a ``DType`` — one output, named ``result``, titled by the ``Result`` on
  the annotation;
* a frozen dataclass — one output per field, named by the field, each
  field annotated ``Annotated[DType, Result(title=...)]``: a **record**,
  and the record is the schema;
* ``Mapping[str, Any]`` — the outputs are whatever ``compute_outputs``
  answered for the placement, and ``run`` returns them by name.

Nothing is positional and nothing is auto-named: a field's name is the
persisted binding key and the ``Ref`` an author wires, so it is a name a
person chose. A node whose ``run`` returns the wrong shape is broken and
says so at the moment of disagreement.

``Result`` is what an *author writes* — a title, a description and, for
a branch, the ``choice`` it belongs to — and an ``Output`` is what
the node *has*. Nothing here mentions downloading: a value is
downloadable because of what it *is*.

A return may declare ``Any`` in place of a ``DType``: the node routes a
value it does not read, and its ``compute_outputs`` types the output
from what arrives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from conductor.dtype import DType, dtype_of
from conductor.metadata import Output

__all__ = ["Result", "outputs_of", "unpack"]

RESULT_KEY = "result"


@dataclass(frozen=True)
class Result:
    """What does the author want said about this output?

    The title, the description and — for one of several exclusive
    alternatives — the ``choice`` group it belongs to; the branches of
    Hvis/ellers all say ``choice="grene"``.

    An author writes one inside the ``Annotated`` on ``run``'s return type, or
    on a field of the record it returns. It is read exactly once, by
    ``outputs_of`` while ``Interface.of`` walks the signature, and what travels
    from there is the ``Output`` it produced. Its opposite number on the input
    side is the widget annotation, which carries the same field-level facts
    for a value coming in.

    Not an ``Output``: this is the authored declaration, that is the derived
    record. Nothing here says how the value is rendered or whether it can be
    downloaded — a value is downloadable because of what it *is*.
    """

    title: str
    description: str | None = None
    choice: str | None = None


def outputs_of(return_hint: Any) -> tuple[Any, tuple[Output, ...]]:
    """The declared return type, and the outputs it declares.

    The type comes back with ``Annotated`` stripped — a ``DType`` (or
    ``Any``), a record class, or ``Mapping`` — and is what ``unpack``
    switches on, so the two read one declaration the same way. A ``DType``
    is tried first: a dtype may itself be a dataclass, and it is then one
    value, not a record.
    """
    declared = get_args(return_hint)[0] if get_origin(return_hint) is Annotated else return_hint
    dtype = dtype_of(declared)
    if dtype is not None:
        result = _result_on(return_hint)
        if result is None:
            raise TypeError("a one-output run() must be annotated Annotated[DType, Result(title=...)]")
        return dtype, (_output(RESULT_KEY, dtype, result),)
    if isinstance(declared, type) and is_dataclass(declared):
        hints = get_type_hints(declared, include_extras=True)
        outputs = []
        for field in fields(declared):
            hint = hints[field.name]
            field_dtype, result = dtype_of(hint), _result_on(hint)
            if field_dtype is None or result is None:
                raise TypeError(
                    f"{declared.__name__}.{field.name} must be Annotated[DType, Result(title=...)] — got {hint!r}"
                )
            outputs.append(_output(field.name, field_dtype, result))
        return declared, tuple(outputs)
    if declared is Mapping or get_origin(declared) is Mapping:
        return Mapping, ()
    raise TypeError(
        f"run() must return a DType, a dataclass of them, or Mapping[str, Any] — got {declared!r}"
    )


def unpack(returns: Any, value: Any, outputs: tuple[Output, ...]) -> dict[str, Any]:
    """Split what ``run`` returned across ``outputs``, by the declared return type.

    ``Mapping`` is the reading for every roster the signature did not
    declare: a ``compute_outputs`` roster, and an ``Interface`` a host gave
    by value — an embedded flow's ``run`` returns its outputs keyed by
    address, and this splits them the same way.
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


def _output(name: str, dtype: Any, result: Result) -> Output:
    return Output(
        name=name, dtype=dtype, title=result.title, description=result.description, choice=result.choice
    )


def _result_on(hint: Any) -> Result | None:
    if get_origin(hint) is not Annotated:
        return None
    return next((extra for extra in get_args(hint)[1:] if isinstance(extra, Result)), None)
