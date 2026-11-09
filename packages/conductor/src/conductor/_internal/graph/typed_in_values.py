"""How a value an author typed into a node's input reads: one value, or many.

A stored ``Static`` holds JSON: a file comes back as a dict, an authored
schema as a list. Each is read here once, through its field's declared type.
A sequence the type can't read as one value is read as a sequence of values,
which keeps a list-shaped scalar (a schema, a list of tags) one value while
three uploaded files are three rows. A value on a parameter typed ``Any`` is
left alone: only an edge can give that parameter a type.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache
from typing import Any

from pydantic import TypeAdapter, ValidationError

from conductor._internal.dtype import name_of
from conductor._internal.errors import Problem, problem
from conductor._internal.field_records import Input
from conductor._internal.graph.binding import static_values
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.table import holds_tables, missing_column_in


def read_typed_in_values(inputs: Iterable[Input], node: GraphNode) -> tuple[dict[str, Any], frozenset[str], list[Problem]]:
    """Every value the author typed into ``node`` on ``inputs``, typed; which of them hold many values; and the values no type could read."""
    values = static_values(node.bindings)
    if not values:
        return {}, frozenset(), []
    declared = {i.name: i for i in inputs}
    typed: dict[str, Any] = {}
    listed: set[str] = set()
    invalid: list[Problem] = []
    for name, value in values.items():
        inp = declared.get(name)
        if inp is None or inp.dtype.id == "any":
            continue
        try:
            typed[name], many = _read_one_value(inp, value)
        except (ValidationError, TypeError, ValueError) as unreadable:
            column = missing_column_in(unreadable) if isinstance(unreadable, ValidationError) else None
            if column is not None:
                invalid.append(problem("table_column_missing", node.id, name, column=column))
                continue
            said = _what_the_type_said(unreadable)
            invalid.append(problem("invalid_static", node.id, name, input=_short_repr(value), expected=name_of(inp.dtype), **({"reason": said} if said else {})))
            continue
        if many:
            listed.add(name)
    return typed, frozenset(listed), invalid


def _read_one_value(inp: Input, value: Any) -> tuple[Any, bool]:
    """``value`` read through ``inp``'s type, and whether it is many values."""
    element = inp.dtype.element
    if element is not None:
        if not _is_sequence(value):
            raise TypeError("a Series input takes a sequence")
        return [None if v is None and inp.takes_blank else _adapter(element).validate_python(v) for v in value], False
    if value is None and inp.takes_blank:
        return None, False
    try:
        return _adapter(inp.dtype).validate_python(value), False
    except ValidationError:
        if _is_sequence(value) and not holds_tables(inp.dtype):  # records are one table, not many
            return [_adapter(inp.dtype).validate_python(v) for v in value], True
        raise


def _short_repr(value: Any) -> str:
    """``value`` as written in a problem's sentence: never a page of it."""
    written = repr(value)
    return written if len(written) <= 60 else f"{written[:57]}..."


def _is_sequence(value: Any) -> bool:
    """A sequence of values, as a list widget or a multi-upload holds one? Text and bytes are one value each."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


@cache
def _adapter(dtype: type) -> TypeAdapter[Any]:
    return TypeAdapter(dtype)


def _what_the_type_said(invalid: Exception) -> str:
    """The sentence the type's own constructor raised, if it raised one.

    A constructor written for people says something specific ("every row
    needs three cells"), which tells an author what to change; anything
    pydantic says on its own is about JSON shape and is not shown.
    """
    if isinstance(invalid, ValidationError):
        errors = invalid.errors()
        cause = errors[0].get("ctx", {}).get("error") if errors else None
        return str(cause) if isinstance(cause, ValueError) else ""
    return str(invalid) if isinstance(invalid, ValueError) else ""
