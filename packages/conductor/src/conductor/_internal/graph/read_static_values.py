"""How a static reads: one value, or many.

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

from conductor._internal.dtype import type_name_for_sentence
from conductor._internal.errors import Problem, make_problem
from conductor._internal.field_records import Input
from conductor._internal.graph.binding import static_values
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.table import is_table_or_series_of_tables, missing_column_in


def read_static_values(inputs: Iterable[Input], node: GraphNode) -> tuple[dict[str, Any], frozenset[str], list[Problem]]:
    """Every static on ``node`` for ``inputs``, read through its type; which of them hold many values; and the problems no type could read."""
    values = static_values(node.bindings)
    if not values:
        return {}, frozenset(), []
    input_by_name = {i.name: i for i in inputs}
    statics: dict[str, Any] = {}
    inputs_given_a_list: set[str] = set()
    problems: list[Problem] = []
    for name, value in values.items():
        input_ = input_by_name.get(name)
        if input_ is None or input_.dtype.id == "any":
            continue
        try:
            statics[name], many = _read_value_or_list(input_, value)
        except (ValidationError, TypeError, ValueError) as unreadable:
            column = missing_column_in(unreadable) if isinstance(unreadable, ValidationError) else None
            if column is not None:
                problems.append(make_problem("table_column_missing", node.id, name, column=column))
                continue
            said = _message_from_type_constructor(unreadable)
            problems.append(make_problem("invalid_static", node.id, name, input=_short_repr(value), expected=type_name_for_sentence(input_.dtype), **({"reason": said} if said else {})))
            continue
        if many:
            inputs_given_a_list.add(name)
    return statics, frozenset(inputs_given_a_list), problems


def _read_value_or_list(input_: Input, value: Any) -> tuple[Any, bool]:
    """``value`` read through ``input_``'s type, and whether it is many values."""
    element = input_.dtype.element
    if element is not None:
        if not _is_sequence(value):
            raise TypeError("a Series input takes a sequence")
        return [None if item is None and input_.takes_blank else _adapter(element).validate_python(item) for item in value], False
    if value is None and input_.takes_blank:
        return None, False
    try:
        return _adapter(input_.dtype).validate_python(value), False
    except ValidationError:
        if _is_sequence(value) and not is_table_or_series_of_tables(input_.dtype):  # records are one table, not many
            return [_adapter(input_.dtype).validate_python(item) for item in value], True
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


def _message_from_type_constructor(unreadable: Exception) -> str:
    """The sentence the type's own constructor raised, if it raised one.

    A constructor written for people says something specific ("every row
    needs three cells"), which tells an author what to change; anything
    pydantic says on its own is about JSON shape and is not shown.
    """
    if isinstance(unreadable, ValidationError):
        errors = unreadable.errors()
        cause = errors[0].get("ctx", {}).get("error") if errors else None
        return str(cause) if isinstance(cause, ValueError) else ""
    return str(unreadable) if isinstance(unreadable, ValueError) else ""
