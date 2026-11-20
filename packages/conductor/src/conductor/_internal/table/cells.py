"""How a table keeps a cell of a declared column type, as polars holds it.

Polars has no type for a host's model, dataclass or enum. A table keeps such a cell as plain data: a model or a
dataclass as the dict of its declared fields, an enum member as its value, and a list or tuple as a list. Say a row
type has a field ``money: Money | None``. The cells ``[Money(amount=1.0), None]`` are kept as the struct column
``[{"amount": 1.0}, None]``.

A cell is written by the fields its column's type declares, as they are, with no host serialization rule: a
``field_serializer`` is not applied, a ``computed_field`` is not added, a field marked ``exclude=True`` is kept,
and an instance of a subclass is written by the declared class's fields. So reading the kept cell back by the same
type gives the same value.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from enum import Enum
from types import NoneType
from typing import Annotated, Any, Literal, get_args, get_origin

import polars as pl

from conductor._internal.types import (
    declares_fields,
    is_alias,
    is_typed_dict,
    type_fields,
    union_members,
)

from .write import CELLS

#: The polars type of a cell of each of Python's own classes, as a saved table keeps it. A ``datetime`` and a
#: ``Decimal`` are left out, since their polars type depends on the cells (a time zone, a scale).
_DTYPES: dict[Any, Any] = {
    bool: pl.Boolean,
    int: pl.Int64,
    float: pl.Float64,
    str: pl.String,
    bytes: pl.Binary,
    datetime.date: pl.Date,
    datetime.time: pl.Time,
    datetime.timedelta: pl.Duration("us"),
}


def kept_column(name: str, cells: list[Any], hint: Any) -> pl.Series:
    """Returns the cells of the column ``name``, each a value of ``hint``, as a table keeps them (``cell_writer``).

    A column of a model, a dataclass or a ``TypedDict`` is built field by field, each field a column of its own, so
    no dict is made per cell: ``[Money(amount=1.0), None]`` gives the column ``amount`` ``[1.0, None]``, made into
    the struct column ``[{"amount": 1.0}, None]``. Any other column is written cell by cell.
    """
    fields = _fields_of(hint)
    if fields is None:
        write = cell_writer(hint)
        return pl.Series(name, [write(cell) for cell in cells], strict=False)
    typed_dict, declared = fields
    parts = [
        kept_column(field, [_field_of(cell, field, typed_dict) for cell in cells], field_hint)
        for field, field_hint in declared.items()
        # a TypedDict's key no cell has, one a NotRequired field lets it leave out, is no field of the struct
        if not typed_dict or any(cell is not None and field in cell for cell in cells)
    ]
    column = pl.DataFrame(parts, height=len(cells)).to_struct(name)
    if None not in cells:
        return column
    present = pl.Series([cell is not None for cell in cells])
    return pl.select(pl.when(present).then(column)).to_series().alias(name)


def _field_of(cell: Any, field: str, typed_dict: bool) -> Any:
    """Returns the field ``field`` of ``cell``, or ``None`` for a cell that is. A ``TypedDict``'s field is read by
    its key, and is ``None`` where the dict leaves it out."""
    if cell is None:
        return None
    return cell.get(field) if typed_dict else getattr(cell, field)


def _fields_of(hint: Any) -> tuple[bool, dict[str, Any]] | None:
    """Returns whether ``hint``, with ``None`` and ``Annotated`` taken off, is a ``TypedDict``, and the fields it
    declares, or ``None`` if it is no model, dataclass or ``TypedDict``."""
    if get_origin(hint) is Annotated:
        return _fields_of(get_args(hint)[0])
    members = [member for member in union_members(hint) if member is not NoneType]
    if len(members) != 1:
        return None
    if members[0] is not hint:
        return _fields_of(members[0])
    if not declares_fields(get_origin(hint) or hint):
        return None
    return is_typed_dict(hint), type_fields(hint)


def cell_writer(hint: Any) -> Callable[[Any], Any]:
    """Returns the function that writes a value of ``hint`` as a table keeps it.

    A ``None`` stays ``None`` at any depth. The writers of what a type holds are made when the first value is
    written, so a type that holds itself, like a tree of nodes, is written as deep as its value goes.
    """
    if is_alias(hint):
        return _once(lambda: cell_writer(hint.__value__))
    origin = get_origin(hint)
    if origin is Annotated:
        return cell_writer(get_args(hint)[0])
    members = [member for member in union_members(hint) if member is not NoneType]
    if len(members) > 1:
        return _union_writer(members)
    if len(members) == 1 and members[0] is not hint:
        return cell_writer(members[0])
    origin = origin or hint
    if declares_fields(origin):
        return _fields_writer(hint)
    arguments = get_args(hint)
    # a table keeps a tuple as a list, of one polars type, so each item is written by the first item's type
    if isinstance(origin, type) and issubclass(origin, (list, tuple, set, frozenset)) and arguments:
        write_item = cell_writer(arguments[0])
        return lambda value: None if value is None else [write_item(one) for one in value]
    return _plain


def _once(make: Callable[[], Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """Returns a writer that makes the writer it calls when it first writes a value, with ``make``."""
    made: list[Callable[[Any], Any]] = []

    def write(value: Any) -> Any:
        if not made:
            made.append(make())
        return made[0](value)

    return write


def _fields_writer(hint: Any) -> Callable[[Any], Any]:
    """Writes a value of a model, a dataclass or a ``TypedDict`` as the dict of the fields ``hint`` declares, each
    written by its own type, so an instance of a subclass is written by the declared class's fields. A
    ``TypedDict``'s key the value leaves out stays out."""
    typed_dict = is_typed_dict(hint)
    fields: dict[str, Callable[[Any], Any]] = {}

    def write(value: Any) -> Any:
        if value is None:
            return None
        if not fields:
            fields.update({name: cell_writer(field) for name, field in type_fields(hint).items()})
        if typed_dict:
            # a key a NotRequired field lets the dict leave out stays out
            return {
                name: write_field(value[name])
                for name, write_field in fields.items()
                if name in value
            }
        return {name: write_field(getattr(value, name)) for name, write_field in fields.items()}

    return write


def _union_writer(members: list[Any]) -> Callable[[Any], Any]:
    """Writes a value of a union of models or dataclasses by the member whose class it is, and anything else as it
    is (``_plain``)."""
    classes = [
        (member, cell_writer(member))
        for member in members
        if isinstance(member, type) and declares_fields(member) and not is_typed_dict(member)
    ]

    def write(value: Any) -> Any:
        for cls, write_member in classes:
            if isinstance(value, cls):
                return write_member(value)
        return _plain(value)

    return write


def _plain(value: Any) -> Any:
    """Writes a value whose type says nothing more, like a cell of ``Any``: a model or dataclass by its own class's
    fields, an enum member as its value, a list or tuple as a list, and a dict with each value so."""
    kind = type(value)
    if value is None or kind in CELLS:
        return value
    if kind is list or kind is tuple:
        return [_plain(one) for one in value]
    if kind is dict:
        return {key: _plain(one) for key, one in value.items()}
    if isinstance(value, Enum):
        return _plain(value.value)
    if declares_fields(kind):
        return cell_writer(kind)(value)
    # any other object stays one, and a saved run refuses its column (``as_saved``)
    return value


def kept_dtype(hint: Any, walked: frozenset[int] = frozenset()) -> Any:
    """Returns the polars type a table keeps a column of ``hint`` as, or ``None`` where the cells decide it.

    ``Money | None``, with ``Money`` declaring ``amount: float``, is ``Struct({"amount": Float64})``, and
    ``list[int]`` is ``List(Int64)``, and a ``Literal`` or an enum of texts ``String``. Any other type is ``None``,
    such as a ``datetime``, a ``Decimal``, a tuple, ``Any``, a union of two types and a type that holds itself, like
    a tree of nodes.
    ``walked`` is the types with fields on the way down.
    """
    if is_alias(hint):
        return None
    origin = get_origin(hint)
    if origin is Annotated:
        return kept_dtype(get_args(hint)[0], walked)
    members = [member for member in union_members(hint) if member is not NoneType]
    if len(members) != 1:
        return None
    if members[0] is not hint:
        return kept_dtype(members[0], walked)
    if origin is Literal:
        return _one_dtype(get_args(hint))
    origin = origin or hint
    if isinstance(origin, type) and issubclass(origin, Enum):
        return _one_dtype([member.value for member in origin])
    if declares_fields(origin):
        if id(origin) in walked:
            return None
        fields = {
            name: kept_dtype(field, walked | {id(origin)})
            for name, field in type_fields(hint).items()
        }
        return None if None in fields.values() else pl.Struct(fields)
    arguments = get_args(hint)
    if isinstance(origin, type) and issubclass(origin, (list, set, frozenset)) and arguments:
        inner = kept_dtype(arguments[0], walked)
        return None if inner is None else pl.List(inner)
    return _DTYPES.get(origin) if isinstance(origin, type) else None


def _one_dtype(values: Any) -> Any:
    """Returns the polars type of ``values``, a ``Literal``'s or an enum's, when they are all of one class it names,
    or ``None``."""
    kinds = {type(value) for value in values}
    return _DTYPES.get(kinds.pop()) if len(kinds) == 1 else None
