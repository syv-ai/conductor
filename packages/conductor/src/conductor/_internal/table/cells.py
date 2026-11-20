"""How a table keeps a cell of a declared column type, as polars holds it.

Polars has no type for a host's model, dataclass or enum. A table keeps such a cell as plain data: a model or a
dataclass as the dict of its declared fields, an enum member as its value, and a list or tuple as a list. Say a row
type has a field ``money: Money | None``. The cells ``[Money(amount=1.0), None]`` are kept as the struct column
``[{"amount": 1.0}, None]``.

A cell is written by the fields its column's type declares, as they are, with no host serialization rule: a
``field_serializer`` is not applied, a ``computed_field`` is not added, a field marked ``exclude=True`` is kept,
and an instance of a subclass is written by the declared class's fields. So reading the kept cell back by the same
type gives the same value.

``CellType`` is the one walk over a column's type. It takes off what doesn't change how a cell is kept (a type
alias, ``Annotated`` and ``None``) and says what the type is: a type with fields, a list of items, a union, named
values, or a plain value. Writing a cell (``CellType.write``), the polars type a column is kept as
(``CellType.dtype``) and the declaration check of a table's row (``CellType.keys_left_out``) all read it.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from enum import Enum
from types import NoneType
from typing import Annotated, Any, Literal, NotRequired, get_args, get_origin

import polars as pl
from pydantic import BaseModel

from conductor._internal.types import (
    declares_fields,
    field_hints,
    is_alias,
    is_typed_dict,
    type_fields,
    union_members,
)

from .write import CELLS

#: The polars type a table keeps a cell of each of Python's own classes as: the type ``CELLS`` names, with polars'
#: own defaults, which are a saved table's (microseconds, and no time zone).
_DTYPES = {kind: dtype() for kind, (dtype, _) in CELLS.items()}


def cell_type(hint: Any) -> CellType:
    """Returns the ``CellType`` of ``hint``, made once per hint. A hint that can't be hashed, like one with a
    ``Param`` whose choices hold a dict, gets a new one each time."""
    try:
        return _cached_cell_type(hint)
    except TypeError:
        return CellType(hint)


@functools.lru_cache(maxsize=4096)  # a server meets many column types
def _cached_cell_type(hint: Any) -> CellType:
    return CellType(hint)


def _kind_of(bare: Any, members: int) -> str:
    """Says what kind of ``CellType`` a type is, from its bare hint and its number of members other than ``None``."""
    origin = get_origin(bare) or bare
    if members > 1:
        return "union"
    if declares_fields(origin):
        return "fields"
    if origin is Literal or (isinstance(origin, type) and issubclass(origin, Enum)):
        return "named"
    if (
        isinstance(origin, type)
        and issubclass(origin, (list, tuple, set, frozenset))
        and get_args(bare)
    ):
        return "items"
    return "value"


class CellType:
    """A column's type, as a table keeps its cells.

    It is one of five kinds, read from the hint with a type alias, ``Annotated`` and ``None`` taken off:

    - ``"fields"``: a model, a dataclass or a ``TypedDict``, kept as a struct of its declared fields;
    - ``"items"``: a list, tuple or set, kept as a list of its items' type;
    - ``"union"``: two types or more, like ``Cat | Dog``;
    - ``"named"``: an enum or a ``Literal``, kept as its values;
    - ``"value"``: anything else, like ``int`` or ``Any``, kept as it is.

    What a type holds (its fields, its items, its members) is made the first time it is asked for, so a type that
    holds itself, like ``type Deep = list[Deep]`` or a tree of nodes, is walked only as deep as a value goes.

    Attributes:
        alias: The name of the type alias the hint was written as, such as ``JsonValue``, or ``None``.
        bare: The one type left once the alias, ``Annotated`` and ``None`` are taken off, or the union.
        kind: What the type is, as above.
    """

    def __init__(self, hint: Any) -> None:
        hint, self.alias = _stripped(hint)
        members = [_stripped(member) for member in union_members(hint)]
        self._members = [member for member, _ in members if member is not NoneType]
        if len(self._members) == 1:
            # ``Alias | None`` is the alias, by its name
            hint, alias = next(
                (member, alias) for member, alias in members if member is not NoneType
            )
            self.alias = self.alias or alias
        self.bare = hint
        self.kind = _kind_of(self.bare, len(self._members))

    @functools.cached_property
    def members(self) -> tuple[CellType, ...]:
        """The ``CellType`` of each member of a union, without ``None``."""
        return tuple(cell_type(member) for member in self._members) if self.kind == "union" else ()

    @functools.cached_property
    def _declared(self) -> dict[str, Any]:
        """Each field a type with fields declares, as written, ``NotRequired`` kept."""
        return dict(field_hints(self.bare)) if self.kind == "fields" else {}

    @functools.cached_property
    def fields(self) -> dict[str, CellType]:
        """The ``CellType`` of each field of a model, a dataclass or a ``TypedDict``, by name."""
        return (
            {name: cell_type(field) for name, field in type_fields(self.bare).items()}
            if self.kind == "fields"
            else {}
        )

    @functools.cached_property
    def item(self) -> CellType | None:
        """The ``CellType`` of a list's items. A tuple's items are all kept as its first item's type, since a
        table keeps a tuple as a list of one polars type."""
        return cell_type(get_args(self.bare)[0]) if self.kind == "items" else None

    @functools.cached_property
    def typed_dict(self) -> bool:
        """Whether the type is a ``TypedDict``, whose value is a dict that may leave out a ``NotRequired`` key."""
        return self.kind == "fields" and is_typed_dict(self.bare)

    def write(self, value: Any) -> Any:
        """Returns ``value``, a value of the type, as a table keeps it.

        A model or a dataclass is the dict of the fields the type declares, each written by its own type, and a
        ``TypedDict``'s key the dict leaves out stays out. A value of a union is written by the member whose class
        it is. Anything else is written by its own class (``_plain``). A ``None`` stays ``None`` at any depth.
        """
        if value is None:
            return None
        if self.typed_dict:
            return {
                name: field.write(value[name])
                for name, field in self.fields.items()
                if name in value
            }
        if self.kind == "fields":
            return {name: field.write(getattr(value, name)) for name, field in self.fields.items()}
        if self.item is not None:
            return [self.item.write(one) for one in value]
        for member in self.members:
            if member.kind == "fields" and not member.typed_dict and isinstance(value, member.bare):
                return member.write(value)
        return _plain(value)

    @functools.cached_property
    def dtype(self) -> Any:
        """The polars type a table keeps a column of the type as, or ``None`` where the cells decide it.

        ``Money | None``, with ``Money`` declaring ``amount: float``, is ``Struct({"amount": Float64})``,
        ``list[int]`` and ``tuple[int, ...]`` are ``List(Int64)``, and a ``Literal`` or an enum whose values are all
        texts ``String``. Any other type is ``None``, such as a dict, ``Any``, a union of two types, and a type met
        again inside itself, like a tree of nodes.
        """
        # a type met again inside itself, while its own polars type is worked out, gives None
        self.__dict__["dtype"] = None
        if self.kind == "fields":
            fields = {name: field.dtype for name, field in self.fields.items()}
            return None if None in fields.values() else pl.Struct(fields)
        if self.item is not None:
            return None if self.item.dtype is None else pl.List(self.item.dtype)
        if self.kind == "named":
            origin = get_origin(self.bare) or self.bare
            return _one_dtype(
                get_args(self.bare) if origin is Literal else [member.value for member in origin]
            )
        return _DTYPES.get(self.bare) if self.kind == "value" else None

    @functools.cached_property
    def keys_left_out(self) -> str | None:
        """Names the type, in this one, whose values may leave out a key, or ``None`` if none may.

        A table keeps a value that has keys as a struct, and a struct gives every row every key, a ``None`` for a
        key a row's value leaves out. So such a value would read back changed. These types' values may leave out a
        key, at any depth:

        - a dict with open keys, like ``dict[str, int]``;
        - a ``TypedDict`` with a ``NotRequired`` field;
        - a model that takes extra keys (``extra="allow"``);
        - a union of models, dataclasses or ``TypedDict``s whose fields differ, like ``Cat | Dog``.

        A type alias is named by its name, as ``JsonValue``. ``Any`` says nothing, so it lets nothing.
        """
        return self._keys_left_out_within(set())

    def _keys_left_out_within(self, walked: set[int]) -> str | None:
        """``keys_left_out``, walking only the types not in ``walked`` yet, so a type that holds itself ends."""
        walked.add(id(self))
        found = self._own_keys_left_out() or next(
            (
                found
                for part in self._parts()
                if id(part) not in walked
                and (found := part._keys_left_out_within(walked)) is not None
            ),
            None,
        )
        return self.alias if found is not None and self.alias is not None else found

    def _own_keys_left_out(self) -> str | None:
        """Names this type, if its own values may leave out a key, as ``keys_left_out`` lists them."""
        origin = get_origin(self.bare) or self.bare
        if self.kind == "value" and isinstance(origin, type) and issubclass(origin, Mapping):
            return origin.__name__ if self.bare is origin else str(self.bare).replace("typing.", "")
        if self.kind == "fields" and (
            any(get_origin(field) is NotRequired for field in self._declared.values())
            or (issubclass(origin, BaseModel) and origin.model_config.get("extra") == "allow")
        ):
            return str(origin.__qualname__)
        with_fields = [member for member in self.members if member.kind == "fields"]
        if len({tuple(member.fields) for member in with_fields}) > 1:
            return " | ".join(
                (get_origin(member.bare) or member.bare).__qualname__ for member in with_fields
            )
        return None

    def _parts(self) -> list[CellType]:
        """The types directly inside this one: its fields, its members and its items."""
        return [
            *self.fields.values(),
            *self.members,
            *([self.item] if self.item is not None else []),
        ]


def _stripped(hint: Any) -> tuple[Any, str | None]:
    """Returns ``hint`` with ``Annotated`` and any type alias taken off, and the name of the first alias, if any."""
    alias = None
    while get_origin(hint) is Annotated or is_alias(hint):
        if is_alias(hint):
            alias = alias or str(hint.__name__)
            hint = hint.__value__
        else:
            hint = get_args(hint)[0]
    return hint, alias


def _one_dtype(values: Any) -> Any:
    """Returns the polars type of ``values``, a ``Literal``'s or an enum's, when they are all of the first one's
    class and it is one of Python's own, or ``None``. ``Literal["a", "b"]`` is ``String``, and ``Literal["a", 1]``
    is ``None``."""
    first = type(values[0])
    return _DTYPES.get(first) if all(type(value) is first for value in values) else None


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
        return cell_type(kind).write(value)
    # any other object stays one, and a saved run refuses its column (``as_saved``)
    return value


def kept_column(name: str, cells: list[Any], cell: CellType) -> pl.Series:
    """Returns the cells of the column ``name``, each a value of ``cell``, as a table keeps them.

    A column of a model, a dataclass or a ``TypedDict`` is built field by field, each field a column of its own, so
    no dict is made per cell: ``[Money(amount=1.0), None]`` gives the column ``amount`` ``[1.0, None]``, made into
    the struct column ``[{"amount": 1.0}, None]``. Any other column is written cell by cell (``CellType.write``).
    """
    if cell.kind != "fields":
        # a type with nothing in it to write by is written by each value's own class, with no step through the type
        write = cell.write if cell.kind in ("items", "union") else _plain
        return pl.Series(name, [write(one) for one in cells], strict=False)
    typed_dict = cell.typed_dict
    parts = [
        kept_column(field, [_field_of(one, field, typed_dict) for one in cells], field_type)
        for field, field_type in cell.fields.items()
    ]
    column = pl.DataFrame(parts, height=len(cells)).to_struct(name)
    present = pl.Series([one is not None for one in cells])
    return pl.select(pl.when(present).then(column)).to_series().alias(name)


def _field_of(cell: Any, field: str, typed_dict: bool) -> Any:
    """Returns the field ``field`` of ``cell``, or ``None`` for a cell that is. A ``TypedDict``'s field is read by
    its key, and is ``None`` where the dict leaves it out."""
    if cell is None:
        return None
    return cell.get(field) if typed_dict else getattr(cell, field)
