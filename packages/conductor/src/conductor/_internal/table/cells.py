"""Which column types a table can keep, and how it keeps a cell of one, as polars holds it.

Polars has a column type for a few of Python's own classes, and for lists and structs of them. So a table keeps a
cell as it is only if its type is one of these, at any depth:

- one of Python's own classes polars has a type for (``CELLS``), like ``int``, ``str`` or ``date``, or a subclass
  of one;
- a ``Literal`` or an enum whose values are all of one of those classes;
- ``list[C]``, or a tuple whose items are all ``C``, for such a ``C``;
- a model, a dataclass or a ``TypedDict`` whose fields are each such a type, that declares every key a value has:
  no ``NotRequired`` field, and no config that allows extra keys;
- ``C | None`` for such a ``C``;
- ``Any``, which says nothing of its values, as a column a hook gives a type it doesn't know yet does.

A table's row declaring any other column type is refused when its node is declared (``table_cell_not_kept``), and
so is a frame returned for a ``Series`` of such rows. A union of two types, a dict, a tuple of different items or a
value that may leave out a key would come back from polars changed.

A kept cell is written by the fields its column's type declares, as they are, with no host serialization rule: a
``field_serializer`` is not applied, a ``computed_field`` is not added, a field marked ``exclude=True`` is kept,
and an instance of a subclass is written by the declared class's fields. A model or a dataclass is the dict of its
fields, an enum member its value, and a tuple a list. Say a row type has a field ``money: Money | None``: the cells
``[Money(amount=1.0), None]`` are kept as the struct column ``[{"amount": 1.0}, None]``.

``CellType`` is the one walk over a column's type, which writing a cell, the polars type a column is kept as, and
the rule above all read.
"""

from __future__ import annotations

import functools
from enum import Enum
from types import NoneType
from typing import Annotated, Any, Literal, NotRequired, get_args, get_origin

import polars as pl

from conductor._internal.types import (
    Type,
    allows_extra,
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

#: Python's own classes polars has a column type for.
_KEPT_CLASSES = tuple(CELLS)


def cell_type(hint: Any) -> CellType:
    """Returns the ``CellType`` of ``hint``, made once per hint. A hint that can't be hashed, like one with a dict in
    its ``Annotated``, gets a new one each time."""
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
    # a set is refused when its node is declared, for having no order, so it is taken as its items here
    if origin in (list, tuple, set, frozenset) and get_args(bare):
        return "items"
    return "value"


class CellType:
    """A column's type, as a table keeps its cells.

    It is one of five kinds, read from the hint with a type alias, ``Annotated`` and ``None`` taken off:

    - ``"fields"``: a model, a dataclass or a ``TypedDict``, kept as a struct of its declared fields;
    - ``"items"``: a list or a tuple, kept as a list of its items' type;
    - ``"named"``: an enum or a ``Literal``, kept as its values;
    - ``"value"``: anything else, like ``int``, kept as it is if polars has a type for it;
    - ``"union"``: two types or more, like ``Cat | Dog``, which a table doesn't keep.

    What a type holds (its fields, its items) is made the first time it is asked for, so a type that holds itself,
    like ``type Deep = list[Deep]`` or a tree of nodes, is walked only as deep as a value goes.

    Attributes:
        alias: The name of the type alias the hint was written as, such as ``JsonValue``, or ``None``.
        bare: The one type left once the alias, ``Annotated`` and ``None`` are taken off, or the union.
        kind: What the type is, as above.
    """

    def __init__(self, hint: Any) -> None:
        hint, self.alias = _stripped(hint)
        members = [_stripped(member) for member in union_members(hint)]
        kept = [(member, alias) for member, alias in members if member is not NoneType]
        if len(kept) == 1:
            # ``Alias | None`` is the alias, by its name
            hint, alias = kept[0]
            self.alias = self.alias or alias
        self.bare = hint
        self.kind = _kind_of(self.bare, len(kept))

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
    def items(self) -> tuple[CellType, ...]:
        """The ``CellType`` of a list's item, or of each of a tuple's items (one, for ``tuple[C, ...]``)."""
        arguments = [argument for argument in get_args(self.bare) if argument is not Ellipsis]
        return tuple(cell_type(argument) for argument in arguments) if self.kind == "items" else ()

    @functools.cached_property
    def typed_dict(self) -> bool:
        """Whether the type is a ``TypedDict``, whose value is a dict."""
        return self.kind == "fields" and is_typed_dict(self.bare)

    @functools.cached_property
    def refusal(self) -> tuple[str, str] | None:
        """Why a table can't keep a cell of the type as it is, as the type it is about and the reason, or ``None``
        if it can. It is the first type, at any depth, that the module's rule doesn't take: ``("dict[str, int]",
        "a table has no column type for it")``."""
        return self._refusal_within(set())

    def _refusal_within(self, walked: set[int]) -> tuple[str, str] | None:
        """``refusal``, walking only the types not in ``walked`` yet, so a type that holds itself ends."""
        walked.add(id(self))
        if (why := self._own_refusal()) is not None:
            return self.name, why
        parts = [*self.fields.values(), *self.items]
        return next(
            (
                found
                for part in parts
                if id(part) not in walked and (found := part._refusal_within(walked)) is not None
            ),
            None,
        )

    @property
    def name(self) -> str:
        """The type as a problem names it: its alias's name, a model's, dataclass's or ``TypedDict``'s class name,
        or its text, such as ``dict[str, int]``."""
        if self.alias is not None:
            return self.alias
        if self.kind == "fields":
            return str((get_origin(self.bare) or self.bare).__qualname__)
        return Type(self.bare).text

    def _own_refusal(self) -> str | None:
        """Why this type itself isn't one a table keeps, leaving out what it holds, or ``None``."""
        if self.kind == "union":
            return "a value of a union may be of either type"
        if self.kind == "fields":
            left_out = [
                name for name, hint in self._declared.items() if get_origin(hint) is NotRequired
            ]
            if left_out:
                return f"a value may leave out its key '{left_out[0]}'"
            if allows_extra(get_origin(self.bare) or self.bare):
                return "a value may have keys it doesn't declare"
            return None
        if self.kind == "items":
            return (
                None if len({id(item) for item in self.items}) == 1 else "its items differ in type"
            )
        if self.kind == "named":
            return None if self.dtype is not None else "its values are of different classes"
        # Any says nothing of its values, as a column a hook gives a type it doesn't know yet, so it is kept as
        # polars reads it
        if self.bare is Any or (
            isinstance(self.bare, type) and issubclass(self.bare, _KEPT_CLASSES)
        ):
            return None
        return "a table has no column type for it"

    def write(self, value: Any) -> Any:
        """Returns ``value``, a value of a type a table keeps, as a table keeps it.

        A model or a dataclass is the dict of the fields the type declares, each written by its own type, an enum
        member is its value, a tuple is a list, and anything else is as it is. A ``None`` stays ``None``.
        """
        if value is None:
            return None
        if self.typed_dict:
            return {name: field.write(value[name]) for name, field in self.fields.items()}
        if self.kind == "fields":
            return {name: field.write(getattr(value, name)) for name, field in self.fields.items()}
        if self.items:
            return [self.items[0].write(one) for one in value]
        if self.bare is Any:
            return _plain(value)
        return value.value if isinstance(value, Enum) else value

    @functools.cached_property
    def dtype(self) -> Any:
        """The polars type a table keeps a column of the type as, or ``None`` where the cells decide it.

        ``Money | None``, with ``Money`` declaring ``amount: float``, is ``Struct({"amount": Float64})``,
        ``list[int]`` and ``tuple[int, ...]`` are ``List(Int64)``, and a ``Literal`` or an enum whose values are all
        texts ``String``. A ``datetime`` with a time zone and a ``Decimal`` of a scale other than 0 differ from
        their class's own polars type, and a type met again inside itself, like a tree of nodes, is ``None``.
        """
        # a type met again inside itself, while its own polars type is worked out, gives None
        self.__dict__["dtype"] = None
        if self.kind == "fields":
            fields = {name: field.dtype for name, field in self.fields.items()}
            return None if None in fields.values() else pl.Struct(fields)
        if self.items:
            return None if self.items[0].dtype is None else pl.List(self.items[0].dtype)
        if self.kind == "named":
            origin = get_origin(self.bare) or self.bare
            return _one_dtype(
                get_args(self.bare) if origin is Literal else [member.value for member in origin]
            )
        return _DTYPES.get(self.bare) if self.kind == "value" else None


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
    """Writes a value of ``Any`` by its own class: a model or dataclass as the dict of its fields, an enum member as
    its value, a list or tuple as a list, and a dict with each value so."""
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
    """Returns the cells of the column ``name``, each a value of ``cell``, a type a table keeps, as a table keeps
    them.

    A column of a model, a dataclass or a ``TypedDict`` is built field by field, each field a column of its own, so
    no dict is made per cell: ``[Money(amount=1.0), None]`` gives the column ``amount`` ``[1.0, None]``, made into
    the struct column ``[{"amount": 1.0}, None]``. Any other column is written cell by cell (``CellType.write``).
    """
    if cell.kind == "value" and cell.bare is not Any:
        return pl.Series(name, cells, strict=False)
    if cell.kind != "fields":
        return pl.Series(name, [cell.write(one) for one in cells], strict=False)
    typed_dict = cell.typed_dict
    parts = [
        kept_column(field, [_field_of(one, field, typed_dict) for one in cells], field_type)
        for field, field_type in cell.fields.items()
    ]
    column = pl.DataFrame(parts, height=len(cells)).to_struct(name)
    present = pl.Series([one is not None for one in cells])
    return pl.select(pl.when(present).then(column)).to_series().alias(name)


def _field_of(cell: Any, field: str, typed_dict: bool) -> Any:
    """Returns the field ``field`` of ``cell``, or ``None`` for a cell that is."""
    if cell is None:
        return None
    return cell[field] if typed_dict else getattr(cell, field)
