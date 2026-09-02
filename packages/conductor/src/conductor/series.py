"""``Series[X]`` — many values of one dtype, on an index — and the only collection.

A node is a scalar function. When a series reaches a scalar input, the
node is *lifted*: it runs once per row and its outputs become series on
the same index. When a node wants the whole series it declares
``Series[X]`` and receives it whole — that is a reduction. The type
says which; nothing on a wire does.

The **index** is where a series' rows came from, and it is what the engine
reasons about rather than lengths. Two series align when they share
one. A child index, made by unfolding a ``Table`` cell, has a parent, and
every row on it is a **path** — ``(i, j)``, the j-th child of parent row
``i`` — so a parent-index series broadcasts down to it and a reduction on
it collapses back up, and neither needs a table beside the index to say
which row came from which. A series may be *sparse* — cover a subset of
its index's rows — which is what a lifted decision produces; there
is no ``NA``.

``Series[Series[X]]`` does not exist. Depth is a ``Table`` cell opened one
level by a node.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from pydantic_core import core_schema

from conductor.dtype import DType, description_of

#: Ordinary generics for the container protocol — ``Series[Text]`` types its
#: values. There is no wire-level type variable: an unconstrained input is
#: declared ``Any`` and typed from what arrives.
T = TypeVar("T")

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

#: A row's address on its index: ``(i,)`` on a root, ``(i, j)`` for the j-th
#: row under parent row ``i``. A path, so lineage is in the row itself.
Row = tuple[int, ...]

__all__ = ["Index", "Row", "Series"]


@dataclass(frozen=True, slots=True, eq=False)
class Index:
    """Where a series' rows come from — the identity two series align on.

    A root is minted by ``Index.fresh()`` wherever many values are born with
    no parent: a documents node's files, a gathered pile, a sheet's rows. A
    child is minted where a node unfolds a ``Table`` cell one level. Compile
    reads them to say which index a node is lifted on and whether two inputs
    align; the ledger births rows on them and seals them; ``Row`` is an
    address on one and ``Series`` carries the one its values sit on.

    Not a length and not a table of lineage beside the rows.

    Compile compares indexes, never lengths: it knows which index a series
    is on and never how many rows it will have. ``id`` is that identity —
    a root is born where a series is born without a parent (a documents
    node's files, a gathered pile, a sheet's rows), and a child is born
    when a node unfolds a ``Table`` cell.

    An index is identity and lineage and nothing else: its ``id``, and
    the index it was unfolded from. Rows live on the series, as paths — a
    child row ``(i, j)`` names its parent row ``(i,)`` by construction — so
    compile can name an index before a single row exists and the
    engine births rows on that same index. Equality and hashing
    read the ``id`` alone; ``parent`` is carried for the hierarchy.
    """

    id: str
    #: The index this one was unfolded from, or ``None`` for a root.
    parent: "Index | None" = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Index) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def fresh(cls) -> "Index":
        """A new root. Two fresh indexes never align — correctly."""
        return cls(uuid4().hex)

    @property
    def depth(self) -> int:
        """How long a row's path is on this index: 1 on a root."""
        return 1 if self.parent is None else self.parent.depth + 1


#: One parameterised subclass per element type, built once and reused, so
#: ``Series[Text] is Series[Text]`` and an annotation compares by identity.
_PARAMETERISED: dict[Any, type["Series[Any]"]] = {}


class Series(DType, Sequence[T]):
    """Many values of one dtype, on an index.

    ``Series[Text]`` is a **real subclass**, not a ``typing`` alias. That is
    "a ``DType`` is a real Python type" taken literally, and it is what
    lets ``Series[Text].describe()`` see its own element: attribute access on
    a ``_GenericAlias`` forwards to the origin, so a classmethod reading
    ``get_args(cls)`` there gets nothing back. It also means
    ``source.element is not None`` answers "is this a series?" without
    unwrapping anything, which is what ``DType.accepts`` asks.

    A ``Series[X]`` input — a series received whole — is a reduction's
    declaration: receive the whole series, of whatever arrives.
    Compile types it from the wire and the roster carries the typed
    series; the class itself is never asked ``accepts``.
    """

    id = "series"
    title = "Serie"

    __slots__ = ("index", "rows", "values")

    def __class_getitem__(cls, item: Any) -> type["Series[Any]"]:
        if getattr(item, "element", None) is not None:
            raise TypeError(
                "Series[Series[...]] does not exist — a nested structure is a "
                "Table cell, opened one level by a node"
            )
        made = _PARAMETERISED.get(item)
        if made is None:
            made = type(
                f"Series[{getattr(item, '__name__', item)}]",
                (cls,),
                {"element": item, "__slots__": ()},
            )
            _PARAMETERISED[item] = made
        return made

    def __init__(
        self,
        index: Index,
        values: Sequence[T],
        rows: Sequence[Row] | None = None,
    ) -> None:
        self.index = index
        self.values: tuple[T, ...] = tuple(values)
        #: One path per value. Left out, the series is dense on a root:
        #: ``(0,), (1,), ...``. A child index has no such default — its rows
        #: are born under parents, and the engine says which.
        self.rows: tuple[Row, ...] = (
            tuple((i,) for i in range(len(self.values))) if rows is None else tuple(rows)
        )
        if len(self.rows) != len(self.values):
            raise ValueError(
                f"rows ({len(self.rows)}) and values ({len(self.values)}) differ in length"
            )
        if not all(isinstance(row, tuple) and all(isinstance(i, int) for i in row) for row in self.rows):
            raise TypeError("a row is a path of ints — (i,) on a root, (i, j) on its child")

    # -- read as a sequence of values ------------------------------------

    def __getitem__(self, position: int) -> T:  # type: ignore[override]
        return self.values[position]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[T]:
        return iter(self.values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Series):
            return (self.index, self.rows, self.values) == (
                other.index,
                other.rows,
                other.values,
            )
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return list(self) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Series({list(self.values)!r})"

    # -- the wire ----------------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Nested rather than a string like ``"series[text]"``: a string
        would have to be parsed, and nothing downstream parses a type.

        An unparameterised ``Series`` raises. A collection whose element
        type nobody declared cannot be type-checked or rendered, so it is a
        declaration bug and says so here rather than serialising
        ``{"of": None}`` for a frontend to trip over.
        """
        if cls.element is None:
            raise ValueError(
                "Series is unparameterised — declare Series[X], never bare Series"
            )
        return {"id": cls.id, "of": description_of(cls.element)}

    # -- the one question, from the series side ---------------------------

    @classmethod
    def accepts(cls, source: Any) -> bool:
        """A ``Series[X]`` input takes a whole series of ``X`` — or gathers
        scalars of ``X`` into one. Either way the element decides."""
        if cls.element is None:
            raise TypeError("an input must declare Series[X], never bare Series")
        inner = source.element if source.element is not None else source
        return cls.element.accepts(inner)

    # -- pydantic ------------------------------------------------------------

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: "GetCoreSchemaHandler"
    ) -> core_schema.CoreSchema:
        """Validate the elements, then put the index and rows back.

        The inner list schema does the real work, so a bad element fails
        with pydantic's own message and the field's name.
        """
        element = getattr(source_type, "element", None)
        element_schema = (
            handler.generate_schema(element)
            if element is not None
            else core_schema.any_schema()
        )

        def validate(value: Any, inner: Any) -> "Series[Any]":
            if isinstance(value, Series):
                return Series(value.index, inner(list(value.values)), rows=value.rows)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return Series(Index.fresh(), inner(list(value)))
            # Wrapping a scalar into a one-element series here would turn a
            # mistake into a plausible-looking success.
            raise TypeError(f"expected a series, got {type(value).__name__}")

        return core_schema.no_info_wrap_validator_function(
            validate,
            core_schema.list_schema(element_schema),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _wire,
                # The values travel through the element's own schema, so a
                # series of files dumps each file's wire form.
                return_schema=core_schema.typed_dict_schema({
                    "index": core_schema.typed_dict_field(core_schema.any_schema()),
                    "rows": core_schema.typed_dict_field(core_schema.any_schema()),
                    "values": core_schema.typed_dict_field(core_schema.list_schema(element_schema)),
                }),
                when_used="json",
            ),
            metadata={"pydantic_js_functions": [lambda _s, _h: _SERIES_WIRE_SCHEMA]},
        )


def _wire(series: "Series[Any]") -> dict[str, Any]:
    """A series on the wire: its index whole, its rows, its values.

    The index travels as lineage — id, parent, parent rows — never as a
    length, so a consumer can group series that share one into a table and
    can tell a child's rows from its parent's.
    """
    return {
        "index": _index_wire(series.index),
        "rows": [list(row) for row in series.rows],
        "values": list(series.values),
    }


def _index_wire(index: Index) -> dict[str, Any]:
    return {
        "id": index.id,
        "parent": None if index.parent is None else _index_wire(index.parent),
    }


#: An index on the wire. ``parent`` is the same shape again, as deep as the
#: lineage goes; a JSON schema cannot name itself from inside a pydantic
#: schema function, so the parent is published as an object and described.
_INDEX_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "parent": {
            "anyOf": [{"type": "object"}, {"type": "null"}],
            "description": "The parent index, in this same shape; null for a root.",
        },
    },
    "required": ["id", "parent"],
}
_SERIES_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": _INDEX_WIRE_SCHEMA,
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "values": {"type": "array"},
    },
    "required": ["index", "rows", "values"],
}
