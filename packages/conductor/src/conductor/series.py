"""``Series[X]`` — many values of one type, on an ``Index``.

A node is written for one value. When a series reaches an input declared
with a scalar type, the engine runs the node once per row ("iteration") and
its outputs become series on the same index. When a node declares
``Series[X]`` it receives the whole series at once — a reduction. The
declared type decides; there is no flag on the wire::

    s = Series[Text](Index("docs"), ["a", "b", "c"])
    list(s)     # ["a", "b", "c"]
    s.rows      # ((0,), (1,), (2,))
    s.index     # Index("docs")

The **index** says where the rows came from, and it is what the engine
compares — never lengths: two series may feed one node only if they share
an index. A child index, ``Index("lines", parent=Index("docs"))``, is made
when a node opens a nested value one level. Each row on it is a path,
``(i, j)`` for the j-th row born under parent row ``(i,)``, so a row knows
its parent by construction: a value on the parent index broadcasts down to
the child, and a reduction on the child collapses back up to the parent.

A series may be **sparse** — carry only some of its index's rows — which is
what a branching node produces when it iterates. There is no ``NA``.

``Series[Series[X]]`` is refused: nesting travels as one value on one
field and is opened one level at a time by a node.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar
from uuid import uuid4

from pydantic_core import core_schema

from conductor.dtype import DType
from conductor.dtype_ref import description_of
from conductor.model import ConductorModel

#: The element type, for the ``Sequence`` protocol: ``Series[Text]`` yields ``Text``.
T = TypeVar("T")

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

#: A row's position on its index, as a path: ``(i,)`` on a root index,
#: ``(i, j)`` for the j-th row born under parent row ``(i,)``.
Row = tuple[int, ...]

class Index(ConductorModel):
    """Where a series' rows come from — the identity two series must share to align.

    An index is an ``id`` and, for a child index, the ``parent`` it was
    opened from. Nothing else: no length, no row table. Equality and hashing
    use the ``id`` alone, so an index can be named at compile time, before
    any row exists, and rows are added to it later::

        docs = Index("docs")                     # a root
        lines = Index("lines", parent=docs)      # one level down
        lines.depth                              # 2
        Index.fresh()                            # a new root with a unique id

    Two ``Index.fresh()`` results never align, which is the point: values
    born separately have no rows in common.
    """

    id: str
    #: The index this one was unfolded from, or ``None`` for a root.
    parent: Index | None = None

    def __init__(self, id: str, parent: Index | None = None) -> None:
        """Positional, as an index reads: ``Index("lines", parent=docs)``."""
        super().__init__(id=id, parent=parent)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Index) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def fresh(cls) -> "Index":
        """A new root index with a unique id."""
        return cls(uuid4().hex)

    @property
    def depth(self) -> int:
        """The length of a row path on this index: 1 on a root, 2 on its child."""
        return 1 if self.parent is None else self.parent.depth + 1


class Series(DType, Sequence[T]):
    """Many values of one type on one index, read as a sequence of the values.

    ``Series[Text]`` is a real subclass created on first use, not a
    ``typing`` alias, so it can carry ``element = Text`` as a class
    attribute: ``describe()`` can nest its element and ``DType.accepts``
    can recognise a series by ``source.element is not None``. Declaring an
    input as ``Series[X]`` means "give me the whole series"; the class
    itself is never a wire type, only its parameterisations are.

    ``rows`` may be left out on a root index, in which case the series is
    dense: ``(0,), (1,), ...``. On a child index the rows must be given,
    since only the node that produced them knows which parent each came
    from. ``rows`` and ``values`` always have the same length.
    """

    id = "series"
    title = "Series"

    __slots__ = ("index", "rows", "values")

    #: One subclass per element type, built once, so ``Series[Text] is Series[Text]``.
    _parameterised: ClassVar[dict[Any, type["Series[Any]"]]] = {}

    def __class_getitem__(cls, item: Any) -> type["Series[Any]"]:
        if getattr(item, "element", None) is not None:
            raise TypeError(
                "Series[Series[...]] does not exist — a nested structure is a "
                "Table cell, opened one level by a node"
            )
        made = Series._parameterised.get(item)
        if made is None:
            made = type(
                f"Series[{getattr(item, '__name__', item)}]",
                (cls,),
                {"element": item, "__slots__": ()},
            )
            Series._parameterised[item] = made
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
        """``{"id": "series", "of": <the element's description>}``.

        Nested rather than a string such as ``"series[text]"``, so nothing
        downstream has to parse it. Bare ``Series`` raises: a series whose
        element type nobody declared is a declaration bug.
        """
        if cls.element is None:
            raise ValueError(
                "Series is unparameterised — declare Series[X], never bare Series"
            )
        return {"id": cls.id, "of": description_of(cls.element)}

    # -- the one question, from the series side ---------------------------

    @classmethod
    def accepts(cls, source: Any) -> bool:
        """May ``source`` land on a ``Series[X]`` input?

        Yes for a series of something ``X`` accepts (received whole) and
        for a scalar ``X`` accepts (gathered into a series). Either way the
        element type decides.
        """
        if cls.element is None:
            raise TypeError("an input must declare Series[X], never bare Series")
        inner = source.element if source.element is not None else source
        return cls.element.accepts(inner)

    # -- pydantic ------------------------------------------------------------

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: "GetCoreSchemaHandler"
    ) -> core_schema.CoreSchema:
        """Validate the values through the element type; keep index and rows.

        The wire form — ``{index, rows, values}`` out, and a series, that
        form or a plain list in — is the codec's (``conductor.codec``);
        this schema only points pydantic at it, with the values written and
        read through the element type's own schema. Imported here rather
        than at the top because the codec builds series.
        """
        from conductor.codec import SERIES_WIRE_SCHEMA, series_from_wire, series_to_wire

        element = getattr(source_type, "element", None)
        element_schema = (
            handler.generate_schema(element)
            if element is not None
            else core_schema.any_schema()
        )

        return core_schema.no_info_wrap_validator_function(
            series_from_wire,
            core_schema.list_schema(element_schema),
            serialization=core_schema.plain_serializer_function_ser_schema(
                series_to_wire,
                # Values serialise through the element type's own schema.
                return_schema=core_schema.typed_dict_schema({
                    "index": core_schema.typed_dict_field(core_schema.any_schema()),
                    "rows": core_schema.typed_dict_field(core_schema.any_schema()),
                    "values": core_schema.typed_dict_field(core_schema.list_schema(element_schema)),
                }),
                when_used="json",
            ),
            metadata={"pydantic_js_functions": [lambda _s, _h: SERIES_WIRE_SCHEMA]},
        )
