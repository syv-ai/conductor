"""``Ref`` — the address of one field on one node.

Everything that moves between nodes is addressed by a ``Ref``: an edge
says which output feeds which input, a caller's answers are keyed by the
input they fill, a problem points at the field it is about. The address
is ``"<node id>.<field name>"``::

    >>> Ref("summary", "text")
    'summary.text'
    >>> Ref("summary.text").node_id, Ref("summary.text").field
    ('summary', 'text')

A ``Ref`` is a ``str`` subclass, so it works as-is wherever a string key
does — a dict key, a JSON object key, a pydantic field name — with nothing
to encode on the way out or decode on the way in. ``node_id`` and
``field`` are the only place the string is split.

The node part is the placement's **id**, never its title. Titles are for
people and may be edited; an id is minted when the node is placed and
never changes. That is the reason an address deserves a type: renaming a
node does not move anything that points at it.

An address has exactly two parts. A field name may itself contain a dot —
that is how a flow embedded as a node keeps its own addresses as field
names — but a node id never does, so the first dot always separates the
two::

    >>> Ref("inner", "total.value").field
    'total.value'
    >>> Ref("inner.total.value") == Ref("inner", Ref("total", "value"))
    True

There is deliberately no third part and no path syntax (``"rows.0.name"``):
a nested value travels whole on one field, and opening it is a node's job.
"""

from __future__ import annotations

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class Ref(str):
    """The address ``"<node id>.<field name>"`` of one field on one node.

    Build one from its two parts, or read one back from the string form;
    both give the same value::

        >>> Ref("summary", "text") == Ref("summary.text") == "summary.text"
        True
    """

    __slots__ = ()

    def __new__(cls, node_id: str, field: str | None = None) -> Ref:
        """``Ref(node_id, field)`` composes an address; ``Ref("node.field")`` reads one.

        Refused: a string with no dot or an empty half (not an address), and
        a node id containing a dot (the first dot would no longer separate
        node from field).
        """
        if field is not None and "." in node_id:
            raise ValueError(f"placement {node_id!r} contains '.'; an address reads 'node.field'")
        address = node_id if field is None else f"{node_id}.{field}"
        head, dot, tail = address.partition(".")
        if not dot or not head or not tail:
            raise ValueError(f"{address!r} is not an address; one reads 'node.field'")
        return super().__new__(cls, address)

    @property
    def node_id(self) -> str:
        """The node part — everything before the first dot."""
        return self.partition(".")[0]

    @property
    def field(self) -> str:
        """The field part — everything after the first dot, dots included."""
        return self.partition(".")[2]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Let pydantic validate a ``Ref`` field into a ``Ref``, not a bare ``str``.

        pydantic validates a ``str`` subclass as plain ``str`` unless told
        otherwise; this validates the string and wraps it, so
        ``Model(ref="a.b").ref.node_id`` works and an invalid address is a
        validation error. It dumps as the plain string.
        """
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
