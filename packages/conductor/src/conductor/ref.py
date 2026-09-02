"""An address: which node, which field.

Everything that moves between nodes is addressed. A ``Ref`` is that
address — one field of one node, an input or an output alike, since both
are named slots on a node. It is what a wire carries, what a caller's
answers are keyed by and what a flow's interface names its fields with,
so that none of them is keyed by a **title**: a title is something a
person wrote, it can be rewritten on a placement, and two placements can
share one.

**The node part is a placement's ``id``, never its name.** An id is minted
when the node is placed — the type and the first free counter, ``currency-1``
— and nothing renames it, in the editor or anywhere else. The name a person
reads sits beside it, as ``GraphNode.title``, and per field as
``FieldContent.title``; an App's own name for a field is a third thing again,
which is why an ``AppField`` carries a ``Ref`` beside its ``name`` rather than
being keyed by it. That split is the whole reason an address is worth a type:
names move, addresses do not. (Short ids — ``sag``, ``godkend`` — appear below
and in tests because they read; nothing mints one.)

**A ``Ref`` is the address, not a rendering of one.** It is a ``str``
subclass, so the one thing a composite key is otherwise bad at — *being
a key* — it is good at: a dict key, a JSON object key, a pydantic field
name and a key in the editor's own maps all take it as it is. There is
no encode step to remember, no second form to store and no second
spelling for a frontend to agree with. ``node_id`` and ``field`` read the
two parts back, and they are the only place in the codebase that splits
the string.

**Two parts, because depth is a node's job.** An address never grows a
third: a nested value travels whole, on one field, and *opening* it is
something a node does — the node that opens it births a child index and
exposes one field per part. So there is nothing below a field to
address and no path to carry. What a host makes of that rule is the
host's: AKA opens a table with «Fold ud».

A field name may itself contain a dot, which is what lets a flow be a
node: its interface names a field ``"sag.value"``, and under the
placement that embeds it that field's address is ``"emb.sag.value"``. The
first dot separates and the rest is the field, so composing is closed and
nesting needs nothing new. A placement id never contains a dot —
``GraphNode`` refuses one.
"""

from __future__ import annotations

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

__all__ = ["Ref"]


class Ref(str):
    """One field on one node."""

    __slots__ = ()

    def __new__(cls, node_id: str, field: str | None = None) -> Ref:
        """``Ref("sag", "value")`` composes an address; ``Ref("sag.value")`` reads one.

        Two arities because there are two questions, not two spellings: an
        author of an address holds the parts, a reader of one holds the
        string. Both produce the same value, so neither is a conversion.
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
        """The placement. A placement id never contains a dot."""
        return self.partition(".")[0]

    @property
    def field(self) -> str:
        """The field on it — everything after the first dot, which is why an
        embedded flow's own address survives as a field name."""
        return self.partition(".")[2]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Validate as the address, come back a ``Ref``, dump as itself.

        Without this pydantic hands back a bare ``str`` and every
        ``ref.node_id`` downstream is an ``AttributeError`` — the same help
        a ``DType`` built on a builtin needs.
        """
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
