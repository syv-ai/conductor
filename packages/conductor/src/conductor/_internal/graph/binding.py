"""Where an input's value comes from, as compile reads it.

A graph record keeps its edges in one list beside the nodes, and each node's
typed-in values in ``values`` and ``defaults``. Loading the record
(``graph_records.load_record``) gathers all three onto the inputs they are
for, one binding per input: an edge from other nodes' outputs (``From``) or a
value (``Static``)::

    {
        "text":  From("reader.text"),   # an edge in
        "limit": Static(200),           # a value or a default
        # "language" absent: the input's declared default applies
    }

Compile and the engine read these; nothing outside the library writes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from conductor._internal.ref import Ref


@dataclass(frozen=True, init=False)
class From:
    """The input's value arrives over one or more edges from other nodes' outputs.

    ``refs`` names those outputs in the order their nodes appear in the
    record. A scalar input has one ref; a ``Series[X]`` input may have
    several, gathered into one list.
    """

    refs: tuple[Ref, ...]

    def __init__(self, *refs: Ref | str) -> None:
        """``From("a.result")``, or several: ``From("a.result", "b.result")``."""
        object.__setattr__(self, "refs", tuple(Ref(ref) for ref in refs))


@dataclass(frozen=True)
class Static:
    """The input's value is one typed into the record, the same on every run.

    Compile validates ``value`` against the input's declared type.
    """

    value: Any


Binding: TypeAlias = From | Static


def static_values(bindings: Mapping[str, Binding]) -> dict[str, Any]:
    """The values as a plain dict, for the engine.

    Only ``Static`` contributes. An absent binding is left absent so the
    node's own default applies.
    """
    return {
        name: binding.value
        for name, binding in bindings.items()
        if isinstance(binding, Static)
    }
