"""Where an input's value comes from.

An input of a node in a graph is bound in one of two ways: by an edge
from another node's output (``From``), or by a value the author set on
it (``Static``, a static for short). A node's ``bindings`` map holds one binding per input the
author touched; an input with no entry takes its declared default::

    {
        "text":  From("reader.text"),   # an edge in
        "limit": Static(200),           # a static
        # "language" absent: the input's declared default applies
    }

Both read as they are written, positionally; stored, they are
``{"refs": [...]}`` and ``{"value": ...}``. A type checker reads ``From(...)``
as ``Any``, so an edge can be written where a value is expected
(``upper.graph_node("b", text=From("a.result"))``): an edge carries what its
source produces, and only compile knows if that fits.

Because one input holds one binding, an edge and a static can never
both claim the same input.

The editor writes both: connecting two handles writes a ``From``,
editing a control writes a ``Static``. Compile reads them: the
dependency map, the engine's maps and the graph's own inputs and outputs
are all derived from the bindings (``topology``, ``graph_interface``); nothing
stores an edge as a record of its own.

A run supplies values for a graph's inputs beside the graph, never inside
it, so a stored graph stays runnable on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Mapping, TypeAlias

from pydantic import Discriminator, Field, Tag

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    # A type checker sees a call of ``From`` as ``Any``, so an edge may stand where a value of the
    # input's type is written; it still knows the class, for an ``isinstance``.
    class From:
        refs: tuple[Ref, ...]

        def __new__(cls, *refs: Ref | str, **stored: Any) -> Any: ...

        def model_dump(self, **options: Any) -> dict[str, Any]: ...
else:
    class From(ConductorModel):
        """The input's value arrives over one or more edges from other nodes' outputs.

        ``refs`` names those outputs in the order the author connected them,
        and that order is the only order there is; an edge with no refs is not
        a binding and raises where it is written. A scalar input has one
        ref; a ``Series[X]`` input may have several, gathered into one list.

        Written by the editor when two handles are connected; read by compile.
        """

        refs: tuple[Ref, ...] = Field(min_length=1)

        def __init__(self, *refs: Ref | str, **stored: Any) -> None:
            """``From("a.result")``, or several: ``From("a.result", "b.result")``. A ref
            is a ``Ref`` or its address as a string. The stored form, ``refs=``,
            is what pydantic passes when it reads a saved graph."""
            if refs and stored:
                raise TypeError("From takes its refs positionally or as refs=, not both")
            super().__init__(**({"refs": refs} if refs else stored))


class Static(ConductorModel):
    """The input's value is one the author set on the node; the graph stores it.

    Fixed in the graph, the same on every run. The compiler validates
    ``value`` against the input's declared type; ``static_values`` hands the
    statics to the engine as one dict.

    Written by the editor when a person edits the control, or by a seed
    when the node is added; read by compile. Not a default: a default
    belongs to the node's declaration and applies when there is no binding
    at all.
    """

    value: Any

    def __init__(self, *value: Any, **stored: Any) -> None:
        """``Static(200)``: the value, as the author wrote it. The stored form,
        ``value=``, is what pydantic passes when it reads a saved graph."""
        if len(value) > 1 or (value and stored):
            raise TypeError("Static takes one value, positionally or as value=")
        super().__init__(**({"value": value[0]} if value else stored))


def _kind_of_binding(value: Any) -> str | None:
    """``From`` or ``Static``, read off what ``value`` holds: the one question a union of the two would otherwise ask by trying both."""
    if isinstance(value, From) or (isinstance(value, Mapping) and "refs" in value):
        return "from"
    if isinstance(value, Static) or (isinstance(value, Mapping) and "value" in value):
        return "static"
    return None


Binding: TypeAlias = Annotated[
    Annotated[From, Tag("from")] | Annotated[Static, Tag("static")],
    Discriminator(_kind_of_binding, custom_error_type="not_a_binding", custom_error_message="A binding is From('node.output') or Static(value)."),
]


def static_values(bindings: Mapping[str, Binding]) -> dict[str, Any]:
    """The statics as a plain dict, for the engine.

    Only ``Static`` contributes. An absent binding is left absent so the
    node's own default applies.
    """
    return {
        name: binding.value
        for name, binding in bindings.items()
        if isinstance(binding, Static)
    }
