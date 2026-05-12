"""Canonical for-each loop markers.

The engine's ``FOR_EACH`` compound node type discovers regions by looking
for nodes whose registered ``base_id`` starts with ``for-each-start`` /
``for-each-end``. These are the canonical implementations.

To use the markers you must pass the ``FOR_EACH`` compound type to
``compile()`` — nothing in this module wires that up for you:

    from conductor import compile
    from conductor.compound.for_each import FOR_EACH
    from conductor_nodes import loop

    loop.register(registry)
    compiled = compile(nodes, edges, registry, compound_types=[FOR_EACH])

Both markers register with ``dynamic_handles=True``: the declared
parameters are templates the host UI can use as starting points, but
the actual input/output handle set is unbounded — wire as many sources
into ``items`` and as many body→end edges as you need. The FOR_EACH
compound parallel-zips all wired ``items`` sources into per-iteration
tuples and transposes per-iteration end inputs into per-slot output
lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor.types import NodeCategory
from conductor.widgets import ConnectionList, Dropdown, Output

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register the for-each-start / for-each-end marker nodes."""

    @registry.node(
        "for-each-start", version=1, name="For Each (Start)",
        description="Iterates over a list of items. Must be paired with for-each-end.",
        category=NodeCategory.CONTROL,
        dynamic_handles=True,
    )
    def for_each_start(
        items: Annotated[list, ConnectionList(label="Items")],
        execution_mode: Annotated[
            str,
            Dropdown(label="Execution", choices=["Sequential", "Parallel"]),
        ] = "Sequential",
    ) -> tuple[
        # Template outputs. The compound runtime emits ``output_1`` =
        # primary Item, ``output_2`` = Index, ``output_3..N`` = Item-2..N
        # for every additional source wired into ``items``. Hosts render
        # the dynamic Item-N handles based on the live edge count.
        Annotated[object, Output(label="Item")],
        Annotated[int, Output(label="Index")],
    ]:
        raise NotImplementedError("Handled by the FOR_EACH compound node")

    @registry.node(
        "for-each-end", version=1, name="For Each (End)",
        description="Collects loop body results into one list per wired source.",
        category=NodeCategory.CONTROL,
        dynamic_handles=True,
    )
    def for_each_end(
        # Single ConnectionList input. Wire as many body→end edges as
        # you need; each ``(source, source_handle)`` pair becomes one
        # collected-output slot. The compound runtime transposes per-
        # iteration tuples into per-slot lists and emits
        # ``output_1..output_N`` accordingly. Legacy per-source target
        # handles (``item``, ``item_2``, …) are still accepted for
        # backward compatibility with flows saved before the
        # ConnectionList switch.
        items: Annotated[dict, ConnectionList(label="Items")] = None,
    ) -> Annotated[list, Output(label="Collected")]:
        raise NotImplementedError("Handled by the FOR_EACH compound node")
