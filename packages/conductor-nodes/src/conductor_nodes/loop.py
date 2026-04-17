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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor.types import NodeCategory
from conductor.widgets import ConnectionList, Dropdown, Output, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register the for-each-start / for-each-end marker nodes."""

    @registry.node(
        "for-each-start", version=1, name="For Each (Start)",
        description="Iterates over a list of items. Must be paired with for-each-end.",
        category=NodeCategory.CONTROL,
    )
    def for_each_start(
        items: Annotated[list, ConnectionList(label="Items")],
        execution_mode: Annotated[
            str,
            Dropdown(label="Execution", choices=["Sequential", "Parallel"]),
        ] = "Sequential",
    ) -> tuple[
        Annotated[object, Output(label="Item")],
        Annotated[int, Output(label="Index")],
    ]:
        raise NotImplementedError("Handled by the FOR_EACH compound node")

    @registry.node(
        "for-each-end", version=1, name="For Each (End)",
        description="Collects loop body results into a list",
        category=NodeCategory.CONTROL,
    )
    def for_each_end(
        item: Annotated[object, Text(label="Item")],
    ) -> Annotated[list, Output(label="Collected")]:
        raise NotImplementedError("Handled by the FOR_EACH compound node")
