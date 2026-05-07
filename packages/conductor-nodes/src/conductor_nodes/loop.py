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
        # output_1 = Item from source 0 (the only Item with one source wired,
        # the first source's element under parallel-zip with multiple sources).
        Annotated[object, Output(label="Item")],
        # output_2 = Index — kept here for backward compat with single-source
        # flows that wired ``output_2`` expecting the iteration index.
        Annotated[int, Output(label="Index")],
        # output_3..output_5 = additional Item slots, one per extra source
        # in parallel-zip mode. Hidden by the frontend until the
        # corresponding source is wired into ``items``.
        Annotated[object, Output(label="Item-2")],
        Annotated[object, Output(label="Item-3")],
        Annotated[object, Output(label="Item-4")],
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
