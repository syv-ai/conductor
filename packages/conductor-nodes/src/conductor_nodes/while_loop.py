"""While-loop markers — canonical node ids the ``WHILE`` compound discovers.

See :mod:`conductor.compound.while_loop` for the runtime behavior. To
actually run a while-loop you must pass the ``WHILE`` compound type to
``compile()``:

    from conductor import compile
    from conductor.compound.while_loop import WHILE
    from conductor_nodes import while_loop

    while_loop.register(registry)
    compiled = compile(nodes, edges, registry, compound_types=[WHILE])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor.types import NodeCategory
from conductor.widgets import Checkbox, CodeEditor, Number, Output, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register the while-start / while-end marker nodes."""

    @registry.node(
        "while-start", version=1, name="While (Start)",
        description=(
            "Loops while a CEL condition is true. The condition has access to "
            "`iteration` (the current 1-based count) and `last` (the body's "
            "most recent return value). Must be paired with while-end."
        ),
        category=NodeCategory.CONTROL,
    )
    def while_start(
        condition: Annotated[str, CodeEditor(label="Condition (CEL)")] = "iteration < 5",
        max_iterations: Annotated[int, Number(label="Max iterations", integer_only=True)] = 1000,
        negate: Annotated[bool, Checkbox(label="Until (negate condition)")] = False,
    ) -> tuple[
        Annotated[int, Output(label="Iteration")],
        Annotated[object, Output(label="Last")],
    ]:
        raise NotImplementedError("Handled by the WHILE compound node")

    @registry.node(
        "while-end", version=1, name="While (End)",
        description="Captures the body's last return value as the loop's result.",
        category=NodeCategory.CONTROL,
    )
    def while_end(
        item: Annotated[object, Text(label="Item")],
    ) -> Annotated[object, Output(label="Last value")]:
        raise NotImplementedError("Handled by the WHILE compound node")
