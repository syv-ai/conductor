"""Control-flow / logic nodes that produce the SKIPPED sentinel to branch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor._sentinel import SKIPPED
from conductor.types import NodeCategory
from conductor.widgets import Checkbox, Output, Text, Textarea

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register every logic node on the supplied registry."""

    @registry.node(
        "logic-if-empty", version=1, name="If Empty",
        description="Routes text based on whether it is empty (after strip)",
        category=NodeCategory.CONTROL,
    )
    def if_empty(
        text: Annotated[str, Textarea(label="Text")],
    ) -> tuple[
        Annotated[str, Output(label="Not empty")],
        Annotated[str, Output(label="Empty")],
    ]:
        if text.strip():
            return (text, SKIPPED)
        return (SKIPPED, text)

    @registry.node(
        "logic-if-equals", version=1, name="If Equals",
        description="Routes based on whether two strings are equal",
        category=NodeCategory.CONTROL,
    )
    def if_equals(
        a: Annotated[str, Text(label="A")],
        b: Annotated[str, Text(label="B")],
        case_sensitive: Annotated[bool, Checkbox(label="Case sensitive")] = True,
    ) -> tuple[
        Annotated[str, Output(label="Equal")],
        Annotated[str, Output(label="Not equal")],
    ]:
        left = a if case_sensitive else a.lower()
        right = b if case_sensitive else b.lower()
        if left == right:
            return (a, SKIPPED)
        return (SKIPPED, a)

    @registry.node(
        "logic-not", version=1, name="Not",
        description="Logical negation of a boolean",
    )
    def not_(
        value: Annotated[bool, Checkbox(label="Value")],
    ) -> Annotated[bool, Output(label="Negated")]:
        return not value
