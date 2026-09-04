"""Logic nodes: ``logic-if-empty``, ``logic-if-equals`` and ``logic-not``.

The two ``if`` nodes route text to one of two outputs and return
``SKIPPED`` on the other, like ``decision`` does for any value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from conductor._sentinel import SKIPPED
from conductor.returns import Result
from conductor.widgets import Switch, Textarea
from conductor.widgets import Text as TextWidget

from conductor_nodes.types import Flag, StdlibNode, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


@dataclass(frozen=True)
class Emptiness:
    """What ``IfEmpty.run`` returns: the text on one output, ``SKIPPED`` on the other.

    Both fields share ``choice="emptiness"``, so exactly one is produced.
    ``Equality`` is the same shape for ``IfEquals``.
    """

    not_empty: Annotated[Text, Result(title="Not empty", choice="emptiness")]
    empty: Annotated[Text, Result(title="Empty", choice="emptiness")]


@dataclass(frozen=True)
class Equality:
    """What ``IfEquals.run`` returns: ``a`` on the output that matched, ``SKIPPED`` on the other.

    Both fields share ``choice="equality"``, so exactly one is produced.
    """

    equal: Annotated[Text, Result(title="Equal", choice="equality")]
    not_equal: Annotated[Text, Result(title="Not equal", choice="equality")]


class IfEmpty(StdlibNode):
    id = "logic-if-empty"
    title = "If Empty"
    description = "Routes text based on whether it is empty (after strip)"
    category = "control"

    def run(
        self, text: Annotated[Text, Textarea(title="Text")]
    ) -> Emptiness:
        if text.strip():
            return Emptiness(not_empty=text, empty=SKIPPED)
        return Emptiness(not_empty=SKIPPED, empty=text)


class IfEquals(StdlibNode):
    id = "logic-if-equals"
    title = "If Equals"
    description = "Routes based on whether two strings are equal"
    category = "control"

    def run(
        self,
        a: Annotated[Text, TextWidget(title="A")],
        b: Annotated[Text, TextWidget(title="B")],
        case_sensitive: Annotated[Flag, Switch(title="Case sensitive")] = Flag(True),
    ) -> Equality:
        left = a if case_sensitive else a.lower()
        right = b if case_sensitive else b.lower()
        if left == right:
            return Equality(equal=a, not_equal=SKIPPED)
        return Equality(equal=SKIPPED, not_equal=a)


class Not(StdlibNode):
    id = "logic-not"
    title = "Not"
    description = "Logical negation of a boolean"
    category = "logic"

    def run(
        self, value: Annotated[Flag, Switch(title="Value")]
    ) -> Annotated[Flag, Result(title="Negated")]:
        return Flag(not value)


NODES = (IfEmpty, IfEquals, Not)


def register(registry: "NodeRegistry") -> None:
    """Register every logic node on the supplied registry."""
    for node_cls in NODES:
        registry.register(node_cls)
