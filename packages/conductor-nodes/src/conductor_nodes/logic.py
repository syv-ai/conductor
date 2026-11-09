"""Logic nodes: ``logic-if-empty``, ``logic-if-equals`` and ``logic-not``.

The two ``if`` nodes route text to one of two outputs and return
``SKIPPED`` on the other, like ``decision`` does for any value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from conductor import SKIPPED, Param, Result
from conductor.widgets import Switch, Textarea, TextWidget

from conductor_nodes.types import stdlib_node


@dataclass(frozen=True)
class Emptiness:
    """What ``IfEmpty.run`` returns: the text on one output, ``SKIPPED`` on the other.

    Both fields share ``choice="emptiness"``, so exactly one is produced.
    ``Equality`` is the same shape for ``IfEquals``.
    """

    not_empty: Annotated[str, Result(title="Not empty", choice="emptiness")]
    empty: Annotated[str, Result(title="Empty", choice="emptiness")]


@dataclass(frozen=True)
class Equality:
    """What ``IfEquals.run`` returns: ``a`` on the output that matched, ``SKIPPED`` on the other.

    Both fields share ``choice="equality"``, so exactly one is produced.
    """

    equal: Annotated[str, Result(title="Equal", choice="equality")]
    not_equal: Annotated[str, Result(title="Not equal", choice="equality")]


@stdlib_node("logic-if-empty", title="If Empty", description="Routes text based on whether it is empty (after strip)", category="control")
def if_empty(text: Annotated[str, Param(title="Text", widget=Textarea())]
) -> Emptiness:
    if text.strip():
        return Emptiness(not_empty=text, empty=SKIPPED)
    return Emptiness(not_empty=SKIPPED, empty=text)


@stdlib_node("logic-if-equals", title="If Equals", description="Routes based on whether two strings are equal", category="control")
def if_equals(
    a: Annotated[str, Param(title="A", widget=TextWidget())],
    b: Annotated[str, Param(title="B", widget=TextWidget())],
    case_sensitive: Annotated[bool, Param(title="Case sensitive", widget=Switch())] = True,
) -> Equality:
    left = a if case_sensitive else a.lower()
    right = b if case_sensitive else b.lower()
    if left == right:
        return Equality(equal=a, not_equal=SKIPPED)
    return Equality(equal=SKIPPED, not_equal=a)


@stdlib_node("logic-not", title="Not", description="Logical negation of a boolean", category="logic")
def not_node(value: Annotated[bool, Param(title="Value", widget=Switch())]
) -> Annotated[bool, Result(title="Negated")]:
    return (not value)


#: Every node this module defines, in palette order.
NODES = (if_empty, if_equals, not_node)
