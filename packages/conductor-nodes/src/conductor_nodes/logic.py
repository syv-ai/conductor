"""Logic nodes: ``logic-if-empty``, ``logic-if-equals`` and ``logic-not``.

The two ``if`` nodes route text to one of two outputs and return
``SKIPPED`` on the other, like ``decision`` does for any value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from conductor import SKIPPED, Result

from conductor_nodes.types import Flag, Line, LongText, stdlib_node


@dataclass(frozen=True)
class Emptiness:
    """What ``IfEmpty.run`` returns: the text on one output, ``SKIPPED`` on the other.

    Both fields share ``exclusive_group="emptiness"``, so exactly one is produced.
    ``Equality`` is the same shape for ``IfEquals``.
    """

    not_empty: Annotated[str, Result(title="Not empty", exclusive_group="emptiness")]
    empty: Annotated[str, Result(title="Empty", exclusive_group="emptiness")]


@dataclass(frozen=True)
class Equality:
    """What ``IfEquals.run`` returns: ``a`` on the output that matched, ``SKIPPED`` on the other.

    Both fields share ``exclusive_group="equality"``, so exactly one is produced.
    """

    equal: Annotated[str, Result(title="Equal", exclusive_group="equality")]
    not_equal: Annotated[str, Result(title="Not equal", exclusive_group="equality")]


@stdlib_node("logic-if-empty", title="If Empty", description="Routes text based on whether it is empty (after strip)", category="control")
def if_empty(text: LongText) -> Emptiness:
    if text.strip():
        return Emptiness(not_empty=text, empty=SKIPPED)
    return Emptiness(not_empty=SKIPPED, empty=text)


@stdlib_node("logic-if-equals", title="If Equals", description="Routes based on whether two strings are equal", category="control")
def if_equals(a: Line, b: Line, case_sensitive: Flag = True) -> Equality:
    left = a if case_sensitive else a.lower()
    right = b if case_sensitive else b.lower()
    if left == right:
        return Equality(equal=a, not_equal=SKIPPED)
    return Equality(equal=SKIPPED, not_equal=a)


@stdlib_node("logic-not", title="Not", description="Logical negation of a boolean", category="logic")
def not_node(value: Flag) -> Annotated[bool, Result(title="Negated")]:
    return (not value)


#: Every node this module defines, in palette order.
NODES = (if_empty, if_equals, not_node)
