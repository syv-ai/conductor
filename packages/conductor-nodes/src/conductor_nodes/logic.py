"""Nodes that route a value or turn a yes or no: ``logic-if-empty``, ``logic-if-equals`` and ``logic-not``.

The two ``if`` nodes give the text on one of two outputs and ``SKIPPED`` on
the other, so what is wired to the branch not taken doesn't run on that row.
"""

from dataclasses import dataclass
from typing import Annotated

from conductor import SKIPPED, NodeRegistry, Param

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="control")


@dataclass(frozen=True)
class Emptiness:
    not_empty: Annotated[str, Param(title="Not empty", exclusive_group="emptiness")]
    empty: Annotated[str, Param(title="Empty", exclusive_group="emptiness")]


@dataclass(frozen=True)
class Equality:
    equal: Annotated[str, Param(title="Equal", exclusive_group="equality")]
    not_equal: Annotated[str, Param(title="Not equal", exclusive_group="equality")]


@registry.node("logic-if-empty", title="If empty")
def if_empty(text: LongText) -> Emptiness:
    """Routes the text by whether it is empty once stripped."""
    if text.strip():
        return Emptiness(not_empty=text, empty=SKIPPED)
    return Emptiness(not_empty=SKIPPED, empty=text)


@registry.node("logic-if-equals", title="If equals")
def if_equals(a: Line, b: Line, case_sensitive: Flag = True) -> Equality:
    """Routes `a` by whether it equals `b`."""
    same = a == b if case_sensitive else a.lower() == b.lower()
    if same:
        return Equality(equal=a, not_equal=SKIPPED)
    return Equality(equal=SKIPPED, not_equal=a)


@registry.node("logic-not", title="Not", category="logic")
def negate(value: Flag) -> Annotated[bool, Param(title="Negated")]:
    """Turns yes into no and no into yes."""
    return not value
