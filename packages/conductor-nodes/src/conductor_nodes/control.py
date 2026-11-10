"""Nodes that send a value one of two ways: ``decision``, ``logic-if-empty`` and ``logic-if-equals``.

Each gives its value on one of two outputs and ``SKIPPED`` on the other, so
what is wired to the way not taken doesn't run on that row. A choice between
more than two ways is a chain of them.

``decision`` sends ``value`` on ``if_true`` or ``if_false`` by ``when``. ``when``
is a yes or no, not an expression to parse: other nodes give one
(``text-contains``, ``regex-match``, ``logic-not``), so the condition is wired
in rather than written in a language this node would have to own. ``value``
takes any value, since the node passes it on without reading it, and the
``compute_outputs`` hook gives both outputs the type that arrives on it.

The two ``logic-if`` nodes route a text by a test of their own.
"""

from dataclasses import dataclass
from typing import Annotated, Any

from conductor import SKIPPED, NodeRegistry, Param

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="control")


def _either_way(value: type) -> dict[str, Any]:
    """Both outputs of the type arriving on ``value``, ``Any`` until it is wired; one holds a value on each row."""
    return {
        "if_true": Annotated[value, Param(title="If true", exclusive_group="when")],
        "if_false": Annotated[value, Param(title="If false", exclusive_group="when")],
    }


@registry.node("decision", compute_outputs=_either_way)
def decision(value: Any, when: Flag = True) -> dict[str, Any]:
    """Sends a value one of two ways by `when`."""
    if when:
        return {"if_true": value, "if_false": SKIPPED}
    return {"if_true": SKIPPED, "if_false": value}


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
