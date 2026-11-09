"""``decision`` — route a value to one of two branches.

The node sends ``value`` to its ``if_true`` or ``if_false`` output
according to ``when`` and returns ``SKIPPED`` on the other, so whatever is
connected to the branch not taken does not run. The engine treats ``SKIPPED``
like any other value; the node has no special role. A choice between more
than two branches is a chain of decisions.

``when`` is a ``bool``, not an expression to parse: the catalog already
produces booleans (``text-contains``, ``regex-match``, ``logic-not``), so
the condition is connected in rather than written in a language this node
would have to own.

``value`` is declared ``Any`` because the node routes it without reading
it and so cannot name its type. ``compute_outputs`` copies the type that
arrives on ``value`` onto both outputs. The two outputs share the
``choice`` ``"when"``: exactly one of them is produced per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from conductor import SKIPPED, Arriving, Output, Param, Result
from conductor.widgets import Switch

from conductor_nodes.types import stdlib_node


@dataclass(frozen=True)
class Branches:
    """What ``Decision.run`` returns: one field per output.

    Both fields share ``choice="when"``, so the interface records that
    exactly one of them is produced; the other holds ``SKIPPED``.
    """

    if_true: Annotated[Any, Result(title="If true", choice="when")]
    if_false: Annotated[Any, Result(title="If false", choice="when")]


def decision_outputs(value: Arriving) -> tuple[Output, Output]:
    """Give both outputs the type that arrives on ``value`` (``Any`` until connected)."""
    return Output("if_true", value | None, title="If true", choice="when"), Output("if_false", value | None, title="If false", choice="when")


@stdlib_node("decision", title="Decision", description="Routes a value to one of two branches according to `when`.", category="control", compute_outputs=decision_outputs)
def decision(
    value: Annotated[Any, Param(title="Value")],
    when: Annotated[bool, Param(title="When", widget=Switch())] = True,
) -> Branches:
    if when:
        return Branches(if_true=value, if_false=SKIPPED)
    return Branches(if_true=SKIPPED, if_false=value)


#: Every node this module defines, in palette order.
NODES = (decision,)
