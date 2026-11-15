"""``control-decide``: send a value one of two ways, by a yes or no.

``control-decide`` returns ``value`` on ``if_true`` or ``if_false``, and ``SKIPPED`` on the other. The two outputs
are a branch: the nodes downstream of the output not chosen are skipped for that call.

``when`` is a yes or no, not an expression to parse. Other nodes give one (``logic-equals``, ``logic-is-empty``,
``text-contains``, ``regex-match``, ``logic-not``), so the condition is wired in rather than written in a language
this node would have to own. ``value`` takes any value, since the node passes it on without reading it: its type is
a type variable, so both outputs have the type of what is wired to it (an ``int`` from ``text-length`` gives two
``int`` outputs).
"""

from dataclasses import dataclass
from typing import Annotated

from conductor import SKIPPED, NodeRegistry, Param

registry = NodeRegistry(category="control")


@dataclass(frozen=True)
class Decided[T]:
    """The outputs of ``control-decide``: ``value`` on one of them, ``SKIPPED`` on the other."""

    if_true: Annotated[T, Param(title="If true", exclusive_group="when")] = SKIPPED
    if_false: Annotated[T, Param(title="If false", exclusive_group="when")] = SKIPPED


@registry.node(
    "control-decide",
    title="Decision",
    description="Sends the value one way when the switch is on and the other way when it is off.",
)
def decision[T](value: T, when: bool = True) -> Decided[T]:
    """Returns ``value`` on ``if_true`` when ``when`` is true, and on ``if_false`` when it is false."""
    return Decided(if_true=value) if when else Decided(if_false=value)
