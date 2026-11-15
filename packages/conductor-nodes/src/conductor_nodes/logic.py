"""Nodes that give a yes or no: ``logic-not``, ``logic-equals`` and ``logic-is-empty``.

Wire their result into ``control-decide.when`` to send a value one way or the other.
"""

from collections.abc import Sized
from typing import Annotated, Any

from conductor import NodeRegistry, Param

registry = NodeRegistry(category="logic")


@registry.node("logic-not", title="Not", description="Turns yes into no and no into yes.")
def negate(value: bool) -> Annotated[bool, Param(title="Negated")]:
    """Turns yes into no and no into yes."""
    return not value


@registry.node(
    "logic-equals",
    title="Equals",
    description="Tells whether the value is the same as the other value.",
)
def equals[T](value: T, other: T) -> Annotated[bool, Param(title="Equal")]:
    """Whether ``value == other``. ``other`` is read by the type wired to ``value``, so a constant ``3`` is a number
    after ``text-length``, and a text is compared with its case: put ``text-lowercase`` in front to ignore it."""
    return value == other


@registry.node(
    "logic-is-empty",
    title="Is empty",
    description="Tells whether the value is missing, or is a text, list or table with nothing in it.",
)
def is_empty(value: Any) -> Annotated[bool, Param(title="Empty")]:
    """True for ``None`` and for a value of length 0: ``""``, ``[]``, ``{}`` or a table of no rows. A text of only
    spaces is not empty (put ``text-trim`` in front for that), and neither is ``0`` or ``False``."""
    return value is None or (isinstance(value, Sized) and len(value) == 0)
