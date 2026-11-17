"""Nodes that answer yes or no: ``logic-not``, ``logic-equals`` and ``logic-is-empty``.

Wire the answer into ``control-decide.when`` to send a value one way or the other.
"""

from collections.abc import Sized
from typing import Annotated, Any

from conductor import NodeRegistry, Param, Policy

registry = NodeRegistry(category="logic")


@registry.node(
    "logic-not",
    title="Not",
    description="Turns yes into no and no into yes.",
    policy=Policy(costs_nothing=True),
)
def negate(value: bool) -> Annotated[bool, Param(title="Negated")]:
    """Turns yes into no and no into yes."""
    return not value


@registry.node(
    "logic-equals",
    title="Equals",
    description="Tells whether the value is the same as the other value.",
    policy=Policy(costs_nothing=True),
)
def equals[T](value: T, other: T) -> Annotated[bool, Param(title="Equal")]:
    """Whether ``value == other``.

    ``other`` takes the type of what is wired to ``value``. So after ``text-length``, a constant ``3`` is the
    number 3. Texts are compared exactly, upper and lower case included. To ignore case, put ``text-lowercase`` in
    front.
    """
    return value == other


@registry.node(
    "logic-is-empty",
    title="Is empty",
    description="Tells whether the value is missing, or is a text, list or table with nothing in it.",
    policy=Policy(costs_nothing=True),
)
def is_empty(value: Any) -> Annotated[bool, Param(title="Empty")]:
    """Whether ``value`` is ``None``, or has a length of 0.

    So ``None``, ``""``, ``[]``, ``{}`` and a table with no rows are empty. ``0`` and ``False`` are not. A text of
    only spaces isn't empty either. Put ``text-trim`` in front to count it as empty.
    """
    return value is None or (isinstance(value, Sized) and len(value) == 0)
