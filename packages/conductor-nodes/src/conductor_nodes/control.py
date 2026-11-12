"""Nodes that send a value one of two ways: ``control-decide``, ``control-if-empty`` and ``control-if-equals``.

Each returns its value on one of two outputs and ``SKIPPED`` on the other. The two
outputs are a branch: the nodes downstream of the output not chosen are skipped for
that call. A choice between more than two ways is a chain of these nodes.

``control-decide`` sends ``value`` on ``if_true`` or ``if_false`` by ``when``. ``when``
is a yes or no, not an expression to parse: other nodes give one
(``text-contains``, ``regex-match``, ``logic-not``), so the condition is wired
in rather than written in a language this node would have to own. ``value``
takes any value, since the node passes it on without reading it: its type is
a type variable, so both outputs have the type of what is wired to it (an ``int``
from ``text-length`` gives two ``int`` outputs).

``control-if-empty`` and ``control-if-equals`` route a text by a test of their own.
"""

from dataclasses import dataclass
from typing import Annotated

from conductor import SKIPPED, NodeRegistry, Param

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="control")


@dataclass(frozen=True)
class Decided[T]:
    """The outputs of ``control-decide``: ``value`` on one of them, ``SKIPPED`` on the other."""

    if_true: Annotated[T, Param(title="If true", exclusive_group="when")] = SKIPPED
    if_false: Annotated[T, Param(title="If false", exclusive_group="when")] = SKIPPED


@registry.node("control-decide", title="Decision", description="Sends the value one way when the switch is on and the other way when it is off.")
def decision[T](value: T, when: Flag = True) -> Decided[T]:
    """Returns ``value`` on ``if_true`` when ``when`` is true, and on ``if_false`` when it is false."""
    return Decided(if_true=value) if when else Decided(if_false=value)


@dataclass(frozen=True)
class Emptiness:
    """The outputs of ``control-if-empty``: the text on one of them, ``SKIPPED`` on the other."""

    not_empty: Annotated[str, Param(title="Not empty", exclusive_group="emptiness")]
    empty: Annotated[str, Param(title="Empty", exclusive_group="emptiness")]


@dataclass(frozen=True)
class Equality:
    """The outputs of ``control-if-equals``: ``a`` on one of them, ``SKIPPED`` on the other."""

    equal: Annotated[str, Param(title="Equal", exclusive_group="equality")]
    not_equal: Annotated[str, Param(title="Not equal", exclusive_group="equality")]


@registry.node("control-if-empty", title="If empty", description="Sends the text one way when it is empty or only spaces, and the other way when it has something in it.")
def if_empty(text: LongText) -> Emptiness:
    """Returns the text on ``empty`` when it is empty or only whitespace, and on ``not_empty`` otherwise."""
    if text.strip():
        return Emptiness(not_empty=text, empty=SKIPPED)
    return Emptiness(not_empty=SKIPPED, empty=text)


@registry.node("control-if-equals", title="If equals", description="Sends the first text one way when it matches the second and the other way when it doesn't.")
def if_equals(a: Line, b: Line, case_sensitive: Flag = True) -> Equality:
    """Returns ``a`` on ``equal`` when it equals ``b`` (ignoring case unless ``case_sensitive``), and on ``not_equal`` otherwise."""
    same = a == b if case_sensitive else a.lower() == b.lower()
    if same:
        return Equality(equal=a, not_equal=SKIPPED)
    return Equality(equal=SKIPPED, not_equal=a)
