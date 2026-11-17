"""Nodes that send a value one way or another: ``control-decide``, ``control-switch`` and ``control-default``.

``control-decide`` and ``control-switch`` send their value down one of several outputs. Each output is a branch: its
default is ``SKIPPED``. One branch gets the value and the others are skipped, so the nodes after them are skipped for
that call. You can wire several branches into one input. That node then takes the value from whichever branch has
it, so you don't need a node to join the branches again.

``control-decide`` picks its branch by ``when``, a yes or no. It doesn't parse a condition. Another node works out
the yes or no, and you wire it in: ``logic-equals``, ``logic-is-empty``, ``text-contains``, ``regex-match`` or
``logic-not``.

All three nodes pass ``value`` on, so it can be anything. ``control-default`` only checks whether it is ``None``.
Its type is a type variable, and the outputs take the type of whatever is wired to it. Wire in an ``int`` from
``text-length`` and the outputs are ``int`` too.
"""

from dataclasses import dataclass
from typing import Annotated, Any

from conductor import SKIPPED, NodeError, NodeRegistry, Param, Policy

registry = NodeRegistry(category="control")

_OTHER = "other"


@dataclass(frozen=True)
class Decided[T]:
    """The two outputs of ``control-decide``. One gets the value, and the other gets ``SKIPPED``."""

    if_true: Annotated[T, Param(title="If true")] = SKIPPED
    if_false: Annotated[T, Param(title="If false")] = SKIPPED


@registry.node(
    "control-decide",
    title="Decision",
    description="Sends the value one way when the switch is on and the other way when it is off.",
    policy=Policy(costs_nothing=True),
)
def decision[T](value: T, when: bool = True) -> Decided[T]:
    """Sends ``value`` to ``if_true`` when ``when`` is true, and to ``if_false`` when it is false."""
    return Decided(if_true=value) if when else Decided(if_false=value)


def _branches(cases: list[str], value: type) -> dict[str, Any]:
    """Works out the outputs of ``control-switch``: one for each name in ``cases``, and one more called ``other``.

    Conductor calls this when the graph is compiled. With ``cases`` set to ``["invoice", "receipt"]``, the outputs
    are ``invoice``, ``receipt`` and ``other``. Each has the type of what is wired to ``value``, and the default
    ``SKIPPED``, which makes it a branch. Only one of them gets the value on a call.

    Raises:
        NodeError: ``case_named_other`` if a case is called ``other``. ``case_given_twice`` if a name is in
            ``cases`` twice.
    """
    if _OTHER in cases:
        raise NodeError(
            "'other' is the branch for a case that has no branch of its own. Give the case another name.",
            code="case_named_other",
        )
    twice = sorted({name for name in cases if cases.count(name) > 1})
    if twice:
        raise NodeError(
            f"The case '{twice[0]}' is given twice. Give each case once.",
            code="case_given_twice",
            ctx={"case": twice[0]},
        )
    return {name: (value, SKIPPED) for name in [*cases, _OTHER]}


@registry.node(
    "control-switch",
    title="Switch",
    description="Sends the value down the branch named by the case, or down 'other' when no branch has that name.",
    compute_outputs=_branches,
    policy=Policy(costs_nothing=True),
)
def switch[T](value: T, case: str, cases: list[str]) -> dict[str, Any]:
    """Sends ``value`` to the output named ``case``, or to ``other`` when ``case`` isn't one of ``cases``.

    ``cases`` decides which outputs the node has, so it must be a constant written in the graph. ``case`` has to
    match a name exactly. To ignore upper and lower case, put ``text-lowercase`` in front.
    """
    taken = case if case in cases else _OTHER
    return {name: value if name == taken else SKIPPED for name in [*cases, _OTHER]}


@registry.node(
    "control-default",
    title="Default",
    description="Gives the value, or the fallback when the value is missing.",
    policy=Policy(costs_nothing=True),
)
def default[T](value: T | None, fallback: T) -> T:
    """Returns ``value``, or ``fallback`` when ``value`` is ``None``.

    When ``value`` is ``None``, the nodes after it get ``fallback``. If ``value`` comes from a skipped branch, this
    node is skipped like any other, and so are the nodes after it. ``fallback`` can be a constant. Wire
    it from another node instead, and you get the first of two values that isn't ``None``.
    """
    return fallback if value is None else value
