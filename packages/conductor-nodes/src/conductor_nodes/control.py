"""Nodes that choose where a value goes: ``control-decide``, ``control-switch`` and ``control-default``.

``control-decide`` and ``control-switch`` return the value on one output and ``SKIPPED`` on the others. Their
outputs are a branch: the nodes after an output not chosen are skipped for that call. Wire several branches into one
input and that node takes whichever branch was chosen, so no node is needed to join them again.

``control-decide`` sends ``value`` on ``if_true`` or ``if_false`` by ``when``. ``when`` is a yes or no, not an
expression to parse. Other nodes give one (``logic-equals``, ``logic-is-empty``, ``text-contains``, ``regex-match``,
``logic-not``), so the condition is wired in rather than written in a language this node would have to own.

``value`` takes any value on all three nodes, since they pass it on without reading it. Its type is a type variable,
so the outputs have the type of what is wired to it: an ``int`` from ``text-length`` gives ``int`` outputs.
"""

from dataclasses import dataclass
from typing import Annotated, Any

from conductor import SKIPPED, NodeError, NodeRegistry, Param

registry = NodeRegistry(category="control")

_OTHER = "other"


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


def _branches(cases: list[str], value: type) -> dict[str, Any]:
    """Returns the outputs of ``control-switch``: one per name in ``cases``, then ``other``.

    Each has the type wired to ``value`` and all are one exclusive group. With ``cases`` ``["invoice", "receipt"]``
    the outputs are ``invoice``, ``receipt`` and ``other``.

    Raises:
        NodeError: A case is named ``other`` (``case_named_other``), or one name is in ``cases`` twice
            (``case_given_twice``).
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
    return {name: Annotated[value, Param(exclusive_group="case")] for name in [*cases, _OTHER]}


@registry.node(
    "control-switch",
    title="Switch",
    description="Sends the value down the branch named by the case, or down 'other' when no branch has that name.",
    compute_outputs=_branches,
)
def switch[T](value: T, case: str, cases: list[str]) -> dict[str, Any]:
    """Returns ``value`` on the output named ``case``, or on ``other`` when ``case`` is not in ``cases``.

    ``cases`` names the branches and is set when the graph is built, so it takes only a constant. ``case`` is
    compared as it is: put ``text-lowercase`` in front to ignore case.
    """
    chosen = case if case in cases else _OTHER
    return {name: value if name == chosen else SKIPPED for name in [*cases, _OTHER]}


@registry.node(
    "control-default",
    title="Default",
    description="Gives the value, or the fallback when the value is missing.",
)
def default[T](value: T | None, fallback: T) -> T:
    """Returns ``value``, or ``fallback`` when ``value`` is ``None``.

    The nodes after it always get a value. Wire ``fallback`` from another node, rather than giving it a constant,
    to take the first of two values that is there.
    """
    return fallback if value is None else value
