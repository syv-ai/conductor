"""Exclusive groups: outputs of a node that are alternatives, so only one of them gets a value.

A node can have outputs that stand for different branches. Say ``route`` sends a line on to ``long`` or to
``short``, but never both. Its author puts the two outputs in one exclusive group, here called ``length``. On each
call, exactly one output of the group gets a value, and the others are ``SKIPPED``. A node that takes no branch at
all returns ``SKIPPED`` itself, which skips every output.

The rule is checked in two places. ``ReturnValidator`` checks what a node returns, and ``AnswerValidator`` checks a
person's answer to a node that asked. Both use the two functions here.
"""

from __future__ import annotations

from collections.abc import Container, Iterator, Mapping
from typing import NamedTuple

from .type import Field


class GroupWithoutOneValue(NamedTuple):
    """An exclusive group that broke the rule on one call: none of its outputs got a value, or several did.

    Attributes:
        group: The group's name.
        members: The group's outputs, in the order the node declares them.
        given: The outputs that got a value. It is empty, or it has two or more names.
    """

    group: str
    members: tuple[str, ...]
    given: list[str]


def exclusive_groups(outputs: Mapping[str, Field]) -> dict[str, tuple[str, ...]]:
    """Returns each exclusive group among a node's outputs, with the outputs in it.

    The outputs keep the order the node declares them in. For ``route``, this is ``{"length": ("long", "short")}``.
    An output in no group is left out.

    Args:
        outputs: The node's outputs, by name.
    """
    groups: dict[str, list[str]] = {}
    for name, field in outputs.items():
        if field.param.exclusive_group is not None:
            groups.setdefault(field.param.exclusive_group, []).append(name)
    return {group: tuple(members) for group, members in groups.items()}


def groups_without_one_value(
    groups: Mapping[str, tuple[str, ...]], given: Container[str]
) -> Iterator[GroupWithoutOneValue]:
    """Yields each group that didn't get exactly one value on a call.

    Say ``route`` returns a value on both ``long`` and ``short``. This yields
    ``GroupWithoutOneValue("length", ("long", "short"), ["long", "short"])``. If it returns neither, the last part
    is ``[]`` instead.

    Args:
        groups: The node's groups, as ``exclusive_groups`` returns them.
        given: The names of the outputs that got a value on the call.
    """
    for group, members in groups.items():
        chosen = [name for name in members if name in given]
        if len(chosen) != 1:
            yield GroupWithoutOneValue(group, members, chosen)
