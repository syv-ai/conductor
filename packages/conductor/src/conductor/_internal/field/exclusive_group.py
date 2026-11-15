"""The rule of an exclusive group: of the outputs of a node that share one, exactly one carries a value on a call, and
the others are ``SKIPPED``.

``route`` has the outputs ``long`` and ``short`` in the group ``length``, and gives a value on one of them per call.
The return validator holds what a node returns to the rule, and the answer validator a person's answer. A node that
chooses no branch returns ``SKIPPED``, which skips every output. The ``Ask`` validator refuses a default for an output
of a group, for the same rule: an answer that leaves that output out would give it the default, so no other output
of the group could be answered.
"""

from __future__ import annotations

from collections.abc import Container, Iterator, Mapping
from typing import NamedTuple

from .type import Field


class GroupWithoutOneValue(NamedTuple):
    """An exclusive group that does not have exactly one output with a value.

    Attributes:
        group: The group's name.
        members: Its outputs, in the order they are declared.
        given: Those of them that carry a value: none, or two or more.
    """

    group: str
    members: tuple[str, ...]
    given: list[str]


def exclusive_groups(outputs: Mapping[str, Field]) -> dict[str, tuple[str, ...]]:
    """Returns the outputs of each exclusive group of ``outputs``, by group, in the order they are declared:
    ``{"length": ("long", "short")}`` for ``route``. A validator builds it once per node."""
    groups: dict[str, list[str]] = {}
    for name, field in outputs.items():
        if field.param.exclusive_group is not None:
            groups.setdefault(field.param.exclusive_group, []).append(name)
    return {group: tuple(members) for group, members in groups.items()}


def groups_without_one_value(
    groups: Mapping[str, tuple[str, ...]], given: Container[str]
) -> Iterator[GroupWithoutOneValue]:
    """Yields each group of ``groups`` that does not have exactly one output among ``given``, the outputs that carry
    a value, in the order of ``groups``. ``route`` returning a value on both ``long`` and ``short`` yields
    ``GroupWithoutOneValue("length", ("long", "short"), ["long", "short"])``, and returning neither the same with
    ``[]``."""
    for group, members in groups.items():
        chosen = [name for name in members if name in given]
        if len(chosen) != 1:
            yield GroupWithoutOneValue(group, members, chosen)
