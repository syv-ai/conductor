"""Branches: the outputs of a node whose default is ``SKIPPED``. A call gives a value to at most one of them.

Say ``route`` sends a line on to ``long`` or to ``short``, but never both. Its author gives both outputs the
default ``SKIPPED``, and returns a value on one of them. The other stays ``SKIPPED``, and what is wired to it is
skipped. Since a call never gives two branches of one node a value, both may be wired into one input.

An output that may be missing but may also have a value beside another output isn't a branch. It is typed
``str | None``, and gets ``None`` when it has no value.

The rule is checked in two places. ``ReturnValidator`` checks what a node returns, and ``AnswerValidator`` checks
a person's answer to a node that asked. Both use the function here.
"""

from __future__ import annotations

from collections.abc import Container, Mapping

from .type import Field


def branches_given(outputs: Mapping[str, Field], given: Container[str]) -> list[str]:
    """Returns the branches among ``outputs`` that got a value on a call, in the order the node declares them.

    For ``route`` returning a value on both ``long`` and ``short``, this is ``["long", "short"]``, which breaks the
    rule. One name or none keeps it.

    Args:
        outputs: The node's outputs, by name.
        given: The names of the outputs that got a value on the call.
    """
    return [name for name, field in outputs.items() if field.is_branch and name in given]
