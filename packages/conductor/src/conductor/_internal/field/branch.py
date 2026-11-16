"""Branches: the outputs of a node whose default is ``SKIPPED``, and the one rule they follow.

The rule is written in the reference, under "Branches" (``python -m conductor.about the-node-contract``). Say
``route`` sends a line on to ``long`` or to ``short``. Its author gives both outputs the default ``SKIPPED``, and
returns a value on one of them. Returning a value on both breaks the rule.

The rule is checked in three places. ``ReturnValidator`` checks what a node returns, ``AnswerValidator`` checks
a person's answer to a node that asked, and ``AskValidator`` checks the defaults an ``Ask`` fills an answer in with.
All three use the function here.
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
