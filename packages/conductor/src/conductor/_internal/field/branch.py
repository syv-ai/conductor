"""Branches: the outputs of a node whose default is ``SKIPPED``, and the one rule they follow.

The rule is written in the reference, under "Branches" (``python -m conductor.about the-node-contract``). Say
``route`` sends a line on to ``long`` or to ``short``. Its author gives both outputs the default ``SKIPPED``, and
returns a value on one of them. Returning a value on both breaks the rule.

The rule is checked in three places. ``ReturnValidator`` checks what a node returns, ``AnswerValidator`` checks
a person's answer to a node that asked, and ``AskValidator`` checks the default an ``Ask`` proposes. All three call
``check_branches``, so each refuses with the same code and message.
"""

from __future__ import annotations

from collections.abc import Container, Mapping

from conductor._internal.problem import conductor_error
from conductor._internal.utils import comma_separated

from .type import Field


def check_branches(outputs: Mapping[str, Field], given: Container[str]) -> list[str]:
    """Refuses a call that gives more than one branch among ``outputs`` a value, and returns the one it gives, if any.

    For ``route`` returning a value on both ``long`` and ``short``, the message is "'long' and 'short' both have a
    value, but they default to SKIPPED, so they are branches of one node". One branch or none keeps the rule.

    Args:
        outputs: The node's outputs, by name.
        given: The names of the outputs that got a value on the call. A return gives an output a value when the
            value isn't ``SKIPPED``. An answer and an ``Ask``'s default give a value to each output they name.

    Returns:
        The branch given a value, in a list, or an empty list when none is.

    Raises:
        PydanticCustomError: ``several_branches_given``, with ``ctx.given`` the branches given a value, in the order
            the node declares them.
    """
    taken = [name for name, field in outputs.items() if field.is_branch and name in given]
    if len(taken) <= 1:
        return taken
    quoted = [f"'{name}'" for name in taken]
    holding = "both" if len(quoted) == 2 else "all"
    raise conductor_error(
        "several_branches_given",
        f"{comma_separated(quoted[:-1])} and {quoted[-1]} {holding} have a value, but they default to SKIPPED, so"
        " they are branches of one node",
        {"given": taken},
    )
