"""``RefusedError``: what a door of the gate raises for a value it does not let in."""

from __future__ import annotations

from collections.abc import Iterable

from conductor._internal.problem import Problem


class RefusedError(Exception):
    """A door refused a value; each problem says what is wrong, with the door's code.

    Example: ``shout`` declares ``-> str`` and returns ``3``. The return door raises ``RefusedError`` with one problem,
    ``invalid_output`` for the output ``result``, which quotes nothing of the value.

    Whoever called the door decides what the refusal becomes. A public method raises ``InvalidCallError`` with the
    problems; the scheduler records the problem as the outcome of the call that failed, and puts the node and the
    call's item indexes in front of its ``loc``. A host never sees this exception.

    Attributes:
        problems: What is wrong, in the order the door found it. A door that a run calls once per node call
            reports the first fault only, so its ``RefusedError`` has one problem.
    """

    def __init__(self, problems: Iterable[Problem]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(problem.msg for problem in self.problems))
