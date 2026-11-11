"""The exceptions of a call to the worker that names no run or that the run's standing refuses, each holding its problems."""

from __future__ import annotations

from conductor import Problem

from conductor_providers.fastapi.router_problem.catalogue import no_such_run


class RefusedError(Exception):
    """A call the worker refused because of how the run stands.

    Attributes:
        problems: Why, each with one of the router's own codes.
    """

    def __init__(self, *problems: Problem) -> None:
        super().__init__(" ".join(problem.msg for problem in problems))
        self.problems = list(problems)


class NoSuchRunError(Exception):
    """No run has the id, or the caller may not see the one that has.

    Attributes:
        problems: The one problem, ``no_such_run``.
    """

    def __init__(self) -> None:
        problem = no_such_run()
        super().__init__(problem.msg)
        self.problems = [problem]
