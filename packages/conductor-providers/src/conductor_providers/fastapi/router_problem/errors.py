"""The exceptions a refused call to the worker raises, each holding the router's problems it was refused for."""

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


class NoSuchRunError(RefusedError):
    """No run has the id, or the caller may not see the one that has."""

    def __init__(self) -> None:
        super().__init__(no_such_run())
