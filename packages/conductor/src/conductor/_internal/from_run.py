"""``FromRun[X]``: marks a parameter that the host fills in by its type when it starts a run. Such a parameter is not
an input of the node.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from conductor._internal.problem import NodeDeclarationError

_T = TypeVar("_T")


class _FromRun:
    """The mark ``FromRun`` leaves on a parameter. ``@node`` takes it off a whole parameter; anywhere else in a
    hint it refuses to be read."""

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Refuses a ``FromRun`` inside a field's type.

        Raises:
            NodeDeclarationError: Always (``from_run_misplaced``).
        """
        raise NodeDeclarationError(
            "FromRun is a whole parameter the host always gives, name: FromRun[Session]; anywhere else, a caller would fill it",
            code="from_run_misplaced",
        )


#: ``FromRun[X]``: a value the host hands the run by its type (a session, who is running), never an input. It takes
#: no edge, is not stored in the record, and is no part of what makes a result reusable. The run is given one
#: with ``from_run={X: value}``. To a type checker the parameter is an ``X``.
FromRun = Annotated[_T, _FromRun()]
