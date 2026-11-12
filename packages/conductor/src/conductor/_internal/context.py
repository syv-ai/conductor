"""``Context[X]``: marks a parameter that is filled from the run's context: values the host passes for this run, by
type, never saved. Such a parameter is not an input of the node.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from conductor._internal.problem import NodeDeclarationError

_T = TypeVar("_T")


class _Context:
    """The mark ``Context`` leaves on a parameter. ``@node`` takes it off a whole parameter; anywhere else in a
    hint it refuses to be read."""

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Refuses a ``Context`` inside a field's type.

        Raises:
            NodeDeclarationError: Always (``context_misplaced``).
        """
        raise NodeDeclarationError(
            "Context is a whole parameter the host always gives, name: Context[Session]; anywhere else, a caller would fill it",
            code="context_misplaced",
        )


#: ``Context[X]``: a value from the run's context, which the host passes by its type (a session, who is running), never an input. It takes
#: no edge, is not stored in the record, and is no part of what makes a result reusable. The run is given one
#: with ``context={X: value}``. To a type checker the parameter is an ``X``.
Context = Annotated[_T, _Context()]
