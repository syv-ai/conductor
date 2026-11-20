"""``ContextValidator``, the check on the ``context=`` a host passes when it starts or resumes a run.

Say the registry names ``HostContext``. Then ``graph.run({"text": "Done"}, context=HostContext(...))`` passes. Given a
``Caller`` instead, the validator raises ``InvalidCallError`` with ``invalid_context``, which names the two classes
and quotes nothing of the object. Given no context at all, when the graph has a node
``sign(ctx: Context[HostContext], ...)``, it raises ``context_missing`` for ``sign``.

A graph where no node takes the context runs without one. A context it is given is still checked, though. It must be
an instance of the registry's class, and a registry that names no class takes no context at all: it refuses one
with ``invalid_context_no_context_type``.

Compiling has already checked each node's ``Context[X]`` against the registry's class. So an instance of that class
is what every node asks for, and checking the class is enough.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from conductor._internal.base import NodePath, Validator
from conductor._internal.problem import InvalidCallError, call_problems

from .type import ContextParameter


class ContextValidator(Validator[Any, Any]):
    """Checks the ``context=`` of a call that starts or resumes a run of one graph.

    One validator is built for each graph.

    Raises (when called):
        InvalidCallError: ``invalid_context`` if the object isn't an instance of the registry's class, or the
            registry names no class. ``context_missing`` if there is no object and a node takes it, one problem per
            such node.
    """

    def __init__(
        self, context_type: type | None, takers: Iterable[tuple[NodePath, ContextParameter]]
    ) -> None:
        """Builds the validator for a graph compiled against a registry whose context class is ``context_type``.

        Args:
            context_type: The registry's class, or ``None`` when it names none.
            takers: Each node of the graph that takes the context, inside subgraphs too, with its node path and its
                ``Context`` parameter.
        """
        self._context_type = context_type
        self._missing = [
            call_problems.context_missing(
                *node_path, parameter.name, parameter=parameter.name, needs=parameter.cls.__name__
            )
            for node_path, parameter in takers
        ]

    def __call__(self, context: Any) -> Any:
        """Checks ``context`` and returns it as the host passed it, which may be ``None`` when no node takes it.

        Raises:
            InvalidCallError: See the class.
        """
        if context is None:
            if self._missing:
                raise InvalidCallError(self._missing)
            return None
        if self._context_type is None:
            raise InvalidCallError(
                [
                    call_problems.invalid_context_no_context_type(
                        "context", got=type(context).__name__
                    )
                ]
            )
        if not isinstance(context, self._context_type):
            raise InvalidCallError(
                [
                    call_problems.invalid_context(
                        "context", expected=self._context_type.__name__, got=type(context).__name__
                    )
                ]
            )
        return context
