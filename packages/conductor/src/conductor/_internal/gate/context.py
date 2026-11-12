"""The context door: the ``context=`` a host passes when it starts or resumes a run.

Example: on a registry that names ``HostContext``, ``graph.run(text="Done", context=HostContext(...))`` passes. Given a
``Caller`` instead, the door raises ``RefusedError`` with ``invalid_context``, which names the two classes and
quotes nothing of the object. Given no context when the graph has a node ``sign(ctx: Context[HostContext], ...)``, it
raises ``context_missing`` for ``sign``. A graph none of whose nodes takes the context runs with one or without.

Compiling has already checked each node's ``Context[X]`` against the registry's class, so an instance of the class
is what every node asks for.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from conductor._internal.base import NodePath
from conductor._internal.context import ContextParameter
from conductor._internal.problem import catalogue as problem

from .refused import RefusedError


class ContextDoor:
    """Checks the ``context=`` of a call that starts or resumes a run of one graph. Built once per graph.

    Raises (when called):
        RefusedError: The object is not an instance of the registry's class (``invalid_context``), or there is none
            and a node takes it (``context_missing``, one problem per such node).
    """

    def __init__(self, context_type: type | None, takers: Iterable[tuple[NodePath, ContextParameter]]) -> None:
        """Builds the door for a graph compiled against a registry that names ``context_type``.

        Args:
            context_type: The registry's class; ``None`` when it names none.
            takers: Each node of the graph, those inside subgraphs too, that takes the context: its node path and
                its ``Context`` parameter.
        """
        self._context_type = context_type
        self._missing = [problem.context_missing(*node_path, parameter.name, parameter=parameter.name, needs=parameter.cls.__name__) for node_path, parameter in takers]

    def __call__(self, context: Any) -> Any:
        """Returns ``context`` as the host passed it, ``None`` included when no node takes it.

        Raises:
            RefusedError: See the class.
        """
        if context is None:
            if self._missing:
                raise RefusedError(self._missing)
            return None
        if self._context_type is None:
            raise RefusedError([problem.invalid_context_no_context_type("context", got=type(context).__name__)])
        if not isinstance(context, self._context_type):
            raise RefusedError([problem.invalid_context("context", expected=self._context_type.__name__, got=type(context).__name__)])
        return context
