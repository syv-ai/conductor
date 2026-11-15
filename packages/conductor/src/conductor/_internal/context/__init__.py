"""The run's context: one object a host passes for a run, which a node can take as its first parameter.

A host often has things that many nodes need, like the person running the graph or an HTTP client. It puts them in
one object, and passes it as ``context=`` when it starts or resumes a run. A node that needs it declares a first
parameter typed ``Context[X]``.

``type.py`` has ``Context``, and the check, when a graph is compiled, that a node's ``Context[X]`` fits the
registry's class. ``validator.py`` has ``ContextValidator``, which checks the object a host passes. The full
rules are in the reference, ``python -m conductor.about``.
"""

from .type import Context, ContextParameter, is_context
from .validator import ContextValidator

__all__ = [
    "Context",
    "ContextParameter",
    "ContextValidator",
    "is_context",
]
