"""``Validator``, the base class of every check on a value that comes into conductor from outside.

Values reach the engine from many places: a caller's graph inputs, a constant in a graph record, what a node
returns, a person's answer. Each is checked once, where it comes in. After that, nothing inside the engine checks
it again.

Every check is a ``Validator``, and each package whose values come from outside has its own, in its
``validator.py``. When a validator refuses a value, whoever called it decides what happens to the problems. A public
method lets them reach its own caller, compiling keeps them in ``graph.problems``, and a run keeps the problem as the
outcome of the call.

Validators check values with the checks a ``Type`` owns (``types/``), and read a value that enters a graph strictly
with ``read_as_argument`` or ``read_as_output``. A few values are checked elsewhere. A saved run is checked in
``run/json/``, where one schema both reads and writes it. A ``NodeError``'s ``ctx`` is checked by ``NodeError``
itself, where a node makes one. And a type's own declarations are checked where the type is declared: ``Param``
checks its fields by its own annotations and its choices against the type it limits (``types/param.py``), and
``@host_type`` checks its arguments.

This module holds the base class, not a concept's validators, so it is the one ``validator.py`` with no
``Validator`` subclass in it.

The class imports nothing, so it stands at the bottom of conductor, where every package can use it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Validator[V, T](ABC):
    """The base class of every validator: a check on one value of type ``V`` that comes in from outside.

    A validator is built with what it checks against, such as a node's outputs, a graph's inputs or the input a
    constant is for, and sometimes where the value belongs, such as a node id. Then it is called with the value
    alone, so every validator is called the same way. It returns the value as a ``T`` the engine can use, or
    refuses it.

    For example, a compiled node's ``return_validator`` checks what the node's function returned. For the node
    ``shout``, it turns ``"B!"`` into the node's outputs::

        outputs = compiled_node.return_validator("B!")   # {"result": "B!"}

    Declare a subclass in a module named ``validator.py``. A subclass declared anywhere else is refused when its
    module is imported.

    Raises (when called):
        InvalidCallError: The value is refused, with a problem for each fault found.
        NodeDeclarationError: Raised instead by ``DeclaredInputValidator`` and ``DeclaredOutputValidator``. They
            check a default the node declares, so a fault there is in the node's own code, found when it is imported.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuses a subclass declared outside a ``validator.py`` module.

        Raises:
            TypeError: The subclass is declared in another module.
        """
        super().__init_subclass__(**kwargs)
        if not cls.__module__.endswith(".validator"):
            raise TypeError(
                f"{cls.__qualname__} is a Validator, so it is declared in a validator.py module, "
                f"not in {cls.__module__}"
            )

    @abstractmethod
    def __call__(self, value: V, /) -> T:
        """Checks ``value`` and returns it ready for the engine.

        Raises:
            InvalidCallError: The value is refused.
        """
