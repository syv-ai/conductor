"""``Validator``, the base class of every validator."""

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
