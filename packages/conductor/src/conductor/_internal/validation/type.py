"""``Validator``, the base class of every validator: what reads a value from outside the engine where it enters.

A validator is built with what it reads a value by (a node's outputs, a graph's inputs, the input a constant is for)
and where the value belongs (a node id), and is called with the value alone::

    outputs = compiled_node.return_validator("B!")   # {"result": "B!"} for shout

It returns the value as the engine keeps it, or raises ``InvalidCallError`` with a problem for each fault it reports;
its caller decides what becomes of the problems. Each package whose values come from outside keeps its validators in
its ``validator.py``, and a subclass declared in any other module is refused when the module is imported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Validator[V, T](ABC):
    """Reads one value of type ``V`` from outside the engine into the ``T`` the engine keeps, or refuses it.

    A subclass takes what it reads by in ``__init__`` and the value alone in ``__call__``, so every validator is
    called the same way. It is declared in a module named ``validator.py``.

    Raises (when called):
        InvalidCallError: The value is refused, with a problem for each fault the validator reports. A default a node
            declares raises ``NodeDeclarationError`` instead: it is a fault in the node's code, met at import.
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
        """Returns ``value`` as the engine keeps it.

        Raises:
            InvalidCallError: The value is refused.
        """
