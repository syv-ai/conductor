"""Node category — classification tag *and* registration unit.

A ``NodeCategory`` serves two purposes:

1. **Classification** — the value that shows up on a node's ``category``
   field and drives frontend palette grouping / styling. ``NodeCategory.IO``
   and ``NodeCategory.CONTROL`` are ready-made built-ins; hosts may add
   their own (e.g. ``NodeCategory("primitives", label="Primitives")``).

2. **Registration unit** — a category collects ``@category.node(...)``
   decorated functions at import time and the host applies them to a
   registry in one shot via ``registry.include(category)``. Each node's
   ``category`` field is auto-assigned, so palette grouping is consistent
   by construction.

Designed to be drop-in compatible with the previous ``str``-enum shape
(``NodeCategory.IO``, ``.value``, string comparison all still work).

Example::

    # primitives.py
    from conductor import NodeCategory
    from conductor.widgets import Textarea, Output

    primitives = NodeCategory(
        "primitives",
        label="Primitives",
        description="Data input nodes",
    )

    @primitives.node("datainput-str", version=1, name="Tekst", description="...")
    def datainput_str(
        value: Annotated[str | None, Textarea(label="Tekst")] = None,
    ) -> Annotated[str | None, Output(label="Tekst")]:
        return value

    # setup.py
    registry = NodeRegistry()
    registry.include(primitives)
"""

from __future__ import annotations

from typing import Any, Callable


class NodeCategory:
    """A named group of nodes that also acts as a registration decorator host."""

    # Built-in well-known categories. Initialized below the class body.
    IO: "NodeCategory"
    CONTROL: "NodeCategory"

    __slots__ = ("id", "label", "description", "_pending")

    def __init__(
        self,
        id: str,
        *,
        label: str | None = None,
        description: str | None = None,
    ) -> None:
        if not id or not all(c.isalnum() or c in "-_" for c in id):
            raise ValueError(
                f"NodeCategory id must be a non-empty alphanumeric slug, got: {id!r}"
            )
        self.id = id
        self.label = label or id.replace("-", " ").replace("_", " ").title()
        self.description = description
        # Pending (base_id, kwargs, func) triples applied when a registry
        # includes this category.
        self._pending: list[tuple[str, dict[str, Any], Callable[..., Any]]] = []

    # ------------------------------------------------------------------
    # Backwards-compatibility with the old ``str``-Enum shape
    # ------------------------------------------------------------------

    @property
    def value(self) -> str:
        """Compatibility shim — returns the category id like the old enum did."""
        return self.id

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"NodeCategory({self.id!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NodeCategory):
            return self.id == other.id
        if isinstance(other, str):
            return self.id == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("NodeCategory", self.id))

    # ------------------------------------------------------------------
    # Registration unit API
    # ------------------------------------------------------------------

    def node(
        self,
        base_id: str,
        *,
        version: int = 1,
        name: str,
        description: str,
        tags: list[str] | None = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        width: int | None = None,
        docs: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that stashes a pending node registration on this category.

        Mirrors ``@registry.node(...)`` exactly, except the ``category``
        argument is omitted: it's auto-set to ``self`` when the registry
        later calls ``include(category)``.
        """
        kwargs: dict[str, Any] = {
            "version": version,
            "name": name,
            "description": description,
            "tags": tags,
            "category": self,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "width": width,
            "docs": docs,
        }

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._pending.append((base_id, kwargs, func))
            return func

        return decorator

    @property
    def nodes(self) -> list[tuple[str, dict[str, Any], Callable[..., Any]]]:
        """Return a snapshot of the pending registrations on this category."""
        return list(self._pending)


# ----------------------------------------------------------------------
# Built-in categories — drop-in replacements for the old enum values.
# Assigned after the class body so they reference the completed class.
# ----------------------------------------------------------------------
NodeCategory.IO = NodeCategory("io", label="I/O")
NodeCategory.CONTROL = NodeCategory("control", label="Control")
