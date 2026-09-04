"""``conductor_nodes`` — the standard node library.

One module per category. ``register_all`` registers every node on a
registry you supply; ``get_default_registry`` builds a fresh one::

    from conductor import NodeRegistry
    from conductor_nodes import register_all, get_default_registry

    registry = NodeRegistry()
    register_all(registry, categories=["text", "math"])   # a subset
    registry = get_default_registry()                     # everything
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor_nodes import (
    decision,
    json_ops,
    logic,
    math,
    regex_ops,
    text,
)

if TYPE_CHECKING:
    from conductor import NodeRegistry

#: Category name -> the module whose ``register`` adds its nodes.
CATEGORIES: dict[str, object] = {
    "text": text,
    "math": math,
    "logic": logic,
    "json": json_ops,
    "regex": regex_ops,
    "decision": decision,
}


def register_all(registry: "NodeRegistry", *, categories: list[str] | None = None) -> None:
    """Register the nodes of ``categories`` (default: all) on ``registry``.

    An unknown category name is a ``KeyError`` naming the known ones.
    """
    for name in (list(CATEGORIES) if categories is None else categories):
        if name not in CATEGORIES:
            raise KeyError(f"Unknown category '{name}'. Known: {sorted(CATEGORIES)}")
        CATEGORIES[name].register(registry)   # type: ignore[attr-defined]


def get_default_registry(*, categories: list[str] | None = None) -> "NodeRegistry":
    """A new ``NodeRegistry`` holding the nodes of ``categories`` (default: all)."""
    from conductor import NodeRegistry

    reg = NodeRegistry()
    register_all(reg, categories=categories)
    return reg
