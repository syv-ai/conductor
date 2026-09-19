"""``conductor_nodes`` — the standard node library.

One module per category. ``registry`` builds a fresh registry holding
them; ``register_all`` adds them to a registry you already have::

    import conductor_nodes

    registry = conductor_nodes.registry()                          # everything
    registry = conductor_nodes.registry(categories=["text"])       # a subset
    conductor_nodes.register_all(my_registry, categories=["math"])

Conductor's core does not know this package exists, so a registry with
the standard nodes in it is asked of this package, never of
``NodeRegistry``: the nodes bring their own vocabulary, which a host with
types of its own does not want.
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


def registry(*, categories: list[str] | None = None) -> "NodeRegistry":
    """A new ``NodeRegistry`` holding the nodes of ``categories`` (default: all)."""
    from conductor import NodeRegistry

    reg = NodeRegistry()
    register_all(reg, categories=categories)
    return reg
