"""conductor-nodes — reusable node library for the conductor DAG engine.

Each submodule exports a ``register(reg)`` function that attaches its nodes
to a ``conductor.NodeRegistry``. Pick categories individually, or call
``register_all()`` for everything.

Example:

    from conductor import NodeRegistry
    from conductor_nodes import register_all

    reg = NodeRegistry()
    register_all(reg)

    # …or just what you need:
    from conductor_nodes import text, math
    reg = NodeRegistry()
    text.register(reg)
    math.register(reg)

Node IDs are prefixed by category (``text-uppercase``, ``math-add``, …) to
minimize collisions with application-level node IDs. The canonical for-each
markers — ``for-each-start`` and ``for-each-end`` — are kept unprefixed to
match the conductor core's compound-region discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor_nodes import json_ops, logic, loop, math, regex_ops, text

if TYPE_CHECKING:
    from conductor import NodeRegistry


CATEGORIES: dict[str, object] = {
    "text": text,
    "math": math,
    "logic": logic,
    "loop": loop,
    "json": json_ops,
    "regex": regex_ops,
}


def register_all(registry: "NodeRegistry", *, categories: list[str] | None = None) -> None:
    """Register every (or a filtered subset of) category's nodes.

    Args:
        registry: The target ``NodeRegistry``.
        categories: If provided, only these category names register. Unknown
            names raise ``KeyError``. If None, every category registers.
    """
    names = list(CATEGORIES) if categories is None else categories
    for name in names:
        if name not in CATEGORIES:
            raise KeyError(
                f"Unknown category '{name}'. Known: {sorted(CATEGORIES)}"
            )
        CATEGORIES[name].register(registry)   # type: ignore[attr-defined]


__all__ = ["CATEGORIES", "register_all", "text", "math", "logic", "loop", "json_ops", "regex_ops"]
