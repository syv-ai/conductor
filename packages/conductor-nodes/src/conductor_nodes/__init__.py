"""``conductor_nodes`` — the standard node library.

One module per kind of node, each exposing ``register(registry)``.
``registry`` builds a fresh registry holding them; ``register_all`` adds
them to a registry you already have. ``categories`` filters on each
node's own ``category`` (``control``, ``json``, ``logic``, ``math``,
``regex``, ``text``)::

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

from collections.abc import Sequence
from typing import TYPE_CHECKING, get_args

from conductor_nodes import (
    decision,
    json_ops,
    logic,
    math,
    regex_ops,
    text,
)
from conductor_nodes.types import Category

if TYPE_CHECKING:
    from conductor import NodeRegistry
    from conductor.node import NodeDefinition

#: Every module of the library, each exposing ``register(registry)``.
MODULES = (text, math, logic, json_ops, regex_ops, decision)


def _nodes(categories: Sequence[Category] | None) -> tuple[type[NodeDefinition], ...]:
    """Every node of the library whose own ``category`` is one of ``categories`` (all when ``None``)."""
    from conductor import NodeRegistry

    if isinstance(categories, str):
        raise TypeError(f"categories is a list of category names, not {categories!r}; say [{categories!r}]")
    known = get_args(Category)
    unknown = sorted(set(categories or ()) - set(known))
    if unknown:
        raise KeyError(f"Unknown category {', '.join(map(repr, unknown))}. Known: {list(known)}")
    everything = NodeRegistry()
    for module in MODULES:
        module.register(everything)
    return tuple(cls for cls in everything.nodes if categories is None or cls.category in categories)


def register_all(registry: "NodeRegistry", *, categories: Sequence[Category] | None = None) -> None:
    """Register the nodes whose category is one of ``categories`` (default: all) on ``registry``.

    A category is the node class's own ``category``. An unknown name is a
    ``KeyError`` naming the known ones; a bare string is a ``TypeError``.
    """
    for cls in _nodes(categories):
        registry.register(cls)


def registry(*, categories: Sequence[Category] | None = None) -> "NodeRegistry":
    """A new ``NodeRegistry`` holding the nodes whose category is one of ``categories`` (default: all)."""
    from conductor import NodeRegistry

    return NodeRegistry(nodes=_nodes(categories))
