"""``conductor_nodes`` — the standard node library.

The nodes live in a handful of modules, each listing its nodes in
``NODES``. ``registry`` builds a fresh registry holding them; ``nodes`` lists
them for a registry of your own. ``categories`` filters on each node's own
``category`` (``control``, ``json``, ``logic``, ``math``, ``regex``, ``text``)::

    import conductor_nodes

    registry = conductor_nodes.registry()                          # everything
    registry = conductor_nodes.registry(categories=["text"])       # a subset
    registry = NodeRegistry([*conductor_nodes.nodes(categories=["math"]), mine])

Conductor's core does not know this package exists, so a registry with
the standard nodes in it is asked of this package, never of
``NodeRegistry``: the nodes bring their own vocabulary, which a host with
types of its own does not want.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, get_args

from conductor import NodeDefinition, NodeRegistry

from conductor_nodes import (
    decision,
    json_ops,
    logic,
    math,
    regex_ops,
    text,
)
from conductor_nodes.types import Category

#: Every module of the library, each listing its nodes in ``NODES``.
_MODULES = (text, math, logic, json_ops, regex_ops, decision)


def nodes(*, categories: Sequence[Category] | None = None) -> tuple[NodeDefinition[..., Any], ...]:
    """Every node in the library, filtered to ``categories``, or all of them when it is ``None``.

    A bare string is a ``TypeError``, an unknown category name a ``KeyError``.
    """
    if isinstance(categories, str):
        raise TypeError(f"categories is a list of category names, not {categories!r}; say [{categories!r}]")
    known = get_args(Category)
    unknown = sorted(set(categories or ()) - set(known))
    if unknown:
        raise KeyError(f"Unknown category {', '.join(map(repr, unknown))}. Known: {list(known)}")
    return tuple(
        definition for module in _MODULES for definition in module.NODES if categories is None or definition.category in categories
    )


def registry(*, categories: Sequence[Category] | None = None) -> NodeRegistry:
    """A new ``NodeRegistry`` holding the nodes whose category is one of ``categories`` (default: all)."""
    return NodeRegistry(nodes(categories=categories))
