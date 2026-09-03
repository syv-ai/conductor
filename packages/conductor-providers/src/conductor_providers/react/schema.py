"""Palette serialization — every definition's ``describe()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conductor import NodeRegistry
    from conductor.node import NodeDescription


def palette_from_registry(registry: "NodeRegistry") -> list["NodeDescription"]:
    """The node palette: one ``NodeDescription`` per registered definition.

    Re-exported from the provider package so a frontend calling the
    provider has one import path for everything ReactFlow-related.
    """
    return [cls.describe() for cls in registry.definitions()]
