"""Palette serialization — thin wrapper over conductor's registry schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor.registry.schema import serialize_registry

if TYPE_CHECKING:
    from conductor import NodeRegistry


def palette_from_registry(registry: "NodeRegistry") -> list[dict]:
    """Return the node palette as a list of dicts — one per registered version.

    The shape is exactly what ``conductor.registry.schema.serialize_registry``
    produces. We re-export it from the provider package so that frontends
    calling the provider have one import path for everything ReactFlow-
    related; if the palette format ever needs provider-specific tweaks, the
    change stays inside this module.
    """
    return serialize_registry(registry)
