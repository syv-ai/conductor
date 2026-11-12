"""A registry as an editor's palette reads it: ``RegistryDescription``, every node and every type, each type a
``TypeDescription`` with the ids of the types a value of it may land on (``registry.describe()``).
"""

from __future__ import annotations

from conductor._internal.base import ConductorModel
from conductor._internal.node import NodeDescription


class TypeDescription(ConductorModel):
    """A type as an editor reads it: its ``id``, its ``title``, and the ids of every type a value of it may land on."""

    id: str
    title: str
    accepted_as: list[str]


class RegistryDescription(ConductorModel):
    """A registry as an editor's palette reads it: every node, and every type."""

    nodes: list[NodeDescription]
    types: list[TypeDescription]
