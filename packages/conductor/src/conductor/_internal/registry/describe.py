"""``RegistryDescription`` and ``TypeDescription``, a registry as an editor's palette is shown it.

``registry.describe()`` gives a ``RegistryDescription``: every node and every type. Each type says which types a value
of it can be wired into, so an editor knows which edges to allow.
"""

from __future__ import annotations

from conductor._internal.base import ConductorModel
from conductor._internal.node import NodeDescription


class TypeDescription(ConductorModel):
    """A value type, as an editor is shown it.

    Attributes:
        id: The type's id.
        title: The type's name, as a person reads it.
        accepted_as: The ids of every type that takes a value of this one, this type included. An edge from an
            output of this type may lead into an input of any of them.
    """

    id: str
    title: str
    accepted_as: list[str]


class RegistryDescription(ConductorModel):
    """A registry as an editor's palette is shown it: every node, and every type.

    Attributes:
        nodes: Every node in the registry.
        types: Every type the nodes use, and every type the host named.
    """

    nodes: list[NodeDescription]
    types: list[TypeDescription]
