"""The registry: the set of nodes a host offers, and the types they use.

Making a node with ``@node`` doesn't offer it anywhere. A host decides which nodes its graphs may use by adding them
to a ``NodeRegistry``, much like routes go on a FastAPI router. A graph is then compiled against that registry.

``type.py`` has ``NodeRegistry``. ``describe.py`` has what an editor's palette is shown of it: every node and every
type. ``upgrade.py`` moves one node of a graph record to a newer version, for ``registry.upgrade``.
"""

from .describe import RegistryDescription, TypeDescription
from .type import NodeRegistry

__all__ = [
    "NodeRegistry",
    "RegistryDescription",
    "TypeDescription",
]
