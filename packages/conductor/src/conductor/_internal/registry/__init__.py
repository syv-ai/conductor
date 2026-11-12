"""The node definitions a host offers: ``type.py`` is ``NodeRegistry``; ``describe.py`` is a registry as an editor's
palette reads it; ``upgrade.py`` moves one node of a graph record to a newer version (``registry.upgrade``).
"""

from .describe import RegistryDescription, TypeDescription
from .type import NodeRegistry
